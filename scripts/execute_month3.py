import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
import joblib

from transformers import EsmTokenizer, EsmModel

print("Starting Month 3 Execution: VUS Re-stratification")

# 1. Define the Hybrid Model (Finalized Weighted BCE Version)
class HybridHCMModel(nn.Module):
    def __init__(self, tabular_dim, esm_dim=1280, hidden_dim=32):
        super().__init__()
        self.tower_tab = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3)
        )
        self.tower_esm = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3)
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x_tab, x_esm):
        out_tab = self.tower_tab(x_tab)
        out_esm = self.tower_esm(x_esm)
        fused = torch.cat([out_tab, out_esm], dim=1)
        return self.fusion(fused)

def train_nn(model, X_tab_tr, X_esm_tr, y_tr, epochs=40, lr=0.003):
    num_class_1 = np.sum(y_tr == 1)
    num_class_0 = np.sum(y_tr == 0)
    total = len(y_tr)
    weight_1 = total / (2.0 * num_class_1)
    weight_0 = total / (2.0 * num_class_0)
    
    sample_weights = np.where(y_tr == 1, weight_1, weight_0)
    sample_weights_t = torch.FloatTensor(sample_weights).unsqueeze(1)
    
    criterion = nn.BCELoss(weight=sample_weights_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    tab_t_tr = torch.FloatTensor(X_tab_tr)
    esm_t_tr = torch.FloatTensor(X_esm_tr)
    y_t_tr = torch.FloatTensor(y_tr).unsqueeze(1)
    
    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        out = model(tab_t_tr, esm_t_tr)
        loss = criterion(out, y_t_tr)
        loss.backward()
        optimizer.step()
    return model

def predict_nn(model, X_tab, X_esm):
    model.eval()
    with torch.no_grad():
        out = model(torch.FloatTensor(X_tab), torch.FloatTensor(X_esm))
    return out.numpy().flatten()


# 2. Train the Final Model on Labeled Data
print("Loading labeled data and training final model...")
data_path = 'data/HCM_labeled_final.csv' if os.path.exists('data/HCM_labeled_final.csv') else 'HCM_labeled_final.csv'
df_train = pd.read_csv(data_path)
if df_train['label'].dtype == object:
    df_train['label'] = (df_train['label'] == 'Pathogenic').astype(int)

leaky_cols = ['pop_freq', 'disease', 'sources', 'genomic_loc', 'review_status', 'clin_sig']
df_clean = df_train.drop(columns=[c for c in leaky_cols if c in df_train.columns]).copy()
exclude_cols = ['label', 'gene', 'accession', 'mutation_str', 'ref_aa', 'alt_aa', 'sequence_window', 'domain_name', 'region_name']
feat_cols = [c for c in df_clean.columns if c not in exclude_cols and df_clean[c].dtype in [np.float64, np.int64, bool]]
X_tab_train = df_clean[feat_cols].fillna(0).astype(float).values
y_train = df_clean['label'].values

emb_path = 'data/esm2_delta_embeddings.npy' if os.path.exists('data/esm2_delta_embeddings.npy') else 'esm2_delta_embeddings.npy'
if os.path.exists(emb_path):
    X_esm_train = np.load(emb_path)
else:
    print("Error: ESM embeddings not found. Please run execute_month1.py first.")
    exit(1)

final_model = HybridHCMModel(tabular_dim=X_tab_train.shape[1], esm_dim=X_esm_train.shape[1])
final_model = train_nn(final_model, X_tab_train, X_esm_train, y_train)

# Fit Platt Scaling on full dataset training predictions
final_prob_tr_uncal = predict_nn(final_model, X_tab_train, X_esm_train)
final_prob_tr_uncal = np.clip(final_prob_tr_uncal, 1e-7, 1-1e-7)
from sklearn.linear_model import LogisticRegression
calibrator = LogisticRegression(C=999, solver='lbfgs')
calibrator.fit(final_prob_tr_uncal.reshape(-1, 1), y_train)

os.makedirs('models', exist_ok=True)
torch.save(final_model.state_dict(), 'models/hcm_final_two_tower_model.pth')
# Save the calibrator coefficient and intercept to a text file for deployment loading or fit it online
import pickle
with open('models/hcm_platt_calibrator.pkl', 'wb') as f:
    pickle.dump(calibrator, f)
print("Saved final trained Two-Tower model weights and Platt calibrator.")

print("Training final RF model on full dataset for VUS restratification...")
final_rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced')
final_rf.fit(np.hstack([X_tab_train, X_esm_train]), y_train)
joblib.dump(final_rf, 'models/hcm_final_rf_model.joblib')
print("Saved final trained RF model to 'models/hcm_final_rf_model.joblib'")

# 3. Load VUS Data
print("Loading all variants and filtering for VUS...")
all_data_path = 'data/HCM_all_variants_v2.csv' if os.path.exists('data/HCM_all_variants_v2.csv') else 'HCM_all_variants_v2.csv'
df_all = pd.read_csv(all_data_path)

vus_mask = df_all['clin_sig'].str.contains('Uncertain', na=False, case=False)
df_vus = df_all[vus_mask].copy()
print(f"Identified {len(df_vus)} VUS variants for re-stratification.")

for c in feat_cols:
    if c not in df_vus.columns:
        df_vus[c] = 0
X_tab_vus = df_vus[feat_cols].fillna(0).astype(float).values

# 4. Extract 650M ESM-2 Embeddings for VUS
print("Loading live ESM-2 Model (650M) for VUS extraction...")
model_name = 'facebook/esm2_t33_650M_UR50D'
tokenizer = EsmTokenizer.from_pretrained(model_name)
esm_model = EsmModel.from_pretrained(model_name)
esm_model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
esm_model.to(device)

def mutate_sequence(row):
    seq = row['sequence_window']
    if len(seq) == 11 and seq[5] == row['ref_aa']:
        return seq[:5] + row['alt_aa'] + seq[6:]
    return seq.replace(row['ref_aa'], row['alt_aa'], 1)

wt_seqs = df_vus['sequence_window'].tolist()
mut_seqs = df_vus.apply(mutate_sequence, axis=1).tolist()

batch_size = 64
delta_embs = []
print(f"Extracting 1280-dim embeddings for {len(wt_seqs)} VUS variants...")
start_time = time.time()
with torch.no_grad():
    for i in range(0, len(wt_seqs), batch_size):
        batch_wt = wt_seqs[i:i+batch_size]
        batch_mut = mut_seqs[i:i+batch_size]
        inputs_wt = tokenizer(batch_wt, return_tensors="pt", padding=True, truncation=True).to(device)
        inputs_mut = tokenizer(batch_mut, return_tensors="pt", padding=True, truncation=True).to(device)
        out_wt = esm_model(**inputs_wt).last_hidden_state.mean(dim=1).cpu().numpy()
        out_mut = esm_model(**inputs_mut).last_hidden_state.mean(dim=1).cpu().numpy()
        delta_embs.append(out_mut - out_wt)
        if i % 500 == 0 and i > 0:
            print(f"Processed {i}/{len(wt_seqs)} VUS sequences...")

X_esm_vus = np.vstack(delta_embs)
print(f"Extraction complete in {time.time() - start_time:.2f} seconds.")

# 5. Predict and Save
print("Running final predictions...")

# Two-Tower Prediction
uncal_vus_probs = predict_nn(final_model, X_tab_vus, X_esm_vus)
uncal_vus_probs = np.clip(uncal_vus_probs, 1e-7, 1-1e-7)
vus_probs_tt = calibrator.predict_proba(uncal_vus_probs.reshape(-1, 1))[:, 1]
df_vus_tt = df_vus.copy()
df_vus_tt['Predicted_Pathogenicity'] = vus_probs_tt
top_vus_tt = df_vus_tt.sort_values(by='Predicted_Pathogenicity', ascending=False)
output_cols = ['gene', 'mutation_str', 'Predicted_Pathogenicity', 'clin_sig', 'disease']
output_cols_tt = [c for c in output_cols if c in top_vus_tt.columns]

# RF Prediction
vus_probs_rf = final_rf.predict_proba(np.hstack([X_tab_vus, X_esm_vus]))[:, 1]
df_vus_rf = df_vus.copy()
df_vus_rf['Predicted_Pathogenicity'] = vus_probs_rf
top_vus_rf = df_vus_rf.sort_values(by='Predicted_Pathogenicity', ascending=False)
output_cols_rf = [c for c in output_cols if c in top_vus_rf.columns]

os.makedirs('results', exist_ok=True)
top_vus_tt[output_cols_tt].to_csv('results/VUS_restratification_two_tower.csv', index=False)
top_vus_rf[output_cols_rf].to_csv('results/VUS_restratification_rf.csv', index=False)

print(f"Successfully evaluated {len(df_vus)} VUS variants.")
print("Saved complete ranked lists to 'results/VUS_restratification_two_tower.csv' and 'results/VUS_restratification_rf.csv'")
print("Month 3 execution complete.")
