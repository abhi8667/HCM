import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import warnings
warnings.filterwarnings('ignore')

from transformers import EsmTokenizer, EsmModel

print("Starting Month 3 Execution: VUS Re-stratification")

# 1. Define the Hybrid Model (Same as Month 2)
class HybridHCMModel(nn.Module):
    def __init__(self, tabular_dim, esm_dim=1280, hidden_dim=64):
        super().__init__()
        self.tower_tab = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.tower_esm = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x_tab, x_esm):
        out_tab = self.tower_tab(x_tab)
        out_esm = self.tower_esm(x_esm)
        fused = torch.cat([out_tab, out_esm], dim=1)
        return self.fusion(fused)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        p_t = torch.where(targets == 1, inputs, 1 - inputs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        epsilon = 1e-8
        loss = -alpha_t * (1 - p_t) ** self.gamma * torch.log(p_t + epsilon)
        return loss.mean()

def train_nn(model, X_tab_train, X_esm_train, y_train, epochs=20, lr=0.01):
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    tab_t = torch.FloatTensor(X_tab_train)
    esm_t = torch.FloatTensor(X_esm_train)
    y_t = torch.FloatTensor(y_train).unsqueeze(1)
    for ep in range(epochs):
        optimizer.zero_grad()
        out = model(tab_t, esm_t)
        loss = criterion(out, y_t)
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

os.makedirs('models', exist_ok=True)
torch.save(final_model.state_dict(), 'models/hcm_final_two_tower_model.pth')
print("Saved final trained Two-Tower model to 'models/hcm_final_two_tower_model.pth'")


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
vus_probs = predict_nn(final_model, X_tab_vus, X_esm_vus)
df_vus['Predicted_Pathogenicity'] = vus_probs

top_vus = df_vus.sort_values(by='Predicted_Pathogenicity', ascending=False)
output_cols = ['gene', 'mutation_str', 'Predicted_Pathogenicity', 'clin_sig', 'disease']
output_cols = [c for c in output_cols if c in top_vus.columns]

os.makedirs('results', exist_ok=True)
top_vus[output_cols].to_csv('results/VUS_restratification_table.csv', index=False)

print(f"Successfully evaluated {len(top_vus)} VUS variants.")
print("Saved complete ranked list to 'results/VUS_restratification_table.csv'")
print("Month 3 execution complete.")
