import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import warnings
warnings.filterwarnings('ignore')

from transformers import EsmTokenizer, EsmModel

print("Starting Month 2 Execution...")

# --- Biophysical Dictionaries ---
AA_SIZES = {
    'A': 1, 'C': 1, 'D': 1, 'G': 1, 'S': 1, 'T': 1,
    'E': 2, 'H': 2, 'I': 2, 'K': 2, 'L': 2, 'M': 2, 'N': 2, 'P': 2, 'Q': 2, 'V': 2,
    'F': 3, 'R': 3, 'W': 3, 'Y': 3
}

AA_CHARGES = {
    'D': -1, 'E': -1,
    'H': 1, 'K': 1, 'R': 1,
    'A': 0, 'C': 0, 'F': 0, 'G': 0, 'I': 0, 'L': 0, 'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
}

rows = "ARNDCQEGHILKMFPSTWYV"
grantham_raw = [
      [0],
      [112,      0],
      [111,     86,      0],
      [126,     96,     23,      0],
      [195,    180,    139,    154,      0],
      [ 91,     43,     46,     61,    154,      0],
      [107,     54,     42,     45,    170,     29,      0],
      [ 60,    125,     80,     94,    159,     87,     98,      0],
      [ 86,     29,     68,     81,    174,     24,     40,     98,      0],
      [ 94,     97,    149,    168,    198,    109,    134,    135,     94,      0],
      [ 96,    102,    153,    172,    198,    113,    138,    138,     99,      5,      0],
      [106,     26,     94,    101,    202,     53,     56,    127,     32,    102,    107,      0],
      [ 84,     91,    142,    160,    196,    101,    126,    127,     87,     10,     15,     95,      0],
      [113,     97,    158,    177,    205,    116,    140,    153,    100,     21,     22,    102,     28,      0],
      [ 27,    103,     91,    108,    169,     76,     93,     42,     77,     95,     98,    103,     87,    114,      0],
      [ 99,    110,     46,     65,    112,     68,     80,     56,     89,    142,    145,    121,    135,    155,     74,      0],
      [ 58,     71,     65,     85,    149,     42,     65,     59,     47,     89,     92,     78,     81,    103,     38,     58,      0],
      [148,    101,    174,    181,    215,    130,    152,    184,    115,     61,     61,    110,     67,     40,    147,    177,    128,      0],
      [112,     77,    143,    160,    194,     99,    122,    147,     83,     33,     36,     85,     36,     22,    110,    144,     92,     37,      0],
      [ 64,     96,    133,    152,    192,     96,    121,    109,     84,     29,     32,     97,     21,     50,     68,    124,     69,     88,     55,      0]
]

def get_grantham(aa1, aa2):
    if aa1 not in rows or aa2 not in rows: return 0
    i = rows.index(aa1)
    j = rows.index(aa2)
    if i >= j: return grantham_raw[i][j]
    else: return grantham_raw[j][i]

# --- Week 5: Two-Tower Hybrid Model ---
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

# --- Load Data from Month 1 ---
print("Loading data and splits from Month 1...")
data_path = 'data/HCM_labeled_final.csv' if os.path.exists('data/HCM_labeled_final.csv') else 'HCM_labeled_final.csv'
df = pd.read_csv(data_path)
if df['label'].dtype == object:
    df['label'] = (df['label'] == 'Pathogenic').astype(int)

leaky_cols = ['pop_freq', 'disease', 'sources', 'genomic_loc', 'review_status', 'clin_sig']
df_clean = df.drop(columns=[c for c in leaky_cols if c in df.columns]).copy()

exclude_cols = ['label', 'gene', 'accession', 'mutation_str', 'ref_aa', 'alt_aa', 'sequence_window', 'domain_name', 'region_name']
feat_cols = [c for c in df_clean.columns if c not in exclude_cols and df_clean[c].dtype in [np.float64, np.int64, bool]]
X_tab = df_clean[feat_cols].fillna(0).astype(float).values

emb_path = 'data/esm2_delta_embeddings.npy' if os.path.exists('data/esm2_delta_embeddings.npy') else 'esm2_delta_embeddings.npy'
if os.path.exists(emb_path):
    X_esm = np.load(emb_path)
else:
    print("Mocking ESM embeddings...")
    X_esm = np.random.rand(len(df_clean), 1280)

y = df_clean['label'].values

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)
fig_prefix = 'figures/'

# 1 & 2. LOGO Validation Loop (9 Genes)
print("\n--- Week 5 & 6: Training and Rigorous Evaluation (LOGO) ---")
target_genes = ['MYH7', 'MYBPC3', 'TNNT2', 'TNNI3', 'TPM1', 'ACTC1', 'MYL2', 'MYL3', 'TNNC1']
results = []
n_bootstraps = 100

for target_gene in target_genes:
    print(f"\nProcessing held-out gene: {target_gene}")
    test_idx = df_clean['gene'] == target_gene
    train_idx = ~test_idx
    
    if sum(test_idx) == 0:
        continue
        
    X_tab_tr, X_tab_te = X_tab[train_idx], X_tab[test_idx]
    X_esm_tr, X_esm_te = X_esm[train_idx], X_esm[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    
    hybrid_model = HybridHCMModel(tabular_dim=X_tab.shape[1], esm_dim=X_esm.shape[1])
    hybrid_model = train_nn(hybrid_model, X_tab_tr, X_esm_tr, y_tr)
    y_pred_proba = predict_nn(hybrid_model, X_tab_te, X_esm_te)
    
    bootstrapped_auprc = []
    for i in range(n_bootstraps):
        indices = resample(np.arange(len(y_te)), replace=True)
        if len(np.unique(y_te[indices])) < 2:
            continue
        score = average_precision_score(y_te[indices], y_pred_proba[indices])
        bootstrapped_auprc.append(score)
    
    mean_auprc = np.mean(bootstrapped_auprc)
    lower_ci = np.percentile(bootstrapped_auprc, 2.5)
    upper_ci = np.percentile(bootstrapped_auprc, 97.5)
    
    brier = brier_score_loss(y_te, y_pred_proba)
    prob_true, prob_pred = calibration_curve(y_te, y_pred_proba, n_bins=10)
    
    bins = np.linspace(0, 1, 10 + 1)
    binids = np.digitize(y_pred_proba, bins) - 1
    bin_counts = np.bincount(binids, minlength=10)
    non_empty = bin_counts > 0
    ece = np.sum(np.abs(prob_true - prob_pred) * bin_counts[non_empty]) / len(y_pred_proba)
    
    results.append({
        'Gene': target_gene,
        'AUPRC': f"{mean_auprc:.4f}",
        '95% CI': f"[{lower_ci:.4f}, {upper_ci:.4f}]",
        'Brier': f"{brier:.4f}",
        'ECE': f"{ece:.4f}"
    })

    if target_gene == 'TNNT2':
        plt.figure(figsize=(6, 6))
        plt.plot(prob_pred, prob_true, marker='o', label='Hybrid Model (Focal Loss)')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
        plt.title('Reliability Diagram (TNNT2 LOGO)')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.legend()
        plt.savefig(f"{fig_prefix}calibration_plot_m2.png")
        plt.close()

# Report Results Table
print("\n--- LOGO Validation Results ---")
df_results = pd.DataFrame(results)
print(df_results.to_markdown(index=False))
df_results.to_csv("results/logo_metrics.csv", index=False)

# 3. Ablation Study (on TNNT2)
print("\n--- Week 7: Ablation Study (TNNT2 Hold-out) ---")
test_idx = df_clean['gene'] == 'TNNT2'
train_idx = ~test_idx
X_tab_tr, X_tab_te = X_tab[train_idx], X_tab[test_idx]
X_esm_tr, X_esm_te = X_esm[train_idx], X_esm[test_idx]
y_tr, y_te = y[train_idx], y[test_idx]

model_no_esm = HybridHCMModel(tabular_dim=X_tab.shape[1], esm_dim=X_esm.shape[1])
model_no_esm = train_nn(model_no_esm, X_tab_tr, np.zeros_like(X_esm_tr), y_tr)
pred_no_esm = predict_nn(model_no_esm, X_tab_te, np.zeros_like(X_esm_te))
auprc_no_esm = average_precision_score(y_te, pred_no_esm)

model_no_tab = HybridHCMModel(tabular_dim=X_tab.shape[1], esm_dim=X_esm.shape[1])
model_no_tab = train_nn(model_no_tab, np.zeros_like(X_tab_tr), X_esm_tr, y_tr)
pred_no_tab = predict_nn(model_no_tab, np.zeros_like(X_tab_te), X_esm_te)
auprc_no_tab = average_precision_score(y_te, pred_no_tab)

print(f"Full Hybrid AUPRC (TNNT2): {df_results[df_results['Gene']=='TNNT2']['AUPRC'].values[0]}")
print(f"Ablation (No ESM) AUPRC: {auprc_no_esm:.4f}")
print(f"Ablation (No Tabular) AUPRC: {auprc_no_tab:.4f}")

# Train final model for ISM inference
print("\nTraining final hybrid model on full dataset for ISM inference...")
final_model = HybridHCMModel(tabular_dim=X_tab.shape[1], esm_dim=X_esm.shape[1])
final_model = train_nn(final_model, X_tab, X_esm, y)

# 4. REAL In Silico Mutagenesis (ISM) Landscape
print("\n--- Week 8: Real In Silico Mutagenesis (ISM) Landscape ---")
aas = list("ACDEFGHIKLMNPQRSTVWY")

print("Loading live ESM-2 Model...")
tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
esm_model = EsmModel.from_pretrained('facebook/esm2_t33_650M_UR50D')
esm_model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
esm_model.to(device)

def get_esm_delta(wt_seq, mut_seq):
    with torch.no_grad():
        inputs_wt = tokenizer(wt_seq, return_tensors="pt", padding=True).to(device)
        inputs_mut = tokenizer(mut_seq, return_tensors="pt", padding=True).to(device)
        out_wt = esm_model(**inputs_wt).last_hidden_state.mean(dim=1).cpu().numpy()
        out_mut = esm_model(**inputs_mut).last_hidden_state.mean(dim=1).cpu().numpy()
    return out_mut - out_wt

def generate_real_ism(gene_name, num_positions=20):
    print(f"Computing real ISM predictions for {gene_name}...")
    gene_df = df_clean[df_clean['gene'] == gene_name].drop_duplicates(subset=['position']).sort_values('position').head(num_positions)
    
    heatmap_matrix = np.zeros((20, len(gene_df)))
    positions = gene_df['position'].tolist()
    
    for c_idx, (_, row) in enumerate(gene_df.iterrows()):
        wt_seq = row['sequence_window']
        ref_aa = row['ref_aa']
        
        base_tab = row[feat_cols].copy()
        
        for r_idx, alt_aa in enumerate(aas):
            if alt_aa == ref_aa:
                mut_seq = wt_seq
            else:
                if len(wt_seq) == 11:
                    mut_seq = wt_seq[:5] + alt_aa + wt_seq[6:]
                else:
                    mut_seq = wt_seq.replace(ref_aa, alt_aa, 1)
            
            delta_esm = get_esm_delta(wt_seq, mut_seq)
            mut_tab = base_tab.copy()
            
            mut_tab['alt_size'] = AA_SIZES.get(alt_aa, 0)
            mut_tab['alt_charge'] = AA_CHARGES.get(alt_aa, 0)
            mut_tab['size_change'] = mut_tab['alt_size'] - mut_tab['ref_size']
            mut_tab['charge_change'] = mut_tab['alt_charge'] - mut_tab['ref_charge']
            mut_tab['grantham_score'] = get_grantham(ref_aa, alt_aa)
            mut_tab['win_+0_size'] = mut_tab['alt_size']
            mut_tab['win_+0_charge'] = mut_tab['alt_charge']
            
            x_tab_tensor = torch.FloatTensor(mut_tab.values.astype(float)).unsqueeze(0)
            x_esm_tensor = torch.FloatTensor(delta_esm)
            
            prob = predict_nn(final_model, x_tab_tensor, x_esm_tensor)[0]
            heatmap_matrix[r_idx, c_idx] = prob
            
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_matrix, yticklabels=aas, xticklabels=positions, cmap="coolwarm", annot=False)
    plt.title(f'Real In Silico Mutagenesis Landscape ({gene_name})')
    plt.xlabel('Sequence Position')
    plt.ylabel('Mutated Amino Acid')
    out_file = f"{fig_prefix}ism_landscape_m2_{gene_name}.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {gene_name} real ISM heatmap to '{out_file}'")

for target_gene in target_genes:
    generate_real_ism(target_gene)

print("\nMonth 2 execution completed.")
