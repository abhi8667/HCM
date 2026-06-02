import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

# Define the Two-Tower Hybrid model structure
class HybridHCMModel(nn.Module):
    def __init__(self, tabular_dim, esm_dim=1280, hidden_dim=128):
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

def compute_ece(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    bin_counts = np.bincount(binids, minlength=n_bins)
    non_empty = bin_counts > 0
    if len(y_prob) == 0:
        return 0
    ece = np.sum(np.abs(prob_true - prob_pred) * bin_counts[non_empty]) / len(y_prob)
    return ece

def train_nn(model, X_tab_tr, X_esm_tr, y_tr, epochs=40, lr=0.003):
    criterion = FocalLoss()
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

def main():
    print("=== Starting Model Trial Grid Search ===")
    
    # 1. Load Data
    data_path = '../data/HCM_labeled_final.csv'
    df = pd.read_csv(data_path)
    if df['label'].dtype == object:
        df['label'] = (df['label'] == 'Pathogenic').astype(int)
        
    leaky_cols = ['pop_freq', 'disease', 'sources', 'genomic_loc', 'review_status', 'clin_sig']
    df_clean = df.drop(columns=[c for c in leaky_cols if c in df.columns]).copy()
    
    exclude_cols = ['label', 'gene', 'accession', 'mutation_str', 'ref_aa', 'alt_aa', 'sequence_window', 'domain_name', 'region_name']
    feat_cols = [c for c in df_clean.columns if c not in exclude_cols and df_clean[c].dtype in [np.float64, np.int64, bool]]
    
    X_tab_full = df_clean[feat_cols].fillna(0).astype(float).values
    X_esm_full = np.load('../data/esm2_delta_embeddings.npy')
    y_full = df_clean['label'].values
    genes = df_clean['gene'].values
    
    target_genes = ['MYH7', 'MYBPC3', 'TNNT2', 'TNNI3', 'TPM1', 'ACTC1', 'MYL2', 'MYL3', 'TNNC1']
    
    hidden_sizes = [32, 64, 128]
    esm_dims = [50, 100, 1280] # 1280 means no PCA
    
    results = []
    
    for hidden_size in hidden_sizes:
        for esm_dim in esm_dims:
            print(f"\\n--- Testing Configuration: Hidden Size={hidden_size}, ESM Dim={esm_dim} ---")
            
            for target_gene in target_genes:
                test_idx = genes == target_gene
                train_idx = ~test_idx
                
                if sum(test_idx) == 0:
                    continue
                    
                X_tab_tr, X_tab_te = X_tab_full[train_idx], X_tab_full[test_idx]
                X_esm_tr_raw, X_esm_te_raw = X_esm_full[train_idx], X_esm_full[test_idx]
                y_tr, y_te = y_full[train_idx], y_full[test_idx]
                
                # Check if test set has both classes
                if len(np.unique(y_te)) < 2:
                    continue
                
                # Apply PCA if needed
                if esm_dim < 1280:
                    pca = PCA(n_components=esm_dim, random_state=42)
                    X_esm_tr = pca.fit_transform(X_esm_tr_raw)
                    X_esm_te = pca.transform(X_esm_te_raw)
                else:
                    X_esm_tr, X_esm_te = X_esm_tr_raw, X_esm_te_raw
                
                # Train model on 100% data (Point 3)
                model = HybridHCMModel(tabular_dim=X_tab_tr.shape[1], esm_dim=esm_dim, hidden_dim=hidden_size)
                model = train_nn(model, X_tab_tr, X_esm_tr, y_tr, epochs=40, lr=0.003)
                
                # Predict uncalibrated probabilities
                y_prob_tr_uncal = predict_nn(model, X_tab_tr, X_esm_tr)
                y_prob_te_uncal = predict_nn(model, X_tab_te, X_esm_te)
                
                # Platt Scaling (Point 5)
                y_prob_tr_uncal = np.clip(y_prob_tr_uncal, 1e-7, 1-1e-7)
                y_prob_te_uncal = np.clip(y_prob_te_uncal, 1e-7, 1-1e-7)
                
                calibrator = LogisticRegression(C=999, solver='lbfgs')
                calibrator.fit(y_prob_tr_uncal.reshape(-1, 1), y_tr)
                
                y_prob_te_cal = calibrator.predict_proba(y_prob_te_uncal.reshape(-1, 1))[:, 1]
                
                # Compute metrics
                auprc = average_precision_score(y_te, y_prob_te_cal)
                auroc = roc_auc_score(y_te, y_prob_te_cal)
                brier = brier_score_loss(y_te, y_prob_te_cal)
                ece = compute_ece(y_te, y_prob_te_cal)
                
                results.append({
                    'Hidden_Size': hidden_size,
                    'ESM_Dim': esm_dim,
                    'Gene': target_gene,
                    'AUPRC': auprc,
                    'AUROC': auroc,
                    'Brier': brier,
                    'ECE': ece
                })
                print(f"      {target_gene}: AUPRC={auprc:.4f}, AUROC={auroc:.4f}, Brier={brier:.4f}, ECE={ece:.4f}")
                    
    df_res = pd.DataFrame(results)
    df_res.to_csv('grid_search_results.csv', index=False)
    
    # Calculate averages
    avg_df = df_res.groupby(['Hidden_Size', 'ESM_Dim']).mean().reset_index()
    avg_df.to_csv('grid_search_summary_averages.csv', index=False)
    print("\\nSaved results to grid_search_results.csv and grid_search_summary_averages.csv")
    
if __name__ == '__main__':
    main()
