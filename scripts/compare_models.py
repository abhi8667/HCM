import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import copy
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Define the Two-Tower Hybrid model structure
class HybridHCMModel(nn.Module):
    def __init__(self, tabular_dim, esm_dim=1280, hidden_dim=128):
        super().__init__()
        # Increased hidden dimension and added dropout for better generalization
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

def train_nn_with_early_stopping(model, X_tab_tr, X_esm_tr, y_tr, X_tab_val, X_esm_val, y_val, max_epochs=150, lr=0.005, patience=20):
    criterion = FocalLoss()
    # Using AdamW for better weight decay handling
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
    
    tab_t_tr = torch.FloatTensor(X_tab_tr)
    esm_t_tr = torch.FloatTensor(X_esm_tr)
    y_t_tr = torch.FloatTensor(y_tr).unsqueeze(1)
    
    tab_t_val = torch.FloatTensor(X_tab_val)
    esm_t_val = torch.FloatTensor(X_esm_val)
    y_t_val = torch.FloatTensor(y_val).unsqueeze(1)
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    
    for ep in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(tab_t_tr, esm_t_tr)
        loss = criterion(out, y_t_tr)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(tab_t_val, esm_t_val)
            val_loss = criterion(val_out, y_t_val).item()
            
        scheduler.step(val_loss)
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"      Early stopping triggered at epoch {ep}. Restoring best weights.")
            break
            
    # Load best weights
    model.load_state_dict(best_model_wts)
    return model

def predict_nn(model, X_tab, X_esm):
    model.eval()
    with torch.no_grad():
        out = model(torch.FloatTensor(X_tab), torch.FloatTensor(X_esm))
    return out.numpy().flatten()

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

def main():
    print("=== Starting Rigorous LOGO Model Tuning and Comparison ===")
    
    # 1. Load Data
    data_path = 'data/HCM_labeled_final.csv'
    df = pd.read_csv(data_path)
    if df['label'].dtype == object:
        df['label'] = (df['label'] == 'Pathogenic').astype(int)
        
    leaky_cols = ['pop_freq', 'disease', 'sources', 'genomic_loc', 'review_status', 'clin_sig']
    df_clean = df.drop(columns=[c for c in leaky_cols if c in df.columns]).copy()
    
    exclude_cols = ['label', 'gene', 'accession', 'mutation_str', 'ref_aa', 'alt_aa', 'sequence_window', 'domain_name', 'region_name']
    feat_cols = [c for c in df_clean.columns if c not in exclude_cols and df_clean[c].dtype in [np.float64, np.int64, bool]]
    
    X_tab = df_clean[feat_cols].fillna(0).astype(float).values
    X_esm = np.load('data/esm2_delta_embeddings.npy')
    y = df_clean['label'].values
    
    # 2. Iterate over all genes
    target_genes = ['MYH7', 'MYBPC3', 'TNNT2', 'TNNI3', 'TPM1', 'ACTC1', 'MYL2', 'MYL3', 'TNNC1']
    
    results = []
    
    for target_gene in target_genes:
        print(f"\nEvaluating on Holdout Gene: {target_gene}")
        test_idx = df_clean['gene'] == target_gene
        train_idx_all = ~test_idx
        
        if sum(test_idx) == 0:
            print(f"Skipping {target_gene}, not found in test set.")
            continue
            
        X_tab_full_tr, X_tab_te = X_tab[train_idx_all], X_tab[test_idx]
        X_esm_full_tr, X_esm_te = X_esm[train_idx_all], X_esm[test_idx]
        y_full_tr, y_te = y[train_idx_all], y[test_idx]
        
        # Check if test set has both classes
        if len(np.unique(y_te)) < 2:
            print(f"Skipping {target_gene}, only one class in test set.")
            continue
            
        # Split train into train and validation (80/20) for Neural Network Early Stopping
        X_tab_tr, X_tab_val, X_esm_tr, X_esm_val, y_tr, y_val = train_test_split(
            X_tab_full_tr, X_esm_full_tr, y_full_tr, test_size=0.2, random_state=42, stratify=y_full_tr
        )
            
        # Train baseline Random Forest on FULL train set
        X_rf_tr = np.hstack([X_tab_full_tr, X_esm_full_tr])
        X_rf_te = np.hstack([X_tab_te, X_esm_te])
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced')
        rf.fit(X_rf_tr, y_full_tr)
        y_prob_rf = rf.predict_proba(X_rf_te)[:, 1]
            
        # Train Two-Tower model with Early Stopping
        print("    Tuning Two-Tower Neural Network...")
        two_tower = HybridHCMModel(tabular_dim=X_tab_tr.shape[1], esm_dim=X_esm_tr.shape[1])
        two_tower = train_nn_with_early_stopping(
            two_tower, X_tab_tr, X_esm_tr, y_tr, X_tab_val, X_esm_val, y_val,
            max_epochs=200, lr=0.003, patience=25
        )
        y_prob_tt = predict_nn(two_tower, X_tab_te, X_esm_te)
        
        # Compute Metrics
        auprc_rf = average_precision_score(y_te, y_prob_rf)
        auroc_rf = roc_auc_score(y_te, y_prob_rf)
        brier_rf = brier_score_loss(y_te, y_prob_rf)
        ece_rf = compute_ece(y_te, y_prob_rf)
        
        auprc_tt = average_precision_score(y_te, y_prob_tt)
        auroc_tt = roc_auc_score(y_te, y_prob_tt)
        brier_tt = brier_score_loss(y_te, y_prob_tt)
        ece_tt = compute_ece(y_te, y_prob_tt)
        
        results.append({'Gene': target_gene, 'Model': 'Baseline RF', 'AUPRC': auprc_rf, 'AUROC': auroc_rf, 'Brier': brier_rf, 'ECE': ece_rf})
        results.append({'Gene': target_gene, 'Model': 'Two-Tower Hybrid (Tuned)', 'AUPRC': auprc_tt, 'AUROC': auroc_tt, 'Brier': brier_tt, 'ECE': ece_tt})
        
        print(f"    RF Baseline        - AUPRC: {auprc_rf:.4f}, AUROC: {auroc_rf:.4f}")
        print(f"    Two-Tower (Tuned)  - AUPRC: {auprc_tt:.4f}, AUROC: {auroc_tt:.4f}")

    # 3. Save comparison metrics CSV
    comparison_df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    comparison_df.to_csv('results/model_comparison_metrics_all_genes_tuned.csv', index=False)
    print("\nSaved fully tuned comparison table to results/model_comparison_metrics_all_genes_tuned.csv")
    
    # 4. Generate beautiful figures
    sns.set_theme(style="whitegrid")
    
    metrics = ['AUPRC', 'AUROC', 'Brier', 'ECE']
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        sns.barplot(data=comparison_df, x='Gene', y=metric, hue='Model', palette=['#95A5A6', '#27AE60'], ax=axes[i])
        axes[i].set_title(f'{metric} Comparison by Gene (Fully Tuned)', fontsize=15, fontweight='bold')
        axes[i].set_xlabel('Held-out Gene', fontsize=12)
        axes[i].set_ylabel(metric, fontsize=12)
        axes[i].tick_params(axis='x', rotation=45)
        if i > 0:
            axes[i].legend_.remove() # only keep one legend
        else:
            axes[i].legend(title='Model Type', fontsize=11)
            
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    fig_path = 'figures/model_metrics_bar_comparison_tuned.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved tuned metrics bar chart comparison to {fig_path}")
    
    # Plot average differences across all genes
    avg_df = comparison_df.groupby('Model')[metrics].mean().reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Higher is better
    avg_df_higher = pd.melt(avg_df, id_vars=['Model'], value_vars=['AUPRC', 'AUROC'])
    sns.barplot(data=avg_df_higher, x='variable', y='value', hue='Model', palette=['#95A5A6', '#27AE60'], ax=axes[0])
    axes[0].set_title('Average Performance across 9 Genes (Higher is Better)', fontweight='bold', fontsize=13)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_xlabel('Metric', fontsize=12)
    
    # Lower is better
    avg_df_lower = pd.melt(avg_df, id_vars=['Model'], value_vars=['Brier', 'ECE'])
    sns.barplot(data=avg_df_lower, x='variable', y='value', hue='Model', palette=['#95A5A6', '#27AE60'], ax=axes[1])
    axes[1].set_title('Average Error across 9 Genes (Lower is Better)', fontweight='bold', fontsize=13)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_xlabel('Metric', fontsize=12)
    
    plt.tight_layout()
    fig_path_avg = 'figures/model_metrics_avg_comparison_tuned.png'
    plt.savefig(fig_path_avg, dpi=300)
    plt.close()
    print(f"Saved tuned average metrics comparison to {fig_path_avg}")
    
    print("=== Model Tuning and Comparison Completed Successfully ===")

if __name__ == "__main__":
    main()
