import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve
import joblib
import torch
import time
import os

try:
    from transformers import EsmTokenizer, EsmModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Transformers not found, please install with `pip install transformers torch`")

def clean_dataset(df):
    print("--- Week 1: Cleaning Dataset ---")
    # Identify leaky columns
    leaky_cols = ['pop_freq', 'disease', 'sources', 'genomic_loc', 'review_status', 'clin_sig']
    cols_to_drop = [c for c in leaky_cols if c in df.columns]
    
    # We want to predict 'label', so we keep that.
    df_clean = df.drop(columns=cols_to_drop).copy()
    
    print(f"Dropped {len(cols_to_drop)} leaky features: {cols_to_drop}")
    return df_clean

def mutate_sequence(row):
    seq = row['sequence_window']
    # The mutation is typically at the center of the 11-mer (index 5)
    # Let's verify if the center is ref_aa
    if len(seq) == 11:
        if seq[5] == row['ref_aa']:
            mut_seq = seq[:5] + row['alt_aa'] + seq[6:]
            return mut_seq
    # Fallback if the above assumption is false
    return seq.replace(row['ref_aa'], row['alt_aa'], 1)

def extract_esm2_embeddings(df, model_name="facebook/esm2_t6_8M_UR50D"):
    print(f"--- Week 3: Extracting ESM-2 Embeddings ({model_name}) ---")
    if not HAS_TRANSFORMERS:
        print("Skipping ESM-2 due to missing libraries, using random features for testing pipeline.")
        return np.random.rand(len(df), 320)
        
    print("Loading ESM-2 model...")
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    wt_seqs = df['sequence_window'].tolist()
    
    # Generate mutant sequences
    print("Generating mutant sequences...")
    mut_seqs = df.apply(mutate_sequence, axis=1).tolist()
    
    batch_size = 64
    delta_embs = []
    
    print(f"Extracting embeddings for {len(wt_seqs)} variants...")
    with torch.no_grad():
        for i in range(0, len(wt_seqs), batch_size):
            batch_wt = wt_seqs[i:i+batch_size]
            batch_mut = mut_seqs[i:i+batch_size]
            
            inputs_wt = tokenizer(batch_wt, return_tensors="pt", padding=True, truncation=True).to(device)
            inputs_mut = tokenizer(batch_mut, return_tensors="pt", padding=True, truncation=True).to(device)
            
            out_wt = model(**inputs_wt).last_hidden_state.mean(dim=1).cpu().numpy()
            out_mut = model(**inputs_mut).last_hidden_state.mean(dim=1).cpu().numpy()
            
            delta = out_mut - out_wt
            delta_embs.append(delta)
            
            if i % 500 == 0 and i > 0:
                print(f"Processed {i}/{len(wt_seqs)} sequences")
                
    return np.vstack(delta_embs)

def get_tabular_features(df):
    # Select numeric/boolean columns for the baseline, excluding target and ID columns
    exclude_cols = ['label', 'gene', 'accession', 'mutation_str', 'ref_aa', 'alt_aa', 'sequence_window', 'domain_name', 'region_name']
    feat_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [np.float64, np.int64, bool]]
    return df[feat_cols].fillna(0).astype(float).values

def train_logo_validation(df, delta_embs, target_gene='MYBPC3'):
    print(f"\n--- Week 2 & 4: LOGO Validation (Hold-out {target_gene}) & Training ---")
    
    tabular_features = get_tabular_features(df)
    X = np.hstack([tabular_features, delta_embs])
    y = df['label'].values
    
    test_idx = df['gene'] == target_gene
    train_idx = ~test_idx
    
    if not any(test_idx):
        print(f"Error: Target gene {target_gene} not found in dataset!")
        return
        
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"Train size: {X_train.shape[0]}, Test size ({target_gene}): {X_test.shape[0]}")
    print(f"Train class balance: {np.mean(y_train):.2f}, Test class balance: {np.mean(y_test):.2f}")
    
    # Train model
    print("Training Random Forest baseline...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    
    auprc = average_precision_score(y_test, y_pred_proba)
    auroc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nResults on {target_gene} Hold-out:")
    print(f"AUPRC: {auprc:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return clf

if __name__ == "__main__":
    start_time = time.time()
    print("Loading data...")
    data_path = 'data/HCM_labeled_final.csv' if os.path.exists('data/HCM_labeled_final.csv') else 'HCM_labeled_final.csv'
    df = pd.read_csv(data_path)
    
    # Map label to 0/1 if it isn't already
    if df['label'].dtype == object:
        df['label'] = (df['label'] == 'Pathogenic').astype(int)
        
    df_clean = clean_dataset(df)
    
    # We'll use a very small ESM-2 model (t6_8M) to make this script run reasonably fast on any machine
    # For publication, use "facebook/esm2_t33_650M_UR50D"
    embs = extract_esm2_embeddings(df_clean, model_name="facebook/esm2_t33_650M_UR50D")
    
    # Save the embeddings so we don't have to recompute them later
    emb_out = 'data/esm2_delta_embeddings.npy' if os.path.exists('data') else 'esm2_delta_embeddings.npy'
    np.save(emb_out, embs)
    print(f"\nSaved ESM-2 embeddings to '{emb_out}'")
    
    model = train_logo_validation(df_clean, embs, target_gene='TNNT2')
    
    # Save the baseline model
    model_out = 'models/hcm_logo_baseline_model.joblib' if os.path.exists('models') else 'hcm_logo_baseline_model.joblib'
    joblib.dump(model, model_out)
    print(f"Saved baseline model to '{model_out}'")
    
    print(f"\nMonth 1 execution completed in {time.time() - start_time:.2f} seconds.")
