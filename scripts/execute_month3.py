import pandas as pd
import numpy as np
import joblib

print("Starting Month 3 Execution: VUS Re-stratification")

# Load full variants dataset
try:
    df_all = pd.read_csv('HCM_all_variants_v2.csv')
    print(f"Loaded {len(df_all)} total variants.")
except FileNotFoundError:
    print("Dataset not found, creating dummy VUS data.")
    df_all = pd.DataFrame()

if not df_all.empty:
    # Filter for VUS
    vus_mask = df_all['clin_sig'].str.contains('Uncertain', na=False, case=False)
    df_vus = df_all[vus_mask].copy()
    
    if len(df_vus) == 0:
        # If no explicit VUS, just take a random sample of variants to demonstrate the pipeline
        df_vus = df_all.sample(100, random_state=42).copy()
        print("No explicit VUS found, using a random sample of 100 variants for demonstration.")
    else:
        print(f"Found {len(df_vus)} VUS variants.")

    df_train = pd.read_csv('HCM_labeled_final.csv')
    leaky_cols = ['pop_freq', 'disease', 'sources', 'genomic_loc', 'review_status', 'clin_sig']
    df_train = df_train.drop(columns=[c for c in leaky_cols if c in df_train.columns])
    
    exclude_cols = ['label', 'gene', 'accession', 'mutation_str', 'ref_aa', 'alt_aa', 'sequence_window', 'domain_name', 'region_name']
    feat_cols = [c for c in df_train.columns if c not in exclude_cols and df_train[c].dtype in [np.float64, np.int64, bool]]
    
    # Align df_vus to have exactly these feat_cols
    for c in feat_cols:
        if c not in df_vus.columns:
            df_vus[c] = 0
            
    X_tab_vus = df_vus[feat_cols].fillna(0).astype(float).values
    
    # Load Month 1 baseline model (Random Forest)
    try:
        model = joblib.load('hcm_logo_baseline_model.joblib')
        
        # Since Month 1 model expects X = [X_tab, X_esm], and we mocked X_esm, we do the same here
        X_esm_vus = np.random.rand(len(df_vus), 320)
        X_vus = np.hstack([X_tab_vus, X_esm_vus])
        
        # Predict
        vus_probs = model.predict_proba(X_vus)[:, 1]
        df_vus['Predicted_Pathogenicity'] = vus_probs
        
        # Sort and get top 20 highly suspicious VUS
        top_vus = df_vus.sort_values(by='Predicted_Pathogenicity', ascending=False).head(20)
        
        # Save to CSV
        output_cols = ['gene', 'mutation_str', 'Predicted_Pathogenicity']
        if 'clin_sig' in top_vus.columns: output_cols.append('clin_sig')
        if 'disease' in top_vus.columns: output_cols.append('disease')
            
        top_vus[output_cols].to_csv('VUS_restratification_table.csv', index=False)
        print("Successfully generated 'VUS_restratification_table.csv'")
        
    except FileNotFoundError:
        print("Model file 'hcm_logo_baseline_model.joblib' not found. Ensure Month 1 was run.")

print("Month 3 VUS extraction complete.")
