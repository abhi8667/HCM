"""
HCM Classifier — Live Inference Backend
Loads the trained Two-Tower PyTorch model, Random Forest model, and serves real-time predictions via REST API.
"""

import os
import sys
import json
import pickle
import logging
import time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# ─── Configure Logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ─── Model Definition (must match training architecture exactly) ─────────────
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


# ─── Globals ─────────────────────────────────────────────────────────────────
TABULAR_DIM = 52
ESM_DIM = 1280
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'hcm_final_two_tower_model.pth')
CALIBRATOR_PATH = os.path.join(ROOT_DIR, 'models', 'hcm_platt_calibrator.pkl')
RF_MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'hcm_final_rf_model.joblib')
DB_PATH = os.path.join(os.path.dirname(__file__), 'data_processed.json')

GENE_LENGTHS = {
    'MYH7': 1935,
    'MYBPC3': 1274,
    'TNNT2': 297,
    'TNNI3': 210,
    'TPM1': 284,
    'MYL2': 166,
    'MYL3': 195,
    'ACTC1': 377,
    'TNNC1': 161
}

TARGET_GENES = ['MYH7', 'MYBPC3', 'TNNT2', 'TNNI3', 'TPM1', 'MYL2', 'MYL3', 'ACTC1', 'TNNC1']

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

ROWS = "ARNDCQEGHILKMFPSTWYV"
GRANTHAM_RAW = [
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

TABULAR_FEATURE_NAMES = [
    "position", "protein_length", "ref_size", "ref_charge", "alt_size", "alt_charge",
    "size_change", "charge_change", "grantham_score", "rel_position",
    "in_domain", "in_coiled", "in_helix", "in_strand", "in_turn", "in_secondary",
    "in_region", "in_disordered", "in_compbias", "in_functional_site", "in_ptm_site"
]
for i in range(-5, 6):
    sign = "+" if i >= 0 else ""
    TABULAR_FEATURE_NAMES.extend([f"win_{sign}{i}_size", f"win_{sign}{i}_charge"])
TABULAR_FEATURE_NAMES.extend([f"is_{g}" for g in TARGET_GENES])


def get_grantham(aa1, aa2):
    if aa1 not in ROWS or aa2 not in ROWS:
        return 0
    i = ROWS.index(aa1)
    j = ROWS.index(aa2)
    if i >= j:
        return GRANTHAM_RAW[i][j]
    else:
        return GRANTHAM_RAW[j][i]


# ─── Initialize Flask & CORS ─────────────────────────────────────────────────
app = Flask(__name__)
# Restrict origins for local development and standard ports
CORS(app, origins=[
    "http://localhost:3000", "http://127.0.0.1:3000",
    "http://localhost:5000", "http://127.0.0.1:5000",
    "http://localhost:8000", "http://127.0.0.1:8000",
    "http://localhost:5500", "http://127.0.0.1:5500",
    "http://localhost:5173", "http://127.0.0.1:5173"
])

# ─── Load Dynamic Database ──────────────────────────────────────────────────
GENE_SEQUENCES = {}
STRUCTURAL_DB = {}
if os.path.exists(DB_PATH):
    try:
        logger.info(f"Loading preprocessed sequence and structural database: {DB_PATH}")
        with open(DB_PATH, 'r') as f:
            db_data = json.load(f)
        GENE_SEQUENCES = db_data.get('sequences', {})
        STRUCTURAL_DB = db_data.get('structural_db', {})
        logger.info("Database loaded successfully.")
    except Exception as db_err:
        logger.error(f"Error loading database: {db_err}")
else:
    logger.warning(f"Database not found at {DB_PATH}. Custom sequence windows must be supplied manually.")

# ─── Load Models on Startup ─────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Loading models on device: {device}")

two_tower = None
calibrator = None
rf_model = None

try:
    logger.info("Loading Two-Tower model weights...")
    two_tower = HybridHCMModel(tabular_dim=TABULAR_DIM, esm_dim=ESM_DIM)
    two_tower.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    two_tower.eval()
    logger.info("Two-Tower model loaded successfully.")
except Exception as e:
    logger.error(f"Could not load Two-Tower model: {e}")

try:
    logger.info("Loading Platt calibrator...")
    with open(CALIBRATOR_PATH, 'rb') as f:
        calibrator_raw = pickle.load(f)
    PLATT_COEF = float(calibrator_raw.coef_[0][0])
    PLATT_INTERCEPT = float(calibrator_raw.intercept_[0])
    logger.info(f"Platt calibrator loaded. coef={PLATT_COEF:.4f}, intercept={PLATT_INTERCEPT:.4f}")
except Exception as e:
    logger.error(f"Could not load Platt calibrator: {e}")
    # Fallback to standard sigmoid coefficients
    PLATT_COEF = 1.0
    PLATT_INTERCEPT = 0.0

try:
    logger.info("Loading Random Forest model...")
    rf_model = joblib.load(RF_MODEL_PATH)
    logger.info("Random Forest model loaded successfully.")
except Exception as e:
    logger.error(f"Could not load Random Forest model: {e}")

def platt_calibrate(raw_prob):
    """Manual Platt scaling: sigmoid(coef * x + intercept)"""
    import math
    logit = PLATT_COEF * raw_prob + PLATT_INTERCEPT
    return 1.0 / (1.0 + math.exp(-logit))

# ─── Load ESM-2 (quantized for CPU, float16 for GPU) ─────────────────────────
esm_model = None
esm_tokenizer = None

try:
    from transformers import EsmTokenizer, EsmModel
    ESM_MODEL_NAME = 'facebook/esm2_t33_650M_UR50D'
    logger.info(f"Loading ESM-2 ({ESM_MODEL_NAME})...")
    esm_tokenizer = EsmTokenizer.from_pretrained(ESM_MODEL_NAME)
    
    if torch.cuda.is_available():
        esm_model = EsmModel.from_pretrained(ESM_MODEL_NAME, torch_dtype=torch.float16).to(device)
        logger.info("ESM-2 loaded in float16 on GPU.")
    else:
        esm_model = EsmModel.from_pretrained(ESM_MODEL_NAME)
        logger.info("Applying 8-bit dynamic quantization to ESM-2 on CPU to reduce memory (from 2.6GB to 650MB)...")
        esm_model = torch.quantization.quantize_dynamic(
            esm_model, {nn.Linear}, dtype=torch.qint8
        )
        logger.info("ESM-2 loaded and quantized successfully on CPU.")
    esm_model.eval()
except ImportError:
    logger.warning("transformers not installed. Using zero-vector ESM fallback.")
except Exception as e:
    logger.error(f"Could not load ESM-2: {e}. Using zero-vector ESM fallback.")


# ─── Rate Limiter ─────────────────────────────────────────────────────────────
request_history = defaultdict(list)
RATE_LIMIT_LIMIT = 60  # max requests
RATE_LIMIT_WINDOW = 60.0  # seconds

def is_rate_limited(ip):
    now = time.time()
    request_history[ip] = [t for t in request_history[ip] if now - t < RATE_LIMIT_WINDOW]
    if len(request_history[ip]) >= RATE_LIMIT_LIMIT:
        return True
    request_history[ip].append(now)
    return False


# ─── Feature Processing ───────────────────────────────────────────────────────
def get_sequence_window(gene, position):
    """Retrieve or construct the 11-AA sequence window from wildtype sequence."""
    seq = GENE_SEQUENCES.get(gene)
    if not seq:
        return None
    p_idx = position - 1  # 0-indexed
    window = []
    for i in range(-5, 6):
        idx = p_idx + i
        if 0 <= idx < len(seq):
            window.append(seq[idx])
        else:
            window.append('X')  # padding
    return "".join(window)


def get_structural_annotations(gene, position):
    """Retrieve structural annotations for a gene position with fallback to closest position."""
    gene_db = STRUCTURAL_DB.get(gene, {})
    pos_str = str(position)
    if pos_str in gene_db:
        return gene_db[pos_str]
    
    # Fallback to closest available position
    available_positions = [int(p) for p in gene_db.keys()]
    if available_positions:
        closest_pos = min(available_positions, key=lambda x: abs(x - position))
        logger.info(f"Position {position} not found in structural DB for {gene}. Falling back to closest position {closest_pos}.")
        return gene_db[str(closest_pos)]
    
    # Default 11 zeros: in_domain, in_coiled, in_helix, in_strand, in_turn, in_secondary, in_region, in_disordered, in_compbias, in_functional_site, in_ptm_site
    return [0] * 11


def compute_esm_delta(wt_seq, mut_seq):
    """Compute the delta embedding (out_mut - out_wt) between wildtype and mutant sequences."""
    if esm_model is None or esm_tokenizer is None:
        return np.zeros((1, ESM_DIM), dtype=np.float32)
    
    with torch.no_grad():
        inputs_wt = esm_tokenizer([wt_seq], return_tensors="pt", padding=True, truncation=True).to(device)
        inputs_mut = esm_tokenizer([mut_seq], return_tensors="pt", padding=True, truncation=True).to(device)
        
        # Determine output extraction (GPU uses float16, CPU uses float32)
        out_wt = esm_model(**inputs_wt).last_hidden_state.mean(dim=1)
        out_mut = esm_model(**inputs_mut).last_hidden_state.mean(dim=1)
        
        delta = (out_mut - out_wt).cpu().numpy()
    return delta


def compute_tabular_features(gene, position, ref_aa, alt_aa, sequence_window):
    """Build the exact 52-dimensional feature vector matching training."""
    features = np.zeros(TABULAR_DIM, dtype=np.float32)
    
    # Basic Position and Length
    protein_length = GENE_LENGTHS.get(gene, 200)
    features[0] = float(position)
    features[1] = float(protein_length)
    
    # Reference and Mutant AA Biophysical Features
    ref_size = AA_SIZES.get(ref_aa, 0)
    ref_charge = AA_CHARGES.get(ref_aa, 0)
    alt_size = AA_SIZES.get(alt_aa, 0)
    alt_charge = AA_CHARGES.get(alt_aa, 0)
    
    features[2] = float(ref_size)
    features[3] = float(ref_charge)
    features[4] = float(alt_size)
    features[5] = float(alt_charge)
    features[6] = float(alt_size - ref_size)
    features[7] = float(alt_charge - ref_charge)
    
    # Grantham Score
    features[8] = float(get_grantham(ref_aa, alt_aa))
    
    # Relative Position
    features[9] = float(position / protein_length) if protein_length > 0 else 0.5
    
    # Structural Annotations (indices 10 to 20)
    struct_vals = get_structural_annotations(gene, position)
    for i, val in enumerate(struct_vals):
        features[10 + i] = float(val)
        
    # Window Residue Features: win_-5 to win_+5 (size and charge)
    # Mutation affects index 5 (position +0)
    for i, aa in enumerate(sequence_window):
        base_idx = 21 + (i * 2)
        if base_idx + 1 < TABULAR_DIM:
            if i == 5:
                # Mutation site size & charge
                features[base_idx] = float(alt_size)
                features[base_idx + 1] = float(alt_charge)
            else:
                # Wildtype flanking residue size & charge (padding 'X' is 0)
                features[base_idx] = float(AA_SIZES.get(aa, 0))
                features[base_idx + 1] = float(AA_CHARGES.get(aa, 0))
                
    # Gene One-Hot Encoding (indices 43 to 51)
    if gene in TARGET_GENES:
        gene_idx = 43 + TARGET_GENES.index(gene)
        features[gene_idx] = 1.0
        
    return features.reshape(1, -1)


# ─── API Routes ──────────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    """
    Run pathogenicity prediction for a specific variant.
    Expected JSON schema matching training:
    {
        "gene": "MYL3",
        "position": 152,
        "ref_aa": "E",
        "alt_aa": "K",
        "sequence_window": "TVMGAELRHVL"   // Optional, auto-resolved if omitted
    }
    """
    # 1. Rate Limiting Check
    ip = request.remote_addr
    if is_rate_limited(ip):
        logger.warning(f"Rate limit hit for IP: {ip}")
        return jsonify({'error': 'Rate limit exceeded. Max 60 requests per minute.'}), 429

    # 2. Extract and Parse Payload
    data = request.get_json() or {}
    gene = data.get('gene', '').upper().strip()
    position = data.get('position')
    ref_aa = data.get('ref_aa', '').upper().strip()
    alt_aa = data.get('alt_aa', '').upper().strip()
    sequence_window = data.get('sequence_window', '').upper().strip()

    # 3. Input Validation
    errors = []
    if not gene:
        errors.append("Missing required field 'gene'.")
    elif gene not in TARGET_GENES:
        errors.append(f"Invalid gene '{gene}'. Supported: {', '.join(TARGET_GENES)}.")

    if position is None:
        errors.append("Missing required field 'position'.")
    else:
        try:
            position = int(position)
            if position <= 0:
                errors.append("'position' must be a positive integer.")
            elif gene in GENE_LENGTHS and position > GENE_LENGTHS[gene]:
                errors.append(f"Position {position} exceeds maximum residue position for {gene} ({GENE_LENGTHS[gene]}).")
        except ValueError:
            errors.append("'position' must be a valid integer.")

    if not ref_aa:
        errors.append("Missing required field 'ref_aa'.")
    elif len(ref_aa) != 1 or ref_aa not in ROWS:
        errors.append(f"Invalid 'ref_aa' '{ref_aa}'. Must be a single uppercase character from: {ROWS}.")

    if not alt_aa:
        errors.append("Missing required field 'alt_aa'.")
    elif len(alt_aa) != 1 or alt_aa not in ROWS:
        errors.append(f"Invalid 'alt_aa' '{alt_aa}'. Must be a single uppercase character from: {ROWS}.")

    if ref_aa and alt_aa and ref_aa == alt_aa:
        errors.append("Wildtype (ref_aa) and Mutant (alt_aa) amino acids cannot be identical.")

    # Auto-resolve sequence window if not provided
    if gene and position and not errors:
        resolved_window = get_sequence_window(gene, position)
        if not sequence_window:
            if resolved_window:
                sequence_window = resolved_window
                logger.info(f"Auto-resolved sequence window for {gene} p.{position}: {sequence_window}")
            else:
                errors.append("Could not auto-resolve sequence window. Please supply 'sequence_window' manually.")
        else:
            if len(sequence_window) != 11:
                errors.append("'sequence_window' must be exactly 11 amino acids.")
            elif not all(c in ROWS or c == 'X' for c in sequence_window):
                errors.append(f"Invalid characters in 'sequence_window'. Must contain only standard amino acids or 'X' padding.")
            elif resolved_window and sequence_window[5] != ref_aa:
                errors.append(f"Mutation consistency error: center residue of window ({sequence_window[5]}) does not match ref_aa ({ref_aa}).")

    if errors:
        return jsonify({'error': 'Input validation failed', 'messages': errors}), 400

    # 4. Feature Extraction & ESM Inference
    # Build wildtype and mutated 11-AA sequence windows matching training ESM inputs
    wt_seq = sequence_window
    mut_seq = sequence_window[:5] + alt_aa + sequence_window[6:]
    
    logger.info(f"Running inference for variant {gene} p.{ref_aa}{position}{alt_aa}")
    logger.info(f"ESM-2 Inputs: WT={wt_seq}, MUT={mut_seq}")

    X_tab = compute_tabular_features(gene, position, ref_aa, alt_aa, sequence_window)
    X_esm = compute_esm_delta(wt_seq, mut_seq)

    # 5. Run Two-Tower Model Prediction & Platt Scale Calibration
    calibrated_prob = 0.5
    raw_prob = 0.5
    if two_tower is not None:
        try:
            with torch.no_grad():
                tab_t = torch.FloatTensor(X_tab)
                esm_t = torch.FloatTensor(X_esm)
                raw_prob = float(two_tower(tab_t, esm_t).numpy().flatten()[0])
            raw_prob_clipped = float(np.clip(raw_prob, 1e-7, 1 - 1e-7))
            calibrated_prob = platt_calibrate(raw_prob_clipped)
        except Exception as pred_err:
            logger.error(f"Error running Two-Tower prediction: {pred_err}")
            
    is_pathogenic = calibrated_prob > 0.5

    # 6. Run Random Forest Model (Explainability Tower)
    rf_prob = 0.5
    explanations = []
    if rf_model is not None:
        try:
            X_rf = np.hstack([X_tab, X_esm])
            rf_prob = float(rf_model.predict_proba(X_rf)[0, 1])
            
            # Extract Feature Importances
            # Group ESM dimensions (dim 52 to 1331) together
            feat_importances = rf_model.feature_importances_
            esm_importance = float(np.sum(feat_importances[52:]))
            
            importances_dict = {"ESM-2 Delta Embedding": esm_importance}
            for i, name in enumerate(TABULAR_FEATURE_NAMES):
                importances_dict[name] = float(feat_importances[i])
                
            sorted_importances = sorted(importances_dict.items(), key=lambda x: x[1], reverse=True)
            
            # Map top features with local actual values for local explainability context
            for name, imp in sorted_importances[:6]:
                val_str = "Active"
                if name == "grantham_score":
                    val_str = str(int(X_tab[0, 8]))
                elif name in TABULAR_FEATURE_NAMES:
                    idx = TABULAR_FEATURE_NAMES.index(name)
                    # format floats and handle integers
                    v = X_tab[0, idx]
                    val_str = f"{v:.3f}" if isinstance(v, float) and abs(v - int(v)) > 1e-5 else str(int(v))
                
                explanations.append({
                    "feature": name,
                    "importance": float(imp),
                    "value": val_str
                })
        except Exception as rf_err:
            logger.error(f"Error running Random Forest explainability: {rf_err}")

    # 7. Confidence Calibration Score
    # High: >= 0.85 or <= 0.15; Medium: >= 0.70 or <= 0.30; Low: otherwise
    if calibrated_prob >= 0.85 or calibrated_prob <= 0.15:
        confidence = "High"
    elif calibrated_prob >= 0.70 or calibrated_prob <= 0.30:
        confidence = "Medium"
    else:
        confidence = "Low"

    # 8. Return JSON Response
    return jsonify({
        'gene': gene,
        'position': position,
        'ref_aa': ref_aa,
        'alt_aa': alt_aa,
        'sequence_window': sequence_window,
        'raw_score': float(raw_prob),
        'calibrated_score': float(calibrated_prob),
        'rf_score': float(rf_prob),
        'prediction': 'Pathogenic' if is_pathogenic else 'Benign',
        'is_pathogenic': is_pathogenic,
        'confidence': confidence,
        'explanations': explanations,
        'model_version': 'v1.0',
        'model': 'Two-Tower Neural Network (Weighted BCE) + Platt Calibration',
        'esm_status': 'live' if esm_model is not None else 'fallback (zero-vector)'
    })


@app.route('/variants', methods=['GET'])
def get_presets():
    """Dynamically parses and serves presets from frontend/data.js."""
    variants = []
    data_js_path = os.path.join(ROOT_DIR, 'frontend', 'data.js')
    if os.path.exists(data_js_path):
        try:
            with open(data_js_path, 'r', encoding='utf-8') as file:
                content = file.read()
            # Extract array after first '='
            json_str = content.split('=', 1)[1].strip()
            if json_str.endswith(';'):
                json_str = json_str[:-1].strip()
            variants = json.loads(json_str)
            logger.info("Successfully loaded dynamic presets from data.js.")
        except Exception as e:
            logger.error(f"Could not parse data.js dynamically: {e}")
    else:
        logger.warning("data.js not found in frontend directory. Presets empty.")
        
    return jsonify({
        'variants': variants,
        'count': len(variants)
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_version': 'v1.0',
        'two_tower': 'loaded' if two_tower is not None else 'failed',
        'calibrator': 'loaded' if PLATT_COEF != 1.0 else 'fallback',
        'rf_model': 'loaded' if rf_model is not None else 'failed',
        'esm2': 'loaded (quantized on CPU)' if esm_model is not None and not torch.cuda.is_available() else ('loaded (GPU)' if esm_model is not None else 'fallback mode'),
        'device': str(device),
        'database_size_loaded': len(GENE_SEQUENCES)
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  HCM Classifier Backend — Ready")
    print("  POST /predict   — Run live inference")
    print("  GET  /variants  — Get variant presets")
    print("  GET  /health    — Check model health")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
