"""
HCM Classifier — Live Inference Backend
Loads the trained Two-Tower PyTorch model and serves real-time predictions via REST API.
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS

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
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'hcm_final_two_tower_model.pth')
CALIBRATOR_PATH = os.path.join(ROOT_DIR, 'models', 'hcm_platt_calibrator.pkl')

app = Flask(__name__)
CORS(app)  # Allow frontend to call this API

# ─── Load Models on Startup ─────────────────────────────────────────────────
print("Loading Two-Tower model...")
two_tower = HybridHCMModel(tabular_dim=TABULAR_DIM, esm_dim=ESM_DIM)
two_tower.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
two_tower.eval()
print("Two-Tower model loaded.")

print("Loading Platt calibrator...")
with open(CALIBRATOR_PATH, 'rb') as f:
    calibrator_raw = pickle.load(f)
# Extract coefficients to avoid sklearn version mismatch issues
PLATT_COEF = float(calibrator_raw.coef_[0][0])
PLATT_INTERCEPT = float(calibrator_raw.intercept_[0])
print(f"Platt calibrator loaded. coef={PLATT_COEF:.4f}, intercept={PLATT_INTERCEPT:.4f}")

def platt_calibrate(raw_prob):
    """Manual Platt scaling: sigmoid(coef * x + intercept)"""
    import math
    logit = PLATT_COEF * raw_prob + PLATT_INTERCEPT
    return 1.0 / (1.0 + math.exp(-logit))

# ─── Try to load ESM-2 (optional, for full inference) ───────────────────────
esm_model = None
esm_tokenizer = None
try:
    from transformers import EsmTokenizer, EsmModel
    ESM_MODEL_NAME = 'facebook/esm2_t33_650M_UR50D'
    print(f"Loading ESM-2 model ({ESM_MODEL_NAME})... This may take 30-60 seconds.")
    esm_tokenizer = EsmTokenizer.from_pretrained(ESM_MODEL_NAME)
    esm_model = EsmModel.from_pretrained(ESM_MODEL_NAME)
    esm_model.eval()
    print("ESM-2 model loaded successfully.")
except ImportError:
    print("WARNING: transformers not installed. Using zero-vector ESM fallback.")
except Exception as e:
    print(f"WARNING: Could not load ESM-2: {e}. Using zero-vector ESM fallback.")


def compute_esm_delta(wt_seq, mut_seq):
    """Compute the delta embedding between wildtype and mutant sequences."""
    if esm_model is None or esm_tokenizer is None:
        # Fallback: return zeros (model will rely on tabular features only)
        return np.zeros((1, ESM_DIM), dtype=np.float32)
    
    with torch.no_grad():
        inputs_wt = esm_tokenizer([wt_seq], return_tensors="pt", padding=True, truncation=True)
        inputs_mut = esm_tokenizer([mut_seq], return_tensors="pt", padding=True, truncation=True)
        out_wt = esm_model(**inputs_wt).last_hidden_state.mean(dim=1).numpy()
        out_mut = esm_model(**inputs_mut).last_hidden_state.mean(dim=1).numpy()
    return out_mut - out_wt


def compute_tabular_features(sequence, position):
    """
    Compute basic tabular features from a raw 11-AA sequence window.
    This approximates the features used during training.
    """
    AA_SIZE = {'G':1,'A':1,'S':1,'T':1,'C':1,'P':2,'V':2,'I':2,'L':2,'M':2,
               'F':3,'Y':3,'W':3,'D':1,'E':2,'N':2,'Q':2,'H':2,'K':2,'R':3}
    AA_CHARGE = {'R':1,'K':1,'H':1,'D':-1,'E':-1}
    
    features = np.zeros(TABULAR_DIM, dtype=np.float32)
    
    if len(sequence) != 11:
        return features.reshape(1, -1)
    
    ref_aa = sequence[5]  # Center of the 11-mer window
    
    # position, protein_length
    features[0] = position
    features[1] = 200  # approximate for MYL3
    
    # ref_size, ref_charge
    features[2] = AA_SIZE.get(ref_aa, 2)
    features[3] = AA_CHARGE.get(ref_aa, 0)
    
    # alt_size, alt_charge (unknown for custom, use ref as placeholder)
    features[4] = AA_SIZE.get(ref_aa, 2)
    features[5] = AA_CHARGE.get(ref_aa, 0)
    
    # size_change, charge_change
    features[6] = 0
    features[7] = 0
    
    # grantham_score
    features[8] = 50  # moderate default
    
    # rel_position
    features[9] = position / 200.0 if position > 0 else 0.5
    
    # in_domain through in_ptm_site (indices 10-20) — default 0
    
    # Window features: win_-5 through win_+5 (size and charge)
    for i, aa in enumerate(sequence):
        base_idx = 21 + (i * 2)
        if base_idx + 1 < TABULAR_DIM:
            features[base_idx] = AA_SIZE.get(aa, 2)
            features[base_idx + 1] = AA_CHARGE.get(aa, 0)
    
    # Gene one-hot (indices 43-51): default all zeros (unknown gene)
    
    return features.reshape(1, -1)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict pathogenicity for a given sequence.
    
    Expected JSON body:
    {
        "sequence": "ACDEFGHIKLM",   // 11-AA window
        "position": 56               // residue position (optional, default 100)
    }
    """
    data = request.get_json()
    sequence = data.get('sequence', '').upper()
    position = data.get('position', 100)
    
    if len(sequence) != 11 or not sequence.isalpha():
        return jsonify({'error': 'Sequence must be exactly 11 uppercase amino acid letters.'}), 400
    
    # 1. Compute tabular features
    X_tab = compute_tabular_features(sequence, position)
    
    # 2. Compute ESM-2 delta embedding
    # For a custom sequence, treat the center residue as the "mutation site"
    # WT = original sequence, MUT = same sequence (delta ≈ 0 without real WT context)
    # In practice, the user provides the mutant window, so we approximate:
    wt_seq = sequence  # wildtype approximation
    mut_seq = sequence  # mutant is the input
    X_esm = compute_esm_delta(wt_seq, mut_seq)
    
    # 3. Run Two-Tower model
    with torch.no_grad():
        tab_t = torch.FloatTensor(X_tab)
        esm_t = torch.FloatTensor(X_esm)
        raw_prob = two_tower(tab_t, esm_t).numpy().flatten()[0]
    
    # 4. Apply Platt scaling calibration (manual sigmoid to avoid sklearn version issues)
    raw_prob_clipped = float(np.clip(raw_prob, 1e-7, 1 - 1e-7))
    calibrated_prob = platt_calibrate(raw_prob_clipped)
    
    is_pathogenic = bool(calibrated_prob > 0.5)
    
    return jsonify({
        'sequence': sequence,
        'position': position,
        'raw_score': float(raw_prob),
        'calibrated_score': float(calibrated_prob),
        'prediction': 'Pathogenic' if is_pathogenic else 'Benign',
        'is_pathogenic': is_pathogenic,
        'model': 'Two-Tower Neural Network + Platt Calibration',
        'esm_status': 'live' if esm_model is not None else 'fallback (zero-vector)'
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'two_tower': 'loaded',
        'calibrator': 'loaded',
        'esm2': 'loaded' if esm_model is not None else 'not loaded (fallback mode)'
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  HCM Classifier Backend — Ready")
    print("  POST /predict   — Run live inference")
    print("  GET  /health    — Check model status")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
