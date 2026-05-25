"""
generate_interpretability_plots.py
====================================
Produces two model-interpretability figures for the HCM stacked ensemble:

1. SHAP summary beeswarm plot
   – Uses TreeExplainer on the XGBoost base model.
   – X_tabular = the 22 window (size/charge) features for all 1,954 variants.
   – Saved as  shap_beeswarm.png

2. Integrated-Gradients saliency heatmap
   – Computes IG w.r.t. the CNN sequence-input branch.
   – Averaged over all correctly-classified Pathogenic variants.
   – Positions are labelled −5 … +5; the mutation site is +0.
   – Saved as  ig_heatmap.png

Usage
-----
    python generate_interpretability_plots.py

Both PNG files are written to the same directory as this script.
"""

import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import joblib
import shap
import tensorflow as tf

# ── constants ──────────────────────────────────────────────────────────────────
WINDOW_HALF  = 5          # residues on each side of the mutation site (+0)
WINDOW_SIZE  = 2 * WINDOW_HALF + 1   # = 11 total positions

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(SCRIPT_DIR, "hcm_final_super_model.joblib")
DATA_PATH    = os.path.join(SCRIPT_DIR, "HCM_labeled_final.csv")
OUT_SHAP     = os.path.join(SCRIPT_DIR, "shap_beeswarm.png")
OUT_IG       = os.path.join(SCRIPT_DIR, "ig_heatmap.png")

# ── load artefacts ──────────────────────────────────────────────────────────────
print("Loading model …")
bundle = joblib.load(MODEL_PATH)
xgb_model        = bundle["xgb"]
cnn_model         = bundle["cnn"]
cnn_scaler_seq    = bundle["cnn_scaler_seq"]
cnn_scaler_aux    = bundle["cnn_scaler_aux"]
feature_names_seq = bundle["feature_names_seq"]   # 22 window features
feature_names_aux = bundle["feature_names_aux"]   # 29 auxiliary features
threshold         = float(bundle["threshold"])

print("Loading data …")
df = pd.read_csv(DATA_PATH)

# ── prepare tabular feature matrices ───────────────────────────────────────────
X_seq = df[feature_names_seq].values.astype(float)   # (N, 22)
X_aux = df[feature_names_aux].values.astype(float)   # (N, 29)
y     = df["label"].values                            # 0 = benign, 1 = pathogenic
N     = len(df)
print(f"Dataset: {N} variants  (benign={int((y==0).sum())}, pathogenic={int((y==1).sum())})")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SHAP beeswarm plot – XGBoost
# ══════════════════════════════════════════════════════════════════════════════
print("\n── SHAP analysis (XGBoost) ──")

# XGBoost was trained on the 29 auxiliary (tabular) features, not the window features.
X_aux_scaled_df = pd.DataFrame(
    cnn_scaler_aux.transform(X_aux), columns=feature_names_aux
)
X_tabular = pd.DataFrame(X_aux, columns=feature_names_aux)

explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer(X_tabular)          # Explanation object  (N, 29)

fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(
    shap_values,
    X_tabular,
    plot_type="dot",          # beeswarm
    show=False,
    max_display=29,
    plot_size=None,
)
plt.title(
    "SHAP Summary – XGBoost feature contributions\n"
    f"({N} variants, 29 tabular features)",
    fontsize=11,
)
plt.tight_layout()
plt.savefig(OUT_SHAP, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {OUT_SHAP}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Integrated-Gradients heatmap – CNN sequence branch
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Integrated Gradients (CNN sequence branch) ──")

# Scale inputs the same way the CNN expects them
X_seq_scaled = cnn_scaler_seq.transform(X_seq)           # (N, 22)
X_aux_scaled = cnn_scaler_aux.transform(X_aux)           # (N, 29)

# Reshape sequence input → (N, 11, 2)  [11 positions × (size, charge)]
X_seq_3d = X_seq_scaled.reshape(N, WINDOW_SIZE, 2).astype(np.float32)
X_aux_2d = X_aux_scaled.astype(np.float32)

# Identify correctly-classified Pathogenic variants
path_idx   = np.where(y == 1)[0]
seq_path   = tf.constant(X_seq_3d[path_idx])          # (M, 11, 2)
aux_path   = tf.constant(X_aux_2d[path_idx])          # (M, 29)

# Predict probabilities to find correctly-classified ones
probs = cnn_model.predict(
    [seq_path.numpy(), aux_path.numpy()], verbose=0
).ravel()
pred_labels       = (probs >= threshold).astype(int)
correct_path_mask = pred_labels == 1
n_correct         = int(correct_path_mask.sum())
print(f"  Pathogenic variants: {len(path_idx)}  |  correctly classified: {n_correct}")

seq_correct = tf.constant(X_seq_3d[path_idx][correct_path_mask])   # (K, 11, 2)
aux_correct = tf.constant(X_aux_2d[path_idx][correct_path_mask])   # (K, 29)


def integrated_gradients_single(seq_input, aux_input, baseline_seq=None, steps=50):
    """Return IG attribution for one sample w.r.t. the sequence branch."""
    if baseline_seq is None:
        baseline_seq = tf.zeros_like(seq_input)               # (11, 2) baseline

    # Interpolate between baseline and input
    alphas       = tf.linspace(0.0, 1.0, steps + 1)          # (steps+1,)
    interp_seq   = baseline_seq + alphas[:, None, None] * (seq_input - baseline_seq)
    # aux is held constant at the actual value
    tiled_aux    = tf.tile(aux_input[None], [steps + 1, 1])   # (steps+1, 29)

    with tf.GradientTape() as tape:
        tape.watch(interp_seq)
        preds = cnn_model([interp_seq, tiled_aux], training=False)  # (steps+1, 1)

    grads   = tape.gradient(preds, interp_seq)                # (steps+1, 11, 2)
    avg_grads = tf.reduce_mean(grads, axis=0)                 # (11, 2)
    ig        = (seq_input - baseline_seq) * avg_grads        # (11, 2)
    return ig.numpy()


print("  Computing integrated gradients …")
K = n_correct
ig_all = np.zeros((K, WINDOW_SIZE, 2), dtype=np.float32)

for i in range(K):
    ig_all[i] = integrated_gradients_single(
        seq_correct[i],    # (11, 2)
        aux_correct[i],    # (29,)
    )
    if (i + 1) % max(1, K // 10) == 0 or i == K - 1:
        print(f"    {i+1}/{K}", end="\r")
print()

# Mean absolute IG over the K variants → (11, 2)
mean_ig = np.mean(np.abs(ig_all), axis=0)   # (11, 2)

# ── plot ──────────────────────────────────────────────────────────────────────
positions    = [f"{i:+d}" if i != 0 else "0\n(mut)" for i in range(-WINDOW_HALF, WINDOW_HALF + 1)]
channel_lbl  = ["Size", "Charge"]

fig, ax = plt.subplots(figsize=(10, 3))
im = ax.imshow(
    mean_ig.T,                    # (2, 11) so rows=channels, cols=positions
    aspect="auto",
    cmap="hot",
    interpolation="nearest",
)
ax.set_xticks(range(WINDOW_SIZE))
ax.set_xticklabels(positions, fontsize=10)
ax.set_yticks([0, 1])
ax.set_yticklabels(channel_lbl, fontsize=10)
ax.set_xlabel("Residue position relative to mutation site", fontsize=11)
ax.set_title(
    "Integrated-Gradients saliency – CNN sequence branch\n"
    f"Mean |IG| over {n_correct} correctly-classified Pathogenic variants",
    fontsize=11,
)
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("Mean |IG|", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_IG, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {OUT_IG}")

print("\nDone.  Both plots written successfully.")
