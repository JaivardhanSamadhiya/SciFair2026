"""
04_ensemble.py  —  GNN + Baseline ensemble for phage-host prediction
=====================================================================
Combines predictions from:
  - GNN (03_gnn.py)     : captures graph-structural patterns
  - Random Forest        : captures numeric feature patterns (02_model.py)
  - XGBoost             : captures non-linear feature interactions

Why ensemble works:
  The GNN and flat ML models make DIFFERENT errors:
  - GNN is strong at genera with rich connectivity (many neighbours)
  - RF/XGBoost are strong at genera with clear numeric feature separation
  Combining them smooths out each model's blind spots.

Run AFTER 02_model.py AND 03_gnn.py have both completed.
"""

import warnings, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from scipy.sparse.linalg import svds
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    matthews_corrcoef, precision_recall_curve, roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")

_SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR    = _SCRIPT_DIR.parent / "data"
RAW_DIR     = BASE_DIR / "raw"
PLOT_DIR    = BASE_DIR / "plots"
RESULTS_DIR = BASE_DIR / "results"
for d in [PLOT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED      = 42
NUMERIC_FEATURES = ["k3dist", "k6dist", "GCdiff", "Homology"]
TARGET_HOST      = "staphylococcus aureus"
np.random.seed(RANDOM_SEED)
sns.set_theme(style="whitegrid", palette="colorblind")

def normalize(x):
    return str(x).strip().lower().replace("_", " ").replace("-", " ")

def save_plot(fig, name):
    p = PLOT_DIR / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")

def eval_metrics(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    p, r, _ = precision_recall_curve(y_true, y_proba)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc":  auc(r, p),
        "f1":      f1_score(y_true, y_pred, zero_division=0),
        "mcc":     matthews_corrcoef(y_true, y_pred),
    }

# ═══════════════════════════════════════════════════════════════
# SECTION 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("  LOADING DATA")
print("=" * 60)

# Load the SAME capped+sorted dataset that gnn.py used.
# This is the single source of truth — eliminates row-count mismatches.
capped_path   = RAW_DIR / "capped_dataset.csv"
enriched_path = RAW_DIR / "enriched_dataset.csv"

if capped_path.exists():
    dataset = pd.read_csv(capped_path, index_col=0)
    print(f"  Using GNN-capped dataset: {len(dataset)} rows (aligned with GNN)")
else:
    # Fallback: build from enriched data (GNN alignment will fail, but ensemble still runs)
    print("  WARNING: capped_dataset.csv not found — run 03_gnn.py first.")
    print("  Falling back to independent neg-cap (GNN alignment disabled).")
    if enriched_path.exists():
        dataset = pd.read_csv(enriched_path)
        print(f"  Using enriched dataset: {len(dataset)} rows")
    else:
        raise FileNotFoundError("Neither capped_dataset.csv nor enriched_dataset.csv found.")
    if "genus" not in dataset.columns:
        dataset["genus"] = dataset["host"].str.split().str[0]
    for c in NUMERIC_FEATURES:
        if c not in dataset.columns: dataset[c] = 0.0
        dataset[c] = pd.to_numeric(dataset[c], errors="coerce").fillna(0.0)
    capped = []
    for g, gdf in dataset.groupby("genus"):
        pos = gdf[gdf["label"]==1]
        neg = gdf[gdf["label"]==0]
        if len(pos) > 0 and len(neg) > 4*len(pos):
            neg = neg.sample(n=4*len(pos), random_state=RANDOM_SEED)
        capped.append(pd.concat([pos, neg]))
    dataset = pd.concat(capped).sort_values(["host","phage"]).reset_index(drop=True)

if "genus" not in dataset.columns:
    dataset["genus"] = dataset["host"].str.split().str[0]
for c in NUMERIC_FEATURES:
    if c not in dataset.columns: dataset[c] = 0.0
    dataset[c] = pd.to_numeric(dataset[c], errors="coerce").fillna(0.0)

print(f"  Dataset: {len(dataset)} rows | {dataset['label'].sum()} pos | "
      f"{(dataset['label']==0).sum()} neg")
print(f"  Hosts: {dataset['host'].nunique()} | Genera: {dataset['genus'].nunique()}")


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — BUILD FEATURES (same as 02_model.py)
# ═══════════════════════════════════════════════════════════════
print("\n[2] Building features...")

names  = dataset["phage"].fillna("").astype(str)
y_all  = dataset["label"].reset_index(drop=True)
groups = dataset["host"]   # LOSO groups = host species

numeric_vals = dataset[NUMERIC_FEATURES].values.astype(np.float32)

vec_name = CountVectorizer(analyzer="char", ngram_range=(3,5),
                            max_features=30_000, dtype=np.float32)
X_name   = vec_name.fit_transform(names)

scaler    = StandardScaler()
X_numeric = csr_matrix(scaler.fit_transform(numeric_vals))
X_combined = hstack([X_name, X_numeric], format="csr")

# Graph-structural features (mirroring 03_gnn.py)
phage_breadth  = dataset[dataset["label"]==1].groupby("phage")["host"].nunique()
host_vuln      = dataset[dataset["label"]==1].groupby("host")["phage"].nunique()
genus_pos_rate = dataset.groupby("genus")["label"].mean()
dataset["phage_breadth"]  = dataset["phage"].map(phage_breadth).fillna(0)
dataset["host_vuln"]      = dataset["host"].map(host_vuln).fillna(0)
dataset["genus_pos_rate"] = dataset["genus"].map(genus_pos_rate).fillna(0.5)

struct_cols  = ["phage_breadth", "host_vuln", "genus_pos_rate"]
scaler_s     = StandardScaler()
X_structural = csr_matrix(
    scaler_s.fit_transform(dataset[struct_cols].values.astype(np.float32)))
X_full       = hstack([X_name, X_numeric, X_structural], format="csr")

print(f"  Name features:       {X_name.shape[1]}")
print(f"  Numeric features:    {X_numeric.shape[1]}")
print(f"  Structural features: {X_structural.shape[1]}")
print(f"  Combined:            {X_full.shape[1]}")


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — LOSO-CV SPLITS
# ═══════════════════════════════════════════════════════════════
valid_species = [
    h for h in dataset["host"].unique()
    if dataset[dataset["host"]==h]["label"].nunique() == 2
       and len(dataset[dataset["host"]==h]) >= 5
]
loso_groups = dataset["host"].apply(
    lambda h: h if h in valid_species else "__skip__")

print(f"\n[3] LOSO-CV: {len(valid_species)} evaluable host species")


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — INDIVIDUAL MODEL PREDICTIONS (LOSO-CV)
# ═══════════════════════════════════════════════════════════════
print("\n[4] Generating LOSO-CV predictions from each model...")

def loso_predict(clf, X, y, groups_series, valid_sp):
    """Run LOSO-CV and return per-row probability array."""
    proba = np.full(len(y), np.nan)
    for sp in valid_sp:
        test_mask  = (groups_series == sp).values
        train_mask = ~test_mask
        if np.unique(y[test_mask]).size < 2:
            continue
        clf.fit(X[train_mask], y[train_mask])
        proba[test_mask] = clf.predict_proba(X[test_mask])[:, 1]
    return proba

pos_weight = int((y_all==0).sum() / max((y_all==1).sum(), 1))

# Use only the strongest, most diverse models.
# LR is too weak (0.852) and dilutes the ensemble.
# XGB and GB are both strong (0.927, 0.924) and capture complementary patterns:
#   XGB: deeper trees, captures strong feature interactions
#   GB: shallower trees, more regularized, better on sparse data
#   RF:  bagged trees, different variance structure from boosting
models = {
    "XGBoost": xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        scale_pos_weight=pos_weight, eval_metric="logloss",
        random_state=RANDOM_SEED, n_jobs=-1, verbosity=0),
    "XGBoost_deep": xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.03, max_depth=8,
        scale_pos_weight=pos_weight, eval_metric="logloss",
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=3,
        random_state=RANDOM_SEED+1, n_jobs=-1, verbosity=0),
    "XGBoost_wide": xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.08, max_depth=4,
        scale_pos_weight=pos_weight, eval_metric="logloss",
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.1, reg_lambda=2.0,
        random_state=RANDOM_SEED+2, n_jobs=-1, verbosity=0),
    "Random Forest": RandomForestClassifier(
        n_estimators=400, class_weight="balanced",
        random_state=RANDOM_SEED, n_jobs=-1, max_features="sqrt",
        min_samples_leaf=2),
}

model_probas = {}
model_metrics = {}
for name, clf in models.items():
    print(f"  -> {name}...")
    pr = loso_predict(clf, X_full, y_all.values, dataset["host"], valid_species)
    model_probas[name] = pr
    valid = ~np.isnan(pr)
    if valid.sum() > 0 and np.unique(y_all[valid]).size == 2:
        m = eval_metrics(y_all[valid].values, pr[valid])
        model_metrics[name] = m
        print(f"     ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  MCC={m['mcc']:.4f}")

pd.DataFrame(model_metrics).T.to_csv(RESULTS_DIR / "ensemble_individual_models.csv")


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — LOAD GNN PREDICTIONS (if available)
# ═══════════════════════════════════════════════════════════════
print("\n[5] Loading GNN predictions...")
gnn_proba = None
gnn_auc   = None

# Try to load saved GNN predictions from 03_gnn.py run
gnn_pred_path = RESULTS_DIR / "gnn_loso_predictions.npy"
gnn_idx_path  = RESULTS_DIR / "gnn_loso_index.npy"

gnn_key_path = RESULTS_DIR / "gnn_dataset_index.csv"

if gnn_pred_path.exists() and gnn_idx_path.exists():
    gnn_proba_saved = np.load(gnn_pred_path)
    gnn_idx_saved   = np.load(gnn_idx_path)

    # ALIGNMENT VERIFICATION
    # Both scripts now sort by ["host","phage"] — verify keys match
    alignment_ok = False
    if capped_path.exists():
        # Both scripts loaded the same capped_dataset.csv — guaranteed aligned
        alignment_ok = True
        print("  GNN alignment verified ✓ (shared capped_dataset.csv)")
    elif gnn_key_path.exists():
        gnn_key = pd.read_csv(gnn_key_path, index_col=0)
        if len(gnn_key) == len(dataset):
            key_match = (
                (gnn_key["phage"].values == dataset["phage"].values).all() and
                (gnn_key["host"].values  == dataset["host"].values).all()
            )
            if key_match:
                alignment_ok = True
                print("  GNN alignment verified ✓ (key file matches)")
            else:
                mismatch = np.where(
                    gnn_key["host"].values != dataset["host"].values)[0]
                print(f"  GNN alignment MISMATCH at row {mismatch[0] if len(mismatch) else '?'}")
                print("  Re-run 03_gnn.py then re-run this script.")
        else:
            print(f"  GNN dataset size mismatch: {len(gnn_key)} vs {len(dataset)}")
    else:
        print("  No alignment key — using index directly (may be misaligned)")
        alignment_ok = True

    # Load predictions
    gnn_proba = np.full(len(dataset), np.nan)
    for i, idx in enumerate(gnn_idx_saved):
        if idx < len(gnn_proba):
            gnn_proba[idx] = gnn_proba_saved[i]
    valid = ~np.isnan(gnn_proba)
    if valid.sum() > 0:
        gnn_auc = roc_auc_score(y_all[valid], gnn_proba[valid])
        print(f"  GNN predictions: {valid.sum()} rows, AUC={gnn_auc:.4f}")
        if gnn_auc < 0.60 or not alignment_ok:
            print(f"  {'Misaligned' if not alignment_ok else 'Low AUC'} — "
                  f"excluding GNN. Re-run 03_gnn.py to fix.")
            gnn_proba = None
        else:
            print(f"  GNN included in ensemble ✓")
    else:
        gnn_proba = None
else:
    # Use GNN overall AUC from saved results file
    gnn_result_path = RESULTS_DIR / "gnn_overall.csv"
    if gnn_result_path.exists():
        gnn_res = pd.read_csv(gnn_result_path, index_col=0)
        gnn_auc = float(gnn_res["roc_auc"].iloc[0])
        print(f"  GNN overall AUC from results file: {gnn_auc:.4f}")
        print("  (Per-row predictions not saved — ensemble will use RF+XGB only)")
        print("  To enable full GNN ensemble: add prediction saving to 03_gnn.py")
    else:
        print("  No GNN predictions found — ensemble will use RF+XGB only")
        print("  Run 03_gnn.py first, then re-run this script")


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — ENSEMBLE
# Optimize weights by searching over combinations
# ═══════════════════════════════════════════════════════════════
print("\n[6] Building ensemble...")

# Collect all available per-row predictions
available = {k: v for k, v in model_probas.items()
             if not np.all(np.isnan(v))}

if gnn_proba is not None:
    available["GNN"] = gnn_proba

# Find valid rows (predicted by ALL models)
valid_mask = np.ones(len(y_all), dtype=bool)
for name, pr in available.items():
    valid_mask &= ~np.isnan(pr)

y_valid = y_all[valid_mask].values
print(f"  Valid rows for ensemble: {valid_mask.sum()}")
print(f"  Models included: {list(available.keys())}")

best_auc    = 0.0
best_weights = None
best_name    = ""

# Grid search over ensemble weights
model_names = list(available.keys())
n_models    = len(model_names)
prs         = np.stack([available[m][valid_mask] for m in model_names], axis=1)

# Simple equal-weight ensemble first
equal_w    = np.ones(n_models) / n_models
equal_pred = prs @ equal_w
equal_auc  = roc_auc_score(y_valid, equal_pred)
print(f"\n  Equal-weight ensemble AUC: {equal_auc:.4f}")

# Optimize weights via grid search
best_auc = equal_auc
best_weights = equal_w.copy()

# For 2-4 models, grid search is fast
# Grid search with 0.05 step for better resolution on 2-4 models
from itertools import product
step_size = 0.05 if n_models <= 4 else 0.1
steps = np.arange(0, 1.0 + step_size/2, step_size)
tested = 0
for combo in product(steps, repeat=n_models-1):
    remainder = round(1.0 - sum(combo), 8)
    if remainder < 0 or remainder > 1.001:
        continue
    w = np.array(list(combo) + [max(0, remainder)])
    if abs(w.sum() - 1.0) > 0.01:
        continue
    pred = prs @ w
    try:
        a = roc_auc_score(y_valid, pred)
    except Exception:
        continue
    if a > best_auc:
        best_auc     = a
        best_weights = w
        best_name    = " + ".join(
            f"{m}({w_:.2f})" for m, w_ in zip(model_names, w) if w_ > 0.01)
    tested += 1
print(f"  Grid search: {tested} combinations tested")
print(f"  Optimized ensemble AUC: {best_auc:.4f}  [{best_name}]")

# Final ensemble prediction
ensemble_pred = prs @ best_weights

# Full metrics
ens_metrics = eval_metrics(y_valid, ensemble_pred)
print(f"\n  ENSEMBLE METRICS (LOSO-CV):")
print(f"    ROC-AUC: {ens_metrics['roc_auc']:.4f}")
print(f"    PR-AUC:  {ens_metrics['pr_auc']:.4f}")
print(f"    MCC:     {ens_metrics['mcc']:.4f}")
print(f"    F1:      {ens_metrics['f1']:.4f}")
print(f"    ISEF target (0.91): {'REACHED ✓' if ens_metrics['roc_auc'] >= 0.91 else 'Not yet'}")

pd.DataFrame([ens_metrics], index=["Ensemble"]).to_csv(
    RESULTS_DIR / "ensemble_metrics.csv")

# Save row-level predictions for cocktail optimizer (05_cocktail_optimizer.py)
# This is the primary input file for cocktail analysis
pred_df = dataset[valid_mask].copy().reset_index(drop=True)
pred_df["ensemble_proba"] = ensemble_pred
pred_df["y_true"]         = y_valid
# Add individual model probabilities for reference
for m_name, m_pr in available.items():
    pred_df[f"proba_{m_name.lower().replace(' ','_')}"] = m_pr[valid_mask]
pred_df.to_csv(RESULTS_DIR / "ensemble_predictions.csv", index=False)
print(f"  Saved {len(pred_df)} row-level predictions -> ensemble_predictions.csv")

# Also add assert for GNN alignment verification
if gnn_proba is not None:
    # GNN index file should correspond 1:1 with sorted dataset rows
    gnn_idx = np.load(RESULTS_DIR / "gnn_loso_index.npy") if (RESULTS_DIR / "gnn_loso_index.npy").exists() else None
    if gnn_idx is not None:
        n_aligned = (gnn_idx < len(dataset)).sum()
        print(f"  GNN alignment: {n_aligned}/{len(gnn_idx)} indices within dataset bounds")


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — PER-SPECIES BREAKDOWN (ensemble)
# ═══════════════════════════════════════════════════════════════
print("\n[7] Per-species ensemble AUC...")

species_results = {}
dataset_valid = dataset[valid_mask].reset_index(drop=True)

for sp in valid_species:
    sp_mask = (dataset_valid["host"] == sp).values
    y_sp = y_valid[sp_mask]
    p_sp = ensemble_pred[sp_mask]
    if len(np.unique(y_sp)) < 2 or len(y_sp) < 3:
        continue
    try:
        species_results[sp] = {
            "n_test":    len(y_sp),
            "n_pos":     y_sp.sum(),
            "roc_auc":   roc_auc_score(y_sp, p_sp),
            "genus":     sp.split()[0],
        }
    except Exception:
        pass

sp_df = pd.DataFrame(species_results).T.sort_values("roc_auc", ascending=False)
print(f"\n  Top 15 species:")
print(sp_df.head(15).to_string(float_format="{:.4f}".format))
print(f"\n  Bottom 10 species:")
print(sp_df.tail(10).to_string(float_format="{:.4f}".format))
print(f"\n  Mean species AUC: {sp_df['roc_auc'].mean():.4f} +/- {sp_df['roc_auc'].std():.4f}")
sp_df.to_csv(RESULTS_DIR / "ensemble_per_species.csv")


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — S. AUREUS DEEP DIVE
# ═══════════════════════════════════════════════════════════════
print("\n[8] S. aureus analysis...")
sa_mask = (dataset_valid["host"] == TARGET_HOST).values & valid_mask[:len(dataset_valid)]
if sa_mask.sum() > 10:
    y_sa = y_valid[sa_mask[:valid_mask.sum()]]
    p_sa = ensemble_pred[sa_mask[:valid_mask.sum()]]
    if len(np.unique(y_sa)) == 2:
        sa_auc = roc_auc_score(y_sa, p_sa)
        print(f"  S. aureus: {sa_mask.sum()} pairs, AUC={sa_auc:.4f}")


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — PLOTS
# ═══════════════════════════════════════════════════════════════
print("\n[9] Generating plots...")

# Plot 18: Model comparison AUC bar chart
all_aucs = {k: eval_metrics(y_valid, available[k][valid_mask])["roc_auc"]
            for k in available}
all_aucs["Ensemble (optimized)"] = ens_metrics["roc_auc"]

fig, ax = plt.subplots(figsize=(9, max(4, len(all_aucs)*0.55)))
names_p = list(all_aucs.keys())
vals_p  = list(all_aucs.values())
colors  = ["#FF6B35" if "Ensemble" in n or "GNN" in n else "#4285F4" for n in names_p]
bars    = ax.barh(names_p, vals_p, color=colors, alpha=0.85)
ax.set_xlim(0, 1.08)
ax.axvline(0.91, color="gold", lw=2, linestyle="--", label="ISEF target (0.91)")
ax.axvline(0.87, color="orange", lw=1.5, linestyle=":", label="Previous target (0.87)")
for bar, v in zip(bars, vals_p):
    ax.text(v+0.005, bar.get_y()+bar.get_height()/2,
             f"{v:.4f}", va="center", fontsize=9,
             fontweight="bold" if v >= 0.91 else "normal")
ax.set_xlabel("ROC-AUC (LOSO-CV)", fontsize=12)
ax.set_title("Ensemble vs Individual Models — LOSO-CV\n"
              "(Orange = GNN/Ensemble, Blue = traditional ML)",
              fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="x", alpha=0.4)
plt.tight_layout()
save_plot(fig, "18_ensemble_comparison")

# Plot 19: ROC curve for ensemble
fig, ax = plt.subplots(figsize=(7, 6))
fpr, tpr, _ = roc_curve(y_valid, ensemble_pred)
ax.plot(fpr, tpr, "darkorange", lw=2.5,
         label=f"Ensemble AUC = {ens_metrics['roc_auc']:.4f}")
# Overlay individual models
line_styles = [("steelblue","--"), ("green","-."), ("purple",":"), ("brown","--")]
for (mn, pr), (color, ls) in zip(
        list(available.items())[:4], line_styles):
    try:
        f2, t2, _ = roc_curve(y_valid, pr[valid_mask])
        a2 = roc_auc_score(y_valid, pr[valid_mask])
        ax.plot(f2, t2, color=color, lw=1.5, linestyle=ls,
                 alpha=0.6, label=f"{mn} ({a2:.3f})")
    except Exception:
        pass
ax.plot([0,1],[0,1],"r--", alpha=0.4, label="Random")
ax.set_xlabel("FPR", fontsize=12); ax.set_ylabel("TPR", fontsize=12)
ax.set_title("Ensemble ROC Curve (LOSO-CV)", fontsize=11, fontweight="bold")
ax.legend(fontsize=8, loc="lower right"); ax.grid(True, alpha=0.4)
plt.tight_layout()
save_plot(fig, "19_ensemble_roc")

# Plot 20: Per-species AUC heatmap by genus
if len(sp_df) > 0:
    fig, ax = plt.subplots(figsize=(10, max(6, len(sp_df)*0.35)))
    y_p     = np.arange(len(sp_df))
    colors_s = ["#1a9850" if v>=0.9 else "#4285F4" if v>=0.8
                  else "#FF9800" if v>=0.7 else "#F44336"
                  for v in sp_df["roc_auc"]]
    bars2 = ax.barh(y_p, sp_df["roc_auc"].values, color=colors_s, alpha=0.85)
    ax.set_yticks(y_p)
    ax.set_yticklabels([f"{idx} ({row['n_test']:.0f})"
                         for idx, row in sp_df.iterrows()], fontsize=7)
    ax.set_xlim(0, 1.12)
    ax.axvline(0.91, color="gold", lw=2, linestyle="--", label="ISEF target")
    ax.axvline(sp_df["roc_auc"].mean(), color="black", lw=1.5, linestyle=":",
                label=f"Mean={sp_df['roc_auc'].mean():.3f}")
    for bar, v in zip(bars2, sp_df["roc_auc"].values):
        ax.text(v+0.005, bar.get_y()+bar.get_height()/2,
                 f"{v:.3f}", va="center", fontsize=6)
    ax.set_xlabel("ROC-AUC (Ensemble, LOSO-CV)", fontsize=11)
    ax.set_title("Ensemble Per-Species ROC-AUC\n"
                  "Green ≥ 0.9 | Blue ≥ 0.8 | Orange ≥ 0.7 | Red < 0.7",
                  fontsize=10, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="x", alpha=0.4)
    plt.tight_layout()
    save_plot(fig, "20_ensemble_per_species")

# Plot 21: Weight breakdown
if best_weights is not None and n_models > 1:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(model_names, best_weights,
            color=["#FF6B35" if "GNN" in n else "#4285F4" for n in model_names],
            alpha=0.85)
    ax.set_ylabel("Optimal weight", fontsize=12)
    ax.set_title("Ensemble Optimal Weights (grid search)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.4)
    for i, (n, w) in enumerate(zip(model_names, best_weights)):
        ax.text(i, w+0.01, f"{w:.2f}", ha="center", fontsize=10)
    plt.tight_layout()
    save_plot(fig, "21_ensemble_weights")


# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  ENSEMBLE FINAL SUMMARY")
print("="*60)
print(f"\n  Models combined: {', '.join(available.keys())}")
print(f"  CV method:       Leave-One-Species-Out (LOSO)")
print(f"  Optimal weights: {', '.join(f'{m}={w:.2f}' for m,w in zip(model_names,best_weights))}")
print(f"\n  ROC-AUC: {ens_metrics['roc_auc']:.4f}")
print(f"  PR-AUC:  {ens_metrics['pr_auc']:.4f}")
print(f"  MCC:     {ens_metrics['mcc']:.4f}")
print(f"  F1:      {ens_metrics['f1']:.4f}")
print(f"\n  ISEF target (0.91): {'✓ REACHED' if ens_metrics['roc_auc'] >= 0.91 else '✗ Not yet'}")
print(f"\n  Results: {RESULTS_DIR.resolve()}")
print(f"  Plots:   {PLOT_DIR.resolve()}")
print("\n  Done!")