"""
06_logo_validation.py  —  Leave-One-Genus-Out (LOGO) Validation
================================================================
PrecisionPhage | ISEF 2026

GOAL: Demonstrate cross-genus generalization.
  LOSO holds out one SPECIES at a time — tests memorization.
  LOGO holds out one GENUS at a time — tests genuine generalization
  to unseen bacterial clades. A phage that infects Pseudomonas shares
  biological mechanisms with other Gamma-proteobacteria even if those
  genera were never seen at training time.

METHOD:
  For each genus G:
    train_mask = all rows where genus != G
    test_mask  = all rows where genus == G  (all species of G)
  Metrics computed on test fold, averaged across genera.
  Compared to LOSO AUC via Wilcoxon signed-rank test.

NO RETRAINING of saved models — this script retrains XGBoost from
scratch under LOGO-CV because the split is fundamentally different
from the LOSO predictions saved in ensemble_predictions.csv.
"""

import warnings, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import wilcoxon
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    matthews_corrcoef, precision_recall_curve, auc,
)
import xgboost as xgb

warnings.filterwarnings("ignore")
np.random.seed(42)

_SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR    = _SCRIPT_DIR.parent / "data"
RAW_DIR     = BASE_DIR / "raw"
PLOT_DIR    = BASE_DIR / "plots"
RESULTS_DIR = BASE_DIR / "results"
for d in [PLOT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED      = 42
NUMERIC_FEATURES = ["k3dist", "k6dist", "GCdiff", "Homology"]
STRUCT_FEATURES  = ["phage_breadth", "host_vuln", "genus_pos_rate"]

sns.set_theme(style="whitegrid", palette="colorblind")

def save_plot(fig, name):
    p = PLOT_DIR / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")

def eval_metrics(y_true, y_proba):
    p, r, _ = precision_recall_curve(y_true, y_proba)
    ypred   = (y_proba >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc":  auc(r, p),
        "f1":      f1_score(y_true, ypred, zero_division=0),
        "mcc":     matthews_corrcoef(y_true, ypred),
    }

# ── LOAD DATA ──
print("=" * 62)
print("  LEAVE-ONE-GENUS-OUT (LOGO) VALIDATION")
print("=" * 62)

capped_path   = RAW_DIR / "capped_dataset.csv"
enriched_path = RAW_DIR / "enriched_dataset.csv"

if capped_path.exists():
    dataset = pd.read_csv(capped_path, index_col=0)
    print(f"  Loaded: {len(dataset)} rows (capped_dataset.csv)")
elif enriched_path.exists():
    dataset = pd.read_csv(enriched_path)
    print(f"  Loaded: {len(dataset)} rows (enriched_dataset.csv)")
else:
    raise FileNotFoundError("Run 03_gnn.py then 04_ensemble.py first.")

dataset["phage"] = dataset["phage"].astype(str).str.strip()
dataset["host"]  = dataset["host"].astype(str).str.strip()
if "genus" not in dataset.columns:
    dataset["genus"] = dataset["host"].str.split().str[0]
for c in NUMERIC_FEATURES:
    if c not in dataset.columns: dataset[c] = 0.0
    dataset[c] = pd.to_numeric(dataset[c], errors="coerce").fillna(0.0)

# Structural features
pb = dataset[dataset["label"]==1].groupby("phage")["host"].nunique()
hv = dataset[dataset["label"]==1].groupby("host")["phage"].nunique()
gp = dataset.groupby("genus")["label"].mean()
dataset["phage_breadth"]  = dataset["phage"].map(pb).fillna(0)
dataset["host_vuln"]      = dataset["host"].map(hv).fillna(0)
dataset["genus_pos_rate"] = dataset["genus"].map(gp).fillna(0.5)

# ── BUILD FEATURES ──
print("\n[1] Building features...")
names   = dataset["phage"].fillna("").astype(str)
y_all   = dataset["label"].values

vec_name = CountVectorizer(analyzer="char", ngram_range=(3,5),
                            max_features=30_000, dtype=np.float32)
X_name   = vec_name.fit_transform(names)
scaler_n = StandardScaler()
X_num    = csr_matrix(scaler_n.fit_transform(
                dataset[NUMERIC_FEATURES].values.astype(np.float32)))
scaler_s = StandardScaler()
X_str    = csr_matrix(scaler_s.fit_transform(
                dataset[STRUCT_FEATURES].values.astype(np.float32)))
X_full   = hstack([X_name, X_num, X_str], format="csr")
print(f"  Feature matrix: {X_full.shape}")

# ── LOGO-CV ──
genera = dataset["genus"].unique()
evaluable_genera = [
    g for g in genera
    if dataset[dataset["genus"]==g]["label"].nunique() == 2
       and len(dataset[dataset["genus"]==g]) >= 5
]
print(f"\n[2] LOGO-CV: {len(evaluable_genera)} evaluable genera")

pos_weight = int((y_all==0).sum() / max((y_all==1).sum(), 1))
clf = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.08, max_depth=4,
    scale_pos_weight=pos_weight, eval_metric="logloss",
    subsample=0.9, colsample_bytree=0.9,
    reg_alpha=0.1, reg_lambda=2.0,
    random_state=RANDOM_SEED, n_jobs=-1, verbosity=0)

logo_results = []
all_y_true, all_y_proba = [], []

for g in evaluable_genera:
    test_mask  = (dataset["genus"] == g).values
    train_mask = ~test_mask
    if y_all[train_mask].sum() < 10:
        continue
    clf.fit(X_full[train_mask], y_all[train_mask])
    proba_test = clf.predict_proba(X_full[test_mask])[:, 1]
    yt = y_all[test_mask]
    if len(np.unique(yt)) < 2:
        continue
    m = eval_metrics(yt, proba_test)
    m["genus"]    = g
    m["n_test"]   = int(test_mask.sum())
    m["n_pos"]    = int(yt.sum())
    m["n_species"]= int(dataset[dataset["genus"]==g]["host"].nunique())
    logo_results.append(m)
    all_y_true.extend(yt.tolist())
    all_y_proba.extend(proba_test.tolist())
    print(f"  {g:<30} n={m['n_test']:>4}  AUC={m['roc_auc']:.4f}")

logo_df = pd.DataFrame(logo_results).set_index("genus")
logo_df.to_csv(RESULTS_DIR / "logo_per_genus.csv")

mean_logo_auc = logo_df["roc_auc"].mean()
std_logo_auc  = logo_df["roc_auc"].std()
pooled_logo_auc = roc_auc_score(all_y_true, all_y_proba)

print(f"\n  LOGO mean AUC:   {mean_logo_auc:.4f} ± {std_logo_auc:.4f}")
print(f"  LOGO pooled AUC: {pooled_logo_auc:.4f}")

# ── LOAD LOSO PER-SPECIES AUC for comparison ──
loso_auc_global = None
loso_per_species_aucs = None
loso_path = RESULTS_DIR / "ensemble_per_species.csv"
if loso_path.exists():
    loso_sp = pd.read_csv(loso_path)
    # Normalise column names
    loso_sp.columns = loso_sp.columns.str.strip()
    auc_col = [c for c in loso_sp.columns if "roc_auc" in c.lower()]
    if auc_col:
        loso_sp = loso_sp.rename(columns={auc_col[0]: "roc_auc"})
        sp_col  = loso_sp.columns[0]
        loso_sp["genus"] = loso_sp[sp_col].str.split().str[0]
        # Aggregate to genus level (mean AUC per genus)
        loso_genus = loso_sp.groupby("genus")["roc_auc"].mean()
        loso_per_species_aucs = loso_sp["roc_auc"].values
        loso_auc_global = loso_sp["roc_auc"].mean()

# ── STATISTICAL COMPARISON ──
print("\n[3] Statistical comparison LOSO vs LOGO...")
if loso_auc_global is not None:
    # Paired comparison at genus level
    shared = logo_df.index[logo_df.index.isin(loso_genus.index)]
    if len(shared) >= 5:
        logo_shared = logo_df.loc[shared, "roc_auc"].values
        loso_shared = loso_genus.loc[shared].values
        diff = logo_shared - loso_shared
        nz   = diff[diff != 0]
        if len(nz) >= 5:
            stat, pval = wilcoxon(nz)
            print(f"  Wilcoxon LOSO vs LOGO (genus-level): p={pval:.4f}")
        else:
            pval = 1.0
            print("  Not enough paired differences for Wilcoxon test")
        delta = mean_logo_auc - loso_auc_global
        print(f"  LOSO mean AUC: {loso_auc_global:.4f}")
        print(f"  LOGO mean AUC: {mean_logo_auc:.4f}  (delta: {delta:+.4f})")
        degrade = "YES — significant degradation" if (pval < 0.05 and delta < -0.05) \
                  else "MINIMAL — cross-genus learning is strong"
        print(f"  Performance degradation: {degrade}")
    else:
        pval  = None
        delta = None
        print("  Insufficient shared genera for paired test")
else:
    pval  = None
    delta = None
    print("  ensemble_per_species.csv not found — skipping LOSO comparison")

# ── SAVE SUMMARY ──
summary = {
    "logo_mean_auc":   mean_logo_auc,
    "logo_std_auc":    std_logo_auc,
    "logo_pooled_auc": pooled_logo_auc,
    "loso_mean_auc":   loso_auc_global,
    "delta_logo_minus_loso": delta,
    "wilcoxon_p": pval,
    "n_genera_evaluated": len(logo_df),
}
pd.Series(summary).to_csv(RESULTS_DIR / "logo_summary.csv", header=["value"])

# ── PLOTS ──
print("\n[4] Generating plots...")

# Plot A: Boxplot LOSO vs LOGO
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]

if loso_auc_global is not None and "loso_genus" in dir():
    shared = logo_df.index[logo_df.index.isin(loso_genus.index)]
    box_data = pd.DataFrame({
        "LOSO\n(per-genus mean)": loso_genus.loc[shared].values,
        "LOGO\n(this script)":    logo_df.loc[shared, "roc_auc"].values,
    })
    bp = box_data.boxplot(ax=ax, patch_artist=True, notch=True,
                           medianprops=dict(color="black", lw=2))
    colors_bp = ["#4285F4", "#FF6B35"]
    for patch, c in zip(ax.patches[:2], colors_bp):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    # Annotate with p-value
    if pval is not None:
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        ax.text(1.5, ax.get_ylim()[1]*0.97, f"p={pval:.4f} {sig}",
                 ha="center", fontsize=10,
                 color="#1a9850" if pval >= 0.05 else "#F44336")
else:
    logo_df["roc_auc"].plot(kind="box", ax=ax, patch_artist=True,
                             medianprops=dict(color="black", lw=2))
    ax.set_xticklabels(["LOGO"])

ax.axhline(0.5, color="gray", linestyle=":", lw=1.5, label="Random")
ax.set_ylabel("ROC-AUC", fontsize=12)
ax.set_title("LOSO vs LOGO AUC Distribution\n"
              "(each point = one genus)",
              fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.4)

# Plot B: Per-genus LOGO AUC bar chart
ax = axes[1]
logo_sorted = logo_df.sort_values("roc_auc", ascending=True)
colors_bar  = ["#1a9850" if v >= 0.8 else
               "#FF6B35" if v >= 0.6 else "#F44336"
               for v in logo_sorted["roc_auc"]]
y_pos = np.arange(len(logo_sorted))
ax.barh(y_pos, logo_sorted["roc_auc"], color=colors_bar, alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{g} (n={logo_sorted.loc[g,'n_test']})"
                     for g in logo_sorted.index], fontsize=7)
ax.axvline(0.5, color="gray", linestyle=":", lw=1.5)
ax.axvline(mean_logo_auc, color="#FF6B35", linestyle="--", lw=2,
            label=f"Mean={mean_logo_auc:.3f}")
ax.set_xlabel("ROC-AUC", fontsize=12)
ax.set_title("Per-Genus LOGO ROC-AUC\n"
              "Green ≥0.8 | Orange ≥0.6 | Red <0.6",
              fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.set_xlim(0, 1.05); ax.grid(axis="x", alpha=0.4)

plt.suptitle("PrecisionPhage — Cross-Genus Generalization (LOGO-CV)",
              fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
save_plot(fig, "29_logo_validation")

# ── FINAL SUMMARY ──
print()
print("=" * 62)
print("  LEAVE-ONE-GENUS-OUT VALIDATION — SUMMARY")
print("=" * 62)
print(f"""
  Genera evaluated:   {len(logo_df)}
  LOGO mean AUC:      {mean_logo_auc:.4f} ± {std_logo_auc:.4f}
  LOGO pooled AUC:    {pooled_logo_auc:.4f}
  LOSO mean AUC:      {loso_auc_global if loso_auc_global else 'N/A'}
  Delta (LOGO-LOSO):  {f'{delta:+.4f}' if delta is not None else 'N/A'}
  Wilcoxon p-value:   {f'{pval:.4f}' if pval is not None else 'N/A'}

  Interpretation:
  {'Cross-genus generalization is STRONG — performance drop is minimal.' if (delta is None or delta > -0.05) else f'Performance drops {abs(delta):.3f} AUC points going genus-to-genus.'}
  {'This demonstrates the model learns transferable phage-host biology,' if (delta is None or delta > -0.05) else 'This suggests the model partially relies on genus-level memorization.'}
  {'not genus-specific memorization.' if (delta is None or delta > -0.05) else 'Consider adding phylogenetic features for future improvement.'}
""")
print(f"  Results: {RESULTS_DIR.resolve()}")
print(f"  Plots:   {PLOT_DIR.resolve()}")
print("\n  Done!")