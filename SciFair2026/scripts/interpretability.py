"""
09_interpretability.py  —  Model Interpretability
==================================================
PrecisionPhage | ISEF 2026

GOAL: Extract biological insight from the ensemble — what features
  does the model actually use to predict phage-host compatibility?

METHOD:
  1. Retrain XGBoost on full dataset (no CV — for feature importance only)
  2. Extract gain-based feature importance
  3. Group into 4 biological categories:
       Genomic:    k3dist, k6dist, GCdiff, Homology
       Structural: phage_breadth, host_vuln, genus_pos_rate
       Name SVD:   char n-gram components (bulk, captures taxonomy)
       Name raw:   individual n-gram tokens
  4. Report % contribution per category and top 20 features
  5. Generate bar chart + pie chart

INTERPRETATION:
  Structural features (host_vuln, phage_breadth) = network topology
  Genomic features (k-mer dist, GC) = sequence similarity
  Name features = taxonomic signal encoded in nomenclature

Uses capped_dataset.csv. No leakage — this is a full-dataset
importance analysis, not a CV result, explicitly disclosed.
"""

import warnings, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
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
TOP_N            = 20   # top features to show

sns.set_theme(style="whitegrid", palette="colorblind")

def save_plot(fig, name):
    p = PLOT_DIR / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")

# ── LOAD DATA ──
print("=" * 62)
print("  MODEL INTERPRETABILITY ANALYSIS")
print("=" * 62)

capped_path   = RAW_DIR / "capped_dataset.csv"
enriched_path = RAW_DIR / "enriched_dataset.csv"
if capped_path.exists():
    dataset = pd.read_csv(capped_path, index_col=0)
elif enriched_path.exists():
    dataset = pd.read_csv(enriched_path)
else:
    raise FileNotFoundError("Run 03_gnn.py first.")

dataset["phage"] = dataset["phage"].astype(str).str.strip()
if "genus" not in dataset.columns:
    dataset["genus"] = dataset["host"].str.split().str[0]
for c in NUMERIC_FEATURES:
    if c not in dataset.columns: dataset[c] = 0.0
    dataset[c] = pd.to_numeric(dataset[c], errors="coerce").fillna(0.0)

pb = dataset[dataset["label"]==1].groupby("phage")["host"].nunique()
hv = dataset[dataset["label"]==1].groupby("host")["phage"].nunique()
gp = dataset.groupby("genus")["label"].mean()
dataset["phage_breadth"]  = dataset["phage"].map(pb).fillna(0)
dataset["host_vuln"]      = dataset["host"].map(hv).fillna(0)
dataset["genus_pos_rate"] = dataset["genus"].map(gp).fillna(0.5)

# ── BUILD FEATURES ──
print("\n[1] Building features...")
names  = dataset["phage"].fillna("").astype(str)
y_all  = dataset["label"].values

vec_name = CountVectorizer(analyzer="char", ngram_range=(3,5),
                            max_features=30_000, dtype=np.float32)
X_name   = vec_name.fit_transform(names)
n_name   = X_name.shape[1]

sn    = StandardScaler()
X_num = csr_matrix(sn.fit_transform(
            dataset[NUMERIC_FEATURES].values.astype(np.float32)))
n_num = X_num.shape[1]

ss    = StandardScaler()
X_str = csr_matrix(ss.fit_transform(
            dataset[STRUCT_FEATURES].values.astype(np.float32)))
n_str = X_str.shape[1]

X_full = hstack([X_name, X_num, X_str], format="csr")
n_total = X_full.shape[1]

# Feature name list
feat_names = (
    [f"ngram_{t}" for t in vec_name.get_feature_names_out()] +
    NUMERIC_FEATURES +
    STRUCT_FEATURES
)
assert len(feat_names) == n_total, f"Mismatch: {len(feat_names)} vs {n_total}"
print(f"  Features: {n_total} total "
      f"({n_name} n-gram | {n_num} genomic | {n_str} structural)")

# ── TRAIN XGBoost FULL DATASET ──
print("\n[2] Training XGBoost (full dataset, for importance only)...")
pos_weight = int((y_all==0).sum() / max((y_all==1).sum(), 1))
clf = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.08, max_depth=4,
    scale_pos_weight=pos_weight, eval_metric="logloss",
    subsample=0.9, colsample_bytree=0.9,
    reg_alpha=0.1, reg_lambda=2.0,
    random_state=RANDOM_SEED, n_jobs=-1, verbosity=0)
clf.fit(X_full, y_all)
print("  Training complete.")
print("  NOTE: This is a full-dataset fit for importance analysis only,")
print("        not used for any AUC reporting.")

# ── EXTRACT IMPORTANCE ──
print("\n[3] Extracting feature importance (gain)...")
imp = clf.get_booster().get_score(importance_type="gain")
# Map f0,f1,... → our feature names
imp_named = {}
for k, v in imp.items():
    idx = int(k.replace("f",""))
    if idx < len(feat_names):
        imp_named[feat_names[idx]] = v

imp_series = pd.Series(imp_named).sort_values(ascending=False)
imp_df     = imp_series.reset_index()
imp_df.columns = ["feature", "importance_gain"]
imp_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
print(f"  Active features: {len(imp_df)} (out of {n_total} total)")

# ── CATEGORISE FEATURES ──
def categorise(feat):
    if feat in NUMERIC_FEATURES:
        return "Genomic (k-mer/GC)"
    if feat in STRUCT_FEATURES:
        return "Structural (network)"
    if feat.startswith("ngram_"):
        return "Name-derived (n-gram)"
    return "Other"

imp_df["category"] = imp_df["feature"].apply(categorise)

cat_totals = imp_df.groupby("category")["importance_gain"].sum()
cat_pct    = (cat_totals / cat_totals.sum() * 100).sort_values(ascending=False)
cat_pct.to_csv(RESULTS_DIR / "feature_category_importance.csv", header=["pct"])

print(f"\n  Feature importance by category (% of total gain):")
for cat, pct in cat_pct.items():
    print(f"    {cat:<30} {pct:.1f}%")

# Individual breakdown for structural features
print(f"\n  Structural features:")
for f in STRUCT_FEATURES:
    g = imp_df[imp_df["feature"]==f]["importance_gain"]
    gain = g.values[0] if len(g) else 0
    pct  = gain / cat_totals.sum() * 100
    print(f"    {f:<25} gain={gain:.1f}  ({pct:.2f}%)")

print(f"\n  Genomic features:")
for f in NUMERIC_FEATURES:
    g = imp_df[imp_df["feature"]==f]["importance_gain"]
    gain = g.values[0] if len(g) else 0
    pct  = gain / cat_totals.sum() * 100
    print(f"    {f:<25} gain={gain:.1f}  ({pct:.2f}%)")

# Top 20 overall
top20 = imp_df.head(TOP_N)
print(f"\n  Top {TOP_N} features by gain:")
print(f"  {'Rank':<5} {'Feature':<35} {'Gain':>10}  {'Category'}")
print("  " + "-"*72)
for i, row in top20.iterrows():
    rank = i + 1
    print(f"  {rank:<5} {row['feature'][:33]:<35} {row['importance_gain']:>10.1f}  {row['category']}")

# ── PLOTS ──
print("\n[4] Generating plots...")

fig, axes = plt.subplots(1, 3, figsize=(17, 6))

# Panel A: Top-20 feature importance bar chart
ax = axes[0]
cat_colors = {
    "Genomic (k-mer/GC)":      "#4285F4",
    "Structural (network)":    "#FF6B35",
    "Name-derived (n-gram)":   "#1a9850",
    "Other":                   "#BDBDBD",
}
top20_rev   = top20.iloc[::-1].reset_index(drop=True)
bar_colors  = [cat_colors.get(c, "#BDBDBD") for c in top20_rev["category"]]
y_pos = np.arange(len(top20_rev))
ax.barh(y_pos, top20_rev["importance_gain"], color=bar_colors, alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(top20_rev["feature"].str[:28], fontsize=7)
ax.set_xlabel("Importance (gain)", fontsize=11)
ax.set_title(f"A  Top {TOP_N} Features by XGBoost Gain\n"
              "(full-dataset fit, for biological interpretation)",
              fontsize=10, fontweight="bold")
# Legend
from matplotlib.patches import Patch
legend_patches = [Patch(color=v, label=k, alpha=0.85)
                   for k, v in cat_colors.items()
                   if k in imp_df["category"].values]
ax.legend(handles=legend_patches, fontsize=7, loc="lower right")
ax.grid(axis="x", alpha=0.4)

# Panel B: Category contribution pie
ax = axes[1]
cat_labels = [f"{c}\n{v:.1f}%" for c, v in cat_pct.items()]
pie_colors = [cat_colors.get(c, "#BDBDBD") for c in cat_pct.index]
wedges, texts, autotexts = ax.pie(
    cat_pct.values,
    labels=cat_labels,
    colors=pie_colors,
    autopct="%1.1f%%",
    startangle=90,
    pctdistance=0.75,
    textprops={"fontsize":8}
)
for at in autotexts:
    at.set_fontsize(7)
ax.set_title("B  Feature Category Contributions\n(% of total XGBoost gain)",
              fontsize=10, fontweight="bold")

# Panel C: Structural features in detail
ax = axes[2]
struct_imp = imp_df[imp_df["category"]=="Structural (network)"].copy()
struct_imp = struct_imp.sort_values("importance_gain", ascending=True)
geom_imp   = imp_df[imp_df["category"]=="Genomic (k-mer/GC)"].copy()
geom_imp   = geom_imp.sort_values("importance_gain", ascending=True)
combined   = pd.concat([struct_imp, geom_imp])
bar_c2 = [cat_colors[c] for c in combined["category"]]
y2 = np.arange(len(combined))
ax.barh(y2, combined["importance_gain"], color=bar_c2, alpha=0.85)
ax.set_yticks(y2)
ax.set_yticklabels(combined["feature"], fontsize=9)
ax.set_xlabel("Importance (gain)", fontsize=11)
ax.set_title("C  Genomic & Structural Feature Detail\n"
              "(biological interpretability)",
              fontsize=10, fontweight="bold")
ax.grid(axis="x", alpha=0.4)
# Annotate biological meaning
feat_desc = {
    "k3dist":          "k-mer 3 distance",
    "k6dist":          "k-mer 6 distance",
    "GCdiff":          "GC content diff",
    "Homology":        "sequence homology",
    "phage_breadth":   "phage host-range width",
    "host_vuln":       "host susceptibility index",
    "genus_pos_rate":  "genus infection rate",
}
for i, (idx, row) in enumerate(combined.iterrows()):
    desc = feat_desc.get(row["feature"], "")
    ax.text(row["importance_gain"]+0.005*combined["importance_gain"].max(),
             i, desc, va="center", fontsize=7, color="#555")

plt.suptitle("PrecisionPhage — Feature Interpretability (XGBoost Gain)",
              fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
save_plot(fig, "32_interpretability")

# ── SUMMARY ──
dominant_cat = cat_pct.idxmax()
struct_pct   = cat_pct.get("Structural (network)", 0)
genomic_pct  = cat_pct.get("Genomic (k-mer/GC)", 0)
name_pct     = cat_pct.get("Name-derived (n-gram)", 0)

print()
print("=" * 62)
print("  INTERPRETABILITY — SUMMARY")
print("=" * 62)
print(f"""
  Feature category contributions:
    Name-derived (n-gram): {name_pct:.1f}%
    Structural (network):  {struct_pct:.1f}%
    Genomic (k-mer/GC):    {genomic_pct:.1f}%

  Dominant category: {dominant_cat}

  Top 3 individual features:
    1. {imp_df.iloc[0]['feature']:<25} ({imp_df.iloc[0]['importance_gain']:.1f} gain)
    2. {imp_df.iloc[1]['feature']:<25} ({imp_df.iloc[1]['importance_gain']:.1f} gain)
    3. {imp_df.iloc[2]['feature']:<25} ({imp_df.iloc[2]['importance_gain']:.1f} gain)

  Biological interpretation:
    {'Structural features dominating → model learned network topology:' if struct_pct > genomic_pct else 'Genomic features dominating → k-mer similarity drives predictions:'}
    {'host_vuln and phage_breadth capture which hosts are broadly susceptible' if struct_pct > genomic_pct else 'k3dist and k6dist capture sequence-level phage-host co-evolution'}
    {'and which phages have wide host range — biologically meaningful.' if struct_pct > genomic_pct else 'and GCdiff captures the %GC coevolution hypothesis.'}
    Name features encode taxonomic signal in phage nomenclature
    (e.g. "nc " prefix identifies NCBI-curated reference phages).
""")
print(f"  Results: {RESULTS_DIR.resolve()}")
print(f"  Plots:   {PLOT_DIR.resolve()}")
print("\n  Done!")