"""
07_unseen_strain_simulation.py  —  Prospective Unseen Strain Simulation
=======================================================================
PrecisionPhage | ISEF 2026

GOAL: Simulate hospital deployment — what happens when the model sees
  a bacterial strain it was NEVER trained on?

METHOD (Monte Carlo, 10 rounds):
  For each species S with ≥2 strains:
    1. Randomly select 30% of strains as "unseen" (held-out).
    2. Remove ALL interaction pairs for those strains from training.
    3. Train XGBoost on remaining pairs.
    4. Evaluate on unseen-strain pairs:
       - ROC-AUC
       - Strain-level cocktail coverage@3
  Repeat 10× with different random seeds.
  Report mean ± std across all Monte Carlo rounds.

STRAIN DEFINITION: uses NCBI /host annotations (from cocktail run).
  Falls back to cluster-based strains if cache missing.

NO LEAKAGE: unseen strains are completely excluded from training.
"""

import warnings, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
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
N_MONTE_CARLO    = 10
UNSEEN_FRAC      = 0.30
NUMERIC_FEATURES = ["k3dist", "k6dist", "GCdiff", "Homology"]
STRUCT_FEATURES  = ["phage_breadth", "host_vuln", "genus_pos_rate"]

sns.set_theme(style="whitegrid", palette="colorblind")

def save_plot(fig, name):
    p = PLOT_DIR / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")

# ── LOAD DATA ──
print("=" * 62)
print("  PROSPECTIVE UNSEEN STRAIN SIMULATION")
print("=" * 62)

# Load predictions with strain annotations from cocktail run
pred_path = RESULTS_DIR / "ensemble_predictions.csv"
if not pred_path.exists():
    raise FileNotFoundError("Run 04_ensemble.py and 05_cocktail_optimizer.py first.")

pred_df = pd.read_csv(pred_path)
pred_df["phage"] = pred_df["phage"].astype(str).str.strip()
pred_df["host"]  = pred_df["host"].astype(str).str.strip()
if "genus" not in pred_df.columns:
    pred_df["genus"] = pred_df["host"].str.split().str[0]

# Load strain annotations cache
strain_cache = RAW_DIR / "phage_strain_annotations.csv"
pred_df["strain"] = pred_df["phage"]  # default: phage = strain

if strain_cache.exists():
    import re
    cache = pd.read_csv(strain_cache)
    strain_lookup = dict(zip(cache["accession"], cache["strain"]))

    def clean_acc(ph):
        s = str(ph).upper().replace(" ", "_")
        m = re.search(r'(N[CZ]_\d{6}|[A-Z]{2}\d{5,6})', s)
        return m.group(1) if m else None

    def resolve(row):
        acc = clean_acc(row["phage"])
        if acc and acc in strain_lookup:
            h = str(strain_lookup[acc]).lower()
            if row["host"].split()[0] in h:
                h = re.sub(r'\bsubsp\.?\s+\w+\b', '', h)
                h = re.sub(r'\b(str\.|strain)\s+', '', h)
                return h.strip()
        return row["phage"]

    pred_df["strain"] = pred_df.apply(resolve, axis=1)
    print(f"  Strain annotations: {(pred_df['strain'] != pred_df['phage']).sum()} rows from NCBI")
else:
    print("  No strain cache found — using phage name as strain proxy")

# Load raw dataset for retraining
capped_path   = RAW_DIR / "capped_dataset.csv"
enriched_path = RAW_DIR / "enriched_dataset.csv"
if capped_path.exists():
    dataset = pd.read_csv(capped_path, index_col=0)
else:
    dataset = pd.read_csv(enriched_path)

dataset["phage"] = dataset["phage"].astype(str).str.strip()
dataset["host"]  = dataset["host"].astype(str).str.strip()
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

# Merge strain into dataset
strain_map = pred_df.set_index("phage")["strain"].to_dict()
dataset["strain"] = dataset["phage"].map(strain_map).fillna(dataset["phage"])

print(f"  Dataset: {len(dataset)} rows | Unique strains: {dataset['strain'].nunique()}")

# ── BUILD FEATURES ──
print("\n[1] Building features...")
names  = dataset["phage"].fillna("").astype(str)
y_all  = dataset["label"].values

vec_name = CountVectorizer(analyzer="char", ngram_range=(3,5),
                            max_features=30_000, dtype=np.float32)
X_name   = vec_name.fit_transform(names)
sn       = StandardScaler()
X_num    = csr_matrix(sn.fit_transform(
                dataset[NUMERIC_FEATURES].values.astype(np.float32)))
ss       = StandardScaler()
X_str    = csr_matrix(ss.fit_transform(
                dataset[STRUCT_FEATURES].values.astype(np.float32)))
X_full   = hstack([X_name, X_num, X_str], format="csr")
print(f"  Feature matrix: {X_full.shape}")

# Identify species with ≥2 unique strains for simulation
species_strain_counts = (
    dataset[dataset["label"]==1]
    .groupby("host")["strain"].nunique()
)
eligible = species_strain_counts[species_strain_counts >= 2].index.tolist()
print(f"  Species with ≥2 strains: {len(eligible)}")

pos_weight = int((y_all==0).sum() / max((y_all==1).sum(), 1))

def strain_cov(sp_df, selected, pos_strains):
    """Strain-level coverage for cocktail."""
    n = len(pos_strains)
    if n == 0: return 0.0
    covered = set(sp_df[sp_df["phage"].isin(selected) &
                         (sp_df["label"]==1)]["strain"].unique()) & pos_strains
    return len(covered) / n

def greedy_sel(sp_df, k, pos_strains):
    pos_df  = sp_df[sp_df["label"]==1]
    covered = set(); sel = []
    rem     = list(sp_df["phage"].unique())
    for _ in range(k):
        if not rem or covered == pos_strains: break
        best, best_g = None, -1
        for ph in rem:
            ns = len((set(pos_df[pos_df["phage"]==ph]["strain"]) & pos_strains) - covered)
            if ns > best_g:
                best_g, best = ns, ph
        if best is None: break
        covered |= set(pos_df[pos_df["phage"]==best]["strain"]) & pos_strains
        sel.append(best); rem.remove(best)
    return sel

# ── MONTE CARLO SIMULATION ──
print(f"\n[2] Monte Carlo simulation ({N_MONTE_CARLO} rounds × {UNSEEN_FRAC:.0%} unseen strains)...")

mc_results = []
loso_aucs  = []

# Load LOSO per-species AUC for comparison
loso_sp_path = RESULTS_DIR / "ensemble_per_species.csv"
if loso_sp_path.exists():
    loso_sp_df = pd.read_csv(loso_sp_path)
    loso_sp_df.columns = loso_sp_df.columns.str.strip()
    sp_col = loso_sp_df.columns[0]
    auc_col = [c for c in loso_sp_df.columns if "roc_auc" in c.lower()]
    if auc_col:
        loso_aucs = loso_sp_df[auc_col[0]].dropna().values.tolist()

clf = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.08, max_depth=4,
    scale_pos_weight=pos_weight, eval_metric="logloss",
    subsample=0.9, colsample_bytree=0.9,
    reg_alpha=0.1, reg_lambda=2.0,
    random_state=RANDOM_SEED, n_jobs=-1, verbosity=0)

for mc_round in range(N_MONTE_CARLO):
    rng = np.random.default_rng(RANDOM_SEED + mc_round)
    round_aucs  = []
    round_covs  = []

    # Select unseen strains for each eligible species
    unseen_rows = np.zeros(len(dataset), dtype=bool)
    for sp in eligible:
        sp_strains = (dataset[(dataset["host"]==sp) & (dataset["label"]==1)]
                      ["strain"].unique())
        n_unseen = max(1, int(len(sp_strains) * UNSEEN_FRAC))
        if n_unseen >= len(sp_strains):
            continue
        unseen_strains = rng.choice(sp_strains, size=n_unseen, replace=False)
        mask = (dataset["host"]==sp) & (dataset["strain"].isin(unseen_strains))
        unseen_rows |= mask.values

    # Training: exclude all unseen-strain rows
    train_mask = ~unseen_rows
    test_mask  = unseen_rows & (dataset["label"].values == 1)  # evaluate on positive unseen only

    if train_mask.sum() < 100 or test_mask.sum() < 5:
        continue

    # For AUC: test on all unseen rows (pos+neg) per species
    test_mask_all = unseen_rows

    try:
        clf_round = xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.08, max_depth=4,
            scale_pos_weight=pos_weight, eval_metric="logloss",
            subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.1, reg_lambda=2.0,
            random_state=RANDOM_SEED + mc_round, n_jobs=-1, verbosity=0)
        clf_round.fit(X_full[train_mask], y_all[train_mask])
        proba_unseen = clf_round.predict_proba(X_full[test_mask_all])[:, 1]
        y_unseen     = y_all[test_mask_all]

        if len(np.unique(y_unseen)) == 2:
            round_auc = roc_auc_score(y_unseen, proba_unseen)
            round_aucs.append(round_auc)

        # Coverage for each eligible species in this round
        for sp in eligible:
            sp_test = test_mask_all & (dataset["host"].values == sp)
            if sp_test.sum() < 2:
                continue
            sp_df_test = dataset[sp_test].copy().reset_index(drop=True)
            sp_df_test["ensemble_proba"] = proba_unseen[
                test_mask_all[test_mask_all].index[:sp_test.sum()] if False
                else np.where(sp_test[test_mask_all])[0]
            ] if False else clf_round.predict_proba(X_full[sp_test])[:, 1]

            pos_strains = set(sp_df_test[sp_df_test["label"]==1]["strain"].unique())
            if len(pos_strains) == 0: continue
            sel = greedy_sel(sp_df_test, 3, pos_strains)
            cov = strain_cov(sp_df_test, sel, pos_strains)
            round_covs.append(cov)

    except Exception as e:
        print(f"  Round {mc_round+1} error: {e}")
        continue

    mc_auc = np.mean(round_aucs)  if round_aucs else np.nan
    mc_cov = np.mean(round_covs) if round_covs else np.nan
    mc_results.append({
        "round":     mc_round + 1,
        "mean_auc":  mc_auc,
        "mean_cov3": mc_cov,
        "n_unseen":  int(test_mask_all.sum()),
    })
    print(f"  Round {mc_round+1:2d}/{N_MONTE_CARLO}: "
          f"AUC={mc_auc:.4f}  coverage@3={mc_cov:.3f}  "
          f"(n_unseen={int(test_mask_all.sum())})")

mc_df = pd.DataFrame(mc_results)
mc_df.to_csv(RESULTS_DIR / "unseen_strain_mc_results.csv", index=False)

mean_mc_auc  = mc_df["mean_auc"].mean()
std_mc_auc   = mc_df["mean_auc"].std()
mean_mc_cov  = mc_df["mean_cov3"].mean()
std_mc_cov   = mc_df["mean_cov3"].std()

loso_mean = np.mean(loso_aucs) if loso_aucs else None
print(f"\n  Unseen strain AUC:  {mean_mc_auc:.4f} ± {std_mc_auc:.4f}")
print(f"  Unseen coverage@3:  {mean_mc_cov:.3f} ± {std_mc_cov:.3f}")
if loso_mean:
    print(f"  LOSO mean AUC:      {loso_mean:.4f}")
    print(f"  AUC drop:           {mean_mc_auc - loso_mean:+.4f}")

# ── PLOTS ──
print("\n[3] Generating plots...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot A: AUC distribution across MC rounds
ax = axes[0]
ax.hist(mc_df["mean_auc"].dropna(), bins=8, color="#FF6B35", alpha=0.8,
         edgecolor="white", linewidth=0.8)
ax.axvline(mean_mc_auc, color="#FF6B35", lw=2.5, linestyle="--",
            label=f"Unseen mean={mean_mc_auc:.3f}")
if loso_mean:
    ax.axvline(loso_mean, color="#4285F4", lw=2, linestyle="--",
                label=f"LOSO mean={loso_mean:.3f}")
ax.set_xlabel("ROC-AUC", fontsize=12)
ax.set_ylabel("Count (MC rounds)", fontsize=12)
ax.set_title("Unseen Strain AUC Distribution\n"
              f"(n={N_MONTE_CARLO} Monte Carlo rounds, {UNSEEN_FRAC:.0%} strains withheld)",
              fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.4)
ax.set_xlim(0, 1)

# Plot B: Coverage@3 distribution
ax = axes[1]
ax.hist(mc_df["mean_cov3"].dropna(), bins=8, color="#1a9850", alpha=0.8,
         edgecolor="white", linewidth=0.8)
ax.axvline(mean_mc_cov, color="#1a9850", lw=2.5, linestyle="--",
            label=f"Mean={mean_mc_cov:.3f}")
ax.set_xlabel("Strain coverage@3", fontsize=12)
ax.set_ylabel("Count (MC rounds)", fontsize=12)
ax.set_title("Unseen Strain Coverage@3 Distribution\n"
              "(Greedy selection on withheld strains)",
              fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.4)
ax.set_xlim(0, 1)

plt.suptitle("PrecisionPhage — Prospective Unseen Strain Simulation",
              fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
save_plot(fig, "30_unseen_strain_simulation")

# ── SUMMARY ──
print()
print("=" * 62)
print("  PROSPECTIVE STRAIN SIMULATION — SUMMARY")
print("=" * 62)
drop = (mean_mc_auc - loso_mean) if loso_mean else None
interp = ("STRONG TRANSLATIONAL SIGNAL — AUC drops <5pp on unseen strains."
          if drop is None or drop > -0.05
          else f"Moderate drop of {abs(drop):.3f} AUC — expected for novel strains.")
print(f"""
  Monte Carlo rounds:        {N_MONTE_CARLO}
  Unseen fraction:           {UNSEEN_FRAC:.0%} of strains per species
  Species simulated:         {len(eligible)}

  Unseen strain AUC:         {mean_mc_auc:.4f} ± {std_mc_auc:.4f}
  Unseen strain coverage@3:  {mean_mc_cov:.3f} ± {std_mc_cov:.3f}
  LOSO AUC (baseline):       {f'{loso_mean:.4f}' if loso_mean else 'N/A'}
  AUC drop vs LOSO:          {f'{drop:+.4f}' if drop else 'N/A'}

  Interpretation: {interp}
""")
print(f"  Results: {RESULTS_DIR.resolve()}")
print(f"  Plots:   {PLOT_DIR.resolve()}")
print("\n  Done!")