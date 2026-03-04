"""
08_clinical_comparison.py  —  Clinical Strategy Comparison
==========================================================
PrecisionPhage | ISEF 2026

GOAL: Compare 4 cocktail selection strategies against each other with
  full statistical testing — framed as the decision a clinician makes.

STRATEGIES (all at k=3):
  Single:  take the single highest-predicted phage, use it alone
  Top-K:   take top-3 by predicted probability
  Random:  randomly sample 3 phages (100-trial average)
  Greedy:  our set-cover greedy maximizing strain coverage

METRICS:
  Mean strain coverage     — what fraction of host strains are treated
  Resistance robustness    — coverage after losing best phage
  % species ≥75% coverage  — practical clinical threshold

STATISTICAL TESTS: paired Wilcoxon for each strategy vs Greedy.

Uses ensemble_predictions.csv + phage_strain_annotations.csv.
No retraining.
"""

import warnings, re, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
np.random.seed(42)

_SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR    = _SCRIPT_DIR.parent / "data"
RAW_DIR     = BASE_DIR / "raw"
PLOT_DIR    = BASE_DIR / "plots"
RESULTS_DIR = BASE_DIR / "results"
for d in [PLOT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="colorblind")
N_RANDOM   = 200
K_COCKTAIL = 3

def save_plot(fig, name):
    p = PLOT_DIR / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")

# ── LOAD & ANNOTATE ──
print("=" * 62)
print("  CLINICAL STRATEGY COMPARISON")
print("=" * 62)

pred_path = RESULTS_DIR / "ensemble_predictions.csv"
if not pred_path.exists():
    raise FileNotFoundError("Run 04_ensemble.py first.")

pred_df = pd.read_csv(pred_path)
pred_df["phage"] = pred_df["phage"].astype(str).str.strip()
pred_df["host"]  = pred_df["host"].astype(str).str.strip()
if "genus" not in pred_df.columns:
    pred_df["genus"] = pred_df["host"].str.split().str[0]

# Attach strain annotations
pred_df["strain"] = pred_df["phage"]
cache = RAW_DIR / "phage_strain_annotations.csv"
if cache.exists():
    strain_lu = dict(zip(
        pd.read_csv(cache)["accession"],
        pd.read_csv(cache)["strain"]))
    def resolve(row):
        m = re.search(r'(N[CZ]_\d{6}|[A-Z]{2}\d{5,6})',
                       str(row["phage"]).upper().replace(" ","_"))
        if m:
            acc = m.group(1)
            if acc in strain_lu:
                h = str(strain_lu[acc]).lower()
                if row["host"].split()[0] in h:
                    h = re.sub(r'\bsubsp\.?\s+\w+\b','',h)
                    h = re.sub(r'\b(str\.|strain)\s+','',h)
                    return h.strip()
        return row["phage"]
    pred_df["strain"] = pred_df.apply(resolve, axis=1)

print(f"  Predictions: {len(pred_df)} rows | "
      f"Species: {pred_df['host'].nunique()} | "
      f"Positives: {pred_df['label'].sum()}")

# ── HELPER FUNCTIONS ──
def get_pos_strains(sp_df):
    return set(sp_df[sp_df["label"]==1]["strain"].unique())

def strain_cov(sp_df, selected, pos_strains):
    n = len(pos_strains)
    if n == 0: return 0.0
    covered = set(sp_df[sp_df["phage"].isin(selected) &
                         (sp_df["label"]==1)]["strain"].unique()) & pos_strains
    return len(covered) / n

def robust_cov(sp_df, selected, pos_strains):
    if len(selected) < 2: return 0.0
    probs = {ph: sp_df[sp_df["phage"]==ph]["ensemble_proba"].max() for ph in selected}
    worst = max(probs, key=probs.get)
    return strain_cov(sp_df, [p for p in selected if p != worst], pos_strains)

def single_sel(sp_df):
    return [sp_df.sort_values("ensemble_proba", ascending=False)["phage"].iloc[0]]

def topk_sel(sp_df, k):
    return sp_df.sort_values("ensemble_proba", ascending=False)["phage"].head(k).tolist()

def greedy_sel(sp_df, k, pos_strains):
    pos_df  = sp_df[sp_df["label"]==1]
    covered = set(); sel = []
    rem     = list(sp_df["phage"].unique())
    for _ in range(k):
        if not rem or covered == pos_strains: break
        best, best_g = None, -1
        for ph in rem:
            ns = len((set(pos_df[pos_df["phage"]==ph]["strain"]) & pos_strains) - covered)
            pb = sp_df[sp_df["phage"]==ph]["ensemble_proba"].max()
            if ns > best_g or (ns==best_g and best is not None and
                    pb > sp_df[sp_df["phage"]==best]["ensemble_proba"].max()):
                best_g, best = ns, ph
        if best is None: break
        covered |= set(pos_df[pos_df["phage"]==best]["strain"]) & pos_strains
        sel.append(best); rem.remove(best)
    return sel

def rand_sel(sp_df, k, rng):
    phages = sp_df["phage"].unique()
    return list(rng.choice(phages, size=min(k, len(phages)), replace=False))

# ── PER-SPECIES COMPARISON ──
print(f"\n[1] Per-species comparison (k={K_COCKTAIL})...")
rng = np.random.default_rng(42)
species_list = sorted(pred_df["host"].unique())
rows = []

for sp in species_list:
    sp_df   = pred_df[pred_df["host"]==sp].copy().reset_index(drop=True)
    n_pos   = (sp_df["label"]==1).sum()
    if n_pos < 2: continue
    pos_str = get_pos_strains(sp_df)
    if len(pos_str) == 0: continue

    single = single_sel(sp_df)
    tk     = topk_sel(sp_df, K_COCKTAIL)
    gr     = greedy_sel(sp_df, K_COCKTAIL, pos_str)
    rand_c = [strain_cov(sp_df, rand_sel(sp_df, K_COCKTAIL, rng), pos_str)
               for _ in range(N_RANDOM)]

    rows.append({
        "species":    sp,
        "genus":      sp_df["genus"].iloc[0],
        "n_pos":      n_pos,
        "n_strains":  len(pos_str),
        "single_cov":      strain_cov(sp_df, single, pos_str),
        "topk_cov":        strain_cov(sp_df, tk, pos_str),
        "random_cov":      np.mean(rand_c),
        "greedy_cov":      strain_cov(sp_df, gr, pos_str),
        "single_robust":   robust_cov(sp_df, single*3, pos_str),  # single used 3×
        "topk_robust":     robust_cov(sp_df, tk, pos_str),
        "greedy_robust":   robust_cov(sp_df, gr, pos_str),
    })

cmp_df = pd.DataFrame(rows)
cmp_df.to_csv(RESULTS_DIR / "clinical_strategy_comparison.csv", index=False)
print(f"  Analyzed: {len(cmp_df)} species")

# ── GLOBAL METRICS ──
print(f"\n[2] Global metrics @ k={K_COCKTAIL}...")
strategies = ["single","topk","random","greedy"]
strategy_labels = {"single":"Single phage","topk":"Top-K","random":"Random","greedy":"Greedy (optimized)"}
means, pct75s, robustness = {}, {}, {}

for s in strategies:
    col = f"{s}_cov"
    means[s]  = cmp_df[col].mean()
    pct75s[s] = (cmp_df[col] >= 0.75).mean() * 100
    rob_col   = f"{s}_robust"
    robustness[s] = cmp_df[rob_col].mean() if rob_col in cmp_df.columns else np.nan

print(f"\n  {'Strategy':<22} {'Mean Cov':>10} {'≥75%':>8} {'Robust':>10}")
print("  " + "-"*55)
for s in strategies:
    print(f"  {strategy_labels[s]:<22} {means[s]:>10.3f} {pct75s[s]:>7.1f}%  {robustness[s]:>10.3f}")

# ── STATISTICAL TESTS ──
print(f"\n[3] Wilcoxon tests vs Greedy @ k={K_COCKTAIL}...")

def wtest(a, b, label, key):
    d  = np.array(a) - np.array(b)
    nz = d[d != 0]
    if len(nz) < 5: return 1.0
    _, p = wilcoxon(nz)
    sig  = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
    print(f"  Greedy vs {label:<16}: p={p:.4f} {sig}  "
          f"(Greedy better by {means['greedy']-means[key]:+.3f})")
    return p

p_vs_single = wtest(cmp_df["greedy_cov"], cmp_df["single_cov"], "Single", "single")
p_vs_topk   = wtest(cmp_df["greedy_cov"], cmp_df["topk_cov"],   "Top-K",  "topk")
p_vs_random = wtest(cmp_df["greedy_cov"], cmp_df["random_cov"], "Random", "random")

improvement = {
    "over_single": means["greedy"] - means["single"],
    "over_topk":   means["greedy"] - means["topk"],
    "over_random": means["greedy"] - means["random"],
}

pd.DataFrame({
    "mean_coverage":  means,
    "pct75_coverage": pct75s,
    "robustness":     robustness,
}).to_csv(RESULTS_DIR / "clinical_strategy_metrics.csv")

# ── PLOTS ──
print("\n[4] Generating plots...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: Coverage comparison bars
ax = axes[0]
strats_ordered = ["single","random","topk","greedy"]
labels_o = [strategy_labels[s] for s in strats_ordered]
vals_o   = [means[s] for s in strats_ordered]
colors_o = ["#F44336","#BDBDBD","#4285F4","#1a9850"]
bars_a   = ax.bar(labels_o, vals_o, color=colors_o, alpha=0.85)
for bar, v in zip(bars_a, vals_o):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.3f}",
             ha="center", fontsize=10, fontweight="bold")
# Improvement arrows
for i, s in enumerate(strats_ordered[:-1]):
    imp = means["greedy"] - means[s]
    ax.annotate("", xy=(3, means["greedy"]), xytext=(i, means[s]),
                  arrowprops=dict(arrowstyle="-", color="gray", lw=0.8, linestyle="dashed"),
                  annotation_clip=False)
ax.axhline(0.75, color="gold", linestyle="--", lw=1.5, label="75% threshold")
ax.set_ylabel("Mean strain coverage", fontsize=12)
ax.set_title(f"A  Clinical Strategy Coverage\n(k={K_COCKTAIL}, strain-level)",
              fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.set_ylim(0, 1.1); ax.grid(axis="y", alpha=0.4)

# Panel B: % Species ≥75% coverage
ax = axes[1]
pcts = [pct75s[s] for s in strats_ordered]
bars_b = ax.bar(labels_o, pcts, color=colors_o, alpha=0.85)
for bar, v in zip(bars_b, pcts):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.5, f"{v:.1f}%",
             ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("% species ≥75% coverage", fontsize=12)
ax.set_title(f"B  % Species Meeting Clinical Threshold\n(≥75% strain coverage @ k={K_COCKTAIL})",
              fontsize=11, fontweight="bold")
ax.set_ylim(0, 105); ax.grid(axis="y", alpha=0.4)

# Panel C: Robustness after resistance
ax = axes[2]
rob_vals   = [robustness.get(s, 0) for s in strats_ordered]
drop_vals  = [means[s] - robustness.get(s, 0) for s in strats_ordered]
x   = np.arange(len(strats_ordered)); w = 0.35
ax.bar(x, [means[s] for s in strats_ordered], w, color=colors_o, alpha=0.85, label="Normal")
ax.bar(x, rob_vals, w*0.7, color=colors_o, alpha=0.4, label="Post-resistance", hatch="//")
for xi, (n, r) in enumerate(zip([means[s] for s in strats_ordered], rob_vals)):
    ax.text(xi, max(n,r)+0.02, f"↓{n-r:.3f}", ha="center", fontsize=8, color="#555")
ax.set_xticks(x); ax.set_xticklabels(labels_o, rotation=10)
ax.set_ylabel("Mean strain coverage", fontsize=12)
ax.set_title("C  Resistance Robustness\n(Solid=normal, Hatched=post-resistance)",
              fontsize=11, fontweight="bold")
ax.legend(fontsize=8); ax.set_ylim(0, 1.15); ax.grid(axis="y", alpha=0.4)

plt.suptitle("PrecisionPhage — Clinical Strategy Comparison (Strain-Level)",
              fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
save_plot(fig, "31_clinical_comparison")

# ── FINAL SUMMARY ──
print()
print("=" * 62)
print("  CLINICAL STRATEGY COMPARISON — SUMMARY")
print("=" * 62)
print(f"""
  Mean strain coverage @ k={K_COCKTAIL}:
    Single phage:       {means['single']:.3f}  (robustness: {robustness['single']:.3f})
    Top-K (naive):      {means['topk']:.3f}  (robustness: {robustness['topk']:.3f})
    Random:             {means['random']:.3f}  (robustness: {robustness['random']:.3f})
    Greedy (optimized): {means['greedy']:.3f}  (robustness: {robustness['greedy']:.3f})

  Improvement over single:  +{improvement['over_single']:.3f}  (p={p_vs_single:.4f})
  Improvement over top-K:   +{improvement['over_topk']:.3f}  (p={p_vs_topk:.4f})
  Improvement over random:  +{improvement['over_random']:.3f}  (p={p_vs_random:.4f})

  % species ≥75% coverage:
    Single:  {pct75s['single']:.1f}%
    Top-K:   {pct75s['topk']:.1f}%
    Random:  {pct75s['random']:.1f}%
    Greedy:  {pct75s['greedy']:.1f}%

  Clinical interpretation:
  Greedy optimization increases the fraction of species reaching
  the 75% clinical threshold by {pct75s['greedy']-pct75s['single']:+.1f}pp vs single phage.
  After resistance to the best phage, the greedy cocktail retains
  {robustness['greedy']:.1%} of its coverage vs {robustness['single']:.1%} for single-phage therapy.
""")
print(f"  Results: {RESULTS_DIR.resolve()}")
print(f"  Plots:   {PLOT_DIR.resolve()}")
print("\n  Done!")