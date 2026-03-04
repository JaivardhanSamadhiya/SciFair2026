"""
11_master_summary.py  —  Master Summary Block
==============================================
PrecisionPhage | ISEF 2026

Reads all previously saved results CSVs and prints the complete
PRECISIONPHAGE TRANSLATIONAL VALIDATION SUMMARY.
Run AFTER all other scripts have completed.
"""

import pandas as pd, numpy as np
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR    = _SCRIPT_DIR.parent / "data"
RESULTS_DIR = BASE_DIR / "results"

def load_val(path, key=None, default="N/A"):
    """Load a single value from a results CSV."""
    try:
        df = pd.read_csv(path, index_col=0, header=0)
        if key:
            return float(df.loc[key].values[0])
        return float(df.iloc[0,0])
    except Exception:
        return default

def load_df(path):
    try: return pd.read_csv(path)
    except Exception: return pd.DataFrame()

print("=" * 64)
print("  Loading results...")

# LOSO (from ensemble)
ens_path = RESULTS_DIR / "ensemble_metrics.csv"
loso_auc = load_val(ens_path, "Ensemble")
ens_df = load_df(ens_path)
loso_pr_auc = float(ens_df["pr_auc"].iloc[0]) if "pr_auc" in ens_df.columns else "N/A"
loso_mcc    = float(ens_df["mcc"].iloc[0])    if "mcc"    in ens_df.columns else "N/A"
loso_f1     = float(ens_df["f1"].iloc[0])     if "f1"     in ens_df.columns else "N/A"

# S. aureus
sp_df = load_df(RESULTS_DIR / "ensemble_per_species.csv")
sa_auc = "N/A"
if len(sp_df):
    sp_df.columns = sp_df.columns.str.strip()
    sp_col = sp_df.columns[0]
    auc_col = [c for c in sp_df.columns if "roc_auc" in c.lower()]
    if auc_col:
        sa_row = sp_df[sp_df[sp_col].str.contains("aureus", case=False, na=False)]
        if len(sa_row):
            sa_auc = f"{float(sa_row[auc_col[0]].iloc[0]):.4f}"

# LOGO
logo_sum  = load_df(RESULTS_DIR / "logo_summary.csv")
logo_auc  = "N/A"; logo_std = ""; logo_pval = "N/A"; loso_logo_delta = "N/A"
if len(logo_sum):
    logo_sum.columns = logo_sum.columns.str.strip()
    idx_col = logo_sum.columns[0]
    logo_sum = logo_sum.set_index(idx_col)
    try:
        logo_auc  = f"{float(logo_sum.loc['logo_mean_auc'].values[0]):.4f}"
        logo_std  = f"±{float(logo_sum.loc['logo_std_auc'].values[0]):.4f}"
        try: logo_pval = f"{float(logo_sum.loc['wilcoxon_p'].values[0]):.4f}"
        except: pass
        try:
            d = float(logo_sum.loc['delta_logo_minus_loso'].values[0])
            loso_logo_delta = f"{d:+.4f}"
        except: pass
    except Exception: pass

# Unseen strain
mc_df   = load_df(RESULTS_DIR / "unseen_strain_mc_results.csv")
mc_auc  = f"{mc_df['mean_auc'].mean():.4f} ± {mc_df['mean_auc'].std():.4f}" if len(mc_df) else "N/A"
mc_cov  = f"{mc_df['mean_cov3'].mean():.3f} ± {mc_df['mean_cov3'].std():.3f}" if len(mc_df) else "N/A"

# Cocktail (strain-level)
gm_path = RESULTS_DIR / "cocktail_global_metrics.csv"
gm_df   = load_df(gm_path)
s_cov3 = "N/A"; s_rand3 = "N/A"; s_pct75_3 = "N/A"; s_pct75_5 = "N/A"
s_rob3 = "N/A"
if len(gm_df):
    gm_df.columns = gm_df.columns.str.strip()
    idx_col = gm_df.columns[0]
    gm_dict = dict(zip(gm_df[idx_col], gm_df.iloc[:,1]))
    s_cov3     = f"{float(gm_dict.get('mean_s_greedy@3', 0)):.3f}"
    s_rand3    = f"{float(gm_dict.get('mean_s_random@3', 0)):.3f}"
    s_pct75_3  = f"{float(gm_dict.get('pct75_s_greedy@3', 0)):.1f}%"
    s_pct75_5  = f"{float(gm_dict.get('pct75_s_greedy@5', 0)):.1f}%"
    drop3      = float(gm_dict.get('mean_s_covdrop@3', 0)) * 100
    s_rob3     = f"{drop3:.1f}% coverage drop"
    over_rand  = float(gm_dict.get('s_greedy_over_random@3', 0)) * 100

# Clinical comparison
clin_df = load_df(RESULTS_DIR / "clinical_strategy_metrics.csv")
clin_single = clin_topk = "N/A"
if len(clin_df):
    clin_df.columns = clin_df.columns.str.strip()
    if "mean_coverage" in clin_df.columns:
        clin_df = clin_df.set_index(clin_df.columns[0])
        try:
            clin_single = f"{float(clin_df.loc['single','mean_coverage']):.3f}"
            clin_topk   = f"{float(clin_df.loc['topk','mean_coverage']):.3f}"
            clin_greedy = f"{float(clin_df.loc['greedy','mean_coverage']):.3f}"
            imp_single  = f"+{float(clin_df.loc['greedy','mean_coverage'])-float(clin_df.loc['single','mean_coverage']):.3f}"
            imp_topk    = f"+{float(clin_df.loc['greedy','mean_coverage'])-float(clin_df.loc['topk','mean_coverage']):.3f}"
        except: pass

# Feature importance
feat_cat = load_df(RESULTS_DIR / "feature_category_importance.csv")
top_feats = load_df(RESULTS_DIR / "feature_importance.csv")
bio_feats = "N/A"
if len(top_feats):
    bio_feats = ", ".join(top_feats.head(5)["feature"].tolist())

# CI
ci_df = load_df(RESULTS_DIR / "cocktail_bootstrap_ci.csv")
ci_str = "N/A"
if len(ci_df):
    ci_df.columns = ci_df.columns.str.strip()
    idx_col = ci_df.columns[0]
    ci_dict = ci_df.set_index(idx_col)
    try:
        lo = float(ci_dict.loc["s_greedy_ci@3","ci_lo"])
        hi = float(ci_dict.loc["s_greedy_ci@3","ci_hi"])
        ci_str = f"[{lo:.3f} – {hi:.3f}]"
    except: pass

print()
print("=" * 64)
print("  PRECISIONPHAGE — TRANSLATIONAL VALIDATION SUMMARY")
print("=" * 64)
print(f"""
  ┌─────────────────── MODEL PERFORMANCE ───────────────────┐
  │  LOSO ROC-AUC:       {str(loso_auc):<8}   PR-AUC: {str(loso_pr_auc):<8}      │
  │  LOSO MCC:           {str(loso_mcc):<8}   F1:     {str(loso_f1):<8}      │
  │  S. aureus AUC:      {sa_auc:<38}  │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────── CROSS-GENUS VALIDATION ──────────────┐
  │  LOGO ROC-AUC:       {logo_auc} {logo_std}                          │
  │  LOSO→LOGO delta:    {loso_logo_delta:<38}  │
  │  Wilcoxon p-value:   {logo_pval:<38}  │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────── PROSPECTIVE VALIDATION ──────────────┐
  │  Unseen strain AUC:  {mc_auc:<38}  │
  │  Unseen coverage@3:  {mc_cov:<38}  │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────── COCKTAIL OPTIMIZATION ───────────────┐
  │  Strain coverage@3 (Optimized): {s_cov3}                      │
  │  Strain coverage@3 (Random):    {s_rand3}                      │
  │  95% CI coverage@3:             {ci_str}                │
  │  Improvement over random:       +{over_rand if isinstance(over_rand,str) else f'{over_rand:.1f}%':<22}  │
  │  % species ≥75% @ k=3:          {s_pct75_3:<22}           │
  │  % species ≥75% @ k=5:          {s_pct75_5:<22}           │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────── CLINICAL COMPARISON ─────────────────┐
  │  Single phage coverage:         {clin_single:<22}           │
  │  Top-K coverage:                {clin_topk:<22}           │
  │  Optimized coverage:            {s_cov3:<22}           │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────── RESISTANCE ROBUSTNESS ───────────────┐
  │  k=3 coverage drop after resistance: {s_rob3:<20}      │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────── BIOLOGICAL FEATURES ─────────────────┐
  │  Top features: {bio_feats[:50]:<50}  │
  └─────────────────────────────────────────────────────────┘
""")
print("=" * 64)
print(f"  Results directory: {RESULTS_DIR.resolve()}")
print("=" * 64)
print("\n  All modules complete. PrecisionPhage pipeline ready.")