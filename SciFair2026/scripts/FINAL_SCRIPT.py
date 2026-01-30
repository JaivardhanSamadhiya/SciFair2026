import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc
from sklearn.base import clone

import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
BASE_DIR = Path("SciFair2026/data/raw")

VIRUS_HOST_INTER = BASE_DIR / "VirusHostInter.csv"
POSITIVE_PAIRS_TXT = BASE_DIR / "phage-bacteria-pairs.txt"
VHR_STAPH_XLSX = BASE_DIR / "VHRStaph.xlsx"

TARGET_HOST = "staphylococcus aureus"
RANDOM_SEED = 42
NEG_RATIO = 3  # 1:3 pos:neg

np.random.seed(RANDOM_SEED)

# =========================
# HELPERS
# =========================
def normalize_name(x):
    return str(x).strip().lower().replace("_", " ")

# =========================
# LOAD MILLARD DATA
# =========================
print("Loading Millard data...")

vhi_df = pd.read_csv(VIRUS_HOST_INTER)
vhi_df = vhi_df.loc[:, ~vhi_df.columns.str.contains("^Unnamed")]

vhi_df = vhi_df.rename(columns={
    "hostname": "host",
    "phagename": "phage",
    "infection": "interaction"
})

for c in ["host", "phage", "interaction"]:
    vhi_df[c] = vhi_df[c].apply(normalize_name)

# =========================
# LOAD PHAGE–BACTERIA PAIRS (SAFE)
# =========================
print("Loading phage-bacteria-pairs.txt...")

pbp_raw = pd.read_csv(POSITIVE_PAIRS_TXT, sep="\t")

pbp_df = pbp_raw.loc[:, ["phage_id", "host_species"]].copy()
pbp_df = pbp_df.rename(columns={
    "phage_id": "phage",
    "host_species": "host"
})

pbp_df["phage"] = pbp_df["phage"].apply(normalize_name)
pbp_df["host"] = pbp_df["host"].apply(normalize_name)

assert pbp_df.columns.is_unique

# =========================
# LOAD VHR STAPH
# =========================
print("Loading VHRStaph.xlsx...")

vhr_df = pd.read_excel(VHR_STAPH_XLSX)
vhr_df = vhr_df.rename(columns={vhr_df.columns[0]: "phage"})
vhr_df["phage"] = vhr_df["phage"].apply(normalize_name)

host_col = [c for c in vhr_df.columns if "staphylococcus" in c.lower()][0]

# =========================
# POSITIVE INTERACTIONS
# =========================
print("Extracting positives...")

pos_vhi = vhi_df[
    (vhi_df["host"] == TARGET_HOST) &
    (vhi_df["interaction"] == "inf")
][["phage", "host"]].copy()
pos_vhi["label"] = 1

pos_pbp = pbp_df[
    pbp_df["host"] == TARGET_HOST
][["phage", "host"]].copy()
pos_pbp["label"] = 1

pos_vhr = vhr_df[
    vhr_df[host_col].notna() & (vhr_df[host_col] != 0)
][["phage"]].copy()
pos_vhr["host"] = TARGET_HOST
pos_vhr["label"] = 1

positives = pd.concat([pos_vhi, pos_pbp, pos_vhr]).drop_duplicates()
print(f"Total positives: {len(positives)}")

# =========================
# NEGATIVE SAMPLING (1:3)
# =========================
print("Generating high-confidence negatives...")

neg_candidates = vhi_df[vhi_df["interaction"] == "noinf"][["phage", "host"]]

pos_set = set(zip(positives["phage"], positives["host"]))
neg_candidates = neg_candidates[
    ~neg_candidates.apply(lambda r: (r["phage"], r["host"]) in pos_set, axis=1)
]

n_pos = len(positives)
n_neg_total = NEG_RATIO * n_pos
n_close = n_neg_total // 3
n_far = n_neg_total - n_close

close_neg = neg_candidates[
    (neg_candidates["host"].str.contains("staphylococcus")) &
    (neg_candidates["host"] != TARGET_HOST)
]

far_neg = neg_candidates[
    ~neg_candidates["host"].str.contains("staphylococcus")
]

close_sample = close_neg.sample(
    n=min(n_close, len(close_neg)), random_state=RANDOM_SEED
)
far_sample = far_neg.sample(
    n=min(n_far, len(far_neg)), random_state=RANDOM_SEED
)

negatives = pd.concat([close_sample, far_sample])
negatives["label"] = 0

print(f"Negatives: {len(negatives)}")

# =========================
# FINAL DATASET
# =========================
dataset = pd.concat([positives, negatives]) \
    .sample(frac=1, random_state=RANDOM_SEED) \
    .reset_index(drop=True)

print(dataset["label"].value_counts())

# =========================
# BASELINE MODEL
# =========================
dataset["pair"] = dataset["phage"]  # phage-only baseline
X = dataset["pair"]
y = dataset["label"]

pipeline = Pipeline([
    ("vectorizer", CountVectorizer(analyzer="char", ngram_range=(3, 5))),
    ("clf", LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=RANDOM_SEED
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

results = cross_validate(
    pipeline, X, y,
    cv=cv,
    scoring=["accuracy", "roc_auc", "average_precision"]
)

print("\n=== CV Results (1:3 pos:neg) ===")
print(f"Accuracy : {results['test_accuracy'].mean():.4f}")
print(f"ROC-AUC  : {results['test_roc_auc'].mean():.4f}")
print(f"PR-AUC   : {results['test_average_precision'].mean():.4f}")

# =========================
# PR-AUC CURVE (CROSS-VALIDATED)
# =========================
print("\nGenerating PR-AUC curve...")

mean_recall = np.linspace(0, 1, 100)
pr_curves = []
aucs = []

for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = clone(pipeline)
    model.fit(X_train, y_train)

    scores = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, scores)
    pr_auc = auc(recall, precision)

    aucs.append(pr_auc)
    interp_prec = np.interp(mean_recall, recall[::-1], precision[::-1])
    pr_curves.append(interp_prec)

mean_precision = np.mean(pr_curves, axis=0)
std_precision = np.std(pr_curves, axis=0)
mean_auc = np.mean(aucs)
std_auc = np.std(aucs)

# =========================
# PLOT
# =========================
plt.figure(figsize=(7, 6))
plt.plot(
    mean_recall,
    mean_precision,
    label=f"Mean PR-AUC = {mean_auc:.3f} ± {std_auc:.3f}"
)

plt.fill_between(
    mean_recall,
    np.maximum(mean_precision - std_precision, 0),
    np.minimum(mean_precision + std_precision, 1),
    alpha=0.2
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (5-fold CV, 1:3 pos:neg)")
plt.legend(loc="lower left")
plt.grid(True)
plt.tight_layout()
plt.show()
