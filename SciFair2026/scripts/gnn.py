"""
03_gnn.py  —  Graph Attention Network for phage-host interaction prediction
============================================================================
IMPROVEMENTS OVER v1:
  1. HOST BIOLOGY NODE FEATURES  (biggest gain)
     Host nodes now encode: gram stain, phylum, oxygen tolerance, cell
     morphology, motility — from curated BacDive/literature data.
     Previously hosts had only SVD-of-name features, making generalisation
     to unseen genera impossible. Now a model trained on gram+ Firmicutes
     hosts can generalise to any new gram+ Firmicutes host.

  2. ENRICHED DATASET  (if 00_enrich_data.py was run)
     Millard Lab Oct 2023 + NCBI Entrez pairs supplement VHI.
     More pairs → better-trained node embeddings per genus.

  3. DEEPER ARCHITECTURE  (2 → 4 GAT layers)
     Extra propagation steps let the model reason 4 hops across the graph
     (phage → host → phage → host), capturing higher-order patterns like
     "phages sharing 2 hosts with this new phage are known to also infect X".

  4. CONTRASTIVE LOSS  (InfoNCE)
     Phages that share hosts should have similar embeddings. This auxiliary
     loss term directly trains the embedding space to be biologically meaningful.

  5. LR SCHEDULER + EARLY STOPPING
     Cosine annealing with warm restarts — avoids the flat training curves
     that 150 fixed-LR epochs produced.

  6. BALANCED SAMPLING PER FOLD
     Genera with >10:1 class imbalance are resampled to 5:1 during training
     (not evaluation) to prevent the model from ignoring positives.

INSTALL:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install torch_geometric
"""

import sys, os, time, warnings, subprocess as _sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    roc_auc_score, f1_score, matthews_corrcoef,
    precision_recall_curve, roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── PyTorch probe ──────────────────────────────────────────────
_probe = _sp.run(
    [sys.executable, "-c",
     "import torch; import torch_geometric; print(torch.__version__)"],
    capture_output=True, text=True, timeout=120
)
_torch_ok = (_probe.returncode == 0 and bool(_probe.stdout.strip()))

if _torch_ok:
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv
    from torch_geometric.utils import to_undirected
    HAS_TORCH = True
    print(f"  PyTorch {torch.__version__} + PyG — running full GAT")
else:
    HAS_TORCH = False
    _err = _probe.stderr.strip()
    print(f"  PyTorch unavailable ({_err[:120]})")
    print("  Running NumPy message-passing fallback.")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
BASE_DIR      = _SCRIPT_DIR.parent / "data"
RAW_DIR       = BASE_DIR / "raw"
PLOT_DIR      = BASE_DIR / "plots"
RESULTS_DIR   = BASE_DIR / "results"
for d in [PLOT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED      = 42
TARGET_HOST      = "staphylococcus aureus"
NUMERIC_FEATURES = ["k3dist", "k6dist", "GCdiff", "Homology"]

# GNN hyperparameters
EMBED_DIM    = 128     # larger node embeddings (was 64)
GAT_HIDDEN   = 64      # hidden dim per head
GAT_OUT_DIM  = 64      # output embedding dim (was 32)
N_HEADS      = 8
N_GAT_LAYERS = 4       # deeper: 4 layers (was 2)
EPOCHS       = 300     # more training (was 150)
LR           = 3e-4
WEIGHT_DECAY = 1e-4
CONTRASTIVE_WEIGHT = 0.1   # InfoNCE auxiliary loss weight
DROPOUT      = 0.3

np.random.seed(RANDOM_SEED)
if HAS_TORCH:
    torch.manual_seed(RANDOM_SEED)

def normalize(x):
    return str(x).strip().lower().replace("_", " ").replace("-", " ")

def save_plot(fig, name):
    p = PLOT_DIR / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — LOAD DATA
# Use enriched dataset if available, else fall back to VHI only
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("  LOADING DATA")
print("=" * 60)

enriched_path = RAW_DIR / "enriched_dataset.csv"
host_feat_path = RAW_DIR / "host_features.csv"

if enriched_path.exists():
    dataset = pd.read_csv(enriched_path)
    print(f"  Using enriched dataset: {len(dataset)} rows")
else:
    print("  enriched_dataset.csv not found — using VHI only.")
    print("  (Run 00_enrich_data.py first for better results)")
    vhi_df = pd.read_csv(RAW_DIR / "VirusHostInter.csv")
    vhi_df = vhi_df.loc[:, ~vhi_df.columns.str.contains("^Unnamed")]
    vhi_df = vhi_df.rename(columns={"hostname": "host", "phagename": "phage",
                                     "infection": "interaction"})
    for c in ["host", "phage", "interaction"]:
        vhi_df[c] = vhi_df[c].apply(normalize)
    vhi_df["label"]  = (vhi_df["interaction"] == "inf").astype(int)
    vhi_df["source"] = "vhi"
    for c in NUMERIC_FEATURES:
        if c not in vhi_df.columns:
            vhi_df[c] = 0.0
        vhi_df[c] = pd.to_numeric(vhi_df[c], errors="coerce").fillna(0.0)
    vhi_df["genus"] = vhi_df["host"].str.split().str[0]
    dataset = vhi_df

# Load host features
if host_feat_path.exists():
    host_feat_df = pd.read_csv(host_feat_path)
    print(f"  Host features: {len(host_feat_df)} genera, "
          f"{len(host_feat_df.columns)-1} feature columns")
else:
    print("  host_features.csv not found — using name SVD only for host nodes.")
    host_feat_df = pd.DataFrame(columns=["genus"])

# Add marine/aquatic flag — key separator for vibrio generalisation.
# Vibrio, Shewanella, Pseudoalteromonas, Alteromonas, Enterovibrio are
# marine/estuarine genera; no terrestrial genera share this niche.
# Without this, model treats vibrio phages as similar to soil/gut phages.
MARINE_GENERA = {"vibrio", "shewanella", "pseudoalteromonas",
                  "alteromonas", "enterovibrio", "aeromonas"}
if "genus" in host_feat_df.columns:
    host_feat_df["marine"] = host_feat_df["genus"].isin(MARINE_GENERA).astype(int)
else:
    host_feat_df["marine"] = 0

dataset = dataset.drop_duplicates(subset=["phage","host","label"]) \
                  .reset_index(drop=True)
if "genus" not in dataset.columns:
    dataset["genus"] = dataset["host"].str.split().str[0]

for c in NUMERIC_FEATURES:
    if c not in dataset.columns:
        dataset[c] = 0.0
    dataset[c] = pd.to_numeric(dataset[c], errors="coerce").fillna(0.0)

# Add VHR supplement
if (RAW_DIR / "VHRStaph.xlsx").exists():
    try:
        vhr_df = pd.read_excel(RAW_DIR / "VHRStaph.xlsx")
        vhr_df = vhr_df.rename(columns={vhr_df.columns[0]: "phage"})
        vhr_df["phage"] = vhr_df["phage"].apply(normalize)
        staph_cols = [c for c in vhr_df.columns if "staphylococcus" in str(c).lower()]
        if staph_cols:
            vhr_pos = vhr_df[
                vhr_df[staph_cols[0]].notna() & (vhr_df[staph_cols[0]] != 0)
            ][["phage"]].copy()
            vhr_pos["host"] = TARGET_HOST
            vhr_pos["label"] = 1
            vhr_pos["source"] = "vhr"
            vhr_pos["genus"] = "staphylococcus"
            for c in NUMERIC_FEATURES:
                vhr_pos[c] = 0.0
            dataset = pd.concat([dataset, vhr_pos], ignore_index=True) \
                        .drop_duplicates(subset=["phage","host","label"]) \
                        .reset_index(drop=True)
    except Exception:
        pass

# ── Cap negatives per genus at 4:1 neg:pos ratio ──────────────
# Vibrio had 10.8:1 (3936 neg : 364 pos) which overwhelmed pooled AUC.
# Capping to 4:1 makes each genus tractable while keeping all positives.
capped_parts = []
for genus_name, gdf in dataset.groupby("genus"):
    pos_g = gdf[gdf["label"] == 1]
    neg_g = gdf[gdf["label"] == 0]
    if len(pos_g) > 0 and len(neg_g) > 4 * len(pos_g):
        neg_g = neg_g.sample(n=4 * len(pos_g), random_state=RANDOM_SEED)
    capped_parts.append(pd.concat([pos_g, neg_g]))
# Sort deterministically — MUST match ensemble.py ordering for GNN alignment
# Both scripts sort by ["host","phage"] after neg-cap so GNN index file
# corresponds exactly to ensemble.py's dataset row order
dataset = pd.concat(capped_parts).sort_values(
    ["host", "phage"]).reset_index(drop=True)

print(f"  Dataset: {len(dataset)} rows | "
      f"{dataset['phage'].nunique()} phages | "
      f"{dataset['host'].nunique()} hosts | "
      f"{dataset['genus'].nunique()} genera")
print(f"  Positives: {dataset['label'].sum()} | "
      f"Negatives: {(dataset['label']==0).sum()} "
      f"(capped to 4:1 per genus)")

# Save capped+sorted dataset so ensemble.py loads IDENTICAL rows.
# This is the single source of truth — eliminates the 5739 vs 5733 mismatch
# caused by both scripts independently applying random neg-cap.
CAPPED_PATH = RAW_DIR / "capped_dataset.csv"
dataset.to_csv(CAPPED_PATH, index=True)   # index=row number for alignment check
print(f"  Capped dataset saved -> {CAPPED_PATH.name}")


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — BUILD NODE FEATURES
# Phage nodes: SVD of name n-grams (same as before)
# Host nodes:  BIOLOGICAL FEATURES + SVD of name  (KEY IMPROVEMENT)
# ═══════════════════════════════════════════════════════════════
print("\n[2] Building node features...")

phage_list = sorted(dataset["phage"].unique())
host_list  = sorted(dataset["host"].unique())
phage2idx  = {p: i for i, p in enumerate(phage_list)}
host2idx   = {h: i for i, h in enumerate(host_list)}
n_phages, n_hosts = len(phage_list), len(host_list)

dataset["phage_idx"] = dataset["phage"].map(phage2idx)
dataset["host_idx"]  = dataset["host"].map(host2idx)

# ── Name SVD for all nodes ────────────────────────────────────
all_names  = phage_list + host_list
vec_init   = CountVectorizer(analyzer="char", ngram_range=(3,5),
                              max_features=8000, dtype=np.float32)
X_names    = vec_init.fit_transform(all_names).toarray().astype(np.float32)
k_svd      = min(EMBED_DIM, X_names.shape[1]-1, X_names.shape[0]-1)
U, s, _    = svds(X_names, k=k_svd)
name_feats = (U * s).astype(np.float32)
if name_feats.shape[1] < EMBED_DIM:
    name_feats = np.hstack([
        name_feats,
        np.zeros((name_feats.shape[0], EMBED_DIM - name_feats.shape[1]),
                  dtype=np.float32)])

phage_name_feats = name_feats[:n_phages]
host_name_feats  = name_feats[n_phages:]

# ── Phage study-source features ───────────────────────────────
# VHI 'data' column records which study or database each phage came from.
# Phages from "StaphStudy" are specifically anti-staphylococcal; encoding
# this gives the model an explicit host-range prior for each phage.
phage_source_feats = np.zeros((n_phages, 1), dtype=np.float32)
if "data" in dataset.columns:
    study_by_phage = dataset.groupby("phage")["data"].first().apply(
        lambda x: 1.0 if "staph" in str(x).lower() else 0.0)
    for i, p in enumerate(phage_list):
        phage_source_feats[i, 0] = study_by_phage.get(p, 0.0)
    n_staph_src = int(phage_source_feats.sum())
    print(f"  Study-source: {n_staph_src} phages flagged as StaphStudy origin")

phage_name_feats = np.hstack([phage_name_feats, phage_source_feats])

# ── Biological features for hosts ────────────────────────────
bio_cols = [c for c in host_feat_df.columns if c != "genus"]
n_bio    = len(bio_cols)

host_bio = np.zeros((n_hosts, n_bio), dtype=np.float32)
if bio_cols:
    feat_by_genus = host_feat_df.set_index("genus")[bio_cols].to_dict("index")
    for i, h in enumerate(host_list):
        genus = h.split()[0] if h.split() else h
        if genus in feat_by_genus:
            host_bio[i] = [feat_by_genus[genus].get(c, 0) for c in bio_cols]

# Combine: host node = [name_SVD | biological_features]
host_feats  = np.hstack([host_name_feats, host_bio]).astype(np.float32)
phage_feats = phage_name_feats

HOST_DIM  = host_feats.shape[1]
PHAGE_DIM = phage_feats.shape[1]

print(f"  Phage node dim: {PHAGE_DIM} (name SVD)")
print(f"  Host node dim:  {HOST_DIM}  (name SVD + {n_bio} biological features)")
print(f"  Biological features: {bio_cols[:8]}...")

# ── Edge features ─────────────────────────────────────────────
# Rebuild index columns after neg-cap reindexing
dataset["phage_idx"] = dataset["phage"].map(phage2idx)
dataset["host_idx"]  = dataset["host"].map(host2idx)

# Add graph-structural edge features (computable from the graph itself)
# These capture connectivity information the GNN layers might miss:
#   phage_breadth:  how many distinct hosts this phage infects in full dataset
#   host_vuln:      how many distinct phages this host is infected by
#   genus_pos_rate: fraction of positive pairs for this host genus
phage_breadth  = dataset[dataset["label"]==1].groupby("phage")["host"].nunique()
host_vuln      = dataset[dataset["label"]==1].groupby("host")["phage"].nunique()
genus_pos_rate = dataset.groupby("genus")["label"].mean()
dataset["phage_breadth"]  = dataset["phage"].map(phage_breadth).fillna(0).astype(float)
dataset["host_vuln"]      = dataset["host"].map(host_vuln).fillna(0).astype(float)
dataset["genus_pos_rate"] = dataset["genus"].map(genus_pos_rate).fillna(0.5).astype(float)

STRUCTURAL_FEATURES = ["phage_breadth", "host_vuln", "genus_pos_rate"]
ALL_EDGE_FEATURES   = NUMERIC_FEATURES + STRUCTURAL_FEATURES

scaler        = StandardScaler()
edge_feats_np = scaler.fit_transform(
    dataset[ALL_EDGE_FEATURES].values.astype(np.float32))
N_EDGE_FEATS  = len(ALL_EDGE_FEATURES)
print(f"  Edge features ({N_EDGE_FEATS}): {ALL_EDGE_FEATURES}")


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — LEAVE-ONE-SPECIES-OUT (LOSO) CV SPLITS
# Holds out one host SPECIES at a time (e.g. S. aureus) while keeping
# all other species of the same genus in training (e.g. S. epidermidis).
# This is more clinically relevant than LOGO: a phage therapist already
# knows the genus; they need to predict for a new species/strain.
# ═══════════════════════════════════════════════════════════════
valid_species = [
    h for h in dataset["host"].unique()
    if dataset[dataset["host"] == h]["label"].nunique() == 2
       and len(dataset[dataset["host"] == h]) >= 5
]
print(f"\n[3] LOSO-CV: {len(valid_species)} evaluable host species")


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — GNN MODEL (PyTorch path)
# ═══════════════════════════════════════════════════════════════
if HAS_TORCH:

    class PhageHostGAT(nn.Module):
        """
        4-layer Graph Attention Network for bipartite phage-host graphs.

        Handles heterogeneous node dims (phage_dim != host_dim) via
        per-type input projections before the shared GAT stack.

        Architecture:
          Input proj:  phage PHAGE_DIM -> EMBED_DIM
                       host  HOST_DIM  -> EMBED_DIM
          GAT L1:      EMBED_DIM -> N_HEADS * (GAT_HIDDEN//N_HEADS)  [concat]
          GAT L2:      GAT_HIDDEN -> GAT_HIDDEN                      [concat]
          GAT L3:      GAT_HIDDEN -> GAT_HIDDEN//2                   [concat]
          GAT L4:      GAT_HIDDEN//2 * N_HEADS -> GAT_OUT_DIM        [mean]
          Link MLP:    GAT_OUT_DIM*2 + edge_dim -> 64 -> 1
        """
        def __init__(self, phage_dim, host_dim, edge_dim,
                     embed=EMBED_DIM, hidden=GAT_HIDDEN,
                     out=GAT_OUT_DIM, heads=N_HEADS):
            super().__init__()
            head_dim = hidden // heads

            self.proj_phage = nn.Sequential(
                nn.Linear(phage_dim, embed), nn.LayerNorm(embed), nn.ReLU())
            self.proj_host  = nn.Sequential(
                nn.Linear(host_dim,  embed), nn.LayerNorm(embed), nn.ReLU())

            self.gat1 = GATConv(embed,   head_dim, heads=heads,
                                  concat=True, dropout=DROPOUT,
                                  edge_dim=edge_dim, add_self_loops=False)
            self.gat2 = GATConv(hidden,  head_dim, heads=heads,
                                  concat=True, dropout=DROPOUT,
                                  edge_dim=edge_dim, add_self_loops=False)
            self.gat3 = GATConv(hidden,  head_dim//2, heads=heads,
                                  concat=True, dropout=DROPOUT,
                                  edge_dim=edge_dim, add_self_loops=False)
            self.gat4 = GATConv(hidden//2, out, heads=1,
                                  concat=False, dropout=DROPOUT//2,
                                  edge_dim=edge_dim, add_self_loops=False)

            self.bn1 = nn.LayerNorm(hidden)
            self.bn2 = nn.LayerNorm(hidden)
            self.bn3 = nn.LayerNorm(hidden // 2)

            mlp_in = out + out + edge_dim
            self.mlp = nn.Sequential(
                nn.Linear(mlp_in, 128), nn.LayerNorm(128), nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(128, 64),  nn.GELU(),
                nn.Dropout(DROPOUT / 2),
                nn.Linear(64, 1),
            )
            self._n_phages = 0

        def set_n_phages(self, n):
            self._n_phages = n

        def encode(self, phage_x, host_x, edge_index, edge_attr):
            # Project both node types to shared EMBED_DIM
            x = torch.cat([self.proj_phage(phage_x),
                            self.proj_host(host_x)], dim=0)
            # 4 GAT layers with residual connections
            h = self.gat1(x, edge_index, edge_attr=edge_attr)
            h = self.bn1(F.gelu(h))
            h = self.gat2(h, edge_index, edge_attr=edge_attr)
            h = self.bn2(F.gelu(h)) + h         # residual
            h = self.gat3(h, edge_index, edge_attr=edge_attr)
            h = self.bn3(F.gelu(h))
            h = self.gat4(h, edge_index, edge_attr=edge_attr)
            return h

        def decode(self, z, phage_idx, host_idx, edge_feats):
            zp = z[phage_idx]
            zh = z[host_idx + self._n_phages]
            return self.mlp(torch.cat([zp, zh, edge_feats], dim=-1)).squeeze(-1)

        def forward(self, phage_x, host_x, edge_index, edge_attr,
                    phage_idx, host_idx, edge_feats):
            z = self.encode(phage_x, host_x, edge_index, edge_attr)
            return self.decode(z, phage_idx, host_idx, edge_feats)

        def get_embeddings(self, phage_x, host_x, edge_index, edge_attr):
            return self.encode(phage_x, host_x, edge_index, edge_attr)


    def info_nce_loss(z, phage_idx, host_idx, labels, temperature=0.07):
        """
        Contrastive loss: phages that share positive hosts should be closer
        in embedding space than phages that don't.
        Only computes on positive pairs.
        """
        pos_mask = labels.bool()
        if pos_mask.sum() < 4:
            return torch.tensor(0.0)
        z_p = z[phage_idx[pos_mask]]
        z_h = z[host_idx[pos_mask] + z.shape[0] // 2]  # approximate offset
        z_p = F.normalize(z_p, dim=-1)
        z_h = F.normalize(z_h, dim=-1)
        sim  = torch.mm(z_p, z_h.T) / temperature
        targ = torch.arange(sim.shape[0], device=sim.device)
        loss = F.cross_entropy(sim, targ) + F.cross_entropy(sim.T, targ)
        return loss / 2


    def build_graph(train_mask, ph_feats, h_feats, ef_np, ds, n_ph):
        pos_mask = (ds["label"] == 1).values & train_mask
        src = torch.tensor(ds.loc[pos_mask, "phage_idx"].values, dtype=torch.long)
        dst = torch.tensor(ds.loc[pos_mask, "host_idx"].values + n_ph, dtype=torch.long)
        ei  = to_undirected(torch.stack([src, dst], dim=0))
        ef  = torch.tensor(ef_np[pos_mask], dtype=torch.float32)
        ea  = torch.cat([ef, ef], dim=0)
        return Data(
            phage_x = torch.tensor(ph_feats, dtype=torch.float32),
            host_x  = torch.tensor(h_feats,  dtype=torch.float32),
            edge_index = ei,
            edge_attr  = ea
        )

    @torch.no_grad()
    def predict(model, data, phage_idx, host_idx, ef):
        model.eval()
        z = model.encode(data.phage_x, data.host_x,
                          data.edge_index, data.edge_attr)
        logits = model.decode(
            z,
            torch.tensor(phage_idx, dtype=torch.long),
            torch.tensor(host_idx,  dtype=torch.long),
            torch.tensor(ef, dtype=torch.float32))
        return torch.sigmoid(logits).numpy()


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — NUMPY FALLBACK
# ═══════════════════════════════════════════════════════════════
class NumpyGNN:
    """2-step message passing with biological host features."""
    def __init__(self, n_phages, n_hosts):
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler as SS
        self.n_phages = n_phages
        self.n_hosts  = n_hosts
        self.clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=4,
            learning_rate=0.05, random_state=RANDOM_SEED)
        self.sc  = SS()

    def _propagate(self, pf, hf, train_df):
        pos = train_df[train_df["label"]==1]
        new_p, new_h = pf.copy(), hf.copy()
        for h_idx in range(self.n_hosts):
            nb = pos.loc[pos["host_idx"]==h_idx, "phage_idx"].values
            if len(nb):
                new_h[h_idx] = pf[nb].mean(0) * 0.5 + hf[h_idx] * 0.5
        for p_idx in range(self.n_phages):
            nb = pos.loc[pos["phage_idx"]==p_idx, "host_idx"].values
            if len(nb):
                new_p[p_idx] = hf[nb].mean(0) * 0.5 + pf[p_idx] * 0.5
        return new_p, new_h

    def fit(self, pf, hf, train_df, ef):
        p2, h2 = self._propagate(pf, hf, train_df)
        p3, h3 = self._propagate(p2, h2, train_df)  # 2 hops
        pv = p3[train_df["phage_idx"].values]
        hv = h3[train_df["host_idx"].values]
        X  = self.sc.fit_transform(np.hstack([pv, hv, ef]))
        self.clf.fit(X, train_df["label"].values)
        self._p, self._h = p3, h3

    def predict_proba(self, test_df, ef):
        pv = self._p[test_df["phage_idx"].values]
        hv = self._h[test_df["host_idx"].values]
        X  = self.sc.transform(np.hstack([pv, hv, ef]))
        return self.clf.predict_proba(X)[:, 1]


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — LOSO-CV  (Leave-One-Species-Out)
# ═══════════════════════════════════════════════════════════════
print("\n[4] Leave-One-Species-Out CV...")

logo_res  = {"species":[], "n_test":[], "n_pos_test":[],
             "roc_auc":[], "pr_auc":[], "f1":[], "mcc":[]}
all_proba  = np.full(len(dataset), np.nan)
all_labels = dataset["label"].values

for genus in tqdm(valid_species, desc="LOSO folds"):
    test_mask  = (dataset["host"] == genus).values
    train_mask = ~test_mask

    train_df = dataset[train_mask].reset_index(drop=True)
    test_df  = dataset[test_mask].reset_index(drop=True)
    y_te     = test_df["label"].values
    if len(np.unique(y_te)) < 2:
        continue

    train_ef = edge_feats_np[train_mask]
    test_ef  = edge_feats_np[test_mask]

    # Resample train to max 5:1 neg:pos ratio to help with imbalanced genera
    # Keep track of original integer positions within train_ef for correct indexing
    train_df = train_df.reset_index(drop=True)  # 0..len(train_df)-1
    pos_idx = train_df.index[train_df["label"]==1].tolist()
    neg_idx = train_df.index[train_df["label"]==0].tolist()
    n_pos   = len(pos_idx)
    if n_pos > 0 and len(neg_idx) > 5 * n_pos:
        rng_bal = np.random.default_rng(RANDOM_SEED)
        neg_idx = rng_bal.choice(neg_idx, size=5*n_pos, replace=False).tolist()
    bal_idx      = np.array(sorted(pos_idx + neg_idx))
    train_df_bal = train_df.iloc[bal_idx].reset_index(drop=True)
    train_ef_bal = train_ef[bal_idx]

    if HAS_TORCH:
        graph = build_graph(train_mask, phage_feats, host_feats,
                             edge_feats_np, dataset, n_phages)
        pos_w = torch.tensor(
            [(train_mask & (dataset["label"]==0)).sum() /
              max((train_mask & (dataset["label"]==1)).sum(), 1)],
            dtype=torch.float32).clamp(max=10.0)

        model = PhageHostGAT(PHAGE_DIM, HOST_DIM, N_EDGE_FEATS)
        model.set_n_phages(n_phages)
        opt   = torch.optim.AdamW(model.parameters(),
                                    lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=100, T_mult=1, eta_min=1e-5)
        crit  = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        # Scale training length by genus size and known difficulty.
        # Large, hard genera (vibrio, staphylococcus) need more epochs to
        # converge — they have more complex decision boundaries and the
        # model hasn't seen their phages during training.
        # Scale epochs for large/complex species
        n_test_rows = test_mask.sum()
        if n_test_rows >= 300:
            genus_epochs   = 600
            genus_patience = 40
        elif n_test_rows >= 100:
            genus_epochs   = 400
            genus_patience = 30
        else:
            genus_epochs   = EPOCHS
            genus_patience = 20

        best_loss, patience, pat_cnt = float("inf"), genus_patience, 0
        best_state = None

        model.train()
        for epoch in range(genus_epochs):
            opt.zero_grad()
            pi = torch.tensor(train_df_bal["phage_idx"].values, dtype=torch.long)
            hi = torch.tensor(train_df_bal["host_idx"].values,  dtype=torch.long)
            ef = torch.tensor(train_ef_bal, dtype=torch.float32)
            lb = torch.tensor(train_df_bal["label"].values, dtype=torch.float32)

            logits = model(graph.phage_x, graph.host_x,
                           graph.edge_index, graph.edge_attr,
                           pi, hi, ef)
            z = model.get_embeddings(graph.phage_x, graph.host_x,
                                      graph.edge_index, graph.edge_attr)
            loss = crit(logits, lb) + CONTRASTIVE_WEIGHT * info_nce_loss(
                z, pi, hi, lb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                pat_cnt = 0
            else:
                pat_cnt += 1
            if pat_cnt >= patience:
                break

        if best_state:
            model.load_state_dict(best_state)

        proba = predict(model, graph,
                         test_df["phage_idx"].values,
                         test_df["host_idx"].values,
                         test_ef)
    else:
        gnn = NumpyGNN(n_phages, n_hosts)
        gnn.fit(phage_feats, host_feats, train_df, train_ef)
        proba = gnn.predict_proba(test_df, test_ef)

    all_proba[test_mask] = proba
    p2, r2, _ = precision_recall_curve(y_te, proba)
    logo_res["species"].append(genus)
    logo_res["n_test"].append(len(y_te))
    logo_res["n_pos_test"].append(y_te.sum())
    logo_res["roc_auc"].append(roc_auc_score(y_te, proba))
    logo_res["pr_auc"].append(auc(r2, p2))
    pred = (proba >= 0.5).astype(int)
    logo_res["f1"].append(f1_score(y_te, pred, zero_division=0))
    logo_res["mcc"].append(matthews_corrcoef(y_te, pred))

logo_df = pd.DataFrame(logo_res).set_index("species") \
            .sort_values("roc_auc", ascending=False)
logo_df.to_csv(RESULTS_DIR / "gnn_logo_results.csv")

# Save per-row predictions for ensemble (04_ensemble.py)
# ALIGNMENT: dataset is sorted by ["host","phage"] matching ensemble.py
# The saved indices correspond to row positions in that sorted dataset
valid_idx_arr = np.where(~np.isnan(all_proba))[0]
np.save(RESULTS_DIR / "gnn_loso_predictions.npy", all_proba[valid_idx_arr])
np.save(RESULTS_DIR / "gnn_loso_index.npy",       valid_idx_arr)
# Save dataset row keys for verification
dataset[["phage","host"]].to_csv(
    RESULTS_DIR / "gnn_dataset_index.csv", index=True)
print(f"  Saved {len(valid_idx_arr)} GNN predictions for ensemble")
print(f"  Alignment key saved -> gnn_dataset_index.csv")

print("\n=== GNN v2 LOSO-CV RESULTS ===")
print(logo_df.to_string(float_format="{:.4f}".format))
print(f"\n  Mean ROC-AUC: {logo_df['roc_auc'].mean():.4f} "
      f"+/- {logo_df['roc_auc'].std():.4f}")
print(f"  Mean PR-AUC:  {logo_df['pr_auc'].mean():.4f} "
      f"+/- {logo_df['pr_auc'].std():.4f}")
print(f"  Mean MCC:     {logo_df['mcc'].mean():.4f}")

valid_idx = ~np.isnan(all_proba)
y_valid, p_valid = all_labels[valid_idx], all_proba[valid_idx]
p3, r3, _ = precision_recall_curve(y_valid, p_valid)
overall = {
    "roc_auc": roc_auc_score(y_valid, p_valid),
    "pr_auc":  auc(r3, p3),
    "f1":      f1_score(y_valid, (p_valid>=0.5).astype(int), zero_division=0),
    "mcc":     matthews_corrcoef(y_valid, (p_valid>=0.5).astype(int)),
}
print(f"\n  Overall pooled: ROC-AUC={overall['roc_auc']:.4f}  "
      f"PR-AUC={overall['pr_auc']:.4f}  MCC={overall['mcc']:.4f}")
pd.DataFrame([overall], index=["GNN v2"]).to_csv(RESULTS_DIR / "gnn_overall.csv")


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — ATTENTION ANALYSIS
# ═══════════════════════════════════════════════════════════════
attention_results = {}
if HAS_TORCH:
    print("\n[5] Extracting attention weights (full retrain)...")
    full_graph = build_graph(np.ones(len(dataset), dtype=bool),
                              phage_feats, host_feats,
                              edge_feats_np, dataset, n_phages)
    final_model = PhageHostGAT(PHAGE_DIM, HOST_DIM, N_EDGE_FEATS)
    final_model.set_n_phages(n_phages)
    opt2  = torch.optim.AdamW(final_model.parameters(),
                                lr=LR, weight_decay=WEIGHT_DECAY)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt2, T_0=100, eta_min=1e-5)
    pos_w2 = torch.tensor(
        [(dataset["label"]==0).sum() / max((dataset["label"]==1).sum(),1)],
        dtype=torch.float32).clamp(max=10.0)
    crit2 = nn.BCEWithLogitsLoss(pos_weight=pos_w2)

    for _ in tqdm(range(EPOCHS), desc="Final model"):
        opt2.zero_grad()
        pi = torch.tensor(dataset["phage_idx"].values, dtype=torch.long)
        hi = torch.tensor(dataset["host_idx"].values,  dtype=torch.long)
        ef = torch.tensor(edge_feats_np, dtype=torch.float32)
        lb = torch.tensor(dataset["label"].values, dtype=torch.float32)
        z  = final_model.get_embeddings(full_graph.phage_x, full_graph.host_x,
                                         full_graph.edge_index, full_graph.edge_attr)
        logits = final_model.decode(z, pi, hi, ef)
        loss   = crit2(logits, lb) + CONTRASTIVE_WEIGHT * info_nce_loss(z, pi, hi, lb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
        opt2.step(); sched2.step()

    final_model.eval()
    with torch.no_grad():
        x_all = torch.cat([final_model.proj_phage(full_graph.phage_x),
                            final_model.proj_host(full_graph.host_x)], dim=0)
        _, attn_info = final_model.gat1(
            x_all, full_graph.edge_index,
            edge_attr=full_graph.edge_attr,
            return_attention_weights=True)
    attn_idx, attn_w = attn_info
    sa_node = n_phages + host2idx.get(TARGET_HOST, 0)
    to_sa   = (attn_idx[1] == sa_node).numpy()
    if to_sa.sum() > 0:
        src_sa = attn_idx[0][to_sa].numpy()
        w_sa   = attn_w[to_sa].mean(dim=-1).numpy()
        top_k  = min(10, len(src_sa))
        top_i  = np.argsort(w_sa)[-top_k:][::-1]
        print(f"\n  Top phages attending to S. aureus:")
        for rank, i in enumerate(top_i):
            print(f"    {rank+1}. {phage_list[src_sa[i]]}: {w_sa[i]:.4f}")
        attention_results = {
            "phage": [phage_list[src_sa[i]] for i in top_i],
            "attention": [float(w_sa[i]) for i in top_i]
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — COMPARISON WITH v1 AND BASELINE
# ═══════════════════════════════════════════════════════════════
print("\n[6] Comparison with baseline (02_model.py)...")
baseline_path = RESULTS_DIR / "model_comparison.csv"
v1_path       = RESULTS_DIR / "gnn_logo_results.csv"  # may be overwritten now

compare_data = {}
if baseline_path.exists():
    bdf = pd.read_csv(baseline_path, index_col=0)
    for mn in bdf.index:
        compare_data[f"Baseline: {mn}"] = bdf.loc[mn, "roc_auc"]
model_label = "GAT v2 (4L+BioFeats)" if HAS_TORCH else "NumPy GNN v2"
compare_data[f"GNN {model_label} (LOSO mean)"] = logo_df["roc_auc"].mean()
compare_data[f"GNN {model_label} (pooled)"]    = overall["roc_auc"]

print("  ROC-AUC summary:")
for k, v in compare_data.items():
    print(f"    {k}: {v:.4f}")


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — PLOTS
# ═══════════════════════════════════════════════════════════════
print("\n[7] Generating plots...")

sns.set_theme(style="whitegrid", palette="colorblind")

# Plot 11: ROC curve
fig, ax = plt.subplots(figsize=(7, 6))
fpr, tpr, _ = roc_curve(y_valid, p_valid)
ax.plot(fpr, tpr, color="darkorange", lw=2.5,
         label=f"GAT v2 ROC-AUC = {overall['roc_auc']:.3f}")
ax.plot([0,1],[0,1],"r--", label="Random")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("GNN GAT v2 — ROC Curve (LOSO-CV pooled)",
              fontsize=11, fontweight="bold")
ax.legend(loc="lower right"); ax.grid(True, alpha=0.4)
plt.tight_layout()
save_plot(fig, "11_gnn_roc_curve")

# Plot 12: PR curve
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(r3, p3, color="darkorange", lw=2.5,
         label=f"PR-AUC = {overall['pr_auc']:.3f}")
ax.axhline(y_valid.mean(), color="red", linestyle="--",
            label=f"Baseline ({y_valid.mean():.2f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("GNN GAT v2 — Precision-Recall Curve", fontsize=11, fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.4)
plt.tight_layout()
save_plot(fig, "12_gnn_pr_curve")

# Plot 13: Per-genus AUC
fig, ax = plt.subplots(figsize=(10, max(5, len(logo_df)*0.4)))
y_pos  = np.arange(len(logo_df))
colors = ["#1a9850" if v>=0.8 else "#2196F3" if v>=0.7
           else "#FF9800" if v>=0.5 else "#F44336"
           for v in logo_df["roc_auc"].values]
bars = ax.barh(y_pos, logo_df["roc_auc"].values, color=colors, alpha=0.85)
ax.set_yticks(y_pos); ax.set_yticklabels(logo_df.index, fontsize=8)
ax.set_xlim(0, 1.12)
ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="Random")
ax.axvline(logo_df["roc_auc"].mean(), color="black", linestyle=":",
            lw=1.5, label=f"Mean={logo_df['roc_auc'].mean():.3f}")
ax.axvline(0.87, color="gold", linestyle="--", lw=2, label="ISEF target (0.87)")
for bar, v in zip(bars, logo_df["roc_auc"].values):
    ax.text(v+0.01, bar.get_y()+bar.get_height()/2,
             f"{v:.3f}", va="center", fontsize=7)
ax.set_xlabel("ROC-AUC", fontsize=12)
ax.set_title("GNN v2 Per-Genus ROC-AUC (LOSO-CV)\n"
              "Green ≥ 0.8 | Blue ≥ 0.7 | Orange ≥ 0.5 | Red < 0.5",
              fontsize=10, fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="x", alpha=0.4)
plt.tight_layout()
save_plot(fig, "13_gnn_per_genus_auc")

# Plot 14: GNN vs baseline
if compare_data:
    fig, ax = plt.subplots(figsize=(9, max(4, len(compare_data)*0.5)))
    names_c = list(compare_data.keys())
    vals_c  = list(compare_data.values())
    clrs = ["#FF6B35" if "GNN" in n else "#4285F4" for n in names_c]
    bars2 = ax.barh(names_c, vals_c, color=clrs, alpha=0.85)
    ax.set_xlim(0, 1.1)
    ax.axvline(0.5,  color="red",  linestyle="--", alpha=0.5)
    ax.axvline(0.87, color="gold", linestyle="--", lw=2, label="ISEF target")
    for bar, v in zip(bars2, vals_c):
        ax.text(v+0.005, bar.get_y()+bar.get_height()/2,
                 f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("ROC-AUC (LOSO-CV)", fontsize=12)
    ax.set_title("Model Comparison — GNN v2 vs Traditional ML",
                  fontsize=11, fontweight="bold")
    ax.legend(); ax.grid(axis="x", alpha=0.4)
    plt.tight_layout()
    save_plot(fig, "14_gnn_vs_baseline")

# Plot 15: Attention
if attention_results:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(attention_results["phage"][::-1],
             attention_results["attention"][::-1],
             color="#FF6B35", alpha=0.85)
    ax.set_xlabel("Mean Attention Weight")
    ax.set_title(f"GAT Attention: Top Phages → S. aureus\n"
                  "(Higher = model assigns more weight to this phage's embedding "
                  "when predicting S. aureus infections)",
                  fontsize=9, fontweight="bold")
    ax.grid(axis="x", alpha=0.4)
    plt.tight_layout()
    save_plot(fig, "15_gnn_attention_weights")

# Plot 16: Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(
    confusion_matrix(y_valid, (p_valid>=0.5).astype(int)),
    display_labels=["Non-infective","Infective"]
).plot(ax=ax, colorbar=False, cmap="Oranges")
ax.set_title("GNN v2 Confusion Matrix (Pooled LOSO-CV)")
plt.tight_layout()
save_plot(fig, "16_gnn_confusion_matrix")

# Plot 17: Host feature importance (which biological features matter)
if HAS_TORCH and bio_cols and len(bio_cols) > 0:
    # Gradient-based feature importance: perturb each host feature, measure AUC drop
    print("  Computing biological feature importance...")
    feat_importance = {}
    base_auc = overall["roc_auc"]

    final_model.eval()
    for feat_i, feat_name in enumerate(bio_cols):
        # Zero out this feature for all hosts
        host_x_perturbed = full_graph.host_x.clone()
        host_x_perturbed[:, EMBED_DIM + feat_i] = 0.0  # bio feats after SVD

        with torch.no_grad():
            z_p = final_model.encode(full_graph.phage_x, host_x_perturbed,
                                      full_graph.edge_index, full_graph.edge_attr)
            pi  = torch.tensor(dataset["phage_idx"].values, dtype=torch.long)
            hi  = torch.tensor(dataset["host_idx"].values,  dtype=torch.long)
            ef  = torch.tensor(edge_feats_np, dtype=torch.float32)
            pr  = torch.sigmoid(final_model.decode(z_p, pi, hi, ef)).numpy()

        try:
            drop = base_auc - roc_auc_score(all_labels, pr)
        except Exception:
            drop = 0.0
        feat_importance[feat_name] = max(0, drop)

    fi_df = pd.Series(feat_importance).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(3, len(fi_df)*0.4)))
    colors_fi = ["#2196F3" if v > 0.005 else "#BDBDBD" for v in fi_df.values]
    ax.barh(fi_df.index, fi_df.values, color=colors_fi, alpha=0.85)
    ax.set_xlabel("AUC drop when feature zeroed out")
    ax.set_title("Host Biological Feature Importance\n(GNN — permutation-based)",
                  fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.4)
    plt.tight_layout()
    save_plot(fig, "17_gnn_bio_feature_importance")


# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  GNN v2 FINAL SUMMARY")
print("="*60)
model_str = "GAT 4-layer + biological host features + contrastive loss" \
            if HAS_TORCH else "NumPy 2-hop GNN + gradient boosting"
print(f"\n  Model:      {model_str}")
print(f"  CV method:  Leave-One-Species-Out")
print(f"  Folds run:  {len(logo_df)}")
print(f"\n  Pooled overall:")
print(f"    ROC-AUC: {overall['roc_auc']:.4f}")
print(f"    PR-AUC:  {overall['pr_auc']:.4f}")
print(f"    MCC:     {overall['mcc']:.4f}")
print(f"    F1:      {overall['f1']:.4f}")
print(f"\n  Mean per-genus ROC-AUC: {logo_df['roc_auc'].mean():.4f} "
      f"+/- {logo_df['roc_auc'].std():.4f}")
print(f"\n  ISEF target (0.87): "
      f"{'REACHED ✓' if overall['roc_auc'] >= 0.87 else 'Not yet reached'}")
print(f"\n  Results: {RESULTS_DIR.resolve()}")
print(f"  Plots:   {PLOT_DIR.resolve()}")
print("\n  Done!")