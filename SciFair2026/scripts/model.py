"""
03_gnn.py  —  Graph Neural Network for phage-host interaction prediction
=========================================================================
INSTALL (run once, then restart your terminal):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install torch_geometric
    pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.4.0+cpu.html

ARCHITECTURE:
    Bipartite graph: phage nodes <-> host nodes
    Edges:  known interactions from VirusHostInter (positive) + sampled negatives
    Edge features:  k3dist, k6dist, GCdiff, Homology  (4 dims)
    Node features:  learned lookup embeddings (64-dim) initialised from name n-grams
                    via SVD so similar names start similar
    Model:  2-layer Graph Attention Network (GAT) -> link-prediction MLP
            GAT layer 1:  64 -> 64  (8 attention heads x 8 dim)
            GAT layer 2:  64 -> 32
            Link MLP:     (32 + 32 + 4) -> 64 -> 1  (concat phage_emb, host_emb, edge_feats)
    Loss:   Binary cross-entropy with pos-weight balancing

EVALUATION:
    Leave-One-Genus-Out CV — same splits as 02_model.py for direct comparison
    Saves results to data/results/gnn_results.csv
    Saves plots  to data/plots/  (11_gnn_*.png)

WHY A GNN?
    Standard ML (02_model.py) treats each (phage, host) pair independently.
    A GNN lets the model reason about *neighbourhoods*: if phage A infects hosts
    X and Y, and phage B also infects host X, the model can infer B may also
    infect Y.  This is message-passing — it is impossible with flat feature vectors.
"""

import sys
import warnings
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
    roc_auc_score, average_precision_score, f1_score,
    matthews_corrcoef, precision_recall_curve, roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Check PyTorch / PyG ───────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, HeteroData
    from torch_geometric.nn import GATConv, to_hetero
    from torch_geometric.utils import negative_sampling, to_undirected
    HAS_TORCH = True
    print(f"  PyTorch {torch.__version__} + PyG available — running full GNN")
except ImportError as e:
    HAS_TORCH = False
    print(f"  PyTorch/PyG not installed ({e})")
    print()
    print("  To install (run in terminal, then re-run this script):")
    print("    pip install torch --index-url https://download.pytorch.org/whl/cpu")
    print("    pip install torch_geometric")
    print()
    print("  Falling back to message-passing GNN implemented in pure NumPy/scipy.")
    print("  This won't be as powerful but demonstrates the same graph reasoning.")

# ─────────────────────────────────────────────
# CONFIG  (mirrors 02_model.py paths)
# ─────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
BASE_DIR      = _SCRIPT_DIR.parent / "data"
RAW_DIR       = BASE_DIR / "raw"
PLOT_DIR      = BASE_DIR / "plots"
RESULTS_DIR   = BASE_DIR / "results"

for d in [PLOT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED      = 42
TARGET_HOST      = "staphylococcus aureus"
NUMERIC_FEATURES = ["k3dist", "k6dist", "GCdiff", "Homology"]
EMBED_DIM        = 64      # node embedding dimension
GAT_OUT_DIM      = 32      # per-node output of GAT stack
N_HEADS          = 8       # attention heads in GAT layer 1
EPOCHS           = 150
LR               = 1e-3
WEIGHT_DECAY     = 1e-4

np.random.seed(RANDOM_SEED)
if HAS_TORCH:
    torch.manual_seed(RANDOM_SEED)

def normalize(x):
    return str(x).strip().lower().replace("_", " ")

def save_plot(fig, name):
    p = PLOT_DIR / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — LOAD DATA  (same as 02_model.py)
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("  LOADING DATA")
print("=" * 60)

vhi_df = pd.read_csv(RAW_DIR / "VirusHostInter.csv")
vhi_df = vhi_df.loc[:, ~vhi_df.columns.str.contains("^Unnamed")]
vhi_df = vhi_df.rename(columns={"hostname": "host", "phagename": "phage",
                                  "infection": "interaction"})
vhi_df["host"]        = vhi_df["host"].apply(normalize)
vhi_df["phage"]       = vhi_df["phage"].apply(normalize)
vhi_df["interaction"] = vhi_df["interaction"].apply(normalize)
for c in NUMERIC_FEATURES:
    if c not in vhi_df.columns:
        vhi_df[c] = 0.0
    vhi_df[c] = pd.to_numeric(vhi_df[c], errors="coerce").fillna(0.0)

vhi_df["label"] = (vhi_df["interaction"] == "inf").astype(int)

# VHRStaph supplement
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
            vhr_pos["interaction"] = "inf"
            vhr_pos["label"] = 1
            vhi_lookup = vhi_df[vhi_df["host"] == TARGET_HOST].set_index("phage")
            for c in NUMERIC_FEATURES:
                vhr_pos[c] = vhr_pos["phage"].map(
                    vhi_lookup[c].to_dict()).fillna(0.0)
            vhi_df = pd.concat([vhi_df, vhr_pos], ignore_index=True)
    except Exception:
        pass

dataset = vhi_df.drop_duplicates(subset=["phage", "host", "label"]) \
                .reset_index(drop=True)
dataset["genus"] = dataset["host"].str.split().str[0]

print(f"  Dataset: {len(dataset)} rows | "
      f"{dataset['phage'].nunique()} phages | "
      f"{dataset['host'].nunique()} hosts")
print(f"  Positives: {dataset['label'].sum()} | "
      f"Negatives: {(dataset['label']==0).sum()}")


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — BUILD GRAPH
# Nodes: phage nodes + host nodes (bipartite)
# Edges: all (phage, host) pairs with features + label
# ═══════════════════════════════════════════════════════════════
print("\n[2] Building bipartite graph...")

# Index phages and hosts separately
phage_list = sorted(dataset["phage"].unique())
host_list  = sorted(dataset["host"].unique())
phage2idx  = {p: i for i, p in enumerate(phage_list)}
host2idx   = {h: i for i, h in enumerate(host_list)}
n_phages   = len(phage_list)
n_hosts    = len(host_list)

dataset["phage_idx"] = dataset["phage"].map(phage2idx)
dataset["host_idx"]  = dataset["host"].map(host2idx)

print(f"  {n_phages} phage nodes + {n_hosts} host nodes = {n_phages + n_hosts} total")
print(f"  {len(dataset)} edges ({dataset['label'].sum()} positive, "
      f"{(dataset['label']==0).sum()} negative)")

# ── Node features: SVD of name n-gram matrix ─────────────────
# Build a name n-gram bag-of-chars for every phage and host,
# then compress to EMBED_DIM via SVD to get dense starting embeddings.
# This seeds the GNN with "similar names -> similar vectors" before any training.
print("  Building name-based node features via SVD...")

all_names = phage_list + host_list
vec_init  = CountVectorizer(analyzer="char", ngram_range=(3, 5),
                             max_features=5000, dtype=np.float32)
X_names   = vec_init.fit_transform(all_names).toarray().astype(np.float32)

k = min(EMBED_DIM, X_names.shape[1] - 1, X_names.shape[0] - 1)
U, s, Vt  = svds(X_names, k=k)
node_feats = (U * s).astype(np.float32)  # shape: (n_phages + n_hosts, k)
# Pad to EMBED_DIM if needed
if node_feats.shape[1] < EMBED_DIM:
    pad = np.zeros((node_feats.shape[0], EMBED_DIM - node_feats.shape[1]),
                   dtype=np.float32)
    node_feats = np.hstack([node_feats, pad])

phage_feats = node_feats[:n_phages]   # (n_phages, EMBED_DIM)
host_feats  = node_feats[n_phages:]   # (n_hosts,  EMBED_DIM)

# ── Edge features ─────────────────────────────────────────────
scaler = StandardScaler()
edge_feats_np = scaler.fit_transform(
    dataset[NUMERIC_FEATURES].values.astype(np.float32))  # (n_edges, 4)

print(f"  Node feature dim: {EMBED_DIM}")
print(f"  Edge feature dim: {len(NUMERIC_FEATURES)}")


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — LOGO-CV SPLITS  (same genus grouping as 02_model.py)
# ═══════════════════════════════════════════════════════════════
valid_genera = [
    g for g in dataset["genus"].unique()
    if dataset[dataset["genus"] == g]["label"].nunique() == 2
]
print(f"\n[3] LOGO-CV: {len(valid_genera)} evaluable genera")


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — GNN MODEL DEFINITION
# ═══════════════════════════════════════════════════════════════

if HAS_TORCH:
    class PhageHostGAT(nn.Module):
        """
        Two-layer Graph Attention Network for bipartite phage-host graphs.

        Message passing:
          Layer 1: each node aggregates from its neighbours with learned
                   attention weights (8 heads).  Phages attend over hosts
                   they interact with; hosts attend over their phages.
          Layer 2: refine embeddings (single-head, output GAT_OUT_DIM).

        Link prediction:
          For a candidate (phage, host) pair:
            score = MLP( concat(phage_emb, host_emb, edge_feats) )
          The edge features (k3dist, k6dist, GCdiff, Homology) are injected
          here so biological similarity directly informs the prediction.
        """
        def __init__(self, in_dim, edge_dim, hidden=64, out_dim=GAT_OUT_DIM):
            super().__init__()
            head_dim = hidden // N_HEADS

            # GAT layer 1: in_dim -> hidden (8 heads, concat)
            self.gat1 = GATConv(in_dim, head_dim, heads=N_HEADS,
                                  concat=True, dropout=0.3,
                                  edge_dim=edge_dim, add_self_loops=False)
            # GAT layer 2: hidden -> out_dim (1 head, mean)
            self.gat2 = GATConv(hidden, out_dim, heads=1,
                                  concat=False, dropout=0.2,
                                  edge_dim=edge_dim, add_self_loops=False)

            # Link prediction MLP
            mlp_in = out_dim + out_dim + edge_dim
            self.mlp = nn.Sequential(
                nn.Linear(mlp_in, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
            self.bn1 = nn.BatchNorm1d(hidden)

        def encode(self, x, edge_index, edge_attr):
            """Run GAT message passing to get node embeddings."""
            # Convert bipartite to homogeneous (treat all nodes same type)
            h = self.gat1(x, edge_index, edge_attr=edge_attr)
            h = self.bn1(h)
            h = F.elu(h)
            h = self.gat2(h, edge_index, edge_attr=edge_attr)
            return h   # (n_nodes, out_dim)

        def decode(self, z, phage_idx, host_idx, edge_feats):
            """Score candidate edges given node embeddings."""
            # Offset host indices to global node index space
            z_phage = z[phage_idx]
            z_host  = z[host_idx + self._n_phages]
            cat     = torch.cat([z_phage, z_host, edge_feats], dim=-1)
            return self.mlp(cat).squeeze(-1)

        def forward(self, data, phage_idx, host_idx, edge_feats):
            z = self.encode(data.x, data.edge_index, data.edge_attr)
            return self.decode(z, phage_idx, host_idx, edge_feats)

        def set_n_phages(self, n):
            self._n_phages = n


    def build_pyg_graph(train_mask, all_phage_feats, all_host_feats,
                         all_edge_feats, dataset_df, n_phages):
        """
        Build a PyG Data object from training edges only.
        Node features = concat of phage features and host features.
        Edge index uses global node IDs (hosts offset by n_phages).
        """
        train_df = dataset_df[train_mask].reset_index(drop=True)

        # Only positive edges for message passing (we don't want the model
        # to propagate "non-infection" signals during neighbourhood aggregation)
        pos_mask = train_df["label"] == 1
        src = torch.tensor(train_df.loc[pos_mask, "phage_idx"].values, dtype=torch.long)
        dst = torch.tensor(train_df.loc[pos_mask, "host_idx"].values + n_phages,
                            dtype=torch.long)

        edge_index = torch.stack([src, dst], dim=0)
        # Make undirected: host -> phage edges too
        edge_index = to_undirected(edge_index)

        # Edge features for the undirected edges
        ef     = torch.tensor(all_edge_feats[train_mask][pos_mask.values],
                               dtype=torch.float32)
        edge_attr = torch.cat([ef, ef], dim=0)  # duplicate for undirected

        # Node features: stack phage + host
        x = torch.cat([
            torch.tensor(all_phage_feats, dtype=torch.float32),
            torch.tensor(all_host_feats,  dtype=torch.float32),
        ], dim=0)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


    def train_epoch(model, optimizer, criterion, data, train_df,
                     edge_feats, pos_weight):
        model.train()
        optimizer.zero_grad()
        phage_t = torch.tensor(train_df["phage_idx"].values, dtype=torch.long)
        host_t  = torch.tensor(train_df["host_idx"].values,  dtype=torch.long)
        ef_t    = torch.tensor(edge_feats, dtype=torch.float32)
        labels  = torch.tensor(train_df["label"].values, dtype=torch.float32)
        logits  = model(data, phage_t, host_t, ef_t)
        loss    = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def predict(model, data, phage_idx, host_idx, edge_feats):
        model.eval()
        phage_t = torch.tensor(phage_idx, dtype=torch.long)
        host_t  = torch.tensor(host_idx,  dtype=torch.long)
        ef_t    = torch.tensor(edge_feats, dtype=torch.float32)
        logits  = model(data, phage_t, host_t, ef_t)
        return torch.sigmoid(logits).numpy()


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — NUMPY FALLBACK GNN
# Simple 1-layer message passing without PyTorch
# ═══════════════════════════════════════════════════════════════

class NumpyGNN:
    """
    1-step message passing GNN in pure NumPy.
    For each node, aggregates (mean) the feature vectors of its neighbours,
    then concatenates [own_features, aggregated_neighbour_features, edge_stats]
    and passes to a logistic regression classifier.

    This captures the core GNN intuition: a phage's prediction is informed by
    all the hosts it's known to infect, and a host's prediction is informed by
    all phages known to infect it.
    """
    def __init__(self, embed_dim=EMBED_DIM, n_phages=0, n_hosts=0):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler as SS
        self.embed_dim = embed_dim
        self.n_phages  = n_phages
        self.n_hosts   = n_hosts
        self.clf       = LogisticRegression(max_iter=1000, class_weight="balanced",
                                             solver="saga", random_state=RANDOM_SEED)
        self.scaler    = SS()

    def _message_pass(self, phage_feats, host_feats, train_df, train_pos_mask):
        """
        One round of mean aggregation over positive training edges only.
        Returns updated phage and host embeddings.
        """
        pos_train = train_df[train_pos_mask]

        # Phage -> host messages: for each host, mean of its infecting phages
        new_host = host_feats.copy()
        for h_idx in range(self.n_hosts):
            nbrs = pos_train.loc[pos_train["host_idx"] == h_idx, "phage_idx"].values
            if len(nbrs):
                new_host[h_idx] = phage_feats[nbrs].mean(axis=0)

        # Host -> phage messages: for each phage, mean of its infected hosts
        new_phage = phage_feats.copy()
        for p_idx in range(self.n_phages):
            nbrs = pos_train.loc[pos_train["phage_idx"] == p_idx, "host_idx"].values
            if len(nbrs):
                new_phage[p_idx] = host_feats[nbrs].mean(axis=0)

        # Residual connection
        new_phage = 0.5 * phage_feats + 0.5 * new_phage
        new_host  = 0.5 * host_feats  + 0.5 * new_host
        return new_phage, new_host

    def fit(self, phage_feats, host_feats, train_df, edge_feats):
        pos_mask = train_df["label"] == 1
        p_emb, h_emb = self._message_pass(phage_feats, host_feats, train_df, pos_mask)

        # Build feature vector: [phage_emb | host_emb | edge_feats]
        p_vecs = p_emb[train_df["phage_idx"].values]
        h_vecs = h_emb[train_df["host_idx"].values]
        X = np.hstack([p_vecs, h_vecs, edge_feats])
        y = train_df["label"].values

        X = self.scaler.fit_transform(X)
        self.clf.fit(X, y)
        self._p_emb = p_emb
        self._h_emb = h_emb

    def predict_proba(self, test_df, edge_feats):
        p_vecs = self._p_emb[test_df["phage_idx"].values]
        h_vecs = self._h_emb[test_df["host_idx"].values]
        X = np.hstack([p_vecs, h_vecs, edge_feats])
        X = self.scaler.transform(X)
        return self.clf.predict_proba(X)[:, 1]


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — LOGO-CV EVALUATION
# ═══════════════════════════════════════════════════════════════
print("\n[4] Running Leave-One-Genus-Out CV...")

logo_results = {
    "genus": [], "n_test": [], "n_pos_test": [],
    "roc_auc": [], "pr_auc": [], "f1": [], "mcc": []
}

all_proba = np.full(len(dataset), np.nan)
all_labels = dataset["label"].values

for genus in tqdm(valid_genera, desc="LOGO folds"):
    test_mask  = (dataset["genus"] == genus).values
    train_mask = ~test_mask

    train_df = dataset[train_mask].reset_index(drop=True)
    test_df  = dataset[test_mask].reset_index(drop=True)

    y_te = test_df["label"].values
    if len(np.unique(y_te)) < 2:
        continue

    train_ef = edge_feats_np[train_mask]
    test_ef  = edge_feats_np[test_mask]

    if HAS_TORCH:
        # ── PyTorch GAT path ─────────────────────────────────
        graph = build_pyg_graph(train_mask, phage_feats, host_feats,
                                  edge_feats_np, dataset, n_phages)
        pos_w = torch.tensor(
            [(train_mask & (dataset["label"] == 0)).sum() /
              max((train_mask & (dataset["label"] == 1)).sum(), 1)],
            dtype=torch.float32)

        model = PhageHostGAT(in_dim=EMBED_DIM,
                              edge_dim=len(NUMERIC_FEATURES))
        model.set_n_phages(n_phages)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                      weight_decay=WEIGHT_DECAY)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        for epoch in range(EPOCHS):
            train_epoch(model, optimizer, criterion, graph,
                         train_df, train_ef, pos_w)

        proba = predict(model, graph,
                         test_df["phage_idx"].values,
                         test_df["host_idx"].values,
                         test_ef)
    else:
        # ── NumPy fallback path ──────────────────────────────
        gnn = NumpyGNN(embed_dim=EMBED_DIM, n_phages=n_phages, n_hosts=n_hosts)
        gnn.fit(phage_feats, host_feats, train_df, train_ef)
        proba = gnn.predict_proba(test_df, test_ef)

    all_proba[test_mask] = proba

    p2, r2, _ = precision_recall_curve(y_te, proba)
    logo_results["genus"].append(genus)
    logo_results["n_test"].append(len(y_te))
    logo_results["n_pos_test"].append(y_te.sum())
    logo_results["roc_auc"].append(roc_auc_score(y_te, proba))
    logo_results["pr_auc"].append(auc(r2, p2))
    pred = (proba >= 0.5).astype(int)
    logo_results["f1"].append(f1_score(y_te, pred))
    logo_results["mcc"].append(matthews_corrcoef(y_te, pred))

logo_df = pd.DataFrame(logo_results).set_index("genus") \
            .sort_values("roc_auc", ascending=False)

print("\n=== GNN LOGO-CV RESULTS ===")
print(logo_df.to_string(float_format="{:.4f}".format))
print(f"\n  Mean ROC-AUC: {logo_df['roc_auc'].mean():.4f} "
      f"+/- {logo_df['roc_auc'].std():.4f}")
print(f"  Mean PR-AUC:  {logo_df['pr_auc'].mean():.4f} "
      f"+/- {logo_df['pr_auc'].std():.4f}")
print(f"  Mean MCC:     {logo_df['mcc'].mean():.4f}")

logo_df.to_csv(RESULTS_DIR / "gnn_logo_results.csv")

# Overall metrics (pool all held-out predictions)
valid_idx = ~np.isnan(all_proba)
y_valid   = all_labels[valid_idx]
p_valid   = all_proba[valid_idx]
p3, r3, _ = precision_recall_curve(y_valid, p_valid)
overall = {
    "roc_auc": roc_auc_score(y_valid, p_valid),
    "pr_auc":  auc(r3, p3),
    "f1":      f1_score(y_valid, (p_valid >= 0.5).astype(int)),
    "mcc":     matthews_corrcoef(y_valid, (p_valid >= 0.5).astype(int)),
}
print(f"\n  Overall (pooled folds):")
for k, v in overall.items():
    print(f"    {k}: {v:.4f}")

pd.DataFrame([overall], index=["GNN overall"]).to_csv(
    RESULTS_DIR / "gnn_overall.csv")


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — ATTENTION ANALYSIS (PyTorch only)
# Show which host neighbours each phage attends to most
# ═══════════════════════════════════════════════════════════════
attention_results = {}
if HAS_TORCH:
    print("\n[5] Extracting attention weights for S. aureus...")
    # Retrain on all data to get final attention weights
    all_pos_mask = dataset["label"] == 1
    full_graph   = build_pyg_graph(np.ones(len(dataset), dtype=bool),
                                    phage_feats, host_feats,
                                    edge_feats_np, dataset, n_phages)
    final_model  = PhageHostGAT(in_dim=EMBED_DIM, edge_dim=len(NUMERIC_FEATURES))
    final_model.set_n_phages(n_phages)
    opt2 = torch.optim.Adam(final_model.parameters(), lr=LR,
                             weight_decay=WEIGHT_DECAY)
    pos_w2 = torch.tensor([(dataset["label"]==0).sum() /
                             max((dataset["label"]==1).sum(), 1)],
                            dtype=torch.float32)
    crit2 = nn.BCEWithLogitsLoss(pos_weight=pos_w2)
    for _ in tqdm(range(EPOCHS), desc="Final model training"):
        train_epoch(final_model, opt2, crit2, full_graph,
                     dataset, edge_feats_np, pos_w2)

    # Extract GAT attention weights for S. aureus host node
    final_model.eval()
    with torch.no_grad():
        _, attn1 = final_model.gat1(
            torch.tensor(np.vstack([phage_feats, host_feats]), dtype=torch.float32),
            full_graph.edge_index,
            edge_attr=full_graph.edge_attr,
            return_attention_weights=True
        )
    attn_idx, attn_weights = attn1
    sa_node = n_phages + host2idx.get(TARGET_HOST, 0)
    # Find edges pointing TO S. aureus
    to_sa   = (attn_idx[1] == sa_node).numpy()
    src_sa  = attn_idx[0][to_sa].numpy()
    w_sa    = attn_weights[to_sa].mean(dim=-1).numpy()

    top_k = min(10, len(src_sa))
    top_idx = np.argsort(w_sa)[-top_k:][::-1]
    print(f"\n  Top {top_k} phages attending to S. aureus (highest attention):")
    for rank, i in enumerate(top_idx):
        print(f"    {rank+1}. {phage_list[src_sa[i]]}: attention={w_sa[i]:.4f}")

    attention_results = {
        "phage": [phage_list[src_sa[i]] for i in top_idx],
        "attention": [w_sa[i] for i in top_idx]
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — COMPARISON WITH 02_model.py RESULTS
# ═══════════════════════════════════════════════════════════════
print("\n[6] Comparison with baseline model (02_model.py)...")
baseline_path = RESULTS_DIR / "model_comparison.csv"
comparison_available = False
if baseline_path.exists():
    baseline_df = pd.read_csv(baseline_path, index_col=0)
    print("\n  Baseline (02_model.py) vs GNN — ROC-AUC:")
    for model_name in baseline_df.index:
        b_auc = baseline_df.loc[model_name, "roc_auc"]
        print(f"    {model_name}: {b_auc:.4f}")
    print(f"    GNN (LOGO-CV mean): {logo_df['roc_auc'].mean():.4f}")
    comparison_available = True
else:
    print("  (Run 02_model.py first to enable comparison)")


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — PLOTS
# ═══════════════════════════════════════════════════════════════
print("\n[7] Generating plots...")

# Plot 11: ROC curve (pooled LOGO predictions)
fig, ax = plt.subplots(figsize=(7, 6))
fpr, tpr, _ = roc_curve(y_valid, p_valid)
ax.plot(fpr, tpr, color="darkorange", lw=2.5,
         label=f"GNN ROC-AUC = {overall['roc_auc']:.3f}")
ax.plot([0,1],[0,1],"r--", label="Random")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
model_label = "GAT" if HAS_TORCH else "Message-Passing GNN (NumPy)"
ax.set_title(f"GNN ({model_label}) — ROC Curve\n(Leave-One-Genus-Out CV, pooled)",
              fontsize=11, fontweight="bold")
ax.legend(loc="lower right"); ax.grid(True, alpha=0.4)
plt.tight_layout()
save_plot(fig, "11_gnn_roc_curve")

# Plot 12: PR curve (pooled)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(r3, p3, color="darkorange", lw=2.5,
         label=f"GNN PR-AUC = {overall['pr_auc']:.3f}")
ax.axhline(y_valid.mean(), color="red", linestyle="--",
            label=f"Baseline ({y_valid.mean():.2f})")
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title(f"GNN ({model_label}) — Precision-Recall Curve",
              fontsize=11, fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.4)
plt.tight_layout()
save_plot(fig, "12_gnn_pr_curve")

# Plot 13: Per-genus ROC-AUC bar chart
fig, ax = plt.subplots(figsize=(10, max(4, len(logo_df) * 0.45)))
y_pos = np.arange(len(logo_df))
colors = ["#2196F3" if v >= 0.7 else "#FF9800" if v >= 0.5 else "#F44336"
          for v in logo_df["roc_auc"].values]
bars = ax.barh(y_pos, logo_df["roc_auc"].values, color=colors, alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(logo_df.index, fontsize=9)
ax.set_xlim(0, 1.1)
ax.axvline(0.5, color="red", linestyle="--", alpha=0.6, label="Random")
ax.axvline(logo_df["roc_auc"].mean(), color="black", linestyle=":",
            lw=1.5, label=f"Mean = {logo_df['roc_auc'].mean():.3f}")
for bar, v in zip(bars, logo_df["roc_auc"].values):
    ax.text(v + 0.01, bar.get_y() + bar.get_height()/2,
             f"{v:.3f}", va="center", fontsize=8)
ax.set_xlabel("ROC-AUC", fontsize=12)
ax.set_title(f"GNN Per-Genus ROC-AUC (Leave-One-Genus-Out CV)\n"
              f"Blue ≥ 0.7, Orange ≥ 0.5, Red < 0.5",
              fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="x", alpha=0.4)
plt.tight_layout()
save_plot(fig, "13_gnn_per_genus_auc")

# Plot 14: GNN vs baseline comparison (if available)
if comparison_available:
    fig, ax = plt.subplots(figsize=(9, 5))
    compare_data = {}
    for model_name in baseline_df.index:
        compare_data[model_name] = baseline_df.loc[model_name, "roc_auc"]
    compare_data[f"GNN ({model_label})"] = logo_df["roc_auc"].mean()
    names_c = list(compare_data.keys())
    vals_c  = list(compare_data.values())
    gnn_color = ["#FF6B35" if "GNN" in n else "#4285F4" for n in names_c]
    bars = ax.barh(names_c, vals_c, color=gnn_color, alpha=0.85)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("ROC-AUC (LOGO-CV)", fontsize=12)
    ax.set_title("Model Comparison — GNN vs Traditional ML\n"
                  "(Leave-One-Genus-Out CV)", fontsize=12, fontweight="bold")
    for bar, v in zip(bars, vals_c):
        ax.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{v:.3f}", va="center", fontsize=9)
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.5)
    ax.grid(axis="x", alpha=0.4)
    plt.tight_layout()
    save_plot(fig, "14_gnn_vs_baseline")

# Plot 15: Attention weights (if available)
if attention_results:
    fig, ax = plt.subplots(figsize=(8, 5))
    phages_a = attention_results["phage"]
    attn_a   = attention_results["attention"]
    ax.barh(phages_a[::-1], attn_a[::-1], color="#FF6B35", alpha=0.85)
    ax.set_xlabel("Mean Attention Weight", fontsize=12)
    ax.set_title(f"GAT Attention: Top Phages Attending to {TARGET_HOST.title()}\n"
                  "(Higher = model weights this phage more when predicting S. aureus infections)",
                  fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.4)
    plt.tight_layout()
    save_plot(fig, "15_gnn_attention_weights")

# Plot 16: Confusion matrix (pooled predictions)
fig, ax = plt.subplots(figsize=(5, 4))
y_pred_pool = (p_valid >= 0.5).astype(int)
ConfusionMatrixDisplay(
    confusion_matrix(y_valid, y_pred_pool),
    display_labels=["Non-infective", "Infective"]
).plot(ax=ax, colorbar=False, cmap="Oranges")
ax.set_title(f"GNN Confusion Matrix (Pooled LOGO-CV folds)")
plt.tight_layout()
save_plot(fig, "16_gnn_confusion_matrix")


# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  GNN FINAL SUMMARY")
print("="*60)
print(f"\n  Model:      {'GAT (2-layer, 8-head)' if HAS_TORCH else 'Message-Passing GNN (NumPy fallback)'}")
print(f"  CV method:  Leave-One-Genus-Out")
print(f"  Folds run:  {len(logo_df)}")
print(f"\n  Overall (pooled folds):")
print(f"    ROC-AUC: {overall['roc_auc']:.4f}")
print(f"    PR-AUC:  {overall['pr_auc']:.4f}")
print(f"    MCC:     {overall['mcc']:.4f}")
print(f"    F1:      {overall['f1']:.4f}")
print(f"\n  Mean per-genus ROC-AUC: {logo_df['roc_auc'].mean():.4f} "
      f"+/- {logo_df['roc_auc'].std():.4f}")
print(f"\n  Results: {RESULTS_DIR.resolve()}")
print(f"  Plots:   {PLOT_DIR.resolve()}")
print("\n  Done!")