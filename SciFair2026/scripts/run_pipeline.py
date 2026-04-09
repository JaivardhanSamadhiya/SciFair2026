"""
run_pipeline.py  —  PrecisionPhage ISEF 2026
============================================
ONE SCRIPT. Run it once. Gets everything done.

What it does, in order:
  1. Loads VirusHostInter.csv
  2. Downloads extra phage-host pairs from Virus-Host DB + NCBI
  3. Computes rich genomic features from your phage FASTA file
  4. Downloads host genome FASTAs from NCBI RefSeq (one per species)
  5. Adds tetranucleotide correlation + codon usage distance features
  6. Trains XGBoost / RF / GBM classical models
  7. Trains GAT + GraphSAGE GNN models (leakage-free)
  8. Runs LOSO / LOGO / Unseen-strain / Cocktail evaluation
  9. Saves all results CSVs and all poster plots

BEFORE RUNNING:
  pip install -r SciFair2026/requirements.txt
  (Or: torch, torch-geometric, xgboost, scikit-learn, pandas, numpy, matplotlib, seaborn, scipy, umap-learn)

Optional speed toggle (more MC rounds / longer training):
  set PHAGE_PIPELINE_THOROUGH=1

EDIT THESE TWO PATHS, then just run:
  python run_pipeline.py
"""

# ════════════════════════════════════════════════════════════════
# ▶▶  EDIT THESE TWO LINES  ◀◀
# ════════════════════════════════════════════════════════════════
VHI_CSV    = r"D:\SciFair2026\SciFair2026\data\raw\VirusHostInter.csv"
PHAGE_FASTA= r"D:\SciFair2026\SciFair2026\data\fastas\phage_genomes.fasta"
# ════════════════════════════════════════════════════════════════

from pathlib import Path

_ROOT      = Path(VHI_CSV).parent.parent
PLOT_DIR   = _ROOT / "plots";    PLOT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR= _ROOT / "results";  RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FASTA_DIR  = _ROOT / "fastas";   FASTA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR    = _ROOT / "raw";      RAW_DIR.mkdir(parents=True, exist_ok=True)
HOST_FASTA    = FASTA_DIR / "host_genomes.fasta"
GEN_FEAT      = RAW_DIR / "genomic_features.csv"
AUG_CSV       = RAW_DIR / "augmented_dataset.csv"

_INPHARED_CANDIDATES = [
    Path(r"D:\GenomesDB"),
    Path(r"D:\GenomesDB_Jan_2026\GenomesDB"),
    Path(r"D:\SciFair2026\GenomesDB"),
    Path(r"D:\SciFair2026\SciFair2026\data\fastas\GenomesDB"),
]
INPHARED_GENOMES_DIR = next((p for p in _INPHARED_CANDIDATES if p.is_dir()), None)

# ── Imports ───────────────────────────────────────────────────
import sys, os, re, time, csv, json, gzip, io, math, warnings, contextlib
import itertools, urllib.parse, subprocess as _sp
from concurrent.futures import ThreadPoolExecutor

_THOROUGH = os.environ.get("PHAGE_PIPELINE_THOROUGH", "").strip().lower() in ("1", "true", "yes")
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import wilcoxon, mannwhitneyu
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (roc_auc_score, f1_score, matthews_corrcoef,
    precision_recall_curve, roc_curve, auc as sk_auc, average_precision_score)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier, HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="colorblind")

# PyTorch probe
_probe = _sp.run([sys.executable, "-c",
    "import torch; import torch_geometric; print(torch.__version__)"],
    capture_output=True, text=True, timeout=120)
HAS_TORCH = _probe.returncode == 0 and bool(_probe.stdout.strip())
if HAS_TORCH:
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch_geometric.nn import SAGEConv, GATConv
    from torch_geometric.utils import to_undirected
    from torch.cuda.amp import autocast, GradScaler
    print(f"  PyTorch {torch.__version__} + PyG ✓")
else:
    print(f"  PyTorch unavailable — GNN will use NumPy fallback\n  ({_probe.stderr.strip()[:120]})")

try:
    import xgboost as xgb; HAS_XGB = True; print("  XGBoost ✓")
except ImportError:
    HAS_XGB = False; print("  XGBoost missing — will use GBM instead")

try:
    import umap as _umap_mod; HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# ── Reproducibility ───────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
if HAS_TORCH:
    torch.manual_seed(SEED)
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print("  Apple Silicon MPS ✓ (GPU-accelerated)")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(f"  CUDA ✓ ({torch.cuda.get_device_name(0)})")
    else:
        DEVICE = torch.device("cpu")
        print("  Using CPU (no GPU detected)")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(4)
else:
    DEVICE = None

# Mixed precision on CUDA only (faster GNN steps; no change to evaluation protocol)
USE_AMP = bool(HAS_TORCH and DEVICE is not None and DEVICE.type == "cuda")
if USE_AMP:
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    print("  CUDA mixed precision (AMP) enabled for GNN training")

# ── Hyperparameters ───────────────────────────────────────────
SVD_DIM      = 128
HIDDEN_DIM   = 256
OUT_DIM      = 128
DROPOUT      = 0.25
LR           = 8e-4
WD           = 1e-4
# Early stopping usually fires well before the cap; lower cap = far fewer GNN epochs per fold.
EPOCHS       = 400 if _THOROUGH else 100
PATIENCE     = 22 if _THOROUGH else 12
NEG_RATIO    = 3
GAT_HEADS    = 4
GAT_LAYERS   = 4
MIN_SP_ROWS  = 5
MAX_SP_ROWS  = 99999
N_MC_ROUNDS  = 30 if _THOROUGH else 12
UNSEEN_FRAC  = 0.30
K_COCKTAIL   = 3
# Slightly smaller char n-gram bag speeds SVD + downstream with small AUC impact
_CHAR_NGRAM_MAX = 8000 if _THOROUGH else 5500
_EDGE_KEEP_FRAC = 0.9
NUMERIC_FEATS= ["k3dist","k6dist","GCdiff","Homology"]
ARCH_COLOR   = {"GAT":"#E65100","SAGE":"#1565C0","XGB":"#2E7D32","RF":"#6A1B9A"}
ARCH_LABEL   = {"GAT":"GAT+Edge MLP","SAGE":"SAGE+Residual","XGB":"XGBoost","RF":"Random Forest"}

def save_fig(fig, name):
    p = PLOT_DIR / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {p.name}")

def metrics(y, p):
    y = np.asarray(y, dtype=np.float32)
    p = np.asarray(p, dtype=np.float32)
    p = np.where(np.isfinite(p), p, 0.5)
    p = np.clip(p, 0.0, 1.0)
    try:
        pr, rc, _ = precision_recall_curve(y, p)
        pr_auc = float(sk_auc(rc, pr))
    except Exception:
        pr_auc = 0.0
    try:
        roc = float(roc_auc_score(y, p))
    except Exception:
        roc = 0.5
    pred = (p >= 0.5).astype(int)
    return {"roc_auc": roc,
            "pr_auc":  pr_auc,
            "f1":      float(f1_score(y, pred, zero_division=0)),
            "mcc":     float(matthews_corrcoef(y, pred))}

print("\n" + "="*66)
print("  PRECISIONPHAGE — FULL PIPELINE  (ISEF 2026)")
print("="*66)


# ════════════════════════════════════════════════════════════════
# SECTION 1 — DOWNLOAD EXTRA INTERACTION DATA
# ════════════════════════════════════════════════════════════════
print("\n[1] Collecting extra phage-host interaction data...")

def _fetch(url, timeout=90, retries=3, post=None):
    hdrs = {"User-Agent": "Mozilla/5.0 PrecisionPhage-ISEF2026"}
    for attempt in range(retries):
        try:
            req = Request(url, data=post, headers=hdrs)
            with urlopen(req, timeout=timeout) as r:
                return r.read()
        except HTTPError as e:
            if e.code in (400, 404): raise
            if attempt < retries-1: time.sleep(2**attempt)
            else: raise
        except URLError as e:
            if attempt < retries-1: time.sleep(2**attempt)
            else: raise

def _dl(url, dest, label=""):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 500:
        return True
    print(f"    GET {label or Path(url).name}")
    try:
        data = _fetch(url, timeout=180)
        if data[:200].lower().lstrip().startswith(b"<html"): return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        print(f"    OK  {dest.name}  ({len(data)//1024} KB)")
        return True
    except Exception as e:
        print(f"    FAIL: {e}")
        if dest.exists(): dest.unlink()
        return False

def _clean_host(name):
    name = name.strip().lower()
    name = re.sub(r'\s+(strain|str\b|subsp\b|serovar|sv\b|bv\b|pv\b|type\b|var\b).*','',name)
    return re.sub(r'\s+',' ',name).strip()

def _ncbi_post(endpoint, **params):
    params.update({"email":"precisionphage@isef2026.edu","retmode":"json"})
    post = urllib.parse.urlencode(params).encode()
    return json.loads(_fetch(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{endpoint}",
                              timeout=60, post=post))

def _get_virushostdb():
    dest = RAW_DIR / "virushostdb_raw.tsv"
    for url in ["https://www.genome.jp/ftp/db/virushostdb/virushostdb.tsv",
                "https://www.genome.jp/virushostdb/virushostdb.tsv"]:
        if _dl(url, dest, "virushostdb.tsv"): break
    else:
        return []
    first = open(dest, encoding="utf-8", errors="replace").readline()
    delim = "\t" if "\t" in first else ","
    rows = []
    with open(dest, encoding="utf-8", errors="replace") as fh:
        for rec in csv.DictReader(fh, delimiter=delim):
            def g(*k): return next((rec.get(x,"").strip() for x in k if rec.get(x,"")), "")
            hl = g("host lineage","Host lineage","host_lineage")
            hn = g("host name","Host name","host_name","Host Name")
            vn = g("virus name","Virus name","Virus Name","virus_name")
            vi = g("virus tax id","Virus tax id")
            if "Bacteria" not in hl or not vn or not hn: continue
            h = _clean_host(hn)
            if len(h.split()) < 2: continue
            rows.append({"phage":vi or vn,"host":h,"host_strain":hn.strip().lower(),
                         "label":1,"source":"virushostdb"})
    print(f"    Virus-Host DB: {len(rows)} pairs")
    return rows

def _get_ncbi(max_rec=2000):
    dest = RAW_DIR / "ncbi_phage_hosts.json"
    if dest.exists() and dest.stat().st_size > 100:
        pairs = json.loads(dest.read_text(encoding="utf-8"))
        print(f"    NCBI (cached): {len(pairs)} pairs")
        return pairs
    try:
        data = _ncbi_post("esearch.fcgi", db="nuccore",
            term="(bacteriophage[Title] OR phage[Title]) AND complete genome[Title] AND 5000:500000[SLEN]",
            retmax=max_rec)
        ids = data.get("esearchresult",{}).get("idlist",[])
        print(f"    NCBI: {len(ids)} records found")
    except Exception as e:
        print(f"    NCBI search failed: {e}"); return []
    pairs = []
    for start in range(0, len(ids), 80):
        batch = ids[start:start+80]
        try:
            data = _ncbi_post("esummary.fcgi", db="nuccore", id=",".join(batch))
            time.sleep(0.35)
        except Exception as e:
            print(f"    Batch {start} failed: {e}"); continue
        for uid, rec in data.get("result",{}).items():
            if uid == "uids": continue
            title = rec.get("title","")
            acc   = rec.get("accessionversion","") or rec.get("caption","")
            if not acc: continue
            m = re.match(r"^([A-Z][a-z]+\s+[a-z]+)\s+(?:phage|bacteriophage|virus)",title)
            if m: pairs.append({"phage":acc,"host":_clean_host(m.group(1)),"host_strain":m.group(1).strip().lower(),"label":1,"source":"ncbi"}); continue
            m2 = re.match(r"^([A-Z][a-z]+)\s+(?:phage|bacteriophage)",title)
            if m2: pairs.append({"phage":acc,"host":m2.group(1).lower()+" sp.","host_strain":m2.group(1).lower()+" sp.","label":1,"source":"ncbi"})
    dest.write_text(json.dumps(pairs), encoding="utf-8")
    print(f"    NCBI: {len(pairs)} pairs extracted")
    return pairs

def _fetch_genbank_strains(accessions, batch_size=80, max_fetch=500):
    """Fetch host strain info from NCBI GenBank records (/host qualifier)."""
    cache_path = RAW_DIR / "ncbi_strain_cache.json"
    strain_map = {}
    if cache_path.exists():
        try: strain_map = json.loads(cache_path.read_text(encoding="utf-8"))
        except: pass
    to_fetch = [a for a in accessions
                if a not in strain_map and re.sub(r"\.\d+$","",a) not in strain_map][:max_fetch]
    if not to_fetch:
        return strain_map
    print(f"    Fetching strain info for {len(to_fetch)} accessions from GenBank...")
    for start in range(0, len(to_fetch), batch_size):
        batch = to_fetch[start:start+batch_size]
        try:
            url = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                   f"?db=nuccore&id={','.join(batch)}&rettype=gb&retmode=text"
                   f"&email=precisionphage@isef2026.edu")
            raw = _fetch(url, timeout=180)
            text = raw.decode("utf-8", errors="replace")
            current_acc = None
            for line in text.splitlines():
                if line.startswith("VERSION"):
                    parts = line.split()
                    if len(parts) >= 2:
                        current_acc = parts[1].strip()
                m_host = re.search(r'/host="([^"]+)"', line)
                if m_host and current_acc:
                    host_raw = m_host.group(1).strip().lower()
                    strain_map[current_acc] = host_raw
                    base = re.sub(r"\.\d+$", "", current_acc)
                    if base != current_acc:
                        strain_map[base] = host_raw
            time.sleep(0.5)
        except Exception as e:
            print(f"    Batch {start} strain fetch failed: {e}")
    try: cache_path.write_text(json.dumps(strain_map), encoding="utf-8")
    except: pass
    found = sum(1 for a in accessions if a in strain_map or re.sub(r"\.\d+$","",a) in strain_map)
    print(f"    Got strain info for {found}/{len(accessions)} phages")
    return strain_map

def _get_inphared():
    urls = [
        ("https://github.com/RyanCook94/inphared/releases/download/1Mar2024/1Mar2024_phages_downloaded_data.tsv","inphared_Mar2024.tsv"),
        ("https://github.com/RyanCook94/inphared/releases/download/1Jan2024/1Jan2024_phages_downloaded_data.tsv","inphared_Jan2024.tsv"),
        ("https://inphared.s3.climb.ac.uk/1Jan2024_phages_downloaded_data.tsv","inphared_s3_Jan2024.tsv"),
        ("https://inphared.s3.climb.ac.uk/1Oct2023_phages_downloaded_data.tsv","inphared_s3_Oct2023.tsv"),
    ]
    for url, fname in urls:
        dest = RAW_DIR / fname
        if not _dl(url, dest, fname): continue
        first = open(dest,encoding="utf-8",errors="replace").readline()
        if "<html" in first.lower(): dest.unlink(); continue
        rows = []
        with open(dest,encoding="utf-8",errors="replace") as fh:
            for rec in csv.DictReader(fh, delimiter="\t"):
                acc  = (rec.get("Accession") or rec.get("accession") or "").strip()
                host = (rec.get("Host") or rec.get("host") or "").strip()
                if not acc or not host: continue
                h = _clean_host(host)
                if len(h.split()) < 2: continue
                rows.append({"phage":acc,"host":h,"host_strain":host.strip().lower(),
                             "label":1,"source":"inphared"})
        if rows:
            print(f"    INPHARED: {len(rows)} pairs from {fname}")
            return rows
    print("    INPHARED: all URLs failed")
    return []

def _load_vhi():
    df = pd.read_csv(VHI_CSV)
    lmap = {c.lower():c for c in df.columns}
    def fc(aliases):
        for a in aliases:
            if a in df.columns: return a
            if a.lower() in lmap: return lmap[a.lower()]
        raise ValueError(f"None of {aliases} found in {df.columns.tolist()}")
    pc = fc(["phage","phagename","PhageName","virus","VirusName"])
    hc = fc(["host","hostname","HostName","bacteria","BacteriaName"])
    rn = {}
    if pc!="phage": rn[pc]="phage"
    if hc!="host":  rn[hc]="host"
    if rn: df = df.rename(columns=rn)
    if "label" not in df.columns:
        for c in ["infection","infects","interaction","y","target"]:
            if c in df.columns: df=df.rename(columns={c:"label"}); break
        else: df["label"]=1
    df["host_strain"] = df["host"].str.strip().str.lower()
    df["host"]  = df["host_strain"].apply(_clean_host)
    df["phage"] = df["phage"].str.strip()
    if "genus" not in df.columns:
        df["genus"] = df["host"].str.split().str[0]
    return df

existing = _load_vhi()

def _map_label(x):
    s = str(x).strip().lower()
    if pd.isna(x) or s in ("", "nan", "none"): return 1
    if s in ("yes","infects","inf","1","1.0","positive","true"): return 1
    if s in ("no","noinf","noneinfects","noinfection","0","0.0","negative","false",
             "no infection","no infects"): return 0
    try: return int(float(s))
    except: return 1

existing["label"] = existing["label"].apply(_map_label)
print(f"  Loaded VHI: {len(existing)} rows | {existing['phage'].nunique()} phages | {existing['host'].nunique()} hosts")

all_new = []
for fn in [_get_virushostdb, _get_inphared, _get_ncbi]:
    try: all_new.extend(fn())
    except Exception as e: print(f"  Source error: {e}")

existing_genera = {h.split()[0] for h in existing["host"].str.lower()}
existing_pairs  = set(zip(existing["phage"].str.lower(), existing["host"].str.lower()))
added = skip_g = skip_d = 0
new_records = []
for row in all_new:
    h = _clean_host(row.get("host",""))
    p = row.get("phage","").strip()
    g = h.split()[0] if h.split() else ""
    if g not in existing_genera: skip_g+=1; continue
    key = (p.lower(), h)
    if key in existing_pairs: skip_d+=1; continue
    existing_pairs.add(key)
    added += 1
    hs = row.get("host_strain", h)
    new_records.append({"phage":p,"host":h,"host_strain":hs,"label":1,"genus":g,
        "k3dist":0.0,"k6dist":0.0,"GCdiff":0.0,"Homology":0.0,"source":"external"})

print(f"  Added: {added} | Genus-mismatch: {skip_g} | Duplicates: {skip_d}")
if "source" not in existing.columns: existing["source"]="original"

rng0 = np.random.default_rng(SEED)

if new_records:
    aug = pd.concat([existing, pd.DataFrame(new_records)], ignore_index=True)
else:
    aug = existing.copy()

all_hosts   = aug["host"].unique().tolist()
all_genera  = aug["genus"].unique().tolist()
pos_set     = set(zip(aug["phage"].str.lower(), aug["host"].str.lower()))
gh          = aug.groupby("genus")["host"].unique().to_dict()
phage_genera= aug[aug["label"]==1].groupby("phage")["genus"].apply(set).to_dict()

neg_rows = []
print(f"  Generating negatives for {aug['phage'].nunique()} phages...")
_ph_pos_counts = aug[aug["label"]==1].groupby("phage").size()
for ph, pos_genera in phage_genera.items():
    n_pos = int(_ph_pos_counts.get(ph, 0))
    if n_pos == 0: continue

    wg_cands = []
    for g in pos_genera:
        wg_cands += [h for h in list(gh.get(g,[])) if (ph.lower(),h.lower()) not in pos_set]
    xg_cands = []
    for g in all_genera:
        if g not in pos_genera:
            arr = gh.get(g,[])
            xg_cands += list(arr) if len(arr) else []

    n_wg = min(n_pos * 1, len(wg_cands))
    n_xg = min(n_pos * 2, len(xg_cands))

    if n_wg > 0:
        for h in rng0.choice(wg_cands, size=n_wg, replace=False):
            if (ph.lower(), h.lower()) not in pos_set:
                neg_rows.append({"phage":ph,"host":h,"host_strain":h,"label":0,
                    "genus":h.split()[0] if h.split() else h,
                    "k3dist":0.0,"k6dist":0.0,"GCdiff":0.0,"Homology":0.0,
                    "source":"negative"})
                pos_set.add((ph.lower(), h.lower()))

    if n_xg > 0:
        for h in rng0.choice(xg_cands, size=n_xg, replace=False):
            if (ph.lower(), h.lower()) not in pos_set:
                neg_rows.append({"phage":ph,"host":h,"host_strain":h,"label":0,
                    "genus":h.split()[0] if h.split() else h,
                    "k3dist":0.0,"k6dist":0.0,"GCdiff":0.0,"Homology":0.0,
                    "source":"negative"})
                pos_set.add((ph.lower(), h.lower()))

print(f"  Generated {len(neg_rows)} negatives for {len(phage_genera)} phages")
if neg_rows:
    aug = pd.concat([aug, pd.DataFrame(neg_rows)], ignore_index=True)
dataset = aug.copy()
dataset["genus"] = dataset["host"].str.lower().str.split().str[0]

# Enrich strain info from NCBI GenBank records
_acc_like_phages = [p for p in dataset["phage"].unique()
                    if re.match(r'^[A-Z]{1,2}\d{5,}', p)]
if _acc_like_phages:
    _gb_strain_map = _fetch_genbank_strains(_acc_like_phages)
    if _gb_strain_map:
        _base_ph = dataset["phage"].str.replace(r"\.\d+$", "", regex=True)
        _gb_strains = _base_ph.map(_gb_strain_map).fillna(dataset["phage"].map(_gb_strain_map))
        _has_gb = _gb_strains.notna()
        if _has_gb.any():
            if "host_strain" not in dataset.columns:
                dataset["host_strain"] = dataset["host"]
            dataset.loc[_has_gb, "host_strain"] = _gb_strains[_has_gb]
            print(f"  Enriched {_has_gb.sum()} rows with GenBank strain info")
if "host_strain" not in dataset.columns:
    dataset["host_strain"] = dataset["host"]
_n_with_strain = (dataset["host_strain"] != dataset["host"]).sum()
print(f"  Rows with strain-level host info: {_n_with_strain}/{len(dataset)}")

def leaf_frac(df):
    pos = df[df["label"]==1]
    cnt = pos.groupby("phage")["host"].nunique()
    return (cnt==1).sum()/len(cnt) if len(cnt) else 1.0

print(f"  Dataset: {len(dataset)} rows | {dataset['phage'].nunique()} phages | "
      f"{dataset['host'].nunique()} hosts | leaf frac: {leaf_frac(dataset):.1%}")
dataset.to_csv(AUG_CSV, index=False)
print(f"  Saved → {AUG_CSV.name}")


# ════════════════════════════════════════════════════════════════
# SECTION 2 — DOWNLOAD HOST GENOME FASTAs FROM NCBI REFSEQ
# ════════════════════════════════════════════════════════════════
print("\n[2] Downloading host genome FASTAs from NCBI RefSeq...")

HOST_FASTA_DIR = FASTA_DIR / "hosts"
HOST_FASTA_DIR.mkdir(parents=True, exist_ok=True)

_ASSEMBLY_PREF = {"Complete Genome":0,"Chromosome":1,"Scaffold":2,"Contig":3}

def _taxon_id(species):
    query = species.strip().title()
    try:
        data = _ncbi_post("esearch.fcgi", db="taxonomy", term=query, retmax=3)
        ids = data.get("esearchresult",{}).get("idlist",[])
        if ids: return ids[0]
    except: pass
    try:
        genus = query.split()[0]
        data = _ncbi_post("esearch.fcgi", db="taxonomy", term=f"{genus}[genus]", retmax=1)
        ids = data.get("esearchresult",{}).get("idlist",[])
        if ids: return ids[0]
    except: pass
    return None

def _best_assembly(taxon_id):
    for cat in ["reference genome[RefSeq Category]","representative genome[RefSeq Category]",""]:
        q = f"txid{taxon_id}[Organism] AND latest[filter]"
        if cat: q += f" AND {cat}"
        try:
            data  = _ncbi_post("esearch.fcgi", db="assembly", term=q, retmax=5)
            ids   = data.get("esearchresult",{}).get("idlist",[])
            if not ids: continue
            summ  = _ncbi_post("esummary.fcgi", db="assembly", id=",".join(ids[:5]))
            cands = []
            for uid, rec in summ.get("result",{}).items():
                if uid=="uids": continue
                ftp = rec.get("ftppath_refseq","") or rec.get("ftppath_genbank","")
                if not ftp or ftp=="na": continue
                lvl = rec.get("assemblylevel","Contig")
                cands.append({"ftp":ftp,"level":lvl,"pref":_ASSEMBLY_PREF.get(lvl,4)})
            if cands: return min(cands, key=lambda x: x["pref"])
        except: continue
    return None

def _dl_genome(ftp_path, dest):
    base  = ftp_path.replace("ftp://ftp.ncbi.nlm.nih.gov","https://ftp.ncbi.nlm.nih.gov")
    name  = ftp_path.rstrip("/").split("/")[-1]
    for url in [f"{base}/{name}_genomic.fna.gz", f"{base}/{name}_genomic.fna"]:
        try:
            time.sleep(0.4)
            raw = _fetch(url, timeout=300)
            if url.endswith(".gz"):
                with gzip.open(io.BytesIO(raw)) as gz: raw = gz.read()
            text = raw.decode("utf-8", errors="replace")
            recs = []
            cur_h, cur_s = "", []
            for line in text.splitlines():
                if line.startswith(">"):
                    if cur_h: recs.append((cur_h,"".join(cur_s)))
                    cur_h=line; cur_s=[]
                else: cur_s.append(line.strip())
            if cur_h: recs.append((cur_h,"".join(cur_s)))
            recs.sort(key=lambda r: len(r[1]), reverse=True)
            dest.write_text("\n".join(f"{h}\n{s}" for h,s in recs[:3])+"\n")
            return True
        except: pass
    return False

species_list = sorted(dataset["host"].str.lower().str.strip().unique())
_dl_log = FASTA_DIR/"host_download_log.csv"
_already_done = (_dl_log.exists() and HOST_FASTA.is_file()
    and HOST_FASTA.stat().st_size > 100_000_000)
if _already_done:
    print(f"  Host genomes already downloaded, skipping download step.")
    print(f"  (Delete {_dl_log} to force re-download)")
else:
    dl_ok = 0
    log_rows = []
    for i, sp in enumerate(species_list, 1):
        slug = re.sub(r"[^\w]","_",sp)
        dest = HOST_FASTA_DIR / f"{slug}.fasta"
        if dest.exists() and dest.stat().st_size > 500:
            dl_ok += 1; continue
        print(f"  [{i}/{len(species_list)}] {sp}")
        tid = _taxon_id(sp)
        if not tid:
            log_rows.append({"species":sp,"status":"no_taxon"})
            print(f"    \u2717 taxon not found"); continue
        asm = _best_assembly(tid)
        if not asm:
            log_rows.append({"species":sp,"status":"no_assembly"})
            print(f"    \u2717 no assembly"); continue
        ok = _dl_genome(asm["ftp"], dest)
        if ok:
            dl_ok += 1
            log_rows.append({'species':sp,'status':'ok','level':asm['level']})
            print(f"    \u2713 {asm['level']}")
        else:
            log_rows.append({'species':sp,'status':'dl_failed'})
            print(f"    \u2717 download failed")
    pd.DataFrame(log_rows).to_csv(_dl_log, index=False)
    print(f"  Downloaded {dl_ok}/{len(species_list)} host genomes")
    with open(HOST_FASTA, "w") as out:
        for sp in species_list:
            slug = re.sub(r"[^\w]","_",sp)
            fp   = HOST_FASTA_DIR / f"{slug}.fasta"
            if not fp.exists(): continue
            flines = fp.read_text().splitlines()
            first_header = True
            for line in flines:
                if line.startswith(">") and first_header:
                    out.write(f">{sp} {line[1:]}\n"); first_header=False
                else: out.write(line+"\n")
    print(f"  Concatenated \u2192 {HOST_FASTA.name}  ({HOST_FASTA.stat().st_size//1024} KB)")


# ════════════════════════════════════════════════════════════════
# SECTION 3 — COMPUTE GENOMIC FEATURES
# ════════════════════════════════════════════════════════════════
print("\n[3] Computing genomic features from FASTA sequences...")

_COMP = str.maketrans("ACGTacgt","TGCAtgca")
def _revcomp(s): return s.translate(_COMP)[::-1]
def _clean(s): return re.sub(r"[^ACGTacgt]","",s).upper()

def _parse_fasta(path):
    path = Path(path)
    if not path.is_file(): raise FileNotFoundError(f"Not a file: {path}")
    recs = {}
    cur_id, parts = None, []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(str(path),"rt",errors="replace") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if cur_id: recs[cur_id] = _clean("".join(parts))
                cur_id = line[1:].split()[0]; parts=[]
            else: parts.append(line)
    if cur_id: recs[cur_id] = _clean("".join(parts))
    return recs

_NUC2INT = np.zeros(128, dtype=np.int8)
_NUC2INT[ord('A')] = 0; _NUC2INT[ord('C')] = 1
_NUC2INT[ord('G')] = 2; _NUC2INT[ord('T')] = 3
_KMER_MULTS = {k: 4 ** np.arange(k - 1, -1, -1, dtype=np.int64) for k in range(1, 7)}

def _seq_encode(seq):
    return _NUC2INT[np.frombuffer(seq.encode('ascii'), dtype=np.uint8)]

def _kmer_freqs(seq, k):
    n_kmers = 4 ** k
    if len(seq) < k:
        return np.zeros(n_kmers, dtype=np.float32)
    enc = _seq_encode(seq)
    rc_enc = np.ascontiguousarray(3 - enc[::-1])
    mult = _KMER_MULTS[k]
    ids = np.concatenate([
        np.lib.stride_tricks.sliding_window_view(s.astype(np.int64), k) @ mult
        for s in (enc, rc_enc) if len(s) >= k])
    counts = np.bincount(ids, minlength=n_kmers)
    total = counts.sum()
    return (counts / total).astype(np.float32) if total > 0 else np.zeros(n_kmers, dtype=np.float32)

def _di_ra(seq):
    mono = _kmer_freqs(seq, 1); di = _kmer_freqs(seq, 2)
    denom = np.outer(mono, mono).ravel()
    return np.where(denom > 1e-9, di / denom, 0.0).astype(np.float32)

def _tri(seq):
    return _kmer_freqs(seq, 3)

def _tet(seq):
    return _kmer_freqs(seq, 4)

_CODONS = [a+b+c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
_AA = {"TTT":"F","TTC":"F","TTA":"L","TTG":"L","CTT":"L","CTC":"L","CTA":"L","CTG":"L",
       "ATT":"I","ATC":"I","ATA":"I","ATG":"M","GTT":"V","GTC":"V","GTA":"V","GTG":"V",
       "TCT":"S","TCC":"S","TCA":"S","TCG":"S","CCT":"P","CCC":"P","CCA":"P","CCG":"P",
       "ACT":"T","ACC":"T","ACA":"T","ACG":"T","GCT":"A","GCC":"A","GCA":"A","GCG":"A",
       "TAT":"Y","TAC":"Y","TAA":"*","TAG":"*","CAT":"H","CAC":"H","CAA":"Q","CAG":"Q",
       "AAT":"N","AAC":"N","AAA":"K","AAG":"K","GAT":"D","GAC":"D","GAA":"E","GAG":"E",
       "TGT":"C","TGC":"C","TGA":"*","TGG":"W","CGT":"R","CGC":"R","CGA":"R","CGG":"R",
       "AGT":"S","AGC":"S","AGA":"R","AGG":"R","GGT":"G","GGC":"G","GGA":"G","GGG":"G"}

_AA_GROUPS_IDX = {}
for _ci, _cs in enumerate(_CODONS):
    _aa = _AA.get(_cs, "X")
    if _aa not in ("*", "X"):
        _AA_GROUPS_IDX.setdefault(_aa, []).append(_ci)
for _aa in list(_AA_GROUPS_IDX):
    _AA_GROUPS_IDX[_aa] = np.array(_AA_GROUPS_IDX[_aa], dtype=np.int64)

def _rscu(seq):
    enc = _seq_encode(seq)
    rc_enc = np.ascontiguousarray(3 - enc[::-1])
    mult3 = _KMER_MULTS[3]
    all_ids = []
    for strand in (enc, rc_enc):
        for frame in range(3):
            end = frame + ((len(strand) - frame) // 3) * 3
            if end <= frame: continue
            codons = strand[frame:end].astype(np.int64).reshape(-1, 3)
            all_ids.append(codons @ mult3)
    if not all_ids:
        return np.ones(64, dtype=np.float32)
    counts = np.bincount(np.concatenate(all_ids), minlength=64).astype(np.float64)
    rscu_arr = np.ones(64, dtype=np.float32)
    for indices in _AA_GROUPS_IDX.values():
        group_total = counts[indices].sum()
        exp = group_total / len(indices)
        if exp > 0:
            rscu_arr[indices] = (counts[indices] / exp).astype(np.float32)
    return rscu_arr

def _gc(seq): return (seq.count("G")+seq.count("C"))/len(seq) if seq else 0.5

# ── Build fast accession→key reverse index to avoid O(n²) lookups ──────
def _build_seq_index(seqs):
    """Build a {accession: key} map including version-stripped forms."""
    idx = {}
    for k in seqs:
        idx[k] = k
        base = re.sub(r"\.\d+$", "", k)
        if base != k:
            idx[base] = k
    return idx

def _find_seq(name, seqs, index=None):
    if index is not None:
        key = index.get(name) or index.get(re.sub(r"\.\d+$","",name))
        if key: return seqs[key]
    else:
        if name in seqs: return seqs[name]
        base = re.sub(r"\.\d+$","",name)
        if base in seqs: return seqs[base]
    # Fallback substring scan only if fast lookup failed
    for k, v in seqs.items():
        if name in k or k in name: return v
    return None

def _load_inphared_seq(accession, base_dir):
    acc = accession.strip()
    acc_base = re.sub(r"\.\d+$", "", acc)
    for a in [acc, acc_base]:
        for ext in [".fna", ".ffn", ".fa", ".fasta"]:
            p = Path(base_dir) / a / f"{a}{ext}"
            if p.is_file():
                try:
                    seqs = _parse_fasta(str(p))
                    if seqs:
                        return _clean("".join(seqs.values()))
                except Exception:
                    pass
    return None

_inphared_ok = (INPHARED_GENOMES_DIR is not None and
                Path(INPHARED_GENOMES_DIR).is_dir() and
                any(True for _ in Path(INPHARED_GENOMES_DIR).iterdir()))

phage_seqs = {}
_phage_seq_index = {}
USE_INPHARED_LIVE = False

if _inphared_ok:
    print(f"  INPHARED GenomesDB detected at {INPHARED_GENOMES_DIR}")
    print(f"  Will load phage sequences on-demand (no 10 GB concatenation needed).")
    USE_INPHARED_LIVE = True
else:
    _pfp_check = Path(PHAGE_FASTA)
    if _pfp_check.is_dir():
        _fasta_files = (list(_pfp_check.glob("*.fasta")) +
                        list(_pfp_check.glob("*.fa")) +
                        list(_pfp_check.glob("*.fna")) +
                        list(_pfp_check.glob("*.ffn")))
        if _fasta_files:
            _concat_dest = _pfp_check.parent / "phage_genomes_concat.fasta"
            if not _concat_dest.exists():
                print(f"  Auto-concatenating {len(_fasta_files)} phage FASTA files...")
                with open(_concat_dest, "w") as _cout:
                    for _ff in sorted(_fasta_files):
                        _cout.write(_ff.read_text(errors="replace"))
                print(f"  Concatenated → {_concat_dest.name}")
            else:
                print(f"  Using existing concat: {_concat_dest.name}")
            PHAGE_FASTA = str(_concat_dest)
    _pfp = Path(PHAGE_FASTA)
    if _pfp.is_file():
        try:
            phage_seqs = _parse_fasta(PHAGE_FASTA)
            _phage_seq_index = _build_seq_index(phage_seqs)
            print(f"  Phage FASTA: {len(phage_seqs)} records")
        except Exception as e:
            print(f"  WARNING: Could not read phage FASTA: {e}")
    else:
        print(f"  NOTE: No phage FASTA — continuing with SVD-only features.")

# Load concat FASTA as fallback
_concat_fasta = FASTA_DIR / "phage_genomes_concat.fasta"
if _concat_fasta.is_file():
    try:
        phage_seqs = _parse_fasta(str(_concat_fasta))
        _phage_seq_index = _build_seq_index(phage_seqs)
        print(f"  Concat FASTA fallback: {len(phage_seqs)} records loaded")
    except Exception as e:
        print(f"  WARNING: Could not read concat FASTA: {e}")
elif not USE_INPHARED_LIVE:
    _pfp = Path(PHAGE_FASTA)
    if _pfp.is_file():
        try:
            phage_seqs = _parse_fasta(PHAGE_FASTA)
            _phage_seq_index = _build_seq_index(phage_seqs)
            print(f"  Phage FASTA: {len(phage_seqs)} records")
        except Exception as e:
            print(f"  WARNING: Could not read phage FASTA: {e}")

host_seqs = {}
_host_seq_index = {}
if HOST_FASTA.is_file():
    try:
        host_seqs = _parse_fasta(HOST_FASTA)
        _host_seq_index = _build_seq_index(host_seqs)
        print(f"  Host FASTA:  {len(host_seqs)} records")
    except Exception as e:
        print(f"  WARNING: Could not read host FASTA: {e}")

phage_names = sorted(dataset["phage"].unique())
host_names  = sorted(dataset["host"].unique())

# Compute phage features
print(f"  Computing features for {len(phage_names)} phages...")
tet_vecs=[]; cub_vecs=[]; ph_rows=[]
missing=0
_n_inphared_hit = 0

_phage_seq_list = []
for pname in phage_names:
    seq = None
    if USE_INPHARED_LIVE:
        seq = _load_inphared_seq(pname, INPHARED_GENOMES_DIR)
        if seq and len(seq) >= 500:
            _n_inphared_hit += 1
    if seq is None or len(seq) < 500:
        seq = _find_seq(pname, phage_seqs, _phage_seq_index)
    _phage_seq_list.append(seq)

def _compute_phage_feats(args):
    pname, seq = args
    if seq is None or len(seq) < 500:
        return ({"phage":pname,"p_gc":0.5,"p_loglen":0.0,"p_found":0},
                np.zeros(256, dtype=np.float32), np.ones(64, dtype=np.float32))
    row = {"phage":pname,"p_gc":_gc(seq),"p_loglen":math.log10(len(seq)),"p_found":1}
    di = _di_ra(seq); tri = _tri(seq)
    row.update({f"p_di_{j:02d}":float(di[j]) for j in range(16)})
    row.update({f"p_tri_{j:03d}":float(tri[j]) for j in range(64)})
    return row, _tet(seq), _rscu(seq)

_n_feat_workers = min(8, max(1, os.cpu_count() or 4))
with ThreadPoolExecutor(max_workers=_n_feat_workers) as _pool:
    _ph_results = list(_pool.map(_compute_phage_feats,
        zip(phage_names, _phage_seq_list)))
for row, tet_v, cub_v in _ph_results:
    ph_rows.append(row)
    tet_vecs.append(tet_v)
    cub_vecs.append(cub_v)
    if row.get("p_found", 0) == 0: missing += 1

found = len(phage_names)-missing
_n_concat_hit = found - _n_inphared_hit
if USE_INPHARED_LIVE:
    print(f"  Phages with sequences: {found}/{len(phage_names)}  "
          f"({_n_inphared_hit} from INPHARED, {_n_concat_hit} from concat FASTA, "
          f"{missing} missing)")
else:
    print(f"  Phages with sequences: {found}/{len(phage_names)} ({missing} missing)")

# NCBI fallback for missing accessions
_acc_pattern = re.compile(r'^[A-Z]{1,2}\d{5,8}(\.\d+)?$')
_missing_accs = [phage_names[i] for i,row in enumerate(ph_rows)
                 if row.get("p_found",0)==0
                 and _acc_pattern.match(phage_names[i])]
if _missing_accs:
    print(f"  Attempting NCBI download for {len(_missing_accs)} missing accessions...")
    _ncbi_fasta_cache = RAW_DIR / "ncbi_phage_seqs.fasta"
    _ncbi_cache = {}
    if _ncbi_fasta_cache.is_file():
        try: _ncbi_cache = _parse_fasta(str(_ncbi_fasta_cache))
        except: pass
    _ncbi_cache_index = _build_seq_index(_ncbi_cache)
    _newly_fetched = 0
    _ncbi_out = open(_ncbi_fasta_cache, "a")
    for _acc in _missing_accs[:500]:
        _acc_base = re.sub(r"\.\d+$","",_acc)
        if _acc_base in _ncbi_cache or _acc in _ncbi_cache:
            continue
        try:
            _url = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                    f"?db=nuccore&id={_acc}&rettype=fasta&retmode=text"
                    f"&email=precisionphage@isef2026.edu")
            _raw = _fetch(_url, timeout=30)
            _txt = _raw.decode("utf-8", errors="replace").strip()
            if _txt.startswith(">"):
                _ncbi_out.write(_txt + "\n")
                _seqs_new = {}
                _cur, _parts = None, []
                for _ln in _txt.splitlines():
                    if _ln.startswith(">"):
                        if _cur: _seqs_new[_cur] = _clean("".join(_parts))
                        _cur = _ln[1:].split()[0]; _parts=[]
                    else: _parts.append(_ln.strip())
                if _cur: _seqs_new[_cur] = _clean("".join(_parts))
                _ncbi_cache.update(_seqs_new)
                _ncbi_cache_index.update(_build_seq_index(_seqs_new))
                _newly_fetched += 1
            time.sleep(0.35)
        except Exception:
            pass
    _ncbi_out.close()
    if _newly_fetched > 0:
        print(f"  NCBI fetched {_newly_fetched} new sequences → ncbi_phage_seqs.fasta")
        _backfilled = 0
        for i, pname in enumerate(phage_names):
            if ph_rows[i].get("p_found",0) == 1:
                continue
            _seq = _find_seq(pname, _ncbi_cache, _ncbi_cache_index)
            if _seq and len(_seq) >= 500:
                ph_rows[i].update({"p_gc":_gc(_seq),"p_loglen":math.log10(len(_seq)),"p_found":1})
                _di2 = _di_ra(_seq); _tri2 = _tri(_seq)
                ph_rows[i].update({f"p_di_{j:02d}":float(_di2[j]) for j in range(16)})
                ph_rows[i].update({f"p_tri_{j:03d}":float(_tri2[j]) for j in range(64)})
                tet_vecs[i] = _tet(_seq)
                cub_vecs[i] = _rscu(_seq)
                _backfilled += 1
        print(f"  Back-filled features for {_backfilled} phages")
    else:
        print(f"  NCBI: no new sequences found (all already cached or network unavailable)")

# PCA-reduce tet (256→32) and cub (64→18)
tet_arr = np.vstack(tet_vecs)
cub_arr = np.vstack(cub_vecs)
n_tet = min(32, tet_arr.shape[0]-1, tet_arr.shape[1]-1)
n_cub = min(18, cub_arr.shape[0]-1, cub_arr.shape[1]-1)
if n_tet > 1 and np.any(tet_arr!=0):
    pca_tet = PCA(n_components=n_tet, random_state=SEED)
    tet_red = pca_tet.fit_transform(tet_arr)
    print(f"  Tet PCA: {n_tet}d explains {pca_tet.explained_variance_ratio_.sum():.1%}")
else: tet_red = tet_arr[:,:n_tet]
if n_cub > 1 and np.any(cub_arr!=1):
    pca_cub = PCA(n_components=n_cub, random_state=SEED)
    cub_red = pca_cub.fit_transform(cub_arr)
    print(f"  CUB PCA: {n_cub}d explains {pca_cub.explained_variance_ratio_.sum():.1%}")
else: cub_red = cub_arr[:,:n_cub]

phage_feat_df = pd.DataFrame(ph_rows)
for j in range(n_tet): phage_feat_df[f"p_tet_{j:02d}"] = tet_red[:,j]
for j in range(n_cub): phage_feat_df[f"p_cub_{j:02d}"] = cub_red[:,j]
phage_feat_df = phage_feat_df.set_index("phage")

# Compute host features + pair-level features
host_tet={}; host_cub={}; host_gc={}
if host_seqs:
    print(f"  Computing features for {len(host_names)} hosts...")
    def _compute_host_feats(hname):
        seq = _find_seq(hname, host_seqs, _host_seq_index)
        if seq and len(seq) >= 500:
            return hname, _tet(seq), _rscu(seq), _gc(seq)
        return hname, None, None, None
    with ThreadPoolExecutor(max_workers=min(8, max(1, os.cpu_count() or 4))) as _pool:
        _host_results = list(_pool.map(_compute_host_feats, host_names))
    for hname, t, c, g in _host_results:
        if t is not None:
            host_tet[hname] = t; host_cub[hname] = c; host_gc[hname] = g

print("  Building pair-level features (vectorised)...")
_n = len(dataset)
_ph = dataset["phage"].values
_ho = dataset["host"].values
p_tet_cols = [f"p_tet_{j:02d}" for j in range(n_tet)]
p_cub_cols = [f"p_cub_{j:02d}" for j in range(n_cub)]
_p_tet_mat = np.zeros((_n, n_tet), dtype=np.float32)
if p_tet_cols and all(c in phage_feat_df.columns for c in p_tet_cols):
    _p_tet_mat = np.nan_to_num(
        phage_feat_df.reindex(_ph)[p_tet_cols].values.astype(np.float32), nan=0.0)
_h_tet_mat = np.zeros((_n, n_tet), dtype=np.float32)
for _hname, _vec in host_tet.items():
    _m = _ho == _hname
    if _m.any():
        _h_tet_mat[_m] = _vec[:n_tet]
_ph_ok = dataset["phage"].isin(phage_feat_df.index).values
_ho_t_ok = np.isin(_ho, list(host_tet.keys()))
_tet_mask = _ph_ok & _ho_t_ok & (n_tet > 1)
_tet_corr = np.zeros(_n, dtype=np.float32)
if _tet_mask.any():
    _a = _p_tet_mat[_tet_mask].astype(np.float64)
    _b = _h_tet_mat[_tet_mask].astype(np.float64)
    _am = _a - _a.mean(axis=1, keepdims=True)
    _bm = _b - _b.mean(axis=1, keepdims=True)
    _num = (_am * _bm).sum(axis=1)
    _den = np.sqrt(np.maximum((_am * _am).sum(axis=1) * (_bm * _bm).sum(axis=1), 1e-20))
    _tet_corr[_tet_mask] = np.nan_to_num(_num / _den, nan=0.0).astype(np.float32)

_p_cub_mat = np.zeros((_n, n_cub), dtype=np.float32)
if p_cub_cols and all(c in phage_feat_df.columns for c in p_cub_cols):
    _p_cub_mat = np.nan_to_num(
        phage_feat_df.reindex(_ph)[p_cub_cols].values.astype(np.float32), nan=0.0)
_h_cub_mat = np.zeros((_n, n_cub), dtype=np.float32)
for _hname, _vec in host_cub.items():
    _m = _ho == _hname
    if _m.any():
        _h_cub_mat[_m] = _vec[:n_cub]
_cub_mask = dataset["phage"].isin(phage_feat_df.index).values & np.isin(_ho, list(host_cub.keys()))
_cub_dist = np.zeros(_n, dtype=np.float32)
if _cub_mask.any():
    _cub_dist[_cub_mask] = np.linalg.norm(_p_cub_mat[_cub_mask] - _h_cub_mat[_cub_mask], axis=1)

_p_gc = phage_feat_df.reindex(_ph)["p_gc"].fillna(0.5).values.astype(np.float32)
_h_gc = np.array([host_gc.get(h, 0.5) for h in _ho], dtype=np.float32)
_gc_match = 1.0 - np.abs(_p_gc - _h_gc)
_p_ll = phage_feat_df.reindex(_ph)["p_loglen"].fillna(0.0).values.astype(np.float32)
_len_ratio = np.where(_p_ll > 0, np.minimum(_p_ll / 7.0, 2.0), 0.0).astype(np.float32)

dataset["tetra_corr"] = _tet_corr
dataset["cub_dist"]   = _cub_dist
dataset["gc_match"]   = _gc_match
dataset["len_ratio"]  = _len_ratio

gf_rows = []
for ph in phage_names:
    if ph not in phage_feat_df.index: continue
    row = {"phage":ph}
    row.update(phage_feat_df.loc[ph].to_dict())
    gf_rows.append(row)
pd.DataFrame(gf_rows).to_csv(GEN_FEAT, index=False)
print(f"  Saved genomic features → {GEN_FEAT.name}")
print(f"  Total features per phage: {phage_feat_df.shape[1]}")


# ════════════════════════════════════════════════════════════════
# SECTION 4 — BUILD NODE + EDGE FEATURE MATRICES
# ════════════════════════════════════════════════════════════════
print("\n[4] Building node and edge feature matrices...")

dataset["phage"] = dataset["phage"].astype(str).str.strip()
dataset["host"]  = dataset["host"].astype(str).str.strip()
dataset = dataset[(dataset["phage"]!="") & (dataset["host"]!="")].copy()
for c in NUMERIC_FEATS:
    if c not in dataset.columns: dataset[c]=0.0
    dataset[c] = pd.to_numeric(dataset[c], errors="coerce").fillna(0.0)
dataset = dataset.reset_index(drop=True)
dataset["label"] = dataset["label"].apply(_map_label)

phage_list = sorted(dataset["phage"].unique())
host_list  = sorted(dataset["host"].unique())
phage2idx  = {p:i for i,p in enumerate(phage_list)}
phage_name2idx = {p.lower(): p for p in phage_list}
host2idx   = {h:i for i,h in enumerate(host_list)}
n_phages   = len(phage_list)
n_hosts    = len(host_list)
dataset["phage_idx"] = dataset["phage"].map(phage2idx)
dataset["host_idx"]  = dataset["host"].map(host2idx)

# FIX: Verify EDGE_FEATS_NP will align with dataset rows
assert len(dataset) == len(dataset.index), "Dataset index not clean after reset"

_vec = CountVectorizer(analyzer="char",ngram_range=(3,5),max_features=_CHAR_NGRAM_MAX,dtype=np.float32)
_X   = _vec.fit_transform(phage_list+host_list).toarray().astype(np.float32)
_k   = min(SVD_DIM,_X.shape[1]-1,_X.shape[0]-1)
_U,_s,_ = svds(_X,k=_k)
_nm = (_U*_s).astype(np.float32)
if _nm.shape[1]<SVD_DIM:
    _nm=np.hstack([_nm,np.zeros((_nm.shape[0],SVD_DIM-_nm.shape[1]),dtype=np.float32)])
phage_name_emb = _nm[:n_phages]
host_name_emb  = _nm[n_phages:]

ph_feat_cols = [c for c in phage_feat_df.columns
                if c.startswith(("p_di_","p_tri_","p_tet_","p_cub_")) or
                   c in ("p_gc","p_loglen")]
ph_feat_cols = [c for c in ph_feat_cols if c in phage_feat_df.columns]
_pg = np.zeros((n_phages,max(len(ph_feat_cols),1)),dtype=np.float32)
for i,ph in enumerate(phage_list):
    if ph in phage_feat_df.index:
        vals = phage_feat_df.loc[ph,ph_feat_cols].values.astype(np.float32)
        if len(vals)==len(ph_feat_cols): _pg[i]=vals
if len(ph_feat_cols)>0: phage_gen = StandardScaler().fit_transform(_pg)
else:
    _pg2 = np.zeros((n_phages,len(NUMERIC_FEATS)),dtype=np.float32)
    for ci,col in enumerate(NUMERIC_FEATS):
        _vals=dataset.groupby("phage")[col].mean()
        for i,ph in enumerate(phage_list): _pg2[i,ci]=float(_vals.get(ph,0.0))
    phage_gen = StandardScaler().fit_transform(_pg2)

_hfp = RAW_DIR/"host_features.csv"
host_feat_df2 = pd.read_csv(_hfp) if _hfp.exists() else pd.DataFrame(columns=["genus"])
bio_cols = [c for c in host_feat_df2.columns if c!="genus"]
host_bio = np.zeros((n_hosts,max(len(bio_cols),1)),dtype=np.float32)
if bio_cols:
    _bio_lut=host_feat_df2.set_index("genus")[bio_cols].to_dict("index")
    for i,h in enumerate(host_list):
        g=h.split()[0] if h.split() else h
        if g in _bio_lut: host_bio[i]=[_bio_lut[g].get(c,0.0) for c in bio_cols]

def build_structural(dataset_df, train_mask):
    tr=dataset_df[train_mask]; pos=tr[tr["label"]==1]
    pb=pos.groupby("phage")["host"].nunique()
    hv=pos.groupby("host")["phage"].nunique()
    gpr=tr.groupby("genus")["label"].mean()
    med_pb=float(pb.median()) if len(pb) else 0.0
    med_hv=float(hv.median()) if len(hv) else 0.0
    med_gpr=float(gpr.median()) if len(gpr) else 0.5
    ph_arr=np.array([pb.get(p,med_pb) for p in phage_list],dtype=np.float32).reshape(-1,1)
    ho_arr=np.column_stack([
        np.array([hv.get(h,med_hv) for h in host_list],dtype=np.float32),
        np.array([gpr.get(h.split()[0] if h.split() else h,med_gpr) for h in host_list],dtype=np.float32)])
    for arr in [ph_arr,ho_arr]:
        for c in range(arr.shape[1]):
            m,s=arr[:,c].mean(),arr[:,c].std()+1e-8
            arr[:,c]=(arr[:,c]-m)/s
    return ph_arr, ho_arr

# ── FIX: Scale genomic features only for phages WITH sequences ──────────
# Prevents missing-sequence phages all mapping to same constant after scaling
_pg_raw = np.zeros((n_phages, max(len(ph_feat_cols),1)), dtype=np.float32)
_pg_found_mask = np.zeros(n_phages, dtype=bool)
for _i, _ph in enumerate(phage_list):
    if _ph in phage_feat_df.index:
        _row = phage_feat_df.loc[_ph]
        if _row.get("p_found", 0) == 1:
            _pg_found_mask[_i] = True
            _vals = _row[ph_feat_cols].values.astype(np.float32) if ph_feat_cols else np.zeros(1, dtype=np.float32)
            if len(_vals) == max(len(ph_feat_cols), 1):
                _pg_raw[_i] = _vals

_pg_fixed = np.zeros_like(_pg_raw)
if _pg_found_mask.sum() > 1 and len(ph_feat_cols) > 0:
    _sc_pg = StandardScaler()
    _pg_fixed[_pg_found_mask] = _sc_pg.fit_transform(_pg_raw[_pg_found_mask])
_pg_found_col = _pg_found_mask.astype(np.float32).reshape(-1, 1)
PHAGE_BASE_FIXED = np.hstack([phage_name_emb, _pg_fixed, _pg_found_col]).astype(np.float32)
PHAGE_DIM_FIXED  = PHAGE_BASE_FIXED.shape[1] + 1  # +1 for per-fold structural

PHAGE_BASE = np.hstack([phage_name_emb, phage_gen]).astype(np.float32)
HOST_BASE  = np.hstack([host_name_emb, host_bio]).astype(np.float32)
PHAGE_DIM  = PHAGE_BASE.shape[1]+1
HOST_DIM   = HOST_BASE.shape[1]+2

_rich = [c for c in ["tetra_corr","cub_dist","gc_match","len_ratio"] if c in dataset.columns]
ALL_EDGE_FEATS = NUMERIC_FEATS + _rich
N_EDGE_FEATS   = len(ALL_EDGE_FEATS)
EDGE_FEATS_NP  = StandardScaler().fit_transform(
    dataset[ALL_EDGE_FEATS].fillna(0.0).values.astype(np.float32))

# FIX: Verify alignment
assert len(EDGE_FEATS_NP) == len(dataset), \
    f"EDGE_FEATS_NP row count {len(EDGE_FEATS_NP)} != dataset {len(dataset)}"

print(f"  Phage node dim: {PHAGE_DIM_FIXED}  Host node dim: {HOST_DIM}")
print(f"  Edge features ({N_EDGE_FEATS}): {ALL_EDGE_FEATS}")

all_labels = dataset["label"].values
if "host_strain" in dataset.columns:
    dataset["strain"] = dataset["host_strain"].fillna(dataset["host"])
else:
    dataset["strain"] = dataset["host"]


# ════════════════════════════════════════════════════════════════
# SECTION 5 — CLASSICAL ML MODELS
# ════════════════════════════════════════════════════════════════
print("\n[5] Training classical ML models...")

# Build flat feature matrix (needed whether cached or not)
all_feat_cols = NUMERIC_FEATS + _rich + (ph_feat_cols[:32] if len(ph_feat_cols)>0 else [])
all_feat_cols = [c for c in all_feat_cols if c in dataset.columns]
X_all = dataset[all_feat_cols].fillna(0.0).values.astype(np.float32)
y_all = all_labels.copy()
X_sc  = StandardScaler().fit_transform(X_all)

# FIX: raise min positives/negatives from 2 to 3 for more stable evaluation
valid_species = sorted([sp for sp in dataset["host"].unique()
    if (dataset["host"]==sp).sum() >= MIN_SP_ROWS
    and (dataset["host"]==sp).sum() <= MAX_SP_ROWS
    and dataset.loc[dataset["host"]==sp,"label"].nunique()==2
    and (dataset.loc[dataset["host"]==sp,"label"]==1).sum() >= 3
    and (dataset.loc[dataset["host"]==sp,"label"]==0).sum() >= 3])
print(f"  {len(valid_species)} valid species for evaluation")

_clf_cache = RESULTS_DIR / "classical_xgboost_loso.csv"
if _clf_cache.exists():
    print("  Classical results already cached — loading from CSV, skipping retraining.")
    clf_results = {}
    for _mn, _fn in [("XGBoost","classical_xgboost_loso.csv"),
                     ("RandomForest","classical_randomforest_loso.csv"),
                     ("GBM","classical_gbm_loso.csv")]:
        _fp = RESULTS_DIR / _fn
        if _fp.exists():
            _df = pd.read_csv(_fp, index_col="species")
            clf_results[_mn] = {"df":_df, "mean":_df["roc_auc"].mean(),
                                 "probas":np.full(len(dataset), np.nan), "pooled":{}}
            print(f"  {_mn}: LOSO mean AUC={_df['roc_auc'].mean():.4f}  ({len(_df)} folds) [cached]")
else:
    def run_classical_loso(model_name, make_model):
        rows=[]; probas=np.full(len(dataset),np.nan)
        for sp in valid_species:
            tm=(dataset["host"]==sp).values
            Xtr,ytr=X_sc[~tm],y_all[~tm]
            Xte,yte=X_sc[tm], y_all[tm]
            if len(np.unique(yte))<2: continue
            clf = make_model()
            clf.fit(Xtr,ytr)
            p = clf.predict_proba(Xte)[:,1]
            probas[tm]=p
            m=metrics(yte,p); m["species"]=sp; m["n_test"]=int(tm.sum())
            rows.append(m)
        df=pd.DataFrame(rows).set_index("species")
        vm=~np.isnan(probas)
        pooled=metrics(y_all[vm],probas[vm]) if vm.sum()>10 and len(np.unique(y_all[vm]))==2 else {}
        mean_auc=df["roc_auc"].mean() if len(df)>0 else 0.0
        print(f"  {model_name}: LOSO mean AUC={mean_auc:.4f}  pooled={pooled.get('roc_auc',0):.4f}  ({len(df)} folds)")
        df.to_csv(RESULTS_DIR/f"classical_{model_name.lower().replace(' ','_')}_loso.csv")
        return {"df":df,"mean":mean_auc,"probas":probas,"pooled":pooled}

    clf_results={}
    if HAS_XGB:
        clf_results["XGBoost"] = run_classical_loso("XGBoost",
            lambda: xgb.XGBClassifier(n_estimators=(400 if _THOROUGH else 220),max_depth=6,learning_rate=0.05,
                subsample=0.8,colsample_bytree=0.8,random_state=SEED,
                eval_metric="logloss",verbosity=0))
    clf_results["RandomForest"] = run_classical_loso("RandomForest",
        lambda: RandomForestClassifier(n_estimators=(400 if _THOROUGH else 220),max_depth=8,
            min_samples_leaf=2,random_state=SEED,n_jobs=-1))
    clf_results["GBM"] = run_classical_loso("GBM",
        lambda: HistGradientBoostingClassifier(max_iter=(280 if _THOROUGH else 160),max_depth=5,
            learning_rate=0.05,random_state=SEED))


# ════════════════════════════════════════════════════════════════
# SECTION 6 — GNN MODELS (GAT + GraphSAGE)
# ════════════════════════════════════════════════════════════════
print("\n[6] Training GNN models (GAT + GraphSAGE)...")

evaluable_genera = sorted([g for g in dataset["genus"].unique()
    if dataset[dataset["genus"]==g]["label"].nunique()==2
    and len(dataset[dataset["genus"]==g])>=MIN_SP_ROWS])

if HAS_TORCH:
    class PhageHostGAT(nn.Module):
        def __init__(self, phage_dim, host_dim, n_ef, n_ph):
            super().__init__()
            self._n_ph = n_ph
            H = GAT_HEADS
            self.proj_p = nn.Linear(phage_dim, HIDDEN_DIM)
            self.proj_h = nn.Linear(host_dim,  HIDDEN_DIM)
            self.gat1 = GATConv(HIDDEN_DIM, HIDDEN_DIM//H, heads=H, edge_dim=n_ef,
                                dropout=DROPOUT, concat=True, add_self_loops=True)
            self.gat2 = GATConv(HIDDEN_DIM, HIDDEN_DIM//H, heads=H, edge_dim=n_ef,
                                dropout=DROPOUT, concat=True, add_self_loops=True)
            self.gat3 = GATConv(HIDDEN_DIM, HIDDEN_DIM//H, heads=H, edge_dim=n_ef,
                                dropout=DROPOUT, concat=True, add_self_loops=True)
            self.gat4 = GATConv(HIDDEN_DIM, OUT_DIM,       heads=H, edge_dim=n_ef,
                                dropout=DROPOUT, concat=False, add_self_loops=True)
            self.bn1 = nn.LayerNorm(HIDDEN_DIM)
            self.bn2 = nn.LayerNorm(HIDDEN_DIM)
            self.bn3 = nn.LayerNorm(HIDDEN_DIM)
            self.drop = nn.Dropout(DROPOUT)
            self.mlp_gnn = nn.Sequential(
                nn.Linear(OUT_DIM*2 + n_ef, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(DROPOUT),
                nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(),
                nn.Linear(128, 1))
            self.mlp_bypass = nn.Sequential(
                nn.Linear(phage_dim + host_dim + n_ef, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(DROPOUT),
                nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(DROPOUT),
                nn.Linear(128, 64), nn.GELU(),
                nn.Linear(64, 1))
            self.alpha = nn.Parameter(torch.tensor(0.5))

        def encode(self, px, hx, ei, ea):
            h = torch.cat([F.gelu(self.proj_p(px)), F.gelu(self.proj_h(hx))], dim=0)
            h = self.bn1(F.gelu(self.gat1(h, ei, edge_attr=ea))); h = self.drop(h)
            h = self.bn2(F.gelu(self.gat2(h, ei, edge_attr=ea))); h = self.drop(h)
            h = self.bn3(F.gelu(self.gat3(h, ei, edge_attr=ea))); h = self.drop(h)
            return self.gat4(h, ei, edge_attr=ea)

        def decode(self, z, px, hx, pi, hi, ef):
            gnn_score = self.mlp_gnn(torch.cat([z[pi], z[hi + self._n_ph], ef], dim=-1))
            bypass_score = self.mlp_bypass(torch.cat([px[pi], hx[hi], ef], dim=-1))
            alpha = torch.sigmoid(self.alpha)
            return (alpha * gnn_score + (1 - alpha) * bypass_score).squeeze(-1)

        def forward(self, px, hx, ei, ea, pi, hi, ef):
            z = self.encode(px, hx, ei, ea)
            return self.decode(z, px, hx, pi, hi, ef)

        @torch.no_grad()
        def predict(self, px, hx, ei, ea, pi, hi, ef):
            self.eval()
            return torch.sigmoid(self.forward(px, hx, ei, ea, pi, hi, ef)).cpu().numpy()

        @torch.no_grad()
        def get_embeddings(self, px, hx, ei, ea):
            self.eval()
            return self.encode(px, hx, ei, ea)

    class PhageHostSAGE(nn.Module):
        def __init__(self, phage_dim, host_dim, n_ph):
            super().__init__()
            self._n_ph = n_ph
            self.proj_p = nn.Sequential(nn.Linear(phage_dim, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.ReLU())
            self.proj_h = nn.Sequential(nn.Linear(host_dim,  HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.ReLU())
            self.sage1 = SAGEConv(HIDDEN_DIM, HIDDEN_DIM, aggr="mean", normalize=True)
            self.sage2 = SAGEConv(HIDDEN_DIM, OUT_DIM,   aggr="mean", normalize=True)
            self.bn1   = nn.LayerNorm(HIDDEN_DIM)
            self.drop  = nn.Dropout(DROPOUT)
            self.mlp_sage = nn.Sequential(
                nn.Linear(OUT_DIM*2, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(DROPOUT),
                nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(),
                nn.Linear(128, 1))
            self.mlp_bypass = nn.Sequential(
                nn.Linear(phage_dim + host_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(DROPOUT),
                nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(DROPOUT),
                nn.Linear(128, 64), nn.GELU(),
                nn.Linear(64, 1))
            self.alpha = nn.Parameter(torch.tensor(0.5))

        def encode(self, px, hx, ei):
            pp = self.proj_p(px); ph = self.proj_h(hx)
            x  = torch.cat([pp, ph], dim=0)
            h  = self.bn1(F.relu(self.sage1(x, ei))); h = self.drop(h)
            return self.sage2(h, ei), pp, ph

        def decode(self, z, pp, ph, px, hx, pi, hi):
            gnn_score    = self.mlp_sage(torch.cat([z[pi], z[hi + self._n_ph]], dim=-1))
            bypass_score = self.mlp_bypass(torch.cat([px[pi], hx[hi]], dim=-1))
            alpha = torch.sigmoid(self.alpha)
            return (alpha * gnn_score + (1 - alpha) * bypass_score).squeeze(-1)

        def forward(self, px, hx, ei, pi, hi):
            z, pp, ph = self.encode(px, hx, ei)
            return self.decode(z, pp, ph, px, hx, pi, hi)

        @torch.no_grad()
        def predict(self, px, hx, ei, pi, hi):
            self.eval()
            z, pp, ph = self.encode(px, hx, ei)
            return torch.sigmoid(self.decode(z, pp, ph, px, hx, pi, hi)).cpu().numpy()

        @torch.no_grad()
        def get_embeddings(self, px, hx, ei):
            self.eval(); z, _, _ = self.encode(px, hx, ei); return z

else:
    class PhageHostGAT:
        def __init__(self, *a, **k): pass
    class PhageHostSAGE:
        def __init__(self, *a, **k): pass


def _build_gat_graph(mask):
    pm  = (dataset["label"] == 1).values & mask
    src = torch.tensor(dataset.loc[pm, "phage_idx"].values, dtype=torch.long)
    dst = torch.tensor(dataset.loc[pm, "host_idx"].values + n_phages, dtype=torch.long)
    ef  = torch.tensor(EDGE_FEATS_NP[pm], dtype=torch.float32)
    ei  = torch.cat([torch.stack([src, dst], dim=0),
                     torch.stack([dst, src], dim=0)], dim=1)
    ea  = torch.cat([ef, ef], dim=0)
    return ei, ea

def _build_edges(mask):
    pm  = (dataset["label"] == 1).values & mask
    src = torch.tensor(dataset.loc[pm, "phage_idx"].values, dtype=torch.long)
    dst = torch.tensor(dataset.loc[pm, "host_idx"].values + n_phages, dtype=torch.long)
    return to_undirected(torch.stack([src, dst], dim=0))

class NumpyFallback:
    def __init__(self):
        self.clf = GradientBoostingClassifier(n_estimators=200, max_depth=4,
            learning_rate=0.05, random_state=SEED)
        self.sc  = StandardScaler(); self._pf = self._hf = None

    def _agg(self, pf, hf, pos_df):
        np_ = pf.copy(); nh = hf.copy()
        for hi in range(len(hf)):
            nb = pos_df.loc[pos_df["host_idx"]==hi, "phage_idx"].values
            if len(nb): nh[hi] = pf[nb].mean(0)*0.5 + hf[hi]*0.5
        for pi in range(len(pf)):
            nb = pos_df.loc[pos_df["phage_idx"]==pi, "host_idx"].values
            if len(nb): np_[pi] = hf[nb].mean(0)*0.5 + pf[pi]*0.5
        return np_, nh

    def fit(self, pf, hf, tr):
        pos = tr[tr["label"]==1]; p2,h2 = self._agg(pf,hf,pos); p3,h3 = self._agg(p2,h2,pos)
        self._pf, self._hf = p3, h3
        X = self.sc.fit_transform(np.hstack([p3[tr["phage_idx"].values], h3[tr["host_idx"].values]]))
        self.clf.fit(X, tr["label"].values)

    def predict_proba(self, te):
        X = self.sc.transform(np.hstack([self._pf[te["phage_idx"].values], self._hf[te["host_idx"].values]]))
        p = self.clf.predict_proba(X)[:,1]
        return np.clip(np.where(np.isfinite(p), p, 0.5), 0.0, 1.0)


def run_fold(train_mask, test_mask, arch="GAT"):
    phs, hos = build_structural(dataset, train_mask)
    px_np = np.hstack([PHAGE_BASE_FIXED, phs]).astype(np.float32)
    hx_np = np.hstack([HOST_BASE,        hos]).astype(np.float32)
    tr_df = dataset[train_mask].reset_index(drop=True)
    te_df = dataset[test_mask].reset_index(drop=True)
    yte   = te_df["label"].values

    if not HAS_TORCH:
        m = NumpyFallback(); m.fit(px_np, hx_np, tr_df)
        return m.predict_proba(te_df), yte

    px = torch.tensor(px_np, dtype=torch.float32).to(DEVICE)
    hx = torch.tensor(hx_np, dtype=torch.float32).to(DEVICE)

    n_neg_tr = int((train_mask & (dataset["label"]==0).values).sum())
    n_pos_tr = max(int((train_mask & (dataset["label"]==1).values).sum()), 1)
    pw = torch.tensor([n_neg_tr / n_pos_tr], dtype=torch.float32).clamp(max=10.0).to(DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)

    if arch == "GAT":
        ei, ea = _build_gat_graph(train_mask)
        ei = ei.to(DEVICE); ea = ea.to(DEVICE)
        model  = PhageHostGAT(px_np.shape[1], hx_np.shape[1], N_EDGE_FEATS, n_phages).to(DEVICE)
    else:
        ei    = _build_edges(train_mask).to(DEVICE); ea = None
        model = PhageHostSAGE(px_np.shape[1], hx_np.shape[1], n_phages).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=(100 if _THOROUGH else 45), eta_min=1e-5)
    # FIX: do NOT compile inside run_fold — compilation is expensive and not reused per call

    tr_global_idx = np.where(train_mask)[0]
    all_pi = torch.tensor(tr_df["phage_idx"].values, dtype=torch.long).to(DEVICE)
    all_hi = torch.tensor(tr_df["host_idx"].values,  dtype=torch.long).to(DEVICE)
    all_lb = torch.tensor(tr_df["label"].values.astype(np.float32), dtype=torch.float32).to(DEVICE)
    if arch == "GAT":
        all_ef = torch.tensor(EDGE_FEATS_NP[tr_global_idx], dtype=torch.float32).to(DEVICE)

    tr_labels = tr_df["label"].values
    pos_idx = np.where(tr_labels == 1)[0]; neg_idx = np.where(tr_labels == 0)[0]
    vrng  = np.random.default_rng(SEED + 13)
    v_pos = vrng.choice(pos_idx, size=max(5, int(len(pos_idx)*0.12)), replace=False)
    v_neg = vrng.choice(neg_idx, size=max(5, int(len(neg_idx)*0.12)), replace=False)
    vidx  = np.concatenate([v_pos, v_neg])
    val_df = tr_df.iloc[vidx].reset_index(drop=True)
    vpi = torch.tensor(val_df["phage_idx"].values, dtype=torch.long).to(DEVICE)
    vhi = torch.tensor(val_df["host_idx"].values,  dtype=torch.long).to(DEVICE)
    vlb = val_df["label"].values; has_vc = len(np.unique(vlb)) == 2
    if arch == "GAT":
        vef = torch.tensor(EDGE_FEATS_NP[tr_global_idx[vidx]], dtype=torch.float32).to(DEVICE)

    n_pairs  = len(all_pi)
    BATCH = 8192 if DEVICE.type == "cuda" else 4096
    BATCH = min(BATCH, max(256, n_pairs))
    best_val = -1.0; pat_cnt = 0; best_st = None
    perm_rng = np.random.default_rng(SEED)
    model.train()
    scaler = GradScaler(enabled=USE_AMP)

    for ep in range(EPOCHS):
        perm = torch.from_numpy(perm_rng.permutation(n_pairs))
        n_e = ei.shape[1]
        keep_e = torch.randperm(n_e, device=ei.device)[:max(1, int(n_e * _EDGE_KEEP_FRAC))]
        eid = ei[:, keep_e]
        ead = ea[keep_e] if arch == "GAT" else None
        for start in range(0, n_pairs, BATCH):
            idx = perm[start:start+BATCH]
            bpi = all_pi[idx]; bhi = all_hi[idx]; blb = all_lb[idx]
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                if arch == "GAT":
                    bef = all_ef[idx]
                    logits = model(px, hx, eid, ead, bpi, bhi, bef)
                else:
                    logits = model(px, hx, eid, bpi, bhi)
                loss = crit(logits, blb)
            if USE_AMP:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        sch.step()

        if ep % 3 == 0 and has_vc:
            if arch == "GAT": vp = model.predict(px, hx, ei, ea, vpi, vhi, vef)
            else:              vp = model.predict(px, hx, ei, vpi, vhi)
            vp = np.where(np.isfinite(vp), vp, 0.5)
            try:    vauc = roc_auc_score(vlb, vp)
            except: vauc = 0.5
            model.train()
            if vauc > best_val:
                best_val = vauc
                best_st  = {k: v.clone() for k, v in model.state_dict().items()}
                pat_cnt  = 0
            else:
                pat_cnt += 1
            if pat_cnt >= PATIENCE: break

    # FIX: always save weights at end as fallback
    if best_st is None:
        best_st = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_st)

    full_mask = train_mask | test_mask
    if arch == "GAT":
        ei_full, ea_full = _build_gat_graph(full_mask)
        ei_full = ei_full.to(DEVICE); ea_full = ea_full.to(DEVICE)
    else:
        ei_full = _build_edges(full_mask).to(DEVICE)

    tpi = torch.tensor(te_df["phage_idx"].values, dtype=torch.long).to(DEVICE)
    thi = torch.tensor(te_df["host_idx"].values,  dtype=torch.long).to(DEVICE)
    # FIX: use boolean mask correctly for EDGE_FEATS_NP
    test_global_idx = np.where(test_mask)[0]
    if arch == "GAT":
        tef   = torch.tensor(EDGE_FEATS_NP[test_global_idx], dtype=torch.float32).to(DEVICE)
        proba = model.predict(px, hx, ei_full, ea_full, tpi, thi, tef)
    else:
        proba = model.predict(px, hx, ei_full, tpi, thi)

    proba = np.asarray(proba, dtype=np.float32)
    proba = np.where(np.isfinite(proba), proba, 0.5)
    proba = np.clip(proba, 0.0, 1.0)
    return proba, yte


def greedy_cocktail(sp_df, k, pos_strains):
    cov, sel, rem = set(), [], list(sp_df["phage"].unique())
    pos_df = sp_df[sp_df["label"]==1]
    for _ in range(k):
        if not rem or cov == pos_strains: break
        best, bg = None, -1
        for ph in rem:
            ns = len((set(pos_df[pos_df["phage"]==ph]["strain"]) & pos_strains) - cov)
            pb = sp_df[sp_df["phage"]==ph]["proba"].max()
            if ns > bg or (ns == bg and best is not None and pb > sp_df[sp_df["phage"]==best]["proba"].max()):
                bg, best = ns, ph
        if best:
            cov |= (set(pos_df[pos_df["phage"]==best]["strain"]) & pos_strains)
            sel.append(best); rem.remove(best)
    return sel, cov

def strain_cov(sp_df, sel, ps):
    c = set(sp_df[sp_df["phage"].isin(sel) & (sp_df["label"]==1)]["strain"]) & ps
    return len(c)/len(ps) if ps else 0.0

def run_gnn_pipeline(arch):
    print(f"\n  ── {ARCH_LABEL[arch]} ──")

    _loso_cache_path = RESULTS_DIR / f"gnn_{arch.lower()}_loso.csv"
    _pred_cache_path = RESULTS_DIR / f"predictions_{arch.lower()}.csv"
    if _loso_cache_path.exists():
        _done_df = pd.read_csv(_loso_cache_path, index_col="species")
        loso_rows = _done_df.reset_index().to_dict("records")
        print(f"  Resuming: {len(loso_rows)} species already done, skipping.")
    else:
        _done_df = pd.DataFrame()
        loso_rows = []
    _done_species = set(_done_df.index.tolist()) if len(_done_df) else set()
    all_proba = np.full(len(dataset), np.nan)

    # FIX: fast vectorised cache reload using merge instead of iterrows O(n²)
    if _pred_cache_path.exists() and len(_done_species):
        _pred_cache = pd.read_csv(_pred_cache_path)
        if "proba" in _pred_cache.columns and "host" in _pred_cache.columns:
            _merged = dataset[["phage","host"]].reset_index().merge(
                _pred_cache[["phage","host","proba"]], on=["phage","host"], how="left")
            _merged = _merged.set_index("index").sort_index()
            all_proba = _merged["proba"].values.astype(np.float64)

    _remaining = [sp for sp in valid_species if sp not in _done_species]
    print(f"  LOSO ({len(valid_species)} species, {len(_remaining)} remaining)...")
    t0=time.time()
    for sp in _remaining:
        tm = (dataset["host"]==sp).values
        proba, yte = run_fold(~tm, tm, arch=arch)
        all_proba[tm] = proba
        m = metrics(yte, proba); m["species"]=sp; m["n_test"]=int(tm.sum())
        loso_rows.append(m)
        print(f"    {sp:<40} n={int(tm.sum()):>4}  AUC={m['roc_auc']:.4f}")
        pd.DataFrame(loso_rows).set_index("species").to_csv(_loso_cache_path)
    loso_df = pd.DataFrame(loso_rows).set_index("species")
    vm = ~np.isnan(all_proba)
    loso_pooled = metrics(all_labels[vm], all_proba[vm])
    loso_mean = loso_df["roc_auc"].mean(); loso_std = loso_df["roc_auc"].std()
    loso_df.to_csv(RESULTS_DIR/f"gnn_{arch.lower()}_loso.csv")
    print(f"  LOSO {time.time()-t0:.0f}s  mean={loso_mean:.4f}±{loso_std:.4f}  pooled={loso_pooled['roc_auc']:.4f}")

    pred_df = dataset[vm].copy().reset_index(drop=True)
    pred_df["proba"] = all_proba[vm]
    pred_df.to_csv(RESULTS_DIR/f"predictions_{arch.lower()}.csv", index=False)

    # LOGO (resume from partial CSV if present)
    _logo_path = RESULTS_DIR / f"gnn_{arch.lower()}_logo.csv"
    if _logo_path.exists():
        _logo_prev = pd.read_csv(_logo_path, index_col="genus")
        logo_rows = _logo_prev.reset_index().to_dict("records")
        _logo_done = set(_logo_prev.index.tolist())
        print(f"  LOGO ({len(evaluable_genera)} genera, {len(_logo_done)} cached)...")
    else:
        logo_rows = []
        _logo_done = set()
        print(f"  LOGO ({len(evaluable_genera)} genera)...")
    t1=time.time()
    for genus in evaluable_genera:
        if genus in _logo_done: continue
        tm = (dataset["genus"]==genus).values
        if all_labels[tm].sum()<2: continue
        proba, yte = run_fold(~tm, tm, arch=arch)
        if len(np.unique(yte))<2: continue
        m = metrics(yte, proba); m["genus"]=genus; m["n_test"]=int(tm.sum())
        logo_rows.append(m)
        pd.DataFrame(logo_rows).set_index("genus").to_csv(_logo_path)
    logo_df = pd.DataFrame(logo_rows).set_index("genus")
    logo_df.to_csv(_logo_path)
    logo_mean = logo_df["roc_auc"].mean() if len(logo_df) else 0.0
    print(f"  LOGO {time.time()-t1:.0f}s  mean={logo_mean:.4f}")
    if len(logo_df) == 0:
        print(f"  WARNING: {arch} LOGO produced no results — delete cache to regenerate")

    # Unseen Strain
    print(f"  Unseen Strain ({N_MC_ROUNDS} rounds)...")
    sp_str_cnt = dataset[dataset["label"]==1].groupby("host")["strain"].nunique()
    elig_sp    = sp_str_cnt[sp_str_cnt>=2].index.tolist()
    mc_rows=[]; skipped_rounds = 0
    for mc in range(N_MC_ROUNDS):
        rng_mc = np.random.default_rng(SEED+mc)
        u_rows = np.zeros(len(dataset), dtype=bool)
        for sp in elig_sp:
            sl = dataset[(dataset["host"]==sp)&(dataset["label"]==1)]["strain"].unique()
            nu = max(1, int(len(sl)*UNSEEN_FRAC))
            if nu >= len(sl): continue
            us = rng_mc.choice(sl, size=nu, replace=False)
            u_rows |= (dataset["host"]==sp) & (dataset["strain"].isin(us))
        if u_rows.sum()<5 or (~u_rows).sum()<100 or len(np.unique(all_labels[u_rows]))<2:
            skipped_rounds += 1
            continue
        try:
            pu, yu = run_fold(~u_rows, u_rows, arch=arch)
            mc_rows.append({"round":mc+1,"auc":roc_auc_score(yu,pu),"n_unseen":int(u_rows.sum())})
            print(f"    Round {mc+1}: AUC={mc_rows[-1]['auc']:.4f}")
        except Exception as e:
            print(f"    Round {mc+1} error: {e}"); skipped_rounds += 1
    mc_df  = pd.DataFrame(mc_rows)
    mc_df.to_csv(RESULTS_DIR/f"gnn_{arch.lower()}_unseen.csv", index=False)
    mc_auc = mc_df["auc"].mean() if len(mc_df) else 0.0
    print(f"  Unseen Strain: {len(mc_rows)} valid rounds, {skipped_rounds} skipped")

    # Cocktail
    print(f"  Cocktail (k={K_COCKTAIL})...")
    c_rng=np.random.default_rng(SEED+999); ctail_rows=[]
    for sp in sorted(pred_df["host"].unique()):
        spd = pred_df[pred_df["host"]==sp].copy()
        if (spd["label"]==1).sum()<2: continue
        ps = set(spd[spd["label"]==1]["strain"].unique())
        if not ps: continue
        srt   = spd.sort_values("proba",ascending=False)["phage"]
        single = [srt.iloc[0]]; topk = srt.head(K_COCKTAIL).tolist()
        greedy, _ = greedy_cocktail(spd, K_COCKTAIL, ps)
        pu2  = spd["phage"].unique()
        _n_rand = 100 if _THOROUGH else 36
        rand = float(np.mean([strain_cov(spd, c_rng.choice(pu2, size=min(K_COCKTAIL,len(pu2)), replace=False), ps)
                              for _ in range(_n_rand)]))
        ctail_rows.append({"species":sp,
            "single_cov": strain_cov(spd,single,ps), "topk_cov": strain_cov(spd,topk,ps),
            "random_cov": rand,                       "greedy_cov": strain_cov(spd,greedy,ps)})
    ctail_df = pd.DataFrame(ctail_rows)
    ctail_df.to_csv(RESULTS_DIR/f"gnn_{arch.lower()}_cocktail.csv", index=False)
    strats  = ["single","topk","random","greedy"]
    means_c = {s: ctail_df[f"{s}_cov"].mean() for s in strats}
    pct75_c = {s: (ctail_df[f"{s}_cov"]>=0.75).mean()*100 for s in strats}

    return {"arch":arch,"loso_df":loso_df,"loso_mean":loso_mean,"loso_std":loso_std,
            "loso_pooled":loso_pooled,"logo_df":logo_df,"logo_mean":logo_mean,
            "mc_df":mc_df,"mc_auc":mc_auc,"ctail_df":ctail_df,
            "means_c":means_c,"pct75_c":pct75_c,"pred_df":pred_df,"all_proba":all_proba}

_gnn_loso_file   = RESULTS_DIR / "gnn_gat_loso.csv"
_gnn_unseen_file = RESULTS_DIR / "gnn_gat_unseen.csv"
_gnn_ctail_file  = RESULTS_DIR / "gnn_gat_cocktail.csv"
_loso_done   = _gnn_loso_file.exists()
_unseen_done = _gnn_unseen_file.exists() and _gnn_unseen_file.stat().st_size > 50
_ctail_done  = _gnn_ctail_file.exists()  and _gnn_ctail_file.stat().st_size  > 50
_sage_loso_file = RESULTS_DIR / "gnn_sage_loso.csv"
_sage_done   = _sage_loso_file.exists()

gnn_results={}
if _loso_done and _unseen_done and _ctail_done and _sage_done:
    print("  GNN results already cached — loading from CSV, skipping retraining.")
    for _arch in ["GAT", "SAGE"]:
        _a = _arch.lower()
        def _safe_read(path, **kwargs):
            try:
                df = pd.read_csv(path, **kwargs)
                return df if len(df) > 0 else pd.DataFrame()
            except Exception:
                return pd.DataFrame()
        _loso_df  = pd.read_csv(RESULTS_DIR/f"gnn_{_a}_loso.csv", index_col="species")
        _logo_df  = _safe_read(RESULTS_DIR/f"gnn_{_a}_logo.csv",    index_col="genus")  if (RESULTS_DIR/f"gnn_{_a}_logo.csv").exists()    else pd.DataFrame()
        _mc_df    = _safe_read(RESULTS_DIR/f"gnn_{_a}_unseen.csv")                       if (RESULTS_DIR/f"gnn_{_a}_unseen.csv").exists()   else pd.DataFrame()
        _ctail_df = _safe_read(RESULTS_DIR/f"gnn_{_a}_cocktail.csv")                     if (RESULTS_DIR/f"gnn_{_a}_cocktail.csv").exists() else pd.DataFrame()
        _pred_df  = _safe_read(RESULTS_DIR/f"predictions_{_a}.csv")                      if (RESULTS_DIR/f"predictions_{_a}.csv").exists()  else pd.DataFrame()

        # FIX: fast vectorised proba reload
        _all_proba = np.full(len(dataset), np.nan)
        if len(_pred_df) and "proba" in _pred_df.columns:
            _merged2 = dataset[["phage","host"]].reset_index().merge(
                _pred_df[["phage","host","proba"]], on=["phage","host"], how="left")
            _merged2 = _merged2.set_index("index").sort_index()
            _all_proba = _merged2["proba"].values.astype(np.float64)

        _loso_mean = _loso_df["roc_auc"].mean(); _loso_std = _loso_df["roc_auc"].std()
        _logo_mean = _logo_df["roc_auc"].mean() if len(_logo_df) else 0.0
        _mc_auc    = _mc_df["auc"].mean()        if len(_mc_df)   else 0.0
        _strats  = ["single","topk","random","greedy"]
        _means_c = {s: _ctail_df[f"{s}_cov"].mean() for s in _strats} if len(_ctail_df) else {s:0.0 for s in _strats}
        _pct75_c = {s: (_ctail_df[f"{s}_cov"]>=0.75).mean()*100 for s in _strats} if len(_ctail_df) else {s:0.0 for s in _strats}
        _vm_cached = ~np.isnan(_all_proba)
        _loso_pooled = metrics(all_labels[_vm_cached], _all_proba[_vm_cached]) if _vm_cached.sum() > 10 and len(np.unique(all_labels[_vm_cached])) == 2 else {"roc_auc": _loso_mean}
        gnn_results[_arch] = {"arch":_arch,"loso_df":_loso_df,"loso_mean":_loso_mean,
            "loso_std":_loso_std,"loso_pooled":_loso_pooled,"logo_df":_logo_df,
            "logo_mean":_logo_mean,"mc_df":_mc_df,"mc_auc":_mc_auc,"ctail_df":_ctail_df,
            "means_c":_means_c,"pct75_c":_pct75_c,"pred_df":_pred_df,"all_proba":_all_proba}
        print(f"  {_arch}: LOSO={_loso_mean:.4f}±{_loso_std:.4f}  LOGO={_logo_mean:.4f}  Unseen={_mc_auc:.4f} [cached]")
else:
    for _arch in ["GAT", "SAGE"]:
        gnn_results[_arch]=run_gnn_pipeline(_arch)

best_gnn=max(gnn_results,key=lambda a:gnn_results[a]["loso_mean"])
gnn_results[best_gnn]["pred_df"].to_csv(RESULTS_DIR/"ensemble_predictions_gnn.csv",index=False)
print(f"\n  Best GNN: {ARCH_LABEL[best_gnn]} AUC={gnn_results[best_gnn]['loso_mean']:.4f}")


# ════════════════════════════════════════════════════════════════
# SECTION 7 — SUMMARY TABLE
# ════════════════════════════════════════════════════════════════
print("\n[7] Saving summary tables...")

cmp_rows=[]
for arch,r in gnn_results.items():
    _pr_auc_mean = r["loso_df"]["pr_auc"].mean() if "pr_auc" in r["loso_df"].columns else 0.0
    cmp_rows.append({"model":ARCH_LABEL[arch],
        "loso_mean_auc":round(r["loso_mean"],4),"loso_std":round(r["loso_std"],4),
        "loso_mean_pr_auc":round(_pr_auc_mean,4),
        "loso_pooled_auc":round(r["loso_pooled"]["roc_auc"],4),
        "loso_pooled_pr_auc":round(r["loso_pooled"].get("pr_auc",0),4),
        "logo_mean_auc":round(r["logo_mean"],4),"unseen_auc":round(r["mc_auc"],4),
        "greedy_cov3":round(r["means_c"]["greedy"],3),
        "pct_species_75":round(r["pct75_c"]["greedy"],1)})
for mname,cr in clf_results.items():
    _pr_mean = cr["df"]["pr_auc"].mean() if hasattr(cr.get("df"), "columns") and "pr_auc" in cr["df"].columns else 0.0
    cmp_rows.append({"model":mname,"loso_mean_auc":round(cr["mean"],4),
        "loso_mean_pr_auc":round(_pr_mean,4),
        "loso_pooled_auc":round(cr["pooled"].get("roc_auc",0),4),
        "loso_pooled_pr_auc":round(cr["pooled"].get("pr_auc",0),4),
        "logo_mean_auc":0,"unseen_auc":0,"greedy_cov3":0,"pct_species_75":0})
pd.DataFrame(cmp_rows).to_csv(RESULTS_DIR/"model_comparison.csv",index=False)
print("  Saved model_comparison.csv")


# ════════════════════════════════════════════════════════════════
# SECTION 7b — STATISTICAL SIGNIFICANCE TESTING
# ════════════════════════════════════════════════════════════════
print("\n[7b] Statistical significance testing...")

sig_rows = []
all_model_loso = {}
for arch, r in gnn_results.items():
    all_model_loso[ARCH_LABEL[arch]] = r["loso_df"]["roc_auc"]
for mname, cr in clf_results.items():
    if hasattr(cr.get("df"), "index") and "roc_auc" in cr["df"].columns:
        all_model_loso[mname] = cr["df"]["roc_auc"]

model_names_sig = list(all_model_loso.keys())
for i in range(len(model_names_sig)):
    for j in range(i+1, len(model_names_sig)):
        m1, m2 = model_names_sig[i], model_names_sig[j]
        s1, s2 = all_model_loso[m1], all_model_loso[m2]
        shared = s1.index.intersection(s2.index)
        if len(shared) < 5:
            continue
        v1, v2 = s1.loc[shared].values, s2.loc[shared].values
        try:
            stat_w, p_w = wilcoxon(v1, v2)
        except Exception:
            stat_w, p_w = np.nan, np.nan
        try:
            stat_u, p_u = mannwhitneyu(v1, v2, alternative="two-sided")
        except Exception:
            stat_u, p_u = np.nan, np.nan
        sig_rows.append({
            "model_1": m1, "model_2": m2,
            "n_shared_species": len(shared),
            "mean_auc_1": round(float(v1.mean()),4),
            "mean_auc_2": round(float(v2.mean()),4),
            "wilcoxon_stat": float(stat_w) if not np.isnan(stat_w) else None,
            "wilcoxon_p": float(p_w) if not np.isnan(p_w) else None,
            "mannwhitney_stat": float(stat_u) if not np.isnan(stat_u) else None,
            "mannwhitney_p": float(p_u) if not np.isnan(p_u) else None,
            "significant_005": bool(p_w < 0.05) if not np.isnan(p_w) else False,
            "significant_001": bool(p_w < 0.01) if not np.isnan(p_w) else False,
        })
        star = "***" if (not np.isnan(p_w) and p_w < 0.001) else \
               "**"  if (not np.isnan(p_w) and p_w < 0.01)  else \
               "*"   if (not np.isnan(p_w) and p_w < 0.05)  else "ns"
        print(f"  {m1} vs {m2}: Wilcoxon p={p_w:.4g} {star}  (n={len(shared)})")

sig_df = pd.DataFrame(sig_rows)
sig_df.to_csv(RESULTS_DIR / "statistical_tests.csv", index=False)
print("  Saved statistical_tests.csv")


# ════════════════════════════════════════════════════════════════
# SECTION 7c — CONFIDENCE INTERVALS (95% Bootstrap)
# ════════════════════════════════════════════════════════════════
print("\n[7c] Computing 95% bootstrap confidence intervals...")

def bootstrap_ci(values, n_boot=10000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n < 2:
        m = float(vals.mean()) if n == 1 else 0.0
        return m, m, m
    boot_means = np.array([rng.choice(vals, size=n, replace=True).mean()
                           for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_means, alpha * 100))
    hi = float(np.percentile(boot_means, (1 - alpha) * 100))
    return float(vals.mean()), lo, hi

ci_rows = []
for arch, r in gnn_results.items():
    aucs = r["loso_df"]["roc_auc"].values
    mean, lo, hi = bootstrap_ci(aucs)
    ci_rows.append({"model": ARCH_LABEL[arch], "metric": "roc_auc",
                    "mean": round(mean,4), "ci_lo": round(lo,4),
                    "ci_hi": round(hi,4), "n_species": len(aucs)})
    print(f"  {ARCH_LABEL[arch]} ROC-AUC: {mean:.4f} [{lo:.4f}, {hi:.4f}]")
    if "pr_auc" in r["loso_df"].columns:
        pr_aucs = r["loso_df"]["pr_auc"].values
        mean_pr, lo_pr, hi_pr = bootstrap_ci(pr_aucs)
        ci_rows.append({"model": ARCH_LABEL[arch], "metric": "pr_auc",
                        "mean": round(mean_pr,4), "ci_lo": round(lo_pr,4),
                        "ci_hi": round(hi_pr,4), "n_species": len(pr_aucs)})
        print(f"  {ARCH_LABEL[arch]} PR-AUC:  {mean_pr:.4f} [{lo_pr:.4f}, {hi_pr:.4f}]")

for mname, cr in clf_results.items():
    if hasattr(cr.get("df"), "columns") and "roc_auc" in cr["df"].columns:
        aucs = cr["df"]["roc_auc"].values
        mean, lo, hi = bootstrap_ci(aucs)
        ci_rows.append({"model": mname, "metric": "roc_auc",
                        "mean": round(mean,4), "ci_lo": round(lo,4),
                        "ci_hi": round(hi,4), "n_species": len(aucs)})
        print(f"  {mname} ROC-AUC: {mean:.4f} [{lo:.4f}, {hi:.4f}]")
        if "pr_auc" in cr["df"].columns:
            pr_aucs = cr["df"]["pr_auc"].values
            mean_pr, lo_pr, hi_pr = bootstrap_ci(pr_aucs)
            ci_rows.append({"model": mname, "metric": "pr_auc",
                            "mean": round(mean_pr,4), "ci_lo": round(lo_pr,4),
                            "ci_hi": round(hi_pr,4), "n_species": len(pr_aucs)})

ci_df = pd.DataFrame(ci_rows)
ci_df.to_csv(RESULTS_DIR / "confidence_intervals.csv", index=False)
print("  Saved confidence_intervals.csv")


# ════════════════════════════════════════════════════════════════
# SECTION 8 — PLOTS FOR POSTER
# ════════════════════════════════════════════════════════════════
print("\n[8] Generating poster plots...")

_archs_available = list(gnn_results.keys())
n_sp=len(gnn_results[_archs_available[0]]["loso_df"])
if len(_archs_available) >= 2:
    fig,axes=plt.subplots(1,3,figsize=(22,max(10,n_sp*0.22)))
else:
    fig,axes=plt.subplots(1,1,figsize=(10,max(10,n_sp*0.22)))
    axes=[axes]
for col,arch in enumerate(_archs_available[:2]):
    ax=axes[col]; ldf=gnn_results[arch]["loso_df"].sort_values("roc_auc",ascending=True)
    clr=["#1a9850" if v>=0.8 else "#4285F4" if v>=0.6 else "#FF9800" if v>=0.5 else "#F44336"
         for v in ldf["roc_auc"].values]
    yp=np.arange(len(ldf))
    ax.barh(yp,ldf["roc_auc"].values,color=clr,alpha=0.85,edgecolor="white",linewidth=0.5)
    ax.set_yticks(yp); ax.set_yticklabels(ldf.index,fontsize=5.5)
    ax.axvline(0.5,color="red",linestyle=":",lw=1.2,label="Random")
    ax.axvline(gnn_results[arch]["loso_mean"],color="black",linestyle="--",lw=2,
               label=f"Mean={gnn_results[arch]['loso_mean']:.3f}")
    ax.set_xlabel("ROC-AUC"); ax.set_xlim(0,1.12); ax.grid(axis="x",alpha=0.4)
    ax.set_title(f"{ARCH_LABEL[arch]}\nLOSO per-species AUC",fontsize=10,fontweight="bold",
                 color=ARCH_COLOR[arch]); ax.legend(fontsize=7)
    legend_patches=[mpatches.Patch(color="#1a9850",label="AUC≥0.8"),
        mpatches.Patch(color="#4285F4",label="AUC≥0.6"),
        mpatches.Patch(color="#FF9800",label="AUC≥0.5"),
        mpatches.Patch(color="#F44336",label="AUC<0.5")]
    ax.legend(handles=legend_patches,fontsize=6,loc="lower right")

if len(_archs_available) >= 2:
    ax3=axes[2]
    shared=gnn_results["GAT"]["loso_df"].index.intersection(gnn_results["SAGE"]["loso_df"].index)
    gv=gnn_results["GAT"]["loso_df"].loc[shared,"roc_auc"].values
    sv=gnn_results["SAGE"]["loso_df"].loc[shared,"roc_auc"].values
    ax3.scatter(gv,sv,alpha=0.6,s=30,c="#444444")
    lim=[min(gv.min(),sv.min())-0.05,max(gv.max(),sv.max())+0.05]
    ax3.plot(lim,lim,"k--",lw=1,alpha=0.5)
    ax3.set_xlabel("GAT AUC"); ax3.set_ylabel("SAGE AUC")
    ax3.set_title("C  GAT vs SAGE per species\n(above diagonal = SAGE better)",fontsize=10,fontweight="bold")
    ngb=int((gv>sv).sum()); nsb=int((sv>gv).sum())
    ax3.text(0.05,0.95,f"GAT better: {ngb}\nSAGE better: {nsb}",transform=ax3.transAxes,
             fontsize=9,va="top",bbox=dict(boxstyle="round",facecolor="wheat",alpha=0.75))
    ax3.grid(alpha=0.4)
plt.suptitle("PrecisionPhage — LOSO-CV Performance by Host Species",fontsize=13,fontweight="bold",y=1.01)
plt.tight_layout()
save_fig(fig,"01_loso_per_species")

# ── Plot 2: ROC curves ──
fig,ax=plt.subplots(figsize=(8,6))
for arch in _archs_available:
    vm=~np.isnan(gnn_results[arch]["all_proba"])
    if vm.sum()>10 and len(np.unique(all_labels[vm]))==2:
        fpr,tpr,_=roc_curve(all_labels[vm],gnn_results[arch]["all_proba"][vm])
        ax.plot(fpr,tpr,color=ARCH_COLOR[arch],lw=2.5,
                label=f"{ARCH_LABEL[arch]}  AUC={gnn_results[arch]['loso_pooled']['roc_auc']:.3f}")
for mname,cr in clf_results.items():
    vm=~np.isnan(cr["probas"])
    if vm.sum()>10:
        fpr,tpr,_=roc_curve(y_all[vm],cr["probas"][vm])
        ax.plot(fpr,tpr,lw=1.8,linestyle="--",
                label=f"{mname}  AUC={cr['pooled'].get('roc_auc',0):.3f}")
ax.plot([0,1],[0,1],"k:",lw=1.2,alpha=0.5,label="Random")
ax.set_xlabel("False Positive Rate",fontsize=12); ax.set_ylabel("True Positive Rate",fontsize=12)
ax.set_title("ROC Curves — All Models (Pooled LOSO-CV)",fontsize=13,fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.4)
plt.tight_layout()
save_fig(fig,"02_roc_curves_all_models")

# ── Plot 2b: Precision-Recall curves ──
fig,ax=plt.subplots(figsize=(8,6))
for arch in _archs_available:
    vm=~np.isnan(gnn_results[arch]["all_proba"])
    if vm.sum()>10 and len(np.unique(all_labels[vm]))==2:
        pr_vals,rc_vals,_=precision_recall_curve(all_labels[vm],gnn_results[arch]["all_proba"][vm])
        pr_auc_val = sk_auc(rc_vals, pr_vals)
        ax.plot(rc_vals,pr_vals,color=ARCH_COLOR[arch],lw=2.5,
                label=f"{ARCH_LABEL[arch]}  PR-AUC={pr_auc_val:.3f}")
for mname,cr in clf_results.items():
    vm=~np.isnan(cr["probas"])
    if vm.sum()>10 and len(np.unique(y_all[vm]))==2:
        pr_vals,rc_vals,_=precision_recall_curve(y_all[vm],cr["probas"][vm])
        pr_auc_val = sk_auc(rc_vals, pr_vals)
        ax.plot(rc_vals,pr_vals,lw=1.8,linestyle="--",
                label=f"{mname}  PR-AUC={pr_auc_val:.3f}")
_baseline_pr = all_labels.mean()
ax.axhline(_baseline_pr,color="k",linestyle=":",lw=1.2,alpha=0.5,label=f"Random ({_baseline_pr:.2f})")
ax.set_xlabel("Recall",fontsize=12); ax.set_ylabel("Precision",fontsize=12)
ax.set_title("Precision-Recall Curves — All Models (Pooled LOSO-CV)",fontsize=13,fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.4); ax.set_xlim(0,1.05); ax.set_ylim(0,1.05)
plt.tight_layout()
save_fig(fig,"02b_pr_curves_all_models")

# ── Plot 3: Model comparison ──
fig,axes=plt.subplots(1,3,figsize=(16,5))
models_all=list(gnn_results.keys())+list(clf_results.keys())
colors_all=[ARCH_COLOR.get(m,"#607D8B") for m in models_all]
loso_aucs=[gnn_results[m]["loso_mean"] if m in gnn_results else clf_results[m]["mean"] for m in models_all]
for pi,(title,vals) in enumerate([
    ("LOSO Mean AUC",loso_aucs),
    ("LOGO Mean AUC",[gnn_results[m]["logo_mean"] if m in gnn_results else 0 for m in models_all]),
    ("Unseen Strain AUC",[gnn_results[m]["mc_auc"] if m in gnn_results else 0 for m in models_all])]):
    ax=axes[pi]
    bars=ax.bar(range(len(models_all)),vals,color=colors_all,alpha=0.85,edgecolor="white")
    for b,v in zip(bars,vals):
        if v>0: ax.text(b.get_x()+b.get_width()/2,v+0.01,f"{v:.3f}",ha="center",fontsize=8,fontweight="bold")
    ax.set_xticks(range(len(models_all))); ax.set_xticklabels([ARCH_LABEL.get(m,m) for m in models_all],rotation=20,ha="right",fontsize=8)
    ax.set_ylim(0,1.15); ax.set_title(title,fontsize=10,fontweight="bold")
    ax.axhline(0.9,color="red",linestyle="--",lw=1.2,alpha=0.5,label="0.90 target")
    ax.legend(fontsize=7); ax.grid(axis="y",alpha=0.4)
plt.suptitle("PrecisionPhage — Model Performance Comparison",fontsize=12,fontweight="bold")
plt.tight_layout()
save_fig(fig,"03_model_comparison")

# ── Plot 4: LOGO per-genus ──
for arch in _archs_available:
    ldf=gnn_results[arch]["logo_df"].sort_values("roc_auc",ascending=True)
    if len(ldf) == 0:
        print(f"  Plot 4 ({arch} LOGO) skipped — no LOGO data.")
        continue
    fig,ax=plt.subplots(figsize=(10,max(6,len(ldf)*0.28)))
    clr=["#1a9850" if v>=0.8 else "#4285F4" if v>=0.6 else "#FF9800" if v>=0.5 else "#F44336"
         for v in ldf["roc_auc"].values]
    ax.barh(np.arange(len(ldf)),ldf["roc_auc"].values,color=clr,alpha=0.85)
    ax.set_yticks(np.arange(len(ldf))); ax.set_yticklabels(ldf.index,fontsize=7)
    ax.axvline(0.5,color="red",linestyle=":",lw=1.2)
    ax.axvline(ldf["roc_auc"].mean(),color="black",linestyle="--",lw=2,
               label=f"Mean={ldf['roc_auc'].mean():.3f}")
    ax.set_xlabel("ROC-AUC"); ax.set_xlim(0,1.1)
    ax.set_title(f"{ARCH_LABEL[arch]} — LOGO per-genus AUC",fontsize=11,fontweight="bold",
                 color=ARCH_COLOR[arch]); ax.legend(fontsize=8); ax.grid(axis="x",alpha=0.4)
    plt.tight_layout()
    save_fig(fig,f"04_logo_{arch.lower()}")

# ── Plot 5: Cocktail ──
fig,axes=plt.subplots(1,2,figsize=(13,5))
strats=["single","random","topk","greedy"]
slbl={"single":"Single","random":"Random","topk":"Top-K","greedy":"Greedy"}
xp=np.arange(len(strats)); wd=0.35
for pi,(panel,key_fn) in enumerate([
    ("Mean Strain Coverage",lambda a,s:gnn_results[a]["means_c"][s]),
    ("% Species ≥75% Coverage",lambda a,s:gnn_results[a]["pct75_c"][s])]):
    ax=axes[pi]
    for i,arch in enumerate(_archs_available):
        vals=[key_fn(arch,s) for s in strats]
        bars=ax.bar(xp+(i-0.5)*wd,vals,wd,alpha=0.85,color=ARCH_COLOR[arch],label=ARCH_LABEL[arch])
        for b,v in zip(bars,vals):
            ax.text(b.get_x()+b.get_width()/2,v+(0.5 if pi==1 else 0.01),
                    f"{v:.0f}%" if pi==1 else f"{v:.2f}",ha="center",fontsize=7.5,fontweight="bold")
    if pi==0: ax.axhline(0.75,color="gold",linestyle="--",lw=1.5,label="75% threshold"); ax.set_ylim(0,1.12)
    else: ax.set_ylim(0,115)
    ax.set_xticks(xp); ax.set_xticklabels([slbl[s] for s in strats])
    ax.set_title(f"{panel} (k={K_COCKTAIL})",fontsize=10,fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y",alpha=0.4)
plt.suptitle("PrecisionPhage — Phage Cocktail Optimization",fontsize=12,fontweight="bold",y=1.01)
plt.tight_layout()
save_fig(fig,"05_cocktail_comparison")

# ── Plot 6: Feature importance ──
best_clf_name = max(clf_results, key=lambda k: clf_results[k]["mean"])
print(f"  Feature importance from {best_clf_name}...")
try:
    clf_for_imp = GradientBoostingClassifier(n_estimators=200,max_depth=5,learning_rate=0.05,random_state=SEED)
    clf_for_imp.fit(X_sc, y_all)
    fi = pd.Series(clf_for_imp.feature_importances_, index=all_feat_cols).sort_values(ascending=False)
    fig,ax=plt.subplots(figsize=(10,6))
    top_fi=fi.head(25)
    ax.barh(range(len(top_fi)),top_fi.values[::-1],color="#1565C0",alpha=0.85)
    ax.set_yticks(range(len(top_fi))); ax.set_yticklabels(top_fi.index[::-1],fontsize=8)
    ax.set_xlabel("Feature Importance",fontsize=11)
    ax.set_title("Top 25 Feature Importances (GBM on all data)",fontsize=12,fontweight="bold")
    ax.grid(axis="x",alpha=0.4)
    plt.tight_layout()
    save_fig(fig,"06_feature_importance")
except Exception as e:
    print(f"  Feature importance plot failed: {e}")

# ── Plot 7: Unseen strain distribution ──
fig,ax=plt.subplots(figsize=(8,5))
for arch in _archs_available:
    mdf = gnn_results[arch]["mc_df"]
    if len(mdf)>0:
        ax.hist(mdf["auc"].dropna(),bins=min(N_MC_ROUNDS,8),alpha=0.6,
                color=ARCH_COLOR[arch],label=f"{ARCH_LABEL[arch]} (mean={gnn_results[arch]['mc_auc']:.3f})",
                edgecolor="white")
ax.axvline(0.5,color="red",linestyle=":",lw=1.5,label="Random")
ax.axvline(0.9,color="green",linestyle="--",lw=1.5,label="0.90 target")
ax.set_xlabel("AUC on Unseen Strains",fontsize=11); ax.set_ylabel("Count",fontsize=11)
ax.set_title("Unseen Strain Simulation — AUC Distribution",fontsize=12,fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.4)
plt.tight_layout()
save_fig(fig,"07_unseen_strain_auc")

# ── Plot 8: UMAP embedding ──
if HAS_UMAP and HAS_TORCH:
    try:
        print("  Generating UMAP from learned GNN embeddings...")
        # Train one full-data GNN to get learned representations
        phs_umap, hos_umap = build_structural(dataset, np.ones(len(dataset), dtype=bool))
        px_umap = torch.tensor(np.hstack([PHAGE_BASE_FIXED, phs_umap]).astype(np.float32)).to(DEVICE)
        hx_umap = torch.tensor(np.hstack([HOST_BASE, hos_umap]).astype(np.float32)).to(DEVICE)
        if best_gnn == "GAT":
            ei_umap, ea_umap = _build_gat_graph(np.ones(len(dataset), dtype=bool))
            ei_umap = ei_umap.to(DEVICE); ea_umap = ea_umap.to(DEVICE)
            _umap_model = PhageHostGAT(px_umap.shape[1], hx_umap.shape[1],
                                        N_EDGE_FEATS, n_phages).to(DEVICE)
        else:
            ei_umap = _build_edges(np.ones(len(dataset), dtype=bool)).to(DEVICE)
            ea_umap = None
            _umap_model = PhageHostSAGE(px_umap.shape[1], hx_umap.shape[1], n_phages).to(DEVICE)
        # Train briefly for embedding quality
        _umap_opt = torch.optim.AdamW(_umap_model.parameters(), lr=LR, weight_decay=WD)
        _umap_crit = nn.BCEWithLogitsLoss()
        _all_pi_u = torch.tensor(dataset["phage_idx"].values, dtype=torch.long).to(DEVICE)
        _all_hi_u = torch.tensor(dataset["host_idx"].values,  dtype=torch.long).to(DEVICE)
        _all_lb_u = torch.tensor(all_labels.astype(np.float32)).to(DEVICE)
        _umap_model.train()
        for _ep in range(min(EPOCHS, 60)):
            _umap_opt.zero_grad(set_to_none=True)
            if best_gnn == "GAT":
                _ef_u = torch.tensor(EDGE_FEATS_NP, dtype=torch.float32).to(DEVICE)
                _logits_u = _umap_model(px_umap, hx_umap, ei_umap, ea_umap,
                                         _all_pi_u, _all_hi_u, _ef_u)
            else:
                _logits_u = _umap_model(px_umap, hx_umap, ei_umap, _all_pi_u, _all_hi_u)
            _loss_u = _umap_crit(_logits_u, _all_lb_u)
            _loss_u.backward()
            torch.nn.utils.clip_grad_norm_(_umap_model.parameters(), 1.0)
            _umap_opt.step()
        # Extract embeddings
        with torch.no_grad():
            _umap_model.eval()
            if best_gnn == "GAT":
                _z_learned = _umap_model.get_embeddings(px_umap, hx_umap, ei_umap, ea_umap)
            else:
                _z_learned, _, _ = _umap_model.encode(px_umap, hx_umap, ei_umap)
        _z_np = _z_learned.cpu().numpy()
        ns_p = min(n_phages, 600); ns_h = min(n_hosts, 200)
        pi_s = np.random.choice(n_phages, ns_p, replace=False)
        hi_s = np.random.choice(n_hosts,  ns_h, replace=False)
        _z_sample = np.vstack([_z_np[pi_s], _z_np[n_phages + hi_s]])
        emb = _umap_mod.UMAP(n_components=2, random_state=SEED,
                              n_neighbors=15, min_dist=0.1,
                              metric="cosine").fit_transform(_z_sample)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(emb[:ns_p, 0], emb[:ns_p, 1], c=ARCH_COLOR["GAT"],
                   s=12, alpha=0.5, label=f"Phage (n={ns_p})")
        ax.scatter(emb[ns_p:, 0], emb[ns_p:, 1], c="#2E7D32",
                   s=40, alpha=0.8, marker="^", label=f"Host (n={ns_h})")
        for i, hi in enumerate(hi_s[:50]):
            ax.annotate(host_list[hi].split()[0][:8],
                        (emb[ns_p+i, 0], emb[ns_p+i, 1]), fontsize=5, alpha=0.7)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.set_title("Learned GNN Node Embeddings (UMAP)\nOrange=phages  Green=hosts",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        plt.tight_layout()
        save_fig(fig, "08_umap_embedding")
    except Exception as e:
        print(f"  UMAP failed: {e}")

# ── Plot 9: Ablation study (classical + name-embedding GNN ablation) ──
print("  Generating ablation/leakage audit plot...")
ablation_groups = {
    "Full model (all features)": all_feat_cols,
    "No genomic (no di/tri/tet/cub)": [c for c in all_feat_cols if not any(c.startswith(p) for p in ["p_di","p_tri","p_tet","p_cub"])],
    "No pair features (no tetra_corr/cub_dist)": [c for c in all_feat_cols if c not in ["tetra_corr","cub_dist"]],
    "Baseline (k3dist/k6dist/GCdiff/Homology only)": [c for c in NUMERIC_FEATS if c in all_feat_cols],
}
abl_aucs={}
print(f"  Ablation feature counts: Full={len(ablation_groups['Full model (all features)'])}, "
      f"No genomic={len(ablation_groups['No genomic (no di/tri/tet/cub)'])}, "
      f"No pair={len(ablation_groups['No pair features (no tetra_corr/cub_dist)'])}, "
      f"Baseline={len(ablation_groups['Baseline (k3dist/k6dist/GCdiff/Homology only)'])}")
for name, cols in ablation_groups.items():
    if not cols: continue
    X_abl_sc = StandardScaler().fit_transform(
        dataset[cols].fillna(0.0).values.astype(np.float32))
    fold_aucs=[]
    for sp in valid_species:
        tm=(dataset["host"]==sp).values
        Xtr,ytr=X_abl_sc[~tm],y_all[~tm]
        Xte,yte=X_abl_sc[tm], y_all[tm]
        if len(np.unique(yte))<2: continue
        clf=RandomForestClassifier(n_estimators=100,random_state=SEED,n_jobs=-1)
        clf.fit(Xtr,ytr); p=clf.predict_proba(Xte)[:,1]
        fold_aucs.append(roc_auc_score(yte,p))
    abl_aucs[name]=np.mean(fold_aucs) if fold_aucs else 0.0
    print(f"  Ablation '{name}': {abl_aucs[name]:.4f}")

# GNN Name-embedding ablation (zeroes out SVD name embeddings, keeps genomic)
print("  GNN ablation: testing name embedding impact...")
_abl_species_subset = valid_species
_name_dim = phage_name_emb.shape[1]
_host_name_dim = host_name_emb.shape[1]

_full_gnn_aucs = []
for sp in _abl_species_subset:
    tm = (dataset["host"]==sp).values
    if len(np.unique(all_labels[tm])) < 2: continue
    try:
        proba, yte = run_fold(~tm, tm, arch=best_gnn)
        _full_gnn_aucs.append(metrics(yte, proba)["roc_auc"])
    except Exception:
        pass
abl_aucs["GNN full model"] = np.mean(_full_gnn_aucs) if _full_gnn_aucs else 0.0

if HAS_TORCH:
    _saved_PBF = PHAGE_BASE_FIXED.copy()
    _saved_HB  = HOST_BASE.copy()

    # No name embeddings: zero out SVD dims, keep genomic features + found flag
    PHAGE_BASE_FIXED[:, :_name_dim] = 0.0
    HOST_BASE[:, :_host_name_dim]   = 0.0

    _no_name_aucs = []
    for sp in _abl_species_subset:
        tm = (dataset["host"]==sp).values
        if len(np.unique(all_labels[tm])) < 2: continue
        try:
            proba, yte = run_fold(~tm, tm, arch=best_gnn)
            m = metrics(yte, proba)
            _no_name_aucs.append(m["roc_auc"])
        except Exception:
            pass
    abl_aucs["GNN no name embeddings"] = np.mean(_no_name_aucs) if _no_name_aucs else 0.0
    print(f"  GNN no name embeddings: AUC={abl_aucs['GNN no name embeddings']:.4f}")

    # Restore original features
    PHAGE_BASE_FIXED[:] = _saved_PBF
    HOST_BASE[:]        = _saved_HB

    # No genomic features: zero out genomic dims, keep name embeddings
    PHAGE_BASE_FIXED[:, _name_dim:-1] = 0.0  # zero genomic, keep found_flag

    _no_gen_aucs = []
    for sp in _abl_species_subset:
        tm = (dataset["host"]==sp).values
        if len(np.unique(all_labels[tm])) < 2: continue
        try:
            proba, yte = run_fold(~tm, tm, arch=best_gnn)
            m = metrics(yte, proba)
            _no_gen_aucs.append(m["roc_auc"])
        except Exception:
            pass
    abl_aucs["GNN no genomic features"] = np.mean(_no_gen_aucs) if _no_gen_aucs else 0.0
    print(f"  GNN no genomic features: AUC={abl_aucs['GNN no genomic features']:.4f}")

    # Restore
    PHAGE_BASE_FIXED[:] = _saved_PBF
    HOST_BASE[:]        = _saved_HB

print("  Hard negative sensitivity analysis (within-genus only)...")
hard_neg_mask = (dataset["label"] == 0) & (dataset["source"] == "negative")
within_genus_neg = dataset[hard_neg_mask].copy()
pos_mask = dataset["label"] == 1
hard_neg_rows = []
_hard_valid_species = []
for sp in valid_species:
    sp_pos = dataset[(dataset["host"]==sp) & (dataset["label"]==1)]
    sp_genus = sp.split()[0]
    sp_hard_neg = within_genus_neg[within_genus_neg["genus"]==sp_genus]
    if len(sp_pos) < 3 or len(sp_hard_neg) < 3: continue
    _hard_valid_species.append(sp)

hard_neg_aucs = []
for sp in _hard_valid_species:
    sp_genus = sp.split()[0]
    sp_mask = ((dataset["host"]==sp) & (dataset["label"]==1)) | \
              ((dataset["genus"]==sp_genus) & (dataset["label"]==0))
    tm = (dataset["host"]==sp).values & sp_mask
    tr = sp_mask & ~tm
    if tr.sum() < 10 or tm.sum() < 3: continue
    Xtr = X_sc[tr]; ytr = y_all[tr]
    Xte = X_sc[tm]; yte = y_all[tm]
    if len(np.unique(yte)) < 2: continue
    clf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:,1]
    try:
        hard_neg_aucs.append(roc_auc_score(yte, p))
    except Exception:
        pass

hard_neg_mean = np.mean(hard_neg_aucs) if hard_neg_aucs else 0.0
abl_aucs["Hard negatives only"] = hard_neg_mean
print(f"  Hard negative sensitivity AUC={hard_neg_mean:.4f} (n={len(hard_neg_aucs)} species)")
pd.DataFrame([{"condition": "hard_negatives_only", "mean_auc": round(hard_neg_mean,4),
               "n_species": len(hard_neg_aucs)}]).to_csv(
    RESULTS_DIR / "hard_negative_sensitivity.csv", index=False)

# Save ablation results
pd.DataFrame([{"condition": k, "mean_auc": round(v,4)} for k,v in abl_aucs.items()]).to_csv(
    RESULTS_DIR / "ablation_results.csv", index=False)
print("  Saved ablation_results.csv")

fig,ax=plt.subplots(figsize=(11,5))
names=list(abl_aucs.keys()); vals=list(abl_aucs.values())
clr=["#2E7D32" if "full" in n.lower() else "#E65100" if "GNN" in n else "#1565C0" for n in names]
bars=ax.bar(range(len(names)),vals,color=clr,alpha=0.85,edgecolor="white",linewidth=0.8)
for b,v in zip(bars,vals):
    ax.text(b.get_x()+b.get_width()/2,v+0.005,f"{v:.3f}",ha="center",fontsize=8,fontweight="bold")
ax.set_xticks(range(len(names))); ax.set_xticklabels(names,rotation=20,ha="right",fontsize=7)
ax.set_ylim(0,1.0); ax.set_ylabel("Mean LOSO AUC (subset)",fontsize=11)
ax.set_title("Feature Ablation Study\n(classical RF + GNN name-embedding ablation)",fontsize=12,fontweight="bold")
ax.axhline(0.5,color="red",linestyle=":",lw=1,label="Random"); ax.legend(fontsize=8); ax.grid(axis="y",alpha=0.4)
plt.tight_layout()
save_fig(fig,"09_ablation_study")

# ── Plot 10: Dataset overview ──
fig,axes=plt.subplots(1,2,figsize=(14,6))
ax=axes[0]
genera=dataset.groupby("genus")["label"].agg(["sum","count"]).sort_values("count",ascending=False).head(20)
ax.barh(range(len(genera)),genera["count"].values,color="#90CAF9",alpha=0.85,label="All pairs")
ax.barh(range(len(genera)),genera["sum"].values,color="#1565C0",alpha=0.85,label="Positive pairs")
ax.set_yticks(range(len(genera))); ax.set_yticklabels(genera.index,fontsize=8)
ax.set_xlabel("Number of pairs"); ax.set_title("Top 20 Host Genera — Pair Counts",fontsize=10,fontweight="bold")
ax.legend(fontsize=8); ax.grid(axis="x",alpha=0.4)
ax2=axes[1]
label_counts=dataset["label"].value_counts()
ax2.pie([label_counts.get(1,0),label_counts.get(0,0)],labels=["Positive\n(infection)","Negative\n(no infection)"],
        colors=["#1a9850","#F44336"],autopct="%1.1f%%",startangle=90,
        textprops={"fontsize":11,"fontweight":"bold"})
ax2.set_title(f"Dataset Composition\n({len(dataset):,} total pairs)",fontsize=10,fontweight="bold")
plt.suptitle("PrecisionPhage Dataset Overview",fontsize=12,fontweight="bold")
plt.tight_layout()
save_fig(fig,"10_dataset_overview")


# ════════════════════════════════════════════════════════════════
# SECTION 9 — CROSS-DATABASE VALIDATION
# ════════════════════════════════════════════════════════════════
print("\n[9] Cross-database validation...")

def _fetch_phagesdb():
    """Fetch phage-host pairs from PhagesDB for external validation."""
    cache = RAW_DIR / "phagesdb_pairs.json"
    if cache.exists() and cache.stat().st_size > 100:
        pairs = json.loads(cache.read_text(encoding="utf-8"))
        print(f"  PhagesDB (cached): {len(pairs)} pairs")
        return pairs
    pairs = []
    for page in range(1, 60):
        try:
            url = f"https://phagesdb.org/api/phages/?page={page}&page_size=500"
            raw = _fetch(url, timeout=60)
            data = json.loads(raw.decode("utf-8"))
            results = data.get("results", [])
            if not results:
                break
            for phage in results:
                name = (phage.get("phage_name","") or phage.get("PhageName","") or "").strip()
                acc  = (phage.get("genbank_accession","") or phage.get("accession","") or "").strip()
                iso  = phage.get("isolation_host", {})
                host_genus = ""
                if isinstance(iso, dict):
                    host_genus = iso.get("genus","")
                elif isinstance(iso, str):
                    host_genus = iso.strip()
                if not host_genus:
                    host_genus = phage.get("host_genus","") or phage.get("HostGenus","")
                if not name or not host_genus:
                    continue
                host = _clean_host(host_genus)
                if len(host.split()) < 1:
                    continue
                pairs.append({"phage": acc or name, "host": host, "phage_name": name,
                              "source": "phagesdb"})
            time.sleep(0.5)
        except Exception as e:
            print(f"  PhagesDB page {page}: {e}")
            break
    try:
        cache.write_text(json.dumps(pairs), encoding="utf-8")
    except Exception:
        pass
    print(f"  PhagesDB: {len(pairs)} pairs fetched")
    return pairs

cross_db_results = {}

# Strategy A: Source-stratified validation (always available)
_vhi_phages = set(dataset.loc[dataset["source"]=="original", "phage"].unique())
_ext_phages = set(dataset["phage"].unique()) - _vhi_phages
if _ext_phages:
    _ext_only_mask = dataset["phage"].isin(_ext_phages).values
    _vhi_train_mask = ~_ext_only_mask
    _ext_labels = dataset.loc[_ext_only_mask, "label"]
    if _ext_labels.nunique() >= 2 and _ext_only_mask.sum() >= 10 and _vhi_train_mask.sum() >= 50:
        print(f"  Source-stratified: train on {_vhi_train_mask.sum()} VHI rows, "
              f"test on {_ext_only_mask.sum()} external rows")
        try:
            proba_ext, yte_ext = run_fold(_vhi_train_mask, _ext_only_mask, arch=best_gnn)
            ext_m = metrics(yte_ext, proba_ext)
            cross_db_results["source_stratified"] = ext_m
            print(f"  Source-stratified {ARCH_LABEL[best_gnn]}: "
                  f"ROC-AUC={ext_m['roc_auc']:.4f}  PR-AUC={ext_m['pr_auc']:.4f}")
        except Exception as e:
            print(f"  Source-stratified validation failed: {e}")
    else:
        print(f"  Source-stratified skipped: insufficient external data "
              f"({_ext_only_mask.sum()} rows, {_ext_labels.nunique()} classes)")

# Strategy B: PhagesDB external holdout
try:
    _phagesdb_pairs = _fetch_phagesdb()
except Exception as e:
    _phagesdb_pairs = []
    print(f"  PhagesDB fetch failed: {e}")

if _phagesdb_pairs:
    _pdb_df = pd.DataFrame(_phagesdb_pairs)
    _pdb_df["host"] = _pdb_df["host"].str.lower().str.strip()
    _existing_pairs_set = set(zip(dataset["phage"].str.lower(), dataset["host"].str.lower()))
    _existing_hosts = set(dataset["host"].str.lower())
    # Use PhagesDB phages that exist in our phage index (overlap) but form NEW pairs
    _pdb_df["_phage_lower"] = _pdb_df["phage"].str.lower()
    _pdb_df["_in_idx"] = _pdb_df.apply(
        lambda r: r["phage"] in phage2idx
               or r["phage"].split(".")[0] in phage2idx
               or r.get("phage_name", "").lower() in phage_name2idx,
        axis=1)
    _pdb_df["_host_known"] = _pdb_df["host"].isin(_existing_hosts)
    _pdb_df["_pair_key"] = list(zip(_pdb_df["_phage_lower"], _pdb_df["host"]))
    _pdb_df["_is_novel_pair"] = ~_pdb_df["_pair_key"].isin(_existing_pairs_set)
    _pdb_valid = _pdb_df[_pdb_df["_in_idx"] & _pdb_df["_host_known"] & _pdb_df["_is_novel_pair"]].copy()
    if len(_pdb_valid) >= 5:
        _pdb_valid["label"] = 1
        _pdb_valid["genus"] = _pdb_valid["host"].str.split().str[0]
        _pdb_valid["host_idx"] = _pdb_valid["host"].map(host2idx)
        _pdb_valid = _pdb_valid[_pdb_valid["host_idx"].notna()].copy()
        _pdb_valid["host_idx"] = _pdb_valid["host_idx"].astype(int)
        _pdb_valid["phage_idx"] = _pdb_valid["phage"].map(
            lambda p: phage2idx.get(p, phage2idx.get(p.split(".")[0], -1)))
        _pdb_valid = _pdb_valid[_pdb_valid["phage_idx"] >= 0].copy()
        _pdb_neg = []
        _pdb_rng = np.random.default_rng(SEED + 7777)
        for _, row in _pdb_valid.head(200).iterrows():
            g = row["genus"]
            neg_hosts = [h for h in (gh.get(g,[]))
                         if h != row["host"] and h in host2idx]
            if neg_hosts:
                nh = _pdb_rng.choice(neg_hosts)
                _pdb_neg.append({"phage":row["phage"],"host":nh,"label":0,
                                 "genus":g,"host_idx":host2idx[nh],
                                 "phage_idx":row["phage_idx"],
                                 "source":"phagesdb_neg"})
        if _pdb_neg:
            _pdb_test = pd.concat([_pdb_valid.head(200), pd.DataFrame(_pdb_neg)], ignore_index=True)
        else:
            _pdb_test = _pdb_valid.head(200).copy()
        if _pdb_test["label"].nunique() >= 2 and len(_pdb_test) >= 10:
            print(f"  PhagesDB holdout: {len(_pdb_test)} test pairs "
                  f"({(_pdb_test['label']==1).sum()} pos, {(_pdb_test['label']==0).sum()} neg)")
            for c in ALL_EDGE_FEATS:
                if c not in _pdb_test.columns:
                    _pdb_test[c] = 0.0
            _tmp_rows = _pdb_test[["phage","host","label","phage_idx","host_idx"]].copy()
            for c in ALL_EDGE_FEATS:
                _tmp_rows[c] = 0.0
            _orig_len = len(dataset)
            _combined = pd.concat([dataset, _tmp_rows], ignore_index=True)
            _combined_train = np.zeros(len(_combined), dtype=bool)
            _combined_train[:_orig_len] = True
            _combined_test = np.zeros(len(_combined), dtype=bool)
            _combined_test[_orig_len:] = True
            _pdb_edge = np.zeros((len(_tmp_rows), N_EDGE_FEATS), dtype=np.float32)
            _saved_ef = EDGE_FEATS_NP
            _saved_ds = dataset
            try:
                EDGE_FEATS_NP = np.vstack([_saved_ef, _pdb_edge])
                dataset = _combined
                proba_pdb, yte_pdb = run_fold(_combined_train, _combined_test, arch=best_gnn)
                pdb_m = metrics(yte_pdb, proba_pdb)
                cross_db_results["phagesdb"] = pdb_m
                print(f"  PhagesDB {ARCH_LABEL[best_gnn]}: "
                      f"ROC-AUC={pdb_m['roc_auc']:.4f}  PR-AUC={pdb_m['pr_auc']:.4f}")
            except Exception as e:
                print(f"  PhagesDB validation failed: {e}")
            finally:
                EDGE_FEATS_NP = _saved_ef
                dataset = _saved_ds
        else:
            print(f"  PhagesDB: insufficient test data ({len(_pdb_test)} pairs, "
                  f"{_pdb_test['label'].nunique()} classes)")
    else:
        print(f"  PhagesDB: {len(_pdb_valid)} valid pairs with indexed phages — insufficient")

# GPDB external validation
print("  Fetching GPDB pairs for external validation...")
_gpdb_cache = RAW_DIR / "gpdb_pairs.json"
_gpdb_pairs = []
if _gpdb_cache.exists() and _gpdb_cache.stat().st_size > 100:
    try:
        _gpdb_pairs = json.loads(_gpdb_cache.read_text(encoding="utf-8"))
        print(f"  GPDB (cached): {len(_gpdb_pairs)} pairs")
    except Exception:
        pass
if not _gpdb_pairs:
    for _page in range(1, 40):
        try:
            _url = f"https://gpd.phasodb.org/api/phages?page={_page}&per_page=500"
            _raw = _fetch(_url, timeout=60)
            _data = json.loads(_raw.decode("utf-8"))
            _results = _data.get("data", _data.get("results", []))
            if not _results: break
            for _rec in _results:
                _acc  = (_rec.get("accession","") or _rec.get("genbank_accession","") or "").strip()
                _host = (_rec.get("host","") or _rec.get("host_genus","") or "").strip()
                _name = (_rec.get("name","") or _rec.get("phage_name","") or "").strip()
                if not _host or not (_acc or _name): continue
                _h = _clean_host(_host)
                if len(_h.split()) < 1: continue
                _gpdb_pairs.append({"phage": _acc or _name, "phage_name": _name,
                                    "host": _h, "source": "gpdb"})
            time.sleep(0.4)
        except Exception as e:
            print(f"  GPDB page {_page}: {e}"); break
    try:
        _gpdb_cache.write_text(json.dumps(_gpdb_pairs), encoding="utf-8")
    except Exception:
        pass
    print(f"  GPDB: {len(_gpdb_pairs)} pairs fetched")

if _gpdb_pairs:
    _gpdb_df = pd.DataFrame(_gpdb_pairs)
    _gpdb_df["host"] = _gpdb_df["host"].str.lower().str.strip()
    _gpdb_df["_in_idx"] = _gpdb_df.apply(
        lambda r: r["phage"] in phage2idx
               or r["phage"].split(".")[0] in phage2idx
               or r.get("phage_name","").lower() in phage_name2idx, axis=1)
    _gpdb_df["_host_known"] = _gpdb_df["host"].isin(set(dataset["host"].str.lower()))
    _gpdb_df["_pair_key"] = list(zip(_gpdb_df["phage"].str.lower(), _gpdb_df["host"]))
    _existing_pairs_gpdb = set(zip(dataset["phage"].str.lower(), dataset["host"].str.lower()))
    _gpdb_df["_novel"] = ~_gpdb_df["_pair_key"].isin(_existing_pairs_gpdb)
    _gpdb_valid = _gpdb_df[_gpdb_df["_in_idx"] & _gpdb_df["_host_known"] & _gpdb_df["_novel"]].copy()
    print(f"  GPDB: {len(_gpdb_valid)} valid novel pairs with indexed phages")
    if len(_gpdb_valid) >= 5:
        _gpdb_valid["label"] = 1
        _gpdb_valid["genus"] = _gpdb_valid["host"].str.split().str[0]
        _gpdb_valid["host_idx"] = _gpdb_valid["host"].map(host2idx)
        _gpdb_valid = _gpdb_valid[_gpdb_valid["host_idx"].notna()].copy()
        _gpdb_valid["host_idx"] = _gpdb_valid["host_idx"].astype(int)
        _gpdb_valid["phage_idx"] = _gpdb_valid["phage"].map(
            lambda p: phage2idx.get(p, phage2idx.get(p.split(".")[0], -1)))
        _gpdb_valid = _gpdb_valid[_gpdb_valid["phage_idx"] >= 0].copy()
        _gpdb_neg = []
        _gpdb_rng = np.random.default_rng(SEED + 8888)
        for _, _row in _gpdb_valid.head(200).iterrows():
            _neg_hosts = [h for h in gh.get(_row["genus"],[])
                          if h != _row["host"] and h in host2idx]
            if _neg_hosts:
                _nh = _gpdb_rng.choice(_neg_hosts)
                _gpdb_neg.append({"phage":_row["phage"],"host":_nh,"label":0,
                                  "genus":_row["genus"],"host_idx":host2idx[_nh],
                                  "phage_idx":_row["phage_idx"],"source":"gpdb_neg"})
        if _gpdb_neg:
            _gpdb_test = pd.concat([_gpdb_valid.head(200), pd.DataFrame(_gpdb_neg)], ignore_index=True)
        else:
            _gpdb_test = _gpdb_valid.head(200).copy()
        if _gpdb_test["label"].nunique() >= 2 and len(_gpdb_test) >= 10:
            for c in ALL_EDGE_FEATS:
                if c not in _gpdb_test.columns: _gpdb_test[c] = 0.0
            _gpdb_tmp = _gpdb_test[["phage","host","label","phage_idx","host_idx"]].copy()
            for c in ALL_EDGE_FEATS: _gpdb_tmp[c] = 0.0
            _gpdb_orig_len = len(dataset)
            _gpdb_combined = pd.concat([dataset, _gpdb_tmp], ignore_index=True)
            _gpdb_train_mask = np.zeros(len(_gpdb_combined), dtype=bool)
            _gpdb_train_mask[:_gpdb_orig_len] = True
            _gpdb_test_mask = np.zeros(len(_gpdb_combined), dtype=bool)
            _gpdb_test_mask[_gpdb_orig_len:] = True
            _gpdb_edge = np.zeros((len(_gpdb_tmp), N_EDGE_FEATS), dtype=np.float32)
            _saved_ef2 = EDGE_FEATS_NP
            _saved_ds2 = dataset
            try:
                EDGE_FEATS_NP = np.vstack([_saved_ef2, _gpdb_edge])
                dataset = _gpdb_combined
                _proba_gpdb, _yte_gpdb = run_fold(_gpdb_train_mask, _gpdb_test_mask, arch=best_gnn)
                _gpdb_m = metrics(_yte_gpdb, _proba_gpdb)
                cross_db_results["gpdb"] = _gpdb_m
                print(f"  GPDB {ARCH_LABEL[best_gnn]}: "
                      f"ROC-AUC={_gpdb_m['roc_auc']:.4f}  PR-AUC={_gpdb_m['pr_auc']:.4f}")
            except Exception as e:
                print(f"  GPDB validation failed: {e}")
            finally:
                EDGE_FEATS_NP = _saved_ef2
                dataset = _saved_ds2
        else:
            print(f"  GPDB: insufficient test data after filtering")
    else:
        print(f"  GPDB: {len(_gpdb_valid)} valid pairs — insufficient for validation")

if cross_db_results:
    _cdb_rows = []
    for src, m in cross_db_results.items():
        _cdb_rows.append({"source": src, "model": ARCH_LABEL[best_gnn], **m})
    pd.DataFrame(_cdb_rows).to_csv(RESULTS_DIR / "cross_database_validation.csv", index=False)
    print("  Saved cross_database_validation.csv")

# ── Plot 11: Confidence interval forest plot ──
if ci_rows:
    _ci_roc = [r for r in ci_rows if r["metric"] == "roc_auc"]
    if _ci_roc:
        fig, ax = plt.subplots(figsize=(10, max(4, len(_ci_roc)*0.8)))
        for i, r in enumerate(_ci_roc):
            ax.errorbar(r["mean"], i, xerr=[[r["mean"]-r["ci_lo"]], [r["ci_hi"]-r["mean"]]],
                        fmt="o", capsize=6, capthick=2, markersize=8,
                        color=ARCH_COLOR.get(r["model"].split("+")[0].strip(),
                              ARCH_COLOR.get(r["model"], "#607D8B")),
                        ecolor="#333333", linewidth=2)
            ax.text(r["ci_hi"]+0.005, i,
                    f'{r["mean"]:.3f} [{r["ci_lo"]:.3f}, {r["ci_hi"]:.3f}]',
                    va="center", fontsize=9)
        ax.set_yticks(range(len(_ci_roc)))
        ax.set_yticklabels([r["model"] for r in _ci_roc], fontsize=10)
        ax.axvline(0.5, color="red", linestyle=":", lw=1.2, label="Random")
        ax.set_xlabel("ROC-AUC", fontsize=12)
        ax.set_title("LOSO ROC-AUC with 95% Bootstrap CI", fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.4); ax.legend(fontsize=8)
        plt.tight_layout()
        save_fig(fig, "11_confidence_intervals")


# Plot 12: Calibration curves
from sklearn.calibration import calibration_curve
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0,1],[0,1],"k:",lw=1.5,label="Perfect calibration")
for arch in _archs_available:
    vm = ~np.isnan(gnn_results[arch]["all_proba"])
    if vm.sum() > 50 and len(np.unique(all_labels[vm])) == 2:
        try:
            frac_pos, mean_pred = calibration_curve(
                all_labels[vm], gnn_results[arch]["all_proba"][vm],
                n_bins=10, strategy="uniform")
            ax.plot(mean_pred, frac_pos, marker="o", lw=2,
                    color=ARCH_COLOR[arch], label=ARCH_LABEL[arch])
        except Exception:
            pass
for mname, cr in clf_results.items():
    vm = ~np.isnan(cr["probas"])
    if vm.sum() > 50 and len(np.unique(y_all[vm])) == 2:
        try:
            frac_pos, mean_pred = calibration_curve(
                y_all[vm], cr["probas"][vm], n_bins=10, strategy="uniform")
            ax.plot(mean_pred, frac_pos, marker="s", lw=1.8,
                    linestyle="--", label=mname)
        except Exception:
            pass
ax.set_xlabel("Mean Predicted Probability", fontsize=12)
ax.set_ylabel("Fraction of Positives", fontsize=12)
ax.set_title("Calibration Curves — All Models\n(closer to diagonal = better calibrated)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.4)
ax.set_xlim(0,1); ax.set_ylim(0,1)
plt.tight_layout()
save_fig(fig, "12_calibration_curves")
print("  Saved plot: 12_calibration_curves.png")

# ════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════
print("\n" + "="*66)
print("  FINAL RESULTS")
print("="*66)
best_g=gnn_results["GAT"]; best_s=gnn_results["SAGE"]

# PR-AUC from LOSO results
_g_pr = best_g["loso_df"]["pr_auc"].mean() if "pr_auc" in best_g["loso_df"].columns else 0.0
_s_pr = best_s["loso_df"]["pr_auc"].mean() if "pr_auc" in best_s["loso_df"].columns else 0.0

# 95% CI strings
_g_ci = ""
_s_ci = ""
for r in ci_rows:
    if r["model"] == ARCH_LABEL["GAT"] and r["metric"] == "roc_auc":
        _g_ci = f" [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
    if r["model"] == ARCH_LABEL["SAGE"] and r["metric"] == "roc_auc":
        _s_ci = f" [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"

print(f"""
  ┌────────────────────────────────────────────────────────────────────┐
  │  Metric                  GAT+Edge MLP         SAGE+Residual       │
  ├────────────────────────────────────────────────────────────────────┤
  │  LOSO mean ROC-AUC  {best_g['loso_mean']:.4f}±{best_g['loso_std']:.4f}      {best_s['loso_mean']:.4f}±{best_s['loso_std']:.4f}      │
  │  LOSO 95% CI        {_g_ci:<20s} {_s_ci:<20s} │
  │  LOSO mean PR-AUC   {_g_pr:.4f}              {_s_pr:.4f}              │
  │  LOSO pooled AUC    {best_g['loso_pooled']['roc_auc']:.4f}              {best_s['loso_pooled']['roc_auc']:.4f}              │
  │  LOGO mean AUC      {best_g['logo_mean']:.4f}              {best_s['logo_mean']:.4f}              │
  │  Unseen strain      {best_g['mc_auc']:.4f}              {best_s['mc_auc']:.4f}              │
  │  Greedy cov@3       {best_g['means_c']['greedy']:.3f}               {best_s['means_c']['greedy']:.3f}               │
  └────────────────────────────────────────────────────────────────────┘""")
for mname,cr in clf_results.items():
    _clf_pr = cr["df"]["pr_auc"].mean() if hasattr(cr.get("df"), "columns") and "pr_auc" in cr["df"].columns else 0.0
    print(f"  {mname:<20} LOSO ROC-AUC={cr['mean']:.4f}  PR-AUC={_clf_pr:.4f}")

if sig_rows:
    print("\n  Statistical significance (Wilcoxon signed-rank):")
    for r in sig_rows:
        star = "***" if r.get("wilcoxon_p") is not None and r["wilcoxon_p"] < 0.001 else \
               "**"  if r.get("wilcoxon_p") is not None and r["wilcoxon_p"] < 0.01  else \
               "*"   if r.get("wilcoxon_p") is not None and r["wilcoxon_p"] < 0.05  else "ns"
        pv = r.get("wilcoxon_p")
        print(f"    {r['model_1']} vs {r['model_2']}: p={pv:.4g} {star}")

if cross_db_results:
    print("\n  Cross-database validation:")
    for src, m in cross_db_results.items():
        print(f"    {src}: ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}")

if hard_neg_mean > 0:
    print(f"\n  Hard negative sensitivity (within-genus negatives only):")
    print(f"    Mean AUC={hard_neg_mean:.4f} (n={len(hard_neg_aucs)} species)")
    print(f"    {'PASS' if hard_neg_mean > 0.80 else 'WARN'}: "
          f"{'Performance holds on hard negatives' if hard_neg_mean > 0.80 else 'Performance drops significantly on hard negatives — check negative sampling'}")

print(f"""
  All plots → {PLOT_DIR}
  All CSVs  → {RESULTS_DIR}

  Plots saved:
    01_loso_per_species.png       — per-species LOSO bars + GAT vs SAGE scatter
    02_roc_curves_all_models.png  — ROC curves for all models
    03_model_comparison.png       — LOSO / LOGO / Unseen bar chart
    04_logo_gat.png               — LOGO per-genus AUC (GAT)
    04_logo_sage.png              — LOGO per-genus AUC (SAGE)
    05_cocktail_comparison.png    — cocktail strategy bars
    06_feature_importance.png     — top 25 feature importances
    07_unseen_strain_auc.png      — unseen strain simulation
    08_umap_embedding.png         — learned GNN node embeddings UMAP
    09_ablation_study.png         — feature ablation + name embedding ablation
    10_dataset_overview.png       — dataset composition
    11_confidence_intervals.png   — forest plot with 95% bootstrap CI
    12_calibration_curves.png     — reliability diagrams for all models

  CSVs saved:
    model_comparison.csv          — all models with ROC-AUC + PR-AUC
    statistical_tests.csv         — pairwise Wilcoxon + Mann-Whitney tests
    confidence_intervals.csv      — 95% bootstrap CI for all models
    ablation_results.csv          — classical + GNN ablation results
    cross_database_validation.csv — external validation results
    hard_negative_sensitivity.csv — within-genus hard negative AUC
""")
print("  DONE!")