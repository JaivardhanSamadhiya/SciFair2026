"""
10_pipeline_diagram.py  —  PrecisionPhage Pipeline Diagram
===========================================================
PrecisionPhage | ISEF 2026

Generates a clean publication-style pipeline diagram showing
the full translational workflow from pathogen genome to therapy.

Output: data/plots/precisionphage_pipeline.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR    = _SCRIPT_DIR.parent / "data"
PLOT_DIR    = BASE_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

print("Generating PrecisionPhage pipeline diagram...")

fig, ax = plt.subplots(figsize=(18, 9))
ax.set_xlim(0, 18); ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#FAFAFA")

# ── COLOR PALETTE ──
C = {
    "input":       "#1565C0",   # deep blue
    "features":    "#2E7D32",   # deep green
    "model":       "#6A1B9A",   # deep purple
    "cocktail":    "#E65100",   # deep orange
    "resistance":  "#B71C1C",   # deep red
    "output":      "#00695C",   # teal
    "arrow":       "#37474F",
    "bg":          "#ECEFF1",
    "white":       "#FFFFFF",
}

def draw_box(ax, x, y, w, h, title, body_lines, color, icon=""):
    """Draw a rounded box with title and body text."""
    # Shadow
    shadow = FancyBboxPatch((x+0.06, y-0.06), w, h,
                              boxstyle="round,pad=0.12",
                              facecolor="#B0BEC5", edgecolor="none",
                              alpha=0.4, zorder=1)
    ax.add_patch(shadow)
    # Main box
    box = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.12",
                           facecolor=C["white"], edgecolor=color,
                           linewidth=2.5, zorder=2)
    ax.add_patch(box)
    # Header bar
    header = FancyBboxPatch((x, y+h-0.72), w, 0.72,
                              boxstyle="round,pad=0.08",
                              facecolor=color, edgecolor="none",
                              alpha=0.92, zorder=3)
    ax.add_patch(header)
    # Title
    ax.text(x+w/2, y+h-0.36, f"{icon} {title}".strip(),
             ha="center", va="center", fontsize=10.5, fontweight="bold",
             color=C["white"], zorder=4)
    # Body
    for i, line in enumerate(body_lines):
        ax.text(x+0.18, y+h-0.95-(i*0.38), line,
                 ha="left", va="center", fontsize=8.2, color="#37474F", zorder=4)

def arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                  arrowprops=dict(
                      arrowstyle="-|>",
                      color=C["arrow"],
                      lw=2.0,
                      mutation_scale=20,
                      connectionstyle="arc3,rad=0.0"
                  ), zorder=5)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my+0.12, label, ha="center", va="bottom",
                 fontsize=7.5, color=C["arrow"], style="italic")

# ── TITLE ──
ax.text(9, 8.65, "PrecisionPhage — Translational Pipeline",
         ha="center", va="center", fontsize=17, fontweight="bold",
         color="#263238")
ax.text(9, 8.28, "From Pathogen Genome to Optimized Phage Cocktail",
         ha="center", va="center", fontsize=11, color="#546E7A")

# ── BOX DEFINITIONS: (x, y, w, h) ──
BW = 2.55; BH = 3.6; GAP = 0.3; YBASE = 3.6

boxes = [
    # Stage 1
    (0.3, YBASE, BW, BH, "Pathogen Genome\nSequencing", [
        "• Clinical isolate",
        "• Whole-genome seq.",
        "• Species identification",
        "• Strain typing (WGS)",
    ], C["input"], "①"),

    # Stage 2
    (0.3 + (BW+GAP)*1, YBASE, BW, BH, "Feature Extraction", [
        "• k-mer 3/6 distances",
        "• GC content difference",
        "• Sequence homology",
        "• Phage char n-grams",
        "• Network topology",
    ], C["features"], "②"),

    # Stage 3
    (0.3 + (BW+GAP)*2, YBASE, BW, BH, "Ensemble Model", [
        "• XGBoost (3 variants)",
        "• Graph Attention Net",
        "• LOSO-CV validated",
        "• AUC = 0.9297",
    ], C["model"], "③"),

    # Stage 4
    (0.3 + (BW+GAP)*3, YBASE, BW, BH, "Cocktail\nOptimization", [
        "• Greedy set-cover",
        "• Strain-level coverage",
        "• k=1–5 evaluated",
        "• 71.6% coverage@3",
    ], C["cocktail"], "④"),

    # Stage 5
    (0.3 + (BW+GAP)*4, YBASE, BW, BH, "Resistance\nRobustness", [
        "• Best phage removed",
        "• Coverage recomputed",
        "• 15.1% avg drop",
        "• p < 0.0001",
    ], C["resistance"], "⑤"),

    # Stage 6
    (0.3 + (BW+GAP)*5, YBASE, BW, BH, "Therapy\nRecommendation", [
        "• Ranked phage list",
        "• Cocktail prescription",
        "• Confidence score",
        "• Experimental flag",
    ], C["output"], "⑥"),
]

for (x, y, w, h, title, body, color, icon) in boxes:
    title_lines = title.split("\n")
    draw_box(ax, x, y, w, h, title_lines[0] + (" " + title_lines[1] if len(title_lines)>1 else ""),
             body, color, icon)

# ── ARROWS ──
for i in range(len(boxes)-1):
    x1 = boxes[i][0]   + boxes[i][2]
    y1 = boxes[i][1]   + boxes[i][3] / 2
    x2 = boxes[i+1][0]
    y2 = boxes[i+1][1] + boxes[i+1][3] / 2
    arrow(ax, x1, y1, x2, y2)

# ── VALIDATION BADGES (below each box) ──
badges = [
    (boxes[0][0] + BW/2, YBASE-0.45, "NCBI / PhagesDB", "#1565C0"),
    (boxes[1][0] + BW/2, YBASE-0.45, "4 + 3 features",  "#2E7D32"),
    (boxes[2][0] + BW/2, YBASE-0.45, "LOSO-CV",         "#6A1B9A"),
    (boxes[3][0] + BW/2, YBASE-0.45, "Wilcoxon p<0.001","#E65100"),
    (boxes[4][0] + BW/2, YBASE-0.45, "Monte Carlo",     "#B71C1C"),
    (boxes[5][0] + BW/2, YBASE-0.45, "Experimental val.","#00695C"),
]
for (bx, by, label, color) in badges:
    badge = FancyBboxPatch((bx-0.9, by-0.18), 1.8, 0.36,
                            boxstyle="round,pad=0.06",
                            facecolor=color, edgecolor="none", alpha=0.15, zorder=2)
    ax.add_patch(badge)
    ax.text(bx, by, label, ha="center", va="center",
             fontsize=7.8, color=color, fontweight="bold", zorder=3)

# ── BOTTOM STATS BAR ──
stats_y = 1.85
ax.add_patch(FancyBboxPatch((0.3, stats_y), 17.4, 1.3,
                              boxstyle="round,pad=0.1",
                              facecolor="#E3F2FD", edgecolor="#1565C0",
                              linewidth=1.5, zorder=1, alpha=0.8))
ax.text(9, stats_y+0.95, "Key Performance Metrics",
         ha="center", fontsize=11, fontweight="bold", color="#1565C0")
metrics = [
    ("Ensemble AUC",   "0.9297", "#6A1B9A"),
    ("S. aureus AUC",  "0.9519", "#2E7D32"),
    ("Strain cov@3",   "71.6%",  "#E65100"),
    ("vs Random",      "+41.0%", "#1565C0"),
    ("≥75% species@3", "60.5%",  "#B71C1C"),
    ("NCBI strains",   "924",    "#00695C"),
]
mx_start = 1.2
mx_step  = 17.4 / len(metrics)
for i, (label, val, color) in enumerate(metrics):
    mx = mx_start + i * mx_step
    ax.text(mx, stats_y+0.58, val, ha="left", fontsize=16, fontweight="bold", color=color)
    ax.text(mx, stats_y+0.25, label, ha="left", fontsize=8, color="#546E7A")

# ── FOOTER ──
ax.text(9, 1.45, "PrecisionPhage  |  ISEF 2026  |  Leave-One-Species-Out Cross-Validation  |  No Data Leakage",
         ha="center", fontsize=8, color="#90A4AE", style="italic")

plt.tight_layout(pad=0.3)
out_path = PLOT_DIR / "precisionphage_pipeline.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  Saved: {out_path}")
print("\n  Done!")