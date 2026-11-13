"""Figures 1-2: architecture and conceptual-workflow schematics (hand-built,
not data-driven -- captions/labels reference only mechanisms implemented
in src/candidates/pirk_final.py)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def box(ax, xy, w, h, text, fc="#e8eef7", ec="#4c72b0", fontsize=9):
    b = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                         fc=fc, ec=ec, linewidth=1.4)
    ax.add_patch(b)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def arrow(ax, p1, p2):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14, color="#333333", linewidth=1.3)
    ax.add_patch(a)


# ---------------- Figure 1: PISTRD architecture ----------------
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis("off")
ax.set_title("Fig. 1. PISTRD architecture", fontsize=11)

box(ax, (0.3, 4.6), 2.6, 1.1, "Dither-encoded\nmeasurement stream\n$z^{enc}_k$ (source-method\nencoding, retained)")
box(ax, (3.4, 5.2), 2.9, 0.9, "Temporal branch:\njoint one-step ridge\npredictor + Mahalanobis\nresidual $d_T(k)$", fc="#fde8e8", ec="#d62728")
box(ax, (3.4, 3.6), 2.9, 0.9, "Spatial branch:\nDC power-flow WLS\nstate-estimation residual\n$d_S(k)$ (topology $H$)", fc="#e8f7ec", ec="#2ca02c")
box(ax, (6.9, 4.4), 2.8, 1.1, "OR-fusion decision:\nflag if $d_T(k){>}\\tau_T$ OR\n$d_S(k){>}\\tau_S$\n(each calibrated on VAL,\ntarget FAR $\\alpha/2$)", fc="#fff6e0", ec="#ff7f0e")

arrow(ax, (2.9, 5.3), (3.4, 5.65))
arrow(ax, (2.9, 4.8), (3.4, 4.05))
arrow(ax, (6.3, 5.65), (6.9, 5.1))
arrow(ax, (6.3, 4.05), (6.9, 4.75))

box(ax, (0.3, 1.4), 2.6, 1.0, "TRAIN split:\nfit ridge model +\ncovariance (temporal),\nfit WLS + variance\n(spatial)")
box(ax, (3.4, 1.4), 2.9, 1.0, "VAL split:\ncalibrate $\\tau_T,\\tau_S$\n(quantile, target FAR)\n-- never re-tuned on TEST")
box(ax, (6.9, 1.4), 2.8, 1.0, "TEST split:\nfrozen decision,\nreport macro-F1,\nFPR, minority recall")
arrow(ax, (1.6, 2.4), (1.6, 4.55))
arrow(ax, (4.85, 2.4), (4.85, 3.55))
arrow(ax, (8.3, 2.4), (8.3, 4.35))

plt.tight_layout()
plt.savefig(FIG / "fig1_architecture.pdf")
plt.close()

# ---------------- Figure 2: source vs proposed workflow ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

ax = axes[0]
ax.set_xlim(0, 6); ax.set_ylim(0, 8); ax.axis("off")
ax.set_title("(a) Source paper (single 2-state\nDC-motor testbed)", fontsize=10)
steps_a = ["Batch least-squares\nQ-learning $H_y$\n(frozen after $L_n$ samples)",
           "Independent per-sensor\nresidual $r_k$",
           "Threshold = sup($g_k$)\nover one calibration run\n(no stated FAR target)",
           "ADR / FAR only\n(no macro-F1, no\nmulti-testbed evidence)"]
y = 6.6
for s in steps_a:
    box(ax, (0.5, y), 5, 1.1, s, fc="#f5f5f5", ec="#888888", fontsize=8.5)
    if y > 1:
        arrow(ax, (3, y), (3, y - 0.5))
    y -= 1.7

ax = axes[1]
ax.set_xlim(0, 6); ax.set_ylim(0, 8); ax.axis("off")
ax.set_title("(b) Proposed (4 IEEE bus testbeds,\nreal load-driven)", fontsize=10)
steps_b = ["Joint (all-sensor) ridge\npredictor + physics-\ninformed WLS branch",
           "Joint Mahalanobis /\nspatial-consistency\nresiduals (2 branches)",
           "Per-branch VAL-calibrated\nquantile threshold\n(explicit target FAR $\\alpha$)",
           "macro-F1, PR-AUC, FPR,\nminority recall across\n4 testbeds, 16 benchmarks,\nWilcoxon+Holm stats"]
y = 6.6
for s in steps_b:
    box(ax, (0.5, y), 5, 1.1, s, fc="#e8eef7", ec="#4c72b0", fontsize=8.5)
    if y > 1:
        arrow(ax, (3, y), (3, y - 0.5))
    y -= 1.7

plt.suptitle("Fig. 2. Conceptual workflow: source paper vs. proposed", fontsize=11)
plt.tight_layout()
plt.savefig(FIG / "fig2_workflow_comparison.pdf")
plt.close()

print("Schematics written.")
