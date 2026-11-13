"""Generates all data-driven figures directly from result CSVs. No numbers
are hand-typed. Figures 1-2 (architecture/workflow schematics) are built
separately in make_schematics.py."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

# ---------------- Figure 3: main benchmark performance ----------------
bench = pd.read_csv(ROOT / "results" / "benchmarks" / "benchmark_results.csv")
agg = bench.groupby("model")["macro_f1"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#d62728" if m == "PISTRD_final" else "#4c72b0" for m in agg.index]
ax.barh(agg.index[::-1], agg.values[::-1], color=colors[::-1])
ax.set_xlabel("Mean macro-F1 (across 4 testbeds, all seeds)")
ax.set_title("Main benchmark comparison: PISTRD vs. 16 benchmarks")
ax.set_xlim(0, 1.05)
for i, v in enumerate(agg.values[::-1]):
    ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "fig3_main_benchmark.pdf")
plt.close()

# ---------------- Figure 4: partial-replay robustness ----------------
pr = pd.read_csv(ROOT / "results" / "robustness" / "partial_replay_results.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, metric, title in zip(axes, ["macro_f1", "minority_recall"], ["macro-F1", "minority recall"]):
    for variant, marker in zip(["Full", "Temporal_only", "Spatial_only"], ["o", "s", "^"]):
        sub = pr[pr.variant == variant].groupby("attack_fraction")[metric].mean().sort_index(ascending=False)
        ax.plot(sub.index, sub.values, marker=marker, label=variant)
    ax.set_xlabel("Fraction of sensors replayed")
    ax.set_title(title)
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Score")
axes[0].legend()
plt.suptitle("Robustness to partial (subset-sensor) replay attacks")
plt.tight_layout()
plt.savefig(FIG / "fig4_partial_replay_robustness.pdf")
plt.close()

# ---------------- Figure 5: ablation + dither-necessity ----------------
dith = pd.read_csv(ROOT / "results" / "ablation" / "dither_necessity_results.csv")
fig, ax = plt.subplots(figsize=(8, 4.5))
piv = dith.pivot_table(index="testbed", columns=["variant", "stream"], values="macro_f1")
testbeds = ["case14", "case39", "case57", "case118"]
x = np.arange(len(testbeds))
width = 0.13
variants = ["Full", "Temporal_only", "Spatial_only"]
streams = ["plain", "enc"]
i = 0
for variant in variants:
    for stream in streams:
        vals = [piv.loc[tb, (variant, stream)] for tb in testbeds]
        label = f"{variant} ({'no dither' if stream=='plain' else 'dither-encoded'})"
        ax.bar(x + i * width, vals, width, label=label,
               hatch="//" if stream == "plain" else None,
               color=plt.cm.tab10(variants.index(variant)), alpha=0.55 if stream == "plain" else 1.0)
        i += 1
ax.set_xticks(x + width * 2.5)
ax.set_xticklabels(testbeds)
ax.set_ylabel("macro-F1")
ax.set_title("Dither-encoding necessity: plain vs. encoded stream, all branches")
ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
ax.legend(fontsize=6, ncol=2, loc="lower right")
plt.tight_layout()
plt.savefig(FIG / "fig5_dither_necessity_ablation.pdf")
plt.close()

# ---------------- Figure 6: performance-efficiency Pareto ----------------
# wall-clock time per (fit + val + test) run, straight from the benchmark log
eff = bench.groupby("model")["elapsed_s"].agg(["mean", "std", "count"])
perf = bench.groupby("model")["macro_f1"].mean()
merged = pd.DataFrame({"time_s": eff["mean"], "macro_f1": perf}).dropna()
fig, ax = plt.subplots(figsize=(7.5, 5.5))
for name, row in merged.iterrows():
    is_final = name == "PISTRD_final"
    ax.scatter(row["time_s"], row["macro_f1"], s=90 if is_final else 45,
               c="#d62728" if is_final else "#4c72b0", zorder=3 if is_final else 2,
               edgecolors="black" if is_final else "none")
    ax.annotate(name, (row["time_s"], row["macro_f1"]), fontsize=6.5,
                xytext=(4, 3), textcoords="offset points")
ax.set_xscale("log")
ax.set_xlabel("Mean wall-clock time per (fit + val + test) run [s, log scale]")
ax.set_ylabel("Mean macro-F1")
ax.set_title("Performance-efficiency Pareto frontier")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "fig6_pareto.pdf")
plt.close()

print("Figures written to", FIG)
for f in sorted(FIG.glob("*.pdf")):
    print(" ", f.name)
