"""Generates the corrected-statistics LaTeX table (table_stats_v2.tex) from
results/stats/wtl_per_testbed_v2.csv, merging original 3-seed DL/candidate
results with the new 20-seed tabular reruns."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "tables"
TAB.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "results" / "stats" / "wtl_per_testbed_v2.csv")
df = df.sort_values("mean_macro_f1")

lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Corrected per-testbed statistical protocol: sign test per testbed, Fisher-combined, Holm-corrected across all 16 comparisons. Seed counts: 20 for stochastic tabular benchmarks, 3 for compute-heavy deep/graph models (disclosed as underpowered for this test), 1 for deterministic methods.}",
    r"\label{tab:stats_v2}",
    r"\begin{tabular}{lccc}",
    r"\toprule",
    r"Benchmark & $n$/testbed & Fisher $p$ (Holm) & Significant? \\",
    r"\midrule",
]
for _, r in df.iterrows():
    n = r["n_seeds_per_testbed"]
    sig = "Yes" if r["holm_significant"] else "No"
    p = r["fisher_p_holm"]
    p_str = f"{p:.4f}" if p < 0.999 else "1.000"
    lines.append(f"{r['model'].replace('_',' ')} & {n} & {p_str} & {sig} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

(TAB / "table_stats_v2.tex").write_text("\n".join(lines), encoding="utf-8")
print("Wrote table_stats_v2.tex")
print("\n".join(lines))
