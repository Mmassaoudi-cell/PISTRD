"""Generates IEEE-ready LaTeX tables directly from result CSVs -- no
hand-typed numbers."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "tables"
TAB.mkdir(exist_ok=True)

df_orig = pd.read_csv(ROOT / "results" / "benchmarks" / "benchmark_results.csv")
seed20_path = ROOT / "results" / "benchmarks" / "benchmark_results_20seed.csv"
if seed20_path.exists():
    df_20 = pd.read_csv(seed20_path)
    replaced = set(df_20["model"].unique())
    df = pd.concat([df_orig[~df_orig.model.isin(replaced)], df_20], ignore_index=True)
else:
    df = df_orig
# wall-clock time per (fit + val + test) run, straight from the benchmark log
eff = df_orig.groupby("model")["elapsed_s"].agg(["mean", "std", "count"])
wtl = pd.read_csv(ROOT / "results" / "stats" / "benchmark_wtl.csv")  # written by src/stats/wtl_analysis.py

# ---------------- Table 1: main comparison ----------------
main_models = [
    ("PISTRD (proposed)", "PISTRD_final"),
    ("Source-method-adapted", "SourceMethodAdapted"),
    ("Logistic Regression (best classical)", "LogReg"),
    ("LightGBM (best boosting)", "LightGBM"),
    ("Transformer (best generic DL)", "Transformer"),
    ("STGAT (best task-specific)", "C2_STGAT"),
]
f1 = df.groupby(["model", "testbed"])["macro_f1"].mean().unstack()
fpr = df.groupby("model")["FPR"].mean()

lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Main benchmark comparison (macro-F1 by testbed, mean FPR, mean wall-clock time). All values computed on the untouched TEST split; the full 16-benchmark comparison is given in Table~\ref{tab:wtl}.}",
    r"\label{tab:main_results}",
    r"\begin{tabular}{lccccccc}",
    r"\toprule",
    r"Method & case14 & case39 & case57 & case118 & Avg. FPR & Time (s) \\",
    r"\midrule",
]
for label, key in main_models:
    row = f1.loc[key]
    lines.append(
        f"{label} & {row['case14']:.3f} & {row['case39']:.3f} & {row['case57']:.3f} & {row['case118']:.3f} & "
        f"{fpr[key]:.4f} & {eff.loc[key,'mean']:.2f} \\\\"
    )
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
(TAB / "table1_main_results.tex").write_text("\n".join(lines), encoding="utf-8")

# ---------------- Table 2: WTL summary ----------------
lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Win/tie/loss of PISTRD against all 16 benchmarks. Paired Wilcoxon signed-rank test, Holm-corrected across all comparisons; rank-biserial effect size.}",
    r"\label{tab:wtl}",
    r"\begin{tabular}{lrrrl}",
    r"\toprule",
    r"Benchmark & Mean $\Delta$F1 & $p$ (Holm) & Effect $r$ & Verdict \\",
    r"\midrule",
]
for _, r in wtl.sort_values("mean_diff_(benchmark_minus_PISTRD)").iterrows():
    verdict = r["verdict"].replace("PISTRD_", "").replace("_", " ")
    p_str = f"{r['p_holm']:.4f}" if r["p_holm"] < 0.999 else "1.000"
    if r["low_power_n4"]:
        p_str += "$^\\dagger$"
    lines.append(
        f"{r['model'].replace('_',' ')} & {r['mean_diff_(benchmark_minus_PISTRD)']:+.3f} & {p_str} & "
        f"{r['rank_biserial_effect_size']:.2f} & {verdict} \\\\"
    )
lines += [
    r"\bottomrule",
    r"\multicolumn{5}{l}{\footnotesize $^\dagger$ n=4 testbed-level pairs only (deterministic method, no seed variance); descriptive, low power.}\\",
    r"\end{tabular}",
    r"\end{table*}",
]
(TAB / "table2_wtl_summary.tex").write_text("\n".join(lines), encoding="utf-8")

print("Wrote:", list(TAB.glob("*.tex")))
