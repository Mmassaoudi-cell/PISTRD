"""Statistical win/tie/loss analysis: PISTRD_final vs every benchmark.

Unit of comparison: (testbed, seed) macro-F1, paired against PISTRD's own
(deterministic, single-valued per testbed) macro-F1 on the same testbed.
For benchmarks with 5 seeds this gives n=20 paired samples per comparison,
for 3-seed DL models n=12, for deterministic benchmarks (LogReg/SVM/kNN/
SourceMethodAdapted/C1_RLS_CUSUM) n=4 -- explicitly flagged as low-power
(scipy still returns a p-value, but with n=4 the test has essentially no
power to detect anything at alpha=0.05; reported as descriptive only).

Wilcoxon signed-rank test (paired, robust to non-normal macro-F1
differences), Holm-Bonferroni correction across all benchmark comparisons,
rank-biserial effect size. Classification per master-prompt Section 15/30:
  - statistically superior: PISTRD mean > benchmark mean AND Holm-adjusted p < 0.05
  - statistically inferior: benchmark mean > PISTRD mean AND Holm-adjusted p < 0.05
  - practically superior: |mean diff| > 0.02 macro-F1 but not statistically significant (n too small)
  - statistically indistinguishable: otherwise
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(ROOT / "results" / "benchmarks" / "benchmark_results.csv")

pistrd = df[df.model == "PISTRD_final"].set_index("testbed")["macro_f1"].to_dict()
print("PISTRD per-testbed macro-F1:", pistrd)

models = [m for m in df.model.unique() if m != "PISTRD_final"]
rows = []
for model in models:
    sub = df[df.model == model]
    diffs = []
    for _, r in sub.iterrows():
        diffs.append(r["macro_f1"] - pistrd[r["testbed"]])
    diffs = np.array(diffs)
    n = len(diffs)
    mean_benchmark = sub["macro_f1"].mean()
    mean_pistrd_matched = np.mean([pistrd[t] for t in sub["testbed"]])
    mean_diff = diffs.mean()

    if n >= 4 and not np.allclose(diffs, 0):
        try:
            stat, p = wilcoxon(diffs)
        except ValueError:
            stat, p = np.nan, 1.0
        n_pos = int((diffs > 0).sum())
        n_neg = int((diffs < 0).sum())
        rank_biserial = (n_pos - n_neg) / n if n > 0 else np.nan
    else:
        p = 1.0
        rank_biserial = np.nan
    rows.append({
        "model": model, "n_pairs": n,
        "mean_macro_f1": round(mean_benchmark, 4),
        "PISTRD_mean_macro_f1_matched": round(mean_pistrd_matched, 4),
        "mean_diff_(benchmark_minus_PISTRD)": round(mean_diff, 4),
        "wilcoxon_p_raw": p,
        "rank_biserial_effect_size": round(rank_biserial, 3) if not np.isnan(rank_biserial) else np.nan,
        "low_power_n4": n <= 4,
    })

res = pd.DataFrame(rows)
reject, p_adj, _, _ = multipletests(res["wilcoxon_p_raw"], alpha=0.05, method="holm")
res["p_holm"] = p_adj
res["holm_significant"] = reject


def classify(row):
    if row["low_power_n4"]:
        basis = "descriptive_only_n4"
    else:
        basis = "tested"
    if row["holm_significant"] and row["mean_diff_(benchmark_minus_PISTRD)"] < 0:
        return "PISTRD_statistically_superior", basis
    if row["holm_significant"] and row["mean_diff_(benchmark_minus_PISTRD)"] > 0:
        return "PISTRD_statistically_inferior", basis
    if abs(row["mean_diff_(benchmark_minus_PISTRD)"]) > 0.02:
        return ("PISTRD_practically_superior" if row["mean_diff_(benchmark_minus_PISTRD)"] < 0
                else "PISTRD_practically_inferior"), basis
    return "statistically_indistinguishable", basis


res[["verdict", "verdict_basis"]] = res.apply(lambda r: pd.Series(classify(r)), axis=1)
res = res.sort_values("mean_diff_(benchmark_minus_PISTRD)")
res.to_csv(ROOT / "BENCHMARK_WTL.csv", index=False)
print(res.to_string(index=False))

wins = (res["verdict"].isin(["PISTRD_statistically_superior", "PISTRD_practically_superior"])).sum()
losses = (res["verdict"].isin(["PISTRD_statistically_inferior", "PISTRD_practically_inferior"])).sum()
ties = (res["verdict"] == "statistically_indistinguishable").sum()
print(f"\nPISTRD: {wins} wins / {ties} ties / {losses} losses (out of {len(res)} benchmarks)")
