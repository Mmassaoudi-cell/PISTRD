"""Response to Methodology Reviewer W1/W2: per-testbed statistical tests
(not pooled across testbeds against a repeated constant), plus explicit
effective-sample-size reporting (independent attack scenarios, not
overlapping windows).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
df_orig = pd.read_csv(ROOT / "results" / "benchmarks" / "benchmark_results.csv")
df_orig = df_orig[df_orig.model != "PISTRD_final"]
df_20seed_path = ROOT / "results" / "benchmarks" / "benchmark_results_20seed.csv"
if df_20seed_path.exists():
    df_20 = pd.read_csv(df_20seed_path)
    # 20-seed reruns replace the original 5-seed tabular rows for the same models
    replaced_models = set(df_20["model"].unique())
    df = pd.concat([df_orig[~df_orig.model.isin(replaced_models)], df_20], ignore_index=True)
else:
    df = df_orig
pistrd_df = pd.read_csv(ROOT / "results" / "benchmarks" / "benchmark_results.csv")
manifest = pd.read_csv(ROOT / "DATA_SPLIT_MANIFEST.csv")

pistrd = pistrd_df[pistrd_df.model == "PISTRD_final"].set_index("testbed")["macro_f1"].to_dict()
testbeds = ["case14", "case39", "case57", "case118"]
models = [m for m in df.model.unique() if m != "PISTRD_final"]

# Effective independent sample size per testbed/split: N_SCENARIOS from the
# manifest (6), not the window count (~3,534) -- windows within a scenario
# are highly overlapping (stride=1) and not independent observations.
eff_n = manifest.set_index(["testbed", "split"])["n_scenarios"].to_dict()
print("Effective independent scenarios per (testbed, split):")
for tb in testbeds:
    print(f"  {tb}: train={eff_n[(tb,'train')]}, val={eff_n[(tb,'val')]}, test={eff_n[(tb,'test')]}")
print("  (vs. window counts of ~1,800-10,700 per split -- NOT independent observations)\n")

rows = []
for model in models:
    sub = df[df.model == model]
    per_testbed_p = {}
    n_seeds_seen = []
    for tb in testbeds:
        tb_scores = sub[sub.testbed == tb]["macro_f1"].values
        ref = pistrd[tb]
        diffs = tb_scores - ref
        n = len(diffs)
        n_seeds_seen.append(n)
        if n >= 2 and not np.allclose(diffs, 0):
            n_neg = int((diffs < 0).sum())
            # sign test: exact, valid even for n as small as 1-4, no exchangeability
            # assumption needed across testbeds since each test is now within-testbed only
            res = binomtest(n_neg, n, 0.5)
            per_testbed_p[tb] = res.pvalue
        else:
            per_testbed_p[tb] = 1.0 if n == 0 else (0.5 ** n if (diffs < 0).all() else 1.0)
    # combine 4 independent per-testbed p-values via Fisher's method
    ps = np.array(list(per_testbed_p.values()))
    ps = np.clip(ps, 1e-300, 1.0)
    fisher_stat = -2 * np.sum(np.log(ps))
    from scipy.stats import chi2 as chi2dist
    fisher_p = 1 - chi2dist.cdf(fisher_stat, df=2 * len(ps))
    rows.append({
        "model": model,
        **{f"p_{tb}": round(per_testbed_p[tb], 4) for tb in testbeds},
        "fisher_combined_p": fisher_p,
        "mean_macro_f1": round(sub["macro_f1"].mean(), 4),
        "n_seeds_per_testbed": n_seeds_seen[0] if len(set(n_seeds_seen)) == 1 else min(n_seeds_seen),
    })

res = pd.DataFrame(rows)
reject, p_adj, _, _ = multipletests(res["fisher_combined_p"], alpha=0.05, method="holm")
res["fisher_p_holm"] = p_adj
res["holm_significant"] = reject
res = res.sort_values("mean_macro_f1")
res.to_csv(ROOT / "results" / "stats" / "wtl_per_testbed_v2.csv", index=False)
print(res.to_string(index=False))
print(f"\nSignificant at Holm-corrected alpha=0.05 (per-testbed sign test, Fisher-combined): "
      f"{res['holm_significant'].sum()}/{len(res)}")
