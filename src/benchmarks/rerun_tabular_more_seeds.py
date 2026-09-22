"""Re-runs ONLY the fast tabular benchmarks at 20 seeds (up from 5) to give
the corrected per-testbed statistical test (wtl_analysis_v2.py) enough power
to reach significance. Writes to a separate CSV; DL/candidate models keep
their original 3-seed results (re-running those at 20 seeds is computationally
infeasible within the available time -- documented as a residual limitation).
"""
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "eval"))
sys.path.insert(0, str(ROOT / "src" / "benchmarks"))

from metrics import evaluate_model
import tabular_benchmarks as tb

TESTBEDS = ("case14", "case39", "case57", "case118")
DATA = ROOT / "results" / "data"
OUT = ROOT / "results" / "benchmarks"
RESULTS_CSV = OUT / "benchmark_results_20seed.csv"

SEEDS = tuple(range(20))
DETERMINISTIC = {"LogReg", "SVM_RBF", "kNN"}  # unchanged, no seed variance


def append_result(row: dict):
    df = pd.DataFrame([row])
    header = not RESULTS_CSV.exists()
    df.to_csv(RESULTS_CSV, mode="a", header=header, index=False)


def main():
    for testbed in TESTBEDS:
        print(f"=== {testbed} ===", flush=True)
        Xtr, ytr, _ = tb.load_xy(str(DATA / f"{testbed}_train_stats.csv"))
        Xv, yv, _ = tb.load_xy(str(DATA / f"{testbed}_val_stats.csv"))
        Xt, yt, _ = tb.load_xy(str(DATA / f"{testbed}_test_stats.csv"))

        for name in tb.get_models(0).keys():
            seeds = (0,) if name in DETERMINISTIC else SEEDS
            for seed in seeds:
                t0 = time.time()
                model = tb.get_models(seed)[name]
                val_s, test_s = tb.fit_score(name, model, Xtr, ytr, Xv, Xt)
                m = evaluate_model(val_s, yv, test_s, yt)
                m.update({"model": name, "testbed": testbed, "seed": seed,
                          "elapsed_s": round(time.time() - t0, 1)})
                append_result(m)
            print(name, testbed, "done,", len(seeds), "seeds", flush=True)


if __name__ == "__main__":
    main()
