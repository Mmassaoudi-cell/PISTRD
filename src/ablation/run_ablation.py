"""Ablation: Full PISTRD vs. Temporal-branch-only vs. Spatial-branch-only,
on the FROZEN test split, proving each component's contribution (master
prompt Section 19). Full model = pirk_final.PISTRD. Temporal-only reuses the
exact mechanism validated in screening (pirk.PIRKSimplified). Spatial-only
is a new minimal wrapper using just PISTRD's physics-informed WLS residual.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "candidates"))
sys.path.insert(0, str(ROOT / "src" / "eval"))
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from pirk_final import PISTRD
from pirk import PIRKSimplified
from metrics import evaluate_model

TESTBEDS = {"case14": 15, "case39": 35, "case57": 63, "case118": 173}
DATA = ROOT / "results" / "data"
OUT = ROOT / "results" / "ablation"
OUT.mkdir(parents=True, exist_ok=True)


def eval_full(testbed, n_line):
    m = PISTRD(testbed).fit(str(DATA / f"{testbed}_train_stats.csv"), n_line)
    m.calibrate(str(DATA / f"{testbed}_val_stats.csv"))
    val_df = m.score_split(str(DATA / f"{testbed}_val_stats.csv"))
    test_df = m.score_split(str(DATA / f"{testbed}_test_stats.csv"))
    return evaluate_model(val_df["score"].values, val_df["label"].values,
                            test_df["score"].values, test_df["label"].values)


def eval_temporal_only(testbed, n_line):
    m = PIRKSimplified().fit(str(DATA / f"{testbed}_train_stats.csv"), n_line)
    val_df = m.score_split(str(DATA / f"{testbed}_val_stats.csv"))
    test_df = m.score_split(str(DATA / f"{testbed}_test_stats.csv"))
    return evaluate_model(val_df["score"].values, val_df["label"].values,
                            test_df["score"].values, test_df["label"].values)


def eval_spatial_only(testbed, n_line):
    full = PISTRD(testbed).fit(str(DATA / f"{testbed}_train_stats.csv"), n_line)
    full.calibrate(str(DATA / f"{testbed}_val_stats.csv"))
    val_df = full.score_split(str(DATA / f"{testbed}_val_stats.csv"))
    test_df = full.score_split(str(DATA / f"{testbed}_test_stats.csv"))
    return evaluate_model(val_df["spatial_score"].values, val_df["label"].values,
                            test_df["spatial_score"].values, test_df["label"].values)


def main():
    rows = []
    for testbed, n_line in TESTBEDS.items():
        for name, fn in (("Full_PISTRD", eval_full),
                          ("Temporal_only", eval_temporal_only),
                          ("Spatial_only", eval_spatial_only)):
            m = fn(testbed, n_line)
            m.update({"variant": name, "testbed": testbed})
            rows.append(m)
            print(testbed, name, "macro_f1=", round(m["macro_f1"], 4),
                  "minority_recall=", round(m["minority_recall"], 4))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ablation_results.csv", index=False)


if __name__ == "__main__":
    main()
