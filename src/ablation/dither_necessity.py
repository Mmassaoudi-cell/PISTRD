"""Tests whether the source paper's stochastic dither-encoding mechanism is
still NECESSARY once the empirical surface is extended to realistic,
non-stationary real-load multi-bus data -- or whether that realism alone
(as speculated in DATA_AUDIT.md) already makes replay attacks detectable to
a plain residual model. Runs Full/Temporal-only/Spatial-only on BOTH the
'plain' (no dither) and 'enc' (dither-encoded, source-method-style) streams.
"""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "candidates"))
sys.path.insert(0, str(ROOT / "src" / "eval"))
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from pirk_final import PISTRD
from metrics import evaluate_model

TESTBEDS = {"case14": 15, "case39": 35, "case57": 63, "case118": 173}
DATA = ROOT / "results" / "data"
OUT = ROOT / "results" / "ablation"
OUT.mkdir(parents=True, exist_ok=True)


def evaluate(testbed, n_line, stream):
    m = PISTRD(testbed).fit(str(DATA / f"{testbed}_train_stats.csv"), n_line, stream=stream)
    m.calibrate(str(DATA / f"{testbed}_val_stats.csv"))
    val_df = m.score_split(str(DATA / f"{testbed}_val_stats.csv"))
    test_df = m.score_split(str(DATA / f"{testbed}_test_stats.csv"))
    full = evaluate_model(val_df["score"].values, val_df["label"].values,
                            test_df["score"].values, test_df["label"].values)
    temporal = evaluate_model(val_df["temporal_score"].values, val_df["label"].values,
                                test_df["temporal_score"].values, test_df["label"].values)
    spatial = evaluate_model(val_df["spatial_score"].values, val_df["label"].values,
                               test_df["spatial_score"].values, test_df["label"].values)
    return full, temporal, spatial


def main():
    rows = []
    for testbed, n_line in TESTBEDS.items():
        for stream in ("plain", "enc"):
            full, temporal, spatial = evaluate(testbed, n_line, stream)
            for variant_name, m in (("Full", full), ("Temporal_only", temporal), ("Spatial_only", spatial)):
                m.update({"testbed": testbed, "stream": stream, "variant": variant_name})
                rows.append(m)
                print(testbed, stream, variant_name, "macro_f1=", round(m["macro_f1"], 4))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "dither_necessity_results.csv", index=False)


if __name__ == "__main__":
    main()
