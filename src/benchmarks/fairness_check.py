"""Devil's Advocate CRITICAL #1 response: does giving the strongest tabular
benchmarks an explicit one-step lag/delta feature (the exact structure
PISTRD's temporal branch is built around) close the gap to PISTRD?

Adds, for every sensor, delta_i = last_i(t) - last_i(t-1) as additional
engineered features (computed from consecutive stats-CSV rows within a
scenario, matching PISTRD's own regressor construction), then re-trains
LogReg and LightGBM (the two strongest benchmarks) with these features
appended to the existing window-statistics feature set.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "benchmarks"))
sys.path.insert(0, str(ROOT / "src" / "eval"))

from metrics import evaluate_model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

DATA = ROOT / "results" / "data"
OUT = ROOT / "results" / "fairness_check"
OUT.mkdir(parents=True, exist_ok=True)

N_LINE = {"case14": 15, "case39": 35, "case57": 63, "case118": 173}
STREAM = "enc"


def load_with_lag(stats_csv_path: str, n_line: int):
    df = pd.read_csv(stats_csv_path)
    last_cols = [f"{STREAM}_stat_{4 * n_line + i}" for i in range(n_line)]
    base_cols = [c for c in df.columns if c.startswith(f"{STREAM}_stat_")]
    df = df.sort_values(["scenario", "t"]).reset_index(drop=True)

    delta = np.zeros((len(df), n_line))
    for _, g in df.groupby("scenario"):
        idx = g.index.values
        last_vals = g[last_cols].values
        d = np.zeros_like(last_vals)
        d[1:] = last_vals[1:] - last_vals[:-1]
        delta[idx] = d

    X_base = df[base_cols].values.astype(np.float32)
    X_aug = np.concatenate([X_base, delta.astype(np.float32)], axis=1)
    y = df["label"].values.astype(np.int64)
    return X_base, X_aug, y


def run_one(name, model_factory, testbed, n_line):
    Xb_tr, Xa_tr, ytr = load_with_lag(str(DATA / f"{testbed}_train_stats.csv"), n_line)
    Xb_v, Xa_v, yv = load_with_lag(str(DATA / f"{testbed}_val_stats.csv"), n_line)
    Xb_t, Xa_t, yt = load_with_lag(str(DATA / f"{testbed}_test_stats.csv"), n_line)

    results = {}
    for tag, (Xtr, Xv, Xt) in (("baseline", (Xb_tr, Xb_v, Xb_t)), ("with_lag_feature", (Xa_tr, Xa_v, Xa_t))):
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xv_s, Xt_s = scaler.transform(Xtr), scaler.transform(Xv), scaler.transform(Xt)
        model = model_factory()
        model.fit(Xtr_s, ytr)
        val_s = model.predict_proba(Xv_s)[:, 1]
        test_s = model.predict_proba(Xt_s)[:, 1]
        m = evaluate_model(val_s, yv, test_s, yt)
        results[tag] = m["macro_f1"]
    return results


def main():
    rows = []
    for testbed, n_line in N_LINE.items():
        for name, factory in (
            ("LogReg", lambda: LogisticRegression(max_iter=2000, class_weight="balanced")),
            ("LightGBM", lambda: lgb.LGBMClassifier(n_estimators=300, class_weight="balanced",
                                                      n_jobs=-1, verbosity=-1)),
        ):
            r = run_one(name, factory, testbed, n_line)
            gap_before = 0.9923 - r["baseline"]  # PISTRD's mean, from BENCHMARK_WTL.csv
            gap_after = 0.9923 - r["with_lag_feature"]
            rows.append({"model": name, "testbed": testbed,
                         "macro_f1_baseline": r["baseline"],
                         "macro_f1_with_lag": r["with_lag_feature"],
                         "gap_to_PISTRD_baseline": gap_before,
                         "gap_to_PISTRD_with_lag": gap_after,
                         "gap_closed_fraction": (gap_before - gap_after) / gap_before if gap_before > 0 else np.nan})
            print(testbed, name, "baseline=", round(r["baseline"], 4),
                  "with_lag=", round(r["with_lag_feature"], 4))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "fairness_check_results.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
