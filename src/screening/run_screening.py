"""Stage 2 validation-only candidate screening (master-prompt Section 9):
evaluates all 4 candidates on all 4 testbeds, 3 seeds each, TRAIN+VAL only
(TEST is never touched here). Writes results/screening/screening_results.csv
incrementally so progress survives if the run is interrupted.
"""
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "candidates"))
sys.path.insert(0, str(ROOT / "src" / "eval"))

from rls_cusum import score_split as c1_score_split
from pirk import PIRKSimplified
from graph_utils import build_line_graph
from stgat import train_stgat, score as stgat_score
from tcn_cp import train_tcn, score as tcn_score, conformal_threshold
from norm_utils import fit_normalizer, apply_normalizer
from metrics import evaluate_model, compute_metrics

TESTBEDS = ("case14", "case39", "case57", "case118")
SEEDS = (0, 1, 2)
DATA = ROOT / "results" / "data"
OUT = ROOT / "results" / "screening"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT / "screening_results.csv"


def load_npz(testbed, split):
    d = np.load(DATA / f"{testbed}_{split}.npz", allow_pickle=True)
    return d["X_encoded"], d["y"], d["edges"], int(d["n_line"])


def append_result(row: dict):
    df = pd.DataFrame([row])
    header = not RESULTS_CSV.exists()
    df.to_csv(RESULTS_CSV, mode="a", header=header, index=False)


def run_c1(testbed):
    n_line = int(np.load(DATA / f"{testbed}_train.npz", allow_pickle=True)["n_line"])
    val_df = c1_score_split(str(DATA / f"{testbed}_val_stats.csv"), n_line)
    test_df = c1_score_split(str(DATA / f"{testbed}_test_stats.csv"), n_line)
    m = evaluate_model(val_df["score"].values, val_df["label"].values,
                        val_df["score"].values, val_df["label"].values)  # validation-only screening: eval on VAL
    return m


def run_c3(testbed):
    n_line = int(np.load(DATA / f"{testbed}_train.npz", allow_pickle=True)["n_line"])
    model = PIRKSimplified().fit(str(DATA / f"{testbed}_train_stats.csv"), n_line)
    val_df = model.score_split(str(DATA / f"{testbed}_val_stats.csv"))
    m = evaluate_model(val_df["score"].values, val_df["label"].values,
                        val_df["score"].values, val_df["label"].values)
    return m


def run_c2(testbed, seed, epochs):
    Xtr, ytr, edges, n_line = load_npz(testbed, "train")
    Xv, yv, _, _ = load_npz(testbed, "val")
    mean, std = fit_normalizer(Xtr)
    Xtr_n, Xv_n = apply_normalizer(Xtr, mean, std), apply_normalizer(Xv, mean, std)
    ei = build_line_graph(edges, n_line)
    adj = np.zeros((n_line, n_line), dtype=bool)
    adj[ei[0], ei[1]] = True
    model = train_stgat(Xtr_n, ytr, Xv_n, yv, adj, n_line, Xtr.shape[1], epochs=epochs, seed=seed)
    val_s = stgat_score(model, Xv_n, adj).ravel()
    m = evaluate_model(val_s, yv, val_s, yv)
    return m


def run_c4(testbed, seed, epochs, alpha=0.05):
    Xtr, ytr, edges, n_line = load_npz(testbed, "train")
    Xv, yv, _, _ = load_npz(testbed, "val")
    mean, std = fit_normalizer(Xtr)
    Xtr_n, Xv_n = apply_normalizer(Xtr, mean, std), apply_normalizer(Xv, mean, std)
    model = train_tcn(Xtr_n, ytr, Xv_n, yv, epochs=epochs, seed=seed)
    val_s = tcn_score(model, Xv_n).ravel()
    # conformal calibration: use normal-only (label==0) half of val as calibration,
    # the other half as the "test-within-screening" evaluation set.
    rng = np.random.default_rng(seed)
    normal_idx = np.where(yv == 0)[0]
    rng.shuffle(normal_idx)
    cal_idx = normal_idx[: len(normal_idx) // 2]
    eval_idx = np.setdiff1d(np.arange(len(yv)), cal_idx)
    thr = conformal_threshold(val_s[cal_idx], alpha=alpha)
    m = compute_metrics(yv[eval_idx], val_s[eval_idx], thr)
    m["threshold_from_val"] = thr
    return m


def main(epochs_dl=25, testbeds=TESTBEDS, seeds=SEEDS):
    for testbed in testbeds:
        print(f"=== {testbed} ===")
        # C1, C3: deterministic given the (fixed) data, run once and replicate across "seeds" label for uniform reporting
        for cname, fn in (("C1_RLS_CUSUM", run_c1), ("C3_PIRK", run_c3)):
            t0 = time.time()
            m = fn(testbed)
            m.update({"candidate": cname, "testbed": testbed, "seed": 0,
                      "elapsed_s": round(time.time() - t0, 1)})
            append_result(m)
            print(cname, testbed, "macro_f1=", round(m["macro_f1"], 3), "time", m["elapsed_s"])

        for seed in seeds:
            t0 = time.time()
            m = run_c2(testbed, seed, epochs_dl)
            m.update({"candidate": "C2_STGAT", "testbed": testbed, "seed": seed,
                      "elapsed_s": round(time.time() - t0, 1)})
            append_result(m)
            print("C2_STGAT", testbed, seed, "macro_f1=", round(m["macro_f1"], 3), "time", m["elapsed_s"])

            t0 = time.time()
            m = run_c4(testbed, seed, epochs_dl)
            m.update({"candidate": "C4_TCN_CP", "testbed": testbed, "seed": seed,
                      "elapsed_s": round(time.time() - t0, 1)})
            append_result(m)
            print("C4_TCN_CP", testbed, seed, "macro_f1=", round(m["macro_f1"], 3), "time", m["elapsed_s"])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs_dl", type=int, default=25)
    ap.add_argument("--testbeds", nargs="+", default=list(TESTBEDS))
    args = ap.parse_args()
    main(epochs_dl=args.epochs_dl, testbeds=args.testbeds)
