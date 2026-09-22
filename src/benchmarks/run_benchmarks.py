"""Runs the final model (PISTRD) + 10+ benchmark suite against the FROZEN
TEST split (FINAL_MODEL_CONFIG.yaml is already frozen; this script does not
alter it). Threshold selection is always fit on VAL only, per src/eval/metrics.py.

Seeds: 5 for fast stochastic tabular models (RF/ExtraTrees/XGBoost/LightGBM/
CatBoost/MLP), 3 for the compute-heavy DL/candidate models (LSTM/Transformer/
STGAT/TCN-CP), matching the screening-stage precedent -- documented here
rather than silently padded to a uniform number.
"""
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "candidates"))
sys.path.insert(0, str(ROOT / "src" / "eval"))
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "src" / "benchmarks"))

from metrics import evaluate_model
from pirk_final import PISTRD
from source_method_adapted import SourceMethodAdapted
from rls_cusum import score_split as c1_score_split
from graph_utils import build_line_graph
from stgat import train_stgat, score as stgat_score
from tcn_cp import train_tcn, score as tcn_score, conformal_threshold
from norm_utils import fit_normalizer, apply_normalizer
import tabular_benchmarks as tb
import sequence_benchmarks as sb

TESTBEDS = ("case14", "case39", "case57", "case118")
N_LINE = {"case14": 15, "case39": 35, "case57": 63, "case118": 173}
DATA = ROOT / "results" / "data"
OUT = ROOT / "results" / "benchmarks"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT / "benchmark_results.csv"

SEEDS_FAST = tuple(range(20))  # bumped from 5->20: with only 5 seeds, a per-testbed
# sign test can never reach p<0.05 (min possible two-sided p at n=5 is 0.0625) --
# see review/STATISTICAL_POWER_FIX.md. 20 seeds gives min p=2*0.5^20=1.9e-6.
SEEDS_DL = (0, 1, 2)
EPOCHS_DL = 30


def append_result(row: dict):
    df = pd.DataFrame([row])
    header = not RESULTS_CSV.exists()
    df.to_csv(RESULTS_CSV, mode="a", header=header, index=False)


def load_npz(testbed, split):
    d = np.load(DATA / f"{testbed}_{split}.npz", allow_pickle=True)
    return d["X_encoded"], d["y"], d["edges"], int(d["n_line"])


def run_pistrd(testbed, n_line):
    m = PISTRD(testbed).fit(str(DATA / f"{testbed}_train_stats.csv"), n_line)
    m.calibrate(str(DATA / f"{testbed}_val_stats.csv"))
    val_df = m.score_split(str(DATA / f"{testbed}_val_stats.csv"))
    test_df = m.score_split(str(DATA / f"{testbed}_test_stats.csv"))
    return evaluate_model(val_df["score"].values, val_df["label"].values,
                            test_df["score"].values, test_df["label"].values)


def run_source_method(testbed, n_line):
    m = SourceMethodAdapted().fit(str(DATA / f"{testbed}_train_stats.csv"), n_line)
    val_df = m.score_split(str(DATA / f"{testbed}_val_stats.csv"))
    test_df = m.score_split(str(DATA / f"{testbed}_test_stats.csv"))
    return evaluate_model(val_df["score"].values, val_df["label"].values,
                            test_df["score"].values, test_df["label"].values)


def run_c1(testbed, n_line):
    val_df = c1_score_split(str(DATA / f"{testbed}_val_stats.csv"), n_line)
    test_df = c1_score_split(str(DATA / f"{testbed}_test_stats.csv"), n_line)
    return evaluate_model(val_df["score"].values, val_df["label"].values,
                            test_df["score"].values, test_df["label"].values)


def run_c2(testbed, seed):
    Xtr, ytr, edges, n_line = load_npz(testbed, "train")
    Xv, yv, _, _ = load_npz(testbed, "val")
    Xt, yt, _, _ = load_npz(testbed, "test")
    mean, std = fit_normalizer(Xtr)
    Xtr_n, Xv_n, Xt_n = (apply_normalizer(X, mean, std) for X in (Xtr, Xv, Xt))
    ei = build_line_graph(edges, n_line)
    adj = np.zeros((n_line, n_line), dtype=bool)
    adj[ei[0], ei[1]] = True
    model = train_stgat(Xtr_n, ytr, Xv_n, yv, adj, n_line, Xtr.shape[1], epochs=EPOCHS_DL, seed=seed)
    val_s = stgat_score(model, Xv_n, adj).ravel()
    test_s = stgat_score(model, Xt_n, adj).ravel()
    return evaluate_model(val_s, yv, test_s, yt)


def run_c4(testbed, seed):
    Xtr, ytr, edges, n_line = load_npz(testbed, "train")
    Xv, yv, _, _ = load_npz(testbed, "val")
    Xt, yt, _, _ = load_npz(testbed, "test")
    mean, std = fit_normalizer(Xtr)
    Xtr_n, Xv_n, Xt_n = (apply_normalizer(X, mean, std) for X in (Xtr, Xv, Xt))
    model = train_tcn(Xtr_n, ytr, Xv_n, yv, epochs=EPOCHS_DL, seed=seed)
    val_s = tcn_score(model, Xv_n).ravel()
    test_s = tcn_score(model, Xt_n).ravel()
    return evaluate_model(val_s, yv, test_s, yt)


def run_tabular(name, model, testbed, n_line):
    Xtr, ytr, _ = tb.load_xy(str(DATA / f"{testbed}_train_stats.csv"))
    Xv, yv, _ = tb.load_xy(str(DATA / f"{testbed}_val_stats.csv"))
    Xt, yt, _ = tb.load_xy(str(DATA / f"{testbed}_test_stats.csv"))
    val_s, test_s = tb.fit_score(name, model, Xtr, ytr, Xv, Xt)
    return evaluate_model(val_s, yv, test_s, yt)


def run_sequence(name, testbed, seed):
    Xtr, ytr, _, n_line = load_npz(testbed, "train")
    Xv, yv, _, _ = load_npz(testbed, "val")
    Xt, yt, _, _ = load_npz(testbed, "test")
    mean, std = fit_normalizer(Xtr)
    Xtr_n, Xv_n, Xt_n = (apply_normalizer(X, mean, std) for X in (Xtr, Xv, Xt))
    train_fn = sb.train_lstm if name == "LSTM" else sb.train_transformer
    model = train_fn(Xtr_n, ytr, Xv_n, yv, epochs=EPOCHS_DL, seed=seed)
    val_s = sb.score(model, Xv_n).ravel()
    test_s = sb.score(model, Xt_n).ravel()
    return evaluate_model(val_s, yv, test_s, yt)


def main():
    for testbed in TESTBEDS:
        n_line = N_LINE[testbed]
        print(f"=== {testbed} ===", flush=True)

        for name, fn in (("PISTRD_final", lambda: run_pistrd(testbed, n_line)),
                          ("SourceMethodAdapted", lambda: run_source_method(testbed, n_line)),
                          ("C1_RLS_CUSUM", lambda: run_c1(testbed, n_line))):
            t0 = time.time()
            m = fn()
            m.update({"model": name, "testbed": testbed, "seed": 0, "elapsed_s": round(time.time() - t0, 1)})
            append_result(m)
            print(name, testbed, "macro_f1=", round(m["macro_f1"], 4), flush=True)

        DETERMINISTIC = {"LogReg", "SVM_RBF", "kNN"}  # no meaningful seed-to-seed variance given fixed data
        for name in tb.get_models(0).keys():
            seeds = (0,) if name in DETERMINISTIC else SEEDS_FAST
            for seed in seeds:
                t0 = time.time()
                model = tb.get_models(seed)[name]
                m = run_tabular(name, model, testbed, n_line)
                m.update({"model": name, "testbed": testbed, "seed": seed, "elapsed_s": round(time.time() - t0, 1)})
                append_result(m)
                print(name, testbed, seed, "macro_f1=", round(m["macro_f1"], 4), flush=True)

        for seed in SEEDS_DL:
            for name, fn in (("C2_STGAT", lambda s=seed: run_c2(testbed, s)),
                              ("C4_TCN_CP", lambda s=seed: run_c4(testbed, s)),
                              ("LSTM", lambda s=seed: run_sequence("LSTM", testbed, s)),
                              ("Transformer", lambda s=seed: run_sequence("Transformer", testbed, s))):
                t0 = time.time()
                m = fn()
                m.update({"model": name, "testbed": testbed, "seed": seed, "elapsed_s": round(time.time() - t0, 1)})
                append_result(m)
                print(name, testbed, seed, "macro_f1=", round(m["macro_f1"], 4), flush=True)


if __name__ == "__main__":
    main()
