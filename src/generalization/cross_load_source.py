"""Response to Devil's Advocate MAJOR #5: the 4 testbeds all use DC power-flow
physics driven by the SAME PJM load series, so they are closer to 1 physical
scenario at 4 scales than 4 independently-varying scenarios. This experiment
tests the already-trained, already-frozen PISTRD model (calibrated only on
PJM-driven TRAIN/VAL) against ENTIRELY NEW, out-of-distribution test data
generated from two different real load datasets (ISO New England, Malaysia
national grid) that PISTRD has never seen -- a genuine cross-load-source
generalization test, using additional real data as requested.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "src" / "candidates"))
sys.path.insert(0, str(ROOT / "src" / "eval"))

from simulate import simulate_true_trajectory
from attacks import generate_scenario, scenario_to_labels
from pirk_final import PISTRD
from metrics import evaluate_model, compute_metrics, best_threshold_by_f1

DATA_DIR = Path(r"C:\Users\MMASSAOUDI\Desktop\Data\Load Data")
OUT = ROOT / "results" / "generalization"
OUT.mkdir(parents=True, exist_ok=True)

T_TEST = 600  # match the original TEST split length
W = 12
N_SCENARIOS = 6
NOISE_FRAC = 0.02
DITHER_FRAC = 0.06
TESTBEDS = {"case14": 15, "case118": 173}


def load_isone(T, start_offset=2000):
    df = pd.read_csv(DATA_DIR / "isone_load.csv", sep=";")
    series = df["load"].values.astype(float)
    series = series[start_offset:start_offset + T]
    return series / series.mean()


def load_malaysia(T, start_offset=2000):
    df = pd.read_csv(DATA_DIR / "malaysia_load.csv", sep=";")
    series = df["load"].values.astype(float)
    series = series[start_offset:start_offset + T]
    return series / series.mean()


def build_ood_test_stats(testbed, n_line, mult, source_name, seed_base):
    traj = simulate_true_trajectory(testbed, T=len(mult), mult=mult)
    line_flow = traj["line_flow"]

    train_std_path = ROOT / "results" / "data" / f"{testbed}_true_traj.npz"
    train_line_flow = np.load(train_std_path)["line_flow"][:1800]
    train_std = train_line_flow.std(axis=0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    noise_std = NOISE_FRAC * train_std
    dither_std = DITHER_FRAC * train_std

    Tb = len(mult)
    rows = []
    for sidx in range(N_SCENARIOS):
        rng = np.random.default_rng(hash((testbed, source_name, sidx, seed_base)) % (2**32))
        scenario = generate_scenario(W, Tb, rng)
        e = rng.normal(0.0, noise_std, size=(Tb, n_line))
        d = rng.normal(0.0, dither_std, size=(Tb, n_line))
        z_meas = line_flow + e
        replay_of, label_full = scenario_to_labels(scenario, Tb)
        z_enc = z_meas.copy()
        for k, ridx in replay_of.items():
            z_enc[k] = z_meas[ridx] + d[ridx] - d[k]

        for t in range(W - 1, Tb):
            window = z_enc[t - W + 1: t + 1]
            stat = np.concatenate([window.mean(0), window.std(0), window.min(0), window.max(0), window[-1]])
            rows.append({"testbed": testbed, "split": "ood_test", "scenario": sidx,
                         "t": t, "label": int(label_full[t]),
                         **{f"enc_stat_{i}": v for i, v in enumerate(stat)}})
    return pd.DataFrame(rows)


def main():
    rows = []
    for testbed, n_line in TESTBEDS.items():
        model = PISTRD(testbed).fit(str(ROOT / "results" / "data" / f"{testbed}_train_stats.csv"), n_line)
        model.calibrate(str(ROOT / "results" / "data" / f"{testbed}_val_stats.csv"))

        # Frozen threshold from the ORIGINAL PJM VAL split (never re-tuned per
        # test set) -- this is what makes the OOD evaluation below a genuine
        # generalization test rather than an oracle re-fit.
        val_df = model.score_split(str(ROOT / "results" / "data" / f"{testbed}_val_stats.csv"))
        frozen_threshold = best_threshold_by_f1(val_df["score"].values, val_df["label"].values)

        # in-distribution reference: original PJM-driven TEST result, same frozen threshold
        orig_test_df = model.score_split(str(ROOT / "results" / "data" / f"{testbed}_test_stats.csv"))
        m_orig = compute_metrics(orig_test_df["label"].values, orig_test_df["score"].values, frozen_threshold)
        rows.append({"testbed": testbed, "load_source": "PJM (in-distribution)", **m_orig})
        print(testbed, "PJM (in-dist)", "macro_f1=", round(m_orig["macro_f1"], 4))

        for source_name, loader in (("ISO-NE", load_isone), ("Malaysia", load_malaysia)):
            mult = loader(T_TEST)
            df_ood = build_ood_test_stats(testbed, n_line, mult, source_name, seed_base=0)
            csv_path = OUT / f"_tmp_{testbed}_{source_name}.csv"
            df_ood.to_csv(csv_path, index=False)
            ood_scored = model.score_split(str(csv_path))
            csv_path.unlink()
            m = compute_metrics(ood_scored["label"].values, ood_scored["score"].values, frozen_threshold)
            rows.append({"testbed": testbed, "load_source": source_name, **m})
            print(testbed, source_name, "macro_f1=", round(m["macro_f1"], 4),
                  "minority_recall=", round(m["minority_recall"], 4))

    pd.DataFrame(rows).to_csv(OUT / "cross_load_source_results.csv", index=False)


if __name__ == "__main__":
    main()
