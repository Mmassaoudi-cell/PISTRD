"""Response to Perspective Reviewer W1 (adaptive-adversary gap): all main
experiments replay data recorded only 5-30 hours earlier -- within real
load's autocorrelation timescale, but NOT deliberately chosen by an attacker
to minimize detectability. This experiment models an INFORMED attacker who
specifically times the replay 24*k hours earlier (k=1,2,3), i.e., the same
hour-of-day on a previous day, exploiting daily load periodicity to make the
replayed segment maximally similar to the live signal an uninformed replay
would not achieve -- the single most obvious evasion strategy against a
temporal-residual detector, and a genuine worst-case (not average-case) test.
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
from metrics import compute_metrics, best_threshold_by_f1

DATA = ROOT / "results" / "data"
OUT = ROOT / "results" / "robustness"
TESTBEDS = {"case14": 15, "case39": 35, "case57": 63, "case118": 173}
W = 12
N_SCENARIOS = 6
DURATION = 8  # hours


def camouflaged_scenario(t0: int, t1: int, offset_days: int, rng: np.random.Generator):
    """Attack window placed at a random valid time; record window placed
    EXACTLY offset_days*24 hours earlier (same hour-of-day), not randomly."""
    margin = DURATION + offset_days * 24 + 20
    a_start = int(rng.integers(t0 + offset_days * 24 + 10, t1 - DURATION - 5))
    a_end = a_start + DURATION
    rec_start = a_start - offset_days * 24
    rec_end = rec_start + DURATION
    return {"record": (rec_start, rec_end), "attacks": [(a_start, a_end)]}


def scenario_to_labels_single(scenario, T):
    replay_of = {}
    label = np.zeros(T, dtype=bool)
    rec_start, rec_end = scenario["record"]
    a_start, a_end = scenario["attacks"][0]
    shift = a_start - rec_start
    for k in range(a_start, a_end):
        ridx = k - shift
        if rec_start <= ridx < rec_end:
            replay_of[k] = ridx
            label[k] = True
    return replay_of, label


def build_camouflaged_test(testbed, n_line, offset_days, t0, t1):
    traj = np.load(DATA / f"{testbed}_true_traj.npz")
    line_flow = traj["line_flow"]
    z_true_block = line_flow[t0:t1]
    Tb = t1 - t0

    train_std = line_flow[:1800].std(axis=0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    noise_std = 0.02 * train_std
    dither_std = 0.06 * train_std

    rows = []
    for sidx in range(N_SCENARIOS):
        rng = np.random.default_rng(hash((testbed, "camo", offset_days, sidx)) % (2**32))
        scenario = camouflaged_scenario(W, Tb, offset_days, rng)
        e = rng.normal(0.0, noise_std, size=(Tb, n_line))
        d = rng.normal(0.0, dither_std, size=(Tb, n_line))
        z_meas = z_true_block + e
        replay_of, label_full = scenario_to_labels_single(scenario, Tb)
        z_enc = z_meas.copy()
        for k, ridx in replay_of.items():
            z_enc[k] = z_meas[ridx] + d[ridx] - d[k]

        for t in range(W - 1, Tb):
            window = z_enc[t - W + 1: t + 1]
            stat = np.concatenate([window.mean(0), window.std(0), window.min(0), window.max(0), window[-1]])
            rows.append({"testbed": testbed, "split": "camo_test", "scenario": sidx, "t": t,
                         "label": int(label_full[t]),
                         **{f"enc_stat_{i}": v for i, v in enumerate(stat)}})
    return pd.DataFrame(rows)


def main():
    rows = []
    manifest = pd.read_csv(ROOT / "DATA_SPLIT_MANIFEST.csv")
    for testbed, n_line in TESTBEDS.items():
        r = manifest[(manifest.testbed == testbed) & (manifest.split == "test")].iloc[0]
        t0, t1 = int(r.t_start), int(r.t_end)

        model = PISTRD(testbed).fit(str(DATA / f"{testbed}_train_stats.csv"), n_line)
        model.calibrate(str(DATA / f"{testbed}_val_stats.csv"))
        val_df = model.score_split(str(DATA / f"{testbed}_val_stats.csv"))
        frozen_threshold = best_threshold_by_f1(val_df["score"].values, val_df["label"].values)

        # offset_days=1,2,3: informed attacker replays same-hour-previous-day(s)
        # data specifically to exploit daily periodicity (the camouflage attack).
        # Compare against the main-results random-offset (5-30h) numbers already
        # reported in Table II as the uninformed-attacker baseline.
        for offset_days in (1, 2, 3):
            df_test = build_camouflaged_test(testbed, n_line, offset_days, t0, t1)
            csv_path = OUT / f"_tmp_camo_{testbed}_{offset_days}.csv"
            df_test.to_csv(csv_path, index=False)
            scored = model.score_split(str(csv_path))
            csv_path.unlink()
            m = compute_metrics(scored["label"].values, scored["score"].values, frozen_threshold)
            m.update({"testbed": testbed, "offset_days": offset_days})
            rows.append(m)
            print(testbed, "offset_days=", offset_days, "macro_f1=", round(m["macro_f1"], 4),
                  "minority_recall=", round(m["minority_recall"], 4), "FPR=", round(m["FPR"], 4))
    pd.DataFrame(rows).to_csv(OUT / "camouflaged_replay_results.csv", index=False)


if __name__ == "__main__":
    main()
