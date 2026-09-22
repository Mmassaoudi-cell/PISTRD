"""
Builds the multi-testbed labeled replay-attack-detection dataset:
  - chronological train/val/test split of the real PJM-load-driven trajectory
    (no shuffling across time -- prevents leakage per DATA_SPLIT_MANIFEST policy)
  - per-split, multiple randomized attack scenarios (attacks.py) injected on
    top of the SAME true trajectory with independent noise/dither draws
  - both a "plain" replay stream (no defense) and an "encoded" replay stream
    (source-method-style dither encoding/decoding) are generated, so later
    benchmarking can test whether the encoding trick still matters once a
    model has access to realistic, non-stationary real-load data
  - noise/dither scale is calibrated from TRAIN-split statistics only
  - outputs: results/data/<testbed>_<split>.npz (raw sequence windows) and
    results/data/<testbed>_<split>_stats.csv (aggregated tabular features)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent))
from simulate import simulate_true_trajectory
from testbeds import TESTBEDS
from attacks import generate_scenario, scenario_to_labels

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "data"
OUT.mkdir(parents=True, exist_ok=True)

T_TOTAL = 3000
SPLITS = {"train": (0, 1800), "val": (1800, 2400), "test": (2400, 3000)}
W = 12                 # window length
N_SCENARIOS = 6         # attack scenarios per split per testbed
NOISE_FRAC = 0.02        # measurement-noise std as a fraction of per-line train std
DITHER_FRAC = 0.06       # dither std as a fraction of per-line train std (> noise, per Theorem 1's spirit)


def window_stats(window: np.ndarray) -> np.ndarray:
    # window: (W, n_line) -> (5*n_line,) [mean, std, min, max, last]
    return np.concatenate([
        window.mean(axis=0), window.std(axis=0),
        window.min(axis=0), window.max(axis=0), window[-1],
    ]).astype(np.float32)


def build_for_testbed(testbed: str):
    print(f"=== {testbed} ===")
    traj_cache = OUT / f"{testbed}_true_traj.npz"
    if traj_cache.exists():
        d = np.load(traj_cache, allow_pickle=True)
        traj = {"line_flow": d["line_flow"], "edges": d["edges"],
                "n_bus": int(d["n_bus"]), "n_line": int(d["n_line"])}
    else:
        traj = simulate_true_trajectory(testbed, T=T_TOTAL)
        np.savez_compressed(traj_cache, line_flow=traj["line_flow"], edges=traj["edges"],
                             n_bus=traj["n_bus"], n_line=traj["n_line"])
    line_flow = traj["line_flow"]
    n_line = traj["n_line"]

    train_t0, train_t1 = SPLITS["train"]
    train_std = line_flow[train_t0:train_t1].std(axis=0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    noise_std = NOISE_FRAC * train_std
    dither_std = DITHER_FRAC * train_std

    manifest_rows = []
    for split, (t0, t1) in SPLITS.items():
        z_true_block = line_flow[t0:t1]
        Tb = t1 - t0

        Xp_list, Xe_list, y_list, stat_rows = [], [], [], []
        for sidx in range(N_SCENARIOS):
            rng = np.random.default_rng(hash((testbed, split, sidx)) % (2**32))
            scenario = generate_scenario(W, Tb, rng)
            e = rng.normal(0.0, noise_std, size=(Tb, n_line))
            d = rng.normal(0.0, dither_std, size=(Tb, n_line))
            z_meas = z_true_block + e

            replay_of, label_full = scenario_to_labels(scenario, Tb)
            z_plain = z_meas.copy()
            z_enc = z_meas.copy()
            for k, ridx in replay_of.items():
                z_plain[k] = z_meas[ridx]
                z_enc[k] = z_meas[ridx] + d[ridx] - d[k]

            for t in range(W - 1, Tb):
                wp = z_plain[t - W + 1: t + 1]
                we = z_enc[t - W + 1: t + 1]
                lab = bool(label_full[t])
                Xp_list.append(wp.astype(np.float32))
                Xe_list.append(we.astype(np.float32))
                y_list.append(lab)
                sp = window_stats(wp)
                se = window_stats(we)
                stat_rows.append({
                    "testbed": testbed, "split": split, "scenario": sidx,
                    "strategy": scenario["strategy"], "t": t, "label": int(lab),
                    **{f"plain_stat_{i}": v for i, v in enumerate(sp)},
                    **{f"enc_stat_{i}": v for i, v in enumerate(se)},
                })

        Xp = np.stack(Xp_list)   # (N, W, n_line)
        Xe = np.stack(Xe_list)
        y = np.array(y_list, dtype=np.int64)
        np.savez_compressed(OUT / f"{testbed}_{split}.npz",
                             X_plain=Xp, X_encoded=Xe, y=y, edges=traj["edges"],
                             n_line=n_line)
        df = pd.DataFrame(stat_rows)
        df.to_csv(OUT / f"{testbed}_{split}_stats.csv", index=False)
        pos_rate = y.mean()
        print(f"  {split}: N={len(y)}  pos_rate={pos_rate:.3f}  X shape={Xp.shape}")
        manifest_rows.append({
            "testbed": testbed, "split": split, "t_start": t0, "t_end": t1,
            "n_windows": len(y), "n_positive": int(y.sum()), "positive_rate": round(float(pos_rate), 4),
            "n_scenarios": N_SCENARIOS, "window_len": W, "n_sensors": n_line,
            "noise_frac": NOISE_FRAC, "dither_frac": DITHER_FRAC,
        })
    return manifest_rows


if __name__ == "__main__":
    all_rows = []
    for tb in TESTBEDS:
        all_rows.extend(build_for_testbed(tb))
    manifest = pd.DataFrame(all_rows)
    manifest.to_csv(ROOT / "DATA_SPLIT_MANIFEST.csv", index=False)
    print(manifest.to_string(index=False))
