"""Robustness experiment: PARTIAL/subset replay attacks -- only a random
fraction of sensors are replayed during the attack window, the rest stay
live. This is the scenario the spatial (physics) branch was actually added
for (MODEL_SELECTION_REPORT.md): a whole-vector replay is spatially
self-consistent by construction (DITHER_NECESSITY_FINDING.md confirms the
spatial branch adds nothing there), but a PARTIAL replay breaks the
physical relationship among lines and should be visible to a topology-aware
residual even when the dither/temporal signal alone might be weaker (fewer
attacked dimensions contributing to any joint statistic).

Builds a small held-out test-only dataset (reusing the cached true
trajectories and TEST split's time range, so nothing here touches
train/val) at several attack fractions, and compares Full PISTRD vs
Temporal-only vs Spatial-only.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "candidates"))
sys.path.insert(0, str(ROOT / "src" / "eval"))
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from attacks import generate_scenario, scenario_to_labels
from pirk_final import PISTRD
from metrics import evaluate_model, compute_metrics

DATA = ROOT / "results" / "data"
OUT = ROOT / "results" / "robustness"
OUT.mkdir(parents=True, exist_ok=True)

TESTBEDS = {"case14": 15, "case39": 35, "case57": 63, "case118": 173}
W = 12
N_SCENARIOS = 6
FRACTIONS = (1.0, 0.5, 0.3, 0.15)  # 1.0 = whole-vector (matches main dataset), smaller = partial


def build_partial_test_stats(testbed: str, n_line: int, fraction: float, noise_std, dither_std, t0, t1):
    d = np.load(DATA / f"{testbed}_true_traj.npz", allow_pickle=True)
    line_flow = d["line_flow"]
    z_true_block = line_flow[t0:t1]
    Tb = t1 - t0

    rows = []
    for sidx in range(N_SCENARIOS):
        rng = np.random.default_rng(hash((testbed, "partial", fraction, sidx)) % (2**32))
        scenario = generate_scenario(W, Tb, rng)
        e = rng.normal(0.0, noise_std, size=(Tb, n_line))
        d_dither = rng.normal(0.0, dither_std, size=(Tb, n_line))
        z_meas = z_true_block + e

        replay_of, label_full = scenario_to_labels(scenario, Tb)
        n_attacked_sensors = max(1, int(round(fraction * n_line)))
        attacked_sensor_idx = rng.choice(n_line, size=n_attacked_sensors, replace=False)

        z_enc = z_meas.copy()
        for k, ridx in replay_of.items():
            z_enc[k, attacked_sensor_idx] = (z_meas[ridx, attacked_sensor_idx]
                                               + d_dither[ridx, attacked_sensor_idx]
                                               - d_dither[k, attacked_sensor_idx])

        for t in range(W - 1, Tb):
            window = z_enc[t - W + 1: t + 1]
            stat = np.concatenate([window.mean(0), window.std(0), window.min(0), window.max(0), window[-1]])
            rows.append({"testbed": testbed, "split": "test_partial", "scenario": sidx,
                          "t": t, "label": int(label_full[t]),
                          **{f"enc_stat_{i}": v for i, v in enumerate(stat)}})
    return pd.DataFrame(rows)


def main():
    rows = []
    for testbed, n_line in TESTBEDS.items():
        manifest = pd.read_csv(ROOT / "DATA_SPLIT_MANIFEST.csv")
        r = manifest[(manifest.testbed == testbed) & (manifest.split == "test")].iloc[0]
        t0, t1 = int(r.t_start), int(r.t_end)
        noise_std = 0.02 * np.load(DATA / f"{testbed}_true_traj.npz")["line_flow"][:1800].std(axis=0)
        dither_std = 0.06 * np.load(DATA / f"{testbed}_true_traj.npz")["line_flow"][:1800].std(axis=0)
        noise_std = np.where(noise_std < 1e-6, 1e-6, noise_std)

        model = PISTRD(testbed).fit(str(DATA / f"{testbed}_train_stats.csv"), n_line)
        model.calibrate(str(DATA / f"{testbed}_val_stats.csv"))

        for frac in FRACTIONS:
            df_test = build_partial_test_stats(testbed, n_line, frac, noise_std, dither_std, t0, t1)
            csv_path = OUT / f"_tmp_{testbed}_{frac}.csv"
            df_test.to_csv(csv_path, index=False)
            test_df = model.score_split(str(csv_path))
            csv_path.unlink()

            # use the model's thresholds already frozen via calibrate() on the
            # STANDARD val split -- this tests whether the original calibration
            # generalizes to a harder attack variant, not a re-tuned oracle threshold
            y = test_df["label"].values
            full = compute_metrics(y, test_df["score"].values, 1.0)  # score already normalized s.t. 1.0 = at either branch's threshold
            temporal = compute_metrics(y, test_df["temporal_score"].values, model.temporal_threshold)
            spatial = compute_metrics(y, test_df["spatial_score"].values, model.spatial_threshold)
            for name, m in (("Full", full), ("Temporal_only", temporal), ("Spatial_only", spatial)):
                m.update({"testbed": testbed, "attack_fraction": frac, "variant": name})
                rows.append(m)
                print(testbed, "frac=", frac, name, "macro_f1=", round(m["macro_f1"], 4),
                      "minority_recall=", round(m["minority_recall"], 4))

    pd.DataFrame(rows).to_csv(OUT / "partial_replay_results.csv", index=False)


if __name__ == "__main__":
    main()
