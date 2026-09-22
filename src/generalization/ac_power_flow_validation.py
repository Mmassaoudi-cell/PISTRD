"""Response to Domain Reviewer W3: the main study uses DC power flow
(no voltage magnitude, no reactive power, no losses). This builds a FULL
parallel case14 train/val/test dataset using AC power flow (pp.runpp) --
the same chronological split, same real PJM load driver, same attack
generation and dither encoding, same PISTRD fit/calibrate/score pipeline --
to test whether the method's performance is an artifact of DC linearization
or holds under genuine nonlinear AC physics. The spatial branch's topology
matrix H is intentionally kept as the DC approximation (not re-derived for
AC), so this simultaneously tests robustness to physics-MODEL mismatch for
that branch specifically.
"""
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import pandapower as pp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "src" / "candidates"))
sys.path.insert(0, str(ROOT / "src" / "eval"))

from testbeds import get_testbed
from simulate import load_multiplier_series
from attacks import generate_scenario, scenario_to_labels
from pirk_final import PISTRD
from metrics import evaluate_model

OUT = ROOT / "results" / "generalization"
OUT.mkdir(parents=True, exist_ok=True)
T_TOTAL = 3000
SPLITS = {"train": (0, 1800), "val": (1800, 2400), "test": (2400, 3000)}
W = 12
N_SCENARIOS = 6
NOISE_FRAC = 0.02
DITHER_FRAC = 0.06


def simulate_ac_trajectory(testbed_name: str, T: int):
    net = get_testbed(testbed_name)
    base_p = net.load.p_mw.values.copy()
    base_q = net.load.q_mvar.values.copy()
    mult, _ = load_multiplier_series(T)
    n_line = len(net.line)
    line_flow = np.zeros((T, n_line))
    vm = np.zeros((T, len(net.bus)))
    n_fail = 0
    for t in range(T):
        net.load.p_mw = base_p * mult[t]
        net.load.q_mvar = base_q * mult[t]
        try:
            pp.runpp(net)
            line_flow[t] = net.res_line.p_from_mw.values
            vm[t] = net.res_bus.vm_pu.values
        except Exception:
            n_fail += 1
            line_flow[t] = line_flow[t - 1] if t > 0 else 0.0
            vm[t] = vm[t - 1] if t > 0 else 1.0
    if n_fail:
        print(f"  ({n_fail}/{T} AC power-flow solves failed to converge, held over from previous step)")
    return line_flow, vm


def build_stats(line_flow, n_line, t0, t1, split_name, train_std):
    z_block = line_flow[t0:t1]
    Tb = t1 - t0
    noise_std = NOISE_FRAC * train_std
    dither_std = DITHER_FRAC * train_std
    rows = []
    for sidx in range(N_SCENARIOS):
        rng = np.random.default_rng(hash(("ac", split_name, sidx)) % (2**32))
        scenario = generate_scenario(W, Tb, rng)
        e = rng.normal(0.0, noise_std, size=(Tb, n_line))
        d = rng.normal(0.0, dither_std, size=(Tb, n_line))
        z_meas = z_block + e
        replay_of, label_full = scenario_to_labels(scenario, Tb)
        z_enc = z_meas.copy()
        for k, ridx in replay_of.items():
            z_enc[k] = z_meas[ridx] + d[ridx] - d[k]
        for t in range(W - 1, Tb):
            window = z_enc[t - W + 1: t + 1]
            stat = np.concatenate([window.mean(0), window.std(0), window.min(0), window.max(0), window[-1]])
            rows.append({"testbed": "case14_AC", "split": split_name, "scenario": sidx, "t": t,
                         "label": int(label_full[t]), **{f"enc_stat_{i}": v for i, v in enumerate(stat)}})
    return pd.DataFrame(rows)


def main():
    testbed = "case14"
    n_line = 15
    t0 = time.time()
    print("Simulating AC power flow for case14 (3000 steps)...")
    line_flow, vm = simulate_ac_trajectory(testbed, T_TOTAL)
    print(f"  done in {time.time()-t0:.1f}s, line_flow range [{line_flow.min():.1f}, {line_flow.max():.1f}] MW, "
          f"voltage range [{vm.min():.4f}, {vm.max():.4f}] p.u.")

    train_std = line_flow[:1800].std(axis=0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)

    csvs = {}
    for split, (t0s, t1s) in SPLITS.items():
        df = build_stats(line_flow, n_line, t0s, t1s, split, train_std)
        path = OUT / f"case14_AC_{split}_stats.csv"
        df.to_csv(path, index=False)
        csvs[split] = path
        print(f"  {split}: {len(df)} windows, pos_rate={df['label'].mean():.3f}")

    model = PISTRD("case14").fit(str(csvs["train"]), n_line)  # DC-derived H matrix, intentionally mismatched
    model.calibrate(str(csvs["val"]))
    val_df = model.score_split(str(csvs["val"]))
    test_df = model.score_split(str(csvs["test"]))
    m = evaluate_model(val_df["score"].values, val_df["label"].values,
                         test_df["score"].values, test_df["label"].values)
    print("PISTRD on AC-power-flow case14 (DC-mismatched spatial branch):",
          {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})

    m_temporal = evaluate_model(val_df["temporal_score"].values, val_df["label"].values,
                                  test_df["temporal_score"].values, test_df["label"].values)
    m_spatial = evaluate_model(val_df["spatial_score"].values, val_df["label"].values,
                                 test_df["spatial_score"].values, test_df["label"].values)
    print("  temporal-only:", round(m_temporal["macro_f1"], 4))
    print("  spatial-only (DC H, mismatched physics):", round(m_spatial["macro_f1"], 4))

    pd.DataFrame([
        {"variant": "Full", **m},
        {"variant": "Temporal_only", **m_temporal},
        {"variant": "Spatial_only_mismatched_physics", **m_spatial},
    ]).to_csv(OUT / "ac_power_flow_results.csv", index=False)


if __name__ == "__main__":
    main()
