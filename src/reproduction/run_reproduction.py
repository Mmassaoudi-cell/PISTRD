"""
Runs the faithful reproduction of the source paper's Table II and Table III
(our-method columns only -- see SOURCE_PAPER_AUDIT.md for why the [19]/[27]
baselines are not byte-for-byte reproduced here) and writes CSV + a
REPRODUCTION_REPORT.md comparing published vs. reproduced numbers.
"""
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cps_qlearning import ExperimentConfig, SourceMethodReproduction

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "reproduction"
OUT.mkdir(parents=True, exist_ok=True)

NOISE_LEVELS = [(0.01, 0.1), (0.02, 0.2), (0.03, 0.3)]
STRATEGIES = [1, 2, 3]

PUBLISHED_TABLE2 = {
    (0.01, 1): (90.00, 0.25), (0.01, 2): (90.50, 1.25), (0.01, 3): (89.58, 1.94),
    (0.02, 1): (90.50, 0.25), (0.02, 2): (89.50, 1.75), (0.02, 3): (91.67, 1.67),
    (0.03, 1): (91.50, 0.50), (0.03, 2): (90.50, 1.50), (0.03, 3): (90.83, 1.67),
}
PUBLISHED_GTH = {0.01: 0.045, 0.02: 0.08, 0.03: 0.12}


def calibrate_threshold(rep: SourceMethodReproduction, n_calib: int, seed0: int):
    """Median of per-realization sup(g_k) over n_calib independent no-attack
    calibration runs. Median (not mean) is used because sup() of a noisy
    statistic is right-skewed across realizations; median best matches a
    single 'representative' calibration run of the kind the paper's Fig. 1(a)
    appears to report (see REPRODUCTION_REPORT.md, threshold-calibration note)."""
    vals = []
    for i in range(n_calib):
        rng = np.random.default_rng(1_000_000 + seed0 + i)
        g, g_th, _ = rep.run_detection(strategy=0, rng=rng, probe_seed=0, g_th=None)
        vals.append(g_th)
    return float(np.median(vals))


def run_grid(n_mc: int, n_calib: int, output_dim: int = 1, probe_std: float = 0.15):
    rows = []
    for sigma_e, sigma_d in NOISE_LEVELS:
        cfg = ExperimentConfig(output_dim=output_dim, sigma_e=sigma_e, sigma_d=sigma_d,
                                probe_std=probe_std)
        rep = SourceMethodReproduction(cfg)
        t0 = time.time()
        g_th = calibrate_threshold(rep, n_calib=n_calib, seed0=int(sigma_e * 1000))
        t_calib = time.time() - t0

        for strategy in STRATEGIES:
            adrs, fars = [], []
            t1 = time.time()
            for i in range(n_mc):
                rng = np.random.default_rng(strategy * 100_000 + int(sigma_e * 100000) + i)
                adr, far, _ = rep.run_detection(strategy=strategy, rng=rng, probe_seed=0, g_th=g_th)
                if not np.isnan(adr):
                    adrs.append(adr)
                if not np.isnan(far):
                    fars.append(far)
            elapsed = time.time() - t1
            pub_adr, pub_far = PUBLISHED_TABLE2[(sigma_e, strategy)]
            rows.append({
                "output_dim": output_dim,
                "sigma_e": sigma_e, "sigma_d": sigma_d, "strategy": strategy,
                "n_mc": n_mc,
                "g_th_reproduced": round(g_th, 4),
                "g_th_published": PUBLISHED_GTH[sigma_e],
                "ADR_reproduced_%": round(float(np.mean(adrs)), 2) if adrs else float("nan"),
                "FAR_reproduced_%": round(float(np.mean(fars)), 2) if fars else float("nan"),
                "ADR_published_%": pub_adr,
                "FAR_published_%": pub_far,
                "calib_time_s": round(t_calib, 1),
                "mc_time_s": round(elapsed, 1),
            })
            print(rows[-1])
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_mc", type=int, default=1000)
    ap.add_argument("--n_calib", type=int, default=200)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.n_mc, args.n_calib = 30, 20

    df1 = run_grid(n_mc=args.n_mc, n_calib=args.n_calib, output_dim=1)
    df1.to_csv(OUT / "table2_reproduced_p1.csv", index=False)
    print(df1.to_string(index=False))
    print("Saved:", OUT / "table2_reproduced_p1.csv")
