"""Candidate 1 (RLS-CUSUM): per-sensor online AR(1) predictor (recursive,
least-mean-squares update) + CUSUM change-detection score, run in
chronological order within each attack scenario (state resets at scenario
boundaries). Operates on the 'last' raw measurement per window (the stats
CSV's enc_stat/plain_stat block index 4), which is exactly z_t at the
window's current time step.

Implements Candidate 1 from MODEL_CANDIDATES.md: replaces the source paper's
frozen batch least-squares H_y with an online recursive update, and replaces
its max-of-sample threshold with a CUSUM statistic against an adaptive
exponential-moving-average residual-energy baseline.
"""
import numpy as np
import pandas as pd


def _rls_cusum_scores(z: np.ndarray, ema_lam: float = 0.98, nu_mult: float = 1.5,
                       lms_mu: float = 0.02) -> np.ndarray:
    """z: (T, n_line) chronological raw values for ONE scenario.
    Returns per-timestep CUSUM anomaly score (T,)."""
    T, n = z.shape
    a = np.ones(n)       # per-sensor AR(1) coefficient (online-updated)
    b = np.zeros(n)       # per-sensor bias (online-updated)
    S = np.zeros(n)       # CUSUM accumulator per sensor
    nu = np.zeros(n)      # adaptive residual-energy baseline (EMA)
    scores = np.zeros(T)
    prev = z[0]
    for t in range(1, T):
        pred = a * prev + b
        resid = z[t] - pred

        nu = ema_lam * nu + (1 - ema_lam) * (resid ** 2)
        S = np.maximum(0.0, S + resid ** 2 - nu_mult * nu)
        S = np.minimum(S, 1e12)  # guard against runaway accumulation on the first few steps
        scores[t] = S.sum()

        # normalized LMS (NLMS): step size scaled by local input energy so the
        # update is stable regardless of each sensor's raw MW magnitude.
        norm = prev ** 2 + 1.0 + 1e-6
        a = a + lms_mu * resid * prev / norm
        b = b + lms_mu * resid / norm
        prev = z[t]
    scores[0] = scores[1] if T > 1 else 0.0
    return scores


def score_split(stats_csv_path: str, n_line: int, stream: str = "enc") -> pd.DataFrame:
    df = pd.read_csv(stats_csv_path)
    last_cols = [f"{stream}_stat_{4 * n_line + i}" for i in range(n_line)]
    df = df.sort_values(["scenario", "t"]).reset_index(drop=True)
    scores = np.zeros(len(df))
    for _, g in df.groupby("scenario"):
        idx = g.index.values
        z = g[last_cols].values
        s = _rls_cusum_scores(z)
        scores[idx] = s
    df["score"] = scores
    return df[["testbed", "split", "scenario", "t", "label", "score"]]
