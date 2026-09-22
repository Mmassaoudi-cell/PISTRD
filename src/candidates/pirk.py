"""Candidate 3 (PIRK, screening-stage simplification): a linear state-space
one-step predictor z_hat_t = A z_{t-1} + b fit by ridge regression on TRAIN
only, with a Mahalanobis-distance detection statistic using a residual
covariance also estimated on TRAIN only (a fixed-gain steady-state
simplification of a full Kalman filter, appropriate for candidate screening
under the agreed time budget -- see MODEL_SELECTION_REPORT.md). If PIRK wins
screening, the final model replaces the data-driven A with the DC
power-flow topology matrix H (true physics-informed version) per
MODEL_CANDIDATES.md Candidate 3.

Score = Mahalanobis distance of the one-step residual at each window's
current time step ('last' raw value, matching RLS-CUSUM's convention).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


class PIRKSimplified:
    def __init__(self, alpha: float = 1.0, cov_reg: float = 1e-3):
        self.alpha = alpha
        self.cov_reg = cov_reg
        self.model = None
        self.Sigma_inv = None

    def fit(self, train_stats_csv: str, n_line: int, stream: str = "enc"):
        df = pd.read_csv(train_stats_csv).sort_values(["scenario", "t"])
        last_cols = [f"{stream}_stat_{4 * n_line + i}" for i in range(n_line)]
        X_prev, X_cur = [], []
        for _, g in df.groupby("scenario"):
            z = g[last_cols].values
            X_prev.append(z[:-1])
            X_cur.append(z[1:])
        X_prev = np.concatenate(X_prev)
        X_cur = np.concatenate(X_cur)
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_prev, X_cur)
        resid = X_cur - self.model.predict(X_prev)
        cov = np.cov(resid.T) + self.cov_reg * np.eye(n_line)
        self.Sigma_inv = np.linalg.inv(cov)
        self.n_line = n_line
        self.stream = stream
        return self

    def score_split(self, stats_csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(stats_csv_path).sort_values(["scenario", "t"]).reset_index(drop=True)
        last_cols = [f"{self.stream}_stat_{4 * self.n_line + i}" for i in range(self.n_line)]
        scores = np.zeros(len(df))
        for _, g in df.groupby("scenario"):
            idx = g.index.values
            z = g[last_cols].values
            pred = self.model.predict(z[:-1])
            resid = z[1:] - pred
            maha = np.einsum("ij,jk,ik->i", resid, self.Sigma_inv, resid)
            scores[idx[1:]] = maha
            scores[idx[0]] = maha[0] if len(maha) > 0 else 0.0
        df["score"] = scores
        return df[["testbed", "split", "scenario", "t", "label", "score"]]
