"""Final model: PISTRD (Physics-Informed Spatio-Temporal Residual Detector).

Two complementary branches, OR-fused with per-branch calibrated thresholds:

  TEMPORAL branch (validated in screening as `PIRKSimplified`): a joint
  one-step linear predictor + full-covariance Mahalanobis residual. This is
  what catches the *whole-vector* replay attacks used in the main dataset
  (replayed data is a real historical snapshot -> spatially self-consistent
  -> invisible to any single-instant physics check -> but temporally
  discontinuous relative to the smooth real-load trajectory -> caught here).

  SPATIAL branch (new): a classical weighted-least-squares power-system
  state-estimation residual using the TRUE DC power-flow topology
  (`physics_utils.get_line_observation_matrix`, verified against
  PandaPower's own line flows). A whole-vector replay is invisible to this
  branch by construction (it IS a valid physical snapshot, just from the
  wrong time) -- this branch exists for the robustness experiments testing
  PARTIAL/subset replay attacks (only some sensors spoofed), where spatial
  self-consistency genuinely breaks and this branch is expected to matter
  (see ROBUSTNESS_RESULTS / SOURCE_WEAKNESS_ANALYSIS.md discussion).

Each branch's threshold is calibrated on VAL alone at a target per-branch
false-alarm rate (alpha/2 each, so the OR-fused overall FAR budget is
approximately controlled at alpha via a Bonferroni-style split) --
replacing the source paper's max-of-sample threshold (W1) with an
analytically stated, controllable false-alarm target.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy.stats import chi2

from physics_utils import get_line_observation_matrix


class PISTRD:
    def __init__(self, testbed_name: str, alpha: float = 0.05, ridge_alpha: float = 1.0,
                 cov_reg: float = 1e-3, spatial_reg: float = 1e-2):
        self.testbed_name = testbed_name
        self.alpha = alpha
        self.ridge_alpha = ridge_alpha
        self.cov_reg = cov_reg
        self.spatial_reg = spatial_reg

    # ---- temporal branch ----
    def _fit_temporal(self, df: pd.DataFrame, n_line: int, stream: str):
        last_cols = [f"{stream}_stat_{4 * n_line + i}" for i in range(n_line)]
        X_prev, X_cur = [], []
        for _, g in df.groupby("scenario"):
            z = g[last_cols].values
            X_prev.append(z[:-1]); X_cur.append(z[1:])
        X_prev, X_cur = np.concatenate(X_prev), np.concatenate(X_cur)
        model = Ridge(alpha=self.ridge_alpha).fit(X_prev, X_cur)
        resid = X_cur - model.predict(X_prev)
        cov = np.cov(resid.T) + self.cov_reg * np.eye(n_line)
        return model, np.linalg.inv(cov)

    def _temporal_scores(self, df: pd.DataFrame, n_line: int, stream: str):
        last_cols = [f"{stream}_stat_{4 * n_line + i}" for i in range(n_line)]
        scores = np.zeros(len(df))
        for _, g in df.groupby("scenario"):
            idx = g.index.values
            z = g[last_cols].values
            pred = self.temporal_model.predict(z[:-1])
            resid = z[1:] - pred
            maha = np.einsum("ij,jk,ik->i", resid, self.temporal_Sigma_inv, resid)
            scores[idx[1:]] = maha
            scores[idx[0]] = maha[0] if len(maha) > 0 else 0.0
        return scores

    # ---- spatial branch ----
    def _spatial_scores(self, df: pd.DataFrame, n_line: int, stream: str):
        last_cols = [f"{stream}_stat_{4 * n_line + i}" for i in range(n_line)]
        Z = df[last_cols].values  # (M, n_line), MW
        H = self.H  # (n_line, n_bus)
        HtH_inv = np.linalg.inv(H.T @ H + self.spatial_reg * np.eye(H.shape[1]))
        theta_hat = Z @ H @ HtH_inv.T  # (M, n_bus)  [least-squares angle estimate per row]
        Z_hat = theta_hat @ H.T
        resid = Z - Z_hat
        # scale-normalize by residual variance estimated on this same resid (diag approx)
        var = resid.var(axis=0) + 1e-6
        stat = (resid ** 2 / var).sum(axis=1)
        return stat, var

    def fit(self, train_stats_csv: str, n_line: int, stream: str = "enc"):
        df = pd.read_csv(train_stats_csv).sort_values(["scenario", "t"]).reset_index(drop=True)
        self.temporal_model, self.temporal_Sigma_inv = self._fit_temporal(df, n_line, stream)
        self.H = get_line_observation_matrix(self.testbed_name)
        assert self.H.shape[0] == n_line, f"H rows {self.H.shape[0]} != n_line {n_line}"
        spatial_train_stat, self.spatial_var = self._spatial_scores(df, n_line, stream)
        self.n_line, self.stream = n_line, stream
        self.n_bus = self.H.shape[1]
        return self

    def calibrate(self, val_stats_csv: str):
        """Sets per-branch thresholds from VAL data only, target alpha/2 each."""
        df = pd.read_csv(val_stats_csv).sort_values(["scenario", "t"]).reset_index(drop=True)
        temporal = self._temporal_scores(df, self.n_line, self.stream)
        spatial, _ = self._spatial_scores(df, self.n_line, self.stream)
        normal = (df["label"].values == 0)
        # empirical (1-alpha/2) quantile on normal-only val data for each branch
        self.temporal_threshold = float(np.quantile(temporal[normal], 1 - self.alpha / 2))
        self.spatial_threshold = float(np.quantile(spatial[normal], 1 - self.alpha / 2))
        return self

    def score_split(self, stats_csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(stats_csv_path).sort_values(["scenario", "t"]).reset_index(drop=True)
        temporal = self._temporal_scores(df, self.n_line, self.stream)
        spatial, _ = self._spatial_scores(df, self.n_line, self.stream)
        out = df[["testbed", "split", "scenario", "t", "label"]].copy()
        out["temporal_score"] = temporal
        out["spatial_score"] = spatial
        # normalized OR-fusion score for a single continuous ranking (used for PR-AUC/ROC-AUC);
        # the actual OPERATING decision uses the two calibrated thresholds directly (see `flag`)
        out["score"] = np.maximum(temporal / max(self.temporal_threshold, 1e-9),
                                    spatial / max(self.spatial_threshold, 1e-9))
        out["flag"] = ((temporal > self.temporal_threshold) | (spatial > self.spatial_threshold)).astype(int)
        return out
