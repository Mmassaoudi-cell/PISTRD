"""PISTRD: Physics-Informed Spatio-Temporal Residual Detector.

Calibrated joint-residual detection of replay attacks in power-system
measurement streams. The detector combines two residual statistics:

  Temporal branch
      A joint one-step ridge predictor over all sensors,
          z_hat(k) = A z(k-1) + b,
      scored with the full residual covariance (Mahalanobis distance),
          d_T(k) = r(k)^T Sigma^{-1} r(k),   r(k) = z(k) - z_hat(k).

  Spatial branch
      A regularized DC state-estimation residual built on the line-flow
      observation matrix H (z = H theta),
          theta_hat(k) = (H^T H + lambda_s I)^{-1} H^T z(k),
          d_S(k) = sum_i (z_i(k) - [H theta_hat(k)]_i)^2 / var_i.

Both branches are fitted on the training split only. Each branch threshold is
the empirical (1 - alpha/2) quantile of its statistic on attack-free
validation samples, so the OR-fused false-alarm budget is approximately alpha
(Bonferroni split). At test time the thresholds are frozen and the fused score
is
    s(k) = max(d_T(k) / tau_T, d_S(k) / tau_S),   alarm if s(k) > 1.

Training is closed form (ridge regression, covariance estimation, regularized
least squares): no gradient descent and no random initialization.

Input convention
    A split is a list of chronological sequences, one array of shape (T, m)
    per scenario, where m is the number of measured line flows (MW). The
    measurements are expected to be the decoded output of a synchronized
    dither-encoding layer, which is what makes a replayed segment visible.

Example
    H = line_observation_matrix("case14")            # (m, n_bus)
    model = PISTRD(H).fit(train_sequences)
    model.calibrate(val_sequences, val_labels)
    out = model.score(test_sequences)                # dict of arrays
    alarms = out["flag"]

Dependencies: numpy, scikit-learn, pandapower (only for line_observation_matrix).
"""
import numpy as np
from sklearn.linear_model import Ridge


def line_observation_matrix(testbed: str) -> np.ndarray:
    """DC power-flow matrix H of shape (n_line, n_bus) such that the line
    flows in MW satisfy z = H @ theta, with theta the bus voltage angles in
    radians. `testbed` is a pandapower network name: "case14", "case39",
    "case57", or "case118"."""
    import pandapower as pp
    import pandapower.networks as pn

    net = getattr(pn, testbed)()
    pp.rundcpp(net)
    ppc = net._ppc
    Bf = ppc["internal"]["Bf"]
    Bf = Bf.toarray() if hasattr(Bf, "toarray") else np.asarray(Bf)
    return Bf[: len(net.line), :] * ppc["baseMVA"]


class PISTRD:
    """Physics-Informed Spatio-Temporal Residual Detector.

    Parameters
    ----------
    H : array (m, n_bus)
        Line-flow observation matrix of the network (see line_observation_matrix).
    alpha : float
        Total target false-alarm rate; each branch is calibrated at alpha / 2.
    ridge_alpha : float
        Ridge penalty of the one-step predictor.
    cov_reg : float
        Diagonal loading added to the temporal residual covariance.
    spatial_reg : float
        Tikhonov regularization of the state-estimation normal equations.
    """

    def __init__(self, H: np.ndarray, alpha: float = 0.05, ridge_alpha: float = 1.0,
                 cov_reg: float = 1e-3, spatial_reg: float = 1e-2):
        self.H = np.asarray(H, dtype=float)
        self.alpha = alpha
        self.ridge_alpha = ridge_alpha
        self.cov_reg = cov_reg
        self.spatial_reg = spatial_reg

    # ------------------------------------------------------------------
    # Temporal branch: joint ridge predictor + Mahalanobis residual
    # ------------------------------------------------------------------
    def _fit_temporal(self, sequences):
        # consecutive pairs are formed within each scenario only
        X_prev = np.concatenate([z[:-1] for z in sequences])
        X_cur = np.concatenate([z[1:] for z in sequences])
        self.predictor = Ridge(alpha=self.ridge_alpha).fit(X_prev, X_cur)
        resid = X_cur - self.predictor.predict(X_prev)
        cov = np.cov(resid.T) + self.cov_reg * np.eye(X_cur.shape[1])
        self.Sigma_inv = np.linalg.inv(cov)

    def _temporal_scores(self, sequences):
        scores = []
        for z in sequences:
            resid = z[1:] - self.predictor.predict(z[:-1])
            maha = np.einsum("ij,jk,ik->i", resid, self.Sigma_inv, resid)
            # the first sample of a sequence has no predecessor: reuse the next score
            first = maha[:1] if len(maha) > 0 else np.zeros(1)
            scores.append(np.concatenate([first, maha]))
        return np.concatenate(scores)

    # ------------------------------------------------------------------
    # Spatial branch: regularized DC state-estimation residual
    # ------------------------------------------------------------------
    def _spatial_scores(self, sequences):
        Z = np.concatenate(sequences)                       # (M, m)
        H = self.H
        HtH_inv = np.linalg.inv(H.T @ H + self.spatial_reg * np.eye(H.shape[1]))
        theta_hat = Z @ H @ HtH_inv.T                       # (M, n_bus) angle estimates
        resid = Z - theta_hat @ H.T
        var = resid.var(axis=0) + 1e-6                      # per-line scale of the scored batch
        return (resid ** 2 / var).sum(axis=1)

    # ------------------------------------------------------------------
    # Fit on TRAIN, calibrate on VAL, score with frozen thresholds
    # ------------------------------------------------------------------
    def fit(self, train_sequences):
        """Fits the temporal predictor and residual covariance on the training split."""
        train_sequences = [np.asarray(z, dtype=float) for z in train_sequences]
        m = train_sequences[0].shape[1]
        if self.H.shape[0] != m:
            raise ValueError(f"H has {self.H.shape[0]} rows but the data have {m} sensors")
        self._fit_temporal(train_sequences)
        return self

    def calibrate(self, val_sequences, val_labels):
        """Sets the two branch thresholds from attack-free validation samples.

        val_labels: list of 0/1 arrays aligned with val_sequences (1 = replayed)."""
        val_sequences = [np.asarray(z, dtype=float) for z in val_sequences]
        normal = np.concatenate([np.asarray(y) for y in val_labels]) == 0
        temporal = self._temporal_scores(val_sequences)
        spatial = self._spatial_scores(val_sequences)
        self.temporal_threshold = float(np.quantile(temporal[normal], 1 - self.alpha / 2))
        self.spatial_threshold = float(np.quantile(spatial[normal], 1 - self.alpha / 2))
        return self

    def score(self, sequences):
        """Returns the branch statistics, the fused score, and the alarm flag
        for every sample, concatenated over the given sequences."""
        sequences = [np.asarray(z, dtype=float) for z in sequences]
        temporal = self._temporal_scores(sequences)
        spatial = self._spatial_scores(sequences)
        fused = np.maximum(temporal / max(self.temporal_threshold, 1e-9),
                           spatial / max(self.spatial_threshold, 1e-9))
        flag = ((temporal > self.temporal_threshold) | (spatial > self.spatial_threshold)).astype(int)
        return {"temporal_score": temporal, "spatial_score": spatial, "score": fused, "flag": flag}
