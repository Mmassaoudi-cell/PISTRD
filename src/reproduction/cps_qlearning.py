"""
Faithful reproduction of:
Qin, Zhang, Zhao, "Data-Based Q-Learning for Replay Attack Detection in
Cyber-Physical Systems," IEEE Trans. Ind. Cyber-Phys. Syst., vol.3, 2025.

Implements, from the paper's own equations:
  - discrete-time LQR closed-loop DC-motor testbed (Eq. 31, Section V)
  - batch least-squares Q-learning output estimator (Eqs. 2-18, Algorithm 1)
  - stochastic measurement encoding/decoding scheme (Eq. 19-20)
  - residual-based detector with data-derived threshold (Eq. 21-24)
  - the three replay-attack strategies of Section V

See SOURCE_PAPER_AUDIT.md and REPRODUCTION_REPORT.md for every place an
assumption had to be made because the paper under-specifies an implementation
detail (persistence-of-excitation injection, output dimension p, N, etc.).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations_with_replacement
import numpy as np
from scipy.linalg import solve_discrete_are


# ----------------------------------------------------------------------
# System definition (Eq. 31)
# ----------------------------------------------------------------------
def make_system(output_dim: int = 1):
    """output_dim=1 reproduces the literal C printed in the paper (Eq. 31).
    output_dim=2 is the documented sensitivity variant (see audit item 1)."""
    A = np.array([[-0.0022, -0.0030],
                  [1.0625, 0.9808]])
    B = np.array([[0.9516, 0.0],
                  [6.1303, 4.5728]])
    if output_dim == 1:
        C = np.array([[0.0, 0.170667]])
    elif output_dim == 2:
        C = np.array([[1.0, 0.0],
                       [0.0, 0.170667]])
    else:
        raise ValueError(output_dim)
    return A, B, C


def dlqr(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


# ----------------------------------------------------------------------
# Quadratic-form vectorization helpers (Eqs. 13-17)
# ----------------------------------------------------------------------
def quad_index_pairs(l: int):
    return list(combinations_with_replacement(range(l), 2))


def quad_features(z: np.ndarray, idx_pairs):
    return np.array([z[i] * z[j] for i, j in idx_pairs])


def hbar_to_matrix(hbar: np.ndarray, idx_pairs, l: int):
    H = np.zeros((l, l))
    for (i, j), val in zip(idx_pairs, hbar):
        if i == j:
            H[i, i] = val
        else:
            H[i, j] = val / 2.0
            H[j, i] = val / 2.0
    return H


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    output_dim: int = 1
    N: int = 2                 # historical window length (assumption #4)
    Ln: int = 400
    Lt: int = 1000
    f: int = 3
    eps: float = 0.1
    QL: float = 0.5
    RL: float = 0.5
    Sy: float = 0.2
    Rq: float = 0.1
    sigma_e: float = 0.01       # measurement/comprehensive noise variance
    sigma_d: float = 0.1        # encoding dither variance
    probe_std: float = 0.03     # persistence-of-excitation probe added to u_k for k<Ln (assumption #7)
    x0: tuple = (0.1, -0.1)
    dt: float = 0.1


ATTACK_STRATEGIES = {
    # (attack windows [(start_s, end_s)], record window (start_s, end_s))
    1: {"record": (30.1, 50.0), "attacks": [(60.1, 80.0)]},
    2: {"record": (30.1, 40.0), "attacks": [(60.1, 70.0), (80.1, 90.0)]},
    3: {"record": (42.1, 50.0), "attacks": [(53.1, 61.0), (65.1, 73.0), (85.1, 93.0)]},
}


def s_to_k(t: float, dt: float) -> int:
    return int(round(t / dt))


# ----------------------------------------------------------------------
# Core reproduction engine
# ----------------------------------------------------------------------
class SourceMethodReproduction:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.A, self.B, self.C = make_system(cfg.output_dim)
        self.n = 2
        self.m = 2
        self.p = cfg.output_dim
        Q = cfg.QL * np.eye(self.n)
        R = cfg.RL * np.eye(self.n)
        self.K, _ = dlqr(self.A, self.B, Q, R)
        self.Sy = cfg.Sy * np.eye(self.p)
        self.Rq = cfg.Rq * np.eye(self.m)

        self.N = cfg.N
        self.l = self.p * (self.N + 1) + self.m * self.N  # dimensionally-consistent
        self.idx_pairs = quad_index_pairs(self.l)
        self.min_samples = self.l * (self.l + 1) // 2

        # slice boundaries within the stacked z vector [ybar(Np); ubar(Nm); y(p)]
        self._yb = slice(0, self.p * self.N)
        self._ub = slice(self.p * self.N, self.p * self.N + self.m * self.N)
        self._yc = slice(self.p * self.N + self.m * self.N, self.l)

        self._true_traj_cache = {}

    # -- true (attack-free) physical trajectory, deterministic given a probe seed --
    def true_trajectory(self, probe_seed: int):
        if probe_seed in self._true_traj_cache:
            return self._true_traj_cache[probe_seed]
        cfg = self.cfg
        rng = np.random.default_rng(probe_seed)
        buffer = self.N + cfg.f + 5
        T = cfg.Lt + buffer
        x = np.array(cfg.x0, dtype=float)
        xs = np.zeros((T, self.n))
        us = np.zeros((T, self.m))
        ys = np.zeros((T, self.p))
        for k in range(T):
            u = -self.K @ x
            if k < cfg.Ln:
                u = u + rng.normal(0.0, cfg.probe_std, size=self.m)
            y = self.C @ x
            xs[k] = x
            us[k] = u
            ys[k] = y
            x = self.A @ x + self.B @ u
        out = (xs, us, ys)
        self._true_traj_cache[probe_seed] = out
        return out

    def _z(self, ybar_hist: np.ndarray, ubar_hist: np.ndarray, y_now: np.ndarray):
        return np.concatenate([ybar_hist, ubar_hist, y_now])

    def _hist(self, arr: np.ndarray, k: int):
        """arr[k-1], arr[k-2], ..., arr[k-N] stacked, matching the paper's
        ybar_{k-1,k-N} = [y_{k-1}^T,...,y_{k-N}^T]^T ordering."""
        return np.concatenate([arr[k - i] for i in range(1, self.N + 1)])

    # -- Q-learning batch least-squares estimation of Hy (Eqs. 13-17, Algorithm 1) --
    def learn_Hy(self, u_seq: np.ndarray, ytil_seq: np.ndarray):
        cfg = self.cfg
        N = self.N
        k_min = N
        k_max = cfg.Ln - 2  # need z_k and z_{k+1} both inside [0, Ln-1]
        ks = np.arange(k_min, k_max + 1)

        Zbar = np.zeros((len(ks) + 1, len(self.idx_pairs)))
        for i, k in enumerate([*ks, ks[-1] + 1]):
            ybar = self._hist(ytil_seq, k)
            ubar = self._hist(u_seq, k)
            z = self._z(ybar, ubar, ytil_seq[k])
            Zbar[i] = quad_features(z, self.idx_pairs)

        J = np.array([
            ytil_seq[k] @ self.Sy @ ytil_seq[k] + u_seq[k] @ self.Rq @ u_seq[k]
            for k in ks
        ])

        # Algorithm-1-style convergence: grow the sample window and iterate
        # the least-squares solve until the H_y estimate stabilizes (< eps).
        chunk = max(self.min_samples, 20)
        Hbar_prev = None
        Hbar = np.zeros(len(self.idx_pairs))
        for end in range(chunk, len(ks) + 1, max(chunk // 4, 5)):
            Psi = (Zbar[:end] - Zbar[1:end + 1]).T          # l(l+1)/2 x end
            Gamma = J[:end]
            Hbar, *_ = np.linalg.lstsq(Psi.T, Gamma, rcond=None)
            if Hbar_prev is not None and np.linalg.norm(Hbar - Hbar_prev) < cfg.eps:
                break
            Hbar_prev = Hbar
        else:
            Psi = (Zbar[:-1] - Zbar[1:]).T
            Hbar, *_ = np.linalg.lstsq(Psi.T, J, rcond=None)

        H = hbar_to_matrix(Hbar, self.idx_pairs, self.l)
        Hyy = H[self._yc, self._yc]
        Hy_yb = H[self._yc, self._yb]
        Hy_ub = H[self._yc, self._ub]
        return Hyy, Hy_yb, Hy_ub

    def estimate_y(self, Hyy, Hy_yb, Hy_ub, ybar, ubar):
        rhs = Hy_yb @ ybar + Hy_ub @ ubar
        return -np.linalg.solve(Hyy, rhs)

    # -- full Monte-Carlo detection run for one strategy at fixed noise level --
    def run_detection(self, strategy: int, rng: np.random.Generator,
                       probe_seed: int, g_th: float | None = None):
        cfg = self.cfg
        xs, us, ys_true = self.true_trajectory(probe_seed)
        T = len(ys_true)

        e = rng.normal(0.0, np.sqrt(cfg.sigma_e), size=(T, self.p))
        d = rng.normal(0.0, np.sqrt(cfg.sigma_d), size=(T, self.p))
        ytil = ys_true + e  # "measured" output stream, Eq. 19

        Hyy, Hy_yb, Hy_ub = self.learn_Hy(us, ytil)

        # build attack map: k -> replay_index (or None)
        replay_of = {}
        if strategy != 0:  # strategy 0 = no-attack calibration run
            spec = ATTACK_STRATEGIES[strategy]
            rec_start = s_to_k(spec["record"][0], cfg.dt)
            rec_end = s_to_k(spec["record"][1], cfg.dt)
            for (a_start_s, a_end_s) in spec["attacks"]:
                a_start = s_to_k(a_start_s, cfg.dt)
                a_end = s_to_k(a_end_s, cfg.dt)
                shift = a_start - rec_start
                for k in range(a_start, a_end + 1):
                    ridx = k - shift
                    if rec_start <= ridx <= rec_end:
                        replay_of[k] = ridx

        # received/decoded stream yd_k (Eq. 19-20)
        yd = ytil.copy()
        for k, ridx in replay_of.items():
            yd[k] = ytil[ridx] + d[ridx] - d[k]

        yhat = np.full((T, self.p), np.nan)
        for k in range(cfg.Ln, cfg.Lt + self.N + 2):
            ybar = self._hist(yd, k)
            ubar = self._hist(us, k)
            yhat[k] = self.estimate_y(Hyy, Hy_yb, Hy_ub, ybar, ubar)

        r = yd - yhat  # NaN outside the valid range, fine
        g = np.full(T, np.nan)
        start = cfg.Ln + cfg.f
        for k in range(start, cfg.Lt):
            window = r[k - cfg.f + 1: k + 1]
            g[k] = np.mean(np.sum(window * window, axis=1))

        is_attacked = np.zeros(T, dtype=bool)
        for k in replay_of:
            is_attacked[k] = True

        if g_th is None:
            g_th = np.nanmax(g[start:cfg.Lt])
            return g, g_th, is_attacked

        valid = ~np.isnan(g[start:cfg.Lt])
        gk = g[start:cfg.Lt][valid]
        atk = is_attacked[start:cfg.Lt][valid]
        detected = gk > g_th
        if atk.any():
            adr = 100.0 * detected[atk].mean()
        else:
            adr = np.nan
        if (~atk).any():
            far = 100.0 * detected[~atk].mean()
        else:
            far = np.nan
        return adr, far, g_th
