"""Per-sensor z-score normalization fit on TRAIN only (never val/test), shared
by every neural candidate (STGAT, TCN-CP) and later DL benchmarks."""
import numpy as np


def fit_normalizer(X_train: np.ndarray):
    # X_train: (N, W, n_line)
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    std = np.where(std < 1e-6, 1e-6, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_normalizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)
