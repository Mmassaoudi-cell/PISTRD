"""SOURCE_METHOD_REPRODUCTION, adapted to the multi-bus power-system windows:
the mandatory benchmark representing the source paper's own recipe applied
to this domain -- independent per-line batch OLS one-step predictor (fit
once on TRAIN, frozen, no online adaptation -- exactly the source paper's
Algorithm 1 structure) and per-line residual energy as the score. Unlike the
paper's own max-of-sample threshold, this benchmark uses the SAME
val-F1-optimal threshold-selection protocol as every other benchmark here,
so the comparison in the main results table is apples-to-apples (this makes
the baseline stronger than the paper's literal max-of-sample rule would,
which is the fair/conservative choice per REPRODUCTION_REPORT.md's
diagnosis of that rule's fragility).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class SourceMethodAdapted:
    def __init__(self):
        self.models = None  # per-line independent OLS

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
        self.models = []
        for i in range(n_line):
            m = LinearRegression().fit(X_prev[:, i:i + 1], X_cur[:, i])
            self.models.append(m)
        self.n_line, self.stream = n_line, stream
        return self

    def score_split(self, stats_csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(stats_csv_path).sort_values(["scenario", "t"]).reset_index(drop=True)
        last_cols = [f"{self.stream}_stat_{4 * self.n_line + i}" for i in range(self.n_line)]
        scores = np.zeros(len(df))
        for _, g in df.groupby("scenario"):
            idx = g.index.values
            z = g[last_cols].values
            pred = np.stack([self.models[i].predict(z[:-1, i:i + 1]) for i in range(self.n_line)], axis=1)
            resid = z[1:] - pred
            g_k = (resid ** 2).mean(axis=1)  # source-paper-style residual-energy statistic
            scores[idx[1:]] = g_k
            scores[idx[0]] = g_k[0] if len(g_k) > 0 else 0.0
        df["score"] = scores
        return df[["testbed", "split", "scenario", "t", "label", "score"]]
