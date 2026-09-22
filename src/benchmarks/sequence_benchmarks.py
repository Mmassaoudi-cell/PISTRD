"""LSTM and Transformer sequence benchmarks operating on raw windowed
measurements (X_encoded from *.npz), trained as supervised classifiers with
the same protocol used for every candidate (train on TRAIN, threshold on
VAL). Also includes a plain-TCN (no conformal calibration) as the direct
ablation counterpart to Candidate 4 (TCN-CP), isolating the calibration
mechanism's contribution from the encoder architecture's."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


class LSTMClassifier(nn.Module):
    def __init__(self, n_line, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_line, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)


class TransformerClassifier(nn.Module):
    def __init__(self, n_line, d_model=32, n_heads=4, n_layers=2):
        super().__init__()
        self.proj = nn.Linear(n_line, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=64,
                                             batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.proj(x)
        h = self.enc(h)
        h = h.mean(dim=1)
        return self.head(h).squeeze(-1)


def _train_generic(model_cls, X_train, y_train, X_val, y_val, epochs=25, lr=3e-3,
                    batch_size=512, seed=0, **model_kwargs):
    torch.manual_seed(seed)
    n_line = X_train.shape[-1]
    model = model_cls(n_line, **model_kwargs)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    raw_pw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    pos_weight = torch.tensor([min(raw_pw, 15.0)])
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    n = len(ytr)
    best_state, best_f1 = None, -1
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = (model(Xv).numpy() > 0).astype(int)
        f1 = f1_score(y_val, vp, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def train_lstm(*args, **kwargs):
    return _train_generic(LSTMClassifier, *args, **kwargs)


def train_transformer(*args, **kwargs):
    return _train_generic(TransformerClassifier, *args, **kwargs)


def score(model, X):
    model.eval()
    with torch.no_grad():
        logit = model(torch.tensor(X, dtype=torch.float32)).numpy()
    return 1 / (1 + np.exp(-logit))
