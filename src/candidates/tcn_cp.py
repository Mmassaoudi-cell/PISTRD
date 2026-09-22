"""Candidate 4 (TCN-CP): temporal convolutional window classifier with
conformal-calibrated decision threshold. The TCN itself is trained as a
standard supervised classifier (for fair comparison against every other
candidate/benchmark under the same protocol); what distinguishes this
candidate is its DECISION LAYER: instead of a val-set F1-maximizing
threshold sweep (used by every other model), the operating threshold is set
via split conformal prediction on a normal-only calibration subset of val,
giving a finite-sample false-alarm guarantee -- directly targeting weakness
W1 (fragile, unguaranteed max-of-sample threshold in the source paper).
"""
import numpy as np
import torch
import torch.nn as nn


class TCNBlock(nn.Module):
    def __init__(self, d_in, d_out, k=3, dilation=1):
        super().__init__()
        pad = (k - 1) * dilation
        self.conv = nn.Conv1d(d_in, d_out, k, padding=pad, dilation=dilation)
        self.relu = nn.ReLU()
        self.pad = pad

    def forward(self, x):
        y = self.conv(x)
        if self.pad > 0:
            y = y[:, :, :-self.pad]  # causal: drop future leakage from padding
        return self.relu(y)


class TCNClassifier(nn.Module):
    def __init__(self, n_line, d_model=32):
        super().__init__()
        self.net = nn.Sequential(
            TCNBlock(n_line, d_model, k=3, dilation=1),
            TCNBlock(d_model, d_model, k=3, dilation=2),
            TCNBlock(d_model, d_model, k=3, dilation=4),
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, W, N) -> (B, N, W) for conv1d
        h = self.net(x.permute(0, 2, 1))
        h = h.mean(dim=-1)
        return self.head(h).squeeze(-1)


def train_tcn(X_train, y_train, X_val, y_val, epochs=40, lr=3e-3, batch_size=512,
               device="cpu", seed=0):
    torch.manual_seed(seed)
    n_line = X_train.shape[-1]
    model = TCNClassifier(n_line).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    raw_pw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    pos_weight = torch.tensor([min(raw_pw, 15.0)], device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    n = len(ytr)
    from sklearn.metrics import f1_score
    best_state, best_f1 = None, -1
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logit = model(Xtr[idx])
            loss = lossf(logit, ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = (model(Xv).cpu().numpy() > 0).astype(int)
        f1 = f1_score(y_val, vp, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def score(model, X, device="cpu"):
    model.eval()
    with torch.no_grad():
        logit = model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
    return 1 / (1 + np.exp(-logit))


def conformal_threshold(cal_scores_normal_only: np.ndarray, alpha: float = 0.05) -> float:
    """Split conformal: threshold = the ceil((1-alpha)(n+1))-th order statistic
    of nonconformity scores on normal-only calibration data. Guarantees
    P(false alarm) <= alpha in finite samples under exchangeability."""
    n = len(cal_scores_normal_only)
    s = np.sort(cal_scores_normal_only)
    k = int(np.ceil((1 - alpha) * (n + 1))) - 1
    k = min(k, n - 1)
    return float(s[k])
