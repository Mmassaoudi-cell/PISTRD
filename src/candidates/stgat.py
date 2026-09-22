"""Candidate 2 (STGAT): spatio-temporal graph-attention window classifier.
Graphs here are small (<=186 nodes), so attention is computed as dense N x N
scores masked by the adjacency (from graph_utils.build_line_graph) rather
than via a sparse-scatter GNN library -- simpler, exact, and fast enough at
this scale.
"""
import numpy as np
import torch
import torch.nn as nn


class DenseGATLayer(nn.Module):
    def __init__(self, d_in, d_out, n_heads=2):
        super().__init__()
        self.n_heads = n_heads
        self.d_out = d_out
        self.W = nn.Linear(d_in, d_out * n_heads, bias=False)
        self.a_src = nn.Parameter(torch.randn(n_heads, d_out) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(n_heads, d_out) * 0.1)
        self.leaky = nn.LeakyReLU(0.2)
        self.out = nn.Linear(d_out * n_heads, d_out)

    def forward(self, h, adj_mask):
        # h: (B, N, d_in), adj_mask: (N, N) bool (True = edge / allowed)
        B, N, _ = h.shape
        Wh = self.W(h).view(B, N, self.n_heads, self.d_out)  # (B,N,H,D)
        src_score = torch.einsum("bnhd,hd->bnh", Wh, self.a_src)  # (B,N,H)
        dst_score = torch.einsum("bnhd,hd->bnh", Wh, self.a_dst)  # (B,N,H)
        e = self.leaky(src_score.unsqueeze(2) + dst_score.unsqueeze(1))  # (B,N,N,H)
        mask = (~adj_mask).unsqueeze(0).unsqueeze(-1)
        e = e.masked_fill(mask, float("-1e9"))
        alpha = torch.softmax(e, dim=2)  # softmax over source nodes j for each target i
        out = torch.einsum("bijh,bjhd->bihd", alpha, Wh)  # (B,N,H,D)
        out = out.reshape(B, N, self.n_heads * self.d_out)
        return torch.relu(self.out(out))


class STGAT(nn.Module):
    def __init__(self, n_line, window_len, d_model=16, n_heads=2):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.gat1 = DenseGATLayer(d_model, d_model, n_heads=n_heads)
        self.gat2 = DenseGATLayer(d_model, d_model, n_heads=n_heads)
        self.readout = nn.Sequential(
            nn.Linear(n_line * d_model, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, adj_mask):
        # x: (B, W, N) -> per-node temporal conv over W
        B, W, N = x.shape
        xt = x.permute(0, 2, 1).reshape(B * N, 1, W)  # (B*N,1,W)
        h = self.temporal(xt)  # (B*N, d_model, W)
        h = h.mean(dim=-1).view(B, N, -1)  # (B,N,d_model) temporal pooling
        h = self.gat1(h, adj_mask)
        h = self.gat2(h, adj_mask)
        logit = self.readout(h.reshape(B, -1)).squeeze(-1)
        return logit


def train_stgat(X_train, y_train, X_val, y_val, adj_mask, n_line, window_len,
                 epochs=40, lr=3e-3, batch_size=512, device="cpu", seed=0):
    torch.manual_seed(seed)
    model = STGAT(n_line, window_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    raw_pw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    pos_weight = torch.tensor([min(raw_pw, 15.0)], device=device)  # capped: uncapped ~30-50x destabilizes early training
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    adj_mask = torch.tensor(adj_mask, device=device)

    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.float32, device=device)
    n = len(ytr)
    best_state, best_f1 = None, -1
    from sklearn.metrics import f1_score
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr[idx], ytr[idx]
            opt.zero_grad()
            logit = model(xb, adj_mask)
            loss = lossf(logit, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_logit = model(Xv, adj_mask).cpu().numpy()
        val_pred = (val_logit > 0).astype(int)
        f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def score(model, X, adj_mask, device="cpu"):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32, device=device)
        adj = torch.tensor(adj_mask, device=device)
        logit = model(x, adj).cpu().numpy()
    return 1 / (1 + np.exp(-logit))
