"""Shared evaluation harness used by every candidate and every benchmark, so
all numbers in MODEL_SELECTION_REPORT.md / the final benchmark tables come
from one consistent implementation. Threshold selection (when needed) must
only ever be fit on VALIDATION scores/labels, never on TEST -- enforced by
callers passing val_scores/val_y separately from test_scores/test_y."""
import numpy as np
from sklearn.metrics import (f1_score, balanced_accuracy_score, recall_score,
                              precision_score, average_precision_score, roc_auc_score,
                              confusion_matrix)


def best_threshold_by_f1(val_scores: np.ndarray, val_y: np.ndarray) -> float:
    """Sweeps candidate thresholds on VALIDATION only, returns the one
    maximizing macro-F1 on validation."""
    order = np.argsort(val_scores)
    candidates = np.unique(val_scores)
    if len(candidates) > 500:
        candidates = np.quantile(val_scores, np.linspace(0, 1, 500))
    best_t, best_f1 = candidates[0], -1.0
    for t in candidates:
        pred = (val_scores >= t).astype(int)
        f1 = f1_score(val_y, pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
    out = {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "minority_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_pos": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "FPR": fpr, "FNR": fnr,
        "n_pos": int(y_true.sum()), "n": len(y_true),
    }
    try:
        out["PR_AUC"] = average_precision_score(y_true, scores)
    except Exception:
        out["PR_AUC"] = float("nan")
    try:
        out["ROC_AUC"] = roc_auc_score(y_true, scores)
    except Exception:
        out["ROC_AUC"] = float("nan")
    return out


def evaluate_model(val_scores, val_y, test_scores, test_y) -> dict:
    """Standard protocol used everywhere: pick threshold on val (macro-F1
    optimal), report metrics on test at that frozen threshold, plus threshold
    -free metrics (PR-AUC/ROC-AUC) on test."""
    thr = best_threshold_by_f1(val_scores, val_y)
    m = compute_metrics(test_y, test_scores, thr)
    m["threshold_from_val"] = thr
    return m
