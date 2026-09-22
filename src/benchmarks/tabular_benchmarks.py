"""Classical/tree/boosting/MLP benchmark suite operating on the aggregated
per-window statistical features (*_stats.csv). All preprocessing (scaling) is
fit on TRAIN only. Each benchmark returns a probability-like score used by
the shared eval harness (src/eval/metrics.py) with a threshold picked on VAL.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


FEATURE_PREFIX = "enc_stat_"  # use the encoded-stream stats by default (source-method-consistent)


def load_xy(stats_csv_path: str, feature_prefix: str = FEATURE_PREFIX):
    df = pd.read_csv(stats_csv_path)
    cols = [c for c in df.columns if c.startswith(feature_prefix)]
    X = df[cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)
    return X, y, cols


def get_models(seed: int = 0):
    return {
        "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=seed),
        "SVM_RBF": SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=seed, cache_size=500),
        "kNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                 max_depth=None, n_jobs=-1, random_state=seed),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                                             n_jobs=-1, random_state=seed),
        "HistGB": HistGradientBoostingClassifier(max_iter=300, random_state=seed),
        "XGBoost": xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                       eval_metric="logloss", n_jobs=-1, random_state=seed,
                                       tree_method="hist"),
        "LightGBM": lgb.LGBMClassifier(n_estimators=300, max_depth=-1, learning_rate=0.1,
                                         class_weight="balanced", n_jobs=-1, random_state=seed, verbosity=-1),
        "CatBoost": CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                                         auto_class_weights="Balanced", verbose=False, random_state=seed),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed, early_stopping=True),
    }


def fit_score(name: str, model, Xtr, ytr, Xval, Xtest):
    scaler = StandardScaler().fit(Xtr)  # fit on TRAIN only
    Xtr_s, Xval_s, Xtest_s = scaler.transform(Xtr), scaler.transform(Xval), scaler.transform(Xtest)

    if name in ("XGBoost",):
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        model.set_params(scale_pos_weight=spw)
    model.fit(Xtr_s, ytr)
    if hasattr(model, "predict_proba"):
        val_s = model.predict_proba(Xval_s)[:, 1]
        test_s = model.predict_proba(Xtest_s)[:, 1]
    else:
        val_s = model.decision_function(Xval_s)
        test_s = model.decision_function(Xtest_s)
    return val_s, test_s
