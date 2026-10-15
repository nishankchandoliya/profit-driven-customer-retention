"""Model pipelines: the device-age rule, logistic regression and XGBoost."""
import json
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from churn.preprocessing import ChurnPreprocessor

SEED = 42

# settings of the original notebook (notebooks/01)
V1_XGB_PARAMS = {
    "n_estimators": 350,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


class DeviceAgeRule(BaseEstimator, ClassifierMixin):
    """Baseline with no learning: flag a customer as a churner if the device is older than `days`."""

    def __init__(self, days=365):
        self.days = days

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return (X["eqpdays"] > self.days).astype(int).to_numpy()

    def predict_proba(self, X):
        flag = self.predict(X)
        return np.column_stack([1 - flag, flag]).astype(float)


def make_xgb_pipeline(params=None, seed=SEED, extra_features=()):
    """XGBoost after the leak-free preprocessing. `params` overrides the original settings,
    `extra_features` adds feature groups from churn.preprocessing.FEATURE_GROUPS."""
    model_params = {**V1_XGB_PARAMS, **(params or {})}
    model = XGBClassifier(
        **model_params,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("prep", ChurnPreprocessor(extra_features=tuple(extra_features))), ("model", model)])


def make_logreg_pipeline(C=1.0):
    """Logistic regression: scaled numbers + one-hot encoded text columns (rare categories grouped)."""
    encode = ColumnTransformer([
        ("num", StandardScaler(), make_column_selector(dtype_include="number")),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50),
         make_column_selector(dtype_exclude="number")),
    ])
    return Pipeline([
        ("prep", ChurnPreprocessor(encode_categoricals=False)),
        ("encode", encode),
        ("model", LogisticRegression(C=C, max_iter=5000)),
    ])


def cross_val_oof(model, X, y, cv):
    """Cross-validation that returns the ROC-AUC of every fold and the out-of-fold probabilities.

    Each fold refits the whole pipeline (cleaning included) on the other folds only.
    """
    oof = np.zeros(len(y))
    aucs = []
    for train_idx, valid_idx in cv.split(X, y):
        fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        prob = fitted.predict_proba(X.iloc[valid_idx])[:, 1]
        oof[valid_idx] = prob
        aucs.append(roc_auc_score(y.iloc[valid_idx], prob))
    return np.array(aucs), oof


def load_params(path):
    """Read tuned XGBoost settings saved by scripts/tune.py."""
    with open(Path(path)) as f:
        return json.load(f)["params"]
