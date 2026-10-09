"""Shared building blocks: preprocessing pipelines, data split, tuning helper."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

SEED = 42


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each numeric column to [lo, hi] percentiles learned on TRAIN only."""

    def __init__(self, lo=0.5, hi=99.5):
        self.lo, self.hi = lo, hi

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanpercentile(X, self.lo, axis=0)
        self.hi_ = np.nanpercentile(X, self.hi, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, names=None):
        return np.asarray(names)


def make_preprocessor(num_cols=None, cat_cols=None, scale=True):
    """All fitting (medians, clip bounds, scaler, encoder) happens inside the
    pipeline, so it only ever sees the training fold."""
    num_cols = num_cols or NUMERIC_FEATURES
    cat_cols = cat_cols or CATEGORICAL_FEATURES
    num_steps = [("impute", SimpleImputer(strategy="median")), ("winsor", Winsorizer())]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                              min_frequency=200), cat_cols),
    ])


def split_data(df, stratify_col="pd_target"):
    """60 / 20 / 20 stratified split -> fit (train), val (threshold, calibration,
    model selection), test (touched once for final reporting)."""
    idx = np.arange(len(df))
    tr_val, te = train_test_split(idx, test_size=0.20, stratify=df[stratify_col], random_state=SEED)
    tr, va = train_test_split(tr_val, test_size=0.25, stratify=df[stratify_col].iloc[tr_val],
                              random_state=SEED)
    split = np.empty(len(df), dtype=object)
    split[tr], split[va], split[te] = "train", "validation", "test"
    return split


def tune(name, pipe, grid, X, y, scoring, classification=True, sample=40_000, cv=3):
    """Grid search with k-fold CV on a (stratified) subsample of the TRAIN split,
    then refit the best configuration on the whole train split."""
    if len(X) > sample:
        if classification:
            Xs, _, ys, _ = train_test_split(X, y, train_size=sample, stratify=y, random_state=SEED)
        else:
            Xs, _, ys, _ = train_test_split(X, y, train_size=sample, random_state=SEED)
    else:
        Xs, ys = X, y
    folds = (StratifiedKFold(cv, shuffle=True, random_state=SEED) if classification
             else KFold(cv, shuffle=True, random_state=SEED))
    gs = GridSearchCV(pipe, grid, scoring=scoring, cv=folds, n_jobs=1, refit=False)
    gs.fit(Xs, ys)
    best = {k: v for k, v in gs.best_params_.items()}
    pipe.set_params(**best)
    pipe.fit(X, y)
    return pipe, best, float(gs.best_score_)


class SigmoidCalibrator:
    """Platt scaling: logistic regression on logit(raw probability), fitted on VALIDATION data.
    Smooth and monotone - unlike isotonic regression it cannot saturate at exactly 0 or 1."""

    def fit(self, p, y):
        self.lr_ = LogisticRegression(C=1e6, max_iter=500).fit(self._z(p), y)
        return self

    @staticmethod
    def _z(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def predict(self, p):
        return self.lr_.predict_proba(self._z(p))[:, 1]
