"""Probability-of-Default modelling: train several models, compare, select."""
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.common import SEED, SigmoidCalibrator, make_preprocessor, tune
from src.preprocessing import FEATURES

LR_PREFERENCE_MARGIN = 0.005   # prefer the interpretable logit if within this ROC-AUC of the best


def candidates():
    return {
        "Logistic Regression": (
            Pipeline([("prep", make_preprocessor(scale=True)),
                      ("clf", LogisticRegression(max_iter=600))]),
            {"clf__C": [0.01, 0.1, 1.0]}),
        "Decision Tree": (
            Pipeline([("prep", make_preprocessor(scale=False)),
                      ("clf", DecisionTreeClassifier(random_state=SEED))]),
            {"clf__max_depth": [4, 6, 8], "clf__min_samples_leaf": [100, 300]}),
        "Random Forest": (
            Pipeline([("prep", make_preprocessor(scale=False)),
                      ("clf", RandomForestClassifier(n_estimators=150, max_features="sqrt",
                                                     n_jobs=1, random_state=SEED))]),
            {"clf__max_depth": [8, 12], "clf__min_samples_leaf": [50, 150]}),
        "Gradient Boosting (HistGB)": (
            Pipeline([("prep", make_preprocessor(scale=False)),
                      ("clf", HistGradientBoostingClassifier(max_iter=250, early_stopping=True,
                                                             l2_regularization=1.0,
                                                             random_state=SEED))]),
            {"clf__learning_rate": [0.05, 0.1], "clf__max_depth": [3, 5]}),
    }


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.nanargmax(f1))])


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def metrics_at(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return {"Threshold": thr, "Accuracy": accuracy_score(y, yhat),
            "Precision": precision_score(y, yhat, zero_division=0),
            "Recall": recall_score(y, yhat), "F1": f1_score(y, yhat),
            "ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p),
            "Gini": 2 * roc_auc_score(y, p) - 1, "KS": ks_stat(y, p),
            "Brier": brier_score_loss(y, p), "TN": tn, "FP": fp, "FN": fn, "TP": tp}


def run_pd_experiments(df, split, log=print):
    X, y = df[FEATURES], df["pd_target"].values
    tr, va, te = (split == "train"), (split == "validation"), (split == "test")
    fitted, rows_val, rows_test, rows_test_05, tuning = {}, [], [], [], []
    curves = {}
    for name, (pipe, grid) in candidates().items():
        t0 = time.time()
        model, best, cv_auc = tune(name, pipe, grid, X[tr], y[tr], "roc_auc")
        raw_val = model.predict_proba(X[va])[:, 1]
        # sigmoid (Platt) calibration on VALIDATION only, so PD is a true probability
        iso = SigmoidCalibrator().fit(raw_val, y[va])
        p_val = iso.predict(raw_val)
        p_test = iso.predict(model.predict_proba(X[te])[:, 1])
        thr = best_f1_threshold(y[va], p_val)
        fitted[name] = {"model": model, "calibrator": iso, "threshold": thr}
        rows_val.append({"Model": name, **metrics_at(y[va], p_val, thr)})
        rows_test.append({"Model": name, **metrics_at(y[te], p_test, thr)})
        rows_test_05.append({"Model": name, **metrics_at(y[te], p_test, 0.5)})
        tuning.append({"Model": name, "best_params": str(best), "cv_roc_auc": cv_auc,
                       "fit_seconds": round(time.time() - t0, 1)})
        fpr, tpr, _ = roc_curve(y[te], p_test)
        pr, rc, _ = precision_recall_curve(y[te], p_test)
        curves[name] = {"fpr": fpr[::20], "tpr": tpr[::20], "prec": pr[::20], "rec": rc[::20]}
        log(f"[PD] {name}: val AUC={rows_val[-1]['ROC-AUC']:.4f} test AUC={rows_test[-1]['ROC-AUC']:.4f} "
            f"({tuning[-1]['fit_seconds']}s) {best}")
    val_tbl, test_tbl = pd.DataFrame(rows_val), pd.DataFrame(rows_test)

    # ---- documented selection rule (validation set only)
    top = val_tbl.loc[val_tbl["ROC-AUC"].idxmax()]
    lr_auc = val_tbl.loc[val_tbl.Model == "Logistic Regression", "ROC-AUC"].iloc[0]
    if top["ROC-AUC"] - lr_auc <= LR_PREFERENCE_MARGIN:
        chosen = "Logistic Regression"
        reason = (f"Logistic Regression is within {LR_PREFERENCE_MARGIN:.3f} ROC-AUC of the best model "
                  f"({top['Model']}, {top['ROC-AUC']:.4f} vs {lr_auc:.4f} on validation); the "
                  "interpretable scorecard-style model is preferred for banking use.")
    else:
        chosen = top["Model"]
        reason = (f"{chosen} has the highest validation ROC-AUC ({top['ROC-AUC']:.4f}); the gap to "
                  f"Logistic Regression ({lr_auc:.4f}) exceeds {LR_PREFERENCE_MARGIN:.3f}, so the "
                  "extra complexity is justified. Explainability is provided via permutation "
                  "importance and local sensitivity analysis.")
    return {"fitted": fitted, "val": val_tbl, "test": test_tbl, "test_at_0.5": pd.DataFrame(rows_test_05),
            "tuning": pd.DataFrame(tuning), "chosen": chosen, "reason": reason, "curves": curves}


def risk_band_thresholds(p_val, q_low=0.40, q_high=0.80):
    """Data-driven bands: cut-offs at the 40th / 80th percentile of calibrated PD on the
    validation set (Low = lowest 40%, Medium = next 40%, High = top 20%)."""
    return {"low_max": float(np.quantile(p_val, q_low)), "high_min": float(np.quantile(p_val, q_high))}


def assign_band(p, th):
    p = np.asarray(p)
    return np.where(p < th["low_max"], "Low", np.where(p < th["high_min"], "Medium", "High"))
