"""Regression experiment runner shared by the EAD and LGD models."""
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.common import SEED, make_preprocessor, tune

LINEAR_MODELS = ("Linear Regression", "Ridge", "Lasso")
LINEAR_PREFERENCE_MARGIN = 0.02   # prefer a linear model if its val RMSE is within 2% of the best


def reg_candidates(num_cols, cat_cols):
    def prep(scale):
        return make_preprocessor(num_cols, cat_cols, scale=scale)
    return {
        "Linear Regression": (Pipeline([("prep", prep(True)), ("reg", LinearRegression())]), {}),
        "Ridge": (Pipeline([("prep", prep(True)), ("reg", Ridge())]),
                  {"reg__alpha": [1.0, 10.0, 100.0]}),
        "Lasso": (Pipeline([("prep", prep(True)), ("reg", Lasso(max_iter=5000))]),
                  {"reg__alpha": [1e-4, 1e-3, 1e-2]}),
        "Decision Tree": (Pipeline([("prep", prep(False)), ("reg", DecisionTreeRegressor(random_state=SEED))]),
                          {"reg__max_depth": [4, 6, 8], "reg__min_samples_leaf": [100, 300]}),
        "Random Forest": (Pipeline([("prep", prep(False)),
                                    ("reg", RandomForestRegressor(n_estimators=120, max_features=0.5,
                                                                  n_jobs=1, random_state=SEED))]),
                          {"reg__max_depth": [8, 12], "reg__min_samples_leaf": [50, 150]}),
        "Gradient Boosting (HistGB)": (Pipeline([("prep", prep(False)),
                                                 ("reg", HistGradientBoostingRegressor(
                                                     max_iter=250, early_stopping=True,
                                                     l2_regularization=1.0, random_state=SEED))]),
                                       {"reg__learning_rate": [0.05, 0.1], "reg__max_depth": [3, 5]}),
    }


def scores(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": r2_score(y, p)}


def run_regression(tag, X, y, split, num_cols, cat_cols, clip=None, log=print):
    """X, y are already restricted to the rows of interest; split aligned to them."""
    tr, va, te = (split == "train"), (split == "validation"), (split == "test")
    fitted, rows_val, rows_test, tuning = {}, [], [], []
    mu = float(np.mean(y[tr]))
    base_row = lambda m: {"Model": "Baseline (train mean)", **scores(y[m], np.full(int(m.sum()), mu))}
    rows_val.append(base_row(va)); rows_test.append(base_row(te))
    for name, (pipe, grid) in reg_candidates(num_cols, cat_cols).items():
        t0 = time.time()
        if grid:
            model, best, cv = tune(name, pipe, grid, X[tr], y[tr], "neg_root_mean_squared_error",
                                   classification=False)
        else:
            model, best, cv = pipe.fit(X[tr], y[tr]), {}, np.nan
        pv, pt = model.predict(X[va]), model.predict(X[te])
        outside = float(np.mean((pv < clip[0]) | (pv > clip[1]))) if clip else np.nan
        if clip:
            pv, pt = np.clip(pv, *clip), np.clip(pt, *clip)
        fitted[name] = model
        rows_val.append({"Model": name, **scores(y[va], pv), "Pct raw preds outside bounds": outside})
        rows_test.append({"Model": name, **scores(y[te], pt)})
        tuning.append({"Model": name, "best_params": str(best), "cv_neg_rmse": cv,
                       "fit_seconds": round(time.time() - t0, 1)})
        log(f"[{tag}] {name}: val RMSE={rows_val[-1]['RMSE']:.4f} R2={rows_val[-1]['R2']:.4f} "
            f"test RMSE={rows_test[-1]['RMSE']:.4f} R2={rows_test[-1]['R2']:.4f}")
    val_tbl, test_tbl = pd.DataFrame(rows_val), pd.DataFrame(rows_test)
    real = val_tbl[val_tbl.Model != "Baseline (train mean)"]
    top = real.loc[real["RMSE"].idxmin()]
    lin = real[real.Model.isin(LINEAR_MODELS)]
    best_lin = lin.loc[lin["RMSE"].idxmin()]
    if best_lin["RMSE"] <= top["RMSE"] * (1 + LINEAR_PREFERENCE_MARGIN):
        chosen = best_lin["Model"]
        reason = (f"{chosen} is within {LINEAR_PREFERENCE_MARGIN:.0%} of the lowest validation RMSE "
                  f"({top['Model']}: {top['RMSE']:.4f} vs {best_lin['RMSE']:.4f}); the simpler, "
                  "more transparent model is preferred.")
    else:
        chosen = top["Model"]
        reason = (f"{chosen} has the lowest validation RMSE ({top['RMSE']:.4f}); the best linear model "
                  f"({best_lin['Model']}, {best_lin['RMSE']:.4f}) is more than "
                  f"{LINEAR_PREFERENCE_MARGIN:.0%} worse.")
    base_r2 = float(val_tbl.loc[val_tbl.Model == chosen, "R2"].iloc[0])
    return {"fitted": fitted, "val": val_tbl, "test": test_tbl, "tuning": pd.DataFrame(tuning),
            "chosen": chosen, "reason": reason, "chosen_val_r2": base_r2}
