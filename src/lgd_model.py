"""LGD proxy model.

Target: lgd_realized_proxy = 1 - (recoveries - recovery fees) / outstanding principal at default,
clipped to [0, 1]. Defined ONLY for defaulted loans, so the model is conditional on default.
This is a realised-loss proxy, NOT a Basel / point-in-time LGD.
"""
from src.preprocessing import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from src.regression_common import run_regression


def run_lgd_experiments(df, split, log=print):
    m = (df["pd_target"] == 1) & df["lgd_realized_proxy"].notna()
    sub, sp = df[m.values], split[m.values]
    X, y = sub[FEATURES], sub["lgd_realized_proxy"].values
    return run_regression("LGD", X, y, sp, NUMERIC_FEATURES, CATEGORICAL_FEATURES, clip=(0.0, 1.0), log=log)
