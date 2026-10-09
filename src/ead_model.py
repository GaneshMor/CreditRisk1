"""EAD proxy model.

Target: ead_origination_proxy (= funded amount). This is NOT regulatory EAD.
The model estimates the *typical funded exposure for a borrower profile*; the
requested amount / instalment-based features are excluded from the predictors
because they are deterministic functions of the target.
"""
import numpy as np
from src.preprocessing import CATEGORICAL_FEATURES, EAD_FEATURES, NUMERIC_FEATURES
from src.regression_common import run_regression

EAD_CLIP = (500.0, 35_000.0)   # observed min / max funded amount in the data


def run_ead_experiments(df, split, log=print):
    num = [c for c in NUMERIC_FEATURES if c in EAD_FEATURES]
    cat = [c for c in CATEGORICAL_FEATURES if c in EAD_FEATURES]
    X, y = df[EAD_FEATURES], df["ead_origination_proxy"].values
    return run_regression("EAD", X, y, split, num, cat, clip=EAD_CLIP, log=log)
