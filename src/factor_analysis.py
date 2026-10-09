"""Factorability tests (KMO, Bartlett) and, only if justified, factor analysis."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

from src.common import SEED

FA_VARS = ["annual_inc", "loan_amnt", "dti", "int_rate", "revol_util", "revol_bal", "delinq_2yrs",
           "inq_last_6mths", "credit_history_yrs", "open_acc", "total_acc", "pub_rec", "sub_grade_num"]


def bartlett(R, n):
    p = R.shape[0]
    chi2 = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(R))
    dof = p * (p - 1) / 2
    return float(chi2), float(stats.chi2.sf(chi2, dof))


def kmo(R):
    inv = np.linalg.inv(R)
    d = np.sqrt(np.outer(np.diag(inv), np.diag(inv)))
    partial = -inv / d
    np.fill_diagonal(partial, 0)
    r2 = (R - np.eye(len(R))) ** 2
    p2 = partial ** 2
    per_var = r2.sum(0) / (r2.sum(0) + p2.sum(0))
    return float(r2.sum() / (r2.sum() + p2.sum())), per_var


def run_factor_analysis(df, log=print):
    X = df[FA_VARS].copy()
    X["annual_inc"], X["revol_bal"] = np.log1p(X["annual_inc"]), np.log1p(X["revol_bal"])
    X = X.dropna().sample(min(60_000, len(X)), random_state=SEED)
    Xs = StandardScaler().fit_transform(X.clip(X.quantile(.01), X.quantile(.99), axis=1))
    R = np.corrcoef(Xs, rowvar=False)
    chi2, pval = bartlett(R, len(Xs))
    k, per = kmo(R)
    eig = np.sort(np.linalg.eigvalsh(R))[::-1]
    n_f = int((eig > 1).sum())
    out = {"kmo": k, "bartlett_chi2": chi2, "bartlett_p": pval, "eigenvalues": eig.tolist(),
           "n_factors_kaiser": n_f, "kmo_per_variable": dict(zip(FA_VARS, per.round(3).tolist()))}
    suitable = k >= 0.60 and pval < 0.05
    out["suitable"] = bool(suitable)
    log(f"[FA] KMO={k:.3f} Bartlett p={pval:.3g} eigenvalues>1: {n_f} suitable={suitable}")
    if suitable:
        fa = FactorAnalysis(n_components=n_f, rotation="varimax", random_state=SEED).fit(Xs)
        load = pd.DataFrame(fa.components_.T, index=FA_VARS, columns=[f"Factor {i+1}" for i in range(n_f)])
        out["loadings"] = load
        out["variance_explained"] = ((load ** 2).sum() / len(FA_VARS)).tolist()
    return out
