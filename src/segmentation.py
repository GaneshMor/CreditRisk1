"""K-Means customer risk segmentation (unsupervised; outcome is NOT used to cluster)."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.common import SEED
from src.preprocessing import SEGMENT_FEATURES


def prepare(X, bounds=None):
    X = X[SEGMENT_FEATURES].copy()
    X["annual_inc"] = np.log1p(X["annual_inc"])
    X["loan_amnt"] = np.log1p(X["loan_amnt"])
    X = X.fillna({"emp_length_yrs": 5.0})
    if bounds is None:
        bounds = (X.quantile(0.01), X.quantile(0.99))
    lo, hi = pd.Series(bounds[0]), pd.Series(bounds[1])   # accepts Series or plain dicts
    return X.clip(lo, hi, axis=1), bounds


def run_segmentation(df, k_range=range(2, 9), k_allowed=(3, 6), log=print):
    Z, bounds = prepare(df)
    Z = Z.fillna(Z.median())
    scaler = StandardScaler().fit(Z)
    A = scaler.transform(Z)
    rng = np.random.RandomState(SEED)
    sil_idx = rng.choice(len(A), 20_000, replace=False)
    rows = []
    for k in k_range:
        km = KMeans(k, n_init=5, random_state=SEED).fit(A)
        sil = silhouette_score(A[sil_idx], km.labels_[sil_idx])
        rows.append({"K": k, "Inertia": km.inertia_, "Silhouette": sil})
        log(f"[SEG] K={k} inertia={km.inertia_:.0f} silhouette={sil:.4f}")
    sel = pd.DataFrame(rows)
    cand = sel[(sel.K >= k_allowed[0]) & (sel.K <= k_allowed[1])]
    k_best = int(cand.loc[cand["Silhouette"].idxmax(), "K"])
    km = KMeans(k_best, n_init=20, random_state=SEED).fit(A)
    lab = km.labels_
    prof = df.assign(_c=lab).groupby("_c").agg(
        size=("pd_target", "size"), default_rate=("pd_target", "mean"),
        income=("annual_inc", "mean"), loan_amnt=("loan_amnt", "mean"), dti=("dti", "mean"),
        int_rate=("int_rate", "mean"), revol_util=("revol_util", "mean"),
        delinq_2yrs=("delinq_2yrs", "mean"), inq_6m=("inq_last_6mths", "mean"),
        history_yrs=("credit_history_yrs", "mean"), emp_len=("emp_length_yrs", "mean"))
    # rank clusters by OBSERVED default rate (after clustering) -> Segment 1 = lowest risk
    order = prof["default_rate"].rank(method="first").astype(int)
    mapping = {int(c): int(order[c]) for c in prof.index}
    # auto-descriptor from the most extreme standardised means
    glob_mu, glob_sd = df[SEGMENT_FEATURES].mean(), df[SEGMENT_FEATURES].std()
    names = {"annual_inc": "income", "loan_amnt": "loan size", "dti": "DTI", "int_rate": "interest rate",
             "revol_util": "revolving utilisation", "delinq_2yrs": "delinquencies",
             "inq_last_6mths": "credit inquiries", "credit_history_yrs": "credit history",
             "emp_length_yrs": "employment length", "open_acc": "open accounts"}
    desc = {}
    for c in prof.index:
        m = df.loc[lab == c, SEGMENT_FEATURES].mean()
        z = ((m - glob_mu) / glob_sd).sort_values()
        hi = [f"high {names[f]}" for f, v in z[::-1].items() if v > 0.35][:3]
        lo = [f"low {names[f]}" for f, v in z.items() if v < -0.35][:3]
        desc[mapping[int(c)]] = ", ".join(hi + lo) or "close to portfolio average"
    prof = prof.rename(index=mapping).sort_index()
    prof["description"] = [desc[i] for i in prof.index]
    pca = PCA(2, random_state=SEED).fit(A)
    pidx = rng.choice(len(A), 8_000, replace=False)
    xy = pca.transform(A[pidx])
    viz = pd.DataFrame({"PC1": xy[:, 0], "PC2": xy[:, 1],
                        "segment": [mapping[int(c)] for c in lab[pidx]]})
    # plain-python types only (no pandas objects) so the pickle loads under any pandas version
    art = {"bounds": (bounds[0].to_dict(), bounds[1].to_dict()), "scaler": scaler, "kmeans": km,
           "mapping": mapping, "k": k_best, "median": {k: float(v) for k, v in Z.median().items()},
           "description": desc}
    seg = np.array([mapping[int(c)] for c in lab])
    return {"artifact": art, "selection": sel, "profile": prof, "pca": viz, "segment": seg,
            "pca_var": pca.explained_variance_ratio_.tolist()}


def assign_segment(art, X):
    Z, _ = prepare(X, art["bounds"])
    Z = Z.fillna(art["median"])
    c = art["kmeans"].predict(art["scaler"].transform(Z))
    return np.array([art["mapping"][int(i)] for i in c])
