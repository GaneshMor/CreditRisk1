"""End-to-end pipeline:  data -> EDA -> split -> PD/EAD/LGD experiments -> selection ->
segmentation -> factor analysis -> explainability -> portfolio scoring -> artifacts.

Usage:  python -m src.run_pipeline --raw path/to/loan_data_2007_2014.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import SEED, split_data                                  # noqa: E402
from src.ead_model import EAD_CLIP, run_ead_experiments                   # noqa: E402
from src.factor_analysis import run_factor_analysis                       # noqa: E402
from src.lgd_model import run_lgd_experiments                             # noqa: E402
from src.pd_model import assign_band, risk_band_thresholds, run_pd_experiments  # noqa: E402
from src.preprocessing import (CATEGORICAL_FEATURES, EAD_FEATURES, FEATURES,   # noqa: E402
                               NUMERIC_FEATURES, POST_ORIGINATION_EXCLUDED,
                               build_modeling_dataset)
from src.segmentation import assign_segment, run_segmentation            # noqa: E402

MODELS, REPORTS, FIGS, DATA = ROOT / "models", ROOT / "reports", ROOT / "reports/figures", ROOT / "data"
t_start = time.time()


def log(msg):
    print(f"[{time.time() - t_start:7.0f}s] {msg}", flush=True)


def save_fig(name):
    plt.tight_layout()
    plt.savefig(FIGS / name, dpi=130)
    plt.close()


def main(raw):
    for p in (MODELS, REPORTS, FIGS, DATA):
        p.mkdir(parents=True, exist_ok=True)
    # ------------------------------------------------------------ 1. data
    rep = {}
    df = build_modeling_dataset(raw, rep)
    log(f"modelling dataset: {df.shape}")
    df.to_csv(DATA / "LendingClub_2007_2014_modeling_ready.csv.gz", index=False, compression="gzip")

    # ------------------------------------------------------------ 2. EDA / data quality
    n_total = rep["raw_rows"]
    quality = {
        "raw_rows": n_total, "duplicate_ids": rep["duplicate_ids"],
        "status_counts": rep["status_counts"],
        "rows_resolved_kept": int(len(df)),
        "rows_excluded_censored": int(n_total - len(df)),
        "default_rate_resolved": float(df.pd_target.mean()),
        "default_rate_by_issue_year": df.groupby("issue_year").pd_target.mean().round(4).to_dict(),
        "missing_pct": (df[FEATURES].isna().mean() * 100).round(2).sort_values(ascending=False).head(12).to_dict(),
        "lgd_proxy_summary": df.lgd_realized_proxy.describe().round(4).to_dict(),
        "lgd_share_at_100pct": float((df.lgd_realized_proxy == 1).sum() / df.lgd_realized_proxy.notna().sum()),
        "no_fico_in_dataset": True,
    }
    df[FEATURES + ["ead_origination_proxy"]].describe().T.round(3).to_csv(REPORTS / "eda_summary_stats.csv")
    corr = df[NUMERIC_FEATURES + ["pd_target"]].corr(numeric_only=True)
    corr.round(3).to_csv(REPORTS / "eda_correlations.csv")
    pairs = (corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack().rename("corr").reset_index())
    pairs = pairs[(pairs["corr"].abs() > 0.7) & (pairs.level_0 != "pd_target") & (pairs.level_1 != "pd_target")]
    quality["redundant_pairs_abs_corr_gt_0.7"] = pairs.round(3).values.tolist()
    quality["corr_with_default"] = corr["pd_target"].drop("pd_target").round(3).sort_values().to_dict()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    df.groupby("grade").pd_target.mean().plot.bar(ax=ax[0], color="#7b4a24", title="Default rate by grade")
    df.groupby("issue_year").pd_target.mean().plot(ax=ax[1], color="#7b4a24", marker="o",
                                                   title="Default rate by issue year (censoring effect)")
    df.lgd_realized_proxy.dropna().plot.hist(bins=40, ax=ax[2], color="#a8692c", title="LGD proxy (defaults)")
    save_fig("eda_overview.png")

    # ------------------------------------------------------------ 3. split
    split = split_data(df)
    df["split"] = split
    log("split: " + str(pd.Series(split).value_counts().to_dict()))

    # ------------------------------------------------------------ 4. PD
    pd_res = run_pd_experiments(df, split, log)
    chosen = pd_res["chosen"]
    fit = pd_res["fitted"][chosen]
    va = split == "validation"
    p_val = fit["calibrator"].predict(fit["model"].predict_proba(df.loc[va, FEATURES])[:, 1])
    bands = risk_band_thresholds(p_val)
    te = split == "test"
    p_te = fit["calibrator"].predict(fit["model"].predict_proba(df.loc[te, FEATURES])[:, 1])
    band_tbl = (pd.DataFrame({"band": assign_band(p_te, bands), "pd": p_te, "y": df.loc[te, "pd_target"].values})
                .groupby("band").agg(n=("y", "size"), mean_pd=("pd", "mean"), observed_default_rate=("y", "mean"))
                .reindex(["Low", "Medium", "High"]))
    log(f"PD chosen: {chosen}; bands {bands}\n{band_tbl}")
    pd_res["val"].to_csv(REPORTS / "pd_model_comparison_validation.csv", index=False)
    pd_res["test"].to_csv(REPORTS / "pd_model_comparison_test.csv", index=False)
    pd_res["test_at_0.5"].to_csv(REPORTS / "pd_model_comparison_test_threshold_0.5.csv", index=False)
    pd_res["tuning"].to_csv(REPORTS / "pd_tuning.csv", index=False)
    band_tbl.to_csv(REPORTS / "pd_risk_band_validation.csv")
    joblib.dump({"name": chosen, "model": fit["model"], "calibrator": fit["calibrator"],
                 "threshold": fit["threshold"], "bands": bands}, MODELS / "pd_model.joblib", compress=3)
    lr = pd_res["fitted"]["Logistic Regression"]
    joblib.dump({"name": "Logistic Regression", "model": lr["model"], "calibrator": lr["calibrator"],
                 "threshold": lr["threshold"]}, MODELS / "pd_logreg.joblib", compress=3)

    # curves + confusion matrix
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for n, c in pd_res["curves"].items():
        ax[0].plot(c["fpr"], c["tpr"], label=n)
        ax[1].plot(c["rec"], c["prec"], label=n)
    ax[0].plot([0, 1], [0, 1], "k--", lw=.8); ax[0].set(title="ROC (test)", xlabel="FPR", ylabel="TPR")
    ax[1].set(title="Precision-Recall (test)", xlabel="Recall", ylabel="Precision"); ax[0].legend(fontsize=8)
    save_fig("pd_roc_pr.png")

    # ------------------------------------------------------------ 5. EAD / LGD
    ead_res = run_ead_experiments(df, split, log)
    lgd_res = run_lgd_experiments(df, split, log)
    for tag, r in (("ead", ead_res), ("lgd", lgd_res)):
        r["val"].to_csv(REPORTS / f"{tag}_model_comparison_validation.csv", index=False)
        r["test"].to_csv(REPORTS / f"{tag}_model_comparison_test.csv", index=False)
        r["tuning"].to_csv(REPORTS / f"{tag}_tuning.csv", index=False)
    joblib.dump({"name": ead_res["chosen"], "model": ead_res["fitted"][ead_res["chosen"]], "clip": EAD_CLIP},
                MODELS / "ead_model.joblib", compress=3)
    joblib.dump({"name": lgd_res["chosen"], "model": lgd_res["fitted"][lgd_res["chosen"]], "clip": (0.0, 1.0)},
                MODELS / "lgd_model.joblib", compress=3)
    log(f"EAD chosen {ead_res['chosen']} | LGD chosen {lgd_res['chosen']}")

    # ------------------------------------------------------------ 6. segmentation + factor analysis
    seg = run_segmentation(df, log=log)
    joblib.dump(seg["artifact"], MODELS / "segmentation.joblib", compress=3)
    seg["selection"].to_csv(REPORTS / "segmentation_k_selection.csv", index=False)
    seg["profile"].to_csv(REPORTS / "segmentation_profile.csv")
    seg["pca"].to_csv(REPORTS / "segmentation_pca_sample.csv", index=False)
    df["segment"] = seg["segment"]
    fa = run_factor_analysis(df, log)
    fa_json = {k: v for k, v in fa.items() if k != "loadings"}
    if "loadings" in fa:
        fa["loadings"].round(3).to_csv(REPORTS / "factor_loadings.csv")

    # ------------------------------------------------------------ 7. explainability
    prep, clf = lr["model"].named_steps["prep"], lr["model"].named_steps["clf"]
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    coefs = pd.DataFrame({"feature": names, "coef_per_1sd_or_dummy": clf.coef_[0]})
    coefs["odds_ratio"] = np.exp(coefs["coef_per_1sd_or_dummy"])
    coefs["direction"] = np.where(coefs["coef_per_1sd_or_dummy"] > 0, "raises default risk", "lowers default risk")
    coefs.sort_values("coef_per_1sd_or_dummy", ascending=False).to_csv(REPORTS / "pd_logreg_coefficients.csv", index=False)
    rs = np.random.RandomState(SEED)
    tix = rs.choice(np.where(te)[0], 8000, replace=False)
    pi = permutation_importance(fit["model"], df.iloc[tix][FEATURES], df.iloc[tix]["pd_target"],
                                scoring="roc_auc", n_repeats=3, random_state=SEED, n_jobs=1)
    pimp = (pd.DataFrame({"feature": FEATURES, "importance": pi.importances_mean, "std": pi.importances_std})
            .sort_values("importance", ascending=False))
    pimp.to_csv(REPORTS / "pd_permutation_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    pimp.head(15).iloc[::-1].plot.barh(x="feature", y="importance", ax=ax, color="#7b4a24", legend=False)
    ax.set_title(f"Permutation importance - {chosen} (drop in test ROC-AUC)")
    save_fig("pd_permutation_importance.png")

    # ------------------------------------------------------------ 8. portfolio scoring
    raw_pd = fit["model"].predict_proba(df[FEATURES])[:, 1]
    df["pd"] = fit["calibrator"].predict(raw_pd)
    df["risk_band"] = assign_band(df["pd"], bands)
    ead_art, lgd_art = joblib.load(MODELS / "ead_model.joblib"), joblib.load(MODELS / "lgd_model.joblib")
    df["ead_proxy"] = np.clip(ead_art["model"].predict(df[EAD_FEATURES]), *EAD_CLIP)
    df["lgd_proxy"] = np.clip(lgd_art["model"].predict(df[FEATURES]), 0, 1)
    df["expected_loss"] = df["pd"] * df["ead_proxy"] * df["lgd_proxy"]
    cols = ["id", "split", "issue_year", "grade", "purpose", "income_band", "home_ownership", "pd_target",
            "pd", "risk_band", "ead_proxy", "lgd_proxy", "expected_loss", "segment", "loan_amnt",
            "annual_inc", "dti", "int_rate", "revol_util"]
    df[cols].round(5).to_csv(DATA / "portfolio_scored.csv.gz", index=False, compression="gzip")

    # ------------------------------------------------------------ 9. metadata for dashboard
    ref = {c: float(df[c].median()) for c in ["loan_amnt", "int_rate", "annual_inc", "dti", "delinq_2yrs",
           "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
           "credit_history_yrs", "collections_12_mths_ex_med", "acc_now_delinq"]}
    meta = {
        "pd_chosen": chosen, "pd_reason": pd_res["reason"], "ead_chosen": ead_res["chosen"],
        "ead_reason": ead_res["reason"], "lgd_chosen": lgd_res["chosen"], "lgd_reason": lgd_res["reason"],
        "ead_chosen_val_r2": ead_res["chosen_val_r2"], "lgd_chosen_val_r2": lgd_res["chosen_val_r2"],
        "pd_threshold_f1": fit["threshold"], "bands": bands, "k_segments": seg["artifact"]["k"],
        "segment_descriptions": {str(k): v for k, v in seg["artifact"]["description"].items()},
        "pca_variance": seg["pca_var"], "factor_analysis": fa_json, "quality": quality,
        "reference_inputs": ref,
        "categories": {c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in CATEGORICAL_FEATURES},
        "leakage_excluded": POST_ORIGINATION_EXCLUDED,
        "split_sizes": pd.Series(split).value_counts().to_dict(),
    }
    (REPORTS / "pipeline_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    log("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    main(ap.parse_args().raw)
