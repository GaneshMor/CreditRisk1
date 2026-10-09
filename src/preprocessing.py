"""Data preparation for the credit-risk project.

Builds the modelling dataset from the raw LendingClub 2007-2014 file and
defines which columns are origination (usable as predictors) versus
post-origination (leakage - used ONLY to construct targets).
"""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- columns
RAW_COLS = [
    "id", "loan_amnt", "funded_amnt", "term", "int_rate", "installment", "grade",
    "sub_grade", "emp_length", "home_ownership", "annual_inc", "verification_status",
    "issue_d", "loan_status", "purpose", "addr_state", "dti", "delinq_2yrs",
    "earliest_cr_line", "inq_last_6mths", "mths_since_last_delinq",
    "mths_since_last_record", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "initial_list_status", "collections_12_mths_ex_med", "acc_now_delinq",
    # post-origination: used ONLY to build the EAD-at-default / LGD targets
    "total_rec_prncp", "recoveries", "collection_recovery_fee",
]

# Resolved outcomes only. Current / In Grace / Late(16-30) are right-censored.
DEFAULT_STATUS = ["Charged Off", "Default", "Late (31-120 days)",
                  "Does not meet the credit policy. Status:Charged Off"]
GOOD_STATUS = ["Fully Paid", "Does not meet the credit policy. Status:Fully Paid"]

# ------------------------------------------------- leakage classification
POST_ORIGINATION_EXCLUDED = [
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp",
    "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d",
    "loan_status", "pymnt_plan",
]

NUMERIC_FEATURES = [
    "loan_amnt", "term_months", "int_rate", "sub_grade_num", "annual_inc", "dti",
    "emp_length_yrs", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "credit_history_yrs",
    "mths_since_last_delinq", "mths_since_last_record", "had_delinq", "had_pub_record",
    "collections_12_mths_ex_med", "acc_now_delinq", "payment_to_income", "loan_to_income",
]
CATEGORICAL_FEATURES = ["home_ownership", "verification_status", "purpose", "initial_list_status"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# EAD model must not see anything that is a deterministic function of the
# funded amount (otherwise the "model" just reads the answer off).
EAD_EXCLUDE = ["loan_amnt", "loan_to_income", "payment_to_income"]
EAD_FEATURES = [f for f in FEATURES if f not in EAD_EXCLUDE]

# Segmentation variables (credit-profile only; no model outputs, no target)
SEGMENT_FEATURES = ["annual_inc", "loan_amnt", "dti", "int_rate", "revol_util",
                    "delinq_2yrs", "inq_last_6mths", "credit_history_yrs",
                    "emp_length_yrs", "open_acc"]

INCOME_BANDS = [0, 30_000, 50_000, 75_000, 100_000, 150_000, np.inf]
INCOME_LABELS = ["<30k", "30-50k", "50-75k", "75-100k", "100-150k", "150k+"]


def _emp_len(s):
    s = s.astype(str).str.strip()
    out = s.str.extract(r"(\d+)")[0].astype(float)
    out[s.str.startswith("<")] = 0.0
    out[s.str.lower().isin(["nan", "none", ""])] = np.nan
    return out


def add_engineered_features(d: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for both the training frame and a single dashboard
    input row. Derives term_months, sub_grade_num, emp_length_yrs, the
    had_* flags and the two affordability ratios when they are missing."""
    d = d.copy()
    if "term_months" not in d:
        d["term_months"] = d["term"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    if "sub_grade_num" not in d:
        g = d["sub_grade"].astype(str)
        d["sub_grade_num"] = (g.str[0].map({c: i for i, c in enumerate("ABCDEFG")}) * 5
                              + g.str[1].astype(float))
    if "emp_length_yrs" not in d:
        d["emp_length_yrs"] = _emp_len(d["emp_length"])
    d["had_delinq"] = d["mths_since_last_delinq"].notna().astype(int)
    d["had_pub_record"] = d["mths_since_last_record"].notna().astype(int)
    d["payment_to_income"] = (d["installment"] * 12) / d["annual_inc"].clip(lower=1)
    d["loan_to_income"] = d["loan_amnt"] / d["annual_inc"].clip(lower=1)
    return d


def build_modeling_dataset(raw_csv: str, report: dict | None = None) -> pd.DataFrame:
    df = pd.read_csv(raw_csv, usecols=RAW_COLS, low_memory=False)
    if report is not None:
        report["raw_rows"] = int(len(df))
        report["raw_cols_used"] = len(RAW_COLS)
        report["status_counts"] = df["loan_status"].value_counts().to_dict()
        report["duplicate_ids"] = int(df["id"].duplicated().sum())

    df = df.drop_duplicates("id")
    df = df[df["loan_status"].isin(DEFAULT_STATUS + GOOD_STATUS)].copy()
    df["pd_target"] = df["loan_status"].isin(DEFAULT_STATUS).astype(int)

    df["issue_date"] = pd.to_datetime(df["issue_d"], format="%b-%y", errors="coerce")
    df["issue_year"] = df["issue_date"].dt.year
    ecl = pd.to_datetime(df["earliest_cr_line"], format="%b-%y", errors="coerce")
    # two-digit-year parsing can push 1960s dates into the 2060s
    ecl = ecl.where(ecl <= df["issue_date"], ecl - pd.DateOffset(years=100))
    df["credit_history_yrs"] = ((df["issue_date"] - ecl).dt.days / 365.25).clip(lower=0)

    df = add_engineered_features(df)

    # ---- targets (built from post-origination columns; never used as features)
    df["ead_origination_proxy"] = df["funded_amnt"]
    outstanding = (df["funded_amnt"] - df["total_rec_prncp"]).clip(lower=0)
    df["ead_at_default_proxy"] = np.where(df["pd_target"] == 1, outstanding, np.nan)
    net_rec = (df["recoveries"] - df["collection_recovery_fee"]).clip(lower=0)
    lgd = 1 - net_rec / outstanding.replace(0, np.nan)
    df["lgd_realized_proxy"] = np.where(df["pd_target"] == 1, lgd.clip(0, 1), np.nan)

    df["income_band"] = pd.cut(df["annual_inc"], INCOME_BANDS, labels=INCOME_LABELS, right=False)
    keep = (["id", "issue_year", "grade", "sub_grade", "addr_state", "loan_status", "pd_target",
             "ead_origination_proxy", "ead_at_default_proxy", "lgd_realized_proxy", "income_band"]
            + FEATURES)
    return df[keep].reset_index(drop=True)
