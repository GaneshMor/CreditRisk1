"""Scoring engine used by the dashboard (importable and testable without Streamlit)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.pd_model import assign_band
from src.preprocessing import EAD_FEATURES, FEATURES, add_engineered_features
from src.segmentation import assign_segment

ROOT = Path(__file__).resolve().parents[1]

CAT_REFERENCE = {"home_ownership": "MORTGAGE", "verification_status": "Verified",
                 "purpose": "debt_consolidation", "initial_list_status": "f"}
OTHER_REFERENCE = {"term_months": 36, "sub_grade": "C1", "emp_length_yrs": 6.0,
                   "mths_since_last_delinq": None, "mths_since_last_record": None}

LABELS = {
    "loan_amnt": "Loan amount", "term_months": "Loan term", "int_rate": "Interest rate",
    "sub_grade": "Lender sub-grade", "annual_inc": "Annual income", "dti": "Debt-to-income (DTI)",
    "emp_length_yrs": "Employment length", "delinq_2yrs": "Delinquencies (2y)",
    "inq_last_6mths": "Credit inquiries (6m)", "open_acc": "Open accounts", "pub_rec": "Public records",
    "revol_bal": "Revolving balance", "revol_util": "Revolving utilisation", "total_acc": "Total accounts",
    "credit_history_yrs": "Credit history length", "mths_since_last_delinq": "Months since last delinquency",
    "mths_since_last_record": "Months since last public record",
    "collections_12_mths_ex_med": "Collections (12m)", "acc_now_delinq": "Accounts now delinquent",
    "home_ownership": "Home ownership", "verification_status": "Income verification",
    "purpose": "Loan purpose", "initial_list_status": "Initial list status",
}

ACTIONS = {
    "Low": "Standard credit review; routine monitoring.",
    "Medium": "Enhanced credit review; consider tighter covenants, lower exposure, or additional verification.",
    "High": "Senior credit-committee review and closer monitoring; consider collateral, guarantor or reduced limit.",
}


def installment(loan, rate_pct, term_m):
    r = rate_pct / 1200.0
    return loan / term_m if r == 0 else loan * r / (1 - (1 + r) ** -term_m)


class Engine:
    def __init__(self, root=ROOT):
        m = Path(root) / "models"
        self.pd = joblib.load(m / "pd_model.joblib")
        self.ead = joblib.load(m / "ead_model.joblib")
        self.lgd = joblib.load(m / "lgd_model.joblib")
        self.seg = joblib.load(m / "segmentation.joblib")
        self.meta = json.loads((Path(root) / "reports" / "pipeline_metadata.json").read_text())
        self.reference = {**self.meta["reference_inputs"], **CAT_REFERENCE, **OTHER_REFERENCE}

    # ------------------------------------------------------------------ frames
    def frame(self, inputs: dict) -> pd.DataFrame:
        d = dict(inputs)
        g = d["sub_grade"]
        d["sub_grade_num"] = "ABCDEFG".index(g[0]) * 5 + int(g[1])
        d["installment"] = installment(d["loan_amnt"], d["int_rate"], d["term_months"])
        for k in ("mths_since_last_delinq", "mths_since_last_record"):
            d[k] = np.nan if d.get(k) is None else float(d[k])
        return add_engineered_features(pd.DataFrame([d]))

    # ----------------------------------------------------------------- scoring
    def _pd(self, X):
        raw = self.pd["model"].predict_proba(X[FEATURES])[:, 1]
        return self.pd["calibrator"].predict(raw)

    def score(self, inputs: dict, exposure_basis: str = "model") -> dict:
        X = self.frame(inputs)
        pd_ = float(self._pd(X)[0])
        ead_model = float(np.clip(self.ead["model"].predict(X[EAD_FEATURES])[0], *self.ead["clip"]))
        lgd = float(np.clip(self.lgd["model"].predict(X[FEATURES])[0], 0, 1))
        ead = float(inputs["loan_amnt"]) if exposure_basis == "requested" else ead_model
        band = str(assign_band([pd_], self.pd["bands"])[0])
        seg = int(assign_segment(self.seg, X)[0])
        return {"pd": pd_, "band": band, "ead_model": ead_model, "ead_used": ead, "lgd": lgd,
                "expected_loss": pd_ * ead * lgd, "segment": seg,
                "segment_desc": self.meta["segment_descriptions"].get(str(seg), ""),
                "action": ACTIONS[band]}

    # -------------------------------------------------------------- explanation
    def explain(self, inputs: dict, top: int = 5):
        """Local, model-agnostic sensitivity analysis (NOT SHAP): for each input, replace it with a
        reference-borrower value and record how the model's PD changes. Positive = raises risk."""
        base = float(self._pd(self.frame(inputs))[0])
        rows = []
        for k, ref in self.reference.items():
            if k not in inputs:
                continue
            alt = dict(inputs)
            alt[k] = ref
            delta = base - float(self._pd(self.frame(alt))[0])
            rows.append({"driver": LABELS.get(k, k), "value": inputs[k], "reference": ref, "delta_pd": delta})
        out = pd.DataFrame(rows).sort_values("delta_pd", ascending=False)
        up = out[out.delta_pd > 0.0005].head(top)
        down = out[out.delta_pd < -0.0005].sort_values("delta_pd").head(top)
        return up, down, out
