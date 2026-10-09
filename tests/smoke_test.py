"""Sanity checks on the trained engine (no Streamlit needed):  python tests/smoke_test.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.scoring import Engine

eng = Engine()
base = dict(loan_amnt=15000, term_months=36, int_rate=13.5, sub_grade="C3", annual_inc=65000, dti=16.0,
            emp_length_yrs=6.0, delinq_2yrs=0, inq_last_6mths=1, open_acc=10, pub_rec=0, revol_bal=11000,
            revol_util=55.0, total_acc=23, credit_history_yrs=14.0, mths_since_last_delinq=None,
            mths_since_last_record=None, collections_12_mths_ex_med=0, acc_now_delinq=0,
            home_ownership="MORTGAGE", verification_status="Verified", purpose="debt_consolidation",
            initial_list_status="f")
good = dict(base, int_rate=7.5, sub_grade="A3", dti=6.0, revol_util=15.0, annual_inc=140000, loan_amnt=8000)
bad = dict(base, int_rate=24.0, sub_grade="F2", dti=34.0, revol_util=92.0, annual_inc=28000, term_months=60,
           delinq_2yrs=3, inq_last_6mths=5, mths_since_last_delinq=6, home_ownership="RENT", purpose="small_business")
r = {n: eng.score(x) for n, x in (("base", base), ("good", good), ("bad", bad))}
for n, v in r.items():
    print(n, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items() if k != "action"})
assert r["good"]["pd"] < r["base"]["pd"] < r["bad"]["pd"], "PD should rise with weaker credit profile"
for v in r.values():
    assert 0 <= v["pd"] <= 1 and 0 <= v["lgd"] <= 1 and v["ead_model"] > 0
    assert abs(v["expected_loss"] - v["pd"] * v["ead_used"] * v["lgd"]) < 1e-6
    assert v["band"] in ("Low", "Medium", "High") and v["segment"] >= 1
up, down, _ = eng.explain(bad)
print("risk-increasing drivers for 'bad':", up.driver.tolist())
assert len(up) > 0
print("OK")
