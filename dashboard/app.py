"""Credit Risk Decision-Support Dashboard (Streamlit).

Run from the project root:   streamlit run dashboard/app.py
All figures come from the trained artifacts in models/ and reports/ - nothing is hard-coded.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scoring import Engine  # noqa: E402

# ------------------------------------------------------------------ theme
BROWN, COPPER, TAN, CREAM, INK = "#4a2c17", "#a8692c", "#d9b98a", "#f6f1ea", "#2b2118"
BAND_COL = {"Low": "#6f8f63", "Medium": "#d1a24a", "High": "#a63d2f"}
SEG_SEQ = ["#4a2c17", "#a8692c", "#d9b98a", "#8c7b6b", "#6f8f63", "#a63d2f"]
DEFAULT_USD_INR = 88.0   # ASSUMPTION - editable in the sidebar; dataset values are USD

st.set_page_config(page_title="Credit Risk Decision Support", page_icon="🏦", layout="wide")
st.markdown(f"""
<style>
.stApp {{ background:{CREAM}; }}
.block-container {{ padding-top:1rem; max-width:1500px; }}
/* readable text regardless of the viewer's dark/light system theme */
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"], section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{ color:{INK} !important; }}
.block-container [data-testid="stMarkdownContainer"] > p, .block-container [data-testid="stMarkdownContainer"] li,
.block-container [data-testid="stCaptionContainer"], .block-container label, .block-container [data-testid="stWidgetLabel"] p {{ color:{INK} !important; }}
.stApp [data-testid="stHeader"] {{ background:{CREAM}; }}
.stApp [data-testid="stHeader"] * {{ color:{INK} !important; }}
.hdr {{ background:linear-gradient(90deg,{BROWN},#7b4a24); color:#fff; padding:14px 22px; border-radius:10px;
        margin-bottom:10px; border-bottom:3px solid {TAN}; }}
.hdr h1 {{ color:#fff; font-size:1.55rem; margin:0; font-weight:700; letter-spacing:.3px; }}
.hdr p {{ color:{TAN}; margin:2px 0 0 0; font-size:.85rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap:6px; background:{BROWN}; padding:6px 10px; border-radius:8px; }}
.stTabs [data-baseweb="tab"] {{ color:#f0e2cf; background:#6b4423; border-radius:4px; padding:6px 16px; height:auto; }}
.stTabs [aria-selected="true"] {{ background:{TAN} !important; color:{BROWN} !important; font-weight:700; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display:none; }}
section[data-testid="stSidebar"] {{ background:#efe6d8; border-right:2px solid {TAN}; }}
.kpi {{ background:#fff; border:1px solid #e3d3bb; border-left:6px solid {COPPER}; border-radius:8px;
        padding:10px 14px; box-shadow:0 1px 2px rgba(74,44,23,.08); min-height:88px; }}
.kpi .l {{ color:#7a6a58; font-size:.72rem; text-transform:uppercase; letter-spacing:.6px; }}
.kpi .v {{ color:{BROWN}; font-size:1.55rem; font-weight:700; line-height:1.25; }}
.kpi .s {{ color:#8a7b6a; font-size:.72rem; }}
.sec {{ color:{BROWN}; font-weight:700; border-bottom:2px solid {TAN}; padding-bottom:3px; margin:14px 0 8px 0; }}
.note {{ background:#fff8ec; border:1px solid {TAN}; border-radius:6px; padding:8px 12px; font-size:.82rem; color:{INK}; }}
.pill {{ display:inline-block; padding:3px 12px; border-radius:14px; color:#fff; font-weight:700; }}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def kpi(label, value, sub=""):
    return f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}&nbsp;</div></div>'


def row_kpis(items):
    cols = st.columns(len(items))
    for c, (l, v, s) in zip(cols, items):
        c.markdown(kpi(l, v, s), unsafe_allow_html=True)


def sec(t):
    st.markdown(f'<div class="sec">{t}</div>', unsafe_allow_html=True)


def indian_group(n: int) -> str:
    s = str(abs(int(n)))
    if len(s) > 3:
        s = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", s[:-3]) + "," + s[-3:]
    return ("-" if n < 0 else "") + s


def money(usd, compact=False):
    """Format a USD amount in the chosen display currency."""
    cur, fx = st.session_state["cur"], st.session_state["fx"]
    if cur == "USD":
        v = usd
        if compact and abs(v) >= 1e6:
            return f"${v/1e6:,.2f}M"
        return f"${v:,.0f}"
    v = usd * fx
    if compact and abs(v) >= 1e7:
        return f"₹{v/1e7:,.2f} Cr"
    if compact and abs(v) >= 1e5:
        return f"₹{v/1e5:,.2f} L"
    return f"₹{indian_group(round(v))}"


def style(fig, h=330, title=None):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=10, r=10, t=40 if title else 15, b=10),
                      title=dict(text=title, font=dict(size=14, color=BROWN)) if title else None,
                      paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font=dict(color=INK),
                      colorway=[BROWN, COPPER, TAN, "#8c7b6b"])
    return fig


def show(fig):
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------ cached loaders
@st.cache_resource
def get_engine():
    return Engine(ROOT)


@st.cache_data
def load_portfolio():
    return pd.read_csv(ROOT / "data" / "portfolio_scored.csv.gz")


@st.cache_data
def load_report(name):
    return pd.read_csv(ROOT / "reports" / name)


needed = ROOT / "models" / "pd_model.joblib"
if not needed.exists():
    st.error("Trained models not found. Run `python -m src.run_pipeline --raw <loan_data_2007_2014.csv>` first.")
    st.stop()

eng, meta, port = get_engine(), get_engine().meta, load_portfolio()

# ------------------------------------------------------------------ sidebar (slicer panel)
with st.sidebar:
    st.markdown(f"<h3 style='color:{BROWN};margin-top:0'>Filters</h3>", unsafe_allow_html=True)
    cur = st.radio("Display currency", ["INR (₹)", "USD ($)"], horizontal=True)
    st.session_state["cur"] = "INR" if cur.startswith("INR") else "USD"
    st.session_state["fx"] = st.number_input("USD → INR rate (assumption, editable)", 50.0, 150.0, DEFAULT_USD_INR, 0.5,
                                             disabled=st.session_state["cur"] == "USD")
    st.caption("Source data is in USD; INR is a straight conversion at the rate above.")
    sample = st.radio("Portfolio sample", ["Test set (out-of-sample)", "All resolved loans"])
    years = st.multiselect("Issue year", sorted(port.issue_year.unique()), default=sorted(port.issue_year.unique()))
    grades = st.multiselect("Grade", sorted(port.grade.unique()), default=sorted(port.grade.unique()))
    purposes = st.multiselect("Loan purpose", sorted(port.purpose.unique()), default=sorted(port.purpose.unique()))
    bands_f = st.multiselect("Risk band", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
    st.caption("Filters apply to Overview, Segmentation and Portfolio Analytics.")

view = port
if sample.startswith("Test"):
    view = view[view.split == "test"]
view = view[view.issue_year.isin(years) & view.grade.isin(grades) & view.purpose.isin(purposes)
            & view.risk_band.isin(bands_f)]

st.markdown(f"""<div class="hdr"><h1>🏦 Credit Risk Decision-Support System</h1>
<p>PD · EAD proxy · LGD proxy · Expected Loss · Customer Segmentation &nbsp;|&nbsp; LendingClub 2007-2014 &nbsp;|&nbsp;
Academic / portfolio model - decision support only, not an automated lending decision</p></div>""", unsafe_allow_html=True)

tabs = st.tabs(["Executive Overview", "Individual Assessment", "Model Performance", "Risk Segmentation",
                "Portfolio Analytics", "Explainability", "Notes & Limitations"])

# ================================================================== 1. OVERVIEW
with tabs[0]:
    if view.empty:
        st.warning("No loans match the current filters.")
    else:
        row_kpis([
            ("Loans analysed", f"{len(view):,}", sample),
            ("Observed default rate", f"{view.pd_target.mean():.1%}", "resolved loans only"),
            ("Average PD (model)", f"{view.pd.mean():.1%}", "calibrated probability"),
            ("Est. exposure (EAD proxy)", money(view.ead_proxy.sum(), True), "sum of proxy exposure"),
            ("Expected loss", money(view.expected_loss.sum(), True), f"{view.expected_loss.sum()/view.ead_proxy.sum():.1%} of exposure"),
            ("High-risk share", f"{(view.risk_band == 'High').mean():.1%}", "PD in top band"),
        ])
        c1, c2, c3 = st.columns([1, 1.2, 1.2])
        with c1:
            d = view.pd_target.mean()
            f = go.Figure(go.Pie(values=[d, 1 - d], labels=["Defaulted", "Repaid"], hole=.68,
                                 marker=dict(colors=[BROWN, TAN]), textinfo="none", sort=False))
            f.add_annotation(text=f"<b>{d:.1%}</b>", showarrow=False, font=dict(size=26, color=BROWN))
            show(style(f, 300, "Observed default rate"))
        with c2:
            g = view.groupby("risk_band").size().reindex(["Low", "Medium", "High"]).reset_index(name="loans")
            f = px.bar(g, x="risk_band", y="loans", color="risk_band", color_discrete_map=BAND_COL, text_auto=",")
            f.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Loans")
            show(style(f, 300, "Risk-band distribution"))
        with c3:
            g = view.groupby("segment").size().reset_index(name="loans")
            g["segment"] = "Segment " + g.segment.astype(str)
            f = px.bar(g, x="segment", y="loans", text_auto=",", color_discrete_sequence=[COPPER])
            f.update_layout(xaxis_title=None, yaxis_title="Loans")
            show(style(f, 300, "Customer-segment distribution"))
        c1, c2 = st.columns(2)
        with c1:
            g = view.groupby("risk_band").agg(el=("expected_loss", "sum")).reindex(["Low", "Medium", "High"]).reset_index()
            g["el_disp"] = g.el * (st.session_state["fx"] if st.session_state["cur"] == "INR" else 1)
            f = px.bar(g, x="risk_band", y="el_disp", color="risk_band", color_discrete_map=BAND_COL)
            f.update_layout(showlegend=False, xaxis_title=None, yaxis_title=f"Expected loss ({st.session_state['cur']})")
            show(style(f, 300, "Expected loss by risk band"))
        with c2:
            g = view.groupby("grade").agg(rate=("pd_target", "mean"), pdm=("pd", "mean")).reset_index()
            f = go.Figure()
            f.add_bar(x=g.grade, y=g.rate, name="Observed default rate", marker_color=BROWN)
            f.add_scatter(x=g.grade, y=g.pdm, name="Average model PD", mode="lines+markers", line=dict(color=COPPER))
            f.update_layout(yaxis_tickformat=".0%", legend=dict(orientation="h", y=-0.2))
            show(style(f, 300, "Observed default rate vs model PD, by grade"))
        st.markdown('<div class="note">Exposure and loss figures use the <b>EAD proxy</b> (funded-amount based) and the '
                    '<b>realised LGD proxy</b>; they are not regulatory EAD / Basel LGD. The sample contains only resolved loans, '
                    'so the default rate is not representative of a live book.</div>', unsafe_allow_html=True)

# ================================================================== 2. INDIVIDUAL
with tabs[1]:
    sec("Borrower inputs")
    cs, fx = st.session_state["cur"], st.session_state["fx"]
    mult = fx if cs == "INR" else 1.0
    sym = "₹" if cs == "INR" else "$"
    cat = meta["categories"]
    a, b, c, d = st.columns(4)
    loan_in = a.number_input(f"Loan amount ({sym})", 500.0 * mult, 35000.0 * mult, round(15000 * mult, -3), 1000.0 * mult, key=f"loan{cs}")
    inc_in = a.number_input(f"Annual income ({sym})", 2000.0 * mult, 2e6 * mult, round(65000 * mult, -3), 5000.0 * mult, key=f"inc{cs}")
    term = a.selectbox("Term (months)", [36, 60])
    rate = a.number_input("Interest rate (%)", 5.0, 27.0, 13.5, 0.25)
    sub_grade = a.selectbox("Lender sub-grade", [f"{g}{i}" for g in "ABCDEFG" for i in range(1, 6)], index=12)
    dti = b.number_input("Debt-to-income (DTI, %)", 0.0, 40.0, 16.0, 0.5)
    emp = b.slider("Employment length (years)", 0, 10, 6)
    hist = b.number_input("Credit history (years)", 0.5, 60.0, 14.0, 0.5)
    ru = b.number_input("Revolving utilisation (%)", 0.0, 150.0, 55.0, 1.0)
    rb_in = b.number_input(f"Revolving balance ({sym})", 0.0, 500000.0 * mult, round(11000 * mult, -2), 500.0 * mult, key=f"rb{cs}")
    home = c.selectbox("Home ownership", cat["home_ownership"], index=cat["home_ownership"].index("MORTGAGE"))
    ver = c.selectbox("Income verification", cat["verification_status"])
    purp = c.selectbox("Loan purpose", cat["purpose"], index=cat["purpose"].index("debt_consolidation"))
    ils = c.selectbox("Initial list status", cat["initial_list_status"])
    tot = c.number_input("Total accounts", 1, 150, 23)
    opn = c.number_input("Open accounts", 0, 76, 10)
    delq = d.number_input("Delinquencies (last 2y)", 0, 30, 0)
    inq = d.number_input("Credit inquiries (last 6m)", 0, 33, 1)
    pub = d.number_input("Public records", 0, 11, 0)
    coll = d.number_input("Collections (12m, ex-medical)", 0, 6, 0)
    nowd = d.number_input("Accounts now delinquent", 0, 5, 0)
    no_dq = d.checkbox("No delinquency on record", True)
    ms_dq = None if no_dq else d.number_input("Months since last delinquency", 0, 152, 24)
    no_pr = d.checkbox("No public record on file", True)
    ms_pr = None if no_pr else d.number_input("Months since last public record", 0, 129, 60)
    basis = st.radio("Exposure basis for expected loss",
                     ["Model-estimated EAD proxy", "Requested loan amount"], horizontal=True)

    inputs = dict(loan_amnt=loan_in / mult, term_months=term, int_rate=rate, sub_grade=sub_grade,
                  annual_inc=inc_in / mult, dti=dti, emp_length_yrs=float(emp), delinq_2yrs=delq,
                  inq_last_6mths=inq, open_acc=opn, pub_rec=pub, revol_bal=rb_in / mult, revol_util=ru,
                  total_acc=tot, credit_history_yrs=hist, mths_since_last_delinq=ms_dq,
                  mths_since_last_record=ms_pr, collections_12_mths_ex_med=coll, acc_now_delinq=nowd,
                  home_ownership=home, verification_status=ver, purpose=purp, initial_list_status=ils)
    st.session_state["last_inputs"] = inputs
    res = eng.score(inputs, "requested" if basis.startswith("Requested") else "model")

    sec("Assessment")
    pill = f'<span class="pill" style="background:{BAND_COL[res["band"]]}">{res["band"].upper()} RISK</span>'
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.markdown(kpi("Probability of default", f"{res['pd']:.1%}", "calibrated model PD"), unsafe_allow_html=True)
    k2.markdown(kpi("Risk category", pill, f"cut-offs {meta['bands']['low_max']:.1%} / {meta['bands']['high_min']:.1%}"), unsafe_allow_html=True)
    k3.markdown(kpi("Est. exposure / EAD proxy", money(res["ead_used"]), "model estimate " + money(res["ead_model"]) if basis.startswith("Requested") else "typical funded exposure for profile"), unsafe_allow_html=True)
    k4.markdown(kpi("Est. LGD (proxy)", f"{res['lgd']:.0%}", "realised-loss proxy, given default"), unsafe_allow_html=True)
    k5.markdown(kpi("Expected loss", money(res["expected_loss"]), "PD × EAD × LGD"), unsafe_allow_html=True)
    k6.markdown(kpi("Customer segment", f"Segment {res['segment']}", res["segment_desc"][:42]), unsafe_allow_html=True)

    g1, g2 = st.columns([1, 1.4])
    with g1:
        lo, hi = meta["bands"]["low_max"], meta["bands"]["high_min"]
        top = max(0.6, hi * 2)
        f = go.Figure(go.Indicator(mode="gauge+number", value=res["pd"] * 100, number=dict(suffix="%", valueformat=".1f"),
                                   gauge=dict(axis=dict(range=[0, top * 100]), bar=dict(color=BROWN),
                                              steps=[dict(range=[0, lo * 100], color="#cfe0c8"),
                                                     dict(range=[lo * 100, hi * 100], color="#f3deb0"),
                                                     dict(range=[hi * 100, top * 100], color="#e6b5ad")])))
        show(style(f, 280, "PD against risk-band cut-offs"))
    with g2:
        up, down, _ = eng.explain(inputs)
        sec("Key risk drivers (this borrower)")
        cu, cd = st.columns(2)
        cu.markdown("**Factors increasing risk**")
        for _, r in up.iterrows():
            cu.markdown(f"• {r.driver} <span style='color:#a63d2f'>(+{r.delta_pd*100:.1f} pp)</span>", unsafe_allow_html=True)
        if up.empty:
            cu.caption("None material")
        cd.markdown("**Factors reducing risk**")
        for _, r in down.iterrows():
            cd.markdown(f"• {r.driver} <span style='color:#6f8f63'>({r.delta_pd*100:.1f} pp)</span>", unsafe_allow_html=True)
        if down.empty:
            cd.caption("None material")
        st.caption("Local sensitivity vs. a reference borrower (each input swapped to the portfolio-typical value); "
                   "percentage-point change in this model's PD. Not SHAP.")
    st.markdown(f'<div class="note"><b>Suggested action (decision support):</b> {res["action"]}<br>'
                'This output is a model estimate for review by a credit officer, not an approval or rejection decision.</div>',
                unsafe_allow_html=True)

# ================================================================== 3. MODEL PERFORMANCE
with tabs[2]:
    def mark(tbl, chosen):
        t = tbl.copy()
        t.insert(0, "Selected", np.where(t.Model == chosen, "✔", ""))
        return t

    sec(f"PD models - held-out test set (selected: {meta['pd_chosen']})")
    t = load_report("pd_model_comparison_test.csv")
    cols = ["Model", "Threshold", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Gini", "KS", "Brier"]
    st.dataframe(mark(t[cols], meta["pd_chosen"]).style.format({c: "{:.3f}" for c in cols[1:]}), hide_index=True)
    st.markdown(f'<div class="note"><b>Why this model:</b> {meta["pd_reason"]}<br>Threshold = F1-optimal cut-off chosen on the '
                'validation set (probabilities are sigmoid (Platt) calibrated). Accuracy is shown but not used for selection - the '
                'portfolio is imbalanced.</div>', unsafe_allow_html=True)
    with st.expander("Same models at the default 0.5 cut-off (recall collapses on imbalanced data)"):
        t5 = load_report("pd_model_comparison_test_threshold_0.5.csv")
        st.dataframe(t5[cols].style.format({c: "{:.3f}" for c in cols[1:]}), hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        sel = t[t.Model == meta["pd_chosen"]].iloc[0]
        cm = np.array([[sel.TN, sel.FP], [sel.FN, sel.TP]])
        f = px.imshow(cm, text_auto=",", x=["Pred non-default", "Pred default"], y=["Actual non-default", "Actual default"],
                      color_continuous_scale=[[0, "#f6f1ea"], [1, BROWN]])
        f.update_coloraxes(showscale=False)
        show(style(f, 320, "Confusion matrix - selected PD model (test)"))
    with c2:
        img = ROOT / "reports" / "figures" / "pd_roc_pr.png"
        if img.exists():
            st.image(str(img))
    st.caption("False negatives (defaulters missed) are costlier than false positives for a lender, which is why recall, "
               "PR-AUC and KS are reviewed alongside accuracy.")
    sec("Risk-band calibration (validation / test check)")
    st.dataframe(load_report("pd_risk_band_validation.csv").style.format(
        {"mean_pd": "{:.1%}", "observed_default_rate": "{:.1%}"}), hide_index=True)

    sec(f"EAD proxy models - test set (selected: {meta['ead_chosen']})")
    e = load_report("ead_model_comparison_test.csv")
    st.dataframe(mark(e, meta["ead_chosen"]).style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "R2": "{:.3f}"}), hide_index=True)
    st.markdown(f'<div class="note">{meta["ead_reason"]} Metrics are in USD. Target = funded amount, a proxy; the loan-amount '
                'variable itself is excluded from the predictors.</div>', unsafe_allow_html=True)
    sec(f"LGD proxy models - test set (selected: {meta['lgd_chosen']})")
    lg = load_report("lgd_model_comparison_test.csv")
    st.dataframe(mark(lg, meta["lgd_chosen"]).style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}"}), hide_index=True)
    st.markdown(f'<div class="note">{meta["lgd_reason"]} Predictions are clipped to [0, 1]. '
                f'<b>Caution:</b> LGD is hard to predict from origination data (R² of the selected model on validation: '
                f'{meta["lgd_chosen_val_r2"]:.3f}); treat LGD as close to a portfolio average.</div>', unsafe_allow_html=True)
    with st.expander("Hyper-parameter tuning log"):
        for tg in ("pd", "ead", "lgd"):
            st.markdown(f"**{tg.upper()}**")
            st.dataframe(load_report(f"{tg}_tuning.csv"), hide_index=True)

# ================================================================== 4. SEGMENTATION
with tabs[3]:
    sel = load_report("segmentation_k_selection.csv")
    prof = load_report("segmentation_profile.csv")
    row_kpis([("Clusters (K)", str(meta["k_segments"]), "chosen by silhouette in 3-6"),
              ("Silhouette (chosen K)", f"{sel.loc[sel.K == meta['k_segments'], 'Silhouette'].iloc[0]:.3f}", "weak-moderate structure is typical"),
              ("PCA variance (2 PCs)", f"{sum(meta['pca_variance']):.0%}", "of standardised variables")])
    c1, c2 = st.columns(2)
    with c1:
        f = go.Figure(go.Scatter(x=sel.K, y=sel.Inertia, mode="lines+markers", line=dict(color=BROWN)))
        f.update_layout(xaxis_title="K", yaxis_title="Inertia (WCSS)")
        show(style(f, 290, "Elbow curve"))
    with c2:
        f = go.Figure(go.Scatter(x=sel.K, y=sel.Silhouette, mode="lines+markers", line=dict(color=COPPER)))
        f.update_layout(xaxis_title="K", yaxis_title="Silhouette")
        show(style(f, 290, "Silhouette score by K"))
    sec("Segment profiles (Segment 1 = lowest observed default rate; ranked after clustering)")
    pf = prof.rename(columns={"_c": "segment"}).copy()
    pf.columns = ["Segment", "Loans", "Default rate", "Avg income ($)", "Avg loan ($)", "Avg DTI", "Avg int. rate",
                  "Avg revol. util.", "Avg delinq. 2y", "Avg inquiries 6m", "Avg history (y)", "Avg emp. (y)", "Description"]
    st.dataframe(pf.style.format({"Loans": "{:,.0f}", "Default rate": "{:.1%}", "Avg income ($)": "{:,.0f}",
                                  "Avg loan ($)": "{:,.0f}", "Avg DTI": "{:.1f}", "Avg int. rate": "{:.1f}",
                                  "Avg revol. util.": "{:.1f}", "Avg delinq. 2y": "{:.2f}",
                                  "Avg inquiries 6m": "{:.2f}", "Avg history (y)": "{:.1f}", "Avg emp. (y)": "{:.1f}"}),
                 hide_index=True)
    c1, c2, c3 = st.columns(3)
    sg = view.groupby("segment").agg(n=("pd", "size"), pdm=("pd", "mean"), dr=("pd_target", "mean")).reset_index()
    sg["segment"] = "Segment " + sg.segment.astype(str)
    with c1:
        show(style(px.bar(sg, x="segment", y="n", text_auto=",", color_discrete_sequence=[COPPER]).update_layout(xaxis_title=None, yaxis_title="Loans"), 300, "Cluster sizes (filtered)"))
    with c2:
        f = go.Figure()
        f.add_bar(x=sg.segment, y=sg.pdm, name="Avg PD", marker_color=BROWN)
        f.add_bar(x=sg.segment, y=sg.dr, name="Observed default rate", marker_color=TAN)
        f.update_layout(barmode="group", yaxis_tickformat=".0%", legend=dict(orientation="h", y=-0.2))
        show(style(f, 300, "Average PD & default rate by segment"))
    with c3:
        pca = load_report("segmentation_pca_sample.csv")
        pca["segment"] = "Segment " + pca.segment.astype(str)
        f = px.scatter(pca, x="PC1", y="PC2", color="segment", opacity=.45, color_discrete_sequence=SEG_SEQ)
        f.update_traces(marker=dict(size=4))
        show(style(f, 300, "PCA projection (8,000-loan sample)"))
    st.caption("Segmentation is unsupervised: default outcomes were NOT used to form clusters, only to describe them afterwards.")

# ================================================================== 5. PORTFOLIO ANALYTICS
with tabs[4]:
    if view.empty:
        st.warning("No loans match the current filters.")
    else:
        fxm = st.session_state["fx"] if st.session_state["cur"] == "INR" else 1.0
        sym = "₹" if st.session_state["cur"] == "INR" else "$"
        c1, c2 = st.columns(2)
        with c1:
            show(style(px.histogram(view, x="pd", nbins=50, color_discrete_sequence=[BROWN]).update_layout(xaxis_tickformat=".0%", xaxis_title="Predicted PD", yaxis_title="Loans"), 290, "PD distribution"))
        with c2:
            v = view.assign(e=view.ead_proxy * fxm)
            show(style(px.histogram(v, x="e", nbins=50, color_discrete_sequence=[COPPER]).update_layout(xaxis_title=f"EAD proxy ({sym})", yaxis_title="Loans"), 290, "EAD-proxy distribution"))
        c1, c2 = st.columns(2)
        with c1:
            show(style(px.histogram(view, x="lgd_proxy", nbins=40, color_discrete_sequence=[TAN]).update_layout(xaxis_tickformat=".0%", xaxis_title="Predicted LGD proxy", yaxis_title="Loans"), 290, "LGD-proxy distribution"))
        with c2:
            v = view.assign(el=view.expected_loss * fxm)
            show(style(px.histogram(v, x="el", nbins=50, color_discrete_sequence=["#8c7b6b"]).update_layout(xaxis_title=f"Expected loss ({sym})", yaxis_title="Loans"), 290, "Expected-loss distribution"))
        c1, c2 = st.columns(2)
        with c1:
            g = view.groupby("segment").expected_loss.sum().reset_index()
            g["segment"] = "Segment " + g.segment.astype(str); g["el"] = g.expected_loss * fxm
            show(style(px.bar(g, x="segment", y="el", color_discrete_sequence=[BROWN]).update_layout(xaxis_title=None, yaxis_title=f"Expected loss ({sym})"), 290, "Expected loss by risk segment"))
        with c2:
            g = view.groupby("risk_band").ead_proxy.sum().reindex(["Low", "Medium", "High"]).reset_index()
            g["x"] = g.ead_proxy * fxm
            show(style(px.bar(g, x="risk_band", y="x", color="risk_band", color_discrete_map=BAND_COL).update_layout(showlegend=False, xaxis_title=None, yaxis_title=f"Exposure ({sym})"), 290, "Exposure by risk category"))
        c1, c2, c3 = st.columns(3)
        for col, key, ttl in ((c1, "grade", "Default rate by grade"), (c2, "purpose", "Default rate by loan purpose"),
                              (c3, "income_band", "Default rate by income band")):
            g = view.groupby(key).pd_target.agg(["mean", "size"]).reset_index()
            if key == "income_band":
                order = ["<30k", "30-50k", "50-75k", "75-100k", "100-150k", "150k+"]
                g[key] = pd.Categorical(g[key], order, ordered=True); g = g.sort_values(key)
            elif key == "purpose":
                g = g.sort_values("mean")
            with col:
                f = px.bar(g, x="mean", y=key, orientation="h", color_discrete_sequence=[COPPER]) if key == "purpose" \
                    else px.bar(g, x=key, y="mean", color_discrete_sequence=[COPPER])
                f.update_layout(xaxis_title=None, yaxis_title=None)
                f.update_xaxes(tickformat=".0%") if key == "purpose" else f.update_yaxes(tickformat=".0%")
                show(style(f, 320, ttl))
        st.caption("Income bands are in USD (annual income at origination).")

# ================================================================== 6. EXPLAINABILITY
with tabs[5]:
    c1, c2 = st.columns(2)
    with c1:
        pi = load_report("pd_permutation_importance.csv").head(15).iloc[::-1]
        f = px.bar(pi, x="importance", y="feature", orientation="h", color_discrete_sequence=[BROWN])
        f.update_layout(xaxis_title="Drop in test ROC-AUC when shuffled", yaxis_title=None)
        show(style(f, 480, f"Top risk drivers - permutation importance ({meta['pd_chosen']})"))
    with c2:
        co = load_report("pd_logreg_coefficients.csv")
        co = pd.concat([co.head(8), co.tail(8)]).drop_duplicates("feature").iloc[::-1]
        f = px.bar(co, x="coef_per_1sd_or_dummy", y="feature", orientation="h",
                   color=np.where(co.coef_per_1sd_or_dummy > 0, "Raises risk", "Lowers risk"),
                   color_discrete_map={"Raises risk": "#a63d2f", "Lowers risk": "#6f8f63"})
        f.update_layout(xaxis_title="Logit coefficient (per 1 SD / dummy)", yaxis_title=None, legend_title=None)
        show(style(f, 480, "Logistic-regression coefficients (reference scorecard model)"))
    with st.expander("Odds ratios (logistic regression)"):
        st.dataframe(co.iloc[::-1][["feature", "coef_per_1sd_or_dummy", "odds_ratio", "direction"]]
                     .style.format({"coef_per_1sd_or_dummy": "{:.3f}", "odds_ratio": "{:.3f}"}), hide_index=True)
    sec("Individual borrower explanation")
    li = st.session_state.get("last_inputs")
    if li:
        up, down, allr = eng.explain(li, top=8)
        r = allr.sort_values("delta_pd")
        f = px.bar(r, x="delta_pd", y="driver", orientation="h",
                   color=np.where(r.delta_pd > 0, "Raises PD", "Lowers PD"),
                   color_discrete_map={"Raises PD": "#a63d2f", "Lowers PD": "#6f8f63"})
        f.update_layout(xaxis_title="Change in PD vs reference borrower", xaxis_tickformat=".1%", yaxis_title=None, legend_title=None)
        show(style(f, 560, "Drivers for the borrower entered on the Individual Assessment tab"))
    st.markdown('<div class="note">SHAP was not used (library unavailable in the build environment). Individual explanations use '
                'model-based sensitivity analysis: each input is replaced by a reference value and the change in the model\'s '
                'PD is recorded. Interactions between inputs mean contributions do not sum exactly to the total difference.</div>',
                unsafe_allow_html=True)

# ================================================================== 7. NOTES
with tabs[6]:
    q = meta["quality"]
    sec("Data & leakage controls")
    st.markdown(f"""
- Raw file: **{q['raw_rows']:,}** loans. Kept **{q['rows_resolved_kept']:,}** with a resolved outcome (Fully Paid vs Charged Off / Default / Late 31-120);
  **{q['rows_excluded_censored']:,}** Current / Grace / Late 16-30 loans were excluded because their outcome is not yet known (right-censoring).
- Default rate in the resolved sample: **{q['default_rate_resolved']:.1%}**. It rises for 2013-14 vintages partly because only early outcomes are resolved - a survivorship/censoring bias.
- **Excluded as post-origination (leakage):** {', '.join(meta['leakage_excluded'])}. These were used only to *construct* the EAD-at-default / LGD targets.
- The dataset has **no FICO score**; lender grade, sub-grade and interest rate (assigned at origination) carry that information.
- Split: stratified 60 / 20 / 20 train / validation / test ({meta['split_sizes']}). Preprocessing is fitted inside pipelines on training folds only; hyper-parameters tuned by 3-fold CV on training data; threshold, calibration and model selection on validation; test touched once for reporting.
""")
    sec("What the numbers are - and are not")
    st.markdown("""
- **PD**: calibrated model probability of default for resolved LendingClub loans.
- **EAD proxy**: typical funded amount for a borrower profile - **not** regulatory EAD.
- **LGD proxy**: realised loss share of outstanding principal after net recoveries, for defaulted loans - **not** Basel LGD, and only weakly predictable.
- **Expected loss** = PD × EAD proxy × LGD proxy: an illustrative calculation, not a provision or capital figure.
- Not validated for production lending: no out-of-time validation, no fairness / regulatory review, US peer-to-peer data from 2007-2014, INR display is a currency conversion only.
""")
    fa = meta["factor_analysis"]
    sec("Factor analysis check")
    st.write(f"KMO = {fa['kmo']:.3f}; Bartlett's test p = {fa['bartlett_p']:.3g}; eigenvalues > 1: {fa['n_factors_kaiser']}. "
             + ("Factorability criteria met - see reports/factor_loadings.csv." if fa["suitable"]
                else "Factorability criteria not met, so factor analysis was not applied."))
