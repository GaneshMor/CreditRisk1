# Credit Risk Assessment & Customer Segmentation (LendingClub 2007-2014)

Academic / portfolio project: PD, EAD proxy, LGD proxy, Expected Loss, K-Means segmentation and a Streamlit
decision-support dashboard. **Not production-ready and not validated for real lending decisions.**

## 1. Business problem
Given a borrower's origination-time characteristics: what is the probability of default, the estimated exposure,
the likely loss severity, the expected monetary loss, and the risk segment?

## 2. Dataset
`loan_data_2007_2014.csv` (raw LendingClub, 466,285 loans). It is NOT included in this folder (240 MB).
The modelling-ready file built from it is `data/LendingClub_2007_2014_modeling_ready.csv.gz`
(237,695 loans with a resolved outcome). The raw file has no `pd_target` / EAD / LGD columns; they are constructed in `src/preprocessing.py`.

## 3. Data preparation and leakage
- Default = Charged Off, Default, Late (31-120), "Does not meet policy: Charged Off". Non-default = Fully Paid. 
  Current / In Grace / Late (16-30) are excluded (outcome unknown, right-censored). Default rate in the kept sample: 21.4%.
- Post-origination fields (payments, recoveries, outstanding principal, last-payment info, recovery fees, loan_status...)
  are never predictors; `total_rec_prncp`, `recoveries`, `collection_recovery_fee` are used only to build targets.
- Dataset has no FICO score; lender sub-grade and interest rate (set at origination) are used instead.
- Split: stratified 60/20/20 train/validation/test. Imputation, winsorising, scaling and encoding live inside sklearn pipelines.
  Hyper-parameters: 3-fold CV on training data. Calibration, cut-off and model selection on validation. Test used for reporting.

## 4. Methodology
| Task | Target | Models compared | Selected |
|---|---|---|---|
| PD | `pd_target` | Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting | Logistic Regression (rule below) |
| EAD proxy | funded amount | Linear, Ridge, Lasso, Decision Tree, Random Forest, HistGB | HistGB (test R² 0.57) |
| LGD proxy | 1 - net recoveries / outstanding principal at default, clipped [0,1]; defaulted loans only | same six | Linear Regression (test R² 0.03) |
| Segments | unsupervised | K-Means, K = 2..8 | K = 4 (best silhouette within 3-6) |

- **PD selection rule:** highest validation ROC-AUC, unless Logistic Regression is within 0.005 (then the interpretable model wins).
  Here HistGB scored 0.7134 vs 0.7087 on validation, so the margin is borderline; HistGB is a defensible alternative (test ROC-AUC 0.714 vs 0.709).
  Probabilities are Platt-calibrated on validation. Cut-off for 0/1 metrics = F1-optimal on validation.
- **Risk bands:** Low < 15.5% PD, Medium 15.5-30.9%, High >= 30.9% (40th / 80th percentile of validation PD); observed default
  rates on test rise monotonically across bands (about 10% / 23% / 42%).
- **EAD proxy:** typical funded exposure for a profile, with loan amount and its ratios excluded from predictors. NOT regulatory EAD.
- **LGD proxy:** realised loss share; weakly predictable from origination data (R² ~0.03, mean LGD ~94%). NOT Basel LGD.
- **Expected loss** = PD x EAD proxy x LGD proxy (illustrative).
- **Factor analysis:** KMO 0.611, Bartlett p < 0.001, 6 factors (Kaiser). Loadings in `reports/factor_loadings.csv`.
- **Explainability:** logistic coefficients / odds ratios, permutation importance, and a local sensitivity analysis for individuals
  (swap each input for a reference value, record the PD change). SHAP was not used.

## 5. Run locally
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard/app.py                          # uses the shipped models/ and reports/
# optional: retrain (about 10 min) after putting the raw CSV anywhere
python -m src.run_pipeline --raw path/to/loan_data_2007_2014.csv
python tests/smoke_test.py                              # scoring sanity checks
```
Dashboard pages: Executive Overview, Individual Assessment, Model Performance, Risk Segmentation, Portfolio Analytics,
Explainability, Notes & Limitations. Currency defaults to INR using an editable USD->INR rate (assumed 88; source data is USD).

## 6. Limitations
- Dashboard code was verified with a fake-Streamlit logic test, not rendered in a live Streamlit session during development: check layout on first run.
- Resolved-loans-only sample: censoring/survivorship bias; no out-of-time validation; 2007-2014 US peer-to-peer lending.
- EAD and LGD are proxies; LGD model has almost no explanatory power. No fairness or regulatory review.
- Segments are weakly separated (silhouette ~0.12); labels are descriptive.

## 7. Future improvements
Out-of-time validation, survival / time-to-default models, true EAD from payment schedules, workout-based LGD, SHAP,
monotonic-constraint scorecard, fairness testing.

## Structure
`data/` model-ready and scored data | `src/` pipeline modules | `models/` joblib artifacts | `reports/` metrics, tables, figures | `dashboard/app.py` | `tests/`
(`notebooks/` is empty: the analysis is scripted in `src/run_pipeline.py`.)
