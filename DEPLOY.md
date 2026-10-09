# Deployment guide

The dashboard only needs: `dashboard/`, `src/`, `models/`, `reports/`, `data/portfolio_scored.csv.gz`,
`.streamlit/`, `requirements.txt`. The raw LendingClub file is NOT needed (and must not be committed - 240 MB).

## Option A - Streamlit Community Cloud (free)
1. Push this folder to a GitHub repo (largest file is ~13 MB, so Git LFS is not needed).
2. share.streamlit.io -> New app -> pick the repo, branch `main`, **Main file path: `dashboard/app.py`**.
3. Advanced settings -> Python **3.12** (scikit-learn 1.8.0 needs Python 3.11+; 3.14 also works if its pandas/numpy install cleanly).
4. Deploy. First build takes a few minutes. If the log shows a scikit-learn version error, retrain on the host's
   version (`python -m src.run_pipeline --raw ...`) or pin that version in requirements.txt.

## Option B - Docker (any cloud: Render, Railway, Azure, AWS, Cloud Run)
```bash
docker build -t credit-risk-dashboard .
docker run -p 8501:8501 credit-risk-dashboard      # open http://localhost:8501
```

## Before you share the link
- This app has no authentication. Streamlit Cloud lets you restrict viewers by email in app settings; on other hosts put it behind a login/VPN.
- Check the USD->INR rate in the sidebar (assumed 88).
- It is an academic model on 2007-2014 data: label it "decision-support demo", not a lending tool.
- First-run checklist: open every tab in INR and USD, run Individual Assessment with a low-risk and a high-risk borrower, confirm no red error boxes.
