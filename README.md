# WattWise Energy Forecasting

An interactive Streamlit dashboard for appliance energy forecasting using time-series machine learning.

## Features

- Chronological, leakage-safe forecasting workflow
- Lag and rolling-window features
- Ridge, tree-based models and Optuna tuning
- Forecast explorer with actual-vs-predicted charts
- Residual and worst-miss error analysis
- Permutation-based model explainability
- Data quality and feature health checks

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.
