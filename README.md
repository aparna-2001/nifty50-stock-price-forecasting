# NIFTY 50 Prediction — Streamlit App

A machine learning dashboard for predicting NIFTY 50 yearly and monthly direction and returns.

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place data file
Make sure `nifty50_25years_ohlcv_1999_2026.csv` is in the same folder as `app.py`.

### 3. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Pages

| Page | Description |
|------|-------------|
| 🏠 Home | 25-year price history, annual returns, project overview |
| 📅 Yearly Prediction | Random Forest classifier, 2026 prediction, walk-forward validation, ablation study |
| 🗓️ Monthly Prediction | XGBoost regressor + classifier, April 2026 prediction, phase comparison table, monthly heatmap |
| ℹ️ About | Methodology, key learnings, limitations, future work |

---

## Models

**Yearly:**
- Algorithm: Random Forest Classifier
- Features: M5 structural (HL Range, Recovery Rate, Body Ratio, Shadow %)
- Walk-forward accuracy: 75%

**Monthly Regression:**
- Algorithm: XGBoost Regressor
- Features: 16 (OHLC + Crude Oil, Gold, USD/INR — current + lagged)
- MAE: ~3.38%

**Monthly Classification:**
- Algorithm: XGBoost Classifier
- Features: 10 (OHLC only — macro excluded)
- Accuracy: ~61.5%

---

## Notes
- Macro data (Crude, Gold, USD/INR) is fetched live from yfinance on app startup
- The app degrades gracefully if yfinance is unavailable (OHLC-only fallback)
- All model predictions are probabilistic — not financial advice
