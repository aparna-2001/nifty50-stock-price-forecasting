import streamlit as st

def show():
    st.title("ℹ️ About This Project")

    st.markdown("""
    ## NIFTY 50 Stock Market Forecasting — ML Portfolio Project

    This project builds machine learning models to predict NIFTY 50 index direction and returns
    across two time horizons: **yearly** and **monthly**. It follows a rigorous six-phase
    methodology and documents both successes and limitations honestly.
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🗂️ Project Structure")
        st.markdown("""
        **Phase 1: Data Collection**
        - NSE CSV files (1999–2006)
        - yfinance API (2007–2026)
        - Macro data: Crude Oil (CL=F), Gold (GC=F), USD/INR (INR=X)

        **Phase 2: EDA & Visualization**
        - Price history, return distributions
        - Feature correlation analysis
        - Stationarity checks

        **Phase 3: Feature Engineering**
        - Candlestick structure features
        - Lag features (previous month/year values)
        - Rolling CAGR, momentum features
        - Macroeconomic change percentages

        **Phase 4: Preprocessing**
        - Chronological 80/20 train/test split
        - Walk-forward validation (TimeSeriesSplit)
        - Matched dataset methodology (Phase 1b)

        **Phase 5: Model Building**
        - Ablation study across feature groups (M1–M5)
        - Multiple algorithms compared per task
        - Classification + Regression tasks

        **Phase 6: Evaluation**
        - Test set metrics (Accuracy, F1, MAE, RMSE)
        - Confusion matrices
        - Feature importance analysis
        - Walk-forward cross-validation
        """)

    with col2:
        st.markdown("### 🧠 Key Learnings")
        st.markdown("""
        **1. Fair comparison requires matched datasets**

        Phase 1 and Phase 2 used different row counts due to
        USDINR data gaps pre-2004. A Phase 1b rerun was added
        to ensure valid comparison — a critical methodological fix.

        **2. Structural features outperform return-based features**

        For yearly prediction, candlestick structure features (Body Ratio,
        Shadow %, Recovery Rate) drove all predictions. Return-based
        features (Annual Return, Candle Strength) contributed nothing.

        **3. Macro features are task-specific**

        Crude Oil, Gold, and USD/INR improved XGBoost regression
        (−0.427 MAE) but hurt all classifiers. Adding macro data to
        direction prediction added noise, not signal.

        **4. Mean reversion as a signal**

        High Recovery Rates (closing near yearly high) historically
        signal bearish reversals — this drove the 2026 yearly prediction.

        **5. Data scarcity limits yearly models**

        Only 25 data points for yearly prediction — too small for
        robust ML. Results should be interpreted cautiously.
        """)

    st.markdown("---")

    st.markdown("### 📊 Model Summary")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Yearly Model**")
        st.table({
            "Attribute": ["Algorithm", "Feature Set", "Features Count", "Test Accuracy", "F1 Score", "Walk-Forward Avg"],
            "Value":     ["Random Forest", "M5 — Structural Only", "7", "~80%", "~0.83", "75% (RF)"]
        })

    with col_b:
        st.markdown("**Monthly Models**")
        st.table({
            "Attribute": ["Regression Algorithm", "Regression Features", "Regression MAE", "Regression RMSE",
                          "Classification Algorithm", "Classification Features", "Classification Accuracy", "Classification F1"],
            "Value":     ["XGBoost Regressor", "16 (OHLC + Macro)", "3.384%", "4.232%",
                          "XGBoost Classifier", "10 (OHLC only)", "61.5%", "0.677"]
        })

    st.markdown("---")

    st.markdown("### ⚠️ Limitations")
    st.warning("""
    - **Yearly model:** Only 25 data points — data scarcity limits statistical robustness
    - **Class imbalance:** Only 4 bearish years in yearly data
    - **Monthly model:** ~60% direction accuracy is modest; market direction is inherently noisy
    - **No supply-side data:** Macro signals (Crude, Gold) don't capture OPEC decisions, EIA inventory
    - **Survivorship bias:** NIFTY 50 composition has changed over 25 years
    - **No sentiment data:** FII/DII flows, news sentiment not included
    - **Predictions are probabilistic:** Not financial advice
    """)

    st.markdown("---")

    st.markdown("### 🔭 Future Work (Phase 3)")
    st.info("""
    - **Daily NIFTY 50 model** — higher frequency prediction
    - **Gold prediction model** — same pipeline applied to MCX Gold
    - **Crude Oil model improvement** — add EIA inventory, OPEC production data
    - **Stacked pipeline** — Crude Oil and Gold sub-models feed predictions into the NIFTY monthly model
    - **Kaggle and GitHub publication** — cleaned datasets and full notebooks
    """)

    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    cols = st.columns(5)
    tools = ["Python", "Pandas", "Scikit-learn", "XGBoost", "Streamlit"]
    icons = ["🐍", "🐼", "⚙️", "🚀", "📊"]
    for col, tool, icon in zip(cols, tools, icons):
        col.markdown(f"**{icon} {tool}**")

    st.markdown("---")
    st.caption("Built by Aparna · NIFTY 50 ML Portfolio Project · Data: NSE + yfinance")
