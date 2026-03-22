import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import load_yearly_data, load_monthly_data

def show():
    st.title("📈 NIFTY 50 — Stock Market Prediction")
    st.markdown(
        """
        Welcome to the **NIFTY 50 Prediction Dashboard** — a machine learning portfolio project
        that forecasts Indian stock market direction and returns using candlestick structure,
        momentum, and macroeconomic signals.
        """
    )

    # ── Key stats row ──────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Data Coverage", "1999 – 2026", "25+ years")
    col2.metric("Yearly Model", "Random Forest", "75% Walk-Forward Acc")
    col3.metric("Monthly Regression", "XGBoost", "MAE 3.38%")
    col4.metric("Monthly Direction", "XGBoost", "61.5% Accuracy")

    st.markdown("---")

    # ── NIFTY price chart ──────────────────────────────────────────────────
    st.subheader("NIFTY 50 — 25 Year Price History")

    with st.spinner("Loading price data…"):
        yearly = load_yearly_data()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly["Year"], y=yearly["Year_Close"],
        mode="lines+markers",
        line=dict(color="#2563EB", width=2),
        marker=dict(size=5),
        name="Year Close",
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.08)"
    ))
    fig.add_trace(go.Scatter(
        x=yearly["Year"], y=yearly["Year_Open"],
        mode="lines",
        line=dict(color="#F59E0B", width=1.5, dash="dash"),
        name="Year Open"
    ))
    fig.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.05),
        xaxis_title="Year",
        yaxis_title="NIFTY 50 Level",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Annual returns bar chart ──────────────────────────────────────────
    st.subheader("Annual Returns (%) — Bull vs Bear Years")
    colors = ["#16A34A" if r > 0 else "#DC2626" for r in yearly["Annual_Return_%"]]
    fig2 = go.Figure(go.Bar(
        x=yearly["Year"], y=yearly["Annual_Return_%"],
        marker_color=colors,
        text=[f"{r:.1f}%" for r in yearly["Annual_Return_%"]],
        textposition="outside",
        textfont=dict(size=9),
    ))
    fig2.add_hline(y=0, line_color="black", line_width=1)
    fig2.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Year",
        yaxis_title="Return (%)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── Project overview ───────────────────────────────────────────────────
    st.subheader("Project Architecture")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("##### 📅 Yearly Prediction")
        st.markdown("""
        - **Task:** Classify next year as Bullish or Bearish
        - **Algorithm:** Random Forest (M5 — structural features)
        - **Features:** HL Range, Recovery Rate, Body Ratio, Shadow %
        - **Methodology:** Ablation study across 5 feature sets (M1–M5)
        - **Validation:** Walk-forward (TimeSeriesSplit, 5 folds)
        - **Key finding:** Structural candlestick features outperform return-based features
        """)

    with col_b:
        st.markdown("##### 🗓️ Monthly Prediction")
        st.markdown("""
        - **Task:** Predict next month's return % + direction
        - **Regression:** XGBoost — 16 features (OHLC + Macro)
        - **Classification:** XGBoost — 10 OHLC features
        - **Macro factors:** Crude Oil, Gold, USD/INR (current + lagged)
        - **Key finding:** Macro features improve regression but add noise to classification
        - **Phase comparison:** Matched 257-row datasets for fair evaluation
        """)

    st.markdown("---")
    st.caption("Navigate using the sidebar → to view predictions and model details.")
