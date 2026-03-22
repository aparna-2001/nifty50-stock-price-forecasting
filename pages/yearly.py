import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import (
    load_yearly_data, train_yearly_model,
    YEARLY_FEATURES_M5, predict_yearly
)

def show():
    st.title("📅 Yearly NIFTY 50 Prediction")
    st.markdown("Predicts whether the **next calendar year** will be Bullish or Bearish using candlestick structure features.")

    # ── Load & train ─────────────────────────────────────────────────────────
    with st.spinner("Training yearly model…"):
        model, metrics, yearly = train_yearly_model()

    # ── 2026 Prediction card ─────────────────────────────────────────────────
    result = predict_yearly(model, yearly)

    st.markdown("---")
    st.subheader(f"🔮 {result['year']} Prediction")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        is_bull = "BULLISH" in result["direction"]
        color   = "#16A34A" if is_bull else "#DC2626"
        icon    = "📈" if is_bull else "📉"
        st.markdown(
            f"""
            <div style='background:{color}18; border-left: 5px solid {color};
                        padding:20px; border-radius:8px;'>
                <h2 style='color:{color}; margin:0'>{icon} {result['direction']}</h2>
                <p style='margin:4px 0 0; color:#555'>Prediction for {result['year']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.metric("Bullish Probability", f"{result['bullish_pct']:.1f}%")
    with col3:
        st.metric("Bearish Probability", f"{result['bearish_pct']:.1f}%")

    # probability donut
    fig_donut = go.Figure(go.Pie(
        labels=["Bullish", "Bearish"],
        values=[result["bullish_pct"], result["bearish_pct"]],
        hole=0.55,
        marker=dict(colors=["#16A34A", "#DC2626"]),
        textinfo="label+percent",
        hoverinfo="label+percent"
    ))
    fig_donut.update_layout(
        height=280, margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False, paper_bgcolor="white"
    )
    st.plotly_chart(fig_donut, use_container_width=True)

    # ── Reasoning ─────────────────────────────────────────────────────────────
    latest_year = int(yearly["Year"].iloc[-1])
    rr = yearly["Recovery_Rate_%"].iloc[-1]
    br = yearly["Body_Ratio"].iloc[-1]
    us = yearly["Upper_Shadow_%"].iloc[-1]

    st.markdown("##### 🧠 Why this prediction?")
    st.info(
        f"**{latest_year} closed** with a Recovery Rate of **{rr:.1f}%** — "
        f"meaning the index finished very close to its yearly high. "
        f"Historically, high recovery rates signal potential mean reversion the following year. "
        f"Combined with a Body Ratio of **{br:.2f}** and Upper Shadow of **{us:.1f}%**, "
        f"the model leans {'Bearish' if not is_bull else 'Bullish'} for {result['year']}."
    )

    # ── Model Performance ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🎯 Model Performance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Algorithm",  "Random Forest")
    c2.metric("Features",   f"{len(YEARLY_FEATURES_M5)} (M5 Structural)")
    c3.metric("Test Accuracy", f"{metrics['accuracy']*100:.1f}%")
    c4.metric("F1 Score",   f"{metrics['f1']:.3f}")

    col_cm, col_fi = st.columns(2)

    with col_cm:
        st.markdown("**Confusion Matrix (Test Set)**")
        cm = metrics["cm"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=["Predicted Bearish", "Predicted Bullish"],
            y=["Actual Bearish", "Actual Bullish"],
            colorscale="Blues",
            text=cm.tolist(),
            texttemplate="%{text}",
            showscale=False,
        ))
        fig_cm.update_layout(
            height=280, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="white"
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption(f"Train: {metrics['train_years']} ({metrics['n_train']} years) | Test: {metrics['test_years']} ({metrics['n_test']} years)")

    with col_fi:
        st.markdown("**Feature Importance**")
        # Use full-data model importances
        imp = pd.Series(model.feature_importances_, index=YEARLY_FEATURES_M5).sort_values()
        fig_fi = go.Figure(go.Bar(
            x=imp.values, y=imp.index,
            orientation="h",
            marker_color="#2563EB",
            text=[f"{v:.3f}" for v in imp.values],
            textposition="outside"
        ))
        fig_fi.update_layout(
            height=280, margin=dict(l=0, r=0, t=10, b=60),
            xaxis_title="Importance Score",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Walk-forward table ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔄 Walk-Forward Validation Results")
    st.markdown("TimeSeriesSplit with 5 folds — mirrors the notebook evaluation exactly.")

    wf_data = {
        "Algorithm": ["XGBoost", "XGBoost", "XGBoost", "XGBoost",
                      "Random Forest", "Random Forest", "Random Forest", "Random Forest",
                      "Logistic Reg", "Logistic Reg", "Logistic Reg", "Logistic Reg"],
        "Fold":      [2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5],
        "Accuracy":  [75.0, 75.0, 50.0, 75.0,
                      75.0, 75.0, 75.0, 75.0,
                      100.0, 75.0, 50.0, 75.0],
    }
    wf_df = pd.DataFrame(wf_data)
    avg = wf_df.groupby("Algorithm")["Accuracy"].mean().reset_index()
    avg.columns = ["Algorithm", "Avg Accuracy"]
    avg["Avg Accuracy"] = avg["Avg Accuracy"].map("{:.1f}%".format)
    avg = avg.sort_values("Avg Accuracy", ascending=False)

    col_wf1, col_wf2 = st.columns([2, 1])
    with col_wf1:
        fig_wf = px.line(
            wf_df, x="Fold", y="Accuracy", color="Algorithm",
            markers=True, color_discrete_sequence=["#2563EB","#16A34A","#F59E0B"]
        )
        fig_wf.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Baseline (50%)")
        fig_wf.update_layout(
            height=280, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Accuracy (%)", yaxis=dict(range=[30, 110]),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_wf, use_container_width=True)

    with col_wf2:
        st.markdown("**Average Walk-Forward Accuracy**")
        st.dataframe(avg, hide_index=True, use_container_width=True)
        st.markdown("""
        **Why Random Forest was selected:**
        - Consistent 75% across all folds
        - XGBoost dropped to 50% in Fold 4
        - Logistic Regression was inconsistent (100% → 50%)
        """)

    # ── Historical predictions table ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Historical Data & Features")

    display_cols = ["Year", "Annual_Return_%", "Recovery_Rate_%", "Body_Ratio",
                    "Upper_Shadow_%", "Lower_Shadow_%", "HL_Range_%", "Target"]
    display_df = yearly[display_cols].copy()
    display_df["Target"] = display_df["Target"].map({1: "Bullish ↑", 0: "Bearish ↓"})
    display_df = display_df.rename(columns={
        "Annual_Return_%": "Annual Return %",
        "Recovery_Rate_%": "Recovery Rate %",
        "Body_Ratio": "Body Ratio",
        "Upper_Shadow_%": "Upper Shadow %",
        "Lower_Shadow_%": "Lower Shadow %",
        "HL_Range_%": "HL Range %",
    })

    def highlight_target(val):
        if val == "Bullish ↑":
            return "background-color: #dcfce7; color: #15803d"
        elif val == "Bearish ↓":
            return "background-color: #fee2e2; color: #991b1b"
        return ""

    styled = display_df.style\
        .format({
            "Annual Return %":  "{:.2f}%",
            "Recovery Rate %":  "{:.1f}%",
            "Body Ratio":       "{:.3f}",
            "Upper Shadow %":   "{:.1f}%",
            "Lower Shadow %":   "{:.1f}%",
            "HL Range %":       "{:.1f}%",
        })\
        .applymap(highlight_target, subset=["Target"])

    st.dataframe(styled, height=400, use_container_width=True)

    # ── Ablation study ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🧪 Ablation Study — Feature Group Comparison")
    st.markdown("All 5 XGBoost models were tested on the same 80/20 chronological split.")

    ablation_data = {
        "Model": ["M1 — All Features", "M2 — No Annual Return", "M3 — No Candle Strength",
                  "M4 — No Return Diff", "M5 — Structural Only ✅"],
        "Features": [10, 9, 9, 9, 7],
        "Accuracy": ["60%", "60%", "60%", "60%", "60%"],
        "Note": [
            "Baseline", "Same result", "Same result",
            "Same result", "Selected — cleaner, no redundant features"
        ]
    }
    st.dataframe(pd.DataFrame(ablation_data), hide_index=True, use_container_width=True)
    st.caption("Key finding: Return-based features (Annual Return, Candle Strength, Return Diff) contribute nothing — structural features drive all predictions.")
