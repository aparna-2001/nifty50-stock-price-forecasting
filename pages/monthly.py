import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from utils import (
    train_monthly_models,
    MONTHLY_FEATURES_REG, MONTHLY_FEATURES_CLF
)

def show():
    st.title("🗓️ Monthly NIFTY 50 Prediction")
    st.markdown(
        "Predicts **next month's return %** (regression) and **direction** (classification) "
        "using OHLC candlestick features and macroeconomic indicators."
    )

    # ── Load & train ─────────────────────────────────────────────────────────
    with st.spinner("Training monthly models… (fetching macro data from yfinance)"):
        reg_model, clf_model, reg_metrics, clf_metrics, monthly_v2, monthly_1b, macro = train_monthly_models()

    # ── Latest prediction ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔮 April 2026 Prediction")
    st.markdown("Based on partial March 2026 data — mirrors the notebook's live prediction approach.")

    with st.spinner("Fetching March 2026 data from yfinance…"):
        try:
            march_ohlc = yf.download("^NSEI", start="2026-03-01", end="2026-03-22", progress=False)
            macro_recent = yf.download(["CL=F", "GC=F", "INR=X"], start="2026-03-01", end="2026-03-22", progress=False)["Close"]

            march_open  = march_ohlc["Open"].iloc[0].item()
            march_close = march_ohlc["Close"].iloc[-1].item()
            march_high  = march_ohlc["High"].max().item()
            march_low   = march_ohlc["Low"].min().item()
            avg_volume  = march_ohlc["Volume"].mean().item()

            monthly_return = (march_close - march_open) / march_open * 100
            hl_range       = (march_high - march_low) / march_open * 100
            body_ratio     = abs(march_close - march_open) / (march_high - march_low)
            upper_shadow   = (march_high - max(march_open, march_close)) / march_open * 100
            lower_shadow   = (min(march_open, march_close) - march_low) / march_open * 100
            recovery_rate  = (march_close - march_low) / (march_high - march_low) * 100

            feb_row = monthly_1b[monthly_1b["Month"] == "2026-02-01"]
            if len(feb_row) == 0:
                feb_row = monthly_1b.iloc[-1:]
            prev_bull = feb_row["Bull_Return_%"].values[0]
            prev_bear = feb_row["Bear_Return_%"].values[0]

            mar25_row = monthly_1b[monthly_1b["Month"] == "2025-03-01"]
            if len(mar25_row):
                rolling_cagr = (march_close / mar25_row["Month_Close"].values[0] - 1) * 100
            else:
                rolling_cagr = monthly_1b["Rolling_CAGR_%"].iloc[-1]

            # Macro changes
            macro_date_col = macro.columns[0]
            feb_macro = macro[pd.to_datetime(macro[macro_date_col]) == "2026-02-01"]
            if len(feb_macro) == 0:
                feb_macro = macro.iloc[-1:]

            crude_feb  = feb_macro["Crude"].values[0]
            gold_feb   = feb_macro["Gold"].values[0]
            usdinr_feb = feb_macro["USDINR"].values[0]

            crude_now  = macro_recent["CL=F"].iloc[-1]  if "CL=F"  in macro_recent else crude_feb
            gold_now   = macro_recent["GC=F"].iloc[-1]  if "GC=F"  in macro_recent else gold_feb
            usdinr_now = macro_recent["INR=X"].iloc[-1] if "INR=X" in macro_recent else usdinr_feb

            crude_change  = (crude_now  - crude_feb)  / crude_feb  * 100
            gold_change   = (gold_now   - gold_feb)   / gold_feb   * 100
            usdinr_change = (usdinr_now - usdinr_feb) / usdinr_feb * 100

            prev_crude_change  = feb_macro["Crude_Change_%"].values[0]
            prev_gold_change   = feb_macro["Gold_Change_%"].values[0]
            prev_usdinr_change = feb_macro["Prev_USDINR_Change_%"].values[0]

            input_reg = pd.DataFrame([{
                "Monthly_Return_%":     monthly_return,
                "HL_Range_%":           hl_range,
                "Body_Ratio":           body_ratio,
                "Upper_Shadow_%":       upper_shadow,
                "Lower_Shadow_%":       lower_shadow,
                "Recovery_Rate_%":      recovery_rate,
                "Avg_Volume":           avg_volume,
                "Prev_Bull_Return_%":   prev_bull,
                "Prev_Bear_Return_%":   prev_bear,
                "Rolling_CAGR_%":       rolling_cagr,
                "Crude_Change_%":       crude_change,
                "Gold_Change_%":        gold_change,
                "USDINR_Change_%":      usdinr_change,
                "Prev_Crude_Change_%":  prev_crude_change,
                "Prev_Gold_Change_%":   prev_gold_change,
                "Prev_USDINR_Change_%": prev_usdinr_change,
            }])
            input_clf = input_reg[MONTHLY_FEATURES_CLF]

            pred_return    = reg_model.predict(input_reg)[0]
            pred_direction = clf_model.predict(input_clf)[0]
            pred_proba     = clf_model.predict_proba(input_clf)[0]
            live_data_ok   = True

        except Exception as e:
            st.warning(f"Could not fetch live March 2026 data ({e}). Showing latest available month prediction.")
            live_data_ok = False
            pred_return    = 0.34
            pred_direction = 0
            pred_proba     = [0.54, 0.46]
            march_close    = None

    if live_data_ok:
        col1, col2, col3, col4 = st.columns(4)
        is_bull = pred_direction == 1
        dir_color = "#16A34A" if is_bull else "#DC2626"
        dir_label = "BULLISH 📈" if is_bull else "BEARISH 📉"

        with col1:
            st.markdown(
                f"""<div style='background:{dir_color}18; border-left:5px solid {dir_color};
                              padding:18px; border-radius:8px;'>
                    <h3 style='color:{dir_color}; margin:0'>{dir_label}</h3>
                    <p style='margin:4px 0 0; color:#555'>April 2026 Direction</p>
                </div>""",
                unsafe_allow_html=True
            )
        with col2:
            st.metric("Predicted Return", f"{pred_return:+.2f}%", "April 2026")
        with col3:
            st.metric("Bullish Probability", f"{pred_proba[1]*100:.1f}%")
        with col4:
            if march_close:
                st.metric("March 2026 Close", f"₹{march_close:,.0f}")

    st.markdown("---")

    # ── Model Performance ─────────────────────────────────────────────────────
    st.subheader("🎯 Model Performance")

    tab1, tab2 = st.tabs(["📉 Regression (Return %)", "🔀 Classification (Direction)"])

    with tab1:
        st.markdown("**XGBoost Regressor — 16 features (OHLC + Macro)**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Algorithm",  "XGBoost Regressor")
        c2.metric("Features",   "16 (OHLC + Macro)")
        c3.metric("MAE",        f"{reg_metrics['mae']:.3f}%")
        c4.metric("RMSE",       f"{reg_metrics['rmse']:.3f}%")

        st.caption(
            f"Train: {reg_metrics['train_period']} ({reg_metrics['n_train']} months) | "
            f"Test: {reg_metrics['test_period']} ({reg_metrics['n_test']} months)"
        )

        # Actual vs Predicted plot
        split = reg_metrics["n_train"]
        y_test = monthly_v2["Next_Month_Return_%"].iloc[split:]
        y_pred = reg_model.predict(monthly_v2[MONTHLY_FEATURES_REG])
        y_pred_test = y_pred[split:]
        months_test = monthly_v2["Month"].iloc[split:]

        fig_reg = go.Figure()
        fig_reg.add_trace(go.Scatter(
            x=months_test, y=y_test,
            mode="lines", name="Actual Return %",
            line=dict(color="#2563EB", width=1.5)
        ))
        fig_reg.add_trace(go.Scatter(
            x=months_test, y=y_pred_test,
            mode="lines", name="Predicted Return %",
            line=dict(color="#F59E0B", width=1.5, dash="dash")
        ))
        fig_reg.add_hline(y=0, line_color="black", line_width=0.8)
        fig_reg.update_layout(
            height=320, margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Month", yaxis_title="Return (%)",
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig_reg, use_container_width=True)

        # Feature importance
        st.markdown("**Feature Importance — XGBoost Regression**")
        imp_reg = pd.Series(reg_model.feature_importances_, index=MONTHLY_FEATURES_REG).sort_values()
        fig_imp = go.Figure(go.Bar(
            x=imp_reg.values, y=imp_reg.index,
            orientation="h", marker_color="#2563EB",
        ))
        fig_imp.update_layout(
            height=420, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with tab2:
        st.markdown("**XGBoost Classifier — 10 OHLC features (macro excluded)**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Algorithm",  "XGBoost Classifier")
        c2.metric("Features",   "10 (OHLC only)")
        c3.metric("Accuracy",   f"{clf_metrics['accuracy']*100:.1f}%")
        c4.metric("F1 Score",   f"{clf_metrics['f1']:.3f}")

        st.caption(
            f"Train: {clf_metrics['train_period']} ({clf_metrics['n_train']} months) | "
            f"Test: {clf_metrics['test_period']} ({clf_metrics['n_test']} months)"
        )

        col_cm, col_fi2 = st.columns(2)

        with col_cm:
            st.markdown("**Confusion Matrix**")
            cm = clf_metrics["cm"]
            fig_cm = go.Figure(go.Heatmap(
                z=cm,
                x=["Predicted Down", "Predicted Up"],
                y=["Actual Down", "Actual Up"],
                colorscale="Blues",
                text=cm.tolist(), texttemplate="%{text}",
                showscale=False,
            ))
            fig_cm.update_layout(
                height=280, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="white"
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_fi2:
            st.markdown("**Feature Importance — XGBoost Classification**")
            imp_clf = pd.Series(clf_model.feature_importances_, index=MONTHLY_FEATURES_CLF).sort_values()
            fig_clf_fi = go.Figure(go.Bar(
                x=imp_clf.values, y=imp_clf.index,
                orientation="h", marker_color="#16A34A",
            ))
            fig_clf_fi.update_layout(
                height=280, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
            )
            st.plotly_chart(fig_clf_fi, use_container_width=True)

        st.info(
            "**Why macro features were excluded from classification:** "
            "Adding Crude Oil, Gold, and USD/INR features reduced XGBoost classification accuracy "
            "from 61.5% to 51.9% — macro features added noise rather than signal for direction prediction."
        )

    # ── Phase comparison table ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Phase Comparison — OHLC Baseline vs OHLC + Macro")
    st.markdown("Fair comparison using matched 257-row dataset (2004–2026). Key methodological decision: Phase 1b rerun.")

    tab_r, tab_c = st.tabs(["Regression", "Classification"])

    with tab_r:
        reg_table = pd.DataFrame({
            "Phase": ["Phase 1b (OHLC)", "Phase 1b (OHLC)", "Phase 1b (OHLC)",
                      "Phase 2 (OHLC + Macro)", "Phase 2 (OHLC + Macro)", "Phase 2 (OHLC + Macro)"],
            "Model": ["Ridge", "Random Forest", "XGBoost"] * 2,
            "MAE":   [2.858, 3.049, 3.811, 2.982, 3.134, 3.384],
            "RMSE":  [3.480, 3.986, 4.709, 3.625, 4.003, 4.232],
            "MAE Δ": ["—", "—", "—", "❌ +0.124", "❌ +0.085", "✅ −0.427"],
        })
        st.dataframe(reg_table, hide_index=True, use_container_width=True)
        st.caption("XGBoost regression improved with macro features (−0.427 MAE). Ridge and RF did not benefit.")

    with tab_c:
        clf_table = pd.DataFrame({
            "Phase": ["Phase 1b (OHLC)", "Phase 1b (OHLC)", "Phase 1b (OHLC)",
                      "Phase 2 (OHLC + Macro)", "Phase 2 (OHLC + Macro)", "Phase 2 (OHLC + Macro)"],
            "Model": ["Logistic Reg", "Random Forest", "XGBoost ✅"] * 2,
            "Accuracy": [0.558, 0.596, 0.615, 0.558, 0.577, 0.519],
            "F1":       [0.716, 0.644, 0.677, 0.716, 0.686, 0.615],
            "Acc Δ":    ["—", "—", "—", "—", "❌ −0.019", "❌ −0.096"],
        })
        st.dataframe(clf_table, hide_index=True, use_container_width=True)
        st.caption("Macro features hurt all classifiers. XGBoost Phase 1b selected as final classification model.")

    # ── Monthly returns heatmap ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📅 Monthly Returns Heatmap")

    monthly_1b["Year"]  = pd.to_datetime(monthly_1b["Month"]).dt.year
    monthly_1b["Month_Num"] = pd.to_datetime(monthly_1b["Month"]).dt.month

    pivot = monthly_1b.pivot_table(
        values="Monthly_Return_%", index="Year", columns="Month_Num"
    )
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig_hm = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[
            [0,   "#991B1B"],
            [0.4, "#FCA5A5"],
            [0.5, "#FFFFFF"],
            [0.6, "#86EFAC"],
            [1,   "#15803D"],
        ],
        zmid=0,
        text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=8),
        showscale=True,
        colorbar=dict(title="Return %"),
    ))
    fig_hm.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Month",
        yaxis_title="Year",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_hm, use_container_width=True)
    st.caption("Green = positive return, Red = negative return. Hover for exact values.")
