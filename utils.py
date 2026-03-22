"""
Shared data loading, feature engineering, and model training functions.
Mirrors the exact pipeline from the notebooks.
"""

import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_absolute_error, root_mean_squared_error,
    confusion_matrix
)
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "nifty50_25years_ohlcv_1999_2026.csv")

# ══════════════════════════════════════════════════════════════════════════════
# YEARLY PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_yearly_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df.set_index("Date", inplace=True)

    yearly = df.groupby(df.index.year).agg(
        Year_Open  = ("Open",  "first"),
        Year_Close = ("Close", "last"),
        Year_High  = ("High",  "max"),
        Year_Low   = ("Low",   "min"),
    ).reset_index().rename(columns={"Date": "Year"})

    # Feature engineering (exact notebook logic)
    yearly["Annual_Return_%"]   = (yearly["Year_Close"] - yearly["Year_Open"]) / yearly["Year_Open"] * 100
    yearly["Candle_Strength_%"] = (yearly["Year_Close"] - yearly["Year_Open"]) / yearly["Year_Close"] * 100
    yearly["Return_Diff"]       = yearly["Annual_Return_%"] - yearly["Candle_Strength_%"]
    yearly["HL_Range_%"]        = (yearly["Year_High"] - yearly["Year_Low"]) / yearly["Year_Low"] * 100
    yearly["Prev_HL_Range_%"]   = yearly["HL_Range_%"].shift(1)
    yearly["Recovery_Rate_%"]   = (yearly["Year_Close"] - yearly["Year_Low"]) / (yearly["Year_High"] - yearly["Year_Low"]) * 100
    yearly["Prev_Recovery_Rate_%"] = yearly["Recovery_Rate_%"].shift(1)
    yearly["Body_Ratio"]        = abs(yearly["Year_Close"] - yearly["Year_Open"]) / (yearly["Year_High"] - yearly["Year_Low"])
    yearly["Upper_Shadow_%"]    = (yearly["Year_High"] - yearly[["Year_Open","Year_Close"]].max(axis=1)) / yearly["Year_High"] * 100
    yearly["Lower_Shadow_%"]    = (yearly[["Year_Open","Year_Close"]].min(axis=1) - yearly["Year_Low"]) / yearly["Year_Low"] * 100

    yearly.dropna(inplace=True)

    # Target: did next year close > open?
    yearly["Target"] = (yearly["Year_Close"].shift(-1) > yearly["Year_Open"].shift(-1)).astype(int)
    yearly = yearly[:-1]  # drop last row (no target)

    return yearly


YEARLY_FEATURES_M5 = [
    "HL_Range_%", "Prev_HL_Range_%",
    "Recovery_Rate_%", "Prev_Recovery_Rate_%",
    "Body_Ratio", "Upper_Shadow_%", "Lower_Shadow_%"
]

@st.cache_resource(show_spinner=False)
def train_yearly_model():
    yearly = load_yearly_data()
    X = yearly[YEARLY_FEATURES_M5]
    y = yearly["Target"]

    split = int(len(yearly) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1":       f1_score(y_test, preds, zero_division=0),
        "cm":       confusion_matrix(y_test, preds),
        "train_years": f"{yearly['Year'].iloc[0]}–{yearly['Year'].iloc[split-1]}",
        "test_years":  f"{yearly['Year'].iloc[split]}–{yearly['Year'].iloc[-1]}",
        "n_train": split,
        "n_test":  len(yearly) - split,
    }

    # Retrain on full data for prediction
    model_full = RandomForestClassifier(n_estimators=100, random_state=42)
    model_full.fit(X, y)

    return model_full, metrics, yearly


def predict_yearly(model, yearly):
    """Predict next year direction using latest year's features."""
    latest = yearly[YEARLY_FEATURES_M5].iloc[-1:]
    pred   = model.predict(latest)[0]
    proba  = model.predict_proba(latest)[0]
    latest_year = int(yearly["Year"].iloc[-1])
    return {
        "year":        latest_year + 1,
        "direction":   "BULLISH 📈" if pred == 1 else "BEARISH 📉",
        "bullish_pct": proba[1] * 100,
        "bearish_pct": proba[0] * 100,
        "features":    latest,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MONTHLY PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_monthly_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df.set_index("Date", inplace=True)

    monthly = df.resample("MS").agg(
        Month_Open  = ("Open",  "first"),
        Month_Close = ("Close", "last"),
        Month_High  = ("High",  "max"),
        Month_Low   = ("Low",   "min"),
        Avg_Volume  = ("Volume","mean"),
    ).reset_index().rename(columns={"Date": "Month"})

    monthly.dropna(inplace=True)

    # OHLC features
    monthly["Monthly_Return_%"] = (monthly["Month_Close"] - monthly["Month_Open"]) / monthly["Month_Open"] * 100
    monthly["HL_Range_%"]       = (monthly["Month_High"]  - monthly["Month_Low"])   / monthly["Month_Open"] * 100
    monthly["Body_Ratio"]       = abs(monthly["Month_Close"] - monthly["Month_Open"]) / (monthly["Month_High"] - monthly["Month_Low"])
    monthly["Upper_Shadow_%"]   = (monthly["Month_High"] - monthly[["Month_Open","Month_Close"]].max(axis=1)) / monthly["Month_Open"] * 100
    monthly["Lower_Shadow_%"]   = (monthly[["Month_Open","Month_Close"]].min(axis=1) - monthly["Month_Low"]) / monthly["Month_Open"] * 100
    monthly["Recovery_Rate_%"]  = (monthly["Month_Close"] - monthly["Month_Low"]) / (monthly["Month_High"] - monthly["Month_Low"]) * 100

    # Targets
    monthly["Next_Month_Return_%"]  = monthly["Monthly_Return_%"].shift(-1)
    monthly["Next_Month_Direction"] = (monthly["Next_Month_Return_%"] > 0).astype(int)
    monthly = monthly.dropna(subset=["Next_Month_Return_%"])

    # Lag features
    monthly["Bull_Return_%"]      = monthly["Monthly_Return_%"].clip(lower=0)
    monthly["Bear_Return_%"]      = monthly["Monthly_Return_%"].clip(upper=0)
    monthly["Prev_Bull_Return_%"] = monthly["Bull_Return_%"].shift(1)
    monthly["Prev_Bear_Return_%"] = monthly["Bear_Return_%"].shift(1)
    monthly = monthly.dropna(subset=["Prev_Bull_Return_%"])

    # Rolling CAGR
    monthly["Rolling_CAGR_%"] = monthly["Month_Close"].pct_change(12) * 100
    monthly = monthly.dropna(subset=["Rolling_CAGR_%"])

    return monthly


@st.cache_data(show_spinner=False)
def load_macro_data():
    """Download macro data from yfinance. Returns empty DataFrame on failure."""
    try:
        crude  = yf.download("CL=F",  start="1999-01-01", end="2026-04-01", progress=False)["Close"].resample("MS").last().squeeze()
        gold   = yf.download("GC=F",  start="1999-01-01", end="2026-04-01", progress=False)["Close"].resample("MS").last().squeeze()
        usdinr = yf.download("INR=X", start="1999-01-01", end="2026-04-01", progress=False)["Close"].resample("MS").last().squeeze()

        if crude.empty or gold.empty or usdinr.empty:
            raise ValueError("One or more macro downloads returned empty data")

        macro = pd.DataFrame({"Crude": crude, "Gold": gold, "USDINR": usdinr}).reset_index()
        # Normalize the date column name
        macro.columns = ["Date"] + [c for c in macro.columns[1:]]

        macro["Crude_Change_%"]       = macro["Crude"].pct_change()  * 100
        macro["Gold_Change_%"]        = macro["Gold"].pct_change()   * 100
        macro["USDINR_Change_%"]      = macro["USDINR"].pct_change() * 100
        macro["Prev_Crude_Change_%"]  = macro["Crude_Change_%"].shift(1)
        macro["Prev_Gold_Change_%"]   = macro["Gold_Change_%"].shift(1)
        macro["Prev_USDINR_Change_%"] = macro["USDINR_Change_%"].shift(1)

        return macro

    except Exception as e:
        # Return an empty DataFrame with the right columns so callers can check
        empty_cols = ["Date", "Crude", "Gold", "USDINR",
                      "Crude_Change_%", "Gold_Change_%", "USDINR_Change_%",
                      "Prev_Crude_Change_%", "Prev_Gold_Change_%", "Prev_USDINR_Change_%"]
        return pd.DataFrame(columns=empty_cols)


@st.cache_data(show_spinner=False)
def build_monthly_v2(monthly, macro):
    monthly["Month"] = pd.to_datetime(monthly["Month"])
    macro_date_col = macro.columns[0]
    macro[macro_date_col] = pd.to_datetime(macro[macro_date_col])

    monthly_v2 = monthly.merge(
        macro[[macro_date_col,
               "Crude_Change_%", "Gold_Change_%", "USDINR_Change_%",
               "Prev_Crude_Change_%", "Prev_Gold_Change_%", "Prev_USDINR_Change_%"]],
        left_on="Month", right_on=macro_date_col, how="left"
    )
    if macro_date_col != "Month":
        monthly_v2.drop(columns=[macro_date_col], inplace=True, errors="ignore")

    monthly_v2.dropna(inplace=True)
    return monthly_v2


MONTHLY_FEATURES_REG = [
    "Monthly_Return_%", "HL_Range_%", "Body_Ratio",
    "Upper_Shadow_%", "Lower_Shadow_%", "Recovery_Rate_%",
    "Avg_Volume", "Prev_Bull_Return_%", "Prev_Bear_Return_%",
    "Rolling_CAGR_%",
    "Crude_Change_%", "Gold_Change_%", "USDINR_Change_%",
    "Prev_Crude_Change_%", "Prev_Gold_Change_%", "Prev_USDINR_Change_%"
]

MONTHLY_FEATURES_CLF = [
    "Monthly_Return_%", "HL_Range_%", "Body_Ratio",
    "Upper_Shadow_%", "Lower_Shadow_%", "Recovery_Rate_%",
    "Avg_Volume", "Prev_Bull_Return_%", "Prev_Bear_Return_%",
    "Rolling_CAGR_%"
]


@st.cache_resource(show_spinner=False)
def train_monthly_models():
    monthly  = load_monthly_data()
    macro    = load_macro_data()
    macro_available = len(macro) > 0

    if macro_available:
        monthly_v2 = build_monthly_v2(monthly, macro)
    else:
        # Fallback: use OHLC-only for both regression and classification
        monthly_v2 = None

    # ── Regression ───────────────────────────────────────────────────────────
    if macro_available and monthly_v2 is not None and len(monthly_v2) > 10:
        X_reg = monthly_v2[MONTHLY_FEATURES_REG]
        y_reg = monthly_v2["Next_Month_Return_%"]
        split_reg = int(len(monthly_v2) * 0.8)
        reg_df = monthly_v2
    else:
        # Fallback: OHLC-only regression on full monthly set
        X_reg = monthly[MONTHLY_FEATURES_CLF]
        y_reg = monthly["Next_Month_Return_%"]
        split_reg = int(len(monthly) * 0.8)
        reg_df = monthly
        macro_available = False

    reg_model = XGBRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_reg.iloc[:split_reg], y_reg.iloc[:split_reg])
    reg_preds = reg_model.predict(X_reg.iloc[split_reg:])
    reg_metrics = {
        "mae":  mean_absolute_error(y_reg.iloc[split_reg:], reg_preds),
        "rmse": root_mean_squared_error(y_reg.iloc[split_reg:], reg_preds),
        "n_train": split_reg,
        "n_test":  len(X_reg) - split_reg,
        "train_period": f"{pd.to_datetime(reg_df['Month'].iloc[0]).strftime('%b %Y')}–{pd.to_datetime(reg_df['Month'].iloc[split_reg-1]).strftime('%b %Y')}",
        "test_period":  f"{pd.to_datetime(reg_df['Month'].iloc[split_reg]).strftime('%b %Y')}–{pd.to_datetime(reg_df['Month'].iloc[-1]).strftime('%b %Y')}",
        "macro_used": macro_available,
        "features": MONTHLY_FEATURES_REG if macro_available else MONTHLY_FEATURES_CLF,
    }

    reg_full = XGBRegressor(n_estimators=100, random_state=42)
    reg_full.fit(X_reg, y_reg)

    # ── Classification (Phase 1b — OHLC only) ────────────────────────────────
    if macro_available and monthly_v2 is not None:
        start_date = monthly_v2["Month"].min()
        monthly_1b = monthly[monthly["Month"] >= start_date].copy()
    else:
        monthly_1b = monthly.copy()

    X_clf = monthly_1b[MONTHLY_FEATURES_CLF]
    y_clf = monthly_1b["Next_Month_Direction"]
    split_clf = int(len(monthly_1b) * 0.8)

    clf_model = XGBClassifier(n_estimators=100, random_state=42)
    clf_model.fit(X_clf.iloc[:split_clf], y_clf.iloc[:split_clf])
    clf_preds = clf_model.predict(X_clf.iloc[split_clf:])
    clf_metrics = {
        "accuracy": accuracy_score(y_clf.iloc[split_clf:], clf_preds),
        "f1":       f1_score(y_clf.iloc[split_clf:], clf_preds, zero_division=0),
        "cm":       confusion_matrix(y_clf.iloc[split_clf:], clf_preds),
        "n_train": split_clf,
        "n_test":  len(monthly_1b) - split_clf,
        "train_period": f"{pd.to_datetime(monthly_1b['Month'].iloc[0]).strftime('%b %Y')}–{pd.to_datetime(monthly_1b['Month'].iloc[split_clf-1]).strftime('%b %Y')}",
        "test_period":  f"{pd.to_datetime(monthly_1b['Month'].iloc[split_clf]).strftime('%b %Y')}–{pd.to_datetime(monthly_1b['Month'].iloc[-1]).strftime('%b %Y')}",
    }

    clf_full = XGBClassifier(n_estimators=100, random_state=42)
    clf_full.fit(X_clf, y_clf)

    return reg_full, clf_full, reg_metrics, clf_metrics, reg_df, monthly_1b, macro
