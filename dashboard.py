# dashboard.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from pathlib import Path
import warnings

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except Exception:
    HAS_ARIMA = False


# -----------------------------
# Helper functions
# -----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    return df

def build_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(
            df["year"].astype(int).astype(str) + "-" + df["month"].astype(int).astype(str) + "-01",
            errors="coerce",
        )
    return df

def fit_forecast(series: pd.Series, steps: int = 3, use_arima_min: int = 12, use_lr_min: int = 6) -> np.ndarray:
    y = series.dropna().astype(float)
    n = len(y)
    if n == 0:
        return np.array([np.nan] * steps)

    # ARIMA
    if HAS_ARIMA and n >= use_arima_min:
        try:
            model = ARIMA(y, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = model.fit(method_kwargs={"maxiter": 50}, disp=0)
            return np.asarray(res.forecast(steps=steps))
        except Exception:
            pass

    # Linear Regression
    if n >= use_lr_min:
        X = np.arange(n).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(X, y.values)
        Xf = np.arange(n, n + steps).reshape(-1, 1)
        return lr.predict(Xf)

    # Naive
    return np.array([y.iloc[-1]] * steps)


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Dealership KPI Dashboard", layout="wide")
st.title("📊 Dealership KPI Dashboard")

uploaded_file = st.file_uploader("Upload KPI CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = normalize_columns(df)

    required = {"account_id", "english_name", "year", "month", "monthly_value"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
        st.stop()

    df = build_date(df)
    df = df.sort_values(["account_id", "date"]).reset_index(drop=True)

    # ---------------- Forecasts ----------------
    st.header("🔮 KPI Forecasts")
    preds = []
    for kpi_id, g in df.groupby("account_id"):
        g = g.set_index("date").asfreq("MS")
        english_name = g["english_name"].dropna().iloc[0] if g["english_name"].notna().any() else str(kpi_id)
        y = g["monthly_value"].astype(float).ffill()

        if y.dropna().shape[0] < 6:
            continue

        fc = fit_forecast(y, steps=3)
        last_date = y.index.max()
        future_idx = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=3, freq="MS")
        preds.append(pd.DataFrame({
            "account_id": kpi_id,
            "english_name": english_name,
            "date": future_idx,
            "predicted_value": fc
        }))

    if preds:
        predictions_df = pd.concat(preds, ignore_index=True)
        kpi_selected = st.selectbox("Select KPI to view forecast:", predictions_df["english_name"].unique())
        kpi_data = predictions_df[predictions_df["english_name"] == kpi_selected]

        fig, ax = plt.subplots(figsize=(10,5))
        actual = df[df["english_name"] == kpi_selected][["date","monthly_value"]].sort_values("date")
        ax.plot(actual["date"], actual["monthly_value"], marker="o", label="Actual")
        ax.plot(kpi_data["date"], kpi_data["predicted_value"], marker="x", label="Forecast")
        ax.set_title(f"{kpi_selected} — 3-Month Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        st.pyplot(fig)
        st.dataframe(kpi_data)
    else:
        st.info("Not enough data to generate forecasts.")

    # ---------------- Correlations ----------------
    st.header("📈 KPI Correlation Matrix")
    pivot_df = (
        df.pivot_table(index="date", columns="english_name", values="monthly_value", aggfunc="mean")
          .sort_index().asfreq("MS").ffill()
    )
    corr = pivot_df.corr()
    st.dataframe(corr.round(2))

    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(corr.values, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title("KPI Correlation Heatmap")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)

    # ---------------- What-If ----------------
    st.header("🤔 What-If Scenario")
    kpi_choice = st.selectbox("Select KPI to shock:", list(pivot_df.columns))
    pct_change = st.slider("Percentage change to apply (%)", -50, 50, 10)
    start_date = st.date_input("Start date for change", value=pivot_df.index.min().date())

    if st.button("Apply What-If Scenario"):
        scenario = pivot_df.copy()
        scenario.loc[scenario.index >= pd.to_datetime(start_date), kpi_choice] *= (1 + pct_change / 100.0)

        st.line_chart(scenario)
        st.dataframe(scenario.tail())

else:
    st.info("Please upload a KPI CSV file to get started.")
