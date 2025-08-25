#!/usr/bin/env python3
"""
Dealership KPI Forecaster + Correlations + What-If Scenarios

Usage (from the folder containing your KPI CSV):
    python index.py --csv sample_data.csv \
        --min-months 6 \
        --whatif-kpi "Sales" \
        --whatif-change 10 \
        --whatif-start 2025-03-01

Outputs (in ./outputs):
  - forecast_results.csv        -> 3-month forecasts per KPI (account_id)
  - correlation_matrix.csv      -> Pearson correlations between KPIs (english_name)
  - what_if_results.csv         -> Scenario after applying the what-if (if provided)
  - plots/*.png                 -> Matplotlib charts saved as images
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Try to import ARIMA; if unavailable, fallback
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except Exception:
    HAS_ARIMA = False

# -----------------------------
# Helpers
# -----------------------------

def ensure_output_dirs(base: Path):
    (base / "plots").mkdir(parents=True, exist_ok=True)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    )
    return df

def build_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(
            df["year"].astype(int).astype(str) + "-" + df["month"].astype(int).astype(str) + "-01",
            errors="coerce",
        )
    return df

def recompute_yearly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "yearly_value" not in df.columns or df["yearly_value"].isna().all():
        df["yearly_value"] = (  
            df.sort_values(["account_id", "date"])  
              .groupby(["account_id", "year"]) ["monthly_value"].cumsum()
        )
    return df

def fit_forecast(series: pd.Series, steps: int = 3, use_arima_min: int = 12, use_lr_min: int = 6) -> np.ndarray:
    """Return next `steps` forecasts with tiered strategy."""
    y = series.dropna().astype(float)
    n = len(y)
    if n == 0:
        return np.array([np.nan] * steps)

    # Option 1: ARIMA
    if HAS_ARIMA and n >= use_arima_min:
        try:
            model = ARIMA(y, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = model.fit(method_kwargs={"maxiter": 50}, disp=0)
            return np.asarray(res.forecast(steps=steps))
        except Exception:
            pass

    # Option 2: Linear Regression
    if n >= use_lr_min:
        X = np.arange(n).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(X, y.values)
        Xf = np.arange(n, n + steps).reshape(-1, 1)
        return lr.predict(Xf)

    # Option 3: Naive
    last = y.iloc[-1]
    return np.array([last] * steps)

def plot_heatmap(corr: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(corr.values, aspect='auto')
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title("KPI Correlation Heatmap")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_actual_vs_forecast(actual_df: pd.DataFrame, forecast_df: pd.DataFrame, kpi_name: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(actual_df["date"], actual_df["monthly_value"], marker='o', label='Actual')
    ax.plot(forecast_df["date"], forecast_df["predicted_value"], marker='x', label='Forecast')
    ax.set_title(f"{kpi_name} — 3-Month Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def build_what_if(pivot_df: pd.DataFrame, corr: pd.DataFrame, kpi_name: str, change_percent: float, start_date: str, damping: float = 1.0) -> pd.DataFrame:
    """Apply a what-if multiplicative change to kpi_name and propagate using correlations."""
    scenario = pivot_df.copy()
    start_date = pd.to_datetime(start_date)
    if kpi_name not in scenario.columns:
        raise ValueError(f"KPI '{kpi_name}' not found. Available: {list(scenario.columns)}")

    # Apply base change
    scenario.loc[scenario.index >= start_date, kpi_name] *= (1 + change_percent / 100.0)

    # Propagate via correlations
    col = corr[kpi_name].fillna(0).clip(-1, 1) * float(damping)
    for other_kpi, c in col.items():
        if other_kpi == kpi_name:
            continue
        scenario.loc[scenario.index >= start_date, other_kpi] *= (1 + c * change_percent / 100.0)
    return scenario

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Dealership KPI Forecaster + Correlations + What-If")
    parser.add_argument('--csv', type=str, default='sample_data.csv', help='Path to KPI CSV file')
    parser.add_argument('--min-months', type=int, default=6, help='Minimum months for forecasting model selection')
    parser.add_argument('--arima-min', type=int, default=12, help='Minimum months to attempt ARIMA')
    parser.add_argument('--whatif-kpi', type=str, default=None, help='KPI english_name to shock')
    parser.add_argument('--whatif-change', type=float, default=0.0, help='% change to apply to the KPI')
    parser.add_argument('--whatif-start', type=str, default=None, help='Start date for scenario (YYYY-MM-DD)')
    parser.add_argument('--damping', type=float, default=1.0, help='Damp correlation propagation (0..1)')
    args = parser.parse_args()

    out_dir = Path('outputs')
    ensure_output_dirs(out_dir)

    # Load --------------------------------------------------
    df = pd.read_csv(args.csv)
    df = normalize_columns(df)

    required = {"account_id", "english_name", "year", "month", "monthly_value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = build_date(df)
    df = df.sort_values(["account_id", "date"]).reset_index(drop=True)
    df = recompute_yearly(df)

    # Forecasting ------------------------------------------
    preds = []
    for kpi_id, g in df.groupby("account_id"):
        g = g.set_index("date").asfreq("MS")
        english_name = g["english_name"].dropna().iloc[0] if g["english_name"].notna().any() else str(kpi_id)

        y = g["monthly_value"].astype(float).ffill()
        if y.dropna().shape[0] < args.min_months:
            print(f"Skipping {kpi_id} ({english_name}) — not enough data (< {args.min_months})")
            continue

        fc = fit_forecast(y, steps=3, use_arima_min=args.arima_min, use_lr_min=args.min_months)
        last_date = y.index.max()
        future_idx = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=3, freq='MS')
        preds.append(pd.DataFrame({
            'account_id': kpi_id,
            'english_name': english_name,
            'date': future_idx,
            'predicted_value': fc
        }))

    if preds:
        predictions_df = pd.concat(preds, ignore_index=True)
        predictions_df.to_csv(out_dir / 'forecast_results.csv', index=False)
        print(f"Saved forecasts -> {out_dir / 'forecast_results.csv'}")
    else:
        predictions_df = pd.DataFrame(columns=['account_id','english_name','date','predicted_value'])
        print("No forecasts generated (insufficient history).")

    # Correlations -----------------------------------------
    pivot_df = (
        df.pivot_table(index='date', columns='english_name', values='monthly_value', aggfunc='mean')
          .sort_index()
          .asfreq('MS')
          .ffill()
    )
    corr = pivot_df.corr()
    corr.to_csv(out_dir / 'correlation_matrix.csv')
    print(f"Saved correlations -> {out_dir / 'correlation_matrix.csv'}")
    plot_heatmap(corr, out_dir / 'plots' / 'correlation_heatmap.png')

    # Forecast plots
    if not predictions_df.empty:
        for name, g in predictions_df.groupby('english_name'):
            actual = df[df['english_name'] == name][['date','monthly_value']].sort_values('date')
            forecast = g[['date','predicted_value']].sort_values('date')
            plot_actual_vs_forecast(actual, forecast, name, out_dir / 'plots' / f"forecast_{name.replace(' ','_')}.png")

    # What-If ----------------------------------------------
    if args.whatif_kpi and args.whatif_start and abs(args.whatif_change) > 0:
        scenario = build_what_if(
            pivot_df=pivot_df,
            corr=corr,
            kpi_name=args.whatif_kpi,
            change_percent=args.whatif_change,
            start_date=args.whatif_start,
            damping=args.damping,
        )
        scenario.to_csv(out_dir / 'what_if_results.csv')
        print(f"Saved what-if results -> {out_dir / 'what_if_results.csv'}")

        fig, ax = plt.subplots(figsize=(10,5))
        for col in [args.whatif_kpi] + [c for c in ["Profit", "Sales", "Revenue"] if c in scenario.columns]:
            ax.plot(scenario.index, scenario[col], label=col)
        ax.set_title(f"What-If: {args.whatif_kpi} {args.whatif_change:+.1f}% from {args.whatif_start}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / 'plots' / 'what_if.png', dpi=150)
        plt.close(fig)

    print("\n✅ All done. See the ./outputs folder for CSVs and plots.")

if __name__ == '__main__':
    main()
    