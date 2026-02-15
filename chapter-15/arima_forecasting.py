"""
Chapter 15: Time Series Analysis
Section 15.3 — Forecasting Models: ARIMA, SARIMA, and Prophet

Demonstrates time series forecasting:
- ARIMA model for trend-based data
- SARIMA model for seasonal data
- Model diagnostics and evaluation
- Forecast visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error


def create_sales_data():
    """Generate synthetic monthly sales data with trend and seasonality."""
    np.random.seed(42)
    n_months = 48  # 4 years

    dates = pd.date_range("2021-01-01", periods=n_months, freq="MS")

    # Components
    trend = np.linspace(100, 200, n_months)
    seasonal = 30 * np.sin(2 * np.pi * np.arange(n_months) / 12)
    noise = np.random.normal(0, 8, n_months)

    sales = trend + seasonal + noise

    ts = pd.Series(sales.round(1), index=dates, name="monthly_sales")
    return ts


def decomposition_demo(ts):
    """Decompose time series into trend, seasonal, and residual components."""
    print("=" * 60)
    print("STEP 1: Time Series Decomposition")
    print("=" * 60)

    result = seasonal_decompose(ts, model="additive", period=12)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    result.observed.plot(ax=axes[0], color="#2B579A", linewidth=1.5)
    axes[0].set_title("Observed", fontweight="bold")
    axes[0].set_ylabel("Sales")

    result.trend.plot(ax=axes[1], color="#2E8B57", linewidth=1.5)
    axes[1].set_title("Trend", fontweight="bold")
    axes[1].set_ylabel("Sales")

    result.seasonal.plot(ax=axes[2], color="#FF8C00", linewidth=1.5)
    axes[2].set_title("Seasonal", fontweight="bold")
    axes[2].set_ylabel("Sales")

    result.resid.plot(ax=axes[3], color="#C00000", linewidth=1.5)
    axes[3].set_title("Residual", fontweight="bold")
    axes[3].set_ylabel("Sales")

    plt.suptitle("Seasonal Decomposition of Monthly Sales", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("decomposition.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved: decomposition.png")
    print(f"\n  Observations: {len(ts)} months")
    print(f"  Date range: {ts.index[0].strftime('%Y-%m')} to {ts.index[-1].strftime('%Y-%m')}")
    print(f"  Mean sales: {ts.mean():.1f}")
    print(f"  Clear upward trend with 12-month seasonal cycle")


def arima_forecast(ts):
    """Fit an ARIMA model and generate forecasts."""
    print("\n\n" + "=" * 60)
    print("STEP 2: ARIMA Forecasting")
    print("=" * 60)

    # Train/test split: last 6 months as test
    train = ts[:-6]
    test = ts[-6:]

    print(f"\n  Train period: {train.index[0].strftime('%Y-%m')} to {train.index[-1].strftime('%Y-%m')} ({len(train)} months)")
    print(f"  Test period:  {test.index[0].strftime('%Y-%m')} to {test.index[-1].strftime('%Y-%m')} ({len(test)} months)")

    # Fit ARIMA(1,1,1) — simple differencing model
    model = ARIMA(train, order=(1, 1, 1))
    fitted = model.fit()

    print(f"\n  ARIMA(1,1,1) Summary:")
    print(f"    AIC: {fitted.aic:.2f}")
    print(f"    BIC: {fitted.bic:.2f}")

    # Forecast
    forecast = fitted.forecast(steps=6)

    # Evaluate
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mape = np.mean(np.abs((test - forecast) / test)) * 100

    print(f"\n  ARIMA Forecast Evaluation:")
    print(f"    MAE:  {mae:.2f}")
    print(f"    RMSE: {rmse:.2f}")
    print(f"    MAPE: {mape:.1f}%")

    return train, test, forecast, mae


def sarima_forecast(ts):
    """Fit a SARIMA model that captures seasonality."""
    print("\n\n" + "=" * 60)
    print("STEP 3: SARIMA Forecasting (with Seasonality)")
    print("=" * 60)

    train = ts[:-6]
    test = ts[-6:]

    # Fit SARIMA(1,1,1)(1,1,1,12)
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    fitted = model.fit(disp=False)

    print(f"\n  SARIMA(1,1,1)(1,1,1,12) Summary:")
    print(f"    AIC: {fitted.aic:.2f}")
    print(f"    BIC: {fitted.bic:.2f}")

    # Forecast
    forecast = fitted.forecast(steps=6)

    # Evaluate
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mape = np.mean(np.abs((test - forecast) / test)) * 100

    print(f"\n  SARIMA Forecast Evaluation:")
    print(f"    MAE:  {mae:.2f}")
    print(f"    RMSE: {rmse:.2f}")
    print(f"    MAPE: {mape:.1f}%")

    return forecast, mae


def visualize_forecasts(ts, train, test, arima_forecast_vals, sarima_forecast_vals):
    """Compare ARIMA and SARIMA forecasts visually."""
    print("\n\n" + "=" * 60)
    print("STEP 4: Forecast Comparison Visualization")
    print("=" * 60)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    # Panel 1: ARIMA
    axes[0].plot(train.index, train, "-", color="#2B579A", linewidth=1.5, label="Training Data")
    axes[0].plot(test.index, test, "o-", color="#2E8B57", linewidth=2, markersize=6, label="Actual (Test)")
    axes[0].plot(test.index, arima_forecast_vals, "s--", color="#C00000", linewidth=2, markersize=6, label="ARIMA Forecast")
    axes[0].set_title("ARIMA(1,1,1) Forecast", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Monthly Sales")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=test.index[0], color="gray", linestyle=":", alpha=0.5)
    axes[0].text(test.index[0], axes[0].get_ylim()[1] * 0.95, " ← Forecast Start",
                fontsize=9, color="gray")

    # Panel 2: SARIMA
    axes[1].plot(train.index, train, "-", color="#2B579A", linewidth=1.5, label="Training Data")
    axes[1].plot(test.index, test, "o-", color="#2E8B57", linewidth=2, markersize=6, label="Actual (Test)")
    axes[1].plot(test.index, sarima_forecast_vals, "s--", color="#FF8C00", linewidth=2, markersize=6, label="SARIMA Forecast")
    axes[1].set_title("SARIMA(1,1,1)(1,1,1,12) Forecast", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Monthly Sales")
    axes[1].set_xlabel("Date")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=test.index[0], color="gray", linestyle=":", alpha=0.5)

    plt.suptitle("Time Series Forecasting: ARIMA vs SARIMA", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("forecast_comparison.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved: forecast_comparison.png")


if __name__ == "__main__":
    # Generate data
    ts = create_sales_data()

    # Step 1: Decompose
    decomposition_demo(ts)

    # Step 2: ARIMA
    train, test, arima_fc, arima_mae = arima_forecast(ts)

    # Step 3: SARIMA
    sarima_fc, sarima_mae = sarima_forecast(ts)

    # Step 4: Compare visually
    visualize_forecasts(ts, train, test, arima_fc, sarima_fc)

    # Final comparison
    print("\n\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n  {'Model':<25} {'MAE':>8}")
    print("  " + "-" * 35)
    print(f"  {'ARIMA(1,1,1)':<25} {arima_mae:>8.2f}")
    print(f"  {'SARIMA(1,1,1)(1,1,1,12)':<25} {sarima_mae:>8.2f}")
    winner = "SARIMA" if sarima_mae < arima_mae else "ARIMA"
    print(f"\n  → {winner} performs better on this seasonal dataset.")
    print(f"    SARIMA captures the 12-month seasonal pattern that ARIMA misses.")
