import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


@dataclass
class ForecastResult:
    df: pd.DataFrame  # columns: date, yhat, yhat_lower, yhat_upper, type(history/forecast)


def linear_regression_forecast(df_ticker: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    """
    Baseline: Linear regression on time index.
    Returns a dataframe with columns: date, close, type (history/forecast)
    """
    df = df_ticker.sort_values("date").copy()
    df["t"] = np.arange(len(df))
    X = df[["t"]].values
    y = df["close"].astype(float).values

    model = LinearRegression()
    model.fit(X, y)

    last_t = df["t"].iloc[-1]
    future_t = np.arange(last_t + 1, last_t + 1 + days)

    future_dates = pd.bdate_range(df["date"].iloc[-1] + pd.Timedelta(days=1), periods=days)
    y_pred = model.predict(future_t.reshape(-1, 1))

    hist = df[["date", "close"]].copy()
    hist["type"] = "history"
    hist = hist.rename(columns={"close": "close"})

    fut = pd.DataFrame({"date": future_dates, "close": y_pred, "type": "forecast"})
    return pd.concat([hist, fut], ignore_index=True)


def arima_forecast(df_ticker: pd.DataFrame, days: int = 30) -> ForecastResult:
    """
    ARIMA forecast with confidence intervals using statsmodels.
    """
    from statsmodels.tsa.arima.model import ARIMA

    df = df_ticker.sort_values("date").copy()
    y = df["close"].astype(float).values

    # Basic sanity
    if len(y) < 50:
        raise ValueError("Not enough data for ARIMA (need at least ~50 points).")

    # Silence warnings
    warnings.filterwarnings("ignore")

    # A decent default ARIMA order; can be tuned later
    model = ARIMA(y, order=(1, 1, 1))
    fitted = model.fit()

    fc = fitted.get_forecast(steps=days)
    mean = fc.predicted_mean
    conf = fc.conf_int(alpha=0.05)  # 95% interval
    lower = conf[:, 0]
    upper = conf[:, 1]

    # Future business dates
    future_dates = pd.bdate_range(df["date"].iloc[-1] + pd.Timedelta(days=1), periods=days)

    hist = pd.DataFrame({
        "date": df["date"].values,
        "yhat": df["close"].astype(float).values,
        "yhat_lower": np.nan,
        "yhat_upper": np.nan,
        "type": "history"
    })
    fut = pd.DataFrame({
        "date": future_dates,
        "yhat": mean,
        "yhat_lower": lower,
        "yhat_upper": upper,
        "type": "forecast"
    })
    return ForecastResult(df=pd.concat([hist, fut], ignore_index=True))


def prophet_forecast(df_ticker: pd.DataFrame, days: int = 30) -> ForecastResult:
    """
    Prophet forecast with confidence intervals.
    If prophet is not installed, this will raise ImportError.
    """
    from prophet import Prophet

    df = df_ticker.sort_values("date").copy()
    if len(df) < 50:
        raise ValueError("Not enough data for Prophet (need at least ~50 points).")

    # Prophet expects ds/y
    p = pd.DataFrame({"ds": df["date"], "y": df["close"].astype(float)})
    m = Prophet(interval_width=0.95, daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(p)

    future = m.make_future_dataframe(periods=days, freq="B")
    fc = m.predict(future)

    # history portion
    hist_len = len(df)
    hist = pd.DataFrame({
        "date": fc["ds"].iloc[:hist_len].values,
        "yhat": fc["yhat"].iloc[:hist_len].values,
        "yhat_lower": np.nan,
        "yhat_upper": np.nan,
        "type": "history"
    })

    fut = pd.DataFrame({
        "date": fc["ds"].iloc[hist_len:].values,
        "yhat": fc["yhat"].iloc[hist_len:].values,
        "yhat_lower": fc["yhat_lower"].iloc[hist_len:].values,
        "yhat_upper": fc["yhat_upper"].iloc[hist_len:].values,
        "type": "forecast"
    })

    return ForecastResult(df=pd.concat([hist, fut], ignore_index=True))


def forecast_with_model(df_ticker: pd.DataFrame, days: int, model_name: str) -> Dict[str, Any]:
    """
    model_name: 'arima' | 'prophet' | 'linreg'
    Returns dict with:
      - model_used
      - df_forecast (ForecastResult.df or legacy df)
      - has_bands
    """
    model_name = (model_name or "arima").lower().strip()

    if model_name == "prophet":
        try:
            res = prophet_forecast(df_ticker, days=days)
            return {"model_used": "prophet", "df_forecast": res.df, "has_bands": True}
        except Exception:
            # fallback to ARIMA if Prophet fails for any reason
            res = arima_forecast(df_ticker, days=days)
            return {"model_used": "arima", "df_forecast": res.df, "has_bands": True}

    if model_name == "linreg":
        df_lr = linear_regression_forecast(df_ticker, days=days)
        # reshape to unified format
        df_lr2 = df_lr.rename(columns={"close": "yhat"})
        df_lr2["yhat_lower"] = np.nan
        df_lr2["yhat_upper"] = np.nan
        return {"model_used": "linreg", "df_forecast": df_lr2[["date", "yhat", "yhat_lower", "yhat_upper", "type"]], "has_bands": False}

    # default: arima
    res = arima_forecast(df_ticker, days=days)
    return {"model_used": "arima", "df_forecast": res.df, "has_bands": True}
