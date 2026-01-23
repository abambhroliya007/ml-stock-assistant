import numpy as np
import pandas as pd

TRADING_DAYS = 252


def compute_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


def total_return(prices: pd.Series) -> float:
    if len(prices) < 2:
        return float("nan")
    return float(prices.iloc[-1] / prices.iloc[0] - 1.0)


def cagr(prices: pd.Series, dates: pd.Series) -> float:
    if len(prices) < 2:
        return float("nan")
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float((prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1)


def volatility_daily(prices: pd.Series) -> float:
    r = compute_returns(prices)
    if r.empty:
        return float("nan")
    return float(r.std())


def volatility_annual(prices: pd.Series) -> float:
    vol_d = volatility_daily(prices)
    if np.isnan(vol_d):
        return float("nan")
    return float(vol_d * np.sqrt(TRADING_DAYS))


def sharpe_ratio(prices: pd.Series, risk_free_rate: float = 0.0) -> float:
    r = compute_returns(prices)
    if r.empty:
        return float("nan")
    excess = r - (risk_free_rate / TRADING_DAYS)
    denom = excess.std()
    if denom == 0 or np.isnan(denom):
        return float("nan")
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / denom)


def max_drawdown(prices: pd.Series) -> float:
    if len(prices) < 2:
        return float("nan")
    cummax = prices.cummax()
    drawdown = prices / cummax - 1.0
    return float(drawdown.min())


def max_drawdown_duration(prices: pd.Series) -> int:
    """
    Longest consecutive run of days while below the previous peak.
    Returns number of days.
    """
    if len(prices) < 2:
        return 0
    cummax = prices.cummax()
    underwater = prices < cummax
    # count consecutive True runs
    durations = underwater.groupby((~underwater).cumsum()).cumcount()
    if durations.empty:
        return 0
    return int(durations.max())


def trailing_return(prices: pd.Series, dates: pd.Series, years: int) -> float:
    if len(prices) < 2:
        return float("nan")
    cutoff = dates.iloc[-1] - pd.DateOffset(years=years)
    mask = dates >= cutoff
    sub = prices[mask]
    if len(sub) < 2:
        return float("nan")
    return float(sub.iloc[-1] / sub.iloc[0] - 1.0)


def risk_label(annual_vol: float, mdd: float) -> str:
    if np.isnan(annual_vol) or np.isnan(mdd):
        return "UNKNOWN"
    if annual_vol < 0.20 and mdd > -0.25:
        return "LOW"
    if annual_vol < 0.35 and mdd > -0.45:
        return "MED"
    return "HIGH"


def summary_metrics(df_ticker: pd.DataFrame) -> dict:
    """
    df_ticker must have columns: date (datetime), close (float)
    """
    prices = df_ticker["close"].astype(float).reset_index(drop=True)
    dates = pd.to_datetime(df_ticker["date"]).reset_index(drop=True)

    tr = total_return(prices)
    cg = cagr(prices, dates)
    vol_d = volatility_daily(prices)
    vol_a = volatility_annual(prices)
    shp = sharpe_ratio(prices, risk_free_rate=0.0)
    mdd = max_drawdown(prices)
    mdd_dur = max_drawdown_duration(prices)

    r1 = trailing_return(prices, dates, 1)
    r3 = trailing_return(prices, dates, 3)
    r5 = trailing_return(prices, dates, 5)

    return {
        "total_return": tr,
        "cagr": cg,
        "vol_daily": vol_d,
        "vol_annual": vol_a,
        "sharpe": shp,
        "max_drawdown": mdd,
        "max_dd_duration_days": mdd_dur,
        "ret_1y": r1,
        "ret_3y": r3,
        "ret_5y": r5,
        "risk": risk_label(vol_a, mdd),
    }
