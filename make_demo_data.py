import os
import numpy as np
import pandas as pd

OUT_PATH = os.path.join("data", "prices.csv")

def generate_prices(
    tickers: list[str],
    start: str = "2005-01-03",
    end: str = "2025-01-03",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates long-format OHLC-lite data: date,ticker,close
    - Business days only
    - Market factor + per-stock beta + idiosyncratic noise
    - Different vol levels across tickers
    """
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(start=start, end=end)  # business days
    n_days = len(dates)
    n_tickers = len(tickers)

    # Market returns (daily)
    # Slight positive drift with moderate volatility
    market_ret = rng.normal(loc=0.00025, scale=0.010, size=n_days)

    # Per-ticker parameters
    betas = rng.uniform(0.7, 1.4, size=n_tickers)             # sensitivity to market
    idio_vol = rng.uniform(0.006, 0.025, size=n_tickers)      # idiosyncratic vol
    drift = rng.uniform(0.00005, 0.00035, size=n_tickers)     # small ticker-specific drift
    start_prices = rng.uniform(10, 600, size=n_tickers)       # starting prices

    # Generate returns matrix: (days, tickers)
    # r_ticker = drift + beta*market + idio_noise
    idio_noise = rng.normal(loc=0.0, scale=1.0, size=(n_days, n_tickers)) * idio_vol
    rets = drift + (market_ret[:, None] * betas[None, :]) + idio_noise

    # Convert returns to prices
    prices = np.zeros((n_days, n_tickers), dtype=np.float64)
    prices[0, :] = start_prices
    for i in range(1, n_days):
        prices[i, :] = prices[i - 1, :] * (1.0 + rets[i, :])

    # Ensure no non-positive prices
    prices = np.clip(prices, 0.5, None)

    # Build long DataFrame
    df = pd.DataFrame(prices, index=dates, columns=tickers)
    out = df.reset_index().melt(id_vars="index", var_name="ticker", value_name="close")
    out = out.rename(columns={"index": "date"})
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["close"] = out["close"].round(2)
    return out

def make_tickers(n: int) -> list[str]:
    """
    Create synthetic tickers like STK001, STK002, ...
    """
    return [f"STK{i:03d}" for i in range(1, n + 1)]

def main():
    os.makedirs("data", exist_ok=True)

    # --- CONTROL SIZE HERE ---
    N_TICKERS = 300       # try 300 first; then 500/800 if your machine handles it
    START_DATE = "2005-01-03"
    END_DATE = "2025-01-03"

    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"] + make_tickers(N_TICKERS)
    df = generate_prices(tickers, start=START_DATE, end=END_DATE, seed=42)

    df.to_csv(OUT_PATH, index=False)
    size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"Wrote: {OUT_PATH}")
    print(f"Rows: {len(df):,} | Tickers: {df['ticker'].nunique():,} | Approx size: {size_mb:.1f} MB")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")

if __name__ == "__main__":
    main()
