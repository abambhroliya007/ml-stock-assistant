import os
import time
from datetime import datetime
from io import StringIO

import pandas as pd
import requests
import yfinance as yf

OUT_PATH = os.path.join("data", "prices.csv")

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def get_sp500_tickers() -> list[str]:
    """
    Pull current S&P 500 constituents from Wikipedia.

    Wikipedia may block default Python user agents (HTTP 403),
    so we fetch HTML with requests + a browser-like User-Agent,
    then parse the HTML using pandas.read_html().

    Returns tickers in Yahoo Finance format (dots converted to dashes).
    Example: BRK.B -> BRK-B, BF.B -> BF-B
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    r = requests.get(WIKI_SP500_URL, headers=headers, timeout=30)
    r.raise_for_status()

    tables = pd.read_html(StringIO(r.text))
    df = tables[0]  # constituents table

    tickers = df["Symbol"].astype(str).str.strip().str.upper().tolist()
    tickers = [t.replace(".", "-") for t in tickers]  # Yahoo uses '-' instead of '.'
    tickers = sorted(set(tickers))
    return tickers

def download_batch(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted close for a batch of tickers using yfinance.
    auto_adjust=True gives split/dividend adjusted prices.
    Output is normalized to: date,ticker,close
    """
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    rows = []

    # Multiple tickers -> MultiIndex columns like (TICKER, Close)
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t not in data.columns.get_level_values(0):
                continue
            if (t, "Close") not in data.columns:
                continue
            s = data[(t, "Close")].dropna()
            for dt, val in s.items():
                rows.append({"date": dt.date().isoformat(), "ticker": t, "close": float(val)})

    # Single ticker -> normal columns like Close, Open...
    else:
        if "Close" in data.columns and len(tickers) == 1:
            t = tickers[0]
            s = data["Close"].dropna()
            for dt, val in s.items():
                rows.append({"date": dt.date().isoformat(), "ticker": t, "close": float(val)})

    return pd.DataFrame(rows)

def main():
    os.makedirs("data", exist_ok=True)

    # Long historical range (change if you want)
    START = "2000-01-01"
    END = datetime.today().date().isoformat()

    tickers = get_sp500_tickers()
    print(f"Found {len(tickers)} S&P 500 tickers (current list).")

    # Download in batches to reduce failures / throttling
    BATCH_SIZE = 50
    all_parts = []

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        print(f"Downloading {i+1}-{i+len(batch)} / {len(tickers)} ... ({batch[0]} .. {batch[-1]})")

        try:
            df_part = download_batch(batch, START, END)
            all_parts.append(df_part)
            # Polite pause reduces the chance of throttling
            time.sleep(1.2)
        except Exception as e:
            print(f"Batch failed ({batch[0]}..{batch[-1]}): {e}")
            time.sleep(3)

    if not all_parts:
        raise RuntimeError("No data downloaded. Check internet access or try smaller batches.")

    df = pd.concat(all_parts, ignore_index=True)

    # Clean + sort
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "close"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    df.to_csv(OUT_PATH, index=False)
    size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)

    print(f"\nWrote {OUT_PATH}")
    print(f"Rows: {len(df):,} | Tickers: {df['ticker'].nunique():,} | Size: {size_mb:.1f} MB")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")

if __name__ == "__main__":
    main()
