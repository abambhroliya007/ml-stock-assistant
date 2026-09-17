import os
import pandas as pd


def load_prices():
    parquet_path = os.path.join("data", "prices.parquet")
    csv_path = os.path.join("data", "prices.csv")

    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)

    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

    else:
        raise FileNotFoundError(
            "No stock dataset found. Expected data/prices.parquet "
            "or data/prices.csv."
        )

    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = df.dropna(subset=["date", "ticker", "close"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    return df