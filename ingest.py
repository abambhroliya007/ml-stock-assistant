import pandas as pd

def load_prices(csv_path: str = "data/prices.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"date", "ticker", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"prices.csv missing columns: {missing}. Required: date,ticker,close")

    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "close"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    return df
