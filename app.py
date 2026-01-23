from ingest import load_prices
from analytics import summary_metrics
from ml_models import linear_regression_forecast

STATE = {"df": None}

def cmd_load():
    STATE["df"] = load_prices()
    print(f"Loaded {len(STATE['df']):,} rows. Tickers: {STATE['df']['ticker'].nunique()}")

def cmd_summary(ticker: str):
    df = STATE["df"]
    if df is None:
        print("Run /load first.")
        return
    t = ticker.upper()
    sub = df[df["ticker"] == t]
    if sub.empty:
        print(f"No data for ticker {t}")
        return
    metrics = summary_metrics(sub["close"])
    print(f"Summary {t}:")
    print(metrics)

def cmd_forecast(ticker: str, days: int):
    df = STATE["df"]
    if df is None:
        print("Run /load first.")
        return
    t = ticker.upper()
    sub = df[df["ticker"] == t]
    if sub.empty:
        print(f"No data for ticker {t}")
        return
    out = linear_regression_forecast(sub, days=days)
    print(out.tail(10))

def main():
    print("ML Stock Assistant CLI")
    print("Commands: /load, summary TICKER, forecast TICKER DAYS, quit")
    while True:
        s = input("> ").strip()
        if not s:
            continue
        if s in {"quit", "exit"}:
            break
        if s == "/load":
            cmd_load()
            continue
        parts = s.split()
        if parts[0] == "summary" and len(parts) == 2:
            cmd_summary(parts[1])
        elif parts[0] == "forecast" and len(parts) == 3:
            cmd_forecast(parts[1], int(parts[2]))
        else:
            print("Unknown command")

if __name__ == "__main__":
    main()
