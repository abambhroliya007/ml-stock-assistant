from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request

from ingest import load_prices
from analytics import summary_metrics
from ml_models import linear_regression_forecast

app = Flask(__name__)

# In-memory state for this serverless instance (not shared across instances)
STATE: Dict[str, Any] = {"df": None}
STATE_LOCK = threading.Lock()


def _ensure_loaded():
    """Load prices into memory if not already loaded."""
    with STATE_LOCK:
        if STATE["df"] is None:
            STATE["df"] = load_prices()
        return STATE["df"]


def _get_ticker_df(ticker: str):
    df = _ensure_loaded()
    t = ticker.upper().strip()
    sub = df[df["ticker"] == t]
    return t, sub


@app.get("/api/health")
def health():
    return jsonify(ok=True)


@app.post("/api/load")
def api_load():
    # Force reload on request
    with STATE_LOCK:
        STATE["df"] = load_prices()
        df = STATE["df"]

    return jsonify(
        loaded_rows=int(len(df)),
        tickers=int(df["ticker"].nunique()),
    )


@app.get("/api/summary/<ticker>")
def api_summary(ticker: str):
    t, sub = _get_ticker_df(ticker)
    if sub.empty:
        return jsonify(error=f"No data for ticker {t}"), 404

    metrics = summary_metrics(sub["close"])

    # Make sure it's JSON serializable
    if hasattr(metrics, "to_dict"):
        metrics_out = metrics.to_dict()
    elif isinstance(metrics, dict):
        metrics_out = metrics
    else:
        metrics_out = {"value": str(metrics)}

    return jsonify(ticker=t, metrics=metrics_out)


@app.get("/api/forecast/<ticker>")
def api_forecast(ticker: str):
    days_raw = request.args.get("days", "30")
    try:
        days = int(days_raw)
        if days <= 0 or days > 3650:
            return jsonify(error="days must be between 1 and 3650"), 400
    except ValueError:
        return jsonify(error="days must be an integer"), 400

    t, sub = _get_ticker_df(ticker)
    if sub.empty:
        return jsonify(error=f"No data for ticker {t}"), 404

    out = linear_regression_forecast(sub, days=days)

    # Return last 10 rows to keep payload small
    if hasattr(out, "tail") and hasattr(out, "to_dict"):
        tail = out.tail(10).to_dict(orient="records")
    else:
        tail = str(out)

    return jsonify(ticker=t, days=days, tail=tail)


# ------------------------
# Optional: keep your CLI
# ------------------------
def cmd_load():
    with STATE_LOCK:
        STATE["df"] = load_prices()
        df = STATE["df"]
    print(f"Loaded {len(df):,} rows. Tickers: {df['ticker'].nunique()}")


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
    print(out.tail(10) if hasattr(out, "tail") else out)


def run_cli():
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
    # Local usage:
    #   python app.py --cli   (runs your old CLI)
    #   python app.py         (runs dev server)
    import sys

    if "--cli" in sys.argv:
        run_cli()
    else:
        port = int(os.environ.get("PORT", "5000"))
        app.run(host="0.0.0.0", port=port, debug=True)