import numbers
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from flask import Flask, render_template, request, jsonify

from ingest import load_prices
from analytics import summary_metrics
from ml_models import forecast_with_model

app = Flask(__name__)
STATE = {"df": None}


def _metrics_to_json(m: dict) -> dict:
    out = {}
    for k, v in m.items():
        if v is None:
            out[k] = None
        elif isinstance(v, numbers.Number):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _parse_tickers(s: str):
    if not s:
        return []
    raw = s.replace(",", " ").split()
    tickers = []
    for x in raw:
        x = x.strip().upper()
        if x and x not in tickers:
            tickers.append(x)
    return tickers


def _to_iso_dates(dts) -> List[str]:
    return [pd.Timestamp(x).strftime("%Y-%m-%d") for x in dts]


def _plotly_price_figure(df_ticker: pd.DataFrame, title: str) -> Dict[str, Any]:
    df = df_ticker.sort_values("date")
    fig = {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Close",
                "x": _to_iso_dates(df["date"]),
                "y": df["close"].astype(float).tolist(),
            }
        ],
        "layout": {
            "title": {"text": title},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "margin": {"l": 50, "r": 30, "t": 50, "b": 40},
            "xaxis": {"title": "Date", "showgrid": True, "gridcolor": "rgba(255,255,255,0.06)"},
            "yaxis": {"title": "Price", "showgrid": True, "gridcolor": "rgba(255,255,255,0.06)"},
            "legend": {"orientation": "h"},
        },
    }
    return fig


def _plotly_compare_figure(df: pd.DataFrame, tickers: List[str], title: str) -> Dict[str, Any]:
    data = []
    for t in tickers:
        sub = df[df["ticker"] == t].sort_values("date")
        if sub.empty:
            continue
        base = float(sub["close"].iloc[0])
        if base <= 0:
            continue
        y = (sub["close"].astype(float) / base) * 100.0
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": t,
                "x": _to_iso_dates(sub["date"]),
                "y": y.tolist(),
            }
        )

    fig = {
        "data": data,
        "layout": {
            "title": {"text": title},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "margin": {"l": 60, "r": 30, "t": 50, "b": 40},
            "xaxis": {"title": "Date", "showgrid": True, "gridcolor": "rgba(255,255,255,0.06)"},
            "yaxis": {"title": "Normalized (Start=100)", "showgrid": True, "gridcolor": "rgba(255,255,255,0.06)"},
            "legend": {"orientation": "h"},
        },
    }
    return fig


def _plotly_forecast_figure(df_fc: pd.DataFrame, title: str) -> Dict[str, Any]:
    """
    df_fc columns: date, yhat, yhat_lower, yhat_upper, type
    """
    df_fc = df_fc.sort_values("date")
    hist = df_fc[df_fc["type"] == "history"]
    fut = df_fc[df_fc["type"] == "forecast"]

    data = [
        {
            "type": "scatter",
            "mode": "lines",
            "name": "History",
            "x": _to_iso_dates(hist["date"]),
            "y": hist["yhat"].astype(float).tolist(),
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Forecast",
            "x": _to_iso_dates(fut["date"]),
            "y": fut["yhat"].astype(float).tolist(),
        },
    ]

    # Confidence band (filled)
    if fut["yhat_lower"].notna().any() and fut["yhat_upper"].notna().any():
        x = _to_iso_dates(fut["date"])
        lower = fut["yhat_lower"].astype(float).tolist()
        upper = fut["yhat_upper"].astype(float).tolist()

        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Upper",
                "x": x,
                "y": upper,
                "line": {"width": 0},
                "showlegend": False,
                "hoverinfo": "skip",
            }
        )
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": "95% CI",
                "x": x,
                "y": lower,
                "fill": "tonexty",
                "fillcolor": "rgba(124,92,255,0.18)",
                "line": {"width": 0},
                "hoverinfo": "skip",
            }
        )

    fig = {
        "data": data,
        "layout": {
            "title": {"text": title},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "margin": {"l": 60, "r": 30, "t": 50, "b": 40},
            "xaxis": {"title": "Date", "showgrid": True, "gridcolor": "rgba(255,255,255,0.06)"},
            "yaxis": {"title": "Price", "showgrid": True, "gridcolor": "rgba(255,255,255,0.06)"},
            "legend": {"orientation": "h"},
        },
    }
    return fig


def ticker_date_range(df_ticker):
    start = df_ticker["date"].min()
    end = df_ticker["date"].max()
    return start.date(), end.date()


def _compare_kpis(rows):
    def best(key, reverse=True):
        valid = [r for r in rows if isinstance(r.get(key), (int, float))]
        if not valid:
            return None
        return sorted(valid, key=lambda r: r[key], reverse=reverse)[0]

    def fmt_pct(x):
        return "n/a" if x is None else f"{x*100:.2f}%"

    def fmt_num(x):
        return "n/a" if x is None else f"{x:.2f}"

    best_cagr = best("cagr", reverse=True)
    best_sharpe = best("sharpe", reverse=True)
    low_vol = best("vol_annual", reverse=False)
    low_dd = best("max_drawdown", reverse=True)

    return {
        "best_cagr": f"{best_cagr['ticker']} {fmt_pct(best_cagr['cagr'])}" if best_cagr else "n/a",
        "best_sharpe": f"{best_sharpe['ticker']} {fmt_num(best_sharpe['sharpe'])}" if best_sharpe else "n/a",
        "lowest_vol": f"{low_vol['ticker']} {fmt_pct(low_vol['vol_annual'])}" if low_vol else "n/a",
        "least_drawdown": f"{low_dd['ticker']} {fmt_pct(low_dd['max_drawdown'])}" if low_dd else "n/a",
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/load", methods=["POST"])
def api_load():
    STATE["df"] = load_prices()
    return jsonify({"response": f"Loaded {len(STATE['df']):,} rows. Tickers: {STATE['df']['ticker'].nunique()}."})


@app.route("/api/summary", methods=["POST"])
def api_summary():
    if STATE["df"] is None:
        return jsonify({"response": "Run Load first."}), 400

    ticker = (request.json.get("ticker") or "").upper().strip()
    if not ticker:
        return jsonify({"response": "Please enter a ticker."}), 400

    sub = STATE["df"][STATE["df"]["ticker"] == ticker]
    if sub.empty:
        return jsonify({"response": f"No data for {ticker}"}), 404

    m = summary_metrics(sub)
    start_dt, end_dt = ticker_date_range(sub)

    text = (
        f"{ticker} summary:\n"
        f"- Data range: {start_dt} → {end_dt} ({len(sub):,} rows)\n"
        f"- CAGR: {m.get('cagr'):.2%}\n"
        f"- Total return: {m.get('total_return'):.2%}\n"
        f"- Volatility (annual): {m.get('vol_annual'):.2%}\n"
        f"- Sharpe (rf=0): {m.get('sharpe'):.2f}\n"
        f"- Max drawdown: {m.get('max_drawdown'):.2%}\n"
        f"- Max DD duration: {m.get('max_dd_duration_days')} days\n"
        f"- 1Y / 3Y / 5Y return: {m.get('ret_1y'):.2%} / {m.get('ret_3y'):.2%} / {m.get('ret_5y'):.2%}\n"
        f"- Risk: {m.get('risk')}"
    )

    fig = _plotly_price_figure(sub, f"{ticker} Close Price")
    return jsonify({"response": text, "metrics": _metrics_to_json(m), "plotly": fig})


@app.route("/api/risk", methods=["POST"])
def api_risk():
    if STATE["df"] is None:
        return jsonify({"response": "Run Load first."}), 400

    ticker = (request.json.get("ticker") or "").upper().strip()
    if not ticker:
        return jsonify({"response": "Please enter a ticker."}), 400

    sub = STATE["df"][STATE["df"]["ticker"] == ticker]
    if sub.empty:
        return jsonify({"response": f"No data for {ticker}"}), 404

    m = summary_metrics(sub)
    start_dt, end_dt = ticker_date_range(sub)

    text = (
        f"{ticker} risk:\n"
        f"- Data range: {start_dt} → {end_dt} ({len(sub):,} rows)\n"
        f"- Risk: {m.get('risk')}\n"
        f"- Volatility (annual): {m.get('vol_annual'):.2%}\n"
        f"- Sharpe (rf=0): {m.get('sharpe'):.2f}\n"
        f"- Max drawdown: {m.get('max_drawdown'):.2%}\n"
        f"- Max DD duration: {m.get('max_dd_duration_days')} days"
    )

    fig = _plotly_price_figure(sub, f"{ticker} Risk View")
    return jsonify({"response": text, "metrics": _metrics_to_json(m), "plotly": fig})


@app.route("/api/compare", methods=["POST"])
def api_compare():
    if STATE["df"] is None:
        return jsonify({"response": "Run Load first."}), 400

    tickers_in = request.json.get("tickers") or request.json.get("ticker") or ""
    tickers = _parse_tickers(str(tickers_in))

    if len(tickers) < 2:
        return jsonify({"response": "Enter at least 2 tickers (e.g., AAPL, NVDA, MSFT)."}), 400
    tickers = tickers[:8]

    df = STATE["df"]
    rows = []
    found = []
    for t in tickers:
        sub = df[df["ticker"] == t]
        if sub.empty:
            continue
        found.append(t)
        m = summary_metrics(sub)
        rows.append({"ticker": t, **_metrics_to_json(m)})

    if len(found) < 2:
        return jsonify({"response": "Not enough tickers with data to compare."}), 400

    fig = _plotly_compare_figure(df, found, f"Compare (Normalized): {' vs '.join(found)}")
    return jsonify({
        "response": "Comparison ready.",
        "plotly": fig,
        "compare": rows,
        "tickers": found,
        "kpis_compare": _compare_kpis(rows)
    })


@app.route("/api/forecast", methods=["POST"])
def api_forecast():
    if STATE["df"] is None:
        return jsonify({"response": "Run Load first."}), 400

    ticker = (request.json.get("ticker") or "").upper().strip()
    days = int(request.json.get("days") or 30)
    days = max(1, min(days, 365))
    model_name = (request.json.get("model") or "arima").lower().strip()

    if not ticker:
        return jsonify({"response": "Please enter a ticker."}), 400

    sub = STATE["df"][STATE["df"]["ticker"] == ticker]
    if sub.empty:
        return jsonify({"response": f"No data for {ticker}"}), 404

    start_dt, end_dt = ticker_date_range(sub)

    try:
        out = forecast_with_model(sub, days=days, model_name=model_name)
        df_fc = out["df_forecast"]
        model_used = out["model_used"]
        fig = _plotly_forecast_figure(df_fc, f"{ticker} Forecast ({model_used.upper()}, {days} days)")
        text = (
            f"{ticker} forecast:\n"
            f"- Data range: {start_dt} → {end_dt} ({len(sub):,} rows)\n"
            f"- Forecast horizon: {days} days\n"
            f"- Model: {model_used.upper()}\n"
            f"- Confidence band: {'YES' if out['has_bands'] else 'NO'}"
        )
    except Exception as e:
        return jsonify({"response": f"Forecast failed: {str(e)}"}), 400

    m = summary_metrics(sub)
    return jsonify({"response": text, "metrics": _metrics_to_json(m), "plotly": fig})


@app.route("/chat", methods=["POST"])
def chat():
    msg = (request.json.get("message") or request.json.get("input") or "").strip()
    if not msg:
        return jsonify({"response": "Type: /load, summary AAPL, compare AAPL NVDA MSFT, forecast AAPL 30 arima"}), 400

    parts = msg.split()
    cmd = parts[0].lower()

    if cmd == "/load":
        STATE["df"] = load_prices()
        return jsonify({"response": f"Loaded {len(STATE['df']):,} rows. Tickers: {STATE['df']['ticker'].nunique()}."})

    if STATE["df"] is None:
        return jsonify({"response": "Run /load first (or press Load)."}), 400

    # Keep chat simple: recommend using buttons
    return jsonify({"response": "Use the buttons (Summary/Risk/Forecast/Compare) for the best experience with interactive charts."}), 400


if __name__ == "__main__":
    app.run(debug=True)
