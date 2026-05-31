import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ingest import load_prices
from analytics import summary_metrics
from ml_models import forecast_with_model


st.set_page_config(
    page_title="ML Stock Assistant",
    page_icon="📈",
    layout="wide",
)

st.title("📈 ML Stock Assistant")
st.caption("S&P 500 analytics, comparison, risk metrics, and ARIMA/Prophet forecasting")


@st.cache_data
def get_data():
    return load_prices()


def fmt_pct(x):
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:.2%}"


def fmt_num(x):
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:.2f}"


def price_chart(df_ticker, ticker):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_ticker["date"],
            y=df_ticker["close"],
            mode="lines",
            name=ticker,
        )
    )
    fig.update_layout(
        title=f"{ticker} Close Price",
        xaxis_title="Date",
        yaxis_title="Close Price",
        template="plotly_dark",
        height=520,
    )
    return fig


def compare_chart(df, tickers):
    fig = go.Figure()

    for ticker in tickers:
        sub = df[df["ticker"] == ticker].sort_values("date")
        if sub.empty:
            continue

        normalized = sub["close"] / sub["close"].iloc[0] * 100
        fig.add_trace(
            go.Scatter(
                x=sub["date"],
                y=normalized,
                mode="lines",
                name=ticker,
            )
        )

    fig.update_layout(
        title="Normalized Comparison — Start = 100",
        xaxis_title="Date",
        yaxis_title="Normalized Price",
        template="plotly_dark",
        height=520,
    )
    return fig


def forecast_chart(df_forecast, ticker, model_used):
    hist = df_forecast[df_forecast["type"] == "history"]
    fut = df_forecast[df_forecast["type"] == "forecast"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hist["date"],
            y=hist["yhat"],
            mode="lines",
            name="History",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=fut["date"],
            y=fut["yhat"],
            mode="lines",
            name="Forecast",
        )
    )

    if fut["yhat_lower"].notna().any() and fut["yhat_upper"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=fut["date"],
                y=fut["yhat_upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                name="Upper Band",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fut["date"],
                y=fut["yhat_lower"],
                mode="lines",
                fill="tonexty",
                line=dict(width=0),
                name="95% Confidence Band",
            )
        )

    fig.update_layout(
        title=f"{ticker} Forecast — {model_used.upper()}",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        height=520,
    )
    return fig


try:
    df = get_data()

except Exception:
    st.warning("Local dataset not found. Downloading S&P 500 data...")

    from fetch_sp500_data import main as fetch_data

    with st.spinner("Downloading market data. This may take a minute..."):
        fetch_data()

    df = get_data()

    st.success("Data loaded successfully.")


tickers_available = sorted(df["ticker"].unique())

with st.sidebar:
    st.header("Controls")

    mode = st.radio(
        "Mode",
        ["Summary", "Risk", "Compare", "Forecast"],
    )

    ticker_input = st.text_input(
        "Ticker(s)",
        value="AAPL",
        help="Use one ticker for Summary/Risk/Forecast. Use commas for Compare: AAPL, NVDA, MSFT",
    )

    forecast_days = st.number_input(
        "Forecast days",
        min_value=1,
        max_value=365,
        value=30,
    )

    model_name = st.selectbox(
        "Forecast model",
        ["arima", "prophet", "linreg"],
        index=0,
    )

    run = st.button("Run Analysis", type="primary")


st.info(f"Loaded {len(df):,} rows across {df['ticker'].nunique()} tickers.")

if not run:
    st.write("Choose a mode from the sidebar and click **Run Analysis**.")
    st.stop()


if mode in ["Summary", "Risk", "Forecast"]:
    ticker = ticker_input.strip().upper()

    if ticker not in tickers_available:
        st.error(f"No data found for ticker: {ticker}")
        st.stop()

    sub = df[df["ticker"] == ticker].sort_values("date")
    metrics = summary_metrics(sub)

    start_dt = sub["date"].min().date()
    end_dt = sub["date"].max().date()

    st.subheader(f"{ticker} Analysis")
    st.caption(f"Data range: {start_dt} → {end_dt} | Rows: {len(sub):,}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR", fmt_pct(metrics["cagr"]))
    c2.metric("Sharpe", fmt_num(metrics["sharpe"]))
    c3.metric("Annual Vol", fmt_pct(metrics["vol_annual"]))
    c4.metric("Max Drawdown", fmt_pct(metrics["max_drawdown"]))
    c5.metric("Risk", metrics["risk"])

    if mode == "Summary":
        st.plotly_chart(price_chart(sub, ticker), use_container_width=True)

        with st.expander("Detailed Metrics", expanded=True):
            st.write(
                {
                    "Total Return": fmt_pct(metrics["total_return"]),
                    "CAGR": fmt_pct(metrics["cagr"]),
                    "Daily Volatility": fmt_num(metrics["vol_daily"]),
                    "Annual Volatility": fmt_pct(metrics["vol_annual"]),
                    "Sharpe Ratio": fmt_num(metrics["sharpe"]),
                    "Max Drawdown": fmt_pct(metrics["max_drawdown"]),
                    "Max Drawdown Duration": f"{metrics['max_dd_duration_days']} days",
                    "1Y Return": fmt_pct(metrics["ret_1y"]),
                    "3Y Return": fmt_pct(metrics["ret_3y"]),
                    "5Y Return": fmt_pct(metrics["ret_5y"]),
                    "Risk": metrics["risk"],
                }
            )

    elif mode == "Risk":
        st.plotly_chart(price_chart(sub, ticker), use_container_width=True)

        st.warning(
            f"{ticker} risk level is **{metrics['risk']}**. "
            f"Annual volatility is {fmt_pct(metrics['vol_annual'])}, "
            f"and max drawdown is {fmt_pct(metrics['max_drawdown'])}."
        )

    elif mode == "Forecast":
        with st.spinner("Generating forecast..."):
            out = forecast_with_model(
                sub,
                days=int(forecast_days),
                model_name=model_name,
            )

        st.success(
            f"Forecast complete using **{out['model_used'].upper()}**. "
            f"Confidence bands: {'Yes' if out['has_bands'] else 'No'}"
        )

        st.plotly_chart(
            forecast_chart(out["df_forecast"], ticker, out["model_used"]),
            use_container_width=True,
        )


elif mode == "Compare":
    tickers = [
        t.strip().upper()
        for t in ticker_input.replace(",", " ").split()
        if t.strip()
    ]

    if len(tickers) < 2:
        st.error("Enter at least 2 tickers, for example: AAPL, NVDA, MSFT")
        st.stop()

    valid = [t for t in tickers if t in tickers_available]
    missing = [t for t in tickers if t not in tickers_available]

    if missing:
        st.warning(f"Missing tickers skipped: {', '.join(missing)}")

    if len(valid) < 2:
        st.error("Not enough valid tickers to compare.")
        st.stop()

    st.subheader("Multi-Stock Comparison")
    st.plotly_chart(compare_chart(df, valid), use_container_width=True)

    rows = []
    for ticker in valid:
        sub = df[df["ticker"] == ticker].sort_values("date")
        m = summary_metrics(sub)
        rows.append(
            {
                "Ticker": ticker,
                "CAGR": m["cagr"],
                "Sharpe": m["sharpe"],
                "Annual Vol": m["vol_annual"],
                "Max Drawdown": m["max_drawdown"],
                "Risk": m["risk"],
            }
        )

    compare_df = pd.DataFrame(rows)

    st.subheader("Comparison Table")

    display_df = compare_df.copy()

    display_df["CAGR"] = display_df["CAGR"].map(lambda x: f"{x:.2%}")
    display_df["Sharpe"] = display_df["Sharpe"].map(lambda x: f"{x:.2f}")
    display_df["Annual Vol"] = display_df["Annual Vol"].map(lambda x: f"{x:.2%}")
    display_df["Max Drawdown"] = display_df["Max Drawdown"].map(lambda x: f"{x:.2%}")

    st.dataframe(display_df, use_container_width=True)