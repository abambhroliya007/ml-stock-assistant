## ML Stock Assistant

ML Stock Assistant is an interactive, end-to-end stock analysis platform built to analyze S&P 500 equities using historical market data and machine learning models.

The application allows users to:
- Explore long-term stock performance and risk metrics
- Compare multiple equities side-by-side using normalized charts and heatmap tables
- Forecast future price movements using ARIMA and Prophet models with confidence intervals
- Interact with dynamic Plotly charts (hover, zoom, toggle overlays)
- Evaluate stocks using professional KPIs such as CAGR, Sharpe ratio, volatility, and max drawdown

The project is designed as a full-stack ML system, combining data engineering, quantitative finance, machine learning, and frontend visualization into a single, production-style application.

# data

This folder holds cached datasets used by the app.

**Do not commit large data files.**

To recreate the historical S&P 500 dataset used by this project, run:

```bash
python fetch_sp500_data.py
