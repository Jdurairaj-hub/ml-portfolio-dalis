# Time Series & Forecasting

Time series modeling is central to this portfolio. It combines classic financial forecasting with modern machine learning and simulation techniques.

## Core Projects

- `volatility-forecasting/`: ML-enhanced volatility prediction with GARCH and Monte Carlo simulation
- `asset-correlation-analysis/`: Trend data correlation analysis using Google Trends and stock prices

## Forecasting Approach

### Data pipeline

- Acquire historical OHLCV data from Polygon.io and Yahoo Finance
- Normalize and align daily and weekly time series
- Construct technical features and risk indicators

### Forecast methods

- **Traditional Monte Carlo** using historical drift and volatility
- **ML-enhanced Monte Carlo** leveraging model predicted drift
- **ML + GARCH Monte Carlo** combining ML drift with GARCH volatility forecasts

## Feature engineering

```python
features = [
    'log_return',
    'ma_10',
    'ma_20',
    'volatility_10',
    'volatility_20',
    'momentum',
]
```

### Forecast evaluation

Key evaluation steps:

- Train/test split consistent with time order
- Compare forecasted return distributions
- Quantify forecast bias and variance
- Validate against actual realized volatility

## Cross-project synergy

| Forecast focus | Related project |
|---|---|
| Volatility and risk distribution | `volatility-forecasting/` |
| Signal persistence and market psychology | `asset-correlation-analysis/` |
| Risk-adjusted decision support | `prediction-model/` |

## Outputs and deliverables

- Simulation comparison plots saved to `output/`
- Forecast statistics including VaR, expected return, probability of profit
- ML model performance metrics for regression and volatility prediction

## Architecture section

1. **Data ingestion** (`src/data.py`)
2. **Feature generation** (`src/features.py`)
3. **Model training** (`src/model.py`)
4. **Volatility modeling** (`src/garch.py`)
5. **Simulations** (`src/simulation.py`)

> Placeholder: Add visual chart of forecast distributions and simulation comparison here.
