# Backtesting

Rigorous backtesting is essential for validating portfolio strategy performance and risk management assumptions.

## Strategy focus

- `prediction-model/`: long/flat SPY strategy using logistic regression and technical features
- Performance analysis includes Sharpe ratio, max drawdown, and win rate

## Backtest process

1. Prepare time series data and features
2. Train a machine learning classifier on historical data
3. Generate trading signals with probability thresholds
4. Simulate portfolio returns and benchmark against buy-and-hold

## Key metrics

| Metric | Definition | Goal |
|---|---|---|
| Total return | Compound performance of strategy | Outperform benchmark |
| Sharpe ratio | Risk-adjusted return | Maximize |
| Max drawdown | Largest peak-to-trough loss | Minimize |
| Win rate | Percent profitable trades | Validate signal quality |

## Example output

```text
Strategy Return: 24.7%
Buy & Hold Return: 18.3%
Sharpe Ratio: 1.23
Max Drawdown: -12.4%
Win Rate: 52.1%
```

## Backtesting architecture

- `ml_trading_strategy.py` orchestrates the full pipeline
- Feature engineering creates RSI, MACD, volatility, moving averages
- Cross-validation uses time-aware folds for realistic evaluation
- Signal persistence and threshold tuning are built into the system

## Practical considerations

- Avoids look-ahead bias by preserving chronological order
- Handles NaN values and missing data with careful alignment
- Compares strategy returns to buy-and-hold for benchmarking
- Produces decision-ready insights for portfolio managers

> Placeholder: Add backtest equity curve visualization or strategy vs benchmark comparison here.
