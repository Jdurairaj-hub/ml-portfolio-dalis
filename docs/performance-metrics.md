# Performance Metrics

Performance metrics are the foundation of research validation, risk assessment, and portfolio decision-making.

## Core metrics

| Metric | Definition | Why it matters |
|---|---|---|
| Sharpe ratio | Risk-adjusted return | Measures strategy efficiency |
| Max drawdown | Largest drawdown | Captures worst-case portfolio loss |
| Win rate | Percent profitable trades | Indicates signal quality |
| VaR | Value-at-risk | Estimates tail risk |
| Expected return | Forecasted return | Sets strategy expectations |

## Backtest evaluation

The `prediction-model/` project includes:

- strategy vs benchmark comparison
- probability-driven signal generation
- cumulative return curves
- risk-adjusted performance tables

## Forecast and simulation metrics

The `volatility-forecasting/` project includes:

- distribution comparisons between simulation methods
- volatility forecast accuracy
- probability of profit estimates
- scenario-based risk diagnostics

## Statistical validation

This portfolio also emphasizes statistical rigor:

- significance testing for correlation results
- multiple correlation methods (Pearson, Spearman, Kendall)
- cross-validation to avoid overfitting
- robust metric reporting for reproducibility

## Example metric summary

```text
Strategy Return: 24.7%
Sharpe Ratio: 1.23
Max Drawdown: -12.4%
Win Rate: 52.1%
```

## Practical takeaway

High-quality performance reporting helps recruiters and stakeholders understand model quality, downside risk, and whether research has production readiness.
