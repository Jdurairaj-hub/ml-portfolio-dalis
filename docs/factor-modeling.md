# Factor Modeling

Factor modeling in this portfolio focuses on alternative drivers, correlation structure, and signal construction.

## Analysis scope

- **Macro and sentiment factors** from Google Trends and tariff discussions
- **Market factors** from technical indicator combinations
- **Statistical correlation** using Pearson, Spearman, and Kendall methods

## Project references

- `asset-correlation-analysis/`: correlation and factor testing
- `sentiment-analysis/`: factor combination with sentiment and technical signals

## Factor examples

| Factor | Description | Use case |
|---|---|---|
| AI search interest | Google Trends volume for AI keywords | Tech sector leading indicator |
| Tariff discussion | Search volume for tariff-related topics | Retail and supply chain risk |
| Momentum | Recent returns and moving averages | Trend-following signal |
| Volatility | Rolling standard deviation | Risk-adjusted position sizing |

## Modeling methodology

1. Collect and align feature time series
2. Compute factor exposures and weights
3. Test correlation with target returns
4. Validate statistical significance using p-values

### Example workflow

```python
aligned = align_series(stock_prices, search_trends)
correlations = compute_correlations(aligned, methods=['pearson', 'spearman', 'kendall'])
results = filter_significant(correlations, alpha=0.05)
```

## Research outcomes

- Strong relationships between AI search interest and large-cap tech names
- Evidence that tariff news correlates with retail performance
- Modular analysis pipeline ready for lagged factor testing

## Business relevance

- **Investment research:** identify persistent factor signals
- **Strategy design:** combine macro sentiment with technical momentum
- **Risk monitoring:** detect regime shifts in factor correlations

> Note: This page is designed to be the quantitative bridge between raw data signals and deployable models.
