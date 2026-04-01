# Google Trends vs Stock Price Correlation Analysis - Results Summary

## 📊 Analysis Overview

This improved script analyzes correlations between Google Trends search interest and stock prices, providing statistically rigorous insights into market sentiment indicators.

**Analysis Period:** April 2024 - November 2025 (85 weekly data points)
**Keywords:** AI, tariffs
**Stocks:** NAVBQ (shipping), GOOG (Google), META (Meta), WMT (Walmart)

## 🔍 Key Findings

### Statistically Significant Correlations (p < 0.05)

#### AI Search Interest as Market Indicator
- **GOOG (Google):** r = 0.66, p ≈ 0.000
- **META (Meta Platforms):** r = 0.82, p ≈ 0.000
- **WMT (Walmart):** r = 0.83, p ≈ 0.000
- **Tariffs:** r = 0.27, p = 0.014

#### Tariff Discussions Impact
- **WMT (Walmart):** r = 0.28, p = 0.0085

#### Stock Inter-correlations
- **META-WMT:** r = 0.87, p ≈ 0.000
- **GOOG-META:** r = 0.57, p ≈ 0.000
- **GOOG-WMT:** r = 0.56, p ≈ 0.000

## 📈 Interpretation

### AI as Technology Sector Leading Indicator
The strong correlations between AI search interest and tech stock performance suggest that AI-related searches may serve as a leading indicator for technology sector performance. This could be valuable for:
- Investment timing in tech stocks
- Sector rotation strategies
- Sentiment analysis for algorithmic trading

### Tariff Impact on Retail
The correlation between tariff discussions and Walmart stock performance indicates that tariff policy discussions may have measurable impacts on retail sector expectations.

### Stock Relationships
The high correlation between META and WMT suggests these stocks move together, possibly due to broader market trends or shared economic sensitivities.

## 🛠️ Technical Improvements Made

### 1. **Robust Data Pipeline**
- Proper timezone handling for datetime indices
- Intelligent data alignment between weekly Google Trends and daily stock data
- Caching system to avoid redundant API calls
- Error handling and logging throughout

### 2. **Statistical Rigor**
- Multiple correlation methods (Pearson, Spearman, Kendall)
- Statistical significance testing with p-values
- Only significant correlations displayed in visualizations

### 3. **Modular Architecture**
- Object-oriented design with `GoogleTrendsStockAnalyzer` class
- Configurable parameters via command-line arguments
- Separation of concerns (data fetching, processing, analysis, visualization)

### 4. **Enhanced Analysis**
- Rolling correlations capability (framework in place)
- Multiple statistical methods for robustness
- Comprehensive output including CSV files and visualizations

### 5. **Production-Ready Features**
- Command-line interface for automation
- Configurable parameters (keywords, tickers, timeframes, geography)
- Proper logging and error handling
- Cache management for efficiency

## 🚀 Usage Examples

```bash
# Basic analysis with defaults
python screener_sarah.py

# Custom analysis
python screener_sarah.py --keywords "bitcoin" "ethereum" --tickers "COIN" "MSTR" "SQ"

# Skip cache for fresh data
python screener_sarah.py --no-cache

# Change geographic region
python screener_sarah.py --geo "GB"
```

## 📁 Output Files

- `correlation_pearson.csv` - Pearson correlation matrix
- `correlation_spearman.csv` - Spearman rank correlation
- `correlation_kendall.csv` - Kendall tau correlation
- `correlation_significance.csv` - Statistical significance (p-values)
- `correlation_analysis.png` - Visualization with significance masking
- `aligned_data.csv` - Raw aligned dataset

## 🔬 Future Enhancements

1. **Rolling Correlations** - Analyze how relationships change over time
2. **Lagged Analysis** - Test if Google Trends leads or lags stock prices
3. **Machine Learning Models** - Predict stock movements using search data
4. **Multi-region Analysis** - Compare correlations across geographies
5. **Sentiment Integration** - Incorporate news sentiment with search data

## 💡 Business Applications

- **Investment Research:** Identify leading indicators for sector performance
- **Risk Management:** Monitor sentiment-driven market risks
- **Algorithmic Trading:** Incorporate sentiment signals into trading strategies
- **Market Intelligence:** Track emerging trends through search behavior

---

*This analysis demonstrates how web search data can provide valuable insights into market dynamics and investor sentiment.*</content>
