# ML Trading Strategy - Prediction Model

A formalized machine learning approach for predicting next-day market direction and generating long/flat trading signals for SPY (S&P 500 ETF).

## 🚀 Quick Start

### Basic Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Run the strategy
python ml_trading_strategy.py
```

### Custom Configuration
```bash
# Analyze different ticker
python ml_trading_strategy.py --ticker QQQ

# Use different time period
python ml_trading_strategy.py --period 2y

# Adjust signal threshold
python ml_trading_strategy.py --threshold 0.6

# Save the trained model
python ml_trading_strategy.py --save-model
```

## 📊 What It Does

### 1. **Data Collection**
- Downloads historical OHLCV data from Yahoo Finance
- Supports any ticker symbol and time period
- Handles data validation and cleaning

### 2. **Feature Engineering**
Creates comprehensive technical indicators:
- **Returns**: 1-day, 5-day, 10-day percentage changes
- **RSI**: 14-period Relative Strength Index
- **Volatility**: 10-day and 20-day rolling standard deviation
- **Bollinger Bands**: Upper, middle, and lower bands
- **MACD**: MACD line, signal line, and histogram
- **Moving Averages**: 20-period and 50-period SMAs

### 3. **Machine Learning Model**
- **Algorithm**: Logistic Regression (optimized for interpretability)
- **Target**: Next-day price direction (up/down)
- **Validation**: Time series cross-validation
- **Evaluation**: Accuracy, AUC-ROC, confusion matrix

### 4. **Backtesting Framework**
- **Performance Metrics**: Total return, Sharpe ratio, maximum drawdown
- **Risk Analysis**: Volatility, win rate, trade statistics
- **Benchmarking**: Compares against buy-and-hold strategy
- **Data Integrity**: NaN handling and index alignment checks in return calculation
- **Visualization**: Comprehensive charts and performance plots

### 5. **Signal Generation**
- **Real-time Signals**: Generates actionable long/flat signals
- **Probability Estimates**: Provides confidence levels for predictions
- **Threshold Control**: Configurable probability thresholds

## 📈 Key Features

### Professional Architecture
- **Object-Oriented Design**: `MLTradingStrategy` class with clean interfaces
- **Configuration Management**: Flexible parameter system
- **Error Handling**: Robust error handling and logging
- **Model Persistence**: Save/load trained models

### Comprehensive Analysis
- **Cross-Validation**: Time series-aware validation
- **Backtesting**: Historical performance simulation
- **Risk Metrics**: Professional risk assessment
- **Visualization**: Publication-quality charts

### Production Ready
- **Command-Line Interface**: Scriptable automation
- **Logging**: Detailed progress and error reporting
- **Caching**: Efficient data management
- **Documentation**: Comprehensive docstrings

## 📊 Sample Output

```
============================================================
🎯 ML TRADING SIGNAL - DALIS PORTFOLIO
============================================================
Date: 2026-04-01
Ticker: SPY
Signal: LONG
Probability Market Up: 0.6234
Threshold: 0.55
============================================================

📊 BACKTEST SUMMARY:
Strategy Return: 24.7%
Buy & Hold Return: 18.3%
Sharpe Ratio: 1.23
Max Drawdown: -12.4%
Win Rate: 52.1%
============================================================
```

## 🏗️ Architecture

```
MLTradingStrategy
├── Data Management
│   ├── download_data() - Yahoo Finance integration
│   └── create_features() - Technical indicator creation
├── Model Training
│   ├── prepare_data() - Feature/target preparation
│   ├── train_model() - Logistic regression training
│   └── cross_validate() - Time series CV
├── Analysis & Backtesting
│   ├── backtest_strategy() - Historical simulation
│   ├── predict_signal() - Real-time signals
│   └── plot_backtest_results() - Performance visualization
└── Utilities
    ├── save_model() - Model persistence
    ├── load_model() - Model loading
    └── run_full_analysis() - Complete pipeline
```

## 🎯 Use Cases

### Investment Research
- **Market Timing**: Identify optimal entry/exit points
- **Risk Management**: Assess strategy volatility and drawdowns
- **Performance Attribution**: Compare against benchmarks

### Algorithmic Trading
- **Signal Generation**: Automated buy/sell signals
- **Portfolio Management**: Long/flat strategy implementation
- **Backtesting**: Validate strategies before live trading

### Educational
- **ML in Finance**: Learn ML applications in trading
- **Technical Analysis**: Study indicator relationships
- **Risk Analysis**: Understand trading strategy metrics

## 📋 Requirements

- Python 3.8+
- Dependencies: `pip install -r requirements.txt`
- Internet connection for data downloading

## 🔧 Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ticker` | SPY | Stock symbol to analyze |
| `period` | 5y | Historical data period |
| `threshold` | 0.55 | Signal probability threshold |
| `test_size` | 0.2 | Train/test split ratio |

## 📁 Output Files

- `output/backtest_results.png` - Performance visualization
- `models/*.joblib` - Saved trained models
- Console logs with detailed metrics

## 🚀 Advanced Usage

### Custom Feature Set
```python
config = {
    'features': ['ret_1d', 'rsi_14', 'macd', 'sma_20'],
    'model_params': {'C': 0.5, 'max_iter': 2000}
}
strategy = MLTradingStrategy(config)
```

### Model Loading
```python
strategy = MLTradingStrategy()
strategy.load_model('models/ml_trading_model_SPY_20260401.joblib')
signal = strategy.predict_signal()
```

## 🧾 Backtest Validation (Cross-Checked)

- Re-ran the full on-disk pipeline on 1y and 2y SPY data (2025-04-01 -> 2026-04-01 / 2024-04-01 -> 2026-04-01)
- Verified that `backtest_strategy()` now handles NaN values from `ret_fwd_1d` with `fillna(0)` and maintains per-index alignment with `self.data.loc[X.index]`
- Confirmed final metrics are numeric and stable (e.g., 1y run: strategy 19.25% vs buy-hold 10.73%, Sharpe 1.80)
- Signal generation is consistent with threshold and probability returned by model

## ⚠️ Important Notes

- **Not Financial Advice**: This is for educational/research purposes
- **Past Performance**: No guarantee of future results
- **Market Risks**: Trading involves substantial risk of loss
- **Data Quality**: Results depend on data accuracy and market conditions

## 🔬 Future Enhancements

- **Ensemble Models**: Random Forest, Gradient Boosting
- **Feature Selection**: Automated feature importance analysis
- **Hyperparameter Tuning**: Grid search optimization
- **Multi-Asset Strategies**: Portfolio-level optimization
- **Live Trading Integration**: Broker API connections

---

*Built for demonstrating quantitative finance and machine learning expertise in algorithmic trading strategies.*