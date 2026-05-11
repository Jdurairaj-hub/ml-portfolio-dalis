# Machine Learning Models

This portfolio uses machine learning models to predict market direction, forecast volatility, and support strategy decisions.

## Model families

- **Logistic regression** for directional trading signals
- **Random Forest** and **Gradient Boosting** for regression and volatility prediction
- **Voting ensembles** to stabilize outputs and reduce overfit
- **GARCH models** for conditional volatility forecasting

## Model applications

| Model type | Project | Goal |
|---|---|---|
| Logistic regression | `prediction-model/` | Predict next-day market direction for SPY |
| Random Forest | `volatility-forecasting/` | Forecast future returns and risk factors |
| Gradient Boosting | `volatility-forecasting/` | Capture nonlinear relationships in market data |
| GARCH(1,1) | `volatility-forecasting/` | Model conditional volatility dynamics |

## Feature engineering

Examples of derived features:

- `ret_1d`, `ret_5d`, `ret_10d`
- `rsi_14`, `macd`, `bollinger_band_width`
- `volatility_10`, `volatility_20`
- `momentum`, `trend_strength`

## Training and validation

- Uses time series-aware train/test splits
- Avoids data leakage through chronological validation
- Tracks model performance with out-of-sample metrics
- Stores reusable model artifacts in `models/`

## Code sample

```python
model = LogisticRegression(C=0.5, max_iter=2000)
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)[:, 1]
```

## Portfolio intelligence

ML models in this repo are designed to support:

- **Risk forecasting** via combined ML + GARCH methods
- **Strategy signals** via probability-based thresholds
- **Research insights** through explainable, repeatable pipelines

> This page ties together the machine learning narrative across prediction, volatility, and signal modeling.
