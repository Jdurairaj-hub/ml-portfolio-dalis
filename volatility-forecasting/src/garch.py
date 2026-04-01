import numpy as np
from arch import arch_model


def fit_garch_model(log_returns_series, p=1, q=1):
    returns = log_returns_series.dropna()
    if returns.empty:
        raise ValueError("No returns available for GARCH fitting.")

    returns_pct = returns * 100.0
    model = arch_model(returns_pct, vol='Garch', p=p, q=q, rescale=False, mean='Constant')
    fitted = model.fit(disp='off')
    return fitted


def forecast_garch_volatility(fitted_garch, horizon):
    forecast = fitted_garch.forecast(horizon=horizon, reindex=False)
    try:
        variance_forecast_matrix = forecast.variance.values
        if variance_forecast_matrix.size == 0:
            raise ValueError("GARCH forecast variance empty.")

        last_row = variance_forecast_matrix[-1, :]
        volatility_forecast = np.sqrt(last_row) / 100.0

        if len(volatility_forecast) < horizon:
            last_val = volatility_forecast[-1] if len(volatility_forecast) > 0 else 0.0
            volatility_forecast = np.pad(volatility_forecast, (0, horizon - len(volatility_forecast)), 'edge')

        return volatility_forecast

    except Exception as e:
        print("Warning: unexpected GARCH forecast structure. Falling back to constant vol. Error:", e)
        longrun_vol = fitted_garch.conditional_volatility[-1] if hasattr(fitted_garch, 'conditional_volatility') else 0.0
        return np.repeat(longrun_vol, horizon)
