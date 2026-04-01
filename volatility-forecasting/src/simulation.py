import numpy as np


def monte_carlo_traditional(S0, mu, sigma, days, simulations, dt=1.0):
    paths = np.zeros((days, simulations))
    paths[0, :] = S0
    for t in range(1, days):
        z = np.random.standard_normal(simulations)
        paths[t, :] = paths[t-1, :] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return paths


def monte_carlo_ml_enhanced(S0, ml_predicted_log_return, sigma, days, simulations, dt=1.0):
    paths = np.zeros((days, simulations))
    paths[0, :] = S0
    mu_ml = ml_predicted_log_return
    for t in range(1, days):
        z = np.random.standard_normal(simulations)
        decay = np.exp(-0.5 * t)
        effective_mu = mu_ml * decay
        paths[t, :] = paths[t-1, :] * np.exp((effective_mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return paths


def monte_carlo_ml_garch(S0, ml_predicted_log_return, garch_volatilities, days, simulations, dt=1.0):
    if len(garch_volatilities) < days:
        last = garch_volatilities[-1] if len(garch_volatilities) > 0 else 0.0
        garch_volatilities = np.concatenate([garch_volatilities, np.repeat(last, days - len(garch_volatilities))])

    paths = np.zeros((days, simulations))
    paths[0, :] = S0
    mu_ml = ml_predicted_log_return
    for t in range(1, days):
        z = np.random.standard_normal(simulations)
        decay = np.exp(-0.5 * t)
        effective_mu = mu_ml * decay
        sigma_t = garch_volatilities[t-1]
        paths[t, :] = paths[t-1, :] * np.exp((effective_mu - 0.5 * sigma_t**2) * dt + sigma_t * np.sqrt(dt) * z)
    return paths
