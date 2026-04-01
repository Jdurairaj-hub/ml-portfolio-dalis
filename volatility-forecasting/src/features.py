import numpy as np


def create_features(df):
    data = df.copy()

    data['log_return'] = np.log(data['close']).diff()
    data['next_day_log_return'] = data['log_return'].shift(-1)

    data['ma5'] = data['close'].rolling(window=5).mean()
    data['ma20'] = data['close'].rolling(window=20).mean()
    data['ma_ratio'] = data['ma5'] / data['ma20']

    data['volatility'] = data['log_return'].rolling(window=20).std()

    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))

    exp1 = data['close'].ewm(span=12, adjust=False).mean()
    exp2 = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = exp1 - exp2

    data['volume_change'] = data['volume'].pct_change()

    for i in range(1, 6):
        data[f'log_return_lag_{i}'] = data['log_return'].shift(i)

    return data


def create_advanced_features(data):
    df = data.copy()

    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_10'] = df['close'].pct_change(10)
    df['momentum_20'] = df['close'].pct_change(20)

    df['vol_ratio'] = df['log_return'].rolling(5).std() / df['log_return'].rolling(20).std()

    df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_momentum'] = df['volume'].pct_change(5)

    df['high_low_ratio'] = (df['close'] - df['low'].rolling(20).min()) / \
                           (df['high'].rolling(20).max() - df['low'].rolling(20).min())

    df['bb_position'] = (df['close'] - df['ma20']) / (df['volatility'] * df['close'] * 2 + 1e-12)

    df['rsi_change'] = df['rsi'].diff(5)

    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    return df


def calculate_risk_metrics(paths, S0):
    final_prices = paths[-1]
    returns = np.log(final_prices / S0)
    metrics = {
        'Expected Log Return': returns.mean(),
        'Volatility (Std of Log Returns)': returns.std(),
        'VaR (95%)': np.percentile(returns, 5),
        'CVaR (95%)': returns[returns <= np.percentile(returns, 5)].mean(),
        'Prob of Profit': (returns > 0).mean(),
        'Expected Price': final_prices.mean(),
        'Median Price': np.median(final_prices)
    }
    return metrics


def plot_monte_carlo_comparison_3methods(traditional_paths, ml_paths, ml_garch_paths, S0, ticker):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle(f'Monte Carlo Simulation Comparison: {ticker}', fontsize=18, fontweight='bold')
    days = traditional_paths.shape[0]

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(traditional_paths[:, :100], alpha=0.25, linewidth=0.5)
    ax1.plot(traditional_paths.mean(axis=1), 'r-', linewidth=2, label='Mean Path')
    ax1.axhline(y=S0, color='black', linestyle='--', label='Current Price')
    ax1.set_title('Traditional MC (Historical μ, σ)')
    ax1.set_xlabel('Days'); ax1.set_ylabel('Price ($)'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ml_paths[:, :100], alpha=0.25, linewidth=0.5)
    ax2.plot(ml_paths.mean(axis=1), 'g-', linewidth=2, label='Mean Path')
    ax2.axhline(y=S0, color='black', linestyle='--')
    ax2.set_title('ML-Enhanced MC (ML μ, Historical σ)')
    ax2.set_xlabel('Days'); ax2.set_ylabel('Price ($)'); ax2.legend(); ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(ml_garch_paths[:, :100], alpha=0.25, linewidth=0.5)
    ax3.plot(ml_garch_paths.mean(axis=1), color='purple', linewidth=2, label='Mean Path')
    ax3.axhline(y=S0, color='black', linestyle='--')
    ax3.set_title('ML+GARCH MC (ML μ, GARCH σ)')
    ax3.set_xlabel('Days'); ax3.set_ylabel('Price ($)'); ax3.legend(); ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(traditional_paths[-1], bins=50, alpha=0.7, edgecolor='black')
    ax4.axvline(x=S0, color='black', linestyle='--', linewidth=2)
    ax4.axvline(x=traditional_paths[-1].mean(), color='red', linewidth=2)
    ax4.set_title(f'Traditional - Day {days} Distribution'); ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(ml_paths[-1], bins=50, alpha=0.7, edgecolor='black')
    ax5.axvline(x=S0, color='black', linestyle='--', linewidth=2)
    ax5.axvline(x=ml_paths[-1].mean(), color='darkgreen', linewidth=2)
    ax5.set_title(f'ML-Enhanced - Day {days} Distribution'); ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(ml_garch_paths[-1], bins=50, alpha=0.7, edgecolor='black')
    ax6.axvline(x=S0, color='black', linestyle='--', linewidth=2)
    ax6.axvline(x=ml_garch_paths[-1].mean(), color='purple', linewidth=2)
    ax6.set_title(f'ML+GARCH - Day {days} Distribution'); ax6.grid(True, alpha=0.3)

    ax7 = fig.add_subplot(gs[2, :])
    trad_pct = np.percentile(traditional_paths, [5, 50, 95], axis=1)
    ml_pct = np.percentile(ml_paths, [5, 50, 95], axis=1)
    garch_pct = np.percentile(ml_garch_paths, [5, 50, 95], axis=1)
    t = np.arange(days)

    ax7.fill_between(t, trad_pct[0], trad_pct[2], alpha=0.2, label='Traditional 90% CI')
    ax7.plot(trad_pct[1], 'b-', lw=2, label='Traditional Median')
    ax7.fill_between(t, ml_pct[0], ml_pct[2], alpha=0.2, color='green', label='ML-Enhanced 90% CI')
    ax7.plot(ml_pct[1], 'g-', lw=2, label='ML Median')
    ax7.fill_between(t, garch_pct[0], garch_pct[2], alpha=0.2, color='purple', label='ML+GARCH 90% CI')
    ax7.plot(garch_pct[1], color='purple', lw=2, label='ML+GARCH Median')
    ax7.axhline(y=S0, color='black', linestyle='--', label='Current Price')
    ax7.set_xlabel('Days'); ax7.set_ylabel('Price ($)'); ax7.set_title('Confidence Intervals - All Methods')
    ax7.legend(loc='best'); ax7.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
