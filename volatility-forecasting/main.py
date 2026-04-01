import os
import sys
import numpy as np
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # Load .env variables

from src.data import fetch_polygon_data
from src.features import create_features, create_advanced_features, calculate_risk_metrics, plot_monte_carlo_comparison_3methods
from src.model import train_ml_model
from src.garch import fit_garch_model, forecast_garch_volatility
from src.simulation import monte_carlo_traditional, monte_carlo_ml_enhanced, monte_carlo_ml_garch

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    TICKER = os.environ.get('TICKER', 'NVDA')
    API_KEY = os.environ.get('POLYGON_API_KEY', None)
    DAYS_TO_FETCH = int(os.environ.get('DAYS_TO_FETCH', 730))
    DAYS_TO_SIMULATE = int(os.environ.get('DAYS_TO_SIMULATE', 30))
    NUM_SIMULATIONS = int(os.environ.get('NUM_SIMULATIONS', 10000))
    RANDOM_SEED = int(os.environ.get('RANDOM_SEED', 42))

    print("="*70)
    print("ML + GARCH ENHANCED MONTE CARLO SIMULATION")
    print("="*70)

    print(f"\n[1] Fetching {TICKER} data from Polygon API...")
    try:
        df = fetch_polygon_data(TICKER, API_KEY, days_back=DAYS_TO_FETCH)
    except Exception as e:
        print("Failed to fetch data:", e)
        sys.exit(1)

    print(f"✓ Loaded {len(df)} days of data")

    print("\n[2] Creating technical features...")
    data = create_features(df)
    print(f"✓ Created features. Data shape: {data.shape}")

    print("\n[3] Training Machine Learning model...")
    model, scaler, feature_cols, X_test_scaled, y_test, data_enhanced = train_ml_model(data)
    print("✓ Model trained successfully")

    data = data_enhanced

    print("\n[4] Making ML prediction for next day...")
    current_price = df['close'].iloc[-1]
    if data.shape[0] == 0:
        raise RuntimeError("No data available after feature engineering.")

    current_features = data[feature_cols].iloc[-1].values.reshape(1, -1)
    current_features_scaled = scaler.transform(current_features)
    ml_prediction_log_return = model.predict(current_features_scaled)[0]
    print(f"Current Price: ${current_price:.2f}")
    print(f"ML Predicted Next-Day Log-Return: {ml_prediction_log_return:.6f} (≈ {ml_prediction_log_return*100:.4f}%)")

    print("\n[5] Calculating historical statistics (log returns)...")
    historical_log_returns = data['log_return'].dropna()
    mu_historical = historical_log_returns.mean()
    sigma_historical = historical_log_returns.std()
    print(f"Historical Mean Log-Return (daily): {mu_historical:.6e}")
    print(f"Historical Volatility (daily std of log returns): {sigma_historical:.6e}")

    print("\n[6] Fitting GARCH model for volatility forecasting...")
    try:
        garch_model = fit_garch_model(data['log_return'])
    except Exception as e:
        print("GARCH fit failed:", e)
        garch_volatilities = np.repeat(sigma_historical, DAYS_TO_SIMULATE)
        garch_model = None
        print("✓ Using historical volatility as fallback.")
    else:
        print("✓ GARCH fitted.")
        try:
            print("   Omega (constant):", garch_model.params.get('omega', np.nan))
            print("   Alpha (ARCH term):", garch_model.params.get('alpha[1]', np.nan))
            print("   Beta (GARCH term):", garch_model.params.get('beta[1]', np.nan))
        except Exception:
            pass

        garch_volatilities = forecast_garch_volatility(garch_model, DAYS_TO_SIMULATE)
        print(f"✓ Forecasted volatility for {DAYS_TO_SIMULATE} days (first day): {garch_volatilities[0]*100:.2f}%")

    print(f"\n[7] Running Traditional Monte Carlo ({NUM_SIMULATIONS:,} simulations)...")
    np.random.seed(RANDOM_SEED)
    traditional_paths = monte_carlo_traditional(current_price, mu_historical, sigma_historical, DAYS_TO_SIMULATE, NUM_SIMULATIONS)
    print("✓ Traditional simulation complete")

    print(f"\n[8] Running ML-Enhanced Monte Carlo ({NUM_SIMULATIONS:,} simulations)...")
    np.random.seed(RANDOM_SEED)
    ml_paths = monte_carlo_ml_enhanced(current_price, ml_prediction_log_return, sigma_historical, DAYS_TO_SIMULATE, NUM_SIMULATIONS)
    print("✓ ML-Enhanced simulation complete")

    print(f"\n[9] Running ML+GARCH Monte Carlo ({NUM_SIMULATIONS:,} simulations)...")
    np.random.seed(RANDOM_SEED)
    ml_garch_paths = monte_carlo_ml_garch(current_price, ml_prediction_log_return, garch_volatilities, DAYS_TO_SIMULATE, NUM_SIMULATIONS)
    print("✓ ML+GARCH simulation complete")

    print("\n[10] Calculating risk metrics...")
    trad_metrics = calculate_risk_metrics(traditional_paths, current_price)
    ml_metrics = calculate_risk_metrics(ml_paths, current_price)
    garch_metrics = calculate_risk_metrics(ml_garch_paths, current_price)

    print("\n" + "="*90)
    print("RISK METRICS COMPARISON - ALL THREE METHODS")
    print("="*90)
    print(f"{'Metric':<34} {'Traditional':>18} {'ML-Enhanced':>18} {'ML+GARCH':>18}")
    print("-"*90)

    def fmt_pct(x): return f"{x*100:>17.3f}%"
    def fmt_price(x): return f"${x:>17.2f}"

    print(f"{'Expected Log Return':<34} {fmt_pct(trad_metrics['Expected Log Return'])} {fmt_pct(ml_metrics['Expected Log Return'])} {fmt_pct(garch_metrics['Expected Log Return'])}")
    print(f"{'Volatility (Std of Log Returns)':<34} {fmt_pct(trad_metrics['Volatility (Std of Log Returns)'])} {fmt_pct(ml_metrics['Volatility (Std of Log Returns)'])} {fmt_pct(garch_metrics['Volatility (Std of Log Returns)'])}")
    print(f"{'VaR (95%) (log-return)':<34} {fmt_pct(trad_metrics['VaR (95%)'])} {fmt_pct(ml_metrics['VaR (95%)'])} {fmt_pct(garch_metrics['VaR (95%)'])}")
    print(f"{'Prob of Profit (final)':<34} {ml_metrics['Prob of Profit']*100:>17.2f}% {ml_metrics['Prob of Profit']*100:>18.2f}% {garch_metrics['Prob of Profit']*100:>18.2f}%")
    print(f"{'Expected Price (final)':<34} {fmt_price(trad_metrics['Expected Price'])} {fmt_price(ml_metrics['Expected Price'])} {fmt_price(garch_metrics['Expected Price'])}")
    print(f"{'Median Price (final)':<34}   {fmt_price(trad_metrics['Median Price'])} {fmt_price(ml_metrics['Median Price'])} {fmt_price(garch_metrics['Median Price'])}")
    print("="*90)

    print("\n[11] Creating visualizations...")
    fig = plot_monte_carlo_comparison_3methods(traditional_paths, ml_paths, ml_garch_paths, current_price, TICKER)
    
    # Save plot to file in output directory (or override with OUTPUT_DIR env var)
    output_dir = os.environ.get('OUTPUT_DIR', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"monte_carlo_{TICKER}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved as: {output_file}")
    
    # Optional: still show if in interactive environment
    try:
        fig.show()
        print("✓ Plot displayed (close window to continue)")
        plt.pause(1)  # Brief pause to let window appear
        input("Press Enter to exit...")  # Wait for user input
    except:
        print("✓ Plot saved (interactive display not available)")

    print("\n✓ Analysis complete!")
