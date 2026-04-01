#!/usr/bin/env python3
"""
ML-Based Long/Flat Trading Strategy for SPY

A formalized machine learning approach to predict next-day market direction
using technical indicators and logistic regression. Includes comprehensive
backtesting, risk metrics, and performance analysis.

Author: ML Portfolio Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD, SMAIndicator
import joblib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Default configuration
DEFAULT_CONFIG = {
    'ticker': 'SPY',
    'period': '5y',
    'test_size': 0.2,
    'random_state': 42,
    'model_params': {
        'max_iter': 1000,
        'random_state': 42,
        'C': 1.0
    },
    'features': [
        'ret_1d', 'ret_5d', 'ret_10d',  # Returns
        'rsi_14',  # RSI
        'vol_10d', 'vol_20d',  # Volatility
        'bb_upper', 'bb_lower', 'bb_middle',  # Bollinger Bands
        'macd', 'macd_signal', 'macd_diff',  # MACD
        'sma_20', 'sma_50'  # Moving averages
    ],
    'output_dir': 'output',
    'model_dir': 'models'
}


class MLTradingStrategy:
    """ML-based long/flat trading strategy with comprehensive backtesting."""

    def __init__(self, config: Dict = None):
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self._update_config(config)

        # Create directories
        self.output_dir = Path(self.config['output_dir'])
        self.model_dir = Path(self.config['model_dir'])
        self.output_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)

        # Initialize components
        self.data = None
        self.features = None
        self.target = None
        self.model = None
        self.scaler = None
        self.feature_cols = None

        logger.info(f"Initialized ML Trading Strategy for {self.config['ticker']}")

    def _update_config(self, config: Dict) -> None:
        """Recursively update configuration."""
        for key, value in config.items():
            if isinstance(value, dict) and key in self.config:
                self.config[key].update(value)
            else:
                self.config[key] = value

    def download_data(self, ticker: str = None, period: str = None) -> pd.DataFrame:
        """
        Download historical data from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol
            period: Time period (1y, 2y, 5y, etc.)

        Returns:
            DataFrame with OHLCV data
        """
        ticker = ticker or self.config['ticker']
        period = period or self.config['period']

        logger.info(f"Downloading {period} of data for {ticker}")

        try:
            data = yf.download(ticker, period=period, progress=False)

            if data.empty:
                raise ValueError(f"No data available for {ticker}")

            # Fix multi-index columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [c[0] for c in data.columns]

            # Ensure we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            logger.info(f"Downloaded {len(data)} days of data from {data.index[0].date()} to {data.index[-1].date()}")
            return data

        except Exception as e:
            logger.error(f"Failed to download data for {ticker}: {e}")
            raise

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators and features for ML model.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with additional feature columns
        """
        logger.info("Creating technical features...")

        df = data.copy()

        # Basic returns
        df['ret_1d'] = df['Close'].pct_change(1)
        df['ret_5d'] = df['Close'].pct_change(5)
        df['ret_10d'] = df['Close'].pct_change(10)

        # RSI
        rsi_indicator = RSIIndicator(df['Close'], window=14)
        df['rsi_14'] = rsi_indicator.rsi()

        # Volatility
        df['vol_10d'] = df['ret_1d'].rolling(10).std()
        df['vol_20d'] = df['ret_1d'].rolling(20).std()

        # Bollinger Bands
        bb_indicator = BollingerBands(df['Close'], window=20, window_dev=2)
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_lower'] = bb_indicator.bollinger_lband()
        df['bb_middle'] = bb_indicator.bollinger_mavg()

        # MACD
        macd_indicator = MACD(df['Close'])
        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        df['macd_diff'] = macd_indicator.macd_diff()

        # Moving Averages
        sma_20 = SMAIndicator(df['Close'], window=20)
        sma_50 = SMAIndicator(df['Close'], window=50)
        df['sma_20'] = sma_20.sma_indicator()
        df['sma_50'] = sma_50.sma_indicator()

        # Target variable: next day return direction
        df['ret_fwd_1d'] = df['Close'].pct_change().shift(-1)
        df['target'] = (df['ret_fwd_1d'] > 0).astype(int)

        logger.info(f"Created {len(self.config['features'])} features")
        return df

    def prepare_data(self, data: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for model training.

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data available. Run download_data() and create_features() first.")
            data = self.data

        # Select features and target
        feature_cols = [col for col in self.config['features'] if col in data.columns]
        self.feature_cols = feature_cols  # Store for later use
        missing_features = [col for col in self.config['features'] if col not in data.columns]

        if missing_features:
            logger.warning(f"Missing features: {missing_features}")

        # Create clean dataset
        df_clean = data[feature_cols + ['target']].dropna()

        if len(df_clean) == 0:
            raise ValueError("No valid data after feature engineering")

        X = df_clean[feature_cols]
        y = df_clean['target']

        logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")

        return X, y

    def train_model(self, X: pd.DataFrame = None, y: pd.Series = None,
                   save_model: bool = True) -> LogisticRegression:
        """
        Train the logistic regression model with time series cross-validation.

        Args:
            X: Feature matrix
            y: Target vector
            save_model: Whether to save the trained model

        Returns:
            Trained model
        """
        if X is None or y is None:
            X, y = self.prepare_data()

        logger.info("Training logistic regression model...")

        # Split data (no shuffle for time series)
        split_idx = int(len(X) * (1 - self.config['test_size']))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model = LogisticRegression(**self.config['model_params'])
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        # Predictions for detailed metrics
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        logger.info(f"Training accuracy: {train_score:.4f}")
        logger.info(f"Test accuracy: {test_score:.4f}")
        logger.info(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")

        # Store feature data for later use
        self.X = X
        self.y = y

        # Save model if requested
        if save_model:
            self.save_model()

        return self.model

    def cross_validate(self, X: pd.DataFrame = None, y: pd.Series = None,
                      n_splits: int = 5) -> Dict:
        """
        Perform time series cross-validation.

        Args:
            X: Feature matrix
            y: Target vector
            n_splits: Number of CV splits

        Returns:
            Dictionary with CV results
        """
        if X is None or y is None:
            X, y = self.prepare_data()

        logger.info(f"Performing {n_splits}-fold time series cross-validation...")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        scaler = StandardScaler()

        cv_scores = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
            y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

            # Scale
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_test_scaled = scaler.transform(X_test_fold)

            # Train and score
            model = LogisticRegression(**self.config['model_params'])
            model.fit(X_train_scaled, y_train_fold)
            score = model.score(X_test_scaled, y_test_fold)
            cv_scores.append(score)
            logger.info(f"Fold {fold + 1}: {score:.4f}")

        cv_results = {
            'cv_scores': cv_scores,
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores)
        }

        logger.info(f"CV Mean: {cv_results['mean_cv_score']:.4f} (+/- {cv_results['std_cv_score']:.4f})")

        return cv_results

    def predict_signal(self, latest_data: pd.DataFrame = None,
                      threshold: float = 0.55) -> Dict:
        """
        Generate trading signal for the next day.

        Args:
            latest_data: Latest market data (if None, uses most recent from training data)
            threshold: Probability threshold for long signal

        Returns:
            Dictionary with signal information
        """
        if self.model is None:
            raise ValueError("Model not trained. Run train_model() first.")

        if latest_data is None:
            # Use most recent data from training set
            if self.data is None:
                raise ValueError("No data available for prediction")
            latest_features = self.data[self.feature_cols].iloc[[-1]]
        else:
            # Use provided data
            latest_features = latest_data[self.feature_cols].iloc[[-1]]

        # Scale features
        latest_scaled = self.scaler.transform(latest_features)

        # Predict
        probability_up = self.model.predict_proba(latest_scaled)[0, 1]
        prediction = self.model.predict(latest_scaled)[0]

        signal = "LONG" if probability_up > threshold else "FLAT"

        result = {
            'date': latest_features.index[-1].date() if hasattr(latest_features.index[-1], 'date')
                   else latest_features.index[-1],
            'probability_up': probability_up,
            'prediction': prediction,
            'signal': signal,
            'threshold': threshold,
            'ticker': self.config['ticker']
        }

        logger.info(f"Generated signal: {signal} (P(up)={probability_up:.4f})")

        return result

    def backtest_strategy(self, X: pd.DataFrame = None, y: pd.Series = None,
                         initial_capital: float = 10000) -> Dict:
        """
        Backtest the strategy on historical data.

        Args:
            X: Feature matrix
            y: Target vector
            initial_capital: Starting capital

        Returns:
            Dictionary with backtest results
        """
        if X is None or y is None:
            X, y = self.prepare_data()

        if self.model is None:
            raise ValueError("Model not trained. Run train_model() first.")

        logger.info("Running strategy backtest...")

        # Get predictions for the entire dataset
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        # Calculate returns - ensure proper alignment
        actual_returns = self.data.loc[X.index, 'ret_fwd_1d']

        # Strategy returns (only take positions when predicted up)
        strategy_returns = predictions * actual_returns

        # Handle NaN values in returns
        strategy_returns = strategy_returns.fillna(0)
        actual_returns = actual_returns.fillna(0)

        # Calculate cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod()
        buy_hold_returns = (1 + actual_returns).cumprod()

        # Performance metrics
        total_return = cumulative_returns.iloc[-1] - 1
        buy_hold_return = buy_hold_returns.iloc[-1] - 1
        volatility = strategy_returns.std() * np.sqrt(252)  # Annualized
        sharpe_ratio = total_return / volatility if volatility > 0 else 0

        # Maximum drawdown
        cumulative = cumulative_returns
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        winning_trades = (strategy_returns > 0).sum()
        total_trades = (predictions == 1).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        results = {
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'cumulative_returns': cumulative_returns,
            'buy_hold_cumulative': buy_hold_returns,
            'predictions': predictions,
            'probabilities': probabilities
        }

        logger.info(f"Backtest Results:")
        logger.info(f"  Total Return: {total_return:.2%}")
        logger.info(f"  Buy & Hold: {buy_hold_return:.2%}")
        logger.info(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"  Max Drawdown: {max_drawdown:.2%}")
        logger.info(f"  Win Rate: {win_rate:.2%}")

        return results

    def plot_backtest_results(self, backtest_results: Dict) -> None:
        """Create comprehensive backtest visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Cumulative returns
        axes[0, 0].plot(backtest_results['cumulative_returns'], label='Strategy', linewidth=2)
        axes[0, 0].plot(backtest_results['buy_hold_cumulative'], label='Buy & Hold', alpha=0.7)
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Returns distribution
        strategy_returns = backtest_results['cumulative_returns'].pct_change().dropna()
        axes[0, 1].hist(strategy_returns, bins=50, alpha=0.7, label='Strategy')
        buy_hold_returns = backtest_results['buy_hold_cumulative'].pct_change().dropna()
        axes[0, 1].hist(buy_hold_returns, bins=50, alpha=0.7, label='Buy & Hold')
        axes[0, 1].set_title('Returns Distribution')
        axes[0, 1].legend()

        # Prediction probabilities over time
        prob_series = pd.Series(backtest_results['probabilities'],
                               index=self.X.index[-len(backtest_results['probabilities']):])
        axes[1, 0].plot(prob_series.index, prob_series.values, alpha=0.7)
        axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
        axes[1, 0].set_title('Prediction Probabilities Over Time')
        axes[1, 0].legend()

        # Performance metrics table
        axes[1, 1].axis('off')
        metrics_text = ".2%"
        metrics_text += f"Buy & Hold Return: {backtest_results['buy_hold_return']:.2%}\n"
        metrics_text += f"Volatility: {backtest_results['volatility']:.2%}\n"
        metrics_text += f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}\n"
        metrics_text += f"Max Drawdown: {backtest_results['max_drawdown']:.2%}\n"
        metrics_text += f"Win Rate: {backtest_results['win_rate']:.2%}\n"
        metrics_text += f"Total Trades: {backtest_results['total_trades']}"

        axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_model(self, filename: str = None) -> None:
        """Save trained model and scaler."""
        if self.model is None:
            raise ValueError("No model to save")

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ml_trading_model_{self.config['ticker']}_{timestamp}.joblib"

        model_path = self.model_dir / filename

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'config': self.config,
            'training_date': datetime.now()
        }

        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")

    def load_model(self, filepath: str) -> None:
        """Load trained model and scaler."""
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_cols = model_data['feature_cols']
        self.config = model_data['config']

        logger.info(f"Model loaded from {filepath}")

    def run_full_analysis(self) -> Dict:
        """Run complete analysis pipeline."""
        logger.info("Starting full ML trading strategy analysis...")

        # Download and process data
        self.data = self.download_data()
        self.data = self.create_features(self.data)

        # Prepare data
        X, y = self.prepare_data()

        # Cross-validation
        cv_results = self.cross_validate(X, y)

        # Train final model
        self.train_model(X, y)

        # Backtest
        backtest_results = self.backtest_strategy(X, y)

        # Generate current signal
        current_signal = self.predict_signal()

        # Create visualizations
        self.plot_backtest_results(backtest_results)

        results = {
            'cv_results': cv_results,
            'backtest_results': backtest_results,
            'current_signal': current_signal,
            'config': self.config
        }

        logger.info("Analysis complete!")
        return results


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description="ML-based long/flat trading strategy for stock market prediction"
    )
    parser.add_argument('--ticker', default='SPY',
                       help='Stock ticker symbol (default: SPY)')
    parser.add_argument('--period', default='5y',
                       help='Historical data period (default: 5y)')
    parser.add_argument('--threshold', type=float, default=0.55,
                       help='Signal threshold probability (default: 0.55)')
    parser.add_argument('--output-dir', default='output',
                       help='Output directory for results')
    parser.add_argument('--no-backtest', action='store_true',
                       help='Skip backtesting')
    parser.add_argument('--save-model', action='store_true',
                       help='Save trained model')

    args = parser.parse_args()

    # Update config
    config = {
        'ticker': args.ticker,
        'period': args.period,
        'output_dir': args.output_dir
    }

    # Run analysis
    strategy = MLTradingStrategy(config)
    results = strategy.run_full_analysis()

    # Print current signal
    signal = results['current_signal']
    print("\n" + "="*60)
    print("🎯 ML TRADING SIGNAL - DALIS PORTFOLIO")
    print("="*60)
    print(f"Date: {datetime.today().strftime('%Y-%m-%d')}")
    print(f"Ticker: {signal['ticker']}")
    print(f"Signal: {signal['signal']}")
    print(f"Probability Market Up: {signal['probability_up']:.4f}")
    print(f"Threshold: {signal['threshold']}")
    print("="*60)

    # Print backtest summary
    if not args.no_backtest:
        bt = results['backtest_results']
        print("\n📊 BACKTEST SUMMARY:")
        print(f"Strategy Return: {bt['total_return']:.2%}")
        print(f"Buy & Hold Return: {bt['buy_hold_return']:.2%}")
        print(f"Sharpe Ratio: {bt['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {bt['max_drawdown']:.2%}")
        print("="*60)

    # Save model if requested
    if args.save_model:
        strategy.save_model()


if __name__ == "__main__":
    main()