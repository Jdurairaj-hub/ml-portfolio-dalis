#!/usr/bin/env python3
"""
Google Trends vs Stock Price Correlation Analyzer

This script analyzes correlations between Google Trends interest data and stock prices
for specified keywords and tickers. It provides comprehensive correlation analysis
using multiple statistical methods and generates insightful visualizations.

Author: ML Portfolio Project
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from scipy import stats
from pytrends.request import TrendReq
import yfinance as yf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "keywords": ["AI", "tariffs"],
    "tickers": ["NAVBQ", "GOOG", "META", "WMT"],
    "timeframe": "2023-01-01 2025-11-17",
    "geo": "US",
    "cache_dir": "cache",
    "output_dir": "output",
    "min_periods": 10,  # Minimum periods for correlation
    "significance_level": 0.05,
}


class GoogleTrendsStockAnalyzer:
    """Main analyzer class for Google Trends vs Stock Price correlations."""

    def __init__(self, config: Dict = None):
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        # Create directories
        self.cache_dir = Path(self.config["cache_dir"])
        self.output_dir = Path(self.config["output_dir"])
        self.cache_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize clients
        self.pytrends = TrendReq(hl="en-US", tz=0)

        logger.info(f"Initialized analyzer with config: {self.config}")

    def get_cache_path(self, data_type: str, identifier: str) -> Path:
        """Get cache file path for given data type and identifier."""
        return (
            self.cache_dir
            / f"{data_type}_{identifier}_{self.config['timeframe'].replace(' ', '_')}.csv"
        )

    def fetch_google_trends_data(
        self, keywords: List[str], timeframe: str, geo: str, use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch Google Trends data with caching and error handling.

        Google Trends provides weekly data, so we get interest over time.
        """
        cache_key = f"trends_{'_'.join(keywords)}_{geo}"
        cache_path = self.get_cache_path("trends", cache_key)

        if use_cache and cache_path.exists():
            logger.info(f"Loading cached Google Trends data from {cache_path}")
            try:
                data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                # Ensure timezone-naive index
                if hasattr(data.index, "tz") and data.index.tz is not None:
                    data.index = data.index.tz_localize(None)
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        logger.info(f"Fetching Google Trends data for keywords: {keywords}")

        try:
            # Build payload
            self.pytrends.build_payload(keywords, timeframe=timeframe, geo=geo)

            # Get interest over time
            data = self.pytrends.interest_over_time()

            # Add random delay to be respectful to the API
            time.sleep(random.uniform(5, 10))

            # Clean data
            if "isPartial" in data.columns:
                data = data.drop(columns=["isPartial"])

            # Ensure we have data
            if data.empty:
                logger.error("No Google Trends data retrieved")
                return None

            # Cache the data
            data.to_csv(cache_path)
            logger.info(f"Cached Google Trends data to {cache_path}")

            return data

        except Exception as e:
            logger.error(f"Failed to fetch Google Trends data: {e}")
            return None

    def fetch_stock_data(
        self, tickers: List[str], use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch stock price data with caching and error handling.

        Returns daily closing prices resampled to weekly frequency to match Google Trends.
        """
        all_data = []

        for ticker in tickers:
            cache_path = self.get_cache_path("stock", ticker)

            if use_cache and cache_path.exists():
                logger.info(f"Loading cached stock data for {ticker}")
                try:
                    data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                    # Ensure timezone-naive index
                    if hasattr(data.index, "tz") and data.index.tz is not None:
                        data.index = data.index.tz_localize(None)
                    all_data.append(data)
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load cache for {ticker}: {e}")

            logger.info(f"Fetching stock data for {ticker}")

            try:
                # Download data
                stock = yf.Ticker(ticker)
                data = stock.history(period="2y", interval="1d")

                if data.empty:
                    logger.warning(f"No data available for {ticker}")
                    continue

                # Keep only closing prices and resample to weekly
                weekly_data = data["Close"].resample("W").last().to_frame()
                weekly_data.columns = [ticker]

                # Ensure timezone-naive index
                if weekly_data.index.tz is not None:
                    weekly_data.index = weekly_data.index.tz_localize(None)

                # Cache the data
                weekly_data.to_csv(cache_path)

                all_data.append(weekly_data)
                logger.info(
                    f"Successfully cached {len(weekly_data)} weeks of data for {ticker}"
                )

                # Be respectful to Yahoo Finance API
                time.sleep(random.uniform(1, 3))

            except Exception as e:
                logger.error(f"Failed to fetch data for {ticker}: {e}")
                continue

        if not all_data:
            raise ValueError("No stock data could be retrieved")

        # Combine all stock data
        combined = pd.concat(all_data, axis=1, join="outer", sort=True)
        combined = combined.sort_index()

        logger.info(f"Combined stock data shape: {combined.shape}")
        return combined

    def align_data_temporal(
        self, trends_data: pd.DataFrame, stock_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align Google Trends (weekly) and stock data (weekly) by date.

        Since both are now weekly, we can merge directly on date.
        """
        logger.info("Aligning temporal data...")

        # Debug: Check data info
        logger.info(f"Trends data shape: {trends_data.shape}, "
                    f"index type: {type(trends_data.index)}, "
                    f"tz: {trends_data.index.tz}")
        logger.info(f"Trends date range: {trends_data.index.min()} "
                    f"to {trends_data.index.max()}")
        logger.info(f"Stock data shape: {stock_data.shape}, "
                    f"index type: {type(stock_data.index)}, "
                    f"tz: {stock_data.index.tz}")
        logger.info(f"Stock date range: {stock_data.index.min()} "
                    f"to {stock_data.index.max()}")

        # Ensure both have datetime index and are timezone-naive
        trends_data.index = pd.to_datetime(trends_data.index)
        stock_data.index = pd.to_datetime(stock_data.index)

        # Remove timezone info if present
        if trends_data.index.tz is not None:
            trends_data.index = trends_data.index.tz_localize(None)
        if stock_data.index.tz is not None:
            stock_data.index = stock_data.index.tz_localize(None)

        # Merge on date (left join to keep all trend dates)
        merged = trends_data.merge(
            stock_data, left_index=True, right_index=True, how="left"
        )

        logger.info(f"After merge shape: {merged.shape}")

        # Forward fill missing stock data (if stock market was closed)
        merged = merged.ffill()

        # Remove rows where we don't have enough data
        # Require at least some valid data points, but be more lenient
        min_valid_points = max(
            3, len(merged.columns) // 2
        )  # At least 3 or half the columns
        merged = merged.dropna(thresh=min_valid_points)

        logger.info(f"Final aligned data shape: {merged.shape}")
        if len(merged) > 0:
            logger.info(f"Date range: {merged.index.min()} to {merged.index.max()}")

        return merged

    def calculate_correlations(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Calculate correlation matrices using different methods.

        Returns:
            Dict with keys: 'pearson', 'spearman', 'kendall'
        """
        logger.info("Calculating correlation matrices...")

        if data.empty:
            logger.warning("No data available for correlation analysis")
            # Return empty correlation matrices with proper structure
            columns = data.columns if not data.empty else []
            empty_corr = pd.DataFrame(index=columns, columns=columns)
            return {
                "pearson": empty_corr,
                "spearman": empty_corr,
                "kendall": empty_corr,
            }

        # Ensure numeric data
        numeric_data = data.select_dtypes(include=[np.number])

        correlations = {
            "pearson": numeric_data.corr(method="pearson"),
            "spearman": numeric_data.corr(method="spearman"),
            "kendall": numeric_data.corr(method="kendall"),
        }

        return correlations

    def test_correlation_significance(
        self, data: pd.DataFrame, method: str = "pearson"
    ) -> pd.DataFrame:
        """
        Test statistical significance of correlations using p-values.
        """
        logger.info("Testing correlation significance...")

        if data.empty or len(data) < 2:
            logger.warning("Insufficient data for significance testing")
            columns = data.columns if not data.empty else []
            return pd.DataFrame(index=columns, columns=columns)

        numeric_data = data.select_dtypes(include=[np.number])
        columns = numeric_data.columns

        significance_matrix = pd.DataFrame(index=columns, columns=columns)

        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i == j:
                    significance_matrix.loc[col1, col2] = (
                        1.0  # Perfect correlation with self
                    )
                else:
                    try:
                        # Get non-null values for both columns
                        valid_data = numeric_data[[col1, col2]].dropna()
                        if len(valid_data) < 2:
                            significance_matrix.loc[col1, col2] = np.nan
                        else:
                            corr, p_value = stats.pearsonr(
                                valid_data[col1], valid_data[col2]
                            )
                            significance_matrix.loc[col1, col2] = p_value
                    except Exception as e:
                        logger.warning(
                            f"Could not calculate significance for {col1} vs {col2}: {e}"
                        )
                        significance_matrix.loc[col1, col2] = np.nan

        return significance_matrix

    def create_correlation_heatmap(
        self,
        corr_matrix: pd.DataFrame,
        significance_matrix: pd.DataFrame,
        title: str,
        ax: plt.Axes,
    ) -> None:
        """Create enhanced correlation heatmap with significance indicators."""
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Create significance mask (show only significant correlations)
        sig_level = self.config["significance_level"]
        sig_mask = significance_matrix > sig_level

        # Combine masks
        final_mask = mask | sig_mask

        # Create heatmap
        sns.heatmap(
            corr_matrix,
            mask=final_mask,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
            ax=ax,
            vmin=-1,
            vmax=1,
            center=0,
        )

        ax.set_title(f"{title}\n(Only significant correlations shown, α={sig_level})")

    def plot_correlation_analysis(
        self, correlations: Dict[str, pd.DataFrame], significance: pd.DataFrame
    ) -> None:
        """Create comprehensive correlation visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Correlation heatmaps
        self.create_correlation_heatmap(
            correlations["pearson"], significance, "Pearson Correlation", axes[0, 0]
        )
        self.create_correlation_heatmap(
            correlations["spearman"], significance, "Spearman Correlation", axes[0, 1]
        )
        self.create_correlation_heatmap(
            correlations["kendall"], significance, "Kendall Correlation", axes[1, 0]
        )

        # Summary statistics
        ax_stats = axes[1, 1]
        ax_stats.axis("off")

        # Calculate key insights
        keywords = self.config["keywords"]
        tickers = self.config["tickers"]

        stats_text = "Analysis Summary:\n\n"
        stats_text += f"Time Period: {self.config['timeframe']}\n"
        stats_text += f"Keywords: {', '.join(keywords)}\n"
        stats_text += f"Stocks: {', '.join(tickers)}\n\n"

        # Find strongest correlations
        pearson_corr = correlations["pearson"]
        max_corr = 0
        max_pair = None

        for i in keywords:
            for j in tickers:
                if i in pearson_corr.index and j in pearson_corr.columns:
                    corr_val = abs(pearson_corr.loc[i, j])
                    if corr_val > max_corr:
                        max_corr = corr_val
                        max_pair = (i, j)

        if max_pair:
            stats_text += f"Strongest correlation: {max_pair[0]} vs {max_pair[1]}\n"
            stats_text += (
                f"Correlation: {pearson_corr.loc[max_pair[0], max_pair[1]]:.3f}\n"
            )

        ax_stats.text(
            0.1,
            0.8,
            stats_text,
            transform=ax_stats.transAxes,
            fontsize=10,
            verticalalignment="top",
        )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "correlation_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

    def analyze_trends_vs_stocks(self, use_cache: bool = True) -> Dict:
        """
        Main analysis pipeline: fetch data, align, correlate, and visualize.
        """
        logger.info("Starting Google Trends vs Stock Price Analysis")

        # Fetch data
        trends_data = self.fetch_google_trends_data(
            self.config["keywords"],
            self.config["timeframe"],
            self.config["geo"],
            use_cache=use_cache,
        )

        if trends_data is None:
            raise ValueError("Failed to fetch Google Trends data")

        stock_data = self.fetch_stock_data(self.config["tickers"], use_cache=use_cache)

        # Align data temporally
        aligned_data = self.align_data_temporal(trends_data, stock_data)

        # Save aligned data
        aligned_data.to_csv(self.output_dir / "aligned_data.csv")

        # Calculate correlations
        correlations = self.calculate_correlations(aligned_data)

        # Test significance
        significance = self.test_correlation_significance(aligned_data)

        # Generate visualizations
        self.plot_correlation_analysis(correlations, significance)

        # Save correlation matrices
        for method, matrix in correlations.items():
            matrix.to_csv(self.output_dir / f"correlation_{method}.csv")

        significance.to_csv(self.output_dir / "correlation_significance.csv")

        logger.info("Analysis complete! Results saved to output directory.")

        return {
            "data": aligned_data,
            "correlations": correlations,
            "significance": significance,
            "config": self.config,
        }


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Analyze correlations between Google Trends and stock prices"
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=DEFAULT_CONFIG["keywords"],
        help="Google Trends keywords to analyze",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_CONFIG["tickers"],
        help="Stock tickers to analyze",
    )
    parser.add_argument(
        "--timeframe",
        default=DEFAULT_CONFIG["timeframe"],
        help="Time period for analysis (YYYY-MM-DD YYYY-MM-DD)",
    )
    parser.add_argument(
        "--geo",
        default=DEFAULT_CONFIG["geo"],
        help="Geographic region for Google Trends",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable caching and fetch fresh data"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_CONFIG["output_dir"],
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Update config from args
    config = DEFAULT_CONFIG.copy()
    config.update(
        {
            "keywords": args.keywords,
            "tickers": args.tickers,
            "timeframe": args.timeframe,
            "geo": args.geo,
            "output_dir": args.output_dir,
        }
    )

    # Run analysis
    analyzer = GoogleTrendsStockAnalyzer(config)
    results = analyzer.analyze_trends_vs_stocks(use_cache=not args.no_cache)

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Keywords analyzed: {', '.join(config['keywords'])}")
    print(f"Stocks analyzed: {', '.join(config['tickers'])}")
    print(f"Time period: {config['timeframe']}")
    print(f"Data points: {len(results['data'])}")
    print(f"Output saved to: {config['output_dir']}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
