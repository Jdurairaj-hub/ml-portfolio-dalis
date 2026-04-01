#!/usr/bin/env python3

"""
Hybrid Technical + Social Sentiment Signal Scanner

A comprehensive trading signal generator that combines:
- Technical analysis (RSI, MACD, volume, breakouts)
- Social sentiment (Reddit mentions, news articles)
- Fundamental analysis (P/E, ROE, debt ratios)
- Machine learning-based decision making

For educational and research purposes only.
"""

__version__ = "1.0.0"
__author__ = "Admin"
__license__ = "MIT"

from .data import (
    SentimentSample,
    SignalResult,
    SentimentSummary,
    HybridConfig,
)
from .indicators import (
    Candle,
    ema,
    rsi,
    sma,
    atr,
    vwma,
    zscore_current,
)
from .sentiment import (
    text_sentiment_score,
    finbert_sentiment_score,
    sentiment_score_for_text,
    fetch_reddit_mentions,
    google_news_rss_query,
    summarize_sentiment,
)
from .scanner import (
    scan_ticker,
    hybrid_decision,
    company_name_map,
)
from .utils import (
    now_utc_iso,
    clamp,
    render_console_table,
)

__all__ = [
    # Data classes
    "SentimentSample",
    "SignalResult",
    "SentimentSummary",
    "HybridConfig",
    "Candle",

    # Indicators
    "ema",
    "rsi",
    "sma",
    "atr",
    "vwma",
    "zscore_current",

    # Sentiment
    "text_sentiment_score",
    "finbert_sentiment_score",
    "sentiment_score_for_text",
    "fetch_reddit_mentions",
    "google_news_rss_query",
    "summarize_sentiment",

    # Scanner
    "scan_ticker",
    "hybrid_decision",
    "company_name_map",

    # Utils
    "now_utc_iso",
    "clamp",
    "render_console_table",
]