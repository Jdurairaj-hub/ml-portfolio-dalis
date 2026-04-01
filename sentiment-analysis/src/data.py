#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

from .indicators import Candle

# Constants
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
_FUNDAMENTALS_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "cache", "fundamentals.json")


@dataclass
class SentimentSample:
    source: str
    title: str
    score: float
    ts: Optional[int] = None
    url: Optional[str] = None


@dataclass
class SignalResult:
    ticker: str
    technical_score: float
    sentiment_score: float
    reddit_sentiment: float
    news_sentiment: float
    mention_count: int
    hybrid_score: float
    action: str
    confidence: str
    mode: str
    reasons: List[str]
    indicators: Dict[str, Any]
    top_sources: List[Dict[str, Any]]
    top_reddit_sources: List[Dict[str, Any]]
    top_news_sources: List[Dict[str, Any]]
    debug: Dict[str, Any]


@dataclass
class SentimentSummary:
    source: str
    score: float
    count: int
    recent_1h_count: int
    recent_6h_count: int
    recency_weighted_count: float
    avg_age_hours: Optional[float]


@dataclass
class HybridConfig:
    use_finbert: bool = False
    event_sentiment_weight: float = 0.50
    event_technical_weight: float = 0.35
    event_fundamental_weight: float = 0.15
    normal_sentiment_weight: float = 0.25
    normal_technical_weight: float = 0.45
    normal_fundamental_weight: float = 0.30
    mention_bonus_scale: float = 0.20
    half_life_hours_reddit: float = 4.0
    half_life_hours_news: float = 6.0
    profile: str = "unified"
    event_min_score: float = 0.70
    disagreement_watch_gap: float = 0.30
    disagreement_high_override: float = 0.60
    gap_watch_threshold: float = 0.08
    rsi_watch_threshold: float = 70.0


def load_fundamentals_cache() -> Dict[str, Any]:
    try:
        with open(_FUNDAMENTALS_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_fundamentals_cache(cache: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(_FUNDAMENTALS_CACHE_PATH), exist_ok=True)
    with open(_FUNDAMENTALS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def safe_get(d: Dict[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def fetch_yahoo_candles(
    ticker: str,
    interval: str = "1d",
    range_: str = "6mo",
    timeout: int = 15,
    min_candles: int = 40,
) -> List[Candle]:
    from .indicators import _PRICE_CACHE  # Import here to avoid circular imports

    cache_key = (ticker.upper(), interval, range_)
    if cache_key in _PRICE_CACHE:
        return _PRICE_CACHE[cache_key]

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"interval": interval, "range": range_}
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    result = safe_get(data, "chart", "result", default=[])
    if not result:
        raise RuntimeError(f"No price data for {ticker}")
    result0 = result[0]
    timestamps = result0.get("timestamp") or []
    quote = safe_get(result0, "indicators", "quote", default=[{}])[0]
    candles: List[Candle] = []
    for i, ts in enumerate(timestamps):
        o = quote.get("open", [None])[i]
        h = quote.get("high", [None])[i]
        l = quote.get("low", [None])[i]
        c = quote.get("close", [None])[i]
        v = quote.get("volume", [None])[i]
        if None in (o, h, l, c, v):
            continue
        candles.append(Candle(ts=int(ts), open=float(o), high=float(h), low=float(l), close=float(c), volume=float(v)))
    if len(candles) < min_candles:
        raise RuntimeError(f"Insufficient candles for {ticker}: {len(candles)} (need {min_candles})")
    _PRICE_CACHE[cache_key] = candles
    return candles


def _extract_raw_metric(node: Any) -> Optional[float]:
    if node is None:
        return None
    if isinstance(node, (int, float)):
        return float(node)
    if isinstance(node, dict):
        raw = node.get("raw")
        if isinstance(raw, (int, float)):
            return float(raw)
    return None


def fetch_yahoo_fundamentals(ticker: str, timeout: int = 15) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
    params = {"modules": "defaultKeyStatistics,financialData,summaryDetail"}
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        result = safe_get(data, "quoteSummary", "result", default=[])
        if result:
            row = result[0]
            fdata = row.get("financialData", {}) if isinstance(row, dict) else {}
            kstats = row.get("defaultKeyStatistics", {}) if isinstance(row, dict) else {}
            sdetail = row.get("summaryDetail", {}) if isinstance(row, dict) else {}
            out = {
                "market_cap": _extract_raw_metric(sdetail.get("marketCap")),
                "forward_pe": _extract_raw_metric(sdetail.get("forwardPE")),
                "trailing_pe": _extract_raw_metric(sdetail.get("trailingPE")),
                "profit_margins": _extract_raw_metric(fdata.get("profitMargins")),
                "operating_margins": _extract_raw_metric(fdata.get("operatingMargins")),
                "debt_to_equity": _extract_raw_metric(fdata.get("debtToEquity")),
                "return_on_equity": _extract_raw_metric(fdata.get("returnOnEquity")),
                "current_ratio": _extract_raw_metric(fdata.get("currentRatio")),
                "quick_ratio": _extract_raw_metric(fdata.get("quickRatio")),
                "beta": _extract_raw_metric(kstats.get("beta")),
            }
    except Exception:
        # Some Yahoo endpoints intermittently return 401/empty; continue to fallbacks.
        out = {}

    quote_fb = {}
    yfin_fb = {}
    try:
        quote_fb = fetch_yahoo_fundamentals_quote_fallback(ticker, timeout=timeout)
    except Exception:
        quote_fb = {}
    try:
        yfin_fb = fetch_yfinance_fundamentals_fallback(ticker, timeout=timeout)
    except Exception:
        yfin_fb = {}

    merged = merge_fundamental_metrics(out, quote_fb)
    merged = merge_fundamental_metrics(merged, yfin_fb)
    return merged


def fetch_yahoo_fundamentals_quote_fallback(ticker: str, timeout: int = 15) -> Dict[str, Optional[float]]:
    url = "https://query1.finance.yahoo.com/v7/finance/quote"
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, params={"symbols": ticker}, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    row = safe_get(data, "quoteResponse", "result", default=[])
    if not row:
        return {}
    q = row[0]
    return {
        "market_cap": _extract_raw_metric(q.get("marketCap")),
        "forward_pe": _extract_raw_metric(q.get("forwardPE")),
        "trailing_pe": _extract_raw_metric(q.get("trailingPE")),
        "profit_margins": _extract_raw_metric(q.get("profitMargins")),
        "operating_margins": _extract_raw_metric(q.get("operatingMargins")),
        "debt_to_equity": _extract_raw_metric(q.get("debtToEquity")),
        "return_on_equity": _extract_raw_metric(q.get("returnOnEquity")),
        "current_ratio": _extract_raw_metric(q.get("currentRatio")),
        "quick_ratio": _extract_raw_metric(q.get("quickRatio")),
        "beta": _extract_raw_metric(q.get("beta")),
    }


def fetch_yfinance_fundamentals_fallback(ticker: str, timeout: int = 15) -> Dict[str, Optional[float]]:
    # Optional fallback path; if yfinance is unavailable, keep silent and return empty.
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return {}
    try:
        tk = yf.Ticker(ticker)
        info = tk.info if isinstance(tk.info, dict) else {}
        # yfinance handles its own HTTP stack; timeout may not be honored per-field.
        _ = timeout
    except Exception:
        return {}
    if not isinstance(info, dict):
        return {}
    return {
        "market_cap": _extract_raw_metric(info.get("marketCap")),
        "forward_pe": _extract_raw_metric(info.get("forwardPE")),
        "trailing_pe": _extract_raw_metric(info.get("trailingPE")),
        "profit_margins": _extract_raw_metric(info.get("profitMargins")),
        "operating_margins": _extract_raw_metric(info.get("operatingMargins")),
        "debt_to_equity": _extract_raw_metric(info.get("debtToEquity")),
        "return_on_equity": _extract_raw_metric(info.get("returnOnEquity")),
        "current_ratio": _extract_raw_metric(info.get("currentRatio")),
        "quick_ratio": _extract_raw_metric(info.get("quickRatio")),
        "beta": _extract_raw_metric(info.get("beta")),
    }


def merge_fundamental_metrics(primary: Dict[str, Optional[float]], secondary: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    if not primary:
        return dict(secondary or {})
    merged = dict(primary)
    for k, v in (secondary or {}).items():
        if merged.get(k) is None and v is not None:
            merged[k] = v
    return merged


def get_cached_fundamentals(ticker: str, timeout: int = 15) -> Dict[str, Any]:
    cache = load_fundamentals_cache()
    key = ticker.upper()
    today = time.strftime("%Y-%m-%d", time.gmtime())
    cached = cache.get(key)
    if isinstance(cached, dict) and cached.get("as_of") == today and isinstance(cached.get("metrics"), dict):
        metrics_cached = cached.get("metrics", {})
        if isinstance(metrics_cached, dict) and any(v is not None for v in metrics_cached.values()):
            return cached
    metrics = fetch_yahoo_fundamentals(ticker, timeout=timeout)
    record = {"as_of": today, "metrics": metrics}
    cache[key] = record
    save_fundamentals_cache(cache)
    return record