#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import io
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

REDDIT_ALLOWED_SUBREDDITS = {"stocks", "investing", "wallstreetbets", "options", "stockmarket"}
REDDIT_MAX_AGE_HOURS = 120.0  # strict recency cap (~5 days)
NEWS_MAX_AGE_HOURS = 24.0 * 30.0  # ~30 days

POSITIVE_WORDS = {
    "beat",
    "beats",
    "bullish",
    "buy",
    "breakout",
    "surge",
    "surges",
    "upside",
    "upgrade",
    "strong",
    "record",
    "growth",
    "profit",
    "profits",
    "partnership",
    "partner",
    "launch",
    "expands",
    "expansion",
    "positive",
    "outperform",
    "momentum",
    "guidance",
    "raise",
    "raised",
    "rebound",
    "hot",
}

NEGATIVE_WORDS = {
    "miss",
    "misses",
    "bearish",
    "sell",
    "downgrade",
    "drop",
    "drops",
    "plunge",
    "plunges",
    "lawsuit",
    "weak",
    "decline",
    "declines",
    "cuts",
    "cut",
    "layoff",
    "layoffs",
    "fraud",
    "risk",
    "negative",
    "slowdown",
    "warning",
    "warns",
    "missed",
    "underperform",
    "crash",
    "crashes",
}

INTENSIFIERS = {"very", "strongly", "significantly", "massively", "huge", "major", "record"}
NOISE_TOKENS = {
    "stock",
    "shares",
    "share",
    "today",
    "news",
    "analysis",
    "update",
    "market",
    "markets",
    "company",
    "inc",
    "corp",
    "earnings",
}


@dataclass
class Candle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float


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


def _title_token_set(text: str) -> set[str]:
    t = text.lower()
    # Remove synthetic source prefixes like "r/stocks: ".
    t = re.sub(r"^r/[a-z0-9_+-]+:\s*", "", t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"[^a-z0-9$ ]+", " ", t)
    toks = [w for w in t.split() if len(w) >= 2 and w not in NOISE_TOKENS]
    return set(toks)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return (inter / union) if union else 0.0


def dedupe_sentiment_samples(
    samples: List[SentimentSample],
    sim_threshold: float = 0.70,
) -> Tuple[List[SentimentSample], Dict[str, Any]]:
    # Newest-first so we keep freshest copy of a repeated story.
    ordered = sorted(samples, key=lambda s: (s.ts or 0), reverse=True)
    kept: List[SentimentSample] = []
    kept_token_sets: List[set[str]] = []
    dropped = 0
    for s in ordered:
        ts = _title_token_set(s.title)
        is_dup = False
        for prev in kept_token_sets:
            if _jaccard(ts, prev) >= sim_threshold:
                is_dup = True
                break
        if is_dup:
            dropped += 1
            continue
        kept.append(s)
        kept_token_sets.append(ts)
    stats = {
        "raw": len(samples),
        "kept": len(kept),
        "dropped_duplicates": dropped,
        "sim_threshold": sim_threshold,
    }
    return kept, stats


_FINBERT_MODEL = None
_FINBERT_TOKENIZER = None
_FINBERT_LOAD_ERROR: Optional[str] = None
_PRICE_CACHE: Dict[Tuple[str, str, str], List["Candle"]] = {}


def now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def today_utc_str() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


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


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def ema(values: List[float], period: int) -> List[Optional[float]]:
    if period <= 0:
        raise ValueError("period must be > 0")
    out: List[Optional[float]] = [None] * len(values)
    if len(values) < period:
        return out
    alpha = 2 / (period + 1)
    sma = sum(values[:period]) / period
    out[period - 1] = sma
    prev = sma
    for i in range(period, len(values)):
        prev = (values[i] - prev) * alpha + prev
        out[i] = prev
    return out


def rsi(values: List[float], period: int = 14) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(values)
    if len(values) <= period:
        return out
    gains = []
    losses = []
    for i in range(1, period + 1):
        delta = values[i] - values[i - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    out[period] = 100 if avg_loss == 0 else 100 - (100 / (1 + (avg_gain / avg_loss)))
    for i in range(period + 1, len(values)):
        delta = values[i] - values[i - 1]
        gain = max(delta, 0)
        loss = max(-delta, 0)
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
        out[i] = 100 if avg_loss == 0 else 100 - (100 / (1 + (avg_gain / avg_loss)))
    return out


def sma(values: List[float], period: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(values)
    if period <= 0:
        return out
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= period:
            running -= values[i - period]
        if i >= period - 1:
            out[i] = running / period
    return out


def atr(candles: List[Candle], period: int = 14) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(candles)
    if len(candles) <= period:
        return out
    trs: List[float] = []
    for i, c in enumerate(candles):
        if i == 0:
            tr = c.high - c.low
        else:
            pc = candles[i - 1].close
            tr = max(c.high - c.low, abs(c.high - pc), abs(c.low - pc))
        trs.append(tr)
    first = sum(trs[1 : period + 1]) / period
    out[period] = first
    prev = first
    for i in range(period + 1, len(candles)):
        prev = ((prev * (period - 1)) + trs[i]) / period
        out[i] = prev
    return out


def vwma(values: List[float], volumes: List[float], period: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(values)
    pv_sum = 0.0
    v_sum = 0.0
    for i, (p, v) in enumerate(zip(values, volumes)):
        pv_sum += p * v
        v_sum += v
        if i >= period:
            pv_sum -= values[i - period] * volumes[i - period]
            v_sum -= volumes[i - period]
        if i >= period - 1 and v_sum > 0:
            out[i] = pv_sum / v_sum
    return out


def zscore_current(values: List[float]) -> float:
    if len(values) < 3:
        return 0.0
    mu = statistics.mean(values)
    sigma = statistics.pstdev(values) or 1e-9
    return (values[-1] - mu) / sigma


def text_sentiment_score(text: str) -> float:
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return 0.0
    score = 0.0
    for i, w in enumerate(words):
        mult = 1.0
        if i > 0 and words[i - 1] in INTENSIFIERS:
            mult = 1.5
        if w in POSITIVE_WORDS:
            score += 1.0 * mult
        elif w in NEGATIVE_WORDS:
            score -= 1.0 * mult
    # Normalize to [-1, 1] while preserving strength.
    return clamp(score / max(3.0, math.sqrt(len(words))), -1.0, 1.0)


def _load_finbert_pipeline():
    global _FINBERT_MODEL, _FINBERT_TOKENIZER, _FINBERT_LOAD_ERROR
    if _FINBERT_MODEL is not None and _FINBERT_TOKENIZER is not None:
        return _FINBERT_MODEL, _FINBERT_TOKENIZER
    if _FINBERT_LOAD_ERROR is not None:
        return None
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
        from transformers.utils import logging as hf_logging  # type: ignore

        model_id = os.environ.get("FINBERT_MODEL_ID", "ProsusAI/finbert")
        # Keep demo output clean: suppress HF progress bars / verbose load report during model init.
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        hf_logging.set_verbosity_error()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _FINBERT_TOKENIZER = AutoTokenizer.from_pretrained(model_id)
            _FINBERT_MODEL = AutoModelForSequenceClassification.from_pretrained(model_id, use_safetensors=True)
        return _FINBERT_MODEL, _FINBERT_TOKENIZER
    except Exception as e:  # pragma: no cover - depends on local env
        _FINBERT_LOAD_ERROR = f"{type(e).__name__}: {e}"
        return None


def finbert_sentiment_score(text: str) -> Optional[float]:
    bundle = _load_finbert_pipeline()
    if bundle is None:
        return None
    model, tokenizer = bundle
    try:
        import torch  # type: ignore

        trimmed = text[:512]
        inputs = tokenizer(trimmed, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        id2label = {int(k): v for k, v in getattr(model.config, "id2label", {}).items()} if hasattr(model.config, "id2label") else {}
        best_idx = int(probs.argmax())
        label = str(id2label.get(best_idx, "")).lower()
        conf = float(probs[best_idx])
        if "positive" in label:
            return clamp(conf, 0.0, 1.0)
        if "negative" in label:
            return clamp(-conf, -1.0, 0.0)
        # Fallback if labels are ordered [positive, negative, neutral] or similar.
        label_map = {str(v).lower(): i for i, v in id2label.items()}
        pos_idx = label_map.get("positive")
        neg_idx = label_map.get("negative")
        if pos_idx is not None and neg_idx is not None:
            return clamp(float(probs[pos_idx] - probs[neg_idx]), -1.0, 1.0)
        return 0.0
    except Exception:
        return None


def sentiment_score_for_text(text: str, use_finbert: bool = False) -> float:
    if use_finbert:
        scored = finbert_sentiment_score(text)
        if scored is not None:
            return scored
    return text_sentiment_score(text)


def fetch_yahoo_candles(
    ticker: str,
    interval: str = "1d",
    range_: str = "6mo",
    timeout: int = 15,
    min_candles: int = 40,
) -> List[Candle]:
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


def compute_fundamental_score(metrics: Dict[str, Optional[float]]) -> Tuple[float, List[str], Dict[str, Any]]:
    score = 0.0
    reasons: List[str] = []
    if not metrics:
        return 0.0, ["No fundamentals data (neutral)"], {}

    pm = metrics.get("profit_margins")
    if pm is not None:
        if pm > 0.20:
            score += 0.15
            reasons.append(f"Strong profit margin ({pm:.0%})")
        elif pm > 0.08:
            score += 0.08
            reasons.append(f"Positive profit margin ({pm:.0%})")
        elif pm < 0:
            score -= 0.15
            reasons.append(f"Negative profit margin ({pm:.0%})")

    roe = metrics.get("return_on_equity")
    if roe is not None:
        if roe > 0.20:
            score += 0.12
            reasons.append(f"High ROE ({roe:.0%})")
        elif roe < 0:
            score -= 0.10
            reasons.append(f"Negative ROE ({roe:.0%})")

    dte = metrics.get("debt_to_equity")
    if dte is not None:
        if dte < 80:
            score += 0.05
            reasons.append(f"Moderate leverage (D/E {dte:.0f})")
        elif dte > 250:
            score -= 0.10
            reasons.append(f"High leverage (D/E {dte:.0f})")

    pe = metrics.get("forward_pe") or metrics.get("trailing_pe")
    if pe is not None:
        if 0 < pe < 15:
            score += 0.06
            reasons.append(f"Reasonable P/E ({pe:.1f})")
        elif pe > 80:
            score -= 0.06
            reasons.append(f"Rich valuation (P/E {pe:.1f})")

    beta = metrics.get("beta")
    if beta is not None and beta > 2.5:
        score -= 0.04
        reasons.append(f"High beta ({beta:.2f})")

    compact = {k: (None if v is None else round(v, 4)) for k, v in metrics.items()}
    return clamp(score, -1.0, 1.0), reasons or ["Fundamentals neutral"], compact


def get_cached_fundamentals(ticker: str, timeout: int = 15) -> Dict[str, Any]:
    cache = load_fundamentals_cache()
    key = ticker.upper()
    today = today_utc_str()
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


def google_news_rss_query(
    query: str,
    timeout: int = 15,
    use_finbert: bool = False,
    ticker: Optional[str] = None,
    company_name: Optional[str] = None,
    max_age_hours: float = NEWS_MAX_AGE_HOURS,
) -> List[SentimentSample]:
    encoded = urllib.parse.quote_plus(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    out: List[SentimentSample] = []
    now_ts = int(time.time())
    ticker_u = (ticker or "").upper()
    company_tokens = [t.lower() for t in re.findall(r"[A-Za-z]+", company_name or "") if len(t) >= 3]
    for item in root.findall(".//item")[:20]:
        title = (item.findtext("title") or "").strip()
        if ticker_u:
            title_u = title.upper()
            title_l = title.lower()
            has_ticker = re.search(rf"(?<![A-Z]){re.escape(ticker_u)}(?![A-Z])", title_u) is not None
            has_company = any(tok in title_l for tok in company_tokens[:3]) if company_tokens else False
            if not (has_ticker or has_company):
                continue
        link = (item.findtext("link") or "").strip() or None
        pub_date = item.findtext("pubDate")
        ts = None
        if pub_date:
            try:
                ts = int(parsedate_to_datetime(pub_date).astimezone(dt.timezone.utc).timestamp())
            except Exception:
                ts = None
        if ts is not None:
            age_h = max(0.0, (now_ts - ts) / 3600.0)
            if age_h > max_age_hours:
                continue
        out.append(
            SentimentSample(
                source="news",
                title=title,
                score=sentiment_score_for_text(title, use_finbert=use_finbert),
                ts=ts,
                url=link,
            )
        )
    return out


def fetch_reddit_mentions(
    ticker: str,
    company_name: Optional[str] = None,
    timeout: int = 15,
    use_finbert: bool = False,
    max_age_hours: float = REDDIT_MAX_AGE_HOURS,
) -> Tuple[List[SentimentSample], Dict[str, Any]]:
    allowed_subs_query = " OR ".join(f"subreddit:{s}" for s in sorted(REDDIT_ALLOWED_SUBREDDITS))
    query_terms = [f"${ticker}", ticker]
    if company_name:
        query_terms.append(company_name)
    stock_query = " OR ".join(query_terms)
    q = f"({allowed_subs_query}) ({stock_query})"
    url = "https://www.reddit.com/search.json"
    # Expand Reddit search window enough so strict age filtering can be user-controlled.
    params = {"q": q, "sort": "new", "limit": 80, "t": "month"}
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    children = safe_get(data, "data", "children", default=[])
    out: List[SentimentSample] = []
    debug = {
        "query": q,
        "raw_results": len(children),
        "max_age_hours": max_age_hours,
        "kept": 0,
        "filtered_subreddit": 0,
        "filtered_empty_text": 0,
        "filtered_no_match": 0,
        "filtered_stale": 0,
        "subreddits_seen": {},
    }
    ticker_pattern = re.compile(rf"(?<![A-Za-z])(\${re.escape(ticker)}|{re.escape(ticker)})(?![A-Za-z])", re.IGNORECASE)
    company_tokens = [tok for tok in re.findall(r"[A-Za-z]+", (company_name or "").lower()) if len(tok) >= 4]
    now_ts = int(time.time())
    for child in children:
        post = child.get("data", {})
        sub = (post.get("subreddit") or "").lower()
        if sub:
            debug["subreddits_seen"][sub] = debug["subreddits_seen"].get(sub, 0) + 1
        if sub and sub not in REDDIT_ALLOWED_SUBREDDITS:
            debug["filtered_subreddit"] += 1
            continue
        title = (post.get("title") or "").strip()
        selftext = (post.get("selftext") or "").strip()
        combined = (title + " " + selftext).strip()
        if not combined:
            debug["filtered_empty_text"] += 1
            continue
        # Strict stock-specific matching: ticker/company must appear in the TITLE.
        title_l = title.lower()
        title_has_ticker = ticker_pattern.search(title) is not None
        title_has_company = any(tok in title_l for tok in company_tokens[:3]) if company_tokens else False
        if not (title_has_ticker or title_has_company):
            debug["filtered_no_match"] += 1
            continue
        created = post.get("created_utc")
        if created is not None:
            age_h = max(0.0, (now_ts - int(created)) / 3600.0)
            if age_h > max_age_hours:
                debug["filtered_stale"] += 1
                continue
        permalink = post.get("permalink")
        url_post = f"https://reddit.com{permalink}" if permalink else None
        score = sentiment_score_for_text(combined, use_finbert=use_finbert)
        # Very lightweight engagement weighting in score.
        ups = float(post.get("ups") or 0)
        comments = float(post.get("num_comments") or 0)
        engagement_bonus = clamp(math.log1p(ups + comments) / 12.0, 0.0, 0.15)
        score = clamp(score + (engagement_bonus if score >= 0 else -engagement_bonus), -0.95, 0.95)
        score *= _source_weight_for_subreddit(f"r/{sub}: {title}")
        score = clamp(score, -0.95, 0.95)
        out.append(
            SentimentSample(
                source="reddit",
                title=f"r/{sub}: {title}",
                score=score,
                ts=int(created) if created is not None else None,
                url=url_post,
            )
        )
    debug["fallback_used"] = False
    debug["fallback_candidates"] = 0
    debug["kept"] = len(out)
    return out, debug


def _recency_weight(sample_ts: Optional[int], now_ts: int, half_life_hours: float) -> float:
    if sample_ts is None:
        return 1.0
    age_hours = max(0.0, (now_ts - sample_ts) / 3600.0)
    if half_life_hours <= 0:
        return 1.0
    return 0.5 ** (age_hours / half_life_hours)


def summarize_sentiment(
    source: str,
    samples: Iterable[SentimentSample],
    now_ts: Optional[int] = None,
    half_life_hours: float = 4.0,
) -> SentimentSummary:
    sample_list = list(samples)
    if now_ts is None:
        now_ts = int(time.time())
    if not sample_list:
        return SentimentSummary(
            source=source,
            score=0.0,
            count=0,
            recent_1h_count=0,
            recent_6h_count=0,
            recency_weighted_count=0.0,
            avg_age_hours=None,
        )

    weighted_num = 0.0
    weighted_den = 0.0
    recent_1h = 0
    recent_6h = 0
    weighted_count = 0.0
    ages: List[float] = []
    for s in sample_list:
        w_strength = 1.0 + abs(s.score)
        w_recency = _recency_weight(s.ts, now_ts, half_life_hours)
        w = w_strength * w_recency
        weighted_num += s.score * w
        weighted_den += w
        weighted_count += w_recency
        if s.ts is not None:
            age_h = max(0.0, (now_ts - s.ts) / 3600.0)
            ages.append(age_h)
            if age_h <= 1.0:
                recent_1h += 1
            if age_h <= 6.0:
                recent_6h += 1
        else:
            # Unknown timestamp gets counted as not-recent.
            pass

    score = clamp(weighted_num / (weighted_den or 1.0), -1.0, 1.0)
    return SentimentSummary(
        source=source,
        score=score,
        count=len(sample_list),
        recent_1h_count=recent_1h,
        recent_6h_count=recent_6h,
        recency_weighted_count=weighted_count,
        avg_age_hours=(statistics.mean(ages) if ages else None),
    )


def top_sentiment_sources(
    reddit_samples: List[SentimentSample],
    news_samples: List[SentimentSample],
    limit: int = 3,
) -> List[Dict[str, Any]]:
    merged = list(reddit_samples) + list(news_samples)
    merged.sort(key=lambda s: (abs(s.score), s.ts or 0), reverse=True)
    out: List[Dict[str, Any]] = []
    for s in merged[: max(0, limit)]:
        title = s.title if len(s.title) <= 140 else s.title[:137] + "..."
        out.append(
            {
                "source": s.source,
                "score": round(s.score, 3),
                "ts": s.ts,
                "title": title,
                "url": s.url,
            }
        )
    return out


def top_sentiment_sources_balanced(samples: List[SentimentSample], limit: int = 5) -> List[Dict[str, Any]]:
    if not samples or limit <= 0:
        return []
    pos = [s for s in samples if s.score > 0]
    neg = [s for s in samples if s.score < 0]
    neu = [s for s in samples if s.score == 0]
    pos.sort(key=lambda s: (s.score, s.ts or 0), reverse=True)
    neg.sort(key=lambda s: (s.score, -(s.ts or 0)))  # most negative first
    neu.sort(key=lambda s: (s.ts or 0), reverse=True)

    out_samples: List[SentimentSample] = []
    # First pass: alternate strongest positive and strongest negative.
    while len(out_samples) < limit and (pos or neg):
        if pos and len(out_samples) < limit:
            out_samples.append(pos.pop(0))
        if neg and len(out_samples) < limit:
            out_samples.append(neg.pop(0))
    # Then fill with neutrals/newest leftovers.
    for bucket in (pos, neg, neu):
        for s in bucket:
            if len(out_samples) >= limit:
                break
            out_samples.append(s)
        if len(out_samples) >= limit:
            break
    return top_sentiment_sources(out_samples, [], limit=limit)


def _source_weight_for_subreddit(label: str) -> float:
    l = label.lower()
    if "r/wallstreetbets" in l:
        return 0.80
    if "r/options" in l:
        return 0.90
    if "r/investing" in l:
        return 0.95
    if "r/stocks" in l or "r/stockmarket" in l:
        return 1.00
    return 0.90


def compute_technicals(
    candles: List[Candle],
    benchmark_rel_5: Optional[float] = None,
    gap_from_prev_daily_close: Optional[float] = None,
) -> Tuple[float, Dict[str, Any], List[str]]:
    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    vols = [c.volume for c in candles]

    rsi14 = rsi(closes, 14)
    ema12 = ema(closes, 12)
    ema26 = ema(closes, 26)
    macd: List[Optional[float]] = []
    for a, b in zip(ema12, ema26):
        macd.append(None if a is None or b is None else a - b)
    macd_vals = [m if m is not None else float("nan") for m in macd]
    macd_clean = [x for x in macd_vals if not math.isnan(x)]
    sig_clean = ema(macd_clean, 9)
    signal_line: List[Optional[float]] = [None] * (len(macd) - len(sig_clean))
    signal_line = [None] * len(macd)
    # Align by replaying across valid macd values.
    j = 0
    for i, m in enumerate(macd):
        if m is None:
            continue
        signal_line[i] = sig_clean[j] if j < len(sig_clean) else None
        j += 1

    vol_sma20 = sma(vols, 20)
    close_sma20 = sma(closes, 20)
    atr14 = atr(candles, 14)
    vwma20 = vwma(closes, vols, 20)

    i = len(candles) - 1
    if i < 30:
        raise RuntimeError("Need more candles for technicals")

    latest_close = closes[i]
    latest_rsi = rsi14[i]
    latest_macd = macd[i]
    latest_signal = signal_line[i]
    prev_macd = macd[i - 1]
    prev_signal = signal_line[i - 1]
    latest_vol = vols[i]
    avg_vol = vol_sma20[i] or 0.0
    vol_ratio_bar = "latest"
    vol_ratio = (latest_vol / avg_vol) if avg_vol else 0.0
    volume_reliable = True
    # Intraday free feeds often return partial/zero volume on the latest bar. Use a completed-bar median fallback.
    if i >= 3:
        recent_completed = vols[max(0, i - 3) : i]  # exclude current bar
        baseline_completed = vols[max(0, i - 23) : max(0, i - 3)]
        if len(recent_completed) >= 2 and len(baseline_completed) >= 10:
            med_recent = statistics.median(recent_completed)
            med_base = statistics.median(baseline_completed) if baseline_completed else 0.0
            if med_base > 0:
                intraday_med_ratio = med_recent / med_base
                # Prefer stable completed-bar median for intraday scans.
                vol_ratio = intraday_med_ratio
                vol_ratio_bar = "median3_prev"
                # Mark unreliable if completed bars still look suspiciously tiny.
                if med_recent <= 0 or med_base <= 0 or med_recent < 0.03 * med_base:
                    volume_reliable = False
                # Free intraday Yahoo can mix session regimes; extreme ratios are often artifacts.
                if intraday_med_ratio > 8.0:
                    volume_reliable = False
            else:
                volume_reliable = False
        else:
            # If there isn't enough intraday context, don't trust volume much.
            volume_reliable = False
    else:
        volume_reliable = False

    # Last-resort single previous bar fallback (mostly for short ranges / edge cases).
    if (not volume_reliable or vol_ratio == 0.0) and i >= 1:
        prev_avg_vol = vol_sma20[i - 1] or 0.0
        prev_vol = vols[i - 1]
        if prev_avg_vol:
            fallback_ratio = prev_vol / prev_avg_vol
            if vol_ratio == 0.0:
                vol_ratio = fallback_ratio
            if vol_ratio_bar == "latest":
                vol_ratio_bar = "prev"
            if fallback_ratio < 0.03:
                volume_reliable = False
    ret_1 = (closes[i] / closes[i - 1] - 1.0) if i >= 1 and closes[i - 1] else 0.0
    ret_5 = (closes[i] / closes[i - 5] - 1.0) if i >= 5 and closes[i - 5] else 0.0
    latest_atr = atr14[i]
    atr_pct = (latest_atr / latest_close) if latest_atr and latest_close else None
    above_vwma20 = vwma20[i] is not None and latest_close > float(vwma20[i])
    breakout_lookback = 20
    recent_high_excl_today = max(highs[i - breakout_lookback : i]) if i - breakout_lookback >= 0 else max(highs[:-1])
    breakout = latest_close > recent_high_excl_today
    macd_bull_cross = (
        latest_macd is not None
        and latest_signal is not None
        and prev_macd is not None
        and prev_signal is not None
        and prev_macd <= prev_signal
        and latest_macd > latest_signal
    )
    macd_bull = latest_macd is not None and latest_signal is not None and latest_macd > latest_signal
    trend_above_sma20 = close_sma20[i] is not None and latest_close > float(close_sma20[i])

    reasons: List[str] = []
    score = 0.0

    if latest_rsi is not None:
        if latest_rsi >= 60:
            score += 0.25
            reasons.append(f"RSI strong ({latest_rsi:.1f})")
        elif latest_rsi >= 50:
            score += 0.10
            reasons.append(f"RSI bullish bias ({latest_rsi:.1f})")
        elif latest_rsi <= 40:
            score -= 0.20
            reasons.append(f"RSI weak ({latest_rsi:.1f})")

    if macd_bull_cross:
        score += 0.30
        reasons.append("MACD bullish crossover")
    elif macd_bull:
        score += 0.15
        reasons.append("MACD above signal")
    elif latest_macd is not None and latest_signal is not None and latest_macd < latest_signal:
        score -= 0.15
        reasons.append("MACD below signal")

    if breakout:
        score += 0.25
        reasons.append("Price breakout above 20-bar high")
    elif trend_above_sma20:
        score += 0.10
        reasons.append("Price above 20-day SMA")
    else:
        score -= 0.05
        reasons.append("No breakout/trend confirmation")

    if above_vwma20:
        score += 0.08
        reasons.append("Price above VWMA20")
    elif vwma20[i] is not None:
        score -= 0.05
        reasons.append("Price below VWMA20")

    if volume_reliable:
        if vol_ratio >= 1.5:
            score += 0.20
            suffix = " using median of prior 3 bars" if vol_ratio_bar == "median3_prev" else (" using prev bar" if vol_ratio_bar == "prev" else "")
            reasons.append(f"High volume ({vol_ratio:.2f}x avg{suffix})")
        elif vol_ratio >= 1.1:
            score += 0.08
            suffix = " using median of prior 3 bars" if vol_ratio_bar == "median3_prev" else (" using prev bar" if vol_ratio_bar == "prev" else "")
            reasons.append(f"Volume improving ({vol_ratio:.2f}x avg{suffix})")
        elif vol_ratio < 0.8:
            score -= 0.08
            suffix = " using median of prior 3 bars" if vol_ratio_bar == "median3_prev" else (" using prev bar" if vol_ratio_bar == "prev" else "")
            reasons.append(f"Low volume ({vol_ratio:.2f}x avg{suffix})")
    else:
        reasons.append("Volume signal unreliable on free intraday feed (neutralized)")

    if benchmark_rel_5 is not None:
        if benchmark_rel_5 > 0.03:
            score += 0.10
            reasons.append(f"Strong relative strength vs benchmark (+{benchmark_rel_5:.1%})")
        elif benchmark_rel_5 > 0.01:
            score += 0.05
            reasons.append(f"Positive relative strength vs benchmark (+{benchmark_rel_5:.1%})")
        elif benchmark_rel_5 < -0.03:
            score -= 0.10
            reasons.append(f"Weak relative strength vs benchmark ({benchmark_rel_5:.1%})")

    if gap_from_prev_daily_close is not None:
        if gap_from_prev_daily_close >= 0.12:
            score -= 0.18
            reasons.append(f"Extended gap from prior close ({gap_from_prev_daily_close:.1%})")
        elif gap_from_prev_daily_close >= 0.08:
            score -= 0.10
            reasons.append(f"Large gap from prior close ({gap_from_prev_daily_close:.1%})")
        elif gap_from_prev_daily_close <= -0.08:
            score -= 0.08
            reasons.append(f"Large downside gap ({gap_from_prev_daily_close:.1%})")

    if atr_pct is not None and atr_pct > 0.04:
        reasons.append(f"High ATR regime ({atr_pct:.1%} of price)")

    indicators = {
        "close": round(latest_close, 2),
        "rsi14": None if latest_rsi is None else round(latest_rsi, 2),
        "macd": None if latest_macd is None else round(latest_macd, 4),
        "macd_signal": None if latest_signal is None else round(latest_signal, 4),
        "volume_ratio_20": round(vol_ratio, 2),
        "volume_ratio_bar": vol_ratio_bar,
        "volume_reliable": volume_reliable,
        "ret_1_bar": round(ret_1, 4),
        "ret_5_bar": round(ret_5, 4),
        "atr14": None if latest_atr is None else round(latest_atr, 3),
        "atr_pct": None if atr_pct is None else round(atr_pct, 4),
        "vwma20": None if vwma20[i] is None else round(float(vwma20[i]), 2),
        "above_vwma20": bool(above_vwma20),
        "rel_strength_5": None if benchmark_rel_5 is None else round(benchmark_rel_5, 4),
        "gap_from_prev_daily_close": None if gap_from_prev_daily_close is None else round(gap_from_prev_daily_close, 4),
        "breakout_20": breakout,
        "above_sma20": bool(trend_above_sma20),
    }
    return clamp(score, -1.0, 1.0), indicators, reasons


def compute_extended_move_penalty(indicators: Dict[str, Any], interval: str) -> Tuple[float, Optional[str]]:
    r1 = indicators.get("ret_1_bar")
    r5 = indicators.get("ret_5_bar")
    rsi_val = indicators.get("rsi14")
    if not isinstance(r1, (int, float)) or not isinstance(r5, (int, float)):
        return 0.0, None

    # Thresholds are looser on 1h bars than daily bars.
    if interval == "1h":
        jump1 = 0.04
        jump5 = 0.10
    else:
        jump1 = 0.08
        jump5 = 0.15

    penalty = 0.0
    if r1 >= jump1:
        penalty -= 0.18
    if r5 >= jump5:
        penalty -= 0.14
    if isinstance(rsi_val, (int, float)) and rsi_val >= 72:
        penalty -= 0.08

    if penalty >= 0:
        return 0.0, None
    reason = f"Extended move penalty applied (r1={r1:.1%}, r5={r5:.1%}, interval={interval})"
    return clamp(penalty, -0.40, 0.0), reason


def hybrid_decision(
    ticker: str,
    technical_score: float,
    indicators: Dict[str, Any],
    technical_reasons: List[str],
    reddit_samples: List[SentimentSample],
    news_samples: List[SentimentSample],
    reddit_display_samples: Optional[List[SentimentSample]] = None,
    fundamental_score: float = 0.0,
    fundamental_reasons: Optional[List[str]] = None,
    fundamental_metrics: Optional[Dict[str, Any]] = None,
    fundamentals_as_of: Optional[str] = None,
    interval: str = "1d",
    config: Optional[HybridConfig] = None,
    debug_info: Optional[Dict[str, Any]] = None,
    always_signal: bool = False,
) -> SignalResult:
    cfg = config or HybridConfig()
    now_ts = int(time.time())
    reddit_summary = summarize_sentiment(
        "reddit", reddit_samples, now_ts=now_ts, half_life_hours=cfg.half_life_hours_reddit
    )
    news_summary = summarize_sentiment(
        "news", news_samples, now_ts=now_ts, half_life_hours=cfg.half_life_hours_news
    )
    mention_count = reddit_summary.count + news_summary.count

    reddit_count_conf = clamp(reddit_summary.count / 3.0, 0.0, 1.0)
    reddit_effective = reddit_summary.score * reddit_count_conf
    # Unified source blend: news anchors, Reddit contributes with confidence gating.
    w_reddit, w_news = 0.40, 0.60

    if mention_count == 0:
        sentiment_score = 0.0
    else:
        sentiment_score = clamp((w_reddit * reddit_effective) + (w_news * news_summary.score), -1.0, 1.0)

    recent_1h_mentions = reddit_summary.recent_1h_count + news_summary.recent_1h_count
    recent_6h_mentions = reddit_summary.recent_6h_count + news_summary.recent_6h_count
    baseline_older = max(1.0, mention_count - recent_6h_mentions)
    mention_velocity = recent_1h_mentions / baseline_older
    event_score = 0.0
    if recent_1h_mentions >= 3:
        event_score += 0.35
    elif recent_1h_mentions >= 2:
        event_score += 0.20
    if mention_velocity >= 0.5:
        event_score += 0.30
    elif mention_velocity >= 0.25:
        event_score += 0.15
    if abs(sentiment_score) >= 0.45 and recent_6h_mentions >= 3:
        event_score += 0.25
    elif abs(sentiment_score) >= 0.30 and recent_6h_mentions >= 2:
        event_score += 0.10
    if news_summary.recent_6h_count >= 3:
        event_score += 0.10
    event_score = clamp(event_score, 0.0, 1.0)
    mode = "EVENT" if event_score >= cfg.event_min_score else "NORMAL"

    if mode == "EVENT":
        w_sent = cfg.event_sentiment_weight
        w_tech = cfg.event_technical_weight
        w_fund = cfg.event_fundamental_weight
    else:
        w_sent = cfg.normal_sentiment_weight
        w_tech = cfg.normal_technical_weight
        w_fund = cfg.normal_fundamental_weight

    mention_activity_score = clamp(math.log1p(mention_count) / 4.0, 0.0, 0.5)
    mention_signed_bonus = mention_activity_score * (1 if sentiment_score >= 0 else -1) * cfg.mention_bonus_scale
    late_penalty, late_penalty_reason = compute_extended_move_penalty(indicators, interval=interval)
    disagreement = abs(technical_score - sentiment_score)

    hybrid_score = clamp(
        (w_tech * technical_score)
        + (w_sent * sentiment_score)
        + (w_fund * fundamental_score)
        + mention_signed_bonus
        + late_penalty,
        -1.0,
        1.0,
    )

    reasons: List[str] = []
    reasons.append(
        f"Market Regime = {mode}, Event Score = {event_score:.2f}, Mention Velocity = {mention_velocity:.2f}"
    )
    reasons.append(
        f"Sentiment Blend = {sentiment_score:+.2f} "
        f"(Reddit = {reddit_summary.score:+.2f}, Confidence = {reddit_count_conf:.2f}; News = {news_summary.score:+.2f})"
    )
    reasons.append(
        f"Mention Activity = {mention_count} items (1h = {recent_1h_mentions}, 6h = {recent_6h_mentions})"
    )
    if fundamentals_as_of:
        fpe = (fundamental_metrics or {}).get("forward_pe") if isinstance(fundamental_metrics, dict) else None
        pm = (fundamental_metrics or {}).get("profit_margins") if isinstance(fundamental_metrics, dict) else None
        mc = (fundamental_metrics or {}).get("market_cap") if isinstance(fundamental_metrics, dict) else None
        summary_bits = [f"as_of={fundamentals_as_of}"]
        if isinstance(fpe, (int, float)):
            summary_bits.append(f"fwdPE={fpe:.1f}")
        if isinstance(pm, (int, float)):
            summary_bits.append(f"margin={pm:.0%}")
        if isinstance(mc, (int, float)):
            summary_bits.append(f"mcap={mc/1e9:.0f}B")
        reasons.append("Fundamentals snapshot: " + ", ".join(summary_bits))
    reasons.extend(list(technical_reasons))
    if fundamental_reasons:
        reasons.extend(fundamental_reasons[:2])
    reasons.append(f"Reddit sentiment {reddit_summary.score:+.2f} ({reddit_summary.count} items, conf {reddit_count_conf:.2f})")
    reasons.append(f"News sentiment {news_summary.score:+.2f} ({news_summary.count} items)")
    if late_penalty_reason:
        reasons.append(late_penalty_reason)
    reasons.append(f"Tech/Sent disagreement={disagreement:.2f}")

    if hybrid_score >= 0.45:
        action = "BUY"
        confidence = "HIGH"
    elif hybrid_score >= 0.20:
        action = "BUY"
        confidence = "MEDIUM"
    elif hybrid_score <= -0.45:
        action = "SELL"
        confidence = "HIGH"
    elif hybrid_score <= -0.20:
        action = "SELL"
        confidence = "MEDIUM"
    else:
        if always_signal:
            action = "BUY" if hybrid_score >= 0 else "SELL"
            confidence = "LOW"
            reasons.append("Low conviction but emitted due to always_signal")
        else:
            action = "WATCH"
            confidence = "LOW"

    # Conflict cap: if technicals and sentiment materially disagree, prefer WATCH unless conviction is strong.
    if action in {"BUY", "SELL"} and (technical_score * sentiment_score < 0) and disagreement >= cfg.disagreement_watch_gap:
        if abs(hybrid_score) < cfg.disagreement_high_override:
            action = "WATCH"
            confidence = "LOW"
            reasons.append("WATCH cap: technicals and sentiment disagree materially")

    # Unified no-chase cap for extended bullish setups after large gaps.
    gap_val = indicators.get("gap_from_prev_daily_close")
    rsi_val = indicators.get("rsi14")
    volx = indicators.get("volume_ratio_20")
    if action == "BUY":
        if isinstance(gap_val, (int, float)) and isinstance(rsi_val, (int, float)):
            if gap_val >= cfg.gap_watch_threshold and rsi_val >= cfg.rsi_watch_threshold:
                if not (sentiment_score >= 0.80 and isinstance(volx, (int, float)) and volx >= 2.0):
                    action = "WATCH"
                    confidence = "LOW"
                    reasons.append("Unified no-chase cap: large gap + high RSI")

    reddit_top = reddit_display_samples if reddit_display_samples is not None else reddit_samples

    return SignalResult(
        ticker=ticker,
        technical_score=technical_score,
        sentiment_score=sentiment_score,
        reddit_sentiment=reddit_summary.score,
        news_sentiment=news_summary.score,
        mention_count=mention_count,
        hybrid_score=hybrid_score,
        action=action,
        confidence=confidence,
        mode=mode,
        reasons=reasons,
        indicators={
            **indicators,
            "fundamental_score": round(fundamental_score, 3),
            "fundamental_reasons": list(fundamental_reasons or []),
            "event_score": round(event_score, 3),
            "mention_velocity": round(mention_velocity, 3),
            "late_penalty": round(late_penalty, 3),
            "disagreement": round(disagreement, 3),
            "fundamentals_as_of": fundamentals_as_of,
            "fundamentals": fundamental_metrics or {},
        },
        top_sources=top_sentiment_sources(reddit_samples, news_samples, limit=5),
        top_reddit_sources=top_sentiment_sources(reddit_top, [], limit=5),
        top_news_sources=top_sentiment_sources_balanced(news_samples, limit=5),
        debug=debug_info or {},
    )


def company_name_map() -> Dict[str, str]:
    return {
        "AMD": "Advanced Micro Devices",
        "NVDA": "NVIDIA",
        "META": "Meta",
        "TSLA": "Tesla",
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "AMZN": "Amazon",
        "GOOGL": "Google",
        "NFLX": "Netflix",
        "PLTR": "Palantir",
    }


def benchmark_for_ticker(ticker: str) -> str:
    t = ticker.upper()
    semis = {"AMD", "NVDA", "AVGO", "MU", "QCOM", "INTC", "TSM", "ASML", "AMAT", "LRCX"}
    if t in semis:
        return "SMH"
    if t in {"META", "GOOGL", "MSFT", "AMZN", "NFLX", "AAPL"}:
        return "QQQ"
    return "SPY"


def relative_strength_lookback(candles: List[Candle], benchmark_candles: List[Candle], lookback: int = 5) -> Optional[float]:
    if len(candles) <= lookback or len(benchmark_candles) <= lookback:
        return None
    # Align loosely by using latest N bars of each feed (good enough for MVP same interval/range).
    c_ret = candles[-1].close / candles[-1 - lookback].close - 1.0
    b_ret = benchmark_candles[-1].close / benchmark_candles[-1 - lookback].close - 1.0
    return c_ret - b_ret


def gap_from_previous_daily_close(current_price: float, daily_candles: List[Candle]) -> Optional[float]:
    if len(daily_candles) < 2:
        return None
    prev_close = daily_candles[-2].close
    if not prev_close:
        return None
    return current_price / prev_close - 1.0


def demo_samples_for_ticker(ticker: str) -> Tuple[List[SentimentSample], List[SentimentSample]]:
    t = ticker.upper()
    now_ts = int(time.time())
    if t == "AMD":
        news = [
            SentimentSample("news", "AMD announces partnership with Meta to accelerate AI infrastructure", 0.95, ts=now_ts - 10 * 60),
            SentimentSample("news", "AMD shares surge after major Meta AI chip collaboration", 0.92, ts=now_ts - 20 * 60),
            SentimentSample("news", "Analysts upgrade AMD on strong AI momentum and partnership outlook", 0.78, ts=now_ts - 45 * 60),
        ]
        reddit = [
            SentimentSample("reddit", "r/stocks: AMD + Meta partnership looks huge for MI300 demand", 0.84, ts=now_ts - 12 * 60),
            SentimentSample("reddit", "r/wallstreetbets: AMD breakout on volume after Meta news", 0.72, ts=now_ts - 30 * 60),
            SentimentSample("reddit", "r/investing: AMD fundamentals improving, AI narrative getting stronger", 0.65, ts=now_ts - 80 * 60),
        ]
        return reddit, news
    # Neutral demo fallback
    return (
        [SentimentSample("reddit", f"r/stocks: Mixed sentiment on {t} today", 0.05, ts=now_ts - 3 * 3600)],
        [SentimentSample("news", f"{t} stock news mixed as investors await catalyst", -0.03, ts=now_ts - 4 * 3600)],
    )


def render_console_table(results: List[SignalResult]) -> str:
    headers = ["Ticker", "Action", "Conf", "Mode", "Hybrid", "Tech", "Sent", "Mentions", "Close", "RSI", "VolX"]
    rows = []
    for r in results:
        rows.append(
            [
                r.ticker,
                r.action,
                r.confidence,
                r.mode,
                f"{r.hybrid_score:+.2f}",
                f"{r.technical_score:+.2f}",
                f"{r.sentiment_score:+.2f}",
                str(r.mention_count),
                str(r.indicators.get("close", "")),
                str(r.indicators.get("rsi14", "")),
                (
                    "UNREL"
                    if r.indicators.get("volume_reliable") is False
                    else str(r.indicators.get("volume_ratio_20", ""))
                ),
            ]
        )
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)] if rows else [len(h) for h in headers]
    lines = []
    lines.append(" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    lines.append("-+-".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        lines.append(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


def save_json_report(path: str, results: List[SignalResult], meta: Dict[str, Any]) -> None:
    payload = {
        "generated_at": now_utc_iso(),
        "meta": meta,
        "results": [
            {
                "ticker": r.ticker,
                "action": r.action,
                "confidence": r.confidence,
                "mode": r.mode,
                "hybrid_score": r.hybrid_score,
                "technical_score": r.technical_score,
                "sentiment_score": r.sentiment_score,
                "reddit_sentiment": r.reddit_sentiment,
                "news_sentiment": r.news_sentiment,
                "mention_count": r.mention_count,
                "indicators": r.indicators,
                "reasons": r.reasons,
                "top_sources": r.top_sources,
                "top_reddit_sources": r.top_reddit_sources,
                "top_news_sources": r.top_news_sources,
            }
            for r in results
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_csv_log(path: str, results: List[SignalResult], profile: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "ts",
        "ticker",
        "action",
        "confidence",
        "mode",
        "profile",
        "hybrid_score",
        "technical_score",
        "sentiment_score",
        "reddit_sentiment",
        "news_sentiment",
        "mention_count",
        "close",
        "rsi14",
        "macd",
        "macd_signal",
        "volume_ratio_20",
        "benchmark",
        "event_score",
        "late_penalty",
    ]
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        ts = now_utc_iso()
        for r in results:
            ind = r.indicators
            writer.writerow(
                {
                    "ts": ts,
                    "ticker": r.ticker,
                    "action": r.action,
                    "confidence": r.confidence,
                    "mode": r.mode,
                    "profile": profile,
                    "hybrid_score": round(r.hybrid_score, 4),
                    "technical_score": round(r.technical_score, 4),
                    "sentiment_score": round(r.sentiment_score, 4),
                    "reddit_sentiment": round(r.reddit_sentiment, 4),
                    "news_sentiment": round(r.news_sentiment, 4),
                    "mention_count": r.mention_count,
                    "close": ind.get("close"),
                    "rsi14": ind.get("rsi14"),
                    "macd": ind.get("macd"),
                    "macd_signal": ind.get("macd_signal"),
                    "volume_ratio_20": ind.get("volume_ratio_20"),
                    "benchmark": ind.get("benchmark"),
                    "event_score": ind.get("event_score"),
                    "late_penalty": ind.get("late_penalty"),
                }
            )


def scan_ticker(
    ticker: str,
    args: argparse.Namespace,
    company_names: Dict[str, str],
) -> SignalResult:
    cfg = HybridConfig(use_finbert=args.use_finbert)
    candles = fetch_yahoo_candles(ticker=ticker, interval=args.interval, range_=args.range)
    bench = benchmark_for_ticker(ticker)
    try:
        bench_candles = fetch_yahoo_candles(ticker=bench, interval=args.interval, range_=args.range)
        rel5 = relative_strength_lookback(candles, bench_candles, lookback=5)
    except Exception:
        rel5 = None
    try:
        daily_context = fetch_yahoo_candles(ticker=ticker, interval="1d", range_="1mo", min_candles=2)
        gap_daily = gap_from_previous_daily_close(candles[-1].close, daily_context)
    except Exception:
        gap_daily = None

    tech_score, indicators, tech_reasons = compute_technicals(
        candles,
        benchmark_rel_5=rel5,
        gap_from_prev_daily_close=gap_daily,
    )
    indicators["benchmark"] = bench
    try:
        fund_record = get_cached_fundamentals(ticker)
        fundamental_metrics_raw = fund_record.get("metrics", {}) if isinstance(fund_record, dict) else {}
        fundamentals_as_of = fund_record.get("as_of") if isinstance(fund_record, dict) else None
    except Exception:
        fundamental_metrics_raw = {}
        fundamentals_as_of = None
    fundamental_score, fundamental_reasons, fundamental_metrics = compute_fundamental_score(fundamental_metrics_raw)

    if args.demo and ticker.upper() == args.demo_ticker.upper():
        reddit_samples, news_samples = demo_samples_for_ticker(ticker)
        reddit_debug = {"demo": True, "kept": len(reddit_samples)}
        news_debug: Dict[str, Any] = {}
    else:
        company = company_names.get(ticker.upper())
        reddit_samples, reddit_debug = fetch_reddit_mentions(
            ticker,
            company_name=company,
            use_finbert=args.use_finbert,
            max_age_hours=float(getattr(args, "reddit_max_age_hours", REDDIT_MAX_AGE_HOURS)),
        )
        # Query can include company to reduce ticker ambiguities (ex: META).
        news_query = f'{ticker} stock {company or ""}'.strip()
        news_samples = google_news_rss_query(
            news_query,
            use_finbert=args.use_finbert,
            ticker=ticker,
            company_name=company,
            max_age_hours=float(getattr(args, "news_max_age_hours", NEWS_MAX_AGE_HOURS)),
        )
        news_debug = {}

    reddit_samples, reddit_dedupe = dedupe_sentiment_samples(reddit_samples, sim_threshold=0.70)
    news_samples, news_dedupe = dedupe_sentiment_samples(news_samples, sim_threshold=0.75)
    reddit_debug = {**(reddit_debug if isinstance(reddit_debug, dict) else {}), "dedupe": reddit_dedupe}
    news_debug = {**(news_debug if isinstance(news_debug, dict) else {}), "dedupe": news_dedupe}

    reddit_display_samples = reddit_samples
    reddit_samples_for_scoring = reddit_samples
    if isinstance(reddit_debug, dict) and bool(reddit_debug.get("fallback_used")):
        # Keep fallback Reddit items for explainability only; avoid injecting
        # unrelated broad-market chatter into ticker sentiment score.
        reddit_samples_for_scoring = []

    return hybrid_decision(
        ticker=ticker.upper(),
        technical_score=tech_score,
        indicators=indicators,
        technical_reasons=tech_reasons,
        reddit_samples=reddit_samples_for_scoring,
        news_samples=news_samples,
        reddit_display_samples=reddit_display_samples,
        fundamental_score=fundamental_score,
        fundamental_reasons=fundamental_reasons,
        fundamental_metrics=fundamental_metrics,
        fundamentals_as_of=fundamentals_as_of,
        interval=args.interval,
        config=cfg,
        debug_info={"reddit": reddit_debug, "news": news_debug},
        always_signal=args.always_signal,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid technical + social sentiment signal scanner")
    p.add_argument("--tickers", default="", help="Comma-separated tickers (leave empty to be prompted)")
    p.add_argument("--interactive", action="store_true", help="Prompt for ticker input interactively")
    p.add_argument("--interval", default="1d", choices=["1d", "1h", "15m", "5m"], help="Yahoo chart interval")
    p.add_argument("--range", default="6mo", help="Yahoo chart range (e.g., 3mo, 6mo, 1y)")
    p.add_argument("--always-signal", action="store_true", help="Emit BUY/SELL even for low confidence")
    p.add_argument("--demo", action="store_true", help="Inject demo sentiment for one ticker (presentation mode)")
    p.add_argument("--demo-ticker", default="AMD", help="Ticker to apply demo sentiment to when --demo is used")
    p.add_argument("--json-out", default="", help="Optional JSON report output path")
    p.add_argument("--csv-log", default="", help="Optional CSV append log path")
    p.add_argument("--show-sources", action="store_true", help="Print top sentiment-driving news/reddit items")
    p.add_argument("--reddit-debug", action="store_true", help="Print Reddit fetch/filter diagnostics per ticker")
    p.add_argument("--reddit-max-age-hours", type=float, default=REDDIT_MAX_AGE_HOURS, help="Max age for Reddit posts (hours)")
    p.add_argument("--news-max-age-hours", type=float, default=NEWS_MAX_AGE_HOURS, help="Max age for news posts (hours)")
    p.add_argument("--source-limit", type=int, default=3, help="Max sources to print per ticker when --show-sources is used")
    p.add_argument("--top-n-reasons", type=int, default=8, help="How many reasons to print per ticker")
    p.add_argument("--use-finbert", action="store_true", help="Use FinBERT sentiment model if installed (falls back if unavailable)")
    p.add_argument(
        "--profile",
        default="balanced",
        choices=["aggressive", "balanced", "conservative"],
        help="Deprecated/no-op compatibility flag (unified decision policy is always used)",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    tickers_arg = args.tickers.strip()
    if args.interactive or not tickers_arg or tickers_arg.upper() == "TICKER":
        try:
            entered = input("Enter ticker(s) (comma-separated): ").strip()
        except EOFError:
            entered = ""
        tickers_arg = entered

    tickers = [t.strip().upper() for t in tickers_arg.split(",") if t.strip()]
    if not tickers:
        print("No tickers provided", file=sys.stderr)
        return 2

    results: List[SignalResult] = []
    errors: List[str] = []
    names = company_name_map()
    for ticker in tickers:
        try:
            results.append(scan_ticker(ticker, args, names))
            time.sleep(0.4)  # be polite to free endpoints
        except Exception as e:
            errors.append(f"{ticker}: {type(e).__name__}: {e}")

    results.sort(key=lambda r: r.hybrid_score, reverse=True)

    mode_label = "DEMO" if args.demo else "LIVE"
    finbert_label = "FinBERT" if args.use_finbert else "Lexicon"
    if args.use_finbert and _FINBERT_LOAD_ERROR:
        finbert_label += f" (fallback: {_FINBERT_LOAD_ERROR})"
    print(f"Hybrid Signal Scan @ {now_utc_iso()} [{mode_label}] ({finbert_label}, unified-policy, free-source MVP)")
    if args.profile != "balanced":
        print(f"Note: --profile={args.profile} is currently compatibility-only (no effect on decisions).")
    print(render_console_table(results))
    print()
    for r in results:
        print(f"[{r.ticker}] {r.action} ({r.confidence}) hybrid={r.hybrid_score:+.2f}")
        for reason in r.reasons[: max(1, args.top_n_reasons)]:
            print(f"  - {reason}")
        if args.show_sources and r.top_sources:
            print("  - Top Reddit sources:")
            if r.top_reddit_sources:
                for src in r.top_reddit_sources[: max(1, args.source_limit)]:
                    print(f"    * [{src['source']}] {src['score']:+.2f} {src['title']}")
            else:
                print("    * none")
            print("  - Top News sources:")
            if r.top_news_sources:
                for src in r.top_news_sources[: max(1, args.source_limit)]:
                    print(f"    * [{src['source']}] {src['score']:+.2f} {src['title']}")
            else:
                print("    * none")
        if args.reddit_debug:
            dbg = r.debug.get("reddit", {}) if isinstance(r.debug, dict) else {}
            ndbg = r.debug.get("news", {}) if isinstance(r.debug, dict) else {}
            print("  - Reddit debug:")
            if not dbg:
                print("    * none")
            else:
                if dbg.get("demo"):
                    print(f"    * demo mode (kept={dbg.get('kept', 0)})")
                else:
                    print(
                        "    * "
                        + ", ".join(
                            [
                                f"max_age_h={dbg.get('max_age_hours', REDDIT_MAX_AGE_HOURS)}",
                                f"raw={dbg.get('raw_results', 0)}",
                                f"kept={dbg.get('kept', 0)}",
                                f"filtered_subreddit={dbg.get('filtered_subreddit', 0)}",
                                f"filtered_empty={dbg.get('filtered_empty_text', 0)}",
                                f"filtered_no_match={dbg.get('filtered_no_match', 0)}",
                                f"filtered_stale={dbg.get('filtered_stale', 0)}",
                                f"fallback_used={bool(dbg.get('fallback_used', False))}",
                                f"fallback_candidates={dbg.get('fallback_candidates', 0)}",
                            ]
                        )
                    )
                    q = dbg.get("query")
                    if q:
                        q_short = q if len(q) <= 120 else q[:117] + "..."
                        print(f"    * query={q_short}")
                dd = dbg.get("dedupe", {})
                if isinstance(dd, dict) and dd:
                    print(
                        "    * reddit_dedupe="
                        + ", ".join(
                            [
                                f"raw={dd.get('raw', 0)}",
                                f"kept={dd.get('kept', 0)}",
                                f"dropped={dd.get('dropped_duplicates', 0)}",
                                f"sim={dd.get('sim_threshold', 0)}",
                            ]
                        )
                    )
                ndd = ndbg.get("dedupe", {}) if isinstance(ndbg, dict) else {}
                if isinstance(ndd, dict) and ndd:
                    print(
                        "    * news_dedupe="
                        + ", ".join(
                            [
                                f"raw={ndd.get('raw', 0)}",
                                f"kept={ndd.get('kept', 0)}",
                                f"dropped={ndd.get('dropped_duplicates', 0)}",
                                f"sim={ndd.get('sim_threshold', 0)}",
                            ]
                        )
                    )
        print()

    if errors:
        print("Errors:")
        for err in errors:
            print(f"  - {err}")
        print()

    if args.json_out:
        save_json_report(
            args.json_out,
            results,
            {
                "tickers": tickers,
                "interval": args.interval,
                "range": args.range,
                "always_signal": args.always_signal,
                "demo": args.demo,
                "demo_ticker": args.demo_ticker,
                "profile": args.profile,
                "errors": errors,
            },
        )
        print(f"JSON report written to {args.json_out}")

    if args.csv_log:
        append_csv_log(args.csv_log, results, profile=args.profile)
        print(f"CSV log appended to {args.csv_log}")

    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
