#!/usr/bin/env python3

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .data import (
    SignalResult,
    HybridConfig,
    get_cached_fundamentals,
    fetch_yahoo_candles,
)
from .indicators import Candle
from .sentiment import (
    fetch_reddit_mentions,
    google_news_rss_query,
    summarize_sentiment,
    top_sentiment_sources,
    top_sentiment_sources_balanced,
    dedupe_sentiment_samples,
)
from .utils import (
    compute_fundamental_score,
    compute_technicals,
    compute_extended_move_penalty,
    gap_from_previous_daily_close,
    demo_samples_for_ticker,
)


def company_name_map() -> Dict[str, str]:
    """Simple hardcoded company name mapping for demo purposes."""
    return {
        "AAPL": "Apple Inc",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc",
        "AMZN": "Amazon.com Inc",
        "TSLA": "Tesla Inc",
        "NVDA": "NVIDIA Corporation",
        "AMD": "Advanced Micro Devices Inc",
        "SPY": "SPDR S&P 500 ETF Trust",
        "QQQ": "Invesco QQQ Trust",
        "META": "Meta Platforms Inc",
    }


def hybrid_decision(
    ticker: str,
    technical_score: float,
    indicators: Dict[str, Any],
    technical_reasons: List[str],
    reddit_samples: List["SentimentSample"],
    news_samples: List["SentimentSample"],
    reddit_display_samples: Optional[List["SentimentSample"]] = None,
    fundamental_score: float = 0.0,
    fundamental_reasons: Optional[List[str]] = None,
    fundamental_metrics: Optional[Dict[str, Any]] = None,
    fundamentals_as_of: Optional[str] = None,
    interval: str = "1d",
    config: Optional[HybridConfig] = None,
    debug_info: Optional[Dict[str, Any]] = None,
    always_signal: bool = False,
) -> SignalResult:
    from .data import SentimentSample

    cfg = config or HybridConfig()
    now_ts = int(time.time())
    reddit_summary = summarize_sentiment(
        "reddit", reddit_samples, now_ts=now_ts, half_life_hours=cfg.half_life_hours_reddit
    )
    news_summary = summarize_sentiment(
        "news", news_samples, now_ts=now_ts, half_life_hours=cfg.half_life_hours_news
    )

    # Combine sentiment scores
    reddit_weight = cfg.event_sentiment_weight if technical_score >= cfg.event_min_score else cfg.normal_sentiment_weight
    news_weight = reddit_weight * 0.8  # News typically less timely
    sentiment_score = (reddit_summary.score * reddit_weight + news_summary.score * news_weight) / (reddit_weight + news_weight)

    # Technical and fundamental weights
    tech_weight = cfg.event_technical_weight if technical_score >= cfg.event_min_score else cfg.normal_technical_weight
    fund_weight = cfg.event_fundamental_weight if technical_score >= cfg.event_min_score else cfg.normal_fundamental_weight

    # Mention bonus
    mention_count = reddit_summary.count + news_summary.count
    mention_bonus = min(cfg.mention_bonus_scale, mention_count * 0.02)

    # Hybrid score calculation
    hybrid_score = (
        technical_score * tech_weight +
        sentiment_score * (reddit_weight + news_weight) +
        fundamental_score * fund_weight +
        mention_bonus
    )

    # Normalize to [-1, 1]
    hybrid_score = max(-1.0, min(1.0, hybrid_score))

    # Extended move penalty
    penalty, penalty_reason = compute_extended_move_penalty(indicators, interval)
    if penalty_reason:
        technical_reasons.append(penalty_reason)
    hybrid_score += penalty
    hybrid_score = max(-1.0, min(1.0, hybrid_score))

    # Decision logic
    action = "HOLD"
    confidence = "LOW"
    mode = "unified"

    if abs(hybrid_score) >= 0.60:
        confidence = "HIGH"
        action = "BUY" if hybrid_score > 0 else "SELL"
    elif abs(hybrid_score) >= 0.30:
        confidence = "MEDIUM"
        action = "BUY" if hybrid_score > 0 else "SELL"
    elif always_signal:
        confidence = "LOW"
        action = "BUY" if hybrid_score > 0 else "SELL"

    # Special handling for disagreement
    disagreement = abs(technical_score - sentiment_score) >= cfg.disagreement_watch_gap
    if disagreement:
        if abs(hybrid_score) >= cfg.disagreement_high_override:
            # Strong consensus overrides disagreement
            pass
        else:
            action = "WATCH"
            confidence = "MEDIUM"
            mode = "disagreement"

    # RSI watch threshold
    rsi_val = indicators.get("rsi14")
    gap_val = indicators.get("gap_from_prev_daily_close")
    if isinstance(rsi_val, (int, float)) and rsi_val >= cfg.rsi_watch_threshold:
        if isinstance(gap_val, (int, float)) and gap_val >= cfg.gap_watch_threshold:
            action = "WATCH"
            confidence = "HIGH"
            mode = "rsi_gap_watch"

    reasons = technical_reasons[:]
    if fundamental_reasons:
        reasons.extend(fundamental_reasons)

    # Add sentiment context
    if reddit_summary.count > 0:
        reasons.append(f"Reddit: {reddit_summary.count} mentions ({reddit_summary.score:+.2f})")
    if news_summary.count > 0:
        reasons.append(f"News: {news_summary.count} mentions ({news_summary.score:+.2f})")

    # Top sources
    top_sources = top_sentiment_sources(reddit_samples, news_samples, limit=3)
    top_reddit_sources = top_sentiment_sources_balanced(reddit_display_samples or reddit_samples, limit=3)
    top_news_sources = top_sentiment_sources_balanced(news_samples, limit=3)

    debug = debug_info or {}
    debug.update({
        "reddit_summary": {
            "score": reddit_summary.score,
            "count": reddit_summary.count,
            "recent_1h": reddit_summary.recent_1h_count,
            "recent_6h": reddit_summary.recent_6h_count,
            "weighted_count": reddit_summary.recency_weighted_count,
            "avg_age_hours": reddit_summary.avg_age_hours,
        },
        "news_summary": {
            "score": news_summary.score,
            "count": news_summary.count,
            "recent_1h": news_summary.recent_1h_count,
            "recent_6h": news_summary.recent_6h_count,
            "weighted_count": news_summary.recency_weighted_count,
            "avg_age_hours": news_summary.avg_age_hours,
        },
        "weights": {
            "reddit_sentiment": reddit_weight,
            "news_sentiment": news_weight,
            "technical": tech_weight,
            "fundamental": fund_weight,
            "mention_bonus": mention_bonus,
        },
        "scores": {
            "technical": technical_score,
            "sentiment": sentiment_score,
            "reddit": reddit_summary.score,
            "news": news_summary.score,
            "fundamental": fundamental_score,
            "hybrid": hybrid_score,
        },
        "decision_factors": {
            "disagreement": disagreement,
            "extended_move_penalty": penalty,
            "rsi_watch": rsi_val >= cfg.rsi_watch_threshold if isinstance(rsi_val, (int, float)) else False,
            "gap_watch": gap_val >= cfg.gap_watch_threshold if isinstance(gap_val, (int, float)) else False,
        },
    })

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
        indicators=indicators,
        top_sources=top_sources,
        top_reddit_sources=top_reddit_sources,
        top_news_sources=top_news_sources,
        debug=debug,
    )


def scan_ticker(ticker: str, args: Any, names: Dict[str, str]) -> SignalResult:
    """Scan a single ticker for trading signals."""
    company = names.get(ticker.upper(), "")

    # Fetch technical data
    candles = fetch_yahoo_candles(ticker, interval=args.interval, range_=args.range)

    # Get fundamentals
    fundamentals = get_cached_fundamentals(ticker)
    fundamental_score, fundamental_reasons, fundamental_metrics = compute_fundamental_score(fundamentals.get("metrics", {}))
    fundamentals_as_of = fundamentals.get("as_of")

    # Compute technicals
    benchmark_rel_5 = None  # Could be implemented for relative strength
    gap_from_prev = None
    if args.interval == "1d":
        # For daily scans, check gap from previous close
        daily_candles = candles
    else:
        # For intraday, fetch daily candles for gap calculation
        try:
            daily_candles = fetch_yahoo_candles(ticker, interval="1d", range_="6mo")
            gap_from_prev = gap_from_previous_daily_close(candles[-1].close, daily_candles)
        except Exception:
            daily_candles = candles

    technical_score, indicators, technical_reasons = compute_technicals(
        candles, benchmark_rel_5=benchmark_rel_5, gap_from_prev_daily_close=gap_from_prev
    )

    # Fetch sentiment data
    if args.demo and ticker.upper() == args.demo_ticker.upper():
        reddit_samples, news_samples = demo_samples_for_ticker(ticker)
        reddit_debug = {"demo_mode": True, "ticker": ticker}
        news_debug = {"demo_mode": True, "ticker": ticker}
    else:
        reddit_samples, reddit_debug = fetch_reddit_mentions(
            ticker,
            company_name=company,
            timeout=15,
            use_finbert=args.use_finbert,
            max_age_hours=getattr(args, "reddit_max_age_hours", 120.0),
        )
        news_samples = google_news_rss_query(
            f'"{ticker}" OR "{company}"' if company else ticker,
            timeout=15,
            use_finbert=args.use_finbert,
            ticker=ticker,
            company_name=company,
            max_age_hours=getattr(args, "news_max_age_hours", 720.0),
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

    cfg = HybridConfig()
    return hybrid_decision(
        ticker=ticker.upper(),
        technical_score=technical_score,
        indicators=indicators,
        technical_reasons=technical_reasons,
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