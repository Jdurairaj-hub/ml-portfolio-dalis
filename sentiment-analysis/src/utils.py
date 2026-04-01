#!/usr/bin/env python3

from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import statistics
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set

from .data import SentimentSample

# Constants
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"


def now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def today_utc_str() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _title_token_set(text: str) -> set[str]:
    t = text.lower()
    # Remove synthetic source prefixes like "r/stocks: ".
    t = re.sub(r"^r/[a-z0-9_+-]+:\s*", "", t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"[^a-z0-9$ ]+", " ", t)
    toks = [w for w in t.split() if len(w) >= 2]
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


def _recency_weight(sample_ts: Optional[int], now_ts: int, half_life_hours: float) -> float:
    if sample_ts is None:
        return 1.0
    age_hours = max(0.0, (now_ts - sample_ts) / 3600.0)
    if half_life_hours <= 0:
        return 1.0
    return 0.5 ** (age_hours / half_life_hours)


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


def gap_from_previous_daily_close(current_price: float, daily_candles: List["Candle"]) -> Optional[float]:
    from .indicators import Candle
    if len(daily_candles) < 2:
        return None
    prev_close = daily_candles[-2].close
    if not prev_close:
        return None
    return current_price / prev_close - 1.0


def compute_technicals(
    candles: List["Candle"],
    benchmark_rel_5: Optional[float] = None,
    gap_from_prev_daily_close: Optional[float] = None,
) -> Tuple[float, Dict[str, Any], List[str]]:
    from .indicators import rsi, ema, sma, atr, vwma

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


def render_console_table(results: List["SignalResult"]) -> str:
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


def save_json_report(path: str, results: List["SignalResult"], meta: Dict[str, Any]) -> None:
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


def append_csv_log(path: str, results: List["SignalResult"], profile: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "ts",
        "ticker",
        "action",
        "confidence",
        "mode",
        "profile",
    ]
    # Add more fields as needed
    fieldnames.extend([
        "hybrid_score", "technical_score", "sentiment_score",
        "reddit_sentiment", "news_sentiment", "mention_count"
    ])

    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in results:
            row = {
                "ts": int(time.time()),
                "ticker": r.ticker,
                "action": r.action,
                "confidence": r.confidence,
                "mode": r.mode,
                "profile": profile,
                "hybrid_score": r.hybrid_score,
                "technical_score": r.technical_score,
                "sentiment_score": r.sentiment_score,
                "reddit_sentiment": r.reddit_sentiment,
                "news_sentiment": r.news_sentiment,
                "mention_count": r.mention_count,
            }
            writer.writerow(row)