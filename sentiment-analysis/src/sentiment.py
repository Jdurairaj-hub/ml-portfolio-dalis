#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import io
import math
import re
import statistics
import time
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from .data import SentimentSample, SentimentSummary, safe_get
from .utils import USER_AGENT, _recency_weight, _source_weight_for_subreddit, dedupe_sentiment_samples

# Constants
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

_FINBERT_MODEL = None
_FINBERT_TOKENIZER = None
_FINBERT_LOAD_ERROR: Optional[str] = None


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
    return min(1.0, max(-1.0, score / max(3.0, math.sqrt(len(words)))))


def _load_finbert_pipeline():
    global _FINBERT_MODEL, _FINBERT_TOKENIZER, _FINBERT_LOAD_ERROR
    if _FINBERT_MODEL is not None and _FINBERT_TOKENIZER is not None:
        return _FINBERT_MODEL, _FINBERT_TOKENIZER
    if _FINBERT_LOAD_ERROR is not None:
        return None
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
        from transformers.utils import logging as hf_logging  # type: ignore

        model_id = "ProsusAI/finbert"
        # Keep demo output clean: suppress HF progress bars / verbose load report during model init.
        import os
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
            return min(1.0, conf)
        if "negative" in label:
            return max(-1.0, -conf)
        # Fallback if labels are ordered [positive, negative, neutral] or similar.
        label_map = {str(v).lower(): i for i, v in id2label.items()}
        pos_idx = label_map.get("positive")
        neg_idx = label_map.get("negative")
        if pos_idx is not None and neg_idx is not None:
            return min(1.0, max(-1.0, float(probs[pos_idx] - probs[neg_idx])))
        return 0.0
    except Exception:
        return None


def sentiment_score_for_text(text: str, use_finbert: bool = False) -> float:
    if use_finbert:
        scored = finbert_sentiment_score(text)
        if scored is not None:
            return scored
    return text_sentiment_score(text)


def google_news_rss_query(
    query: str,
    timeout: int = 15,
    use_finbert: bool = False,
    ticker: Optional[str] = None,
    company_name: Optional[str] = None,
    max_age_hours: float = NEWS_MAX_AGE_HOURS,
) -> List[SentimentSample]:
    import xml.etree.ElementTree as ET

    encoded = requests.utils.quote(query)
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
                ts = int(parsedate_to_datetime(pub_date).astimezone(time.timezone.utc).timestamp())
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
        engagement_bonus = min(0.15, math.log1p(ups + comments) / 12.0)
        score = min(0.95, max(-0.95, score + (engagement_bonus if score >= 0 else -engagement_bonus)))
        score *= _source_weight_for_subreddit(f"r/{sub}: {title}")
        score = min(0.95, max(-0.95, score))
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

    score = min(1.0, max(-1.0, weighted_num / (weighted_den or 1.0)))
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