#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict, List, Optional

from .scanner import scan_ticker, company_name_map
from .utils import now_utc_iso, render_console_table, save_json_report, append_csv_log


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
    p.add_argument("--reddit-max-age-hours", type=float, default=120.0, help="Max age for Reddit posts (hours)")
    p.add_argument("--news-max-age-hours", type=float, default=720.0, help="Max age for news posts (hours)")
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

    results = []
    errors = []
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
    if args.use_finbert:
        from .sentiment import _FINBERT_LOAD_ERROR
        if _FINBERT_LOAD_ERROR:
            finbert_label += f" (fallback: {_FINBERT_LOAD_ERROR})"
    print(f"Hybrid Signal Scan @ {now_utc_iso()} [{mode_label}] ({finbert_label}, unified-policy, free-source MVP)")
    if args.profile != "balanced":
        print(f"Note: --profile={args.profile} is currently compatibility-only (no effect on decisions).")
    print(render_console_table(results))
    print()

    # Show detailed results
    for i, r in enumerate(results):
        print(f"#{i+1}: {r.ticker} - {r.action} ({r.confidence}) [Hybrid: {r.hybrid_score:+.2f}]")
        print(f"  Technical: {r.technical_score:+.2f} | Sentiment: {r.sentiment_score:+.2f} | Mentions: {r.mention_count}")
        print(f"  Close: ${r.indicators.get('close', 'N/A')} | RSI: {r.indicators.get('rsi14', 'N/A')} | Mode: {r.mode}")

        # Show top reasons
        top_reasons = r.reasons[:args.top_n_reasons]
        if top_reasons:
            print("  Reasons:")
            for reason in top_reasons:
                print(f"    • {reason}")

        # Show top sources if requested
        if args.show_sources and r.top_sources:
            print("  Top Sources:")
            for source in r.top_sources[:args.source_limit]:
                score = source.get("score", 0)
                title = source.get("title", "")[:80]
                print(f"    • [{source.get('source', 'unknown')}] {score:+.2f}: {title}")

        # Reddit debug info
        if args.reddit_debug and "reddit" in r.debug:
            reddit_debug = r.debug["reddit"]
            print("  Reddit Debug:")
            print(f"    Raw results: {reddit_debug.get('raw_results', 0)}")
            print(f"    Filtered: subreddit={reddit_debug.get('filtered_subreddit', 0)}, empty={reddit_debug.get('filtered_empty_text', 0)}, no_match={reddit_debug.get('filtered_no_match', 0)}, stale={reddit_debug.get('filtered_stale', 0)}")
            print(f"    Kept: {reddit_debug.get('kept', 0)}")
            if reddit_debug.get("subreddits_seen"):
                top_subs = sorted(reddit_debug["subreddits_seen"].items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"    Top subreddits: {', '.join(f'{sub}({count})' for sub, count in top_subs)}")

        print()

    # Error summary
    if errors:
        print(f"Errors ({len(errors)}):")
        for error in errors:
            print(f"  {error}")
        print()

    # Save reports if requested
    meta = {
        "mode": mode_label.lower(),
        "finbert": args.use_finbert,
        "interval": args.interval,
        "range": args.range,
        "tickers_scanned": len(results),
        "errors": len(errors),
        "profile": args.profile,
    }

    if args.json_out:
        save_json_report(args.json_out, results, meta)
        print(f"JSON report saved to: {args.json_out}")

    if args.csv_log:
        append_csv_log(args.csv_log, results, args.profile)
        print(f"CSV log appended to: {args.csv_log}")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())