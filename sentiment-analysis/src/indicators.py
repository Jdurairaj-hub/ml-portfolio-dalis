#!/usr/bin/env python3

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import List, Optional

_PRICE_CACHE: dict = {}


@dataclass
class Candle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float


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