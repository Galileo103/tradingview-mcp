"""
Technical Indicators Calculator — pure Python stdlib, zero dependencies.

All functions take a list of float closing prices (or OHLCV dicts)
and return computed indicator values.
"""
from __future__ import annotations

import math
from typing import Optional


# ─── EMA ──────────────────────────────────────────────────────────────────────

def calc_ema(closes: list[float], period: int) -> list[Optional[float]]:
    """Exponential Moving Average. First (period-1) values are None."""
    result: list[Optional[float]] = [None] * len(closes)
    if len(closes) < period:
        return result
    k = 2 / (period + 1)
    # seed with SMA
    sma = sum(closes[:period]) / period
    result[period - 1] = sma
    for i in range(period, len(closes)):
        result[i] = closes[i] * k + result[i - 1] * (1 - k)
    return result


# ─── SMA ──────────────────────────────────────────────────────────────────────

def calc_sma(closes: list[float], period: int) -> list[Optional[float]]:
    """Simple Moving Average."""
    result: list[Optional[float]] = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        result[i] = sum(closes[i - period + 1 : i + 1]) / period
    return result


# ─── RSI ──────────────────────────────────────────────────────────────────────

def calc_rsi(closes: list[float], period: int = 14) -> list[Optional[float]]:
    """
    Relative Strength Index (Wilder's smoothing).
    First (period) values are None.
    """
    result: list[Optional[float]] = [None] * len(closes)
    if len(closes) < period + 1:
        return result

    gains, losses = [], []
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100 - (100 / (1 + rs))

    for i in range(period + 1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gain = max(diff, 0)
        loss = max(-diff, 0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100 - (100 / (1 + rs))

    return result


# ─── Bollinger Bands ──────────────────────────────────────────────────────────

def calc_bollinger(
    closes: list[float], period: int = 20, std_mult: float = 2.0
) -> dict[str, list[Optional[float]]]:
    """
    Bollinger Bands.
    Returns dict with 'upper', 'middle' (SMA), 'lower' lists.
    """
    middle = calc_sma(closes, period)
    upper: list[Optional[float]] = [None] * len(closes)
    lower: list[Optional[float]] = [None] * len(closes)

    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1 : i + 1]
        mean = middle[i]
        variance = sum((x - mean) ** 2 for x in window) / period
        std = math.sqrt(variance)
        upper[i] = mean + std_mult * std
        lower[i] = mean - std_mult * std

    return {"upper": upper, "middle": middle, "lower": lower}


# ─── MACD ─────────────────────────────────────────────────────────────────────

def calc_macd(
    closes: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, list[Optional[float]]]:
    """
    MACD = EMA(fast) - EMA(slow).
    Signal = EMA(MACD, signal_period).
    Histogram = MACD - Signal.
    """
    ema_fast = calc_ema(closes, fast)
    ema_slow = calc_ema(closes, slow)

    n = len(closes)
    macd_line: list[Optional[float]] = [None] * n
    for i in range(n):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line[i] = ema_fast[i] - ema_slow[i]

    # Signal line = EMA of MACD line (only over non-None values)
    signal_line: list[Optional[float]] = [None] * n
    histogram: list[Optional[float]] = [None] * n

    # Find first valid macd index
    macd_values = [(i, v) for i, v in enumerate(macd_line) if v is not None]
    if len(macd_values) >= signal:
        # Compute EMA of macd values
        start_idx = macd_values[0][0]
        macd_only = [v for _, v in macd_values]
        sig_ema = calc_ema(macd_only, signal)
        for j, (orig_i, _) in enumerate(macd_values):
            if sig_ema[j] is not None:
                signal_line[orig_i] = sig_ema[j]
                histogram[orig_i] = macd_line[orig_i] - sig_ema[j]

    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}
