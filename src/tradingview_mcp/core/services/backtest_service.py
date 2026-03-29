"""
Backtesting Service for tradingview-mcp.

Runs trading strategy simulations on Yahoo Finance historical OHLCV data.
Pure Python — no pandas, no numpy, no external backtesting libraries.

Supported strategies:
  - rsi         : RSI oversold/overbought
  - bollinger   : Bollinger Band mean reversion
  - macd        : MACD crossover
  - ema_cross   : EMA Golden/Death Cross

Usage:
    from tradingview_mcp.core.services.backtest_service import run_backtest

    result = run_backtest(symbol="AAPL", strategy="rsi", period="1y")
"""
from __future__ import annotations

import json
import math
import urllib.request
from datetime import datetime, timezone
from typing import Optional

from tradingview_mcp.core.services.indicators_calc import (
    calc_rsi,
    calc_bollinger,
    calc_macd,
    calc_ema,
)

_UA = "tradingview-mcp/0.5.0 backtest-bot"

# Yahoo Finance HTTPS works fine without proxy (tested).
# We try direct first, then proxy as fallback to avoid CONNECT tunnel hangs.
_YF_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"


# Allowed periods & intervals for Yahoo Finance
_VALID_PERIODS   = {"1mo", "3mo", "6mo", "1y", "2y"}
_VALID_INTERVALS = {"1d"}   # MVP: daily only


# ─── Data Fetching ────────────────────────────────────────────────────────────

def _fetch_ohlcv(symbol: str, period: str, interval: str = "1d") -> list[dict]:
    """
    Fetch historical OHLCV data from Yahoo Finance.
    Tries direct connection first (faster, avoids proxy CONNECT tunnel hangs).
    Falls back to proxy only if direct fails.
    Returns a list of dicts: {date, open, high, low, close, volume}
    """
    url = f"{_YF_BASE}/{symbol}?interval={interval}&range={period}"
    req = urllib.request.Request(url, headers={"User-Agent": _UA})

    # Try direct first
    data = None
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        pass

    # Fallback: try via proxy
    if data is None:
        try:
            from tradingview_mcp.core.services.proxy_manager import build_opener_with_proxy
            opener = build_opener_with_proxy(_UA)
            with opener.open(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"Both direct and proxy connections failed: {e}")

    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    q = result["indicators"]["quote"][0]

    candles = []
    for i, ts in enumerate(timestamps):
        o = q["open"][i]
        h = q["high"][i]
        l = q["low"][i]
        c = q["close"][i]
        v = q["volume"][i]
        if None in (o, h, l, c):
            continue
        candles.append({
            "date":   datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d"),
            "open":   round(o, 4),
            "high":   round(h, 4),
            "low":    round(l, 4),
            "close":  round(c, 4),
            "volume": v or 0,
        })
    return candles


# ─── Strategy Engines ─────────────────────────────────────────────────────────

def _run_rsi(candles: list[dict], oversold: float = 30, overbought: float = 70, period: int = 14) -> list[dict]:
    closes = [c["close"] for c in candles]
    rsi    = calc_rsi(closes, period)
    trades, position = [], None

    for i in range(1, len(candles)):
        if rsi[i] is None:
            continue
        price = candles[i]["close"]
        date  = candles[i]["date"]

        if position is None and rsi[i] < oversold:
            position = {"entry_date": date, "entry_price": price}

        elif position is not None and rsi[i] > overbought:
            ret_pct = (price - position["entry_price"]) / position["entry_price"] * 100
            trades.append({**position, "exit_date": date, "exit_price": price, "return_pct": round(ret_pct, 3)})
            position = None

    return trades


def _run_bollinger(candles: list[dict], period: int = 20, std_mult: float = 2.0) -> list[dict]:
    closes = [c["close"] for c in candles]
    bb     = calc_bollinger(closes, period, std_mult)
    trades, position = [], None

    for i in range(1, len(candles)):
        if bb["lower"][i] is None:
            continue
        price  = candles[i]["close"]
        date   = candles[i]["date"]
        lower  = bb["lower"][i]
        middle = bb["middle"][i]

        if position is None and price < lower:
            position = {"entry_date": date, "entry_price": price}

        elif position is not None and price > middle:
            ret_pct = (price - position["entry_price"]) / position["entry_price"] * 100
            trades.append({**position, "exit_date": date, "exit_price": price, "return_pct": round(ret_pct, 3)})
            position = None

    return trades


def _run_macd(candles: list[dict], fast: int = 12, slow: int = 26, signal: int = 9) -> list[dict]:
    closes = [c["close"] for c in candles]
    macd   = calc_macd(closes, fast, slow, signal)
    trades, position = [], None

    for i in range(1, len(candles)):
        m = macd["macd"][i]
        s = macd["signal"][i]
        m_prev = macd["macd"][i - 1]
        s_prev = macd["signal"][i - 1]
        if None in (m, s, m_prev, s_prev):
            continue

        price = candles[i]["close"]
        date  = candles[i]["date"]

        # Golden cross: MACD crosses above signal
        if position is None and m_prev < s_prev and m >= s:
            position = {"entry_date": date, "entry_price": price}

        # Death cross: MACD crosses below signal
        elif position is not None and m_prev > s_prev and m <= s:
            ret_pct = (price - position["entry_price"]) / position["entry_price"] * 100
            trades.append({**position, "exit_date": date, "exit_price": price, "return_pct": round(ret_pct, 3)})
            position = None

    return trades


def _run_ema_cross(candles: list[dict], fast_period: int = 20, slow_period: int = 50) -> list[dict]:
    closes   = [c["close"] for c in candles]
    ema_fast = calc_ema(closes, fast_period)
    ema_slow = calc_ema(closes, slow_period)
    trades, position = [], None

    for i in range(1, len(candles)):
        f, s       = ema_fast[i], ema_slow[i]
        f_prev, s_prev = ema_fast[i - 1], ema_slow[i - 1]
        if None in (f, s, f_prev, s_prev):
            continue

        price = candles[i]["close"]
        date  = candles[i]["date"]

        if position is None and f_prev < s_prev and f >= s:          # golden cross
            position = {"entry_date": date, "entry_price": price}

        elif position is not None and f_prev > s_prev and f <= s:    # death cross
            ret_pct = (price - position["entry_price"]) / position["entry_price"] * 100
            trades.append({**position, "exit_date": date, "exit_price": price, "return_pct": round(ret_pct, 3)})
            position = None

    return trades


_STRATEGY_MAP = {
    "rsi":       _run_rsi,
    "bollinger": _run_bollinger,
    "macd":      _run_macd,
    "ema_cross": _run_ema_cross,
}


# ─── Metrics ──────────────────────────────────────────────────────────────────

def _calc_metrics(trades: list[dict], initial_capital: float) -> dict:
    if not trades:
        return {
            "total_trades": 0, "win_rate_pct": 0, "winning_trades": 0,
            "losing_trades": 0, "total_return_pct": 0, "final_capital": initial_capital,
            "avg_gain_pct": 0, "avg_loss_pct": 0, "max_drawdown_pct": 0,
            "profit_factor": 0, "best_trade": None, "worst_trade": None,
        }

    winners = [t for t in trades if t["return_pct"] > 0]
    losers  = [t for t in trades if t["return_pct"] <= 0]

    # Compound capital through trades
    capital = initial_capital
    peak    = capital
    max_dd  = 0.0
    for t in trades:
        capital *= (1 + t["return_pct"] / 100)
        peak = max(peak, capital)
        dd   = (peak - capital) / peak * 100
        max_dd = max(max_dd, dd)

    avg_gain = sum(t["return_pct"] for t in winners) / len(winners) if winners else 0
    avg_loss = sum(t["return_pct"] for t in losers)  / len(losers)  if losers  else 0
    gross_profit = sum(t["return_pct"] for t in winners)
    gross_loss   = abs(sum(t["return_pct"] for t in losers))
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

    best  = max(trades, key=lambda t: t["return_pct"])
    worst = min(trades, key=lambda t: t["return_pct"])

    return {
        "total_trades":     len(trades),
        "winning_trades":   len(winners),
        "losing_trades":    len(losers),
        "win_rate_pct":     round(len(winners) / len(trades) * 100, 1),
        "final_capital":    round(capital, 2),
        "total_return_pct": round((capital - initial_capital) / initial_capital * 100, 2),
        "avg_gain_pct":     round(avg_gain, 2),
        "avg_loss_pct":     round(avg_loss, 2),
        "max_drawdown_pct": round(-max_dd, 2),
        "profit_factor":    profit_factor,
        "best_trade":       {k: best[k] for k in ("entry_date", "exit_date", "return_pct")},
        "worst_trade":      {k: worst[k] for k in ("entry_date", "exit_date", "return_pct")},
    }


# ─── Buy & Hold Benchmark ─────────────────────────────────────────────────────

def _buy_and_hold_return(candles: list[dict]) -> float:
    if len(candles) < 2:
        return 0.0
    first = candles[0]["close"]
    last  = candles[-1]["close"]
    return round((last - first) / first * 100, 2)


# ─── Public API ───────────────────────────────────────────────────────────────

def run_backtest(
    symbol: str,
    strategy: str,
    period: str = "1y",
    initial_capital: float = 10_000.0,
) -> dict:
    """
    Run a backtest for the given symbol and strategy.

    Args:
        symbol:          Yahoo Finance symbol (AAPL, BTC-USD, ^GSPC…)
        strategy:        One of: rsi, bollinger, macd, ema_cross
        period:          Data period: 1mo, 3mo, 6mo, 1y, 2y
        initial_capital: Starting capital in USD

    Returns:
        Full performance report dict.
    """
    strategy = strategy.lower().strip()
    period   = period.lower().strip()

    if strategy not in _STRATEGY_MAP:
        return {"error": f"Unknown strategy '{strategy}'. Choose: {', '.join(_STRATEGY_MAP)}"}
    if period not in _VALID_PERIODS:
        return {"error": f"Invalid period '{period}'. Choose: {', '.join(_VALID_PERIODS)}"}

    try:
        candles = _fetch_ohlcv(symbol, period, "1d")
    except Exception as e:
        return {"error": f"Failed to fetch data for '{symbol}': {e}"}

    if len(candles) < 30:
        return {"error": f"Not enough data for '{symbol}' ({len(candles)} candles). Try a longer period."}

    fn     = _STRATEGY_MAP[strategy]
    trades = fn(candles)
    metrics = _calc_metrics(trades, initial_capital)
    bnh     = _buy_and_hold_return(candles)

    strategy_labels = {
        "rsi":       "RSI (oversold/overbought)",
        "bollinger": "Bollinger Band Mean Reversion",
        "macd":      "MACD Crossover",
        "ema_cross": "EMA 20/50 Golden/Death Cross",
    }

    return {
        "symbol":              symbol.upper(),
        "strategy":            strategy,
        "strategy_label":      strategy_labels[strategy],
        "period":              period,
        "timeframe":           "Daily (1d)",
        "candles_analyzed":    len(candles),
        "date_from":           candles[0]["date"],
        "date_to":             candles[-1]["date"],
        "initial_capital":     round(initial_capital, 2),
        **metrics,
        "buy_and_hold_return_pct": bnh,
        "vs_buy_and_hold_pct": round(metrics["total_return_pct"] - bnh, 2),
        "trade_log":           trades[-10:],    # last 10 trades to keep response manageable
        "data_source":         "Yahoo Finance",
        "disclaimer":          "Past performance does not guarantee future results. This is for educational purposes only.",
        "timestamp":           datetime.now(timezone.utc).isoformat(),
    }


def compare_strategies(
    symbol: str,
    period: str = "1y",
    initial_capital: float = 10_000.0,
) -> dict:
    """
    Run all 4 strategies on the same symbol and return a ranked comparison.
    Fetches OHLCV once and runs all strategies on the cached data.
    """
    try:
        candles = _fetch_ohlcv(symbol, period, "1d")
    except Exception as e:
        return {"error": f"Failed to fetch data for '{symbol}': {e}"}

    if len(candles) < 30:
        return {"error": f"Not enough data for '{symbol}' ({len(candles)} candles)."}

    results = []
    for strat, fn in _STRATEGY_MAP.items():
        trades  = fn(candles)
        metrics = _calc_metrics(trades, initial_capital)
        results.append({
            "strategy":         strat,
            "strategy_label":   {"rsi": "RSI", "bollinger": "Bollinger Band",
                                  "macd": "MACD Crossover", "ema_cross": "EMA 20/50"}[strat],
            "total_return_pct": metrics["total_return_pct"],
            "win_rate_pct":     metrics["win_rate_pct"],
            "total_trades":     metrics["total_trades"],
            "profit_factor":    metrics["profit_factor"],
            "max_drawdown_pct": metrics["max_drawdown_pct"],
        })

    results.sort(key=lambda x: x["total_return_pct"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    bnh = _buy_and_hold_return(candles)

    return {
        "symbol":                  symbol.upper(),
        "period":                  period,
        "timeframe":               "Daily (1d)",
        "candles_analyzed":        len(candles),
        "date_from":               candles[0]["date"],
        "date_to":                 candles[-1]["date"],
        "initial_capital":         round(initial_capital, 2),
        "buy_and_hold_return_pct": bnh,
        "winner":                  results[0]["strategy"] if results else None,
        "ranking":                 results,
        "disclaimer":              "Past performance does not guarantee future results.",
        "timestamp":               datetime.now(timezone.utc).isoformat(),
    }
