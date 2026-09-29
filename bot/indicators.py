"""
Vectorised technical indicators and the genome-driven signal generator.

`generate_signals()` is the single source of truth for entry/exit rules - it is
used by the historical backtester AND by the live Alpaca loop, so a bot behaves
identically in both worlds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bot.genome import TradingGenome

REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average (NaN until `period` bars exist)."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (NaN until `period` bars exist)."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's Relative Strength Index, 0-100."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # No losses in the window -> maximally overbought; keep NaN during warm-up.
    return out.where(avg_loss != 0.0, 100.0).where(avg_gain.notna())


def macd(close: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, signal_line, macd_line - signal_line


def generate_signals(bars: pd.DataFrame, genome: TradingGenome) -> pd.DataFrame:
    """
    Evaluate the genome's rules on every bar (signals are known at the bar CLOSE).

    Long-only rule set:
      BUY  (when flat)
        - Momentum entry: fast SMA > slow SMA  AND  MACD crosses above signal  AND  RSI < overbought
        - Reversion entry: RSI crosses back UP through the oversold level
      SELL (when long)
        - RSI > overbought  OR  fast SMA crosses below slow SMA
        - OR MACD crosses below signal while the trend is already down
    Stop-loss / take-profit / trailing-stop exits are handled bar-by-bar by the
    trader (they fire as CLOSE_POSITION, not as a SELL signal).
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in bars.columns]
    if missing:
        raise ValueError(f"bars is missing columns: {missing}")

    g = genome
    close = bars["close"].astype(float)

    fast = sma(close, g.fast_sma)
    slow = sma(close, g.slow_sma)
    rsi_now = rsi(close, g.rsi_period)
    rsi_prev = rsi_now.shift(1)
    macd_line, signal_line, hist = macd(close, g.macd_fast, g.macd_slow, g.macd_signal)

    valid = (
        fast.notna() & slow.notna() & rsi_prev.notna()
        & signal_line.notna() & signal_line.shift(1).notna()
    )

    trend_up = fast > slow
    macd_above = macd_line > signal_line
    macd_cross_up = macd_above & ~macd_above.shift(1, fill_value=False)
    macd_cross_down = ~macd_above & macd_above.shift(1, fill_value=False)
    sma_cross_down = ~trend_up & trend_up.shift(1, fill_value=False)
    rsi_reversal = (rsi_prev < g.rsi_oversold) & (rsi_now >= g.rsi_oversold)

    momentum_entry = trend_up & macd_cross_up & (rsi_now < g.rsi_overbought)
    buy = valid & (momentum_entry | rsi_reversal)

    rsi_exit = rsi_now > g.rsi_overbought
    trend_exit = sma_cross_down | (macd_cross_down & ~trend_up)
    sell = valid & (rsi_exit | trend_exit)

    return pd.DataFrame(
        {
            "close": close,
            "fast_sma": fast,
            "slow_sma": slow,
            "rsi": rsi_now,
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": hist,
            "valid": valid,
            "buy": buy,
            "sell": sell,
            "momentum_entry": valid & momentum_entry,
            "reversion_entry": valid & rsi_reversal,
            "rsi_exit": valid & rsi_exit,
            "trend_exit": valid & trend_exit,
        },
        index=bars.index,
    )


def describe_signal(row: pd.Series) -> str:
    """Human-readable reason for the signal on one row (used in live mode)."""
    if not bool(row.get("valid", False)):
        return "warming up indicators"
    reasons: list[str] = []
    if row.get("momentum_entry"):
        reasons.append("trend up + MACD bull cross")
    if row.get("reversion_entry"):
        reasons.append("RSI bounced off oversold")
    if row.get("rsi_exit"):
        reasons.append("RSI overbought")
    if row.get("trend_exit"):
        reasons.append("trend broke down")
    return ", ".join(reasons) if reasons else f"RSI {row.get('rsi', float('nan')):.0f}, no trigger"
