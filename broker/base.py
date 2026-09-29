"""
Shared broker primitives: market-data container, cost model and interfaces.

Both brokers speak the same language (`MarketDataSource.get_history` returns a
lower-case OHLCV DataFrame indexed by tz-naive dates), so the rest of the code
never cares whether bars came from yfinance, Alpaca or the synthetic generator.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

import pandas as pd

log = logging.getLogger(__name__)

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
Side = Literal["buy", "sell"]
DateLike = str | date | datetime | pd.Timestamp


class BrokerError(RuntimeError):
    """Any failure talking to a data provider or broker."""


class InsufficientDataError(BrokerError):
    """Not enough bars to warm up indicators and still have something to trade."""


def is_crypto(symbol: str) -> bool:
    """yfinance-style crypto tickers look like BTC-USD / ETH-USD."""
    return symbol.upper().endswith(("-USD", "/USD", "-USDT", "/USDT"))


def normalize_ohlcv(df: pd.DataFrame, symbol: str, intraday: bool | None = None) -> pd.DataFrame:
    """
    Lower-case columns, tz-naive index, drop junk rows, sort.

    Daily bars are labelled by their exchange-local DATE. Intraday bars keep
    their full timestamp, converted to UTC so stocks and 24/7 crypto line up.
    """
    if intraday is None:
        import config

        intraday = config.IS_INTRADAY
    if df is None or df.empty:
        raise BrokerError(f"No bars returned for {symbol}")
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    missing = [c for c in OHLCV_COLUMNS[:4] if c not in out.columns]
    if missing:
        raise BrokerError(f"{symbol}: bars missing columns {missing}")
    if "volume" not in out.columns:
        out["volume"] = 0.0
    out = out[list(OHLCV_COLUMNS)].astype(float)

    idx = pd.DatetimeIndex(pd.to_datetime(out.index, utc=intraday))
    if intraday:
        out.index = idx.tz_convert("UTC").tz_localize(None)  # naive UTC timestamps
    else:
        if idx.tz is not None:
            idx = idx.tz_localize(None)  # keep local wall-clock date, drop tz
        out.index = idx.normalize()
    out.index.name = "date"

    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out[(out["close"] > 0) & (out["open"] > 0)]
    return out


@dataclass(frozen=True)
class CostModel:
    """Commission + slippage applied to every simulated fill."""

    commission_pct: float = 0.0
    slippage_pct: float = 0.0
    crypto_commission_pct: float | None = None   # None = same as commission_pct

    @classmethod
    def from_config(cls) -> "CostModel":
        import config

        return cls(config.COMMISSION_PCT, config.SLIPPAGE_PCT, config.CRYPTO_COMMISSION_PCT)

    def fill_price(self, side: Side, reference_price: float) -> float:
        """Adverse slippage: buys fill higher, sells fill lower."""
        if side == "buy":
            return reference_price * (1.0 + self.slippage_pct)
        return reference_price * (1.0 - self.slippage_pct)

    def commission_rate(self, symbol: str = "") -> float:
        if self.crypto_commission_pct is not None and is_crypto(symbol):
            return self.crypto_commission_pct
        return self.commission_pct

    def commission(self, notional: float, symbol: str = "") -> float:
        return abs(notional) * self.commission_rate(symbol)


@dataclass(frozen=True)
class Windows:
    """Train / test date ranges produced by `MarketData.split`."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp | None
    test_end: pd.Timestamp | None

    @property
    def has_test(self) -> bool:
        return self.test_start is not None and self.test_end is not None


@dataclass
class MarketData:
    """Daily bars for every symbol in the arena."""

    bars: dict[str, pd.DataFrame]
    source: str
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.bars:
            raise BrokerError("MarketData needs at least one symbol")
        self._calendar = pd.DatetimeIndex(
            sorted(set().union(*(df.index for df in self.bars.values())))
        )

    @property
    def symbols(self) -> list[str]:
        return list(self.bars)

    @property
    def calendar(self) -> pd.DatetimeIndex:
        """Union of all trading dates (crypto trades weekends, stocks don't)."""
        return self._calendar

    def split(self, train_fraction: float, warmup_bars: int) -> Windows:
        """
        Reserve `warmup_bars` per symbol for indicator warm-up, then split the
        remaining calendar into an in-sample (evolution) window and an
        out-of-sample (honest test) window.
        """
        first_tradeable = max(
            df.index[min(warmup_bars, len(df) - 1)] for df in self.bars.values()
        )
        tradeable = self.calendar[self.calendar >= first_tradeable]
        if len(tradeable) < 60:
            raise InsufficientDataError(
                f"Only {len(tradeable)} tradeable days after a {warmup_bars}-bar warm-up. "
                "Move HISTORY_START earlier."
            )
        cut = int(len(tradeable) * train_fraction)
        cut = min(max(cut, 30), len(tradeable))
        train = tradeable[:cut]
        test = tradeable[cut:]
        if len(test) < 20:
            return Windows(train[0], tradeable[-1], None, None)
        return Windows(train[0], train[-1], test[0], test[-1])


class MarketDataSource(ABC):
    """Anything that can hand us historical daily bars."""

    name: str = "abstract"

    @abstractmethod
    def get_history(self, symbol: str, start: DateLike, end: DateLike | None = None) -> pd.DataFrame:
        """Return normalised OHLCV bars for `symbol` between `start` and `end` (inclusive)."""
