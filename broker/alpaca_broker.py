"""
AlpacaBroker - live PAPER execution through alpaca-py.

Safety first: the TradingClient is always created with paper=True and there is
no switch to change it. alpaca-py is imported lazily so Mode 1 (historical
evolution) works even if alpaca-py is not installed.

Symbol conventions:
    config / yfinance : "AAPL", "BTC-USD"
    Alpaca orders     : "AAPL", "BTC/USD"
    Alpaca positions  : "AAPL", "BTCUSD"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from broker.base import (
    BrokerError,
    DateLike,
    MarketDataSource,
    Side,
    is_crypto,
    normalize_ohlcv,
)

log = logging.getLogger(__name__)


def to_alpaca_symbol(symbol: str) -> str:
    """'BTC-USD' -> 'BTC/USD'; stocks unchanged."""
    s = symbol.upper()
    if is_crypto(s) and "-" in s:
        base, quote = s.split("-", 1)
        return f"{base}/{quote}"
    return s


def position_key(symbol: str) -> str:
    """Alpaca reports crypto positions without the slash ('BTCUSD')."""
    return symbol.upper().replace("/", "").replace("-", "")


@dataclass(frozen=True)
class AccountSummary:
    equity: float
    cash: float
    buying_power: float
    status: str
    trading_blocked: bool
    last_equity: float | None = None   # equity at yesterday's close (for the daily loss limit)


@dataclass(frozen=True)
class BrokerPosition:
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float


@dataclass(frozen=True)
class OrderReceipt:
    symbol: str
    side: Side
    order_id: str
    status: str
    notional: float | None = None
    qty: float | None = None


class AlpacaBroker(MarketDataSource):
    """Thin, defensive wrapper around alpaca-py's trading + market-data clients."""

    name = "alpaca"

    def __init__(self, api_key: str | None, secret_key: str | None) -> None:
        if not api_key or not secret_key:
            raise BrokerError(
                "Alpaca API keys missing. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
                "(environment or .env file) using keys from your PAPER trading dashboard."
            )
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as exc:
            raise BrokerError("alpaca-py is not installed (pip install alpaca-py)") from exc

        self._api_key = api_key
        self._secret_key = secret_key
        # paper=True is hard-coded on purpose - this project never touches real money.
        self._trading = TradingClient(api_key, secret_key, paper=True)
        self._stock_data: Any = None
        self._crypto_data: Any = None

    # ------------------------------------------------------------------ #
    # Account / clock
    # ------------------------------------------------------------------ #
    def get_account_summary(self) -> AccountSummary:
        acct = self._call("get_account", self._trading.get_account)
        return AccountSummary(
            equity=float(acct.equity or 0.0),
            cash=float(acct.cash or 0.0),
            buying_power=float(acct.buying_power or 0.0),
            status=str(getattr(acct.status, "value", acct.status)),
            trading_blocked=bool(acct.trading_blocked),
            last_equity=float(acct.last_equity) if getattr(acct, "last_equity", None) else None,
        )

    def is_market_open(self) -> bool:
        """US equity session status (crypto trades 24/7 and ignores this)."""
        clock = self._call("get_clock", self._trading.get_clock)
        return bool(clock.is_open)

    # ------------------------------------------------------------------ #
    # Positions / orders
    # ------------------------------------------------------------------ #
    def get_positions(self) -> dict[str, BrokerPosition]:
        """All open positions keyed by normalised symbol ('AAPL', 'BTCUSD')."""
        raw = self._call("get_all_positions", self._trading.get_all_positions)
        out: dict[str, BrokerPosition] = {}
        for p in raw:
            qty = float(p.qty)
            if str(getattr(p.side, "value", p.side)).lower() == "short" and qty > 0:
                qty = -qty  # shorts are negative quantities everywhere in this project
            out[position_key(p.symbol)] = BrokerPosition(
                symbol=p.symbol,
                qty=qty,
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price or p.avg_entry_price),
                market_value=float(p.market_value or 0.0),
                unrealized_pl=float(p.unrealized_pl or 0.0),
            )
        return out

    def get_position(self, symbol: str) -> BrokerPosition | None:
        return self.get_positions().get(position_key(symbol))

    def submit_market_order(self, symbol: str, side: Side, *, notional: float | None = None, qty: float | None = None) -> OrderReceipt:
        """Market order by dollar notional (fractional) or share quantity."""
        if (notional is None) == (qty is None):
            raise BrokerError("Provide exactly one of notional or qty")
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        crypto = is_crypto(symbol)
        request = MarketOrderRequest(
            symbol=to_alpaca_symbol(symbol),
            notional=round(notional, 2) if notional is not None else None,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            # Fractional/notional equity orders must be DAY; crypto requires GTC/IOC.
            time_in_force=TimeInForce.GTC if crypto else TimeInForce.DAY,
        )
        order = self._call(f"submit_order {symbol}", self._trading.submit_order, order_data=request)
        return OrderReceipt(
            symbol=symbol,
            side=side,
            order_id=str(order.id),
            status=str(getattr(order.status, "value", order.status)),
            notional=notional,
            qty=qty,
        )

    def close_position(self, symbol: str) -> OrderReceipt:
        """Liquidate the entire position in `symbol`."""
        order = self._call(f"close_position {symbol}", self._trading.close_position, position_key(symbol))
        return OrderReceipt(
            symbol=symbol,
            side="sell",
            order_id=str(getattr(order, "id", "")),
            status=str(getattr(getattr(order, "status", ""), "value", getattr(order, "status", ""))),
        )

    # ------------------------------------------------------------------ #
    # Market data
    # ------------------------------------------------------------------ #
    def get_history(self, symbol: str, start: DateLike, end: DateLike | None = None) -> pd.DataFrame:
        """Bars at config.BAR_INTERVAL from Alpaca (IEX feed for stocks - free tier friendly)."""
        import config
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        timeframe = TimeFrame(config.BAR_MINUTES, TimeFrameUnit.Minute) if config.IS_INTRADAY else TimeFrame.Day

        start_dt = pd.Timestamp(start).to_pydatetime().replace(tzinfo=timezone.utc)
        end_dt = (pd.Timestamp(end).to_pydatetime().replace(tzinfo=timezone.utc)
                  if end is not None else datetime.now(timezone.utc) - timedelta(minutes=16))
        alpaca_symbol = to_alpaca_symbol(symbol)

        if is_crypto(symbol):
            from alpaca.data.historical import CryptoHistoricalDataClient
            from alpaca.data.requests import CryptoBarsRequest

            if self._crypto_data is None:
                self._crypto_data = CryptoHistoricalDataClient(self._api_key, self._secret_key)
            req = CryptoBarsRequest(symbol_or_symbols=alpaca_symbol, timeframe=timeframe, start=start_dt, end=end_dt)
            bars = self._call(f"crypto bars {symbol}", self._crypto_data.get_crypto_bars, req)
            tz = "UTC"
        else:
            from alpaca.data.enums import Adjustment, DataFeed
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest

            if self._stock_data is None:
                self._stock_data = StockHistoricalDataClient(self._api_key, self._secret_key)
            req = StockBarsRequest(
                symbol_or_symbols=alpaca_symbol,
                timeframe=timeframe,
                start=start_dt,
                end=end_dt,
                feed=DataFeed.IEX,
                adjustment=Adjustment.ALL,
            )
            bars = self._call(f"stock bars {symbol}", self._stock_data.get_stock_bars, req)
            tz = "America/New_York"

        df = bars.df
        if df is None or df.empty:
            raise BrokerError(f"Alpaca returned no bars for {symbol}")
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(alpaca_symbol, level="symbol")
        df.index = pd.DatetimeIndex(df.index).tz_convert(tz)  # label daily bars by exchange date
        return normalize_ohlcv(df, symbol)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _call(label: str, fn, *args, **kwargs):
        """Run an alpaca-py call and convert every failure into a BrokerError."""
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # APIError, requests errors, pydantic validation...
            try:
                status = getattr(exc, "status_code", None)
            except Exception:  # APIError.status_code can itself raise without an HTTP response
                status = None
            detail = f" (HTTP {status})" if status else ""
            raise BrokerError(f"Alpaca {label} failed{detail}: {exc}") from exc
