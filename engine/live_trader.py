"""
LiveTrader - deploys an evolved genome to an Alpaca PAPER account.

Each cycle, per symbol:
  1. Pull recent bars (5-minute or daily, per config.TIMEFRAME) and drop the
     still-forming bar, so the signal is computed exactly like the backtest
     (on a completed close).
  2. If holding: update the peak price, check stop-loss / trailing stop /
     take-profit against the live price -> CLOSE_POSITION; otherwise a SELL
     signal closes the position.
  3. If flat: a BUY signal opens a position sized by position_size_pct.
  4. Each bar's signal is acted on at most once (tracked in live_state.json),
     so a stop-out can't immediately re-buy on the same stale signal.

Stops are checked every poll (not tick-by-tick), so live exits can be later
than the backtest's intrabar assumption. `--dry-run` computes everything but
sends no orders, and works without Alpaca keys.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

import config
from bot.genome import TradingGenome
from bot.indicators import describe_signal, generate_signals
from bot.trader import Action, TradingBot
from broker.alpaca_broker import AccountSummary, AlpacaBroker, BrokerPosition, position_key
from broker.base import BrokerError, MarketDataSource, is_crypto
from utils.display import Display

log = logging.getLogger(__name__)


@dataclass
class LiveDecision:
    symbol: str
    price: float | None
    position_qty: float
    signal_date: str | None
    action: Action
    reason: str
    order_status: str = "-"


@dataclass
class LiveState:
    """Persisted between polls/restarts: trailing-stop peaks and acted-on signal dates."""

    peaks: dict[str, float] = field(default_factory=dict)
    last_signal_date: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "LiveState":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(peaks={k: float(v) for k, v in data.get("peaks", {}).items()},
                       last_signal_date=dict(data.get("last_signal_date", {})))
        except FileNotFoundError:
            return cls()
        except (OSError, ValueError, AttributeError) as exc:
            log.warning("Ignoring unreadable live state %s: %s", path, exc)
            return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"peaks": self.peaks, "last_signal_date": self.last_signal_date}, indent=2), encoding="utf-8")
        os.replace(tmp, path)


def _forming_bar_start(symbol: str) -> pd.Timestamp:
    """
    Start of the bar that is still forming right now, in the same labelling
    the data layer uses: naive UTC timestamps for intraday bars, the exchange-
    local date for daily bars. Everything before it is a completed bar.
    """
    if config.IS_INTRADAY:
        return pd.Timestamp.now(tz="UTC").floor(f"{config.BAR_MINUTES}min").tz_localize(None)
    tz = "UTC" if is_crypto(symbol) else "America/New_York"
    return pd.Timestamp.now(tz=tz).normalize().tz_localize(None)


class LiveTrader:
    def __init__(
        self,
        genome: TradingGenome,
        broker: AlpacaBroker | None,
        data_source: MarketDataSource,
        symbols: list[str],
        display: Display,
        *,
        dry_run: bool = True,
        state_path: Path = config.CHECKPOINT_DIR / config.LIVE_STATE_FILE,
        lookback_days: int = config.LIVE_LOOKBACK_DAYS,
    ) -> None:
        if broker is None and not dry_run:
            raise BrokerError("A connected Alpaca broker is required unless running with --dry-run")
        self.bot = TradingBot(genome, bot_id=0)
        self.genome = genome
        self.broker = broker
        self.data = data_source
        self.symbols = symbols
        self.display = display
        self.dry_run = dry_run
        self.state_path = state_path
        # Daily bars need ~2x warm-up days of history; intraday uses the configured window
        # (yfinance only serves ~60 days of 5-minute bars).
        self.lookback_days = lookback_days if config.IS_INTRADAY else max(lookback_days, genome.warmup_bars * 2)
        self.state = LiveState.load(state_path)

    # ------------------------------------------------------------------ #
    # Loop
    # ------------------------------------------------------------------ #
    def run(self, poll_seconds: int = config.LIVE_POLL_SECONDS, max_cycles: int | None = None) -> None:
        cycle = 0
        try:
            while max_cycles is None or cycle < max_cycles:
                cycle += 1
                decisions = self.run_cycle()
                self.display.live_cycle(cycle, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), decisions)
                self.state.save(self.state_path)
                if max_cycles is not None and cycle >= max_cycles:
                    break
                self.display.info(f"Next check in {poll_seconds}s - press Ctrl+C to stop.")
                time.sleep(poll_seconds)
        except KeyboardInterrupt:
            self.state.save(self.state_path)
            self.display.warn("Live loop stopped by user. Open paper positions were left untouched.")

    def account(self) -> AccountSummary | None:
        return self.broker.get_account_summary() if self.broker else None

    def run_cycle(self) -> list[LiveDecision]:
        account = self.account()
        if account and account.trading_blocked:
            raise BrokerError("Alpaca reports trading is blocked on this paper account")
        positions: dict[str, BrokerPosition] = self.broker.get_positions() if self.broker else {}
        market_open = self.broker.is_market_open() if self.broker else True
        equity = account.equity if account else config.INITIAL_BALANCE

        decisions: list[LiveDecision] = []
        for symbol in self.symbols:
            try:
                decisions.append(self._handle_symbol(symbol, positions.get(position_key(symbol)), market_open, equity, account))
            except BrokerError as exc:
                log.warning("%s: %s", symbol, exc)
                decisions.append(LiveDecision(symbol, None, 0.0, None, Action.HOLD, f"error: {exc}"[:80], "skipped"))
        return decisions

    # ------------------------------------------------------------------ #
    # Per-symbol logic
    # ------------------------------------------------------------------ #
    def _handle_symbol(
        self,
        symbol: str,
        position: BrokerPosition | None,
        market_open: bool,
        equity: float,
        account: AccountSummary | None,
    ) -> LiveDecision:
        qty = position.qty if position else 0.0
        if not is_crypto(symbol) and not market_open:
            return LiveDecision(symbol, position.current_price if position else None, qty, None, Action.HOLD, "US market closed")

        forming = _forming_bar_start(symbol)
        start = (pd.Timestamp.now(tz="UTC") - timedelta(days=self.lookback_days)).date()
        bars = self.data.get_history(symbol, start, None)
        completed = bars[bars.index < forming]
        if len(completed) < self.genome.warmup_bars + 2:
            return LiveDecision(symbol, None, qty, None, Action.HOLD, f"only {len(completed)} bars of history")

        signals = generate_signals(completed, self.genome)
        last = signals.iloc[-1]
        last_bar = completed.index[-1]
        signal_date = f"{last_bar:%m-%d %H:%M} UTC" if config.IS_INTRADAY else str(last_bar.date())
        fresh_signal = self.state.last_signal_date.get(symbol) != signal_date
        price = position.current_price if position else float(bars["close"].iloc[-1])

        # ---- Holding: risk exits first, then strategy exit -------------
        if position is not None:
            peak = max(self.state.peaks.get(symbol, position.avg_entry_price), price)
            self.state.peaks[symbol] = peak
            risk = self.bot.check_risk(position.avg_entry_price, peak, price)
            if risk is not None:
                status = self._close(symbol)
                self.state.peaks.pop(symbol, None)
                return LiveDecision(symbol, price, qty, signal_date, Action.CLOSE_POSITION, risk.value.replace("_", " ").lower(), status)

            action = TradingBot.decide(bool(last["buy"]), bool(last["sell"]), in_position=True)
            if action is Action.SELL and fresh_signal:
                status = self._close(symbol)
                self.state.last_signal_date[symbol] = signal_date
                self.state.peaks.pop(symbol, None)
                return LiveDecision(symbol, price, qty, signal_date, Action.SELL, describe_signal(last), status)
            return LiveDecision(symbol, price, qty, signal_date, Action.HOLD, describe_signal(last))

        # ---- Flat: look for an entry -----------------------------------
        self.state.peaks.pop(symbol, None)
        action = TradingBot.decide(bool(last["buy"]), bool(last["sell"]), in_position=False)
        if action is Action.BUY and fresh_signal:
            notional = equity * self.genome.position_size_pct
            if account is not None:
                notional = min(notional, account.cash if is_crypto(symbol) else account.buying_power)
            if notional < config.MIN_ORDER_NOTIONAL:
                return LiveDecision(symbol, price, 0.0, signal_date, Action.HOLD, "BUY signal but insufficient buying power")
            status = self._buy(symbol, notional)
            self.state.last_signal_date[symbol] = signal_date
            return LiveDecision(symbol, price, 0.0, signal_date, Action.BUY, f"{describe_signal(last)} · ${notional:,.0f}", status)
        return LiveDecision(symbol, price, 0.0, signal_date, Action.HOLD, describe_signal(last))

    # ------------------------------------------------------------------ #
    # Order helpers
    # ------------------------------------------------------------------ #
    def _buy(self, symbol: str, notional: float) -> str:
        if self.dry_run or self.broker is None:
            return f"DRY RUN: would buy ${notional:,.2f}"
        receipt = self.broker.submit_market_order(symbol, "buy", notional=notional)
        log.info("BUY %s $%.2f -> order %s (%s)", symbol, notional, receipt.order_id, receipt.status)
        return f"submitted {receipt.order_id[:8]} ({receipt.status})"

    def _close(self, symbol: str) -> str:
        if self.dry_run or self.broker is None:
            return "DRY RUN: would close position"
        receipt = self.broker.close_position(symbol)
        log.info("CLOSE %s -> order %s (%s)", symbol, receipt.order_id, receipt.status)
        return f"close sent {receipt.order_id[:8]} ({receipt.status})"


def describe_checkpoint(data: dict[str, Any]) -> str:
    survivor = data.get("survivor", {})
    return (f"Generation {data.get('generation')} Alpha · Bot #{survivor.get('bot_id')} · "
            f"DNA {data.get('dna', '?')} · in-sample PnL {survivor.get('net_profit_pct', 0):+.2f}%")
