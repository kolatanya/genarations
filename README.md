# 🧬 Evolution Arena: Genetic Algorithm Trading Bots

Six trading bots enter. **One survives.** The survivor clones itself with random mutations, and the cycle repeats: natural selection applied to trading strategies, rendered for a screen recording.

> ⚠️ **Educational and entertainment project. Not financial advice.** Mode 2 is hard-wired to Alpaca's **paper** (fake-money) endpoint. There is no real-money switch.

---

## 🧫 Evolution Lab (v2), menu options 3–7

v1 (menu 1–2) tunes the numbers in one fixed strategy. **The Lab evolves the strategies themselves**, on much more data, and spends most of its effort proving whether the result is skill or luck.

| Menu | Mode | What it does |
|---|---|---|
| 3 | `--mode lab` | Island evolution over **2 years × 20 stocks** of 5-minute bars (Alpaca) |
| 4 | `--mode monkey` | **Luck detector**: the same evolution on real vs shuffled "monkey" markets |
| 5 | `--mode walkforward` | Evolve on the past, trade the next month, slide forward, repeat |
| 6 | `--mode lab-live` | The Lab's **ensemble** trades your Alpaca paper account |
| 7 | `--mode reevolve` | Weekly: evolve a **shadow challenger** on the newest data |
| – | `--mode lab-promote` | Promote the shadow challenger to live (the old ensemble is backed up) |

**What's inside:**

- **Data:** 2 years of consolidated (SIP) 5-minute bars for 20 liquid stocks plus SPY/QQQ from your free Alpaca account. Also the VIX, earnings dates, Fed meeting days, and headline sentiment from Alpaca's free news feed. Everything is cached after the first download.
- **Senses (33 features):** returns over 4 horizons, RSI ×3, distance from 4 moving averages, 2 MACDs, VWAP distance, ATR volatility, volume spikes, time of day, 1-hour and daily trend, opening gap, SPY/QQQ mood, VIX level and change, cross-stock strength rank, earnings/Fed flags, and news sentiment and volume. Every feature is tested to never peek at the future.
- **Bots invent their rules:** genetic-programming trees such as `rsi_14 < 31 AND (vwap_dev < -0.2% OR NOT fomc_day)`, for long entries, short entries and exits. Risk genes cover ATR stops, targets and trailing stops, volatility sizing, trading hours, a daily loss limit, and avoiding event days.
- **Evolution:** 4 islands, each with 1 Alpha and 5 challengers:
  - 3 mutants
  - 1 crossover child (subtree grafting)
  - 1 wildcard: a migrant every 5 generations, a Hall-of-Fame veteran, a random newcomer, or an explorer

  Each gene has its own self-adapting mutation strength, and each island has adaptive mutation. Other rules:
  - Duplicates are re-mutated.
  - Two islands evolving the same bot, or a 20-generation reign, triggers a **meteor strike** that re-seeds the island.
- **The crown is hard to win:** a challenger must out-score the Alpha on 4 random eras of 6, with jittered costs (noise injection). It must also survive the **wobble test** (genes nudged ±10%) and beat the Alpha on **unseen validation data**.
- **Fitness:** Sharpe, then consistency across eras and profit in up/sideways/down markets, then drawdown. It also includes a complexity penalty and the ~20-minute trading pace.
- **Honest scoring:** a 60/20/20 train/validation/**final test** split; the final test isn't touched until the end. The final report adds a Monte Carlo luck range, the probability of profit, and the **deflated Sharpe ratio**, which corrects for how many bots were tried. It also keeps a Pareto archive of trade-offs.
- **Realism:** spreads and slippage scale with volatility, positions close at 15:55 ET, short selling is allowed, and a daily loss limit applies.
- **Live:** a Hall-of-Fame **ensemble vote**, weighted by how each member performs in today's market regime. Safety systems:
  - a daily loss limit
  - flatten before the close
  - a **drift alarm** that benches the bot if live results fall below the backtest's worst 5%
  - a **kill switch** (create `checkpoints/lab/STOP`)
  - a **shadow challenger** that paper-trades virtually and is recommended for promotion only if it out-earns the live ensemble for 10 days

**Choices made for this build:**

- **Rule trees rather than neural networks (NEAT).** Trees are readable, so you can show on camera exactly what a bot learned. With this much data, neural nets overfit badly.
- **A free built-in word list for news sentiment, not a paid AI model.**
- **A virtual shadow book instead of a second paid/paper account.**
- **The weekly re-evolve is a command you run,** not a scheduled task. Ask if you want it automated.

---

## The survival loop

```
            ┌──────────────────────────────────────────────────────────┐
            │  GENERATION N                                            │
            │  👑 Alpha  +  🧪 Clone  🧪 Clone  🧪 Clone  🧪 Clone  🧪 Clone │
            └──────────────────────────────────────────────────────────┘
                                   │  all 6 trade the same market data
                                   ▼
                  Fitness = Net Profit % × Sharpe × (1 − Max Drawdown)
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
          👑 1 SURVIVOR                    ☠ 5 ELIMINATED
          saved to checkpoints/            permanently deleted from memory
          alpha_gen_N.json                 (verified by garbage-collection check)
                    │
                    ▼  replicates 5× with Gaussian DNA mutations (±15%)
            GENERATION N+1  ...and forever
```

1. **Generation creation:** 1 Alpha + 5 mutated clones = 6 bots.
2. **Trading evaluation:** every bot trades AAPL, NVDA, TSLA and BTC-USD independently with its own $10,000.
3. **Extinction:** bots are ranked by fitness. The top bot lives. **The other 5 are deleted.** The engine checks with weak references that every eliminated bot really has been garbage-collected, and stops with an error if any survive.
4. **Replication:** the survivor becomes the new Alpha and spawns 5 clones. Each clone's genes drift by Gaussian noise: `new = old × (1 + N(0, 0.15))`.

A challenger must **strictly beat** the Alpha to take the crown (ties go to the incumbent).

---

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python main.py                                   # interactive menu
python main.py --mode evolve --generations 3 --fast --seed 42   # 3-generation dry run
```

Requires Python 3.10+. Tested on Python 3.14 with pandas 3, numpy 2.5, yfinance 1.7 and alpaca-py 0.44.

### Modes & flags

| Command | What it does |
|---|---|
| `python main.py` | Menu: **1** Fast Evolution Loop, **2** Live Paper Execution |
| `--mode evolve --generations 30` | Mode 1: evolve over historical data (20–50 recommended) |
| `--resume` | Continue the lineage from `checkpoints/latest_alpha.json` (generation numbers keep counting) |
| `--random-start` | Generation 1 Alpha gets random DNA instead of textbook settings |
| `--seed 42` | Reproducible episode: same seed and same data give the same show |
| `--synthetic` | Offline rehearsal on clearly labelled random-walk prices |
| `--fast` | Removes the dramatic pauses (for testing) |
| `--mode live --dry-run --once` | Mode 2: compute today's signals once, send **no** orders (no keys needed) |
| `--mode live` | Mode 2: trade the champion on your Alpaca **paper** account |
| `--checkpoint path.json` | Deploy a specific genome instead of the latest Alpha |
| `--poll 300` | Seconds between live cycles |

---

## Alpaca paper trading setup (Mode 2)

1. Create a free account at [alpaca.markets](https://alpaca.markets).
2. In the dashboard, switch to your **Paper Trading** account (account selector, top-left).
3. Under **API Keys**, click **Generate New Keys**. Copy the key and the secret; the secret is shown only once.
4. Copy `.env.example` to `.env` and paste the keys:
   ```
   ALPACA_API_KEY=PK...
   ALPACA_SECRET_KEY=...
   ```
   Environment variables work too, and so do the official `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` names.
5. Evolve a champion first (Mode 1), then check the signals without sending orders:
   ```bash
   python main.py --mode live --dry-run --once
   ```
6. Go live on paper: `python main.py --mode live`. Stop with **Ctrl+C**. Open positions are left as they are.

**How live mode behaves:**
- Signals are computed on the last **completed** daily bar, exactly like the backtest, and each daily signal is acted on at most once.
- Stop-loss, trailing-stop and take-profit are checked on every poll against the live price. The backtest checks them intrabar, so live exits can happen later.
- Stocks only trade while the US market is open. BTC-USD trades 24/7.
- Orders are notional (fractional) market orders: `DAY` for stocks, `GTC` for crypto.
- By default signal data comes from yfinance, which keeps it consistent with training. Set `LIVE_DATA_SOURCE=alpaca` to use Alpaca's free IEX feed instead.
- Trailing-stop peaks and acted-on signal dates persist in `checkpoints/live_state.json`.

---

## How a bot thinks (for viewers)

### The genome (DNA)

| Gene | Range | Textbook start |
|---|---|---|
| `rsi_period` | 5–30 | 14 |
| `rsi_overbought` / `rsi_oversold` | 55–90 / 10–45 | 70 / 30 |
| `fast_sma` / `slow_sma` | 5–60 / 20–200 | 20 / 50 |
| `macd_fast` / `macd_slow` / `macd_signal` | 4–20 / 15–50 / 3–15 | 12 / 26 / 9 |
| `stop_loss_pct` | 1–15% | 5% |
| `take_profit_pct` | 2–50% | 12% |
| `trailing_stop_pct` | 1–20% | 6% |
| `position_size_pct` | 5–50% of equity | 25% |

After every mutation, genes are clamped to these ranges and repaired so they stay sensible (fast SMA < slow SMA, MACD fast < slow). A mutant can never be an invalid strategy.

### The trading rules (long only, daily bars)

| Action | When |
|---|---|
| **BUY** | *Momentum:* fast SMA > slow SMA **and** MACD crosses above its signal **and** RSI < overbought. *Or reversion:* RSI crosses back up through oversold |
| **SELL** | RSI > overbought, **or** fast SMA crosses below slow SMA, **or** MACD crosses down while the trend is down |
| **CLOSE_POSITION** | Forced risk exit: stop-loss, trailing stop, take-profit, or end of the evaluation window |
| **HOLD** | Nothing triggered |

Signals are read at the **close** and filled at the **next open**, so there is no look-ahead. Stops fill at the stop price, or at the open if the price gapped through it. If a stop and a target are both touched in one bar, the stop is assumed to hit first. Every fill pays 5 bps commission and 5 bps slippage.

### Fitness

```
Fitness = Net Profit (%) × Sharpe Ratio × (1 − Max Drawdown)
```

The formula breaks down for losing bots: −20% profit × −1.5 Sharpe would come out positive and beat a real winner. So bots that lose money are scored `Net Profit (%) × (1 + Max Drawdown)`. That is always negative, and a deeper drawdown ranks lower. A bot that never trades scores 0.

### The honest part: overfitting

By default every generation trades the same **in-sample** window, the first 70% of history. Bots that evolve on one dataset will memorise it. So the last **30% is held out**: the bots never see it during evolution. At the end, the champion, the original Generation-1 genome and an equal-weight buy & hold all face that unseen data in the **Showdown** table and chart. The in-sample numbers flatter the bots. The out-of-sample column is the one to trust.

### Timeframe: fast (5-minute) or slow (daily) trading

`TIMEFRAME` in `config.py` (or `TIMEFRAME=1d` in `.env`) picks the style:

| | `5m` (default) | `1d` |
|---|---|---|
| Bars | 5-minute | daily |
| Trading pace | about **1 trade every 20 minutes** while the US market is open | a few trades a month |
| History | last ~58 days (yfinance limit) | since 2019 |
| Symbols | AAPL, NVDA, TSLA | AAPL, NVDA, TSLA, BTC-USD |
| Stops / targets (textbook) | 0.8% / 1.2% | 5% / 12% |
| Live bot checks | every 60 s | every 5 min |

**The pace rule:** in 5-minute mode, only bots averaging one trade every 10–40 minutes (`PACE_TOLERANCE = 2`) can win the crown. If none do yet, the bot closest to the pace wins. So evolution locks onto the pace first, then works on profit. The **Every** column on the leaderboard shows each bot's pace; yellow means off pace.

**Why no Bitcoin in 5-minute mode:** Alpaca charges 0.25% per crypto trade each way. A typical 20-minute price move is ~0.03%, so every frequent BTC trade loses money. In testing, BTC caused almost all of the losses. Stock trades on Alpaca are commission-free. Add `"BTC-USD"` back to the 5m symbol list in `config.py` if you want to show that on camera.

Checkpoints remember their timeframe. A daily Alpha can't be resumed or deployed in 5-minute mode (you'll get a clear message), because its genes mean different things.

### Evolution switches (in `config.py`)

| Switch | What it does | Default |
|---|---|---|
| `POPULATION_SIZE = 11` | 10 clones per generation instead of 5, so more attempts at improving | 6 |
| `EVAL_FOLDS = 4` | Scores each bot on 4 separate time periods and averages them | 1 |
| `WIN_RATE_WEIGHT` / `PROFIT_FACTOR_WEIGHT = 1.0` | Fitness bonus for winning more often and winning bigger | 0 (off) |
| `MIN_TRADES = 20` | Disqualifies bots with too few trades to prove anything | 20 (on) |
| `ADAPTIVE_MUTATION = True` | Mutates harder when stuck, gentler after a new champion | off |

The defaults come from a test: 10 evolution runs per setting, scored on the held-out data. Adaptive mutation and 10 clones each scored slightly better on their own. The win-rate bonus and the 4 periods scored worse. Combining the "good" ones scored worse too, so the differences are mostly luck between runs. Treat these as experiments to try on camera, not guaranteed upgrades.

---

## Project structure

```
config.py                   every tunable setting (balance, population, mutation rate, symbols, mode...)
main.py                     interactive CLI: Mode 1 evolve / Mode 2 live
bot/
  genome.py                 TradingGenome: genes, bounds, mutate() -> mutated deep clone
  indicators.py             SMA / EMA / RSI / MACD + generate_signals() (shared by backtest & live)
  trader.py                 TradingBot: BUY/SELL/HOLD/CLOSE_POSITION, equity curve, stops, win rate
engine/
  evaluator.py              fitness, ranking, 1 victor, verified purge of the other 5
  evolution.py              generation loop, integrity checks, checkpoints, genetic-drift log, showdown
  live_trader.py            deploys a genome to Alpaca paper
broker/
  base.py                   MarketData, cost model, data-source interface
  simulated_broker.py       yfinance (CSV-cached) + synthetic fallback
  alpaca_broker.py          alpaca-py paper orders, positions, bars
utils/
  display.py                rich UI: extinction log, survivor banner, mutation lab, showdown
  charts.py                 matplotlib evolution report PNG
  logger.py                 rotating file log
tests/test_core.py          offline unit tests of the survival rules
```

### Outputs

| File | Contents |
|---|---|
| `checkpoints/alpha_gen_N.json` | Each generation's survivor: genome, metrics, eliminated bots, settings |
| `checkpoints/latest_alpha.json` | The current Alpha, used by `--resume` and Mode 2 |
| `checkpoints/evolution_report.png` | Fitness per generation, plus the out-of-sample equity race |
| `checkpoints/archive/run_*/` | Previous runs. A fresh run archives old checkpoints; it never deletes them |
| `logs/evolution_history.jsonl` | One line per selection and replication, with every mutation (genetic drift) |
| `logs/evobot.log` | Diagnostic log |
| `data_cache/` | Cached daily bars (one file per symbol per day) |

---

## Tests

```bash
python -m unittest discover -s tests -v
```

Covers: mutations stay in bounds across 2,000 aggressive generations; the parent's DNA is never modified; the fitness sign trap; stop/gap fill rules; portfolio accounting (final equity equals the sum of realised trade P&L); exactly one survivor with a verified purge; and a full 3-generation run plus resume, all offline on synthetic data.

---

## Recording tips 🎥

- Use a terminal at least **140 columns** wide so the mutation table fits.
- `--seed 42` makes an episode replayable. Rehearse, then record the same run.
- `DISPLAY_DELAY` in `config.py` controls the dramatic pauses between eliminations.
- `--synthetic` lets you rehearse offline. The UI clearly labels it as synthetic.
- Put `checkpoints/evolution_report.png` on screen for the final reveal.
