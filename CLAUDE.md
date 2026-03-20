# CLAUDE.md - AI Assistant Guide for Momentum Trading Model

## Project Overview

This is a **momentum-based long/short equity trading strategy** implemented in Python. It forms an end-to-end pipeline: universe selection → data collection → signal calculation → portfolio construction → backtesting → performance evaluation.

The model targets US equities (S&P 500/400/600), uses 15+ technical momentum signals, and applies rigorous risk management (volatility targeting, sector caps, stop-losses).

---

## Repository Structure

```
Momentum-trading-model-/
├── main.py                    # CLI entry point, orchestrates full pipeline
├── universe_selection.py      # Build tradable stock universe (S&P 500/400/600)
├── data_collection.py         # Download OHLCV data via yfinance
├── signal_calculation.py      # Compute momentum signals and composite scores
├── portfolio_construction.py  # Position sizing and risk constraints
├── backtesting.py             # Simulate trading, track P&L
├── performance_evaluation.py  # Calculate metrics and generate reports
└── README.md                  # Project documentation
```

**No configuration files exist** (no requirements.txt, pyproject.toml, setup.py, or .env). Dependencies must be inferred from imports.

---

## Pipeline Architecture

Data flows strictly sequentially through six stages:

```
main.py
  ↓ 1. universe_selection.py   → List of ~100-500 ticker symbols
  ↓ 2. data_collection.py      → OHLCV DataFrame (dates × tickers)
  ↓ 3. signal_calculation.py   → Ranked momentum scores
  ↓ 4. portfolio_construction.py → Position sizes & capital allocation
  ↓ 5. backtesting.py          → Daily portfolio value & returns series
  ↓ 6. performance_evaluation.py → PNG chart + XLSX report
```

---

## Module Reference

### `main.py` (178 lines)
CLI entry point using `argparse`. Orchestrates the full pipeline and exports results.

**CLI arguments:**
```bash
python main.py [--sample] [--sample-size N] [--capital AMOUNT]
               [--holding-period D] [--rebalance-freq D]
               [--period STR] [--output PREFIX]
```

**Defaults:** capital=100000, holding-period=3, rebalance-freq=3, period=1y, output=backtest_results

**Example:**
```bash
python main.py --period 1y --capital 100000 --holding-period 3 --rebalance-freq 3
# For fast testing:
python main.py --sample --sample-size 20 --period 6mo
```

### `universe_selection.py` (327 lines)
Fetches S&P 500/400/600 constituents from Wikipedia, applies liquidity/price/market-cap filters.

**Key filters:** min daily volume 2M, min price $5, min market cap $5B
**Fallback:** Hardcoded ticker lists if Wikipedia fetch fails (offline safe)
**Note:** Disables SSL verification for macOS cert issues (line 17)

### `data_collection.py` (339 lines)
Downloads OHLCV data using `yfinance`. Also fetches VIX (regime filter) and SPY (benchmark).

**Key functions:**
- `fetch_price_data()` - Batch OHLCV download
- `get_vix_data()` / `get_spy_data()` - Auxiliary data
- `check_vix_regime()` / `check_spy_regime()` - Risk-on/bullish checks
- `validate_data()` - Drops tickers with <60 days; forward-fills gaps (limit=5 days)

### `signal_calculation.py` (812 lines) — largest module
Computes 15+ technical indicators and blends them into a composite momentum score.

**Signal types:**
1. Momentum (5d/10d/20d weighted returns)
2. Volatility-adjusted momentum (Sharpe-like)
3. Relative strength vs SPY
4. RSI(14), RSI(2), RSI(5)
5. Moving average crossover (20/50 SMA)
6. Volume trend (recent vs average)
7. Bollinger Bands %B
8. ATR (Average True Range)
9. Donchian Channels breakout
10. Volume confirmation filter (1.5x multiplier)
11. Rolling z-score normalization (60-day window)

**Key functions:**
- `calculate_composite_score()` - Blends all signals with weighted averaging
- `calculate_momentum_score()` - Enhanced momentum with relative strength
- `get_latest_scores()` - Extract current scores for all tickers
- `rank_stocks()` - Top N = longs, bottom N = shorts
- `ml_rank_signals()` - Optional ML ranking (LightGBM/RandomForest)

### `portfolio_construction.py` (391 lines)
Handles capital allocation with inverse-volatility weighting and risk constraints.

**Risk parameters (defaults):**
- Max gross leverage: 1.5x
- Max sector weight: 30%
- Cash buffer: 5%
- Transaction costs: 0.1% per trade
- Stop-loss: 5% (trailing or fixed)
- Volatility target: 15% annualized

### `backtesting.py` (374 lines)
`BacktestEngine` class simulates daily mark-to-market P&L over a configurable holding period.

**Outputs:** Portfolio value series, daily returns, trade history DataFrame

### `performance_evaluation.py` (232 lines)
Calculates performance metrics and generates output files.

**Metrics:** Total return, CAGR, Sharpe ratio, max drawdown, win rate, win/loss ratio
**Exports:**
- PNG chart: portfolio value, returns distribution, drawdown (vs SPY)
- XLSX file: Metrics, Portfolio Value, Returns, Trades, Universe sheets

---

## Dependencies

No requirements.txt exists. Install manually:

```bash
pip install pandas numpy yfinance matplotlib seaborn openpyxl
# Optional ML support:
pip install scikit-learn lightgbm
```

**ML libraries are optional** — `_check_ml_libraries()` tests availability at runtime and falls back to simple averaging if unavailable.

---

## Development Conventions

### Code Style
- All functions include **docstrings** with `Args:` and `Returns:` sections
- **Type hints** used throughout (`Optional`, `Dict`, `List`, `Tuple`)
- Vectorized pandas/numpy operations preferred over loops
- Division-by-zero guards use `+ 1e-10` in denominators

### Testing
No pytest/unittest framework. Each module has an `if __name__ == "__main__"` test block using synthetic data:

```python
# Run module-level tests:
python universe_selection.py
python data_collection.py
python signal_calculation.py
# etc.
```

Test blocks generate random-walk price data and verify basic output shapes/counts.

### Error Handling Patterns
- Wikipedia fetching → fallback to hardcoded ticker lists
- yfinance batch download → fallback to individual per-ticker downloads
- Missing price data → forward-fill (max 5 days), then drop if <60 days
- ML imports → graceful fallback to averaging

### Pandas Conventions
- `.iloc` for integer-position indexing, `.loc` for label-based
- `.pct_change()` for return calculations
- `.reindex()` to align DataFrames before arithmetic
- Avoid in-place operations; prefer assignment

---

## Output Files

Results are saved with timestamp suffixes in the working directory:

| File | Contents |
|------|----------|
| `backtest_results_TIMESTAMP.png` | 3-panel chart: portfolio value, returns histogram, drawdown |
| `backtest_results_TIMESTAMP.xlsx` | 5-sheet Excel: Metrics, Portfolio Value, Returns, Trades, Universe |

---

## Key Architectural Decisions

1. **Sequential pipeline** — Each stage's output is the next stage's input. Do not skip or reorder stages.
2. **No state between runs** — Each execution is independent; no database or cache.
3. **Sampling mode** — Use `--sample --sample-size N` during development to avoid slow data downloads.
4. **Regime filtering** — VIX and SPY regime checks can reduce or halt trading in risk-off environments.
5. **Lazy ML imports** — ML signal ranking is opt-in; base functionality never requires sklearn/lightgbm.

---

## Common Tasks for AI Assistants

### Adding a new signal
1. Implement the signal in `signal_calculation.py` following existing patterns (vectorized, z-score normalized)
2. Add it to the `calculate_composite_score()` blend with an appropriate weight
3. Test via the `if __name__ == "__main__"` block

### Modifying risk parameters
Risk defaults live in `portfolio_construction.py` function signatures. Change default parameter values there.

### Changing the backtest period or rebalancing
Use CLI flags (`--period`, `--rebalance-freq`, `--holding-period`) — do not hardcode.

### Adding a new output metric
Add calculation in `performance_evaluation.py` and ensure it is written to the XLSX Metrics sheet.

### Debugging data issues
- Set `--sample --sample-size 5` to run with minimal tickers
- Check `validate_data()` in `data_collection.py` for filtering logic
- yfinance rate limits: add `time.sleep()` between calls if needed

---

## Git Workflow

- **Main branch:** `master` (upstream: `origin/main`)
- **Development branches:** prefixed with `claude/` for AI-assisted work
- **No CI/CD** configured; no automated tests on push

```bash
git checkout -b feature/my-change
# make changes
git add <specific-files>
git commit -m "Descriptive message"
git push -u origin feature/my-change
```

---

## Security Notes

- SSL verification is disabled in `universe_selection.py` (line 17) for Wikipedia requests. This is a known workaround for macOS cert issues — do not extend this pattern to other HTTP calls.
- No API keys or secrets are used. All data is fetched from public sources (Yahoo Finance, Wikipedia).
- Do not add credential files or `.env` files to the repository.
