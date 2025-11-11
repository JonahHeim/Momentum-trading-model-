# Momentum-trading-model-
What the Program Does (End-to-End)

This program runs a momentum-based long/short equity strategy from start to finish. It fetches market data, generates trading signals, constructs a portfolio under risk constraints, backtests performance, and produces detailed reports.

Core Components and Workflow

Universe Selection (universe_selection.py)
Builds the tradable universe (default: S&P 500, with optional mid- and small-cap stocks).
Filters stocks by liquidity, price, and market cap, with optional sampling for faster testing.
Uses yfinance for data and includes safeguards for SSL and API issues.

Data Collection (data_collection.py)
Downloads OHLCV data for all selected tickers using yfinance.
Optionally pulls benchmark data (SPY) and market regime indicators (VIX).
Cleans and validates the data, ensuring sufficient history and handling missing values.

Signal Calculation (signal_calculation.py)
Calculates momentum signals (e.g., returns over different lookback windows, composite momentum scores). Ranks all tickers and splits them into long and short baskets.
Supports machine learning–based signals (LightGBM, RandomForest) with lazy loading—no ML dependencies unless explicitly used.

Portfolio Construction (portfolio_construction.py)
Allocates capital across the long and short baskets.
Enforces risk and exposure constraints such as gross exposure limits, sector weights, cash buffers, transaction costs, and optional stop-loss rules.
Outputs position sizes and total portfolio exposure.

Backtesting Engine (backtesting.py)
Simulates the strategy over time with configurable holding periods and rebalance frequencies.
Tracks portfolio value, returns, and trade history.
Optionally compares performance against a benchmark (SPY).

Performance Evaluation (performance_evaluation.py)
Generates performance metrics and visualizations, including equity curves, drawdowns, and return distributions.
Optionally benchmarks results against SPY for relative performance analysis.

Exports results as:
A PNG plot showing performance.
An Excel report (XLSX) containing metrics, portfolio values, returns, trades, and the universe used.

Entry Point (main.py)
Serves as the command-line interface for the entire pipeline.
Key arguments: universe type (sample/full), sample size, initial capital, holding period, rebalance frequency, historical period, and output prefix.
Orchestrates the full flow:
universe → data → validation → optional VIX → backtest → report → save outputs.
Example Run
python main.py --period 1y --capital 100000 --holding-period 3 --rebalance-freq 3
The program displays progress updates, runs the backtest, and outputs:
A performance plot (.png)
A detailed Excel report (.xlsx) with metrics, equity curve, returns, trades, and universe details.
Inputs & Outputs
Inputs: Market data from yfinance (tickers, optional VIX/SPY).

Outputs:
Performance plot (.png)
Metrics and detailed results (.xlsx)

Dependencies
Core: pandas, numpy, yfinance
Visualization: matplotlib, seaborn
Export: openpyxl
