"""
Main Execution Script
Advanced Momentum Trading Strategy
"""

import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from universe_selection import get_universe
from data_collection import fetch_price_data, validate_data, get_vix_data
from backtesting import BacktestEngine
from performance_evaluation import generate_report


def main():
    """
    Main execution function for the momentum trading strategy.
    """
    parser = argparse.ArgumentParser(description='Advanced Momentum Trading Strategy')
    parser.add_argument('--sample', action='store_true', help='Use sample universe for faster testing')
    parser.add_argument('--sample-size', type=int, default=50, help='Sample size if using sample mode')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--holding-period', type=int, default=3, help='Holding period in days')
    parser.add_argument('--rebalance-freq', type=int, default=3, help='Rebalance frequency in days')
    parser.add_argument('--period', type=str, default='1y', help='Data period (e.g., 6mo, 1y, 2y)')
    parser.add_argument('--output', type=str, default='backtest_results', help='Output file prefix')
    
    args = parser.parse_args()
    
    print("="*70)
    print("ADVANCED MOMENTUM TRADING STRATEGY")
    print("="*70)
    print(f"Initial Capital: ${args.capital:,.2f}")
    print(f"Holding Period: {args.holding_period} days")
    print(f"Rebalance Frequency: {args.rebalance_freq} days")
    print(f"Data Period: {args.period}")
    print("="*70 + "\n")
    
    # Step 1: Universe Selection (S&P500 + Mid-cap + Small-cap)
    print("STEP 1: Universe Selection (S&P 500 + Mid-cap + Small-cap)")
    print("-" * 70)
    universe = get_universe(
        use_sample=args.sample,
        sample_size=args.sample_size,
        min_volume=2_000_000,
        min_price=5.0,
        min_market_cap=5_000_000_000,
        include_mid_cap=True,
        include_small_cap=True
    )
    
    if len(universe) < 20:
        print(f"Warning: Universe too small ({len(universe)} tickers). Need at least 20.")
        return
    
    print(f"Universe size: {len(universe)} tickers\n")
    
    # Step 2: Data Collection
    print("STEP 2: Data Collection")
    print("-" * 70)
    prices = fetch_price_data(universe, period=args.period, interval="1d")
    
    if prices.empty:
        print("Error: Failed to fetch price data")
        return
    
    # Validate data
    prices = validate_data(prices, min_days=60)
    
    if len(prices.columns) < 20:
        print(f"Warning: Insufficient valid tickers ({len(prices.columns)}). Need at least 20.")
        return
    
    print(f"Valid tickers: {len(prices.columns)}\n")
    
    # Optional: Get VIX data for regime filtering
    print("Fetching VIX data for regime analysis...")
    vix_data = get_vix_data(period=args.period)
    if not vix_data.empty:
        # Filter for low volatility periods (VIX < 25)
        vix_threshold = 25
        low_vol_dates = vix_data[vix_data < vix_threshold].index
        print(f"Low volatility periods (VIX < {vix_threshold}): {len(low_vol_dates)}/{len(vix_data)} days")
    
    # Step 3: Backtesting
    print("\nSTEP 3: Backtesting")
    print("-" * 70)
    
    engine = BacktestEngine(
        initial_capital=args.capital,
        holding_period=args.holding_period,
        rebalance_freq=args.rebalance_freq,
        stop_loss_pct=0.05,
        max_gross_leverage=1.5,
        max_sector_weight=0.30,
        transaction_cost=0.001,
        cash_buffer=0.05
    )
    
    results = engine.run_backtest(prices)
    
    # Step 4: Performance Evaluation
    print("\nSTEP 4: Performance Evaluation")
    print("-" * 70)
    
    # Get benchmark (S&P 500)
    try:
        spy = fetch_price_data(['SPY'], period=args.period)
        if not spy.empty and len(spy.columns) > 0:
            benchmark = spy.iloc[:, 0]
            # Normalize benchmark to start at same value as strategy
            benchmark_normalized = benchmark / benchmark.iloc[0] * args.capital
        else:
            benchmark_normalized = None
    except:
        benchmark_normalized = None
        print("Warning: Could not fetch benchmark data")
    
    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"{args.output}_{timestamp}.png"
    
    metrics_df = generate_report(
        results,
        save_path=plot_path,
        benchmark=benchmark_normalized
    )
    
    # Save detailed results
    excel_path = f"{args.output}_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        metrics_df.to_excel(writer, sheet_name='Metrics')
        
        # Portfolio value
        portfolio_df = pd.DataFrame({
            'Date': results['portfolio_value'].index,
            'Portfolio Value': results['portfolio_value'].values
        })
        portfolio_df.to_excel(writer, sheet_name='Portfolio Value', index=False)
        
        # Returns
        returns_df = pd.DataFrame({
            'Date': results['returns'].index,
            'Returns': results['returns'].values
        })
        returns_df.to_excel(writer, sheet_name='Returns', index=False)
        
        # Trades
        if 'trades' in results and not results['trades'].empty:
            results['trades'].to_excel(writer, sheet_name='Trades', index=False)
        
        # Universe
        universe_df = pd.DataFrame({'Ticker': universe})
        universe_df.to_excel(writer, sheet_name='Universe', index=False)
    
    print(f"\nResults saved to:")
    print(f"  Plot: {plot_path}")
    print(f"  Excel: {excel_path}")
    
    # Summary
    print("\n" + "="*70)
    print("BACKTEST SUMMARY")
    print("="*70)
    print(f"Initial Capital: ${args.capital:,.2f}")
    print(f"Final Capital: ${results['final_capital']:,.2f}")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Universe Size: {len(universe)} tickers")
    print(f"Total Trades: {len(results['trades']) if 'trades' in results else 0}")
    print("="*70)


if __name__ == "__main__":
    main()

