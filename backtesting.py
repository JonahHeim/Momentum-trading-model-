"""
Backtesting Engine
Simulates strategy performance over historical data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from signal_calculation import (
    calculate_composite_score,
    calculate_momentum_score,
    get_latest_scores,
    rank_stocks
)
from portfolio_construction import construct_portfolio, apply_stop_loss


class BacktestEngine:
    """
    Backtesting engine for momentum strategy.
    """
    
    def __init__(
        self,
        initial_capital: float = 100_000,
        holding_period: int = 3,
        rebalance_freq: int = 3,
        stop_loss_pct: float = 0.05,
        max_gross_leverage: float = 1.5,
        max_sector_weight: float = 0.30,
        transaction_cost: float = 0.001,  # 0.1% per trade
        cash_buffer: float = 0.05  # 5% cash buffer
    ):
        """
        Initialize backtesting engine.
        
        Args:
            initial_capital: Starting capital
            holding_period: Days to hold positions
            rebalance_freq: Rebalance every N days
            stop_loss_pct: Stop-loss percentage
            max_gross_leverage: Maximum gross leverage
            max_sector_weight: Maximum sector weight
            transaction_cost: Transaction cost per trade (as fraction)
            cash_buffer: Cash buffer as fraction of capital
        """
        self.initial_capital = initial_capital
        self.holding_period = holding_period
        self.rebalance_freq = rebalance_freq
        self.stop_loss_pct = stop_loss_pct
        self.max_gross_leverage = max_gross_leverage
        self.max_sector_weight = max_sector_weight
        self.transaction_cost = transaction_cost
        self.cash_buffer = cash_buffer
        
        self.portfolio_value = []
        self.returns = []
        self.positions_history = []
        self.trades = []
    
    def run_backtest(
        self,
        prices: pd.DataFrame,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None
    ) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            prices: DataFrame with price data
            start_date: Start date for backtest (defaults to first date with enough data)
            end_date: End date for backtest (defaults to last date)
            
        Returns:
            Dictionary with backtest results
        """
        print("Starting backtest...")
        
        # Set date range
        if start_date is None:
            start_date = prices.index[60]  # Need at least 60 days for calculations
        if end_date is None:
            end_date = prices.index[-1]
        
        # Filter prices to date range
        prices = prices.loc[start_date:end_date]
        
        # Calculate composite scores (momentum, RSI, MA, vol, volume)
        print("Calculating composite signals (momentum, RSI, MA, volatility, volume)...")
        # Note: For backtesting, we'll use composite scores if volumes are available
        # Otherwise fall back to momentum-only
        try:
            # Try to get volumes for enhanced signals
            from data_collection import fetch_price_data
            _, volumes = fetch_price_data(prices.columns.tolist(), period="6mo", return_volumes=True)
            from data_collection import get_spy_data
            spy_prices = get_spy_data(period="6mo")
            
            if not volumes.empty and not spy_prices.empty:
                scores = calculate_composite_score(
                    prices,
                    volumes=volumes,
                    benchmark_prices=spy_prices
                )
                print("  Using composite signals (all factors)")
            else:
                scores = calculate_momentum_score(prices)
                print("  Using momentum signals (fallback)")
        except:
            scores = calculate_momentum_score(prices)
            print("  Using momentum signals (fallback)")
        
        # Initialize tracking variables
        capital = self.initial_capital
        current_positions = {}
        entry_prices = {}
        entry_date = None
        
        # Iterate through trading days
        lookback_window = 60
        dates_to_trade = prices.index[lookback_window::self.rebalance_freq]
        
        print(f"Backtesting from {start_date.date()} to {end_date.date()}")
        print(f"Trading on {len(dates_to_trade)} dates")
        
        for i, trade_date in enumerate(dates_to_trade):
            try:
                # Get date index
                date_idx = prices.index.get_loc(trade_date)
                
                # Check if we need to rebalance
                if entry_date is None or (date_idx - prices.index.get_loc(entry_date)) >= self.holding_period:
                    # Close existing positions
                    if current_positions:
                        capital = self._close_positions(
                            current_positions,
                            entry_prices,
                            prices.iloc[date_idx],
                            capital
                        )
                        current_positions = {}
                        entry_prices = {}
                    
                    # Calculate new signals
                    if date_idx >= lookback_window:
                        latest_scores = get_latest_scores(scores, trade_date)
                        
                        if len(latest_scores) > 0:
                            # Rank stocks (pick 10 for each side)
                            long_tickers, short_tickers = rank_stocks(latest_scores, n_stocks=10)
                            
                            # Construct portfolio with volatility targeting
                            prices_to_date = prices.iloc[:date_idx+1]
                            portfolio = construct_portfolio(
                                list(long_tickers),
                                list(short_tickers),
                                prices_to_date,
                                capital=capital * (1 - self.cash_buffer),  # Reserve cash
                                max_gross_leverage=self.max_gross_leverage,
                                max_sector_weight=self.max_sector_weight,
                                target_vol=0.15  # 15% target volatility
                            )
                            
                            # Extract positions
                            current_positions = {
                                'long_positions': portfolio['long_positions'],
                                'short_positions': portfolio['short_positions']
                            }
                            
                            # Record entry prices
                            current_prices = prices.iloc[date_idx]
                            for ticker in list(long_tickers) + list(short_tickers):
                                if ticker in current_prices.index:
                                    entry_prices[ticker] = current_prices[ticker]
                            
                            entry_date = trade_date
                            
                            # Apply transaction costs
                            total_trades = len(portfolio['long_positions']) + len(portfolio['short_positions'])
                            capital -= capital * self.transaction_cost * total_trades
                            
                            # Record trade
                            self.trades.append({
                                'date': trade_date,
                                'long_positions': len(portfolio['long_positions']),
                                'short_positions': len(portfolio['short_positions']),
                                'capital': capital,
                                'gross_exposure': portfolio['gross_exposure']
                            })
                    
                # Calculate current portfolio value
                if current_positions:
                    portfolio_value = self._calculate_portfolio_value(
                        current_positions,
                        entry_prices,
                        prices.iloc[date_idx],
                        capital
                    )
                else:
                    portfolio_value = capital
                
                # Apply stop losses (with trailing stops)
                if current_positions and entry_prices:
                    # Initialize peak_prices if not exists
                    if not hasattr(self, 'peak_prices'):
                        self.peak_prices = {}
                    
                    updated_positions = apply_stop_loss(
                        current_positions,
                        prices.iloc[:date_idx+1],
                        entry_prices,
                        self.stop_loss_pct,
                        use_trailing_stop=True,
                        peak_prices=self.peak_prices
                    )
                    
                    # Close stopped-out positions
                    for side in ['long_positions', 'short_positions']:
                        for ticker, pos in updated_positions.get(side, {}).items():
                            if pos.get('stop_loss', False):
                                if ticker in current_positions.get(side, {}):
                                    current_price = prices.iloc[date_idx][ticker]
                                    if side == 'long_positions':
                                        shares = current_positions[side][ticker]['shares']
                                        capital += shares * current_price * (1 - self.transaction_cost)
                                    else:
                                        shares = current_positions[side][ticker]['shares']
                                        capital += shares * entry_prices[ticker] * (1 - self.transaction_cost)  # Short profit
                                    
                                    del current_positions[side][ticker]
                                    if ticker in entry_prices:
                                        del entry_prices[ticker]
                
                # Record portfolio value
                self.portfolio_value.append({
                    'date': trade_date,
                    'value': portfolio_value,
                    'capital': capital
                })
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(dates_to_trade)} dates...")
                    
            except Exception as e:
                print(f"Error on date {trade_date}: {e}")
                continue
        
        # Close final positions
        if current_positions:
            final_date_idx = len(prices) - 1
            capital = self._close_positions(
                current_positions,
                entry_prices,
                prices.iloc[final_date_idx],
                capital
            )
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.portfolio_value)
        results_df.set_index('date', inplace=True)
        
        # Calculate returns
        results_df['returns'] = results_df['value'].pct_change()
        
        print("\nBacktest complete!")
        
        return {
            'portfolio_value': results_df['value'],
            'returns': results_df['returns'],
            'trades': pd.DataFrame(self.trades),
            'final_capital': capital,
            'total_return': (capital - self.initial_capital) / self.initial_capital
        }
    
    def _close_positions(
        self,
        positions: Dict,
        entry_prices: Dict[str, float],
        current_prices: pd.Series,
        capital: float
    ) -> float:
        """
        Close all positions and calculate P&L.
        
        Args:
            positions: Current positions dictionary
            entry_prices: Entry prices for positions
            current_prices: Current market prices
            capital: Current capital
            
        Returns:
            Updated capital after closing positions
        """
        # Close long positions
        for ticker, pos in positions.get('long_positions', {}).items():
            if ticker in current_prices.index:
                shares = pos['shares']
                entry_price = entry_prices.get(ticker, pos.get('entry_price', current_prices[ticker]))
                current_price = current_prices[ticker]
                pnl = shares * (current_price - entry_price)
                capital += shares * current_price * (1 - self.transaction_cost)
        
        # Close short positions
        for ticker, pos in positions.get('short_positions', {}).items():
            if ticker in current_prices.index:
                shares = pos['shares']
                entry_price = entry_prices.get(ticker, pos.get('entry_price', current_prices[ticker]))
                current_price = current_prices[ticker]
                pnl = shares * (entry_price - current_price)  # Inverse for shorts
                capital += shares * entry_price * (1 - self.transaction_cost)  # Return borrowed shares
        
        return capital
    
    def _calculate_portfolio_value(
        self,
        positions: Dict,
        entry_prices: Dict[str, float],
        current_prices: pd.Series,
        cash: float
    ) -> float:
        """
        Calculate current portfolio value.
        
        Args:
            positions: Current positions
            entry_prices: Entry prices
            current_prices: Current market prices
            cash: Cash on hand
            
        Returns:
            Total portfolio value
        """
        total_value = cash
        
        # Value long positions
        for ticker, pos in positions.get('long_positions', {}).items():
            if ticker in current_prices.index:
                shares = pos['shares']
                current_price = current_prices[ticker]
                total_value += shares * current_price
        
        # Value short positions (mark-to-market)
        for ticker, pos in positions.get('short_positions', {}).items():
            if ticker in current_prices.index:
                shares = pos['shares']
                entry_price = entry_prices.get(ticker, current_prices[ticker])
                current_price = current_prices[ticker]
                # Short P&L = entry_price - current_price (inverse of long)
                pnl = shares * (entry_price - current_price)
                total_value += pnl
        
        return total_value


if __name__ == "__main__":
    # Test the backtest engine
    dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='D')
    test_prices = pd.DataFrame(
        np.random.randn(len(dates), 50).cumsum(axis=0) + 100,
        index=dates,
        columns=[f'TICKER_{i}' for i in range(50)]
    )
    
    engine = BacktestEngine(initial_capital=100000, holding_period=3, rebalance_freq=3)
    results = engine.run_backtest(test_prices)
    
    print(f"\nBacktest Results:")
    print(f"  Final capital: ${results['final_capital']:,.2f}")
    print(f"  Total return: {results['total_return']:.2%}")

