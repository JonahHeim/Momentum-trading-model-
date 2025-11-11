"""
Portfolio Construction Module
Handles position sizing, weighting, and risk management.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import yfinance as yf


def calculate_volatility_weights(
    tickers: List[str],
    prices: pd.DataFrame,
    vol_window: int = 20
) -> pd.Series:
    """
    Calculate inverse volatility weights for equal risk contribution.
    
    Formula: w_i = (1/σ_i) / Σ(1/σ_j)
    
    Args:
        tickers: List of ticker symbols
        prices: DataFrame with price data
        vol_window: Window for volatility calculation
        
    Returns:
        Series with weights (sums to 1)
    """
    # Calculate volatility
    daily_returns = prices[tickers].pct_change()
    volatility = daily_returns.rolling(window=vol_window).std().iloc[-1]
    
    # Inverse volatility (handle zeros)
    inv_vol = 1 / (volatility + 1e-6)
    
    # Normalize to sum to 1
    weights = inv_vol / inv_vol.sum()
    
    return weights


def get_sector_weights(tickers: List[str]) -> Dict[str, float]:
    """
    Calculate sector weights for the portfolio.
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        Dictionary mapping sectors to weights
    """
    sector_counts = {}
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            sector = info.get('sector', 'Unknown')
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        except:
            sector_counts['Unknown'] = sector_counts.get('Unknown', 0) + 1
    
    total = sum(sector_counts.values())
    sector_weights = {sector: count / total for sector, count in sector_counts.items()}
    
    return sector_weights


def calculate_portfolio_volatility(
    positions: Dict,
    prices: pd.DataFrame,
    vol_window: int = 20
) -> float:
    """
    Calculate expected portfolio volatility based on positions.
    
    Args:
        positions: Dictionary with long and short positions
        prices: DataFrame with price data
        vol_window: Window for volatility calculation
        
    Returns:
        Expected portfolio volatility (annualized)
    """
    if not positions.get('long_positions') and not positions.get('short_positions'):
        return 0.0
    
    # Get all tickers
    all_tickers = list(positions.get('long_positions', {}).keys()) + \
                  list(positions.get('short_positions', {}).keys())
    
    if not all_tickers:
        return 0.0
    
    # Calculate individual volatilities
    daily_returns = prices[all_tickers].pct_change()
    individual_vols = daily_returns.rolling(window=vol_window).std().iloc[-1] * np.sqrt(252)
    
    # Get position weights
    total_long_value = sum(p['value'] for p in positions.get('long_positions', {}).values())
    total_short_value = sum(p['value'] for p in positions.get('short_positions', {}).values())
    total_value = total_long_value + total_short_value
    
    if total_value == 0:
        return 0.0
    
    # Calculate weighted portfolio volatility (simplified - assumes low correlation)
    portfolio_vol = 0.0
    for ticker in all_tickers:
        if ticker in individual_vols.index:
            if ticker in positions.get('long_positions', {}):
                weight = positions['long_positions'][ticker]['value'] / total_value
            else:
                weight = positions['short_positions'][ticker]['value'] / total_value
            
            portfolio_vol += (weight * individual_vols[ticker]) ** 2
    
    # Square root for portfolio volatility (simplified correlation assumption)
    portfolio_vol = np.sqrt(portfolio_vol)
    
    return portfolio_vol


def apply_volatility_target(
    positions: Dict,
    prices: pd.DataFrame,
    target_vol: float = 0.15,
    vol_window: int = 20
) -> Dict:
    """
    Adjust position sizes to target portfolio volatility.
    
    Args:
        positions: Dictionary with positions
        prices: DataFrame with price data
        target_vol: Target annualized volatility (default 15%)
        vol_window: Window for volatility calculation
        
    Returns:
        Updated positions dictionary with volatility-adjusted sizes
    """
    current_vol = calculate_portfolio_volatility(positions, prices, vol_window)
    
    if current_vol == 0 or current_vol < 0.01:
        return positions  # No adjustment needed
    
    # Calculate scaling factor
    vol_scale = target_vol / current_vol
    
    # Scale positions proportionally
    updated_positions = {
        'long_positions': {},
        'short_positions': {}
    }
    
    for ticker, pos in positions.get('long_positions', {}).items():
        updated_pos = pos.copy()
        updated_pos['value'] = pos['value'] * vol_scale
        updated_pos['shares'] = int(updated_pos['value'] / prices[ticker].iloc[-1])
        updated_pos['weight'] = pos['weight'] * vol_scale
        updated_positions['long_positions'][ticker] = updated_pos
    
    for ticker, pos in positions.get('short_positions', {}).items():
        updated_pos = pos.copy()
        updated_pos['value'] = pos['value'] * vol_scale
        updated_pos['shares'] = int(updated_pos['value'] / prices[ticker].iloc[-1])
        updated_pos['weight'] = pos['weight'] * vol_scale
        updated_positions['short_positions'][ticker] = updated_pos
    
    return updated_positions


def construct_portfolio(
    long_tickers: List[str],
    short_tickers: List[str],
    prices: pd.DataFrame,
    capital: float = 100_000,
    max_gross_leverage: float = 1.5,
    max_sector_weight: float = 0.30,
    vol_window: int = 20,
    target_vol: Optional[float] = None
) -> Dict:
    """
    Construct portfolio with risk management rules.
    
    Args:
        long_tickers: List of tickers for long positions
        short_tickers: List of tickers for short positions
        prices: DataFrame with price data
        capital: Initial capital
        max_gross_leverage: Maximum gross leverage ratio
        max_sector_weight: Maximum weight per sector
        vol_window: Window for volatility calculation
        target_vol: Optional target annualized volatility (e.g., 0.15 for 15%)
        
    Returns:
        Dictionary with portfolio allocation
    """
    # Calculate volatility-adjusted weights for longs
    if len(long_tickers) > 0:
        long_weights = calculate_volatility_weights(long_tickers, prices, vol_window)
    else:
        long_weights = pd.Series(dtype=float)
    
    # Calculate volatility-adjusted weights for shorts
    if len(short_tickers) > 0:
        short_weights = calculate_volatility_weights(short_tickers, prices, vol_window)
    else:
        short_weights = pd.Series(dtype=float)
    
    # Apply leverage constraint
    # Allocate capital: 50% to longs, 50% to shorts (within leverage limit)
    long_capital = capital * 0.5 * max_gross_leverage
    short_capital = capital * 0.5 * max_gross_leverage
    
    # Check sector constraints
    all_tickers = list(long_tickers) + list(short_tickers)
    sector_weights = get_sector_weights(all_tickers)
    
    # Apply sector limits (simplified - would need more sophisticated logic)
    # For now, we'll flag if any sector exceeds limit
    sector_violations = [s for s, w in sector_weights.items() if w > max_sector_weight]
    
    # Calculate position sizes
    long_positions = {}
    for ticker in long_tickers:
        if ticker in long_weights.index:
            weight = long_weights[ticker]
            position_value = long_capital * weight
            current_price = prices[ticker].iloc[-1]
            shares = int(position_value / current_price)
            long_positions[ticker] = {
                'shares': shares,
                'value': shares * current_price,
                'weight': weight
            }
    
    short_positions = {}
    for ticker in short_tickers:
        if ticker in short_weights.index:
            weight = short_weights[ticker]
            position_value = short_capital * weight
            current_price = prices[ticker].iloc[-1]
            shares = int(position_value / current_price)
            short_positions[ticker] = {
                'shares': shares,
                'value': shares * current_price,
                'weight': weight
            }
    
    # Apply volatility targeting if specified
    if target_vol is not None:
        temp_portfolio = {
            'long_positions': long_positions,
            'short_positions': short_positions
        }
        adjusted_positions = apply_volatility_target(
            temp_portfolio,
            prices,
            target_vol=target_vol,
            vol_window=vol_window
        )
        long_positions = adjusted_positions['long_positions']
        short_positions = adjusted_positions['short_positions']
    
    # Calculate actual gross exposure
    total_long_value = sum(p['value'] for p in long_positions.values())
    total_short_value = sum(p['value'] for p in short_positions.values())
    gross_exposure = (total_long_value + total_short_value) / capital
    
    # Calculate portfolio volatility
    portfolio_vol = calculate_portfolio_volatility(
        {'long_positions': long_positions, 'short_positions': short_positions},
        prices,
        vol_window
    )
    
    portfolio = {
        'long_positions': long_positions,
        'short_positions': short_positions,
        'total_long_value': total_long_value,
        'total_short_value': total_short_value,
        'gross_exposure': gross_exposure,
        'net_exposure': (total_long_value - total_short_value) / capital,
        'sector_weights': sector_weights,
        'sector_violations': sector_violations,
        'portfolio_volatility': portfolio_vol,
        'target_volatility': target_vol
    }
    
    return portfolio


def apply_stop_loss(
    positions: Dict,
    prices: pd.DataFrame,
    entry_prices: Dict[str, float],
    stop_loss_pct: float = 0.05,
    use_trailing_stop: bool = True,
    peak_prices: Optional[Dict[str, float]] = None
) -> Dict:
    """
    Apply stop-loss rules to positions (fixed or trailing).
    
    Args:
        positions: Dictionary with current positions
        prices: DataFrame with price data
        entry_prices: Dictionary mapping tickers to entry prices
        stop_loss_pct: Stop-loss percentage (e.g., 0.05 for 5%)
        use_trailing_stop: If True, use trailing stop (from peak), else fixed from entry
        peak_prices: Dictionary mapping tickers to peak prices (for trailing stop)
        
    Returns:
        Updated positions dictionary with stop-losses applied
    """
    current_prices = prices.iloc[-1]
    updated_positions = positions.copy()
    
    if peak_prices is None:
        peak_prices = {}
    
    # Check long positions
    for ticker, pos in positions.get('long_positions', {}).items():
        if ticker in entry_prices:
            entry = entry_prices[ticker]
            current = current_prices[ticker]
            
            # Update peak price for trailing stop
            if use_trailing_stop:
                if ticker not in peak_prices:
                    peak_prices[ticker] = entry
                peak_prices[ticker] = max(peak_prices[ticker], current)
                reference_price = peak_prices[ticker]
            else:
                reference_price = entry
            
            # Calculate loss from reference price
            loss_pct = (reference_price - current) / reference_price
            
            if loss_pct >= stop_loss_pct:
                # Stop loss triggered
                updated_positions['long_positions'][ticker]['stop_loss'] = True
                stop_type = "trailing" if use_trailing_stop else "fixed"
                print(f"Stop loss triggered for {ticker} (long, {stop_type}): {loss_pct:.2%} from {reference_price:.2f}")
    
    # Check short positions
    for ticker, pos in positions.get('short_positions', {}).items():
        if ticker in entry_prices:
            entry = entry_prices[ticker]
            current = current_prices[ticker]
            
            # Update trough price for trailing stop (inverse for shorts)
            if use_trailing_stop:
                if ticker not in peak_prices:
                    peak_prices[ticker] = entry
                peak_prices[ticker] = min(peak_prices[ticker], current)  # Lower is better for shorts
                reference_price = peak_prices[ticker]
            else:
                reference_price = entry
            
            # Calculate loss from reference price (inverse for shorts)
            loss_pct = (current - reference_price) / reference_price
            
            if loss_pct >= stop_loss_pct:
                # Stop loss triggered
                updated_positions['short_positions'][ticker]['stop_loss'] = True
                stop_type = "trailing" if use_trailing_stop else "fixed"
                print(f"Stop loss triggered for {ticker} (short, {stop_type}): {loss_pct:.2%} from {reference_price:.2f}")
    
    return updated_positions


if __name__ == "__main__":
    # Test the module
    dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='D')
    test_prices = pd.DataFrame(
        np.random.randn(len(dates), 20).cumsum(axis=0) + 100,
        index=dates,
        columns=[f'TICKER_{i}' for i in range(20)]
    )
    
    long_tickers = [f'TICKER_{i}' for i in range(10)]
    short_tickers = [f'TICKER_{i}' for i in range(10, 20)]
    
    portfolio = construct_portfolio(long_tickers, short_tickers, test_prices)
    print(f"Portfolio constructed:")
    print(f"  Long positions: {len(portfolio['long_positions'])}")
    print(f"  Short positions: {len(portfolio['short_positions'])}")
    print(f"  Gross exposure: {portfolio['gross_exposure']:.2f}x")

