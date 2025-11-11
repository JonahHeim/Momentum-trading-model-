"""
Data Collection Module
Handles fetching and preprocessing price data from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


def fetch_price_data(
    tickers: List[str],
    period: str = "6mo",
    interval: str = "1d",
    return_volumes: bool = False
) -> pd.DataFrame:
    """
    Fetch historical price data for a list of tickers.
    
    Args:
        tickers: List of ticker symbols
        period: Time period (e.g., "6mo", "1y", "2y")
        interval: Data interval (e.g., "1d", "1h")
        return_volumes: If True, also return volume DataFrame
        
    Returns:
        DataFrame with adjusted close prices (columns = tickers, index = dates)
        If return_volumes=True, returns tuple (prices, volumes)
    """
    print(f"Fetching price data for {len(tickers)} tickers...")
    
    try:
        # Use individual downloads - more reliable than batch
        print(f"Downloading data for {len(tickers)} tickers (this may take a moment)...")
        adj_close = pd.DataFrame()
        volumes = pd.DataFrame() if return_volumes else None
        successful = 0
        failed = 0
        
        for i, ticker in enumerate(tickers):
            try:
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i + 1}/{len(tickers)} tickers...")
                
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period, interval=interval)
                
                if hist.empty:
                    failed += 1
                    continue
                
                # Create DataFrame with same index if first ticker
                if adj_close.empty:
                    adj_close = pd.DataFrame(index=hist.index)
                    if return_volumes:
                        volumes = pd.DataFrame(index=hist.index)
                
                # Use Adj Close if available, otherwise use Close
                if 'Adj Close' in hist.columns:
                    adj_close[ticker] = hist['Adj Close']
                elif 'Close' in hist.columns:
                    adj_close[ticker] = hist['Close']
                else:
                    failed += 1
                    continue
                
                # Store volume if requested
                if return_volumes and 'Volume' in hist.columns:
                    volumes[ticker] = hist['Volume']
                
                successful += 1
                
            except Exception as e:
                failed += 1
                if (i + 1) <= 5:  # Only print errors for first few
                    print(f"  ⚠️  Error fetching {ticker}: {str(e)[:50]}")
                continue
        
        print(f"  ✅ Successfully fetched {successful} tickers, {failed} failed")
        
        # Remove any columns with all NaN values
        if not adj_close.empty:
            adj_close = adj_close.dropna(axis=1, how='all')
            
            # Remove rows with all NaN
            adj_close = adj_close.dropna(axis=0, how='all')
            
            if return_volumes and volumes is not None:
                volumes = volumes.dropna(axis=1, how='all')
                volumes = volumes.dropna(axis=0, how='all')
        
        if adj_close.empty:
            print("⚠️  No price data retrieved")
            return (pd.DataFrame(), pd.DataFrame()) if return_volumes else pd.DataFrame()
        
        print(f"Successfully fetched data for {len(adj_close.columns)} tickers")
        print(f"Date range: {adj_close.index[0]} to {adj_close.index[-1]}")
        print(f"Total days: {len(adj_close)}")
        
        if return_volumes:
            return adj_close, volumes
        return adj_close
        
    except Exception as e:
        print(f"Error fetching price data: {e}")
        import traceback
        print("Trying alternative download method...")
        # Fallback: download individually
        adj_close = pd.DataFrame()
        for ticker in tickers[:20]:  # Limit to first 20 to avoid timeout
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period, interval=interval)
                if not hist.empty:
                    if adj_close.empty:
                        adj_close = pd.DataFrame(index=hist.index)
                    if 'Adj Close' in hist.columns:
                        adj_close[ticker] = hist['Adj Close']
                    else:
                        adj_close[ticker] = hist['Close']
            except:
                continue
        
        if not adj_close.empty:
            print(f"✅ Retrieved {len(adj_close.columns)} tickers using fallback method")
            return adj_close
        
        return pd.DataFrame()


def get_vix_data(period: str = "6mo") -> pd.Series:
    """
    Fetch VIX index data for regime filtering.
    
    Args:
        period: Time period
        
    Returns:
        Series with VIX closing values
    """
    try:
        vix = yf.Ticker("^VIX")
        vix_data = vix.history(period=period)
        return vix_data['Close']
    except Exception as e:
        print(f"Error fetching VIX data: {e}")
        return pd.Series()


def get_spy_data(period: str = "6mo") -> pd.Series:
    """
    Fetch SPY (S&P 500 ETF) data for relative strength calculation.
    
    Args:
        period: Time period
        
    Returns:
        Series with SPY adjusted close prices
    """
    try:
        spy = yf.Ticker("SPY")
        spy_data = spy.history(period=period)
        if 'Adj Close' in spy_data.columns:
            return spy_data['Adj Close']
        else:
            return spy_data['Close']
    except Exception as e:
        print(f"Error fetching SPY data: {e}")
        return pd.Series()


def check_vix_regime(vix_data: pd.Series, threshold: float = 25.0, date: Optional[pd.Timestamp] = None) -> bool:
    """
    Check if market is in risk-on regime (VIX below threshold).
    
    Args:
        vix_data: Series with VIX values
        threshold: VIX threshold (default 25)
        date: Specific date to check (defaults to most recent)
        
    Returns:
        True if VIX < threshold (risk-on), False otherwise
    """
    if vix_data.empty:
        return True  # Default to trading if VIX unavailable
    
    if date is None:
        current_vix = vix_data.iloc[-1]
    else:
        # Find closest date
        closest_date = vix_data.index[vix_data.index <= date]
        if len(closest_date) == 0:
            return True  # Default to trading if no data
        current_vix = vix_data.loc[closest_date[-1]]
    
    return current_vix < threshold


def check_spy_regime(
    spy_prices: pd.Series,
    lookback_days: int = 20,
    threshold_pct: float = -0.05,
    date: Optional[pd.Timestamp] = None
) -> bool:
    """
    Check if SPY is in bullish regime (trending up).
    
    Args:
        spy_prices: Series with SPY prices
        lookback_days: Number of days to look back for trend
        threshold_pct: Minimum return threshold (default -5% means must be above -5% loss)
        date: Specific date to check (defaults to most recent)
        
    Returns:
        True if SPY is in bullish regime, False otherwise
    """
    if spy_prices.empty:
        return True  # Default to trading if SPY unavailable
    
    if date is None:
        check_date = spy_prices.index[-1]
        price_data = spy_prices
    else:
        # Find closest date
        closest_date = spy_prices.index[spy_prices.index <= date]
        if len(closest_date) == 0:
            return True
        check_date = closest_date[-1]
        price_data = spy_prices.loc[:check_date]
    
    if len(price_data) < lookback_days:
        return True  # Not enough data
    
    # Calculate return over lookback period
    current_price = price_data.iloc[-1]
    past_price = price_data.iloc[-lookback_days]
    return_pct = (current_price / past_price - 1)
    
    # Also check if SPY is above its moving average (trend filter)
    ma = price_data.rolling(window=lookback_days).mean().iloc[-1]
    above_ma = current_price > ma
    
    # Bullish if: return > threshold AND above MA
    is_bullish = (return_pct > threshold_pct) and above_ma
    
    return is_bullish


def check_market_regime(
    vix_data: pd.Series,
    spy_prices: pd.Series,
    vix_threshold: float = 25.0,
    spy_lookback: int = 20,
    spy_threshold: float = -0.05,
    date: Optional[pd.Timestamp] = None
) -> Dict[str, bool]:
    """
    Check both VIX and SPY regime conditions.
    
    Args:
        vix_data: Series with VIX values
        spy_prices: Series with SPY prices
        vix_threshold: VIX threshold
        spy_lookback: SPY lookback days
        spy_threshold: SPY return threshold
        date: Specific date to check
        
    Returns:
        Dictionary with regime checks: {'vix_ok', 'spy_ok', 'overall_ok'}
    """
    vix_ok = check_vix_regime(vix_data, threshold=vix_threshold, date=date)
    spy_ok = check_spy_regime(spy_prices, lookback_days=spy_lookback, threshold_pct=spy_threshold, date=date)
    
    return {
        'vix_ok': vix_ok,
        'spy_ok': spy_ok,
        'overall_ok': vix_ok and spy_ok  # Both must be OK
    }


def get_sector_info(ticker: str) -> str:
    """
    Get sector information for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Sector name
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get('sector', 'Unknown')
    except:
        return 'Unknown'


def validate_data(prices: pd.DataFrame, min_days: int = 60) -> pd.DataFrame:
    """
    Validate and clean price data.
    
    Args:
        prices: DataFrame with price data
        min_days: Minimum number of days required
        
    Returns:
        Cleaned DataFrame
    """
    # Remove tickers with insufficient data
    valid_tickers = []
    for col in prices.columns:
        non_null_count = prices[col].notna().sum()
        if non_null_count >= min_days:
            valid_tickers.append(col)
    
    cleaned = prices[valid_tickers].copy()
    
    # Forward fill missing values (within reason)
    cleaned = cleaned.ffill(limit=5)
    
    # Drop rows with all NaN
    cleaned = cleaned.dropna(how='all')
    
    print(f"Data validation: {len(valid_tickers)}/{len(prices.columns)} tickers passed")
    
    return cleaned


if __name__ == "__main__":
    # Test the module
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    prices = fetch_price_data(test_tickers, period="3mo")
    print(f"\nSample data shape: {prices.shape}")
    print(f"\nFirst few rows:\n{prices.head()}")

