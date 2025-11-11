"""
Universe Selection Module
Fetches S&P 500, Mid-cap (S&P 400), and Small-cap (S&P 600) constituents 
and filters based on liquidity and market cap criteria.
"""

import pandas as pd
import yfinance as yf
import numpy as np
from typing import List, Dict
import warnings
import ssl
import urllib.request
warnings.filterwarnings('ignore')

# SSL context for macOS certificate issues
ssl._create_default_https_context = ssl._create_unverified_context


def get_sp500_tickers_fallback() -> List[str]:
    """
    Fallback list of S&P 500 tickers (top 100 most liquid).
    Used when Wikipedia fetch fails.
    """
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'XOM',
        'JNJ', 'V', 'JPM', 'WMT', 'PG', 'MA', 'CVX', 'LLY', 'ABBV', 'AVGO',
        'PEP', 'COST', 'ADBE', 'MRK', 'TMO', 'CSCO', 'ACN', 'NFLX', 'DHR', 'ABT',
        'WFC', 'LIN', 'PM', 'CRM', 'DIS', 'VZ', 'NKE', 'BMY', 'UPS', 'TXN',
        'QCOM', 'RTX', 'AMD', 'AMGN', 'SPGI', 'INTU', 'HON', 'AMAT', 'DE', 'ISRG',
        'GE', 'BKNG', 'LOW', 'ADP', 'C', 'ELV', 'TJX', 'AXP', 'BLK', 'SYK',
        'ZTS', 'ADI', 'CB', 'ICE', 'CME', 'EQIX', 'REGN', 'KLAC', 'CDNS', 'MCO',
        'SHW', 'MNST', 'SNPS', 'FTNT', 'CTAS', 'CPRT', 'PAYX', 'FAST', 'CHTR', 'ANET',
        'NXPI', 'APH', 'MCHP', 'FTV', 'DASH', 'TDG', 'ODFL', 'CRWD', 'CDW', 'DXCM',
        'GPN', 'ON', 'ANSS', 'WST', 'AON', 'FDS', 'KEYS', 'CTSH', 'BR', 'TSCO'
    ]


def get_sp500_tickers() -> List[str]:
    """
    Scrape S&P 500 ticker symbols from Wikipedia.
    Falls back to hardcoded list if SSL/fetch fails.
    
    Returns:
        List of ticker symbols
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        # Try with SSL context disabled (for macOS certificate issues)
        tables = pd.read_html(url)
        sp500_df = tables[0]
        tickers = sp500_df['Symbol'].tolist()
        # Clean ticker symbols (remove dots for class shares)
        tickers = [t.replace('.', '-') for t in tickers]
        print(f"✅ Successfully fetched {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers
    except Exception as e:
        print(f"⚠️  Error fetching S&P 500 list from Wikipedia: {e}")
        print("Using fallback list of popular S&P 500 tickers...")
        return get_sp500_tickers_fallback()


def get_sp400_tickers_fallback() -> List[str]:
    """
    Fallback list of S&P 400 Mid-cap tickers (popular mid-cap stocks).
    Used when Wikipedia fetch fails.
    """
    return [
        # Technology & SaaS
        'ETSY', 'ROKU', 'DOCU', 'ZM', 'CRWD', 'NET', 'DDOG', 'SNOW', 'PLTR', 'RBLX',
        'FROG', 'ESTC', 'MDB', 'OKTA', 'ZS', 'SPLK', 'TEAM', 'NOW', 'VEEV', 'WDAY',
        'COUP', 'BILL', 'DOCN', 'FRSH', 'ASAN', 'MNDY', 'NCNO', 'AI', 'PATH', 'ESTA',
        'GTLB', 'DOMO', 'QLYS', 'FTCH', 'ALRM',
        # Financial Services & Fintech
        'SOFI', 'UPST', 'AFRM', 'HOOD', 'LC', 'COIN', 'MARA', 'RIOT', 'HUT', 'BITF',
        # Consumer & Retail
        'PTON', 'RKT', 'UWMC', 'CLOV', 'WISH', 'SPCE', 'OPEN',
        # Energy & Mining
        'ARBK', 'CIFR', 'IREN', 'WULF',
        # Healthcare & Biotech
        'BMRN', 'IONS', 'ALKS', 'EXAS', 'INCY', 'SGEN', 'FOLD', 'RGNX', 'ARWR', 'ALNY',
        # Industrial & Materials
        'AXON', 'TTEK', 'WWD', 'VMI', 'AWI', 'FIX', 'GVA', 'MTZ', 'PRIM', 'ROAD'
    ]


def get_sp400_tickers() -> List[str]:
    """
    Scrape S&P 400 Mid-cap ticker symbols from Wikipedia.
    Falls back to hardcoded list if fetch fails.
    
    Returns:
        List of ticker symbols
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
        tables = pd.read_html(url)
        sp400_df = tables[0]
        tickers = sp400_df['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        print(f"✅ Successfully fetched {len(tickers)} S&P 400 (Mid-cap) tickers from Wikipedia")
        return tickers
    except Exception as e:
        print(f"⚠️  Error fetching S&P 400 list from Wikipedia: {e}")
        print("Using fallback list of popular S&P 400 (Mid-cap) tickers...")
        return get_sp400_tickers_fallback()


def get_sp600_tickers_fallback() -> List[str]:
    """
    Fallback list of S&P 600 Small-cap tickers (popular small-cap stocks).
    Used when Wikipedia fetch fails.
    """
    return [
        # Meme/Retail favorites
        'AMC', 'GME', 'BB', 'NOK', 'SNDL', 'TLRY', 'CGC', 'ACB', 'HEXO', 'OGI',
        # Fintech & Crypto
        'LC', 'COIN', 'MARA', 'RIOT', 'HUT', 'BITF',
        # Real Estate & Housing
        'RKT', 'UWMC', 'OPEN', 'SPCE', 'CLOV', 'WISH',
        # Small Tech & SaaS
        'AI', 'PATH', 'ESTA', 'ALRM', 'FROG', 'DOCN', 'FRSH', 'ASAN', 'MNDY', 'NCNO',
        'GTLB', 'DOMO', 'QLYS', 'FTCH', 'ESTC',
        # Energy & Mining
        'ARBK', 'CIFR', 'IREN', 'WULF',
        # Healthcare Small-cap
        'BMRN', 'IONS', 'ALKS', 'EXAS', 'INCY', 'SGEN', 'FOLD', 'RGNX', 'ARWR', 'ALNY',
        # Consumer Small-cap
        'AXON', 'TTEK', 'WWD', 'VMI', 'AWI', 'FIX', 'GVA', 'MTZ', 'PRIM', 'ROAD',
        # Industrial Small-cap
        'HCSG', 'CSWI', 'VSEC', 'TNC', 'KBR', 'DY', 'ESE', 'FORM', 'GFF', 'HURN'
    ]


def get_sp600_tickers() -> List[str]:
    """
    Scrape S&P 600 Small-cap ticker symbols from Wikipedia.
    Falls back to hardcoded list if fetch fails.
    
    Returns:
        List of ticker symbols
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
        tables = pd.read_html(url)
        sp600_df = tables[0]
        tickers = sp600_df['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        print(f"✅ Successfully fetched {len(tickers)} S&P 600 (Small-cap) tickers from Wikipedia")
        return tickers
    except Exception as e:
        print(f"⚠️  Error fetching S&P 600 list from Wikipedia: {e}")
        print("Using fallback list of popular S&P 600 (Small-cap) tickers...")
        return get_sp600_tickers_fallback()


def get_stock_info(ticker: str) -> Dict:
    """
    Get stock information from yfinance.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with stock info (market_cap, volume, price, sector)
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get recent price data for volume check
        hist = stock.history(period="3mo")
        if hist.empty:
            return None
            
        avg_volume = hist['Volume'].mean()
        current_price = hist['Close'].iloc[-1]
        market_cap = info.get('marketCap', 0)
        sector = info.get('sector', 'Unknown')
        
        return {
            'ticker': ticker,
            'market_cap': market_cap,
            'avg_volume': avg_volume,
            'price': current_price,
            'sector': sector
        }
    except Exception as e:
        print(f"Error fetching info for {ticker}: {e}")
        return None


def filter_universe(
    tickers: List[str],
    min_volume: float = 2_000_000,
    min_price: float = 5.0,
    min_market_cap: float = 5_000_000_000,
    sample_size: int = None
) -> List[str]:
    """
    Filter universe based on liquidity and market cap criteria.
    
    Args:
        tickers: List of ticker symbols
        min_volume: Minimum average daily volume (shares)
        min_price: Minimum stock price
        min_market_cap: Minimum market capitalization
        sample_size: If provided, limit to sample_size for faster testing
        
    Returns:
        Filtered list of ticker symbols
    """
    print(f"Filtering universe from {len(tickers)} tickers...")
    
    if sample_size:
        tickers = tickers[:sample_size]
        print(f"Testing with sample of {sample_size} tickers...")
    
    valid_tickers = []
    failed_count = 0
    
    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(tickers)} tickers...")
            
        info = get_stock_info(ticker)
        if info is None:
            failed_count += 1
            continue
            
        # Apply filters
        if (info['avg_volume'] >= min_volume and 
            info['price'] >= min_price and 
            info['market_cap'] >= min_market_cap):
            valid_tickers.append(ticker)
    
    print(f"\nFiltering complete:")
    print(f"  Valid tickers: {len(valid_tickers)}")
    print(f"  Failed/invalid: {failed_count}")
    print(f"  Total processed: {len(tickers)}")
    
    return valid_tickers


def get_universe(
    use_sample: bool = False,
    sample_size: int = 100,
    min_volume: float = 2_000_000,
    min_price: float = 5.0,
    min_market_cap: float = 5_000_000_000,
    include_mid_cap: bool = True,
    include_small_cap: bool = True
) -> List[str]:
    """
    Main function to get filtered universe from S&P 500, Mid-cap, and Small-cap indices.
    
    Args:
        use_sample: If True, use a sample for faster testing
        sample_size: Size of sample if use_sample is True
        min_volume: Minimum average daily volume
        min_price: Minimum stock price
        min_market_cap: Minimum market capitalization (lowered for mid/small cap)
        include_mid_cap: Include S&P 400 Mid-cap stocks
        include_small_cap: Include S&P 600 Small-cap stocks
        
    Returns:
        List of filtered ticker symbols
    """
    all_tickers = []
    
    # Fetch S&P 500 (Large-cap)
    print("Fetching S&P 500 (Large-cap) constituents...")
    sp500_tickers = get_sp500_tickers()
    if sp500_tickers:
        all_tickers.extend(sp500_tickers)
        print(f"  Added {len(sp500_tickers)} S&P 500 tickers")
    
    # Fetch S&P 400 (Mid-cap)
    if include_mid_cap:
        print("Fetching S&P 400 (Mid-cap) constituents...")
        sp400_tickers = get_sp400_tickers()
        if sp400_tickers:
            all_tickers.extend(sp400_tickers)
            print(f"  Added {len(sp400_tickers)} S&P 400 tickers")
    
    # Fetch S&P 600 (Small-cap)
    if include_small_cap:
        print("Fetching S&P 600 (Small-cap) constituents...")
        sp600_tickers = get_sp600_tickers()
        if sp600_tickers:
            all_tickers.extend(sp600_tickers)
            print(f"  Added {len(sp600_tickers)} S&P 600 tickers")
    
    # Remove duplicates
    all_tickers = list(set(all_tickers))
    
    if not all_tickers:
        print("⚠️  No tickers available. Using minimal fallback list...")
        all_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'JNJ']
    
    print(f"\nTotal unique tickers: {len(all_tickers)}")
    
    # Adjust market cap filter for mid/small cap (use lower threshold)
    # But still filter by liquidity
    effective_min_market_cap = min_market_cap
    if include_mid_cap or include_small_cap:
        # Lower threshold for mid/small cap stocks
        effective_min_market_cap = max(1_000_000_000, min_market_cap * 0.2)  # At least $1B for mid/small
    
    # Filter universe
    filtered_tickers = filter_universe(
        all_tickers,
        min_volume=min_volume,
        min_price=min_price,
        min_market_cap=effective_min_market_cap,
        sample_size=sample_size if use_sample else None
    )
    
    return filtered_tickers


if __name__ == "__main__":
    # Test the module
    universe = get_universe(use_sample=True, sample_size=50)
    print(f"\nFinal universe: {len(universe)} tickers")
    print(f"Sample tickers: {universe[:10]}")

