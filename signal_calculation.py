"""
Signal Calculation Module
Computes multi-factor signals: momentum, RSI, MA, volatility, and volume.
Normalizes and blends signals into composite scores.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict

# Lazy loading of ML libraries to avoid import errors when not needed
LIGHTGBM_AVAILABLE = False
SKLEARN_AVAILABLE = False

def _check_ml_libraries():
    """Check if ML libraries are available (lazy import)."""
    global LIGHTGBM_AVAILABLE, SKLEARN_AVAILABLE
    
    if not LIGHTGBM_AVAILABLE:
        try:
            import lightgbm as lgb
            LIGHTGBM_AVAILABLE = True
        except (ImportError, OSError, Exception):
            LIGHTGBM_AVAILABLE = False
    
    if not SKLEARN_AVAILABLE:
        try:
            from sklearn.ensemble import RandomForestRegressor
            SKLEARN_AVAILABLE = True
        except ImportError:
            SKLEARN_AVAILABLE = False
    
    return LIGHTGBM_AVAILABLE or SKLEARN_AVAILABLE


def calculate_returns(prices: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    """
    Calculate returns for multiple lookback periods.
    
    Args:
        prices: DataFrame with adjusted close prices
        periods: List of lookback periods (e.g., [5, 10, 20])
        
    Returns:
        Dictionary with returns for each period
    """
    returns = {}
    for period in periods:
        returns[period] = prices.pct_change(period)
    return returns


def calculate_volatility(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate rolling volatility (standard deviation of returns).
    
    Args:
        prices: DataFrame with adjusted close prices
        window: Rolling window size in days
        
    Returns:
        DataFrame with rolling volatility
    """
    daily_returns = prices.pct_change()
    volatility = daily_returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    return volatility


def calculate_rsi(prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Relative Strength Index (RSI).
    RSI ranges from 0-100, with >70 overbought, <30 oversold.
    
    Args:
        prices: DataFrame with adjusted close prices
        period: RSI period (default 14)
        
    Returns:
        DataFrame with RSI values (0-100)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / (loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_rsi_mean_reversion(prices: pd.DataFrame, period: int = 2) -> pd.DataFrame:
    """
    Calculate short-period RSI for mean reversion detection.
    RSI(2) or RSI(5) can detect extreme overbought/oversold conditions.
    
    Args:
        prices: DataFrame with adjusted close prices
        period: RSI period (2 or 5 for mean reversion)
        
    Returns:
        DataFrame with RSI values (0-100)
        Low values (<20) indicate oversold (buy signal)
        High values (>80) indicate overbought (sell signal)
    """
    return calculate_rsi(prices, period=period)


def calculate_moving_average_signals(
    prices: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 50
) -> pd.DataFrame:
    """
    Calculate moving average crossover signals.
    Positive when short MA > long MA (bullish), negative when short MA < long MA (bearish).
    
    Args:
        prices: DataFrame with adjusted close prices
        short_window: Short moving average window
        long_window: Long moving average window
        
    Returns:
        DataFrame with MA signals (normalized to -1 to 1 range)
    """
    short_ma = prices.rolling(window=short_window).mean()
    long_ma = prices.rolling(window=long_window).mean()
    
    # Calculate distance between MAs as percentage
    ma_ratio = (short_ma / (long_ma + 1e-10) - 1)
    
    # Normalize to -1 to 1 range (clip extreme values)
    ma_signal = ma_ratio.clip(-0.2, 0.2) / 0.2
    
    return ma_signal


def calculate_volume_trend(volumes: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Calculate volume trend - positive if volume is increasing.
    
    Args:
        volumes: DataFrame with volume data
        window: Window for volume trend calculation
        
    Returns:
        DataFrame with volume trend scores (positive = increasing volume)
    """
    # Calculate rolling average volume
    avg_volume = volumes.rolling(window=window).mean()
    recent_avg = volumes.rolling(window=3).mean()
    
    # Volume trend: recent avg vs longer avg
    volume_trend = (recent_avg / (avg_volume + 1e-6) - 1)
    
    return volume_trend


def calculate_relative_strength(
    prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    periods: Tuple[int, int, int] = (5, 10, 20),
    weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)
) -> pd.DataFrame:
    """
    Calculate relative strength vs benchmark (SPY).
    Uses weighted average of relative returns across multiple periods.
    
    Args:
        prices: DataFrame with stock prices
        benchmark_prices: Series with benchmark (SPY) prices
        periods: Lookback periods for returns
        weights: Weights for each period (same as momentum weights)
        
    Returns:
        DataFrame with relative strength scores
    """
    # Align benchmark prices with stock prices
    aligned_bench = benchmark_prices.reindex(prices.index).ffill()
    
    # Calculate stock returns
    stock_returns = calculate_returns(prices, list(periods))
    
    # Calculate benchmark returns
    benchmark_returns = {}
    for period in periods:
        benchmark_returns[period] = aligned_bench.pct_change(period)
    
    # Calculate weighted relative strength (stock return - benchmark return)
    relative_strength = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    
    for i, period in enumerate(periods):
        stock_ret = stock_returns[period]
        bench_ret = benchmark_returns[period]
        
        # Calculate relative return for this period
        rel_ret = stock_ret.subtract(bench_ret, axis=0)
        
        # Add weighted contribution
        if relative_strength.empty or relative_strength.isna().all().all():
            relative_strength = weights[i] * rel_ret
        else:
            relative_strength = relative_strength + weights[i] * rel_ret
    
    return relative_strength


def rolling_zscore_normalize(signals: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Rolling z-score normalization: Compare each stock's performance to its own history.
    This helps identify when a stock is performing unusually well/poorly relative to itself.
    
    Args:
        signals: DataFrame with raw signals
        window: Rolling window for z-score calculation
        
    Returns:
        DataFrame with rolling z-score normalized signals
    """
    rolling_mean = signals.rolling(window=window, min_periods=20).mean()
    rolling_std = signals.rolling(window=window, min_periods=20).std()
    z_scores = (signals - rolling_mean) / (rolling_std + 1e-10)
    return z_scores


def calculate_bollinger_bands(
    prices: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: DataFrame with prices
        window: Rolling window
        num_std: Number of standard deviations
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle_band = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    return upper_band, middle_band, lower_band


def calculate_bollinger_percent_b(
    prices: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0
) -> pd.DataFrame:
    """
    Calculate Bollinger Band %B position.
    %B = (Price - Lower Band) / (Upper Band - Lower Band)
    Values > 1 = above upper band, < 0 = below lower band, 0.5 = at middle band.
    
    Args:
        prices: DataFrame with prices
        window: Rolling window
        num_std: Number of standard deviations
        
    Returns:
        DataFrame with %B values
    """
    upper_band, middle_band, lower_band = calculate_bollinger_bands(prices, window, num_std)
    percent_b = (prices - lower_band) / (upper_band - lower_band + 1e-10)
    return percent_b


def calculate_atr(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 14
) -> pd.DataFrame:
    """
    Calculate Average True Range (ATR).
    ATR measures volatility based on high-low range.
    
    Args:
        high: DataFrame with high prices
        low: DataFrame with low prices
        close: DataFrame with close prices
        window: Rolling window
        
    Returns:
        DataFrame with ATR values
    """
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    # Calculate true range for each ticker
    true_range = pd.DataFrame(index=high.index, columns=high.columns)
    for col in high.columns:
        tr_values = pd.concat([
            high_low[col],
            high_close[col],
            low_close[col]
        ], axis=1).max(axis=1)
        true_range[col] = tr_values
    
    atr = true_range.rolling(window=window).mean()
    return atr


def calculate_donchian_channels(
    prices: pd.DataFrame,
    window: int = 20
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate Donchian Channels (highest high and lowest low over window).
    
    Args:
        prices: DataFrame with prices
        window: Rolling window
        
    Returns:
        Tuple of (upper_channel, middle_channel, lower_channel)
    """
    upper_channel = prices.rolling(window=window).max()
    lower_channel = prices.rolling(window=window).min()
    middle_channel = (upper_channel + lower_channel) / 2
    return upper_channel, middle_channel, lower_channel


def calculate_volume_confirmation(
    volumes: pd.DataFrame,
    multiplier: float = 1.5,
    window: int = 20
) -> pd.DataFrame:
    """
    Volume confirmation filter: Only signals with volume above multiplier × average.
    
    Args:
        volumes: DataFrame with volume data
        multiplier: Volume multiplier threshold (default 1.5x)
        window: Window for average volume calculation
        
    Returns:
        DataFrame with binary filter (1 if volume confirmed, 0 otherwise)
    """
    avg_volume = volumes.rolling(window=window).mean()
    volume_ratio = volumes / (avg_volume + 1e-10)
    confirmed = (volume_ratio >= multiplier).astype(float)
    return confirmed


def ml_rank_signals(
    signals_dict: Dict[str, pd.DataFrame],
    forward_returns: Optional[pd.Series] = None,
    use_lightgbm: bool = True
) -> pd.Series:
    """
    Use machine learning (LightGBM or Random Forest) to combine multiple signals
    into one composite ranking score.
    
    Args:
        signals_dict: Dictionary mapping signal names to DataFrames
        forward_returns: Optional forward returns for training (if None, uses equal weights)
        use_lightgbm: Use LightGBM if available, else Random Forest
        
    Returns:
        Series with ML-based composite scores
    """
    if not signals_dict:
        return pd.Series(dtype=float)
    
    # Get the latest date and all tickers
    latest_date = None
    all_tickers = set()
    for signal_df in signals_dict.values():
        if not signal_df.empty:
            if latest_date is None or signal_df.index[-1] > latest_date:
                latest_date = signal_df.index[-1]
            all_tickers.update(signal_df.columns)
    
    if latest_date is None:
        return pd.Series(dtype=float)
    
    # Extract latest signal values for each ticker
    feature_data = []
    ticker_list = []
    
    for ticker in all_tickers:
        features = {}
        valid = True
        for signal_name, signal_df in signals_dict.items():
            if ticker in signal_df.columns:
                try:
                    value = signal_df.loc[latest_date, ticker]
                    if pd.isna(value):
                        valid = False
                        break
                    features[signal_name] = value
                except (KeyError, IndexError):
                    valid = False
                    break
            else:
                valid = False
                break
        
        if valid and len(features) > 0:
            feature_data.append(features)
            ticker_list.append(ticker)
    
    if not feature_data:
        # Fallback to simple weighted average
        return pd.Series(dtype=float)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(feature_data, index=ticker_list)
    
    # If we have forward returns for training, use ML
    if forward_returns is not None and len(forward_returns) > 0:
        # Check ML libraries availability
        _check_ml_libraries()
        
        # Align forward returns with features
        aligned_returns = forward_returns.reindex(ticker_list)
        valid_mask = ~aligned_returns.isna()
        
        if valid_mask.sum() > 10:  # Need at least 10 samples
            X_train = features_df[valid_mask]
            y_train = aligned_returns[valid_mask]
            
            try:
                if use_lightgbm and LIGHTGBM_AVAILABLE:
                    import lightgbm as lgb
                    model = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
                    model.fit(X_train, y_train)
                    predictions = model.predict(features_df)
                elif SKLEARN_AVAILABLE:
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                    model.fit(X_train, y_train)
                    predictions = model.predict(features_df)
                else:
                    # Fallback to simple average
                    predictions = features_df.mean(axis=1).values
                
                return pd.Series(predictions, index=ticker_list)
            except (ImportError, OSError, Exception) as e:
                # Fallback to simple average if ML fails
                pass
    
    # Fallback: Simple weighted average (equal weights)
    composite_scores = features_df.mean(axis=1)
    return composite_scores


def normalize_signals(signals: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
    """
    Normalize signals to comparable scales.
    
    Args:
        signals: DataFrame with raw signals
        method: Normalization method ('zscore', 'minmax', 'rank')
        
    Returns:
        DataFrame with normalized signals
    """
    if method == 'zscore':
        # Z-score normalization: (x - mean) / std
        mean = signals.mean(axis=1)
        std = signals.std(axis=1)
        normalized = signals.sub(mean, axis=0).div(std + 1e-10, axis=0)
    elif method == 'minmax':
        # Min-max normalization to [0, 1]
        min_val = signals.min(axis=1)
        max_val = signals.max(axis=1)
        normalized = signals.sub(min_val, axis=0).div(max_val - min_val + 1e-10, axis=0)
    elif method == 'rank':
        # Rank-based normalization (percentile rank)
        normalized = signals.rank(axis=1, pct=True) * 2 - 1  # Scale to [-1, 1]
    else:
        normalized = signals
    
    return normalized


def calculate_composite_score(
    prices: pd.DataFrame,
    volumes: Optional[pd.DataFrame] = None,
    benchmark_prices: Optional[pd.Series] = None,
    signal_weights: Optional[Dict[str, float]] = None,
    momentum_periods: Tuple[int, int, int] = (5, 10, 20),
    momentum_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    vol_window: int = 20,
    rsi_period: int = 14,
    ma_short: int = 20,
    ma_long: int = 50,
    use_rolling_zscore: bool = True,
    use_rsi_mean_reversion: bool = True,
    use_bollinger: bool = True,
    use_volume_confirmation: bool = True,
    volume_multiplier: float = 1.5,
    use_breakouts: bool = True,
    use_ml_ranking: bool = False,
    high_prices: Optional[pd.DataFrame] = None,
    low_prices: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Calculate enhanced composite score with all new features:
    - Multi-timeframe momentum (5d, 10d, 20d)
    - Volatility-adjusted momentum (Sharpe-like)
    - Residual momentum (vs SPY benchmark)
    - Rolling z-score normalization
    - RSI(2) and RSI(5) for mean reversion
    - Bollinger Band %B
    - Volume confirmation (1.5x filter)
    - ATR and Donchian Channel breakouts
    - Optional ML ranking
    
    Args:
        prices: DataFrame with adjusted close prices
        volumes: Optional DataFrame with volume data
        benchmark_prices: Optional Series with benchmark (SPY) prices
        signal_weights: Dictionary with weights for each signal type
        momentum_periods: Lookback periods for momentum
        momentum_weights: Weights for momentum periods
        vol_window: Window for volatility calculation
        rsi_period: Period for RSI calculation
        ma_short: Short MA window
        ma_long: Long MA window
        use_rolling_zscore: Use rolling z-score normalization
        use_rsi_mean_reversion: Include RSI(2) and RSI(5) signals
        use_bollinger: Include Bollinger Band %B
        use_volume_confirmation: Apply volume confirmation filter
        volume_multiplier: Volume multiplier threshold (default 1.5x)
        use_breakouts: Include ATR/Donchian breakout signals
        use_ml_ranking: Use ML to rank signals (requires lightgbm or sklearn)
        high_prices: Optional DataFrame with high prices (for ATR)
        low_prices: Optional DataFrame with low prices (for ATR)
        
    Returns:
        DataFrame with composite scores
    """
    # Default signal weights (updated with new signals)
    if signal_weights is None:
        signal_weights = {
            'momentum': 0.20,
            'rsi': 0.15,
            'rsi_2': 0.10,
            'rsi_5': 0.10,
            'ma': 0.15,
            'volatility': 0.10,
            'volume': 0.10,
            'bollinger': 0.05,
            'breakout': 0.05
        }
    
    all_signals = {}
    
    # 1. Momentum signal (multi-timeframe, volatility-adjusted, residual)
    returns = calculate_returns(prices, list(momentum_periods))
    momentum_raw = (
        momentum_weights[0] * returns[momentum_periods[0]] +
        momentum_weights[1] * returns[momentum_periods[1]] +
        momentum_weights[2] * returns[momentum_periods[2]]
    )
    
    # Add residual momentum (relative strength vs SPY)
    if benchmark_prices is not None:
        relative_strength = calculate_relative_strength(prices, benchmark_prices, momentum_periods)
        momentum_raw = 0.6 * momentum_raw + 0.4 * relative_strength
    
    # Volatility adjustment (Sharpe-like)
    volatility = calculate_volatility(prices, window=vol_window)
    momentum_adj = momentum_raw / (volatility + 1e-6)
    
    # Apply rolling z-score normalization if enabled
    if use_rolling_zscore:
        momentum_adj = rolling_zscore_normalize(momentum_adj, window=60)
    
    all_signals['momentum'] = momentum_adj
    
    # 2. RSI signal (standard)
    rsi_raw = calculate_rsi(prices, period=rsi_period)
    rsi_signal = (rsi_raw - 50) / 20  # Normalize: 50->0, 70->1, 30->-1
    rsi_signal = rsi_signal.clip(-1, 1)
    all_signals['rsi'] = rsi_signal
    
    # 3. RSI mean reversion signals (RSI(2) and RSI(5))
    if use_rsi_mean_reversion:
        rsi_2 = calculate_rsi_mean_reversion(prices, period=2)
        # Low RSI(2) = oversold (positive signal), High RSI(2) = overbought (negative)
        rsi_2_signal = (50 - rsi_2) / 50  # Invert: low RSI = high signal
        rsi_2_signal = rsi_2_signal.clip(-1, 1)
        all_signals['rsi_2'] = rsi_2_signal
        
        rsi_5 = calculate_rsi_mean_reversion(prices, period=5)
        rsi_5_signal = (50 - rsi_5) / 50
        rsi_5_signal = rsi_5_signal.clip(-1, 1)
        all_signals['rsi_5'] = rsi_5_signal
    
    # 4. Moving Average signal
    ma_signal = calculate_moving_average_signals(prices, short_window=ma_short, long_window=ma_long)
    all_signals['ma'] = ma_signal
    
    # 5. Volatility signal (inverse volatility)
    vol_signal = 1 / (volatility + 1e-6)
    all_signals['volatility'] = vol_signal
    
    # 6. Volume signal
    if volumes is not None:
        volume_signal = calculate_volume_trend(volumes, window=5)
        all_signals['volume'] = volume_signal
        
        # Volume confirmation filter
        if use_volume_confirmation:
            volume_confirmed = calculate_volume_confirmation(volumes, multiplier=volume_multiplier)
            # Apply filter: multiply signals by confirmation (0 or 1)
            for signal_name in list(all_signals.keys()):
                if signal_name != 'volume':
                    all_signals[signal_name] = all_signals[signal_name] * volume_confirmed
    else:
        all_signals['volume'] = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    
    # 7. Bollinger Band %B
    if use_bollinger:
        bb_percent_b = calculate_bollinger_percent_b(prices, window=20)
        # Normalize: %B > 1 = above upper (overbought, negative), %B < 0 = below lower (oversold, positive)
        bb_signal = (0.5 - bb_percent_b) * 2  # Invert: low %B = high signal
        bb_signal = bb_signal.clip(-1, 1)
        all_signals['bollinger'] = bb_signal
    
    # 8. Breakout signals (ATR and Donchian)
    if use_breakouts:
        # Donchian Channel breakout
        upper_ch, middle_ch, lower_ch = calculate_donchian_channels(prices, window=20)
        # Breakout above upper channel = positive signal
        donchian_signal = (prices - middle_ch) / (upper_ch - lower_ch + 1e-10)
        donchian_signal = donchian_signal.clip(-1, 1)
        all_signals['breakout'] = donchian_signal
    
    # Normalize each signal
    normalized_signals = {}
    for signal_name, signal_df in all_signals.items():
        normalized_signals[signal_name] = normalize_signals(signal_df, method='zscore')
    
    # Use ML ranking if enabled and available
    if use_ml_ranking:
        # Check ML libraries availability (lazy check)
        ml_available = _check_ml_libraries()
        if ml_available:
            try:
                # Get latest scores using ML
                latest_scores = ml_rank_signals(normalized_signals)
                # Convert back to DataFrame format
                composite = pd.DataFrame(index=prices.index, columns=prices.columns)
                for ticker in latest_scores.index:
                    if ticker in composite.columns:
                        composite.loc[:, ticker] = latest_scores[ticker]
                composite = composite.fillna(0)
            except (ImportError, OSError, Exception):
                # Fallback to weighted average if ML fails
                ml_available = False
        
        if not ml_available:
            # Fallback to weighted average
            composite = pd.DataFrame(0, index=prices.index, columns=prices.columns)
            for signal_name, weight in signal_weights.items():
                if signal_name in normalized_signals:
                    composite = composite + weight * normalized_signals[signal_name]
    else:
        # Blend signals with weights
        composite = pd.DataFrame(0, index=prices.index, columns=prices.columns)
        
        for signal_name, weight in signal_weights.items():
            if signal_name in normalized_signals:
                composite = composite + weight * normalized_signals[signal_name]
    
    return composite


def calculate_momentum_score(
    prices: pd.DataFrame,
    weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    periods: Tuple[int, int, int] = (5, 10, 20),
    vol_window: int = 20,
    volumes: Optional[pd.DataFrame] = None,
    benchmark_prices: Optional[pd.Series] = None,
    use_relative_strength: bool = True,
    use_volume_confirmation: bool = True
) -> pd.DataFrame:
    """
    Calculate enhanced volatility-adjusted momentum score with relative strength and volume confirmation.
    
    Formula:
        Momentum Score = 0.5 * R_5d + 0.3 * R_10d + 0.2 * R_20d
        Relative Strength = Stock Return - SPY Return
        AdjScore = (Momentum Score + Relative Strength) / Volatility_20d * Volume_Confirmation
    
    Args:
        prices: DataFrame with adjusted close prices
        weights: Weights for 5d, 10d, 20d returns
        periods: Lookback periods for returns
        vol_window: Window for volatility calculation
        volumes: Optional DataFrame with volume data for volume confirmation
        benchmark_prices: Optional Series with benchmark (SPY) prices for relative strength
        use_relative_strength: Whether to use relative strength vs SPY
        use_volume_confirmation: Whether to use volume confirmation
        
    Returns:
        DataFrame with enhanced momentum scores
    """
    # Calculate returns for each period
    returns = calculate_returns(prices, list(periods))
    
    # Calculate composite momentum score
    momentum_score = (
        weights[0] * returns[periods[0]] +
        weights[1] * returns[periods[1]] +
        weights[2] * returns[periods[2]]
    )
    
    # Add relative strength vs SPY if available
    if use_relative_strength and benchmark_prices is not None:
        relative_strength = calculate_relative_strength(prices, benchmark_prices, periods)
        # Weight relative strength (50% momentum, 50% relative strength)
        momentum_score = 0.6 * momentum_score + 0.4 * relative_strength
    
    # Calculate volatility
    volatility = calculate_volatility(prices, window=vol_window)
    
    # Adjust for volatility (avoid division by zero)
    volatility = volatility.replace(0, np.nan)
    adj_score = momentum_score / volatility
    
    # Apply volume confirmation if available
    if use_volume_confirmation and volumes is not None:
        volume_trend = calculate_volume_trend(volumes, window=5)
        # Align indices
        volume_trend = volume_trend.reindex(adj_score.index).fillna(0)
        
        # Boost scores for stocks with increasing volume, penalize decreasing volume
        # Volume multiplier: 1.0 + (volume_trend * 0.3), capped between 0.7 and 1.3
        volume_multiplier = 1.0 + (volume_trend * 0.3).clip(-0.3, 0.3)
        adj_score = adj_score * volume_multiplier
    
    return adj_score


def get_latest_scores(
    scores: pd.DataFrame,
    date: Optional[pd.Timestamp] = None
) -> pd.Series:
    """
    Get latest momentum scores for all tickers.
    
    Args:
        scores: DataFrame with momentum scores
        date: Specific date (defaults to most recent)
        
    Returns:
        Series with scores sorted in descending order
    """
    if date is None:
        latest = scores.iloc[-1]
    else:
        latest = scores.loc[date]
    
    # Drop NaN values and sort
    latest = latest.dropna().sort_values(ascending=False)
    
    return latest


def rank_stocks(scores: pd.Series, n_stocks: int = 10) -> Tuple[pd.Index, pd.Index]:
    """
    Rank stocks and identify top/bottom performers.
    
    Args:
        scores: Series with momentum scores
        n_stocks: Number of stocks to pick for each side (default 10)
                  Picks top N for longs and bottom N for shorts
        
    Returns:
        Tuple of (long_tickers, short_tickers)
    """
    # Sort by score (descending)
    sorted_scores = scores.sort_values(ascending=False)
    
    # Pick top N for longs, bottom N for shorts
    n_long = min(n_stocks, len(sorted_scores))
    n_short = min(n_stocks, len(sorted_scores))
    
    long_tickers = sorted_scores.head(n_long).index
    short_tickers = sorted_scores.tail(n_short).index
    
    return long_tickers, short_tickers


if __name__ == "__main__":
    # Test the module
    dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='D')
    test_prices = pd.DataFrame(
        np.random.randn(len(dates), 10).cumsum(axis=0) + 100,
        index=dates,
        columns=[f'TICKER_{i}' for i in range(10)]
    )
    
    scores = calculate_momentum_score(test_prices)
    latest = get_latest_scores(scores)
    longs, shorts = rank_stocks(latest, n_stocks=10)
    
    print(f"Momentum scores calculated")
    print(f"Long positions: {len(longs)}")
    print(f"Short positions: {len(shorts)}")

