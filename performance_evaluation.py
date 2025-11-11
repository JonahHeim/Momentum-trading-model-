"""
Performance Evaluation Module
Calculates performance metrics and creates visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


def calculate_metrics(returns: pd.Series, portfolio_value: pd.Series) -> Dict:
    """
    Calculate performance metrics.
    
    Args:
        returns: Series of period returns
        portfolio_value: Series of portfolio values
        
    Returns:
        Dictionary with performance metrics
    """
    returns = returns.dropna()
    
    if len(returns) == 0:
        return {}
    
    # Basic metrics
    total_return = (portfolio_value.iloc[-1] - portfolio_value.iloc[0]) / portfolio_value.iloc[0]
    mean_return = returns.mean()
    volatility = returns.std() * np.sqrt(252)  # Annualized
    
    # Sharpe Ratio (assuming risk-free rate = 0)
    sharpe_ratio = (mean_return * 252) / volatility if volatility > 0 else 0
    
    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win rate
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
    
    # Average win/loss
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    # CAGR
    if len(portfolio_value) > 1:
        days = (portfolio_value.index[-1] - portfolio_value.index[0]).days
        years = days / 365.25
        cagr = ((portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (1 / years) - 1) if years > 0 else 0
    else:
        cagr = 0
    
    metrics = {
        'Total Return': total_return,
        'CAGR': cagr,
        'Mean Daily Return': mean_return,
        'Volatility (Annualized)': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Average Win': avg_win,
        'Average Loss': avg_loss,
        'Win/Loss Ratio': win_loss_ratio,
        'Total Trades': len(returns)
    }
    
    return metrics


def plot_performance(
    portfolio_value: pd.Series,
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    save_path: Optional[str] = None
):
    """
    Create performance visualization plots.
    
    Args:
        portfolio_value: Series of portfolio values
        returns: Series of period returns
        benchmark: Optional benchmark series for comparison
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Advanced Momentum Strategy - Backtest Performance', fontsize=16, fontweight='bold')
    
    # Plot 1: Portfolio Value Over Time
    ax1 = axes[0]
    ax1.plot(portfolio_value.index, portfolio_value.values, label='Strategy', linewidth=2, color='#1A237E')
    
    if benchmark is not None:
        # Normalize benchmark to start at same value
        benchmark_norm = benchmark / benchmark.iloc[0] * portfolio_value.iloc[0]
        ax1.plot(benchmark_norm.index, benchmark_norm.values, label='Benchmark (S&P 500)', 
                linewidth=2, color='#FFD700', alpha=0.7)
    
    ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Returns Distribution
    ax2 = axes[1]
    returns_clean = returns.dropna()
    ax2.hist(returns_clean, bins=50, alpha=0.7, color='#1A237E', edgecolor='black')
    ax2.axvline(returns_clean.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns_clean.mean():.4f}')
    ax2.set_title('Returns Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Daily Return')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Drawdown
    ax3 = axes[2]
    cumulative = (1 + returns_clean).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    ax3.fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color='red')
    ax3.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1.5)
    ax3.set_title('Drawdown Over Time', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Drawdown')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def print_metrics_table(metrics: Dict):
    """
    Print performance metrics in a formatted table.
    
    Args:
        metrics: Dictionary with performance metrics
    """
    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'Return' in key or 'CAGR' in key or 'Rate' in key or 'Drawdown' in key:
                print(f"{key:.<40} {value:>15.2%}")
            elif 'Ratio' in key:
                print(f"{key:.<40} {value:>15.2f}")
            elif 'Volatility' in key:
                print(f"{key:.<40} {value:>15.2%}")
            else:
                print(f"{key:.<40} {value:>15.4f}")
        else:
            print(f"{key:.<40} {value:>15}")
    
    print("="*60 + "\n")


def generate_report(
    results: Dict,
    save_path: Optional[str] = None,
    benchmark: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Generate comprehensive performance report.
    
    Args:
        results: Dictionary with backtest results
        save_path: Optional path to save report
        benchmark: Optional benchmark for comparison
        
    Returns:
        DataFrame with metrics
    """
    portfolio_value = results['portfolio_value']
    returns = results['returns']
    
    # Calculate metrics
    metrics = calculate_metrics(returns, portfolio_value)
    
    # Print metrics
    print_metrics_table(metrics)
    
    # Create visualization
    plot_performance(portfolio_value, returns, benchmark, save_path)
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame([metrics]).T
    metrics_df.columns = ['Value']
    
    # Save to Excel if path provided
    if save_path:
        excel_path = save_path.replace('.png', '.xlsx') if save_path.endswith('.png') else save_path + '.xlsx'
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            metrics_df.to_excel(writer, sheet_name='Metrics')
            pd.DataFrame({'Date': portfolio_value.index, 'Portfolio Value': portfolio_value.values}).to_excel(
                writer, sheet_name='Portfolio Value', index=False
            )
            pd.DataFrame({'Date': returns.index, 'Returns': returns.values}).to_excel(
                writer, sheet_name='Returns', index=False
            )
            if 'trades' in results and not results['trades'].empty:
                results['trades'].to_excel(writer, sheet_name='Trades', index=False)
        print(f"Report saved to {excel_path}")
    
    return metrics_df


if __name__ == "__main__":
    # Test the module
    dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='D')
    test_returns = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)
    test_portfolio = 100000 * (1 + test_returns).cumprod()
    
    metrics = calculate_metrics(test_returns, test_portfolio)
    print_metrics_table(metrics)

