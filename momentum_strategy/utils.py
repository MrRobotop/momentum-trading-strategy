"""
Utility Functions and Visualization Module
==========================================

Contains utility functions, visualization tools, and helper classes
for the momentum trading strategy framework.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class Visualizer:
    """
    Creates professional visualizations for strategy analysis
    
    This class provides comprehensive visualization tools including:
    - Performance charts and dashboards
    - Risk analysis plots
    - Portfolio composition analysis
    - Interactive Plotly dashboards
    - Publication-ready matplotlib charts
    """
    
    def __init__(self, results: Dict, metrics: Dict = None):
        """
        Initialize Visualizer
        
        Args:
            results: Backtest results dictionary
            metrics: Performance metrics dictionary
        """
        self.results = results
        self.metrics = metrics or {}
        
        # Set up color scheme
        self.colors = {
            'strategy': '#2563eb',      # Blue
            'benchmark': '#dc2626',     # Red  
            'positive': '#16a34a',      # Green
            'negative': '#dc2626',      # Red
            'neutral': '#64748b',       # Gray
            'accent': '#8b5cf6'         # Purple
        }
    
    def create_performance_dashboard(self, save_path: Optional[str] = None) -> go.Figure:
        """
        Create comprehensive interactive performance dashboard
        
        Args:
            save_path: Path to save the dashboard HTML file
            
        Returns:
            Plotly figure object
        """
        logger.info("Creating interactive performance dashboard...")
        
        # Get data
        strategy_returns = self.results['strategy_returns'].dropna()
        benchmark_returns = self.results['benchmark_returns']
        
        # Align series
        common_dates = strategy_returns.index.intersection(benchmark_returns.index)
        strat_ret = strategy_returns[common_dates]
        bench_ret = benchmark_returns[common_dates]
        
        # Calculate cumulative returns
        strat_cum = (1 + strat_ret).cumprod()
        bench_cum = (1 + bench_ret).cumprod()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Cumulative Performance', 'Rolling Sharpe Ratio (252D)',
                'Drawdown Analysis', 'Monthly Returns Distribution',
                'Position Concentration', 'Rolling Volatility (252D)'
            ],
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}]],
            vertical_spacing=0.12
        )
        
        # 1. Cumulative Performance
        fig.add_trace(
            go.Scatter(
                x=strat_cum.index, y=strat_cum.values,
                name='Strategy', line=dict(color=self.colors['strategy'], width=2)
            ), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=bench_cum.index, y=bench_cum.values,
                name='Benchmark', line=dict(color=self.colors['benchmark'], width=2)
            ), row=1, col=1
        )
        
        # 2. Rolling Sharpe Ratio
        rolling_sharpe = self._calculate_rolling_sharpe(strat_ret)
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index, y=rolling_sharpe.values,
                name='Rolling Sharpe', line=dict(color=self.colors['accent'], width=2),
                showlegend=False
            ), row=1, col=2
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=2)
        
        # 3. Drawdown
        rolling_max = strat_cum.expanding().max()
        drawdown = (strat_cum - rolling_max) / rolling_max
        fig.add_trace(
            go.Scatter(
                x=drawdown.index, y=drawdown.values,
                name='Drawdown', fill='tonexty', 
                line=dict(color=self.colors['negative']),
                showlegend=False
            ), row=2, col=1
        )
        
        # 4. Monthly Returns Distribution
        monthly_rets = strat_ret.resample('M').apply(lambda x: (1+x).prod()-1)
        fig.add_trace(
            go.Histogram(
                x=monthly_rets.values, nbinsx=20,
                name='Monthly Returns',
                marker_color=self.colors['strategy'],
                opacity=0.7,
                showlegend=False
            ), row=2, col=2
        )
        
        # 5. Position Concentration
        if 'positions' in self.results:
            positions = self.results['positions']
            concentration = (positions > 0).sum(axis=1).rolling(21).mean()
            fig.add_trace(
                go.Scatter(
                    x=concentration.index, y=concentration.values,
                    name='Active Positions', 
                    line=dict(color=self.colors['accent']),
                    showlegend=False
                ), row=3, col=1
            )
        
        # 6. Rolling Volatility
        rolling_vol = strat_ret.rolling(252).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index, y=rolling_vol.values,
                name='Rolling Volatility',
                line=dict(color=self.colors['positive']),
                showlegend=False
            ), row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Momentum Strategy Performance Dashboard",
            title_x=0.5,
            title_font_size=20,
            showlegend=True,
            legend=dict(x=0, y=1, xanchor='left', yanchor='top')
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=2)
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)
        fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        fig.update_yaxes(title_text="Number of Positions", row=3, col=1)
        fig.update_yaxes(title_text="Volatility", row=3, col=2)
        
        if save_path:
            fig.write_html(f"{save_path}_dashboard.html")
            logger.info(f"Dashboard saved to {save_path}_dashboard.html")
        
        return fig
    
    def plot_performance_summary(self, figsize: Tuple[int, int] = (15, 10),
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive matplotlib performance summary
        
        Args:
            figsize: Figure size tuple
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        logger.info("Creating performance summary plot...")
        
        # Get data
        strategy_returns = self.results['strategy_returns'].dropna()
        benchmark_returns = self.results['benchmark_returns']
        
        # Align series
        common_dates = strategy_returns.index.intersection(benchmark_returns.index)
        strat_ret = strategy_returns[common_dates]
        bench_ret = benchmark_returns[common_dates]
        
        # Calculate cumulative returns
        strat_cum = (1 + strat_ret).cumprod()
        bench_cum = (1 + bench_ret).cumprod()
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Momentum Strategy Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Cumulative Returns
        axes[0,0].plot(strat_cum.index, strat_cum.values, 
                      label='Strategy', linewidth=2, color=self.colors['strategy'])
        axes[0,0].plot(bench_cum.index, bench_cum.values, 
                      label='Benchmark', linewidth=2, color=self.colors['benchmark'])
        axes[0,0].set_title('Cumulative Performance', fontweight='bold')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_ylabel('Cumulative Return')
        
        # 2. Drawdown
        rolling_max = strat_cum.expanding().max()
        drawdown = (strat_cum - rolling_max) / rolling_max
        axes[0,1].fill_between(drawdown.index, drawdown.values, 0, 
                              alpha=0.3, color=self.colors['negative'])
        axes[0,1].plot(drawdown.index, drawdown.values, 
                      color=self.colors['negative'], linewidth=1)
        axes[0,1].set_title('Strategy Drawdown', fontweight='bold')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].set_ylabel('Drawdown %')
        
        # 3. Rolling Returns (Annual)
        rolling_ret = strat_ret.rolling(252).apply(lambda x: (1+x).prod()-1)
        axes[1,0].plot(rolling_ret.index, rolling_ret.values, 
                      color=self.colors['positive'], linewidth=2)
        axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1,0].set_title('Rolling 1-Year Returns', fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].set_ylabel('Annual Return')
        axes[1,0].set_xlabel('Date')
        
        # 4. Monthly Returns Distribution
        monthly_rets = strat_ret.resample('M').apply(lambda x: (1+x).prod()-1)
        axes[1,1].hist(monthly_rets.values, bins=20, alpha=0.7, 
                      color=self.colors['strategy'], edgecolor='black')
        axes[1,1].axvline(x=monthly_rets.mean(), color=self.colors['negative'], 
                         linestyle='--', label=f'Mean: {monthly_rets.mean():.2%}')
        axes[1,1].set_title('Monthly Returns Distribution', fontweight='bold')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].set_xlabel('Monthly Return')
        axes[1,1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_performance_summary.png", dpi=300, bbox_inches='tight')
            logger.info(f"Performance summary saved to {save_path}_performance_summary.png")
        
        return fig
    
    def plot_risk_analysis(self, figsize: Tuple[int, int] = (15, 8),
                          save_path: Optional[str] = None) -> plt.Figure:
        """Create detailed risk analysis charts"""
        
        logger.info("Creating risk analysis plots...")
        
        strategy_returns = self.results['strategy_returns'].dropna()
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Risk Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Return Distribution
        axes[0,0].hist(strategy_returns.values, bins=50, alpha=0.7, 
                      color=self.colors['strategy'], density=True)
        
        # Overlay normal distribution
        mu, sigma = strategy_returns.mean(), strategy_returns.std()
        x = np.linspace(strategy_returns.min(), strategy_returns.max(), 100)
        axes[0,0].plot(x, (1/(sigma * np.sqrt(2 * np.pi))) * 
                      np.exp(-0.5 * ((x - mu) / sigma) ** 2), 
                      'r--', label='Normal')
        axes[0,0].set_title('Return Distribution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Q-Q Plot
        from scipy import stats
        stats.probplot(strategy_returns, dist="norm", plot=axes[0,1])
        axes[0,1].set_title('Q-Q Plot (Normality)')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Rolling VaR
        rolling_var = strategy_returns.rolling(252).quantile(0.05)
        axes[0,2].plot(rolling_var.index, rolling_var.values, 
                      color=self.colors['negative'], linewidth=2)
        axes[0,2].set_title('Rolling VaR (95%)')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Volatility Clustering
        rolling_vol = strategy_returns.rolling(21).std() * np.sqrt(252)
        axes[1,0].plot(rolling_vol.index, rolling_vol.values, 
                      color=self.colors['accent'], linewidth=1)
        axes[1,0].set_title('Rolling Volatility (21D)')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Autocorrelation
        from statsmodels.tsa.stattools import acf
        lags = 20
        autocorr = acf(strategy_returns.dropna(), nlags=lags)
        axes[1,1].bar(range(lags+1), autocorr, color=self.colors['strategy'], alpha=0.7)
        axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1,1].set_title('Return Autocorrelation')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Drawdown Duration
        cumret = (1 + strategy_returns).cumprod()
        rolling_max = cumret.expanding().max()
        drawdown = (cumret - rolling_max) / rolling_max
        
        # Find drawdown periods
        is_drawdown = drawdown < -0.01  # 1% threshold
        drawdown_periods = []
        current_period = 0
        
        for is_dd in is_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if drawdown_periods:
            axes[1,2].hist(drawdown_periods, bins=10, alpha=0.7, 
                          color=self.colors['negative'])
            axes[1,2].set_title('Drawdown Duration Distribution')
            axes[1,2].set_xlabel('Days')
            axes[1,2].set_ylabel('Frequency')
        else:
            axes[1,2].text(0.5, 0.5, 'No Significant\nDrawdowns', 
                          ha='center', va='center', transform=axes[1,2].transAxes)
            axes[1,2].set_title('Drawdown Duration Distribution')
        
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_risk_analysis.png", dpi=300, bbox_inches='tight')
            logger.info(f"Risk analysis saved to {save_path}_risk_analysis.png")
        
        return fig
    
    def plot_portfolio_analysis(self, figsize: Tuple[int, int] = (15, 10),
                               save_path: Optional[str] = None) -> plt.Figure:
        """Create portfolio composition and turnover analysis"""
        
        logger.info("Creating portfolio analysis plots...")
        
        if 'positions' not in self.results:
            logger.warning("No position data available for portfolio analysis")
            return None
        
        positions = self.results['positions']
        turnover = self.results.get('turnover', pd.Series())
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Portfolio Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Position Size Evolution (Top 5 assets)
        top_assets = positions.sum().nlargest(5).index
        for asset in top_assets:
            axes[0,0].plot(positions.index, positions[asset], 
                          label=asset, linewidth=2)
        axes[0,0].set_title('Top 5 Position Sizes Over Time')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_ylabel('Position Size')
        
        # 2. Number of Active Positions
        active_positions = (positions > 0.01).sum(axis=1)  # 1% threshold
        axes[0,1].plot(active_positions.index, active_positions.values, 
                      color=self.colors['accent'], linewidth=2)
        axes[0,1].set_title('Number of Active Positions')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].set_ylabel('Count')
        
        # 3. Turnover Analysis
        if not turnover.empty:
            axes[1,0].plot(turnover.index, turnover.values, 
                          color=self.colors['negative'], linewidth=1)
            axes[1,0].plot(turnover.index, turnover.rolling(21).mean().values, 
                          color=self.colors['strategy'], linewidth=2, label='21D MA')
            axes[1,0].set_title('Portfolio Turnover')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].set_ylabel('Turnover')
            axes[1,0].set_xlabel('Date')
        
        # 4. Position Size Distribution
        all_positions = positions.values.flatten()
        all_positions = all_positions[all_positions > 0.001]  # Remove zeros and tiny positions
        
        if len(all_positions) > 0:
            axes[1,1].hist(all_positions, bins=30, alpha=0.7, 
                          color=self.colors['strategy'], edgecolor='black')
            axes[1,1].axvline(x=all_positions.mean(), color=self.colors['negative'], 
                             linestyle='--', label=f'Mean: {all_positions.mean():.1%}')
            axes[1,1].set_title('Position Size Distribution')
            axes[1,1].legend()
            axes[1,1].set_xlabel('Position Size')
            axes[1,1].set_ylabel('Frequency')
        
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_portfolio_analysis.png", dpi=300, bbox_inches='tight')
            logger.info(f"Portfolio analysis saved to {save_path}_portfolio_analysis.png")
        
        return fig
    
    def plot_signal_analysis(self, figsize: Tuple[int, int] = (15, 8),
                           save_path: Optional[str] = None) -> plt.Figure:
        """Create momentum signal analysis charts"""
        
        logger.info("Creating signal analysis plots...")
        
        if 'momentum_scores' not in self.results:
            logger.warning("No momentum scores available for signal analysis")
            return None
        
        momentum_scores = self.results['momentum_scores']
        rankings = self.results.get('rankings', pd.DataFrame())
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Momentum Signal Analysis', fontsize=16, fontweight='bold')
        
        # 1. Signal Distribution
        all_scores = momentum_scores.values.flatten()
        all_scores = all_scores[~np.isnan(all_scores)]
        
        axes[0,0].hist(all_scores, bins=50, alpha=0.7, 
                      color=self.colors['strategy'], density=True)
        axes[0,0].set_title('Momentum Score Distribution')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_xlabel('Momentum Score')
        axes[0,0].set_ylabel('Density')
        
        # 2. Signal Stability (Top 5 assets by exposure)
        if 'positions' in self.results:
            top_assets = self.results['positions'].sum().nlargest(5).index
            for asset in top_assets:
                if asset in momentum_scores.columns:
                    axes[0,1].plot(momentum_scores.index, momentum_scores[asset], 
                                  label=asset, alpha=0.8)
            axes[0,1].set_title('Momentum Scores - Top Holdings')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].set_ylabel('Momentum Score')
        
        # 3. Cross-sectional Signal Dispersion
        signal_std = momentum_scores.std(axis=1)
        axes[1,0].plot(signal_std.index, signal_std.values, 
                      color=self.colors['accent'], linewidth=2)
        axes[1,0].set_title('Cross-Sectional Signal Dispersion')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].set_ylabel('Signal Std Dev')
        axes[1,0].set_xlabel('Date')
        
        # 4. Signal Rank Stability
        if not rankings.empty:
            # Calculate rank correlation over time
            rank_corr = []
            for i in range(21, len(rankings)):  # 21-day lookback
                current_ranks = rankings.iloc[i].dropna()
                past_ranks = rankings.iloc[i-21].dropna()
                common_assets = current_ranks.index.intersection(past_ranks.index)
                
                if len(common_assets) > 5:
                    corr = current_ranks[common_assets].corr(past_ranks[common_assets], method='spearman')
                    rank_corr.append(corr)
                else:
                    rank_corr.append(np.nan)
            
            rank_corr_series = pd.Series(rank_corr, index=rankings.index[21:])
            axes[1,1].plot(rank_corr_series.index, rank_corr_series.values, 
                          color=self.colors['positive'], linewidth=2)
            axes[1,1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            axes[1,1].set_title('21-Day Rank Correlation')
            axes[1,1].grid(True, alpha=0.3)
            axes[1,1].set_ylabel('Rank Correlation')
            axes[1,1].set_xlabel('Date')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_signal_analysis.png", dpi=300, bbox_inches='tight')
            logger.info(f"Signal analysis saved to {save_path}_signal_analysis.png")
        
        return fig
    
    def create_tearsheet(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create a comprehensive strategy tearsheet"""
        
        logger.info("Creating strategy tearsheet...")
        
        fig = plt.figure(figsize=(20, 24))
        
        # Create a grid layout
        gs = fig.add_gridspec(6, 3, hspace=0.3, wspace=0.3)
        
        # Get data
        strategy_returns = self.results['strategy_returns'].dropna()
        benchmark_returns = self.results['benchmark_returns']
        common_dates = strategy_returns.index.intersection(benchmark_returns.index)
        strat_ret = strategy_returns[common_dates]
        bench_ret = benchmark_returns[common_dates]
        
        # 1. Header with key metrics (spans full width)
        ax_header = fig.add_subplot(gs[0, :])
        ax_header.axis('off')
        
        # Calculate key metrics for header
        total_ret = (1 + strat_ret).prod() - 1
        annual_ret = (1 + total_ret) ** (252 / len(strat_ret)) - 1
        vol = strat_ret.std() * np.sqrt(252)
        sharpe = (annual_ret - 0.02) / vol  # Assuming 2% risk-free rate
        max_dd = ((1 + strat_ret).cumprod() / (1 + strat_ret).cumprod().expanding().max() - 1).min()
        
        header_text = f"""
        MOMENTUM STRATEGY TEARSHEET
        
        Total Return: {total_ret:.1%}    Annual Return: {annual_ret:.1%}    Volatility: {vol:.1%}    Sharpe Ratio: {sharpe:.2f}    Max Drawdown: {max_dd:.1%}
        
        Period: {strat_ret.index[0].strftime('%Y-%m-%d')} to {strat_ret.index[-1].strftime('%Y-%m-%d')}    Total Days: {len(strat_ret)}
        """
        
        ax_header.text(0.5, 0.5, header_text, ha='center', va='center', 
                      fontsize=14, fontweight='bold', transform=ax_header.transAxes)
        
        # 2. Performance charts
        # Cumulative returns
        ax1 = fig.add_subplot(gs[1, :2])
        strat_cum = (1 + strat_ret).cumprod()
        bench_cum = (1 + bench_ret).cumprod()
        ax1.plot(strat_cum.index, strat_cum.values, label='Strategy', 
                color=self.colors['strategy'], linewidth=2)
        ax1.plot(bench_cum.index, bench_cum.values, label='Benchmark', 
                color=self.colors['benchmark'], linewidth=2)
        ax1.set_title('Cumulative Performance', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        ax2 = fig.add_subplot(gs[1, 2])
        rolling_max = strat_cum.expanding().max()
        drawdown = (strat_cum - rolling_max) / rolling_max
        ax2.fill_between(drawdown.index, drawdown.values, 0, 
                        alpha=0.3, color=self.colors['negative'])
        ax2.plot(drawdown.index, drawdown.values, 
                color=self.colors['negative'], linewidth=1)
        ax2.set_title('Drawdown', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Continue with more charts...
        # You can add more visualization components here
        
        plt.suptitle('Strategy Performance Tearsheet', fontsize=20, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(f"{save_path}_tearsheet.png", dpi=300, bbox_inches='tight')
            logger.info(f"Tearsheet saved to {save_path}_tearsheet.png")
        
        return fig
    
    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """Calculate rolling Sharpe ratio"""
        
        excess_returns = returns - 0.02 / 252  # Assuming 2% risk-free rate
        rolling_mean = excess_returns.rolling(window).mean() * 252
        rolling_std = returns.rolling(window).std() * np.sqrt(252)
        
        return rolling_mean / rolling_std

# Utility Functions
class StrategyUtils:
    """Utility functions for strategy development and analysis"""
    
    @staticmethod
    def format_metrics_for_display(metrics: Dict) -> Dict:
        """Format metrics dictionary for display purposes"""
        
        formatted = {}
        
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'ratio' in key.lower() or 'beta' in key.lower():
                    formatted[key] = f"{value:.3f}"
                elif 'return' in key.lower() or 'alpha' in key.lower() or '%' in str(value):
                    formatted[key] = f"{value:.2%}"
                elif 'var' in key.lower() or 'drawdown' in key.lower():
                    formatted[key] = f"{value:.2%}"
                else:
                    formatted[key] = f"{value:.4f}"
            else:
                formatted[key] = str(value)
        
        return formatted
    
    @staticmethod
    def calculate_performance_attribution(strategy_returns: pd.Series,
                                       factor_returns: pd.DataFrame) -> Dict:
        """
        Calculate performance attribution to risk factors
        
        Args:
            strategy_returns: Strategy return series
            factor_returns: DataFrame with factor returns
            
        Returns:
            Attribution analysis results
        """
        # Simple linear regression attribution
        from sklearn.linear_model import LinearRegression
        
        # Align data
        common_dates = strategy_returns.index.intersection(factor_returns.index)
        y = strategy_returns[common_dates].values.reshape(-1, 1)
        X = factor_returns.loc[common_dates].values
        
        # Fit regression
        model = LinearRegression().fit(X, y.ravel())
        
        # Calculate attribution
        factor_exposures = model.coef_
        alpha = model.intercept_
        r_squared = model.score(X, y.ravel())
        
        attribution = {
            'alpha': alpha * 252,  # Annualized
            'factor_exposures': dict(zip(factor_returns.columns, factor_exposures)),
            'r_squared': r_squared
        }
        
        return attribution
    
    @staticmethod
    def generate_summary_statistics(returns: pd.Series) -> Dict:
        """Generate comprehensive summary statistics"""
        
        stats = {
            'count': len(returns),
            'mean': returns.mean(),
            'std': returns.std(),
            'min': returns.min(),
            'max': returns.max(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'median': returns.median(),
            'percentile_5': returns.quantile(0.05),
            'percentile_95': returns.quantile(0.95)
        }
        
        return stats

# Export functions
def save_results_to_excel(results: Dict, metrics: Dict, file_path: str):
    """Save complete results to Excel file"""
    
    logger.info(f"Saving results to {file_path}.xlsx...")
    
    with pd.ExcelWriter(f"{file_path}.xlsx", engine='openpyxl') as writer:
        # Performance metrics
        if metrics:
            metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
            metrics_df.to_excel(writer, sheet_name='Performance_Metrics')
        
        # Returns
        if 'strategy_returns' in results:
            results['strategy_returns'].to_excel(writer, sheet_name='Strategy_Returns')
        
        if 'benchmark_returns' in results:
            results['benchmark_returns'].to_excel(writer, sheet_name='Benchmark_Returns')
        
        # Positions
        if 'positions' in results:
            results['positions'].to_excel(writer, sheet_name='Positions')
        
        # Momentum scores
        if 'momentum_scores' in results:
            results['momentum_scores'].to_excel(writer, sheet_name='Momentum_Scores')
        
        # Turnover
        if 'turnover' in results:
            results['turnover'].to_excel(writer, sheet_name='Turnover')
    
    logger.info("Results saved successfully")

def load_results_from_excel(file_path: str) -> Dict:
    """Load results from Excel file"""
    
    logger.info(f"Loading results from {file_path}.xlsx...")
    
    results = {}
    
    try:
        # Load each sheet
        xl_file = pd.ExcelFile(f"{file_path}.xlsx")
        
        for sheet_name in xl_file.sheet_names:
            df = pd.read_excel(xl_file, sheet_name=sheet_name, index_col=0)
            results[sheet_name.lower()] = df
        
        logger.info("Results loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        return {}
    
    return results