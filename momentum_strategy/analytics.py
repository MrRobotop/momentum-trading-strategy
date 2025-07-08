"""
Performance Analytics Module
===========================

Advanced performance and risk analytics for momentum trading strategies.
Implements institutional-grade metrics and attribution analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
from config import StrategyConfig

logger = logging.getLogger(__name__)

class PerformanceAnalytics:
    """
    Advanced performance and risk analytics
    
    This class provides comprehensive performance analysis including:
    - Standard performance metrics (Sharpe, Sortino, Calmar ratios)
    - Risk metrics (VaR, CVaR, Maximum Drawdown)
    - Attribution analysis (factor decomposition)
    - Statistical significance testing
    - Regime analysis and performance breakdown
    """
    
    def __init__(self, results: Dict, config: StrategyConfig):
        """
        Initialize PerformanceAnalytics
        
        Args:
            results: Backtest results dictionary
            config: Strategy configuration
        """
        self.results = results
        self.config = config
        self.metrics = {}
        
    def calculate_comprehensive_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Returns:
            Dictionary with all performance metrics
        """
        logger.info("Calculating comprehensive performance metrics...")
        
        # Get aligned return series
        strategy_returns, benchmark_returns = self._get_aligned_returns()
        
        # Basic performance metrics
        basic_metrics = self._calculate_basic_metrics(strategy_returns, benchmark_returns)
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(strategy_returns)
        
        # Risk-adjusted metrics
        risk_adj_metrics = self._calculate_risk_adjusted_metrics(strategy_returns, benchmark_returns)
        
        # Drawdown analysis
        drawdown_metrics = self._calculate_drawdown_metrics(strategy_returns)
        
        # Higher moment analysis
        moment_metrics = self._calculate_moment_metrics(strategy_returns)
        
        # Factor analysis
        factor_metrics = self._calculate_factor_metrics(strategy_returns, benchmark_returns)
        
        # Portfolio specific metrics
        portfolio_metrics = self._calculate_portfolio_metrics()
        
        # Combine all metrics
        self.metrics = {
            **basic_metrics,
            **risk_metrics,
            **risk_adj_metrics,
            **drawdown_metrics,
            **moment_metrics,
            **factor_metrics,
            **portfolio_metrics
        }
        
        logger.info("Performance metrics calculation complete")
        
        return self.metrics
    
    def _get_aligned_returns(self) -> Tuple[pd.Series, pd.Series]:
        """Get aligned strategy and benchmark returns"""
        
        strategy_returns = self.results['strategy_returns'].dropna()
        benchmark_returns = self.results['benchmark_returns']
        
        # Align series
        common_dates = strategy_returns.index.intersection(benchmark_returns.index)
        strategy_aligned = strategy_returns[common_dates]
        benchmark_aligned = benchmark_returns[common_dates]
        
        return strategy_aligned, benchmark_aligned
    
    def _calculate_basic_metrics(self, strategy_returns: pd.Series, 
                               benchmark_returns: pd.Series) -> Dict:
        """Calculate basic performance metrics"""
        
        # Total returns
        strategy_total = (1 + strategy_returns).prod() - 1
        benchmark_total = (1 + benchmark_returns).prod() - 1
        
        # Annualized returns
        years = len(strategy_returns) / 252
        strategy_annual = (1 + strategy_total) ** (1/years) - 1 if years > 0 else 0
        benchmark_annual = (1 + benchmark_total) ** (1/years) - 1 if years > 0 else 0
        
        # Volatility
        strategy_vol = strategy_returns.std() * np.sqrt(252)
        benchmark_vol = benchmark_returns.std() * np.sqrt(252)
        
        # Excess returns
        excess_returns = strategy_returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        return {
            'Total Return (Strategy)': strategy_total,
            'Total Return (Benchmark)': benchmark_total,
            'Annualized Return (Strategy)': strategy_annual,
            'Annualized Return (Benchmark)': benchmark_annual,
            'Excess Return (Annual)': strategy_annual - benchmark_annual,
            'Annualized Volatility (Strategy)': strategy_vol,
            'Annualized Volatility (Benchmark)': benchmark_vol,
            'Tracking Error': tracking_error,
            'Active Return': excess_returns.mean() * 252
        }
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict:
        """Calculate risk metrics"""
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Conditional Value at Risk (CVaR/Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        
        # Worst day/week/month
        worst_day = returns.min()
        worst_week = returns.rolling(5).sum().min()
        worst_month = returns.rolling(21).sum().min()
        
        # Consecutive losses
        consecutive_losses = self._calculate_consecutive_losses(returns)
        
        return {
            'VaR (95%)': var_95,
            'VaR (99%)': var_99,
            'CVaR (95%)': cvar_95,
            'CVaR (99%)': cvar_99,
            'Downside Deviation': downside_deviation,
            'Worst Day': worst_day,
            'Worst Week': worst_week,
            'Worst Month': worst_month,
            'Max Consecutive Losses': consecutive_losses['max_consecutive'],
            'Avg Consecutive Loss': consecutive_losses['avg_consecutive']
        }
    
    def _calculate_risk_adjusted_metrics(self, strategy_returns: pd.Series,
                                       benchmark_returns: pd.Series) -> Dict:
        """Calculate risk-adjusted performance metrics"""
        
        # Sharpe Ratio
        excess_strategy = strategy_returns.mean() * 252 - self.config.risk_free_rate
        strategy_vol = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = excess_strategy / strategy_vol if strategy_vol > 0 else 0
        
        # Benchmark Sharpe
        excess_benchmark = benchmark_returns.mean() * 252 - self.config.risk_free_rate
        benchmark_vol = benchmark_returns.std() * np.sqrt(252)
        benchmark_sharpe = excess_benchmark / benchmark_vol if benchmark_vol > 0 else 0
        
        # Information Ratio
        excess_returns = strategy_returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = (excess_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
        
        # Sortino Ratio
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else strategy_vol
        sortino_ratio = excess_strategy / downside_std if downside_std > 0 else 0
        
        # Calmar Ratio
        cumulative = (1 + strategy_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        calmar_ratio = (strategy_returns.mean() * 252) / max_drawdown if max_drawdown > 0 else 0
        
        # Treynor Ratio (requires beta calculation)
        beta = self._calculate_beta(strategy_returns, benchmark_returns)
        treynor_ratio = excess_strategy / beta if beta > 0 else 0
        
        # Jensen's Alpha
        jensen_alpha = excess_strategy - beta * (benchmark_returns.mean() * 252 - self.config.risk_free_rate)
        
        return {
            'Sharpe Ratio (Strategy)': sharpe_ratio,
            'Sharpe Ratio (Benchmark)': benchmark_sharpe,
            'Information Ratio': information_ratio,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': calmar_ratio,
            'Treynor Ratio': treynor_ratio,
            'Jensen Alpha': jensen_alpha,
            'Beta': beta
        }
    
    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Dict:
        """Calculate detailed drawdown metrics"""
        
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Drawdown duration analysis
        is_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        
        for is_dd in is_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:  # Add final period if still in drawdown
            drawdown_periods.append(current_period)
        
        # Underwater analysis
        underwater_days = is_drawdown.sum()
        total_days = len(returns)
        underwater_pct = underwater_days / total_days
        
        # Recovery analysis
        max_dd_idx = drawdown.idxmin()
        max_dd_date = max_dd_idx
        
        # Find recovery (if any)
        post_dd = cumulative[cumulative.index > max_dd_date]
        recovery_value = rolling_max.loc[max_dd_date]
        recovery_date = None
        
        for date, value in post_dd.items():
            if value >= recovery_value:
                recovery_date = date
                break
        
        if recovery_date:
            recovery_days = (recovery_date - max_dd_date).days
        else:
            recovery_days = None  # Still recovering
        
        return {
            'Maximum Drawdown': max_drawdown,
            'Max Drawdown Date': max_dd_date.strftime('%Y-%m-%d'),
            'Recovery Date': recovery_date.strftime('%Y-%m-%d') if recovery_date else 'Not recovered',
            'Recovery Days': recovery_days,
            'Avg Drawdown Duration': np.mean(drawdown_periods) if drawdown_periods else 0,
            'Max Drawdown Duration': max(drawdown_periods) if drawdown_periods else 0,
            'Underwater Percentage': underwater_pct,
            'Number of Drawdown Periods': len(drawdown_periods)
        }
    
    def _calculate_moment_metrics(self, returns: pd.Series) -> Dict:
        """Calculate higher moment statistics"""
        
        # Skewness and Kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns, fisher=True)  # Excess kurtosis
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        
        # Win/Loss ratio
        winning_days = (returns > 0).sum()
        losing_days = (returns < 0).sum()
        win_rate = winning_days / len(returns)
        
        # Average win/loss
        avg_win = returns[returns > 0].mean() if winning_days > 0 else 0
        avg_loss = returns[returns < 0].mean() if losing_days > 0 else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        
        # Profit factor
        total_wins = returns[returns > 0].sum()
        total_losses = abs(returns[returns < 0].sum())
        profit_factor = total_wins / total_losses if total_losses > 0 else np.inf
        
        return {
            'Skewness': skewness,
            'Excess Kurtosis': kurtosis,
            'Jarque-Bera p-value': jb_pvalue,
            'Win Rate': win_rate,
            'Average Win': avg_win,
            'Average Loss': avg_loss,
            'Win/Loss Ratio': win_loss_ratio,
            'Profit Factor': profit_factor
        }
    
    def _calculate_factor_metrics(self, strategy_returns: pd.Series,
                                benchmark_returns: pd.Series) -> Dict:
        """Calculate factor exposure and attribution metrics"""
        
        # Beta calculation
        beta = self._calculate_beta(strategy_returns, benchmark_returns)
        
        # R-squared (correlation with benchmark)
        correlation = strategy_returns.corr(benchmark_returns)
        r_squared = correlation ** 2
        
        # Active share (portfolio-specific, simplified)
        # In practice, this would require position-level benchmark comparison
        active_share = 0.5  # Placeholder - would calculate from actual positions
        
        # Up/Down capture ratios
        up_capture, down_capture = self._calculate_capture_ratios(strategy_returns, benchmark_returns)
        
        return {
            'Beta vs Benchmark': beta,
            'Correlation with Benchmark': correlation,
            'R-Squared': r_squared,
            'Active Share': active_share,
            'Up Capture Ratio': up_capture,
            'Down Capture Ratio': down_capture
        }
    
    def _calculate_portfolio_metrics(self) -> Dict:
        """Calculate portfolio-specific metrics"""
        
        if 'turnover' not in self.results:
            return {}
        
        # Turnover analysis
        turnover = self.results['turnover']
        avg_turnover = turnover.mean()
        turnover_vol = turnover.std()
        
        # Position analysis
        positions = self.results['positions']
        avg_positions = (positions > 0).sum(axis=1).mean()
        max_position = positions.max().max()
        concentration = self._calculate_concentration_metrics(positions)
        
        return {
            'Average Turnover': avg_turnover,
            'Turnover Volatility': turnover_vol,
            'Average Active Positions': avg_positions,
            'Maximum Position Size': max_position,
            'Portfolio Concentration (HHI)': concentration['hhi'],
            'Effective Number of Positions': concentration['effective_positions']
        }
    
    def _calculate_beta(self, strategy_returns: pd.Series, 
                      benchmark_returns: pd.Series) -> float:
        """Calculate beta vs benchmark"""
        
        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        return covariance / benchmark_variance if benchmark_variance > 0 else 0
    
    def _calculate_consecutive_losses(self, returns: pd.Series) -> Dict:
        """Calculate consecutive loss statistics"""
        
        losses = returns < 0
        consecutive_counts = []
        current_count = 0
        
        for loss in losses:
            if loss:
                current_count += 1
            else:
                if current_count > 0:
                    consecutive_counts.append(current_count)
                current_count = 0
        
        if current_count > 0:
            consecutive_counts.append(current_count)
        
        return {
            'max_consecutive': max(consecutive_counts) if consecutive_counts else 0,
            'avg_consecutive': np.mean(consecutive_counts) if consecutive_counts else 0
        }
    
    def _calculate_capture_ratios(self, strategy_returns: pd.Series,
                                benchmark_returns: pd.Series) -> Tuple[float, float]:
        """Calculate up and down capture ratios"""
        
        # Identify up and down market periods
        up_markets = benchmark_returns > 0
        down_markets = benchmark_returns < 0
        
        # Calculate capture ratios
        if up_markets.sum() > 0:
            up_strategy = strategy_returns[up_markets].mean()
            up_benchmark = benchmark_returns[up_markets].mean()
            up_capture = up_strategy / up_benchmark if up_benchmark != 0 else 0
        else:
            up_capture = 0
        
        if down_markets.sum() > 0:
            down_strategy = strategy_returns[down_markets].mean()
            down_benchmark = benchmark_returns[down_markets].mean()
            down_capture = down_strategy / down_benchmark if down_benchmark != 0 else 0
        else:
            down_capture = 0
        
        return up_capture, down_capture
    
    def _calculate_concentration_metrics(self, positions: pd.DataFrame) -> Dict:
        """Calculate portfolio concentration metrics"""
        
        # Herfindahl-Hirschman Index
        hhi_daily = (positions ** 2).sum(axis=1)
        avg_hhi = hhi_daily.mean()
        
        # Effective number of positions
        effective_positions = 1 / avg_hhi if avg_hhi > 0 else 0
        
        return {
            'hhi': avg_hhi,
            'effective_positions': effective_positions
        }
    
    def calculate_rolling_metrics(self, window: int = 252) -> Dict:
        """Calculate rolling performance metrics"""
        
        strategy_returns, benchmark_returns = self._get_aligned_returns()
        
        # Rolling Sharpe ratio
        excess_returns = strategy_returns - self.config.risk_free_rate / 252
        rolling_sharpe = (excess_returns.rolling(window).mean() * 252) / \
                        (strategy_returns.rolling(window).std() * np.sqrt(252))
        
        # Rolling correlation
        rolling_corr = strategy_returns.rolling(window).corr(benchmark_returns)
        
        # Rolling beta
        def rolling_beta(window_data):
            if len(window_data) < window // 2:
                return np.nan
            strat = window_data.iloc[:, 0]
            bench = window_data.iloc[:, 1]
            return np.cov(strat, bench)[0, 1] / np.var(bench)
        
        combined = pd.concat([strategy_returns, benchmark_returns], axis=1)
        rolling_beta_series = combined.rolling(window).apply(rolling_beta, raw=False).iloc[:, 0]
        
        # Rolling volatility
        rolling_vol = strategy_returns.rolling(window).std() * np.sqrt(252)
        
        return {
            'rolling_sharpe': rolling_sharpe,
            'rolling_correlation': rolling_corr,
            'rolling_beta': rolling_beta_series,
            'rolling_volatility': rolling_vol
        }
    
    def regime_analysis(self, regimes: Optional[pd.Series] = None) -> Dict:
        """
        Analyze performance across different market regimes
        
        Args:
            regimes: Series indicating market regimes (if None, will create simple bull/bear)
            
        Returns:
            Performance breakdown by regime
        """
        strategy_returns, benchmark_returns = self._get_aligned_returns()
        
        if regimes is None:
            # Simple bull/bear market definition based on benchmark returns
            rolling_return = benchmark_returns.rolling(63).sum()  # 3-month rolling
            regimes = pd.Series('Bull', index=benchmark_returns.index)
            regimes[rolling_return < 0] = 'Bear'
        
        regime_analysis = {}
        
        for regime in regimes.unique():
            regime_mask = regimes == regime
            regime_strategy = strategy_returns[regime_mask]
            regime_benchmark = benchmark_returns[regime_mask]
            
            if len(regime_strategy) > 0:
                regime_analysis[regime] = {
                    'periods': len(regime_strategy),
                    'strategy_return': regime_strategy.mean() * 252,
                    'benchmark_return': regime_benchmark.mean() * 252,
                    'strategy_vol': regime_strategy.std() * np.sqrt(252),
                    'excess_return': (regime_strategy.mean() - regime_benchmark.mean()) * 252,
                    'win_rate': (regime_strategy > 0).mean(),
                    'sharpe': (regime_strategy.mean() * 252 - self.config.risk_free_rate) / \
                             (regime_strategy.std() * np.sqrt(252)) if regime_strategy.std() > 0 else 0
                }
        
        return regime_analysis
    
    def statistical_significance_tests(self) -> Dict:
        """Run statistical significance tests on strategy performance"""
        
        strategy_returns, benchmark_returns = self._get_aligned_returns()
        excess_returns = strategy_returns - benchmark_returns
        
        # t-test for excess returns
        t_stat, t_pvalue = stats.ttest_1samp(excess_returns, 0)
        
        # Sharpe ratio significance (Jobson-Korkie test approximation)
        n = len(strategy_returns)
        sharpe_strategy = (strategy_returns.mean() * 252 - self.config.risk_free_rate) / \
                         (strategy_returns.std() * np.sqrt(252))
        sharpe_se = np.sqrt((1 + 0.5 * sharpe_strategy**2) / n)
        sharpe_t_stat = sharpe_strategy / sharpe_se
        sharpe_pvalue = 2 * (1 - stats.norm.cdf(abs(sharpe_t_stat)))
        
        # Information ratio significance
        ir = excess_returns.mean() * 252 / (excess_returns.std() * np.sqrt(252))
        ir_se = 1 / np.sqrt(n)
        ir_t_stat = ir / ir_se
        ir_pvalue = 2 * (1 - stats.norm.cdf(abs(ir_t_stat)))
        
        return {
            'excess_return_t_stat': t_stat,
            'excess_return_p_value': t_pvalue,
            'sharpe_t_stat': sharpe_t_stat,
            'sharpe_p_value': sharpe_pvalue,
            'information_ratio_t_stat': ir_t_stat,
            'information_ratio_p_value': ir_pvalue,
            'significant_at_5pct': t_pvalue < 0.05
        }
    
    def export_analytics(self, file_path: str):
        """Export comprehensive analytics to Excel file"""
        
        with pd.ExcelWriter(f"{file_path}_performance_analytics.xlsx") as writer:
            # Main metrics
            if self.metrics:
                metrics_df = pd.DataFrame.from_dict(self.metrics, orient='index', columns=['Value'])
                metrics_df.to_excel(writer, sheet_name='Performance_Metrics')
            
            # Rolling metrics
            rolling_metrics = self.calculate_rolling_metrics()
            for metric_name, metric_data in rolling_metrics.items():
                metric_data.to_excel(writer, sheet_name=f'Rolling_{metric_name}')
            
            # Regime analysis
            regime_results = self.regime_analysis()
            regime_df = pd.DataFrame.from_dict(regime_results, orient='index')
            regime_df.to_excel(writer, sheet_name='Regime_Analysis')
            
            # Statistical tests
            sig_tests = self.statistical_significance_tests()
            sig_df = pd.DataFrame.from_dict(sig_tests, orient='index', columns=['Value'])
            sig_df.to_excel(writer, sheet_name='Statistical_Tests')
        
        logger.info(f"Performance analytics exported to {file_path}_performance_analytics.xlsx")