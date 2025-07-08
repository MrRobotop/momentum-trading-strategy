"""
Backtesting Engine Module
========================

Comprehensive backtesting framework for momentum trading strategies.
Features robust testing, performance tracking, and detailed analytics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
import logging
from datetime import datetime, timedelta
from config import StrategyConfig
from signals import MomentumSignals
from portfolio import PortfolioConstructor

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Comprehensive backtesting engine for momentum strategies
    
    This class orchestrates the complete backtesting process including:
    - Signal generation and portfolio construction
    - Performance calculation and tracking
    - Risk management and monitoring
    - Detailed results storage and analysis
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize BacktestEngine
        
        Args:
            config: Strategy configuration object
        """
        self.config = config
        self.results = {}
        self.performance_tracker = {}
        self.risk_tracker = {}
        
    def run_backtest(self, prices: pd.DataFrame, 
                    returns: pd.DataFrame,
                    benchmark_returns: Optional[pd.Series] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> Dict:
        """
        Run complete backtest of the momentum strategy
        
        Args:
            prices: Asset price data
            returns: Asset return data
            benchmark_returns: Benchmark returns (optional, will create equal-weight if None)
            start_date: Backtest start date (optional)
            end_date: Backtest end date (optional)
            
        Returns:
            Dictionary containing all backtest results
        """
        logger.info("Starting momentum strategy backtest...")
        
        # Filter data to backtest period if specified
        if start_date or end_date:
            prices, returns = self._filter_date_range(prices, returns, start_date, end_date)
        
        # Create benchmark if not provided
        if benchmark_returns is None:
            benchmark_returns = returns.mean(axis=1)
            logger.info("Using equal-weight benchmark")
        
        # Initialize strategy components
        signal_generator = MomentumSignals(self.config)
        portfolio_constructor = PortfolioConstructor(self.config)
        
        # Step 1: Generate momentum signals
        logger.info("Step 1: Generating momentum signals...")
        momentum_scores = signal_generator.calculate_momentum_scores(prices)
        rankings = signal_generator.generate_rankings(momentum_scores)
        
        # Step 2: Construct portfolio positions
        logger.info("Step 2: Constructing portfolio positions...")
        positions = portfolio_constructor.calculate_positions(rankings, returns)
        
        # Step 3: Calculate strategy returns
        logger.info("Step 3: Calculating strategy performance...")
        strategy_returns = self._calculate_strategy_returns(positions, returns)
        
        # Step 4: Apply transaction costs
        logger.info("Step 4: Applying transaction costs...")
        net_returns, turnover = portfolio_constructor.apply_transaction_costs(
            positions, strategy_returns
        )
        
        # Step 5: Track performance and risk
        self._track_performance(net_returns, benchmark_returns, positions)
        self._track_risk_metrics(net_returns, positions, returns)
        
        # Step 6: Compile results
        self.results = {
            'strategy_returns': net_returns,
            'gross_returns': strategy_returns,
            'benchmark_returns': benchmark_returns,
            'positions': positions,
            'momentum_scores': momentum_scores,
            'rankings': rankings,
            'turnover': turnover,
            'prices': prices,
            'returns': returns,
            'performance_tracker': self.performance_tracker,
            'risk_tracker': self.risk_tracker,
            'config': self.config
        }
        
        logger.info("Backtest completed successfully")
        
        return self.results
    
    def _filter_date_range(self, prices: pd.DataFrame, returns: pd.DataFrame,
                          start_date: Optional[str], end_date: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filter data to specified date range"""
        
        if start_date:
            start_idx = prices.index >= start_date
            prices = prices[start_idx]
            returns = returns[start_idx]
        
        if end_date:
            end_idx = prices.index <= end_date
            prices = prices[end_idx]
            returns = returns[end_idx]
        
        return prices, returns
    
    def _calculate_strategy_returns(self, positions: pd.DataFrame, 
                                  returns: pd.DataFrame) -> pd.Series:
        """
        Calculate gross strategy returns before transaction costs
        
        Args:
            positions: Portfolio positions
            returns: Asset returns
            
        Returns:
            Series with strategy returns
        """
        # Align positions and returns
        common_dates = positions.index.intersection(returns.index)
        common_assets = positions.columns.intersection(returns.columns)
        
        aligned_positions = positions.loc[common_dates, common_assets]
        aligned_returns = returns.loc[common_dates, common_assets]
        
        # Calculate portfolio returns (use previous day's positions)
        lagged_positions = aligned_positions.shift(1)
        strategy_returns = (lagged_positions * aligned_returns).sum(axis=1)
        
        return strategy_returns.dropna()
    
    def _track_performance(self, strategy_returns: pd.Series, 
                          benchmark_returns: pd.Series,
                          positions: pd.DataFrame):
        """Track key performance metrics during backtest"""
        
        # Align series
        common_dates = strategy_returns.index.intersection(benchmark_returns.index)
        strategy_aligned = strategy_returns[common_dates]
        benchmark_aligned = benchmark_returns[common_dates]
        
        # Calculate cumulative returns
        strategy_cumret = (1 + strategy_aligned).cumprod()
        benchmark_cumret = (1 + benchmark_aligned).cumprod()
        
        # Rolling performance metrics
        rolling_sharpe = self._calculate_rolling_sharpe(strategy_aligned)
        rolling_alpha = self._calculate_rolling_alpha(strategy_aligned, benchmark_aligned)
        rolling_beta = self._calculate_rolling_beta(strategy_aligned, benchmark_aligned)
        
        # Drawdown analysis
        drawdown = self._calculate_drawdown(strategy_cumret)
        
        self.performance_tracker = {
            'cumulative_returns': strategy_cumret,
            'benchmark_cumulative': benchmark_cumret,
            'rolling_sharpe': rolling_sharpe,
            'rolling_alpha': rolling_alpha,
            'rolling_beta': rolling_beta,
            'drawdown': drawdown,
            'excess_returns': strategy_aligned - benchmark_aligned
        }
    
    def _track_risk_metrics(self, strategy_returns: pd.Series,
                           positions: pd.DataFrame,
                           asset_returns: pd.DataFrame):
        """Track risk metrics during backtest"""
        
        # Portfolio volatility
        portfolio_vol = strategy_returns.rolling(252).std() * np.sqrt(252)
        
        # Value at Risk (VaR)
        var_95 = strategy_returns.rolling(252).quantile(0.05)
        
        # Expected Shortfall (CVaR)
        def rolling_cvar(returns, window=252, alpha=0.05):
            return returns.rolling(window).apply(
                lambda x: x[x <= x.quantile(alpha)].mean()
            )
        
        cvar_95 = rolling_cvar(strategy_returns)
        
        # Maximum Drawdown
        cumret = (1 + strategy_returns).cumprod()
        rolling_max = cumret.expanding().max()
        max_dd = ((cumret - rolling_max) / rolling_max).rolling(252).min()
        
        # Position concentration risk
        concentration = self._calculate_position_concentration(positions)
        
        self.risk_tracker = {
            'portfolio_volatility': portfolio_vol,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': max_dd,
            'position_concentration': concentration
        }
    
    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """Calculate rolling Sharpe ratio"""
        
        excess_returns = returns - self.config.risk_free_rate / 252
        rolling_mean = excess_returns.rolling(window).mean() * 252
        rolling_std = returns.rolling(window).std() * np.sqrt(252)
        
        return rolling_mean / rolling_std
    
    def _calculate_rolling_alpha(self, strategy_returns: pd.Series,
                               benchmark_returns: pd.Series, window: int = 252) -> pd.Series:
        """Calculate rolling alpha vs benchmark"""
        
        # Simple rolling alpha calculation
        excess_returns = strategy_returns - benchmark_returns
        rolling_alpha = excess_returns.rolling(window).mean() * 252

        return rolling_alpha
    
    def _calculate_rolling_beta(self, strategy_returns: pd.Series,
                              benchmark_returns: pd.Series, window: int = 252) -> pd.Series:
        """Calculate rolling beta vs benchmark"""
        
        # Simple rolling beta calculation using correlation and volatility
        correlation = strategy_returns.rolling(window).corr(benchmark_returns)
        strategy_vol = strategy_returns.rolling(window).std()
        benchmark_vol = benchmark_returns.rolling(window).std()

        rolling_beta = correlation * (strategy_vol / benchmark_vol)
        rolling_beta = rolling_beta.fillna(1.0)  # Default beta of 1

        return rolling_beta
    
    def _calculate_drawdown(self, cumulative_returns: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        
        return drawdown
    
    def _calculate_position_concentration(self, positions: pd.DataFrame) -> pd.Series:
        """Calculate position concentration over time using Herfindahl index"""
        
        # Herfindahl index: sum of squared weights
        concentration = (positions ** 2).sum(axis=1)
        
        return concentration
    
    def run_walk_forward_analysis(self, prices: pd.DataFrame, returns: pd.DataFrame,
                                 training_period: int = 252,
                                 rebalance_freq: int = 63) -> Dict:
        """
        Run walk-forward analysis to test strategy robustness
        
        Args:
            prices: Price data
            returns: Return data
            training_period: Days of data for training
            rebalance_freq: How often to retrain (days)
            
        Returns:
            Walk-forward analysis results
        """
        logger.info("Running walk-forward analysis...")
        
        walk_forward_results = []
        start_idx = training_period
        
        while start_idx < len(prices) - rebalance_freq:
            # Define training and testing periods
            train_start = start_idx - training_period
            train_end = start_idx
            test_start = start_idx
            test_end = min(start_idx + rebalance_freq, len(prices))
            
            # Get training and testing data
            train_prices = prices.iloc[train_start:train_end]
            train_returns = returns.iloc[train_start:train_end]
            test_prices = prices.iloc[test_start:test_end]
            test_returns = returns.iloc[test_start:test_end]
            
            try:
                # Run backtest on training period (for parameter validation)
                train_results = self.run_backtest(train_prices, train_returns)
                
                # Apply strategy to test period
                signal_gen = MomentumSignals(self.config)
                momentum_scores = signal_gen.calculate_momentum_scores(
                    pd.concat([train_prices, test_prices])
                )
                rankings = signal_gen.generate_rankings(momentum_scores)
                
                # Get test period performance
                test_rankings = rankings.loc[test_prices.index]
                portfolio_constructor = PortfolioConstructor(self.config)
                test_positions = portfolio_constructor.calculate_positions(
                    test_rankings, test_returns
                )
                
                test_strategy_returns = self._calculate_strategy_returns(
                    test_positions, test_returns
                )
                
                # Store results
                period_result = {
                    'train_period': (train_prices.index[0], train_prices.index[-1]),
                    'test_period': (test_prices.index[0], test_prices.index[-1]),
                    'train_sharpe': self._calculate_period_sharpe(train_results['strategy_returns']),
                    'test_sharpe': self._calculate_period_sharpe(test_strategy_returns),
                    'test_returns': test_strategy_returns.sum(),
                    'test_volatility': test_strategy_returns.std() * np.sqrt(252)
                }
                
                walk_forward_results.append(period_result)
                
            except Exception as e:
                logger.warning(f"Walk-forward period failed: {e}")
                continue
            
            start_idx += rebalance_freq
        
        # Compile walk-forward statistics
        wf_analysis = {
            'period_results': walk_forward_results,
            'avg_oos_sharpe': np.mean([r['test_sharpe'] for r in walk_forward_results]),
            'sharpe_consistency': np.std([r['test_sharpe'] for r in walk_forward_results]),
            'hit_rate': len([r for r in walk_forward_results if r['test_returns'] > 0]) / len(walk_forward_results)
        }
        
        logger.info(f"Walk-forward analysis complete. Average OOS Sharpe: {wf_analysis['avg_oos_sharpe']:.3f}")
        
        return wf_analysis
    
    def _calculate_period_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio for a period"""
        
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_return = returns.mean() * 252 - self.config.risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        
        return excess_return / volatility
    
    def run_monte_carlo_simulation(self, returns: pd.DataFrame, 
                                  n_simulations: int = 1000,
                                  simulation_length: int = 252) -> Dict:
        """
        Run Monte Carlo simulation to assess strategy robustness
        
        Args:
            returns: Historical returns
            n_simulations: Number of simulation runs
            simulation_length: Length of each simulation (days)
            
        Returns:
            Monte Carlo simulation results
        """
        logger.info(f"Running Monte Carlo simulation ({n_simulations} runs)...")
        
        simulation_results = []
        
        for sim in range(n_simulations):
            # Bootstrap returns
            simulated_returns = self._bootstrap_returns(returns, simulation_length)
            
            # Create simulated prices
            simulated_prices = (1 + simulated_returns).cumprod()
            
            try:
                # Run backtest on simulated data
                sim_results = self.run_backtest(simulated_prices, simulated_returns)
                
                # Extract key metrics
                final_return = sim_results['strategy_returns'].sum()
                volatility = sim_results['strategy_returns'].std() * np.sqrt(252)
                max_dd = sim_results['performance_tracker']['drawdown'].min()
                sharpe = self._calculate_period_sharpe(sim_results['strategy_returns'])
                
                simulation_results.append({
                    'total_return': final_return,
                    'volatility': volatility,
                    'max_drawdown': max_dd,
                    'sharpe_ratio': sharpe
                })
                
            except Exception as e:
                logger.warning(f"Simulation {sim} failed: {e}")
                continue
        
        # Compile statistics
        mc_stats = {
            'n_successful_sims': len(simulation_results),
            'avg_return': np.mean([r['total_return'] for r in simulation_results]),
            'return_std': np.std([r['total_return'] for r in simulation_results]),
            'avg_sharpe': np.mean([r['sharpe_ratio'] for r in simulation_results]),
            'sharpe_std': np.std([r['sharpe_ratio'] for r in simulation_results]),
            'win_rate': len([r for r in simulation_results if r['total_return'] > 0]) / len(simulation_results),
            'percentiles': {
                'return_5th': np.percentile([r['total_return'] for r in simulation_results], 5),
                'return_95th': np.percentile([r['total_return'] for r in simulation_results], 95),
                'sharpe_5th': np.percentile([r['sharpe_ratio'] for r in simulation_results], 5),
                'sharpe_95th': np.percentile([r['sharpe_ratio'] for r in simulation_results], 95)
            }
        }
        
        logger.info(f"Monte Carlo complete. Average return: {mc_stats['avg_return']:.2%}, Win rate: {mc_stats['win_rate']:.1%}")
        
        return mc_stats
    
    def _bootstrap_returns(self, returns: pd.DataFrame, length: int) -> pd.DataFrame:
        """Bootstrap returns for Monte Carlo simulation"""
        
        # Randomly sample returns with replacement
        n_assets = len(returns.columns)
        n_days = len(returns)
        
        # Sample random indices
        random_indices = np.random.choice(n_days, size=length, replace=True)
        
        # Create bootstrap sample
        bootstrapped = returns.iloc[random_indices].reset_index(drop=True)
        
        return bootstrapped
    
    def get_backtest_summary(self) -> Dict:
        """Get comprehensive backtest summary"""
        
        if not self.results:
            return {'status': 'No backtest results available'}
        
        strategy_returns = self.results['strategy_returns']
        benchmark_returns = self.results['benchmark_returns']
        
        # Align returns
        common_dates = strategy_returns.index.intersection(benchmark_returns.index)
        strat_aligned = strategy_returns[common_dates]
        bench_aligned = benchmark_returns[common_dates]
        
        summary = {
            'backtest_period': {
                'start': strat_aligned.index[0].strftime('%Y-%m-%d'),
                'end': strat_aligned.index[-1].strftime('%Y-%m-%d'),
                'total_days': len(strat_aligned)
            },
            'returns': {
                'strategy_total': (1 + strat_aligned).prod() - 1,
                'benchmark_total': (1 + bench_aligned).prod() - 1,
                'strategy_annual': (1 + strat_aligned).prod() ** (252 / len(strat_aligned)) - 1,
                'benchmark_annual': (1 + bench_aligned).prod() ** (252 / len(bench_aligned)) - 1,
                'excess_annual': ((1 + strat_aligned).prod() ** (252 / len(strat_aligned)) - 1) - 
                                ((1 + bench_aligned).prod() ** (252 / len(bench_aligned)) - 1)
            },
            'risk_metrics': {
                'strategy_volatility': strat_aligned.std() * np.sqrt(252),
                'benchmark_volatility': bench_aligned.std() * np.sqrt(252),
                'sharpe_ratio': self._calculate_period_sharpe(strat_aligned),
                'max_drawdown': self.results['performance_tracker']['drawdown'].min()
            },
            'portfolio_metrics': {
                'avg_turnover': self.results['turnover'].mean(),
                'avg_positions': (self.results['positions'] > 0).sum(axis=1).mean()
            }
        }
        
        return summary