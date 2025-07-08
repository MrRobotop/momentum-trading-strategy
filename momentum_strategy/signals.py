"""
Signal Generation Module
=======================

Implements momentum signal generation and ranking algorithms.
Features multiple momentum measures and risk-adjusted signal combination.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging
from config import StrategyConfig

logger = logging.getLogger(__name__)

class MomentumSignals:
    """
    Generates momentum signals and rankings
    
    This class implements various momentum measures including:
    - Multi-timeframe price momentum
    - Risk-adjusted momentum (Sharpe ratios)
    - Volatility-adjusted momentum
    - Cross-sectional ranking algorithms
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize MomentumSignals
        
        Args:
            config: Strategy configuration object
        """
        self.config = config
        self.momentum_scores = None
        self.individual_signals = {}
        
    def calculate_momentum_scores(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive momentum scores using multiple measures
        
        Args:
            prices: DataFrame with asset prices (index: dates, columns: assets)
            
        Returns:
            DataFrame with momentum scores for each asset and date
        """
        logger.info("Calculating momentum scores...")
        
        # Calculate individual momentum components
        signals = {}
        
        # 1. Price momentum (multiple timeframes)
        signals['mom_3m'] = self._calculate_price_momentum(prices, self.config.short_mom_period)
        signals['mom_6m'] = self._calculate_price_momentum(prices, self.config.medium_mom_period)  
        signals['mom_12m'] = self._calculate_price_momentum(prices, self.config.long_mom_period)
        
        # 2. Risk-adjusted momentum (Sharpe ratios)
        signals['sharpe_12m'] = self._calculate_sharpe_momentum(prices, self.config.long_mom_period)
        
        # 3. Volatility-adjusted momentum
        signals['vol_adj_mom'] = self._calculate_volatility_adjusted_momentum(
            prices, self.config.long_mom_period
        )
        
        # Store individual signals for analysis
        self.individual_signals = signals
        
        # 4. Combine signals using configured weights
        momentum_score = self._combine_signals(signals, self.config.momentum_weights)
        
        # 5. Apply signal smoothing if configured
        if hasattr(self.config, 'signal_smoothing') and self.config.signal_smoothing:
            momentum_score = self._smooth_signals(momentum_score)
        
        # Store results
        self.momentum_scores = momentum_score.dropna()
        
        logger.info(f"Momentum scores calculated for {len(self.momentum_scores.columns)} assets")
        logger.info(f"Score range: {self.momentum_scores.min().min():.3f} to {self.momentum_scores.max().max():.3f}")
        
        return self.momentum_scores
    
    def _calculate_price_momentum(self, prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """
        Calculate simple price momentum over specified lookback period
        
        Args:
            prices: Price data
            lookback: Lookback period in days
            
        Returns:
            DataFrame with price momentum values
        """
        # Calculate momentum as percentage change over lookback period
        momentum = prices.pct_change(lookback)
        
        # Handle edge cases
        momentum = momentum.replace([np.inf, -np.inf], np.nan)
        
        return momentum
    
    def _calculate_sharpe_momentum(self, prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """
        Calculate risk-adjusted momentum using rolling Sharpe ratios
        
        Args:
            prices: Price data
            lookback: Lookback period in days
            
        Returns:
            DataFrame with Sharpe momentum values
        """
        returns = prices.pct_change()
        
        def rolling_sharpe(ret_series, window=lookback):
            """Calculate rolling Sharpe ratio"""
            rolling_mean = ret_series.rolling(window).mean() * 252
            rolling_std = ret_series.rolling(window).std() * np.sqrt(252)
            
            # Subtract risk-free rate
            excess_return = rolling_mean - self.config.risk_free_rate
            
            # Calculate Sharpe ratio, handling division by zero
            sharpe = np.where(rolling_std > 0, excess_return / rolling_std, 0)
            
            return pd.Series(sharpe, index=ret_series.index)
        
        # Apply rolling Sharpe calculation to each asset
        sharpe_momentum = returns.apply(lambda col: rolling_sharpe(col, lookback))
        
        return sharpe_momentum
    
    def _calculate_volatility_adjusted_momentum(self, prices: pd.DataFrame, 
                                              lookback: int) -> pd.DataFrame:
        """
        Calculate volatility-adjusted momentum
        
        This adjusts raw momentum by the volatility of returns to account for
        risk differences across assets.
        
        Args:
            prices: Price data
            lookback: Lookback period in days
            
        Returns:
            DataFrame with volatility-adjusted momentum values
        """
        returns = prices.pct_change()
        
        # Calculate raw momentum
        raw_momentum = prices.pct_change(lookback)
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(lookback).std() * np.sqrt(252)
        
        # Adjust momentum by volatility (higher vol gets penalized)
        vol_adj_momentum = raw_momentum / (rolling_vol + 1e-8)  # Add small epsilon to avoid division by zero
        
        # Handle edge cases
        vol_adj_momentum = vol_adj_momentum.replace([np.inf, -np.inf], np.nan)
        
        return vol_adj_momentum
    
    def _combine_signals(self, signals: Dict[str, pd.DataFrame], 
                        weights: Dict[str, float]) -> pd.DataFrame:
        """
        Combine individual momentum signals using specified weights
        
        Args:
            signals: Dictionary of signal DataFrames
            weights: Dictionary of signal weights
            
        Returns:
            Combined momentum score DataFrame
        """
        # Standardize signals before combining (z-score normalization)
        standardized_signals = {}
        
        for signal_name, signal_data in signals.items():
            if signal_name in weights:
                # Calculate rolling z-scores for each signal
                standardized_signals[signal_name] = self._standardize_signal(signal_data)
        
        # Combine standardized signals using weights
        combined_score = None
        
        for signal_name, weight in weights.items():
            if signal_name in standardized_signals:
                weighted_signal = standardized_signals[signal_name] * weight
                
                if combined_score is None:
                    combined_score = weighted_signal
                else:
                    combined_score = combined_score.add(weighted_signal, fill_value=0)
        
        return combined_score
    
    def _standardize_signal(self, signal: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
        """
        Standardize signal using rolling z-score normalization
        
        Args:
            signal: Raw signal data
            lookback: Period for calculating rolling statistics
            
        Returns:
            Standardized signal data
        """
        # Calculate rolling mean and standard deviation
        rolling_mean = signal.rolling(lookback, min_periods=60).mean()
        rolling_std = signal.rolling(lookback, min_periods=60).std()
        
        # Calculate z-scores
        z_scores = (signal - rolling_mean) / (rolling_std + 1e-8)
        
        # Cap extreme z-scores to reduce impact of outliers
        z_scores = z_scores.clip(lower=-3, upper=3)
        
        return z_scores
    
    def _smooth_signals(self, signals: pd.DataFrame, smoothing_window: int = 5) -> pd.DataFrame:
        """
        Apply signal smoothing to reduce noise
        
        Args:
            signals: Raw signals
            smoothing_window: Window for moving average smoothing
            
        Returns:
            Smoothed signals
        """
        return signals.rolling(smoothing_window, min_periods=1).mean()
    
    def generate_rankings(self, momentum_scores: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate cross-sectional rankings from momentum scores
        
        Args:
            momentum_scores: Momentum scores (uses stored scores if None)
            
        Returns:
            DataFrame with rankings (1 = highest momentum, higher numbers = lower momentum)
        """
        if momentum_scores is None:
            momentum_scores = self.momentum_scores
            
        if momentum_scores is None:
            raise ValueError("No momentum scores available. Run calculate_momentum_scores first.")
        
        logger.info("Generating cross-sectional rankings...")
        
        # Rank assets by momentum score (1 = highest momentum)
        rankings = momentum_scores.rank(axis=1, ascending=False, method='min')
        
        # Store rankings for analysis
        self.rankings = rankings
        
        return rankings
    
    def get_top_assets(self, date: pd.Timestamp, n_assets: int) -> pd.Series:
        """
        Get top N assets by momentum score for a specific date
        
        Args:
            date: Date to get rankings for
            n_assets: Number of top assets to return
            
        Returns:
            Series with top N assets and their momentum scores
        """
        if self.momentum_scores is None:
            raise ValueError("No momentum scores calculated")
        
        if date not in self.momentum_scores.index:
            raise ValueError(f"Date {date} not found in momentum scores")
        
        day_scores = self.momentum_scores.loc[date].dropna()
        top_assets = day_scores.nlargest(n_assets)
        
        return top_assets
    
    def analyze_signal_stability(self, lookback_days: int = 63) -> Dict:
        """
        Analyze stability and persistence of momentum signals
        
        Args:
            lookback_days: Period to analyze signal stability
            
        Returns:
            Dictionary with stability metrics
        """
        if self.momentum_scores is None:
            raise ValueError("No momentum scores to analyze")
        
        # Calculate rank correlation over time (Spearman)
        rank_correlations = []
        
        for i in range(lookback_days, len(self.momentum_scores)):
            current_ranks = self.momentum_scores.iloc[i].rank()
            previous_ranks = self.momentum_scores.iloc[i-lookback_days].rank()
            
            # Calculate Spearman correlation
            common_assets = current_ranks.index.intersection(previous_ranks.index)
            if len(common_assets) > 5:  # Need minimum assets for meaningful correlation
                corr = current_ranks[common_assets].corr(previous_ranks[common_assets], method='spearman')
                rank_correlations.append(corr)
        
        # Signal turnover analysis
        if hasattr(self, 'rankings'):
            turnover_metrics = self._calculate_signal_turnover()
        else:
            turnover_metrics = {}
        
        stability_metrics = {
            'avg_rank_correlation': np.mean(rank_correlations) if rank_correlations else np.nan,
            'rank_correlation_std': np.std(rank_correlations) if rank_correlations else np.nan,
            'signal_persistence': len([c for c in rank_correlations if c > 0.5]) / len(rank_correlations) if rank_correlations else 0,
            **turnover_metrics
        }
        
        return stability_metrics
    
    def _calculate_signal_turnover(self) -> Dict:
        """Calculate turnover metrics for momentum signals"""
        
        if not hasattr(self, 'rankings'):
            return {}
        
        # Calculate how often top decile assets change
        top_decile_changes = []
        
        for i in range(1, len(self.rankings)):
            current_top = set(self.rankings.iloc[i].nsmallest(len(self.rankings.columns)//10).index)
            previous_top = set(self.rankings.iloc[i-1].nsmallest(len(self.rankings.columns)//10).index)
            
            # Calculate percentage of assets that changed
            intersection = len(current_top.intersection(previous_top))
            turnover = 1 - (intersection / len(current_top)) if len(current_top) > 0 else 1
            top_decile_changes.append(turnover)
        
        return {
            'avg_top_decile_turnover': np.mean(top_decile_changes) if top_decile_changes else np.nan,
            'max_top_decile_turnover': np.max(top_decile_changes) if top_decile_changes else np.nan
        }
    
    def get_signal_attribution(self, date: pd.Timestamp, asset: str) -> Dict:
        """
        Get signal attribution for a specific asset and date
        
        Args:
            date: Date to analyze
            asset: Asset symbol
            
        Returns:
            Dictionary with signal component contributions
        """
        if not self.individual_signals:
            raise ValueError("Individual signals not available")
        
        attribution = {}
        
        for signal_name, signal_data in self.individual_signals.items():
            if date in signal_data.index and asset in signal_data.columns:
                raw_value = signal_data.loc[date, asset]
                weight = self.config.momentum_weights.get(signal_name, 0)
                contribution = raw_value * weight
                
                attribution[signal_name] = {
                    'raw_value': raw_value,
                    'weight': weight,
                    'contribution': contribution
                }
        
        return attribution
    
    def export_signals(self, file_path: str):
        """Export signals and rankings to file"""
        
        with pd.ExcelWriter(f"{file_path}_momentum_analysis.xlsx") as writer:
            # Export combined momentum scores
            if self.momentum_scores is not None:
                self.momentum_scores.to_excel(writer, sheet_name='Momentum_Scores')
            
            # Export individual signals
            for signal_name, signal_data in self.individual_signals.items():
                signal_data.to_excel(writer, sheet_name=f'Signal_{signal_name}')
            
            # Export rankings if available
            if hasattr(self, 'rankings'):
                self.rankings.to_excel(writer, sheet_name='Rankings')
        
        logger.info(f"Signals exported to {file_path}_momentum_analysis.xlsx")