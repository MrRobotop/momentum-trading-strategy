"""
Portfolio Construction Module
============================

Implements portfolio construction and risk management algorithms.
Features dynamic position sizing, risk controls, and transaction cost optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import logging
from scipy.optimize import minimize
from config import StrategyConfig

logger = logging.getLogger(__name__)

class PortfolioConstructor:
    """
    Constructs and manages portfolio positions with risk controls
    
    This class handles:
    - Position sizing based on momentum rankings
    - Risk-based portfolio optimization
    - Position size constraints and limits
    - Transaction cost optimization
    - Sector and concentration constraints
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize PortfolioConstructor
        
        Args:
            config: Strategy configuration object
        """
        self.config = config
        self.positions = None
        self.previous_positions = None
        self.turnover_history = []
        
    def calculate_positions(self, rankings: pd.DataFrame, 
                          returns: pd.DataFrame,
                          volumes: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate portfolio positions based on momentum rankings
        
        Args:
            rankings: Asset rankings from momentum analysis
            returns: Historical returns data
            volumes: Trading volume data (optional)
            
        Returns:
            DataFrame with portfolio positions (weights)
        """
        logger.info("Calculating portfolio positions...")
        
        positions = pd.DataFrame(0.0, index=rankings.index, columns=rankings.columns)
        
        for date in rankings.index:
            try:
                # Get daily positions
                daily_positions = self._calculate_daily_positions(
                    date, rankings, returns, volumes
                )
                positions.loc[date] = daily_positions
                
            except Exception as e:
                logger.warning(f"Error calculating positions for {date}: {e}")
                # Use previous day's positions if available
                if date in positions.index and len(positions.loc[:date]) > 1:
                    prev_date = positions.loc[:date].index[-2]
                    positions.loc[date] = positions.loc[prev_date]
        
        # Store positions
        self.positions = positions
        
        # Calculate turnover
        self._calculate_turnover()
        
        logger.info("Portfolio positions calculated successfully")
        
        return positions
    
    def _calculate_daily_positions(self, date: pd.Timestamp, 
                                 rankings: pd.DataFrame, 
                                 returns: pd.DataFrame,
                                 volumes: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Calculate positions for a single date
        
        Args:
            date: Date to calculate positions for
            rankings: Asset rankings
            returns: Returns data
            volumes: Volume data
            
        Returns:
            Series with position weights for each asset
        """
        # Get rankings for this date
        if date not in rankings.index:
            return pd.Series(0.0, index=rankings.columns)
        
        day_rankings = rankings.loc[date].dropna()
        
        # Select top N assets based on rankings
        top_assets = day_rankings.nsmallest(self.config.top_n_assets).index.tolist()
        
        if len(top_assets) == 0:
            return pd.Series(0.0, index=rankings.columns)
        
        # Calculate position weights
        if self.config.enable_risk_targeting:
            weights = self._calculate_risk_weighted_positions(
                date, top_assets, returns, volumes
            )
        else:
            weights = self._calculate_equal_weighted_positions(top_assets)
        
        # Apply position constraints
        weights = self._apply_position_constraints(weights, top_assets)
        
        # Apply sector constraints if enabled
        if self.config.enable_sector_constraints:
            weights = self._apply_sector_constraints(weights, top_assets)
        
        # Create full position series
        full_positions = pd.Series(0.0, index=rankings.columns)
        full_positions[weights.index] = weights.values
        
        return full_positions
    
    def _calculate_risk_weighted_positions(self, date: pd.Timestamp,
                                         top_assets: List[str],
                                         returns: pd.DataFrame,
                                         volumes: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Calculate risk-weighted position sizes using inverse volatility weighting
        
        Args:
            date: Current date
            top_assets: List of selected assets
            returns: Returns data
            volumes: Volume data
            
        Returns:
            Series with risk-weighted position sizes
        """
        # Get historical returns up to current date
        historical_returns = returns.loc[:date, top_assets]
        
        # Use last 63 days (quarter) for volatility estimation
        lookback_returns = historical_returns.tail(63)
        
        if len(lookback_returns) < 30:  # Need minimum history
            return self._calculate_equal_weighted_positions(top_assets)
        
        # Calculate asset volatilities
        volatilities = lookback_returns.std() * np.sqrt(252)
        volatilities = volatilities.fillna(volatilities.median())
        
        # Handle zero volatilities
        volatilities = volatilities.replace(0, volatilities[volatilities > 0].min())
        
        # Calculate inverse volatility weights
        inv_vol_weights = 1 / volatilities
        inv_vol_weights = inv_vol_weights / inv_vol_weights.sum()
        
        # Apply volatility targeting if configured
        if hasattr(self.config, 'volatility_target'):
            inv_vol_weights = self._apply_volatility_targeting(
                inv_vol_weights, lookback_returns
            )
        
        return inv_vol_weights
    
    def _calculate_equal_weighted_positions(self, top_assets: List[str]) -> pd.Series:
        """Calculate equal-weighted positions"""
        
        equal_weight = 1.0 / len(top_assets)
        weights = pd.Series(equal_weight, index=top_assets)
        
        return weights
    
    def _apply_volatility_targeting(self, weights: pd.Series, 
                                  returns: pd.DataFrame) -> pd.Series:
        """
        Apply portfolio-level volatility targeting
        
        Args:
            weights: Asset weights
            returns: Historical returns
            
        Returns:
            Volatility-targeted weights
        """
        # Calculate portfolio volatility
        cov_matrix = returns.cov() * 252  # Annualize
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        
        # Scale weights to achieve target volatility
        if portfolio_vol > 0:
            vol_scalar = self.config.volatility_target / portfolio_vol
            # Cap the scaling to avoid extreme leverage
            vol_scalar = min(vol_scalar, 2.0)
            weights = weights * vol_scalar
        
        return weights
    
    def _apply_position_constraints(self, weights: pd.Series, 
                                  top_assets: List[str]) -> pd.Series:
        """
        Apply position size constraints (min/max position sizes)
        
        Args:
            weights: Current weights
            top_assets: Selected assets
            
        Returns:
            Constrained weights
        """
        # Apply min/max constraints
        constrained_weights = weights.clip(
            lower=self.config.min_position_size,
            upper=self.config.max_position_size
        )
        
        # Renormalize to ensure weights sum to target (accounting for cash buffer)
        target_invested = 1.0 - self.config.cash_buffer
        current_sum = constrained_weights.sum()
        
        if current_sum > 0:
            constrained_weights = constrained_weights * (target_invested / current_sum)
        
        # Final check - ensure no weight exceeds maximum
        constrained_weights = constrained_weights.clip(upper=self.config.max_position_size)
        
        return constrained_weights
    
    def _apply_sector_constraints(self, weights: pd.Series, 
                                top_assets: List[str]) -> pd.Series:
        """
        Apply sector concentration limits
        
        Args:
            weights: Current weights
            top_assets: Selected assets
            
        Returns:
            Sector-constrained weights
        """
        # Simple sector mapping (in practice, would use external data)
        sector_mapping = self._get_sector_mapping()
        
        # Calculate current sector exposures
        sector_weights = {}
        for asset in top_assets:
            sector = sector_mapping.get(asset, 'Other')
            sector_weights[sector] = sector_weights.get(sector, 0) + weights[asset]
        
        # Apply sector constraints
        adjusted_weights = weights.copy()
        
        for sector, sector_weight in sector_weights.items():
            if sector_weight > self.config.max_sector_weight:
                # Scale down assets in this sector
                sector_assets = [a for a in top_assets 
                               if sector_mapping.get(a, 'Other') == sector]
                
                scale_factor = self.config.max_sector_weight / sector_weight
                
                for asset in sector_assets:
                    adjusted_weights[asset] *= scale_factor
        
        # Renormalize
        target_invested = 1.0 - self.config.cash_buffer
        current_sum = adjusted_weights.sum()
        
        if current_sum > 0:
            adjusted_weights = adjusted_weights * (target_invested / current_sum)
        
        return adjusted_weights
    
    def _get_sector_mapping(self) -> Dict[str, str]:
        """
        Get sector mapping for assets (simplified version)
        In practice, this would come from external data sources
        """
        sector_mapping = {
            # US Equity
            'SPY': 'US Equity', 'QQQ': 'US Tech', 'IWM': 'US Small Cap',
            'VTI': 'US Equity',
            
            # International
            'EFA': 'Developed Markets', 'EEM': 'Emerging Markets',
            'VEA': 'Developed Markets', 'VWO': 'Emerging Markets',
            
            # Fixed Income
            'TLT': 'Treasury Bonds', 'IEF': 'Treasury Bonds',
            'LQD': 'Corporate Bonds', 'HYG': 'High Yield',
            
            # Commodities
            'GLD': 'Precious Metals', 'SLV': 'Precious Metals',
            'DBC': 'Commodities', 'USO': 'Energy',
            
            # REITs
            'VNQ': 'REITs', 'VNQI': 'REITs',
            
            # Currencies
            'UUP': 'Currencies', 'FXE': 'Currencies'
        }
        
        return sector_mapping
    
    def apply_transaction_costs(self, positions: pd.DataFrame, 
                              returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply transaction costs to portfolio returns
        
        Args:
            positions: Portfolio positions over time
            returns: Asset returns
            
        Returns:
            Tuple of (adjusted_returns, turnover_series)
        """
        logger.info("Applying transaction costs...")
        
        # Calculate position changes (turnover)
        position_changes = positions.diff().abs()
        turnover = position_changes.sum(axis=1)
        
        # Calculate transaction costs
        tc_drag = turnover * self.config.transaction_cost
        
        # Calculate portfolio returns before costs
        portfolio_returns = (positions.shift(1) * returns).sum(axis=1)
        
        # Apply transaction costs
        adjusted_returns = portfolio_returns - tc_drag
        
        # Store turnover history
        self.turnover_history = turnover.dropna()
        
        logger.info(f"Average turnover: {turnover.mean():.2%}")
        logger.info(f"Average transaction cost drag: {tc_drag.mean():.3%}")
        
        return adjusted_returns, turnover
    
    def _calculate_turnover(self):
        """Calculate portfolio turnover metrics"""
        
        if self.positions is None:
            return
        
        # Calculate position changes
        position_changes = self.positions.diff().abs()
        daily_turnover = position_changes.sum(axis=1)
        
        # Store turnover metrics
        self.turnover_metrics = {
            'daily_turnover': daily_turnover,
            'avg_daily_turnover': daily_turnover.mean(),
            'annual_turnover': daily_turnover.mean() * 252,
            'max_daily_turnover': daily_turnover.max(),
            'turnover_volatility': daily_turnover.std()
        }
    
    def optimize_portfolio(self, expected_returns: pd.Series, 
                         covariance_matrix: pd.DataFrame,
                         current_weights: Optional[pd.Series] = None) -> pd.Series:
        """
        Optimize portfolio using mean-variance optimization
        
        Args:
            expected_returns: Expected returns for assets
            covariance_matrix: Asset covariance matrix
            current_weights: Current portfolio weights (for transaction cost consideration)
            
        Returns:
            Optimized portfolio weights
        """
        n_assets = len(expected_returns)
        
        # Objective function: maximize return - risk penalty - transaction costs
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            
            # Risk penalty (can be adjusted)
            risk_penalty = 0.5 * portfolio_risk ** 2
            
            # Transaction cost penalty if current weights provided
            tc_penalty = 0
            if current_weights is not None:
                turnover = np.sum(np.abs(weights - current_weights))
                tc_penalty = self.config.transaction_cost * turnover
            
            return -(portfolio_return - risk_penalty - tc_penalty)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - (1.0 - self.config.cash_buffer)}  # Full investment minus cash buffer
        ]
        
        # Bounds (position size limits)
        bounds = [(0, self.config.max_position_size) for _ in range(n_assets)]
        
        # Initial guess (equal weights or current weights)
        if current_weights is not None:
            x0 = current_weights.values
        else:
            x0 = np.ones(n_assets) / n_assets * (1.0 - self.config.cash_buffer)
        
        # Optimize
        result = minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = pd.Series(result.x, index=expected_returns.index)
            # Apply minimum position constraint
            optimal_weights = optimal_weights.where(
                optimal_weights >= self.config.min_position_size, 0
            )
            # Renormalize
            optimal_weights = optimal_weights / optimal_weights.sum() * (1.0 - self.config.cash_buffer)
        else:
            logger.warning("Portfolio optimization failed, using equal weights")
            optimal_weights = pd.Series(
                (1.0 - self.config.cash_buffer) / n_assets, 
                index=expected_returns.index
            )
        
        return optimal_weights
    
    def get_portfolio_analytics(self) -> Dict:
        """
        Get comprehensive portfolio analytics
        
        Returns:
            Dictionary with portfolio metrics
        """
        if self.positions is None:
            return {'status': 'No positions calculated'}
        
        # Position concentration metrics
        avg_positions = (self.positions > 0).sum(axis=1).mean()
        max_position = self.positions.max().max()
        avg_position_size = self.positions[self.positions > 0].mean().mean()
        
        # Turnover metrics
        if hasattr(self, 'turnover_metrics'):
            turnover_stats = self.turnover_metrics
        else:
            turnover_stats = {}
        
        # Weight distribution
        weight_stats = {
            'avg_max_weight': self.positions.max(axis=1).mean(),
            'avg_min_weight': self.positions[self.positions > 0].min(axis=1).mean(),
            'weight_concentration': self._calculate_herfindahl_index()
        }
        
        analytics = {
            'avg_active_positions': avg_positions,
            'max_single_position': max_position,
            'avg_position_size': avg_position_size,
            'weight_statistics': weight_stats,
            'turnover_statistics': turnover_stats
        }
        
        return analytics
    
    def _calculate_herfindahl_index(self) -> float:
        """Calculate portfolio concentration using Herfindahl index"""
        
        if self.positions is None:
            return np.nan
        
        # Calculate daily Herfindahl indices
        daily_hhi = (self.positions ** 2).sum(axis=1)
        
        return daily_hhi.mean()
    
    def export_portfolio_analytics(self, file_path: str):
        """Export portfolio analytics to file"""
        
        analytics = self.get_portfolio_analytics()
        
        with pd.ExcelWriter(f"{file_path}_portfolio_analytics.xlsx") as writer:
            # Export positions
            if self.positions is not None:
                self.positions.to_excel(writer, sheet_name='Positions')
            
            # Export turnover
            if hasattr(self, 'turnover_history'):
                self.turnover_history.to_excel(writer, sheet_name='Turnover')
            
            # Export analytics summary
            analytics_df = pd.DataFrame.from_dict(analytics, orient='index')
            analytics_df.to_excel(writer, sheet_name='Analytics_Summary')
        
        logger.info(f"Portfolio analytics exported to {file_path}_portfolio_analytics.xlsx")