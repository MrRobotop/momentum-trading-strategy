"""
Strategy Configuration Module
============================

Contains configuration classes and parameters for the momentum trading strategy.
Centralizes all strategy parameters for easy modification and testing.
"""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class StrategyConfig:
    """
    Configuration parameters for the momentum strategy
    
    This class centralizes all strategy parameters to enable easy parameter
    sweeps, optimization, and A/B testing of different configurations.
    
    Attributes:
        lookback_period (int): Number of days for momentum calculation (default: 252)
        rebalance_freq (int): Rebalancing frequency in days (default: 21)  
        top_n_assets (int): Number of top momentum assets to hold (default: 10)
        transaction_cost (float): Transaction costs as decimal (default: 0.001)
        risk_free_rate (float): Risk-free rate for Sharpe calculation (default: 0.02)
        max_position_size (float): Maximum position size as decimal (default: 0.15)
        min_position_size (float): Minimum position size as decimal (default: 0.05)
        volatility_target (float): Target portfolio volatility (default: 0.15)
        momentum_weights (dict): Weights for different momentum timeframes
        enable_risk_targeting (bool): Whether to use volatility targeting
        enable_sector_constraints (bool): Whether to apply sector limits
        max_sector_weight (float): Maximum weight in any sector
        cash_buffer (float): Minimum cash buffer to maintain
    """
    
    # Core Strategy Parameters
    lookback_period: int = 252  # 1 year momentum lookback
    rebalance_freq: int = 21    # Monthly rebalancing (21 trading days)
    top_n_assets: int = 10      # Number of top momentum assets to hold
    
    # Cost and Risk Parameters  
    transaction_cost: float = 0.001     # 10 basis points transaction costs
    risk_free_rate: float = 0.02       # 2% risk-free rate for Sharpe ratio
    volatility_target: float = 0.15    # 15% annualized volatility target
    
    # Position Sizing Constraints
    max_position_size: float = 0.15     # 15% maximum position size
    min_position_size: float = 0.05     # 5% minimum position size
    cash_buffer: float = 0.02           # 2% cash buffer
    
    # Advanced Parameters
    momentum_weights: dict = None       # Custom momentum timeframe weights
    enable_risk_targeting: bool = True  # Enable volatility targeting
    enable_sector_constraints: bool = True  # Enable sector constraints
    max_sector_weight: float = 0.40     # 40% maximum sector weight
    
    # Signal Generation Parameters
    short_mom_period: int = 63          # 3-month momentum (quarter)
    medium_mom_period: int = 126        # 6-month momentum
    long_mom_period: int = 252          # 12-month momentum
    
    # Risk Management Parameters
    max_drawdown_limit: float = 0.20    # 20% maximum drawdown before halt
    var_confidence: float = 0.05        # 95% VaR confidence level
    correlation_lookback: int = 252     # Correlation calculation period
    
    def __post_init__(self):
        """Post-initialization validation and default setup"""
        
        # Set default momentum weights if not provided
        if self.momentum_weights is None:
            self.momentum_weights = {
                'mom_3m': 0.2,   # 3-month momentum weight
                'mom_6m': 0.3,   # 6-month momentum weight  
                'mom_12m': 0.3,  # 12-month momentum weight
                'sharpe_12m': 0.1,  # 12-month Sharpe ratio weight
                'vol_adj_mom': 0.1  # Volatility-adjusted momentum weight
            }
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate strategy parameters for consistency"""
        
        # Position size validation
        if self.min_position_size >= self.max_position_size:
            raise ValueError("min_position_size must be less than max_position_size")
        
        # Portfolio constraint validation  
        if self.top_n_assets * self.min_position_size > 1.0 - self.cash_buffer:
            raise ValueError("Minimum position sizes exceed available capital")
        
        # Momentum weights validation
        if abs(sum(self.momentum_weights.values()) - 1.0) > 1e-6:
            raise ValueError("Momentum weights must sum to 1.0")
        
        # Time period validation
        if self.rebalance_freq >= self.lookback_period:
            raise ValueError("Rebalancing frequency should be less than lookback period")
        
        # Risk parameter validation
        if not 0 < self.volatility_target < 1:
            raise ValueError("Volatility target must be between 0 and 1")
        
        if not 0 < self.max_drawdown_limit < 1:
            raise ValueError("Maximum drawdown limit must be between 0 and 1")
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization"""
        return {
            'lookback_period': self.lookback_period,
            'rebalance_freq': self.rebalance_freq,
            'top_n_assets': self.top_n_assets,
            'transaction_cost': self.transaction_cost,
            'risk_free_rate': self.risk_free_rate,
            'volatility_target': self.volatility_target,
            'max_position_size': self.max_position_size,
            'min_position_size': self.min_position_size,
            'momentum_weights': self.momentum_weights,
            'enable_risk_targeting': self.enable_risk_targeting,
            'enable_sector_constraints': self.enable_sector_constraints
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'StrategyConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict)
    
    def copy(self) -> 'StrategyConfig':
        """Create a copy of the configuration"""
        return StrategyConfig(**self.to_dict())

# Predefined Strategy Configurations
class StrategyPresets:
    """Predefined strategy configurations for different use cases"""
    
    @staticmethod
    def conservative() -> StrategyConfig:
        """Conservative momentum strategy with lower risk"""
        return StrategyConfig(
            top_n_assets=8,
            volatility_target=0.10,  # Lower vol target
            max_position_size=0.12,  # Smaller max positions
            transaction_cost=0.0015,  # Higher assumed costs
            max_sector_weight=0.30   # More sector diversification
        )
    
    @staticmethod
    def aggressive() -> StrategyConfig:
        """Aggressive momentum strategy with higher risk/return"""
        return StrategyConfig(
            top_n_assets=12,
            volatility_target=0.20,  # Higher vol target
            max_position_size=0.20,  # Larger max positions
            transaction_cost=0.0005, # Lower assumed costs
            max_sector_weight=0.50   # Less sector diversification
        )
    
    @staticmethod
    def low_turnover() -> StrategyConfig:
        """Low turnover strategy to minimize transaction costs"""
        return StrategyConfig(
            rebalance_freq=63,       # Quarterly rebalancing
            lookback_period=504,     # 2-year momentum
            transaction_cost=0.002,  # Higher cost assumption
            momentum_weights={
                'mom_3m': 0.1,       # Less weight on short-term
                'mom_6m': 0.2,
                'mom_12m': 0.5,      # More weight on long-term
                'sharpe_12m': 0.1,
                'vol_adj_mom': 0.1
            }
        )
    
    @staticmethod
    def high_frequency() -> StrategyConfig:
        """Higher frequency strategy with shorter lookbacks"""
        return StrategyConfig(
            lookback_period=126,     # 6-month momentum
            rebalance_freq=5,        # Weekly rebalancing
            transaction_cost=0.0005, # Assume low-cost execution
            momentum_weights={
                'mom_3m': 0.4,       # More weight on short-term
                'mom_6m': 0.4,
                'mom_12m': 0.1,      # Less weight on long-term
                'sharpe_12m': 0.05,
                'vol_adj_mom': 0.05
            }
        )
