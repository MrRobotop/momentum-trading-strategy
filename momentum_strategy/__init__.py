"""
Multi-Asset Momentum Trading Strategy Package
============================================

A professional quantitative momentum trading strategy implementation featuring:
- Multi-timeframe momentum signal generation
- Risk-adjusted portfolio construction  
- Comprehensive backtesting framework
- Advanced performance analytics
- Interactive visualization dashboard

Author: Quantitative Developer Portfolio Project
Version: 1.0.0
"""

from .config import StrategyConfig
from .data_manager import DataManager
from .signals import MomentumSignals
from .portfolio import PortfolioConstructor
from .backtest import BacktestEngine
from .analytics import PerformanceAnalytics
from .utils import Visualizer

__version__ = "1.0.0"
__author__ = "Quantitative Developer"
__email__ = "quant.dev@example.com"

__all__ = [
    "StrategyConfig",
    "DataManager", 
    "MomentumSignals",
    "PortfolioConstructor",
    "BacktestEngine",
    "PerformanceAnalytics",
    "Visualizer"
]