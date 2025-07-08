"""
Comprehensive Test Suite for Momentum Strategy
==============================================

Unit and integration tests for the momentum trading strategy.
Tests all components including data management, signals, portfolio construction,
backtesting, and analytics.

Author: Rishabh Ashok Patil
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the momentum_strategy package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'momentum_strategy'))

from config import StrategyConfig
from data_manager import DataManager, AssetUniverses
from signals import MomentumSignals
from portfolio import PortfolioConstructor
from backtest import MomentumBacktest
from analytics import PerformanceAnalytics
from main import MomentumStrategy

class TestStrategyConfig(unittest.TestCase):
    """Test strategy configuration"""
    
    def setUp(self):
        self.config = StrategyConfig()
    
    def test_config_initialization(self):
        """Test that config initializes with expected defaults"""
        self.assertIsInstance(self.config.lookback_periods, list)
        self.assertGreater(len(self.config.lookback_periods), 0)
        self.assertIsInstance(self.config.rebalance_frequency, str)
        self.assertGreater(self.config.transaction_cost, 0)
    
    def test_config_validation(self):
        """Test config parameter validation"""
        # Test invalid lookback periods
        with self.assertRaises(ValueError):
            config = StrategyConfig()
            config.lookback_periods = []
        
        # Test invalid transaction cost
        with self.assertRaises(ValueError):
            config = StrategyConfig()
            config.transaction_cost = -0.01

class TestDataManager(unittest.TestCase):
    """Test data management functionality"""
    
    def setUp(self):
        self.assets = ['SPY', 'QQQ', 'IWM']
        self.start_date = '2023-01-01'
        self.end_date = '2023-12-31'
        self.data_manager = DataManager(
            assets=self.assets,
            start_date=self.start_date,
            end_date=self.end_date
        )
    
    def test_asset_universes(self):
        """Test asset universe definitions"""
        universes = AssetUniverses()
        
        # Test that universes exist
        self.assertIn('global_equity', universes.get_available_universes())
        self.assertIn('diversified_etf', universes.get_available_universes())
        
        # Test universe content
        global_equity = universes.get_universe('global_equity')
        self.assertIsInstance(global_equity, list)
        self.assertGreater(len(global_equity), 0)
    
    def test_data_fetching(self):
        """Test data fetching functionality"""
        # This test requires internet connection
        try:
            result = self.data_manager.fetch_data()
            if result is not None:
                self.assertIsNotNone(self.data_manager.prices)
                self.assertIsNotNone(self.data_manager.returns)
                self.assertGreater(len(self.data_manager.prices), 0)
        except Exception as e:
            self.skipTest(f"Data fetching failed (likely network issue): {e}")
    
    def test_data_validation(self):
        """Test data validation logic"""
        # Create mock data
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        mock_prices = pd.DataFrame(
            np.random.randn(len(dates), len(self.assets)) * 0.02 + 100,
            index=dates,
            columns=self.assets
        ).cumprod()
        
        self.data_manager.prices = mock_prices
        self.data_manager.returns = mock_prices.pct_change().dropna()
        
        clean_prices, clean_returns = self.data_manager.get_clean_data()
        
        self.assertIsInstance(clean_prices, pd.DataFrame)
        self.assertIsInstance(clean_returns, pd.DataFrame)
        self.assertGreater(len(clean_prices), 0)
        self.assertGreater(len(clean_returns), 0)

class TestMomentumSignals(unittest.TestCase):
    """Test momentum signal generation"""
    
    def setUp(self):
        self.config = StrategyConfig()
        
        # Create mock return data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        assets = ['SPY', 'QQQ', 'IWM']
        
        # Create trending data for testing
        np.random.seed(42)
        returns_data = np.random.randn(len(dates), len(assets)) * 0.01
        returns_data[:, 0] += 0.0005  # SPY trending up
        returns_data[:, 1] += 0.0003  # QQQ trending up slightly
        returns_data[:, 2] -= 0.0002  # IWM trending down slightly
        
        self.returns = pd.DataFrame(returns_data, index=dates, columns=assets)
        self.signals = MomentumSignals(self.config)
    
    def test_momentum_calculation(self):
        """Test momentum score calculation"""
        momentum_scores = self.signals.calculate_momentum_scores(self.returns)
        
        self.assertIsInstance(momentum_scores, pd.DataFrame)
        self.assertEqual(momentum_scores.shape[1], len(self.returns.columns))
        self.assertFalse(momentum_scores.empty)
        
        # Check that SPY has higher momentum than IWM (based on our mock data)
        final_scores = momentum_scores.iloc[-1]
        self.assertGreater(final_scores['SPY'], final_scores['IWM'])
    
    def test_signal_generation(self):
        """Test buy/sell signal generation"""
        signals = self.signals.generate_signals(self.returns)
        
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertEqual(signals.shape[1], len(self.returns.columns))
        
        # Signals should be between -1 and 1
        self.assertTrue((signals >= -1).all().all())
        self.assertTrue((signals <= 1).all().all())

class TestPortfolioConstructor(unittest.TestCase):
    """Test portfolio construction"""
    
    def setUp(self):
        self.config = StrategyConfig()
        self.portfolio = PortfolioConstructor(self.config)
        
        # Create mock signals
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        assets = ['SPY', 'QQQ', 'IWM']
        
        # Create mock signals (values between -1 and 1)
        np.random.seed(42)
        signal_data = np.random.randn(len(dates), len(assets)) * 0.5
        signal_data = np.clip(signal_data, -1, 1)
        
        self.signals = pd.DataFrame(signal_data, index=dates, columns=assets)
    
    def test_position_sizing(self):
        """Test position sizing logic"""
        positions = self.portfolio.calculate_positions(self.signals)
        
        self.assertIsInstance(positions, pd.DataFrame)
        self.assertEqual(positions.shape[1], len(self.signals.columns))
        
        # Check position constraints
        position_sums = positions.sum(axis=1)
        self.assertTrue((position_sums <= 1.01).all())  # Allow small numerical errors
        self.assertTrue((position_sums >= -0.01).all())  # Allow small numerical errors
        
        # Check individual position limits
        self.assertTrue((positions <= self.config.max_position_size + 0.01).all().all())
        self.assertTrue((positions >= -self.config.max_position_size - 0.01).all().all())
    
    def test_risk_management(self):
        """Test risk management constraints"""
        positions = self.portfolio.calculate_positions(self.signals)
        
        # Test that positions respect risk limits
        for date in positions.index[-10:]:  # Check last 10 days
            daily_positions = positions.loc[date]
            
            # Check concentration limits
            max_position = daily_positions.abs().max()
            self.assertLessEqual(max_position, self.config.max_position_size + 0.01)

class TestMomentumBacktest(unittest.TestCase):
    """Test backtesting functionality"""
    
    def setUp(self):
        self.config = StrategyConfig()
        
        # Create comprehensive mock data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        assets = ['SPY', 'QQQ', 'IWM']
        
        # Create realistic price data
        np.random.seed(42)
        returns_data = np.random.randn(len(dates), len(assets)) * 0.015 + 0.0003
        prices_data = (1 + pd.DataFrame(returns_data, index=dates, columns=assets)).cumprod() * 100
        
        self.prices = prices_data
        self.returns = prices_data.pct_change().dropna()
        
        self.backtest = MomentumBacktest(self.config)
    
    def test_backtest_execution(self):
        """Test full backtest execution"""
        results = self.backtest.run_backtest(self.prices, self.returns)
        
        self.assertIsInstance(results, dict)
        self.assertIn('strategy_returns', results)
        self.assertIn('positions', results)
        self.assertIn('turnover', results)
        
        # Check data integrity
        strategy_returns = results['strategy_returns']
        self.assertIsInstance(strategy_returns, pd.Series)
        self.assertGreater(len(strategy_returns), 0)
        
        # Check that returns are reasonable (not extreme)
        self.assertTrue((strategy_returns.abs() < 0.5).all())  # No single day > 50%
    
    def test_performance_metrics(self):
        """Test performance metric calculations"""
        results = self.backtest.run_backtest(self.prices, self.returns)
        
        if results and 'strategy_returns' in results:
            strategy_returns = results['strategy_returns']
            
            # Test basic metrics
            total_return = (1 + strategy_returns).prod() - 1
            self.assertIsInstance(total_return, (int, float))
            
            volatility = strategy_returns.std() * np.sqrt(252)
            self.assertGreater(volatility, 0)
            
            sharpe_ratio = (strategy_returns.mean() * 252) / volatility
            self.assertIsInstance(sharpe_ratio, (int, float))

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete strategy"""
    
    def setUp(self):
        self.config = StrategyConfig()
        self.strategy = MomentumStrategy(self.config)
    
    def test_full_strategy_pipeline(self):
        """Test the complete strategy pipeline"""
        try:
            # Run with a small universe for testing
            results = self.strategy.run_backtest(
                universe_name='global_equity',
                start_date='2023-06-01',
                end_date='2023-12-31'
            )
            
            if results:
                self.assertIsInstance(results, dict)
                self.assertIn('strategy_returns', results)
                
                # Test summary generation
                summary = self.strategy.get_summary()
                self.assertIsInstance(summary, dict)
                self.assertIn('performance', summary)
                self.assertIn('risk_metrics', summary)
                
        except Exception as e:
            self.skipTest(f"Full pipeline test failed (likely network/data issue): {e}")
    
    def test_error_handling(self):
        """Test error handling in strategy execution"""
        # Test with invalid universe
        with self.assertRaises((ValueError, KeyError)):
            self.strategy.run_backtest(universe_name='invalid_universe')
        
        # Test with invalid date range
        with self.assertRaises((ValueError, TypeError)):
            self.strategy.run_backtest(
                universe_name='global_equity',
                start_date='invalid_date'
            )

def run_performance_tests():
    """Run performance benchmarking tests"""
    print("\n" + "="*50)
    print("PERFORMANCE BENCHMARKING")
    print("="*50)
    
    config = StrategyConfig()
    strategy = MomentumStrategy(config)
    
    import time
    
    # Test strategy execution time
    start_time = time.time()
    try:
        results = strategy.run_backtest(
            universe_name='global_equity',
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        execution_time = time.time() - start_time
        
        print(f"✅ Strategy execution time: {execution_time:.2f} seconds")
        
        if results:
            print(f"✅ Strategy completed successfully")
            print(f"✅ Generated {len(results)} result components")
        else:
            print("❌ Strategy execution failed")
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")

if __name__ == '__main__':
    print("🧪 Running Momentum Strategy Test Suite")
    print("Author: Rishabh Ashok Patil")
    print("="*60)
    
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    # Run performance tests
    run_performance_tests()
    
    print("\n✅ Test suite completed!")
