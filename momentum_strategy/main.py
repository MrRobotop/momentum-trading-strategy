"""
Multi-Asset Momentum Trading Strategy - Main Execution Script
============================================================

This is the main execution script for the momentum trading strategy.
It demonstrates the complete workflow from data acquisition to 
performance analysis and visualization.

Usage:
    python main.py [--config CONFIG_FILE] [--universe UNIVERSE_NAME] [--start START_DATE] [--end END_DATE]
    
Example:
    python main.py --universe diversified_etf --start 2019-01-01 --end 2024-01-01
"""

import sys
import os
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
import warnings
import pandas as pd

# Add the momentum_strategy package to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import strategy components
from config import StrategyConfig, StrategyPresets
from data_manager import DataManager, AssetUniverses
from signals import MomentumSignals
from portfolio import PortfolioConstructor
from backtest import BacktestEngine
from analytics import PerformanceAnalytics
from database import DatabaseManager
from monitoring import setup_monitoring, StrategyMonitor
from utils import Visualizer, StrategyUtils, save_results_to_excel

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('momentum_strategy.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MomentumStrategy:
    """
    Main strategy class that orchestrates the complete momentum trading strategy
    
    This class provides a high-level interface for:
    - Running backtests with different configurations
    - Analyzing strategy performance
    - Generating visualizations and reports
    - Exporting results for further analysis
    """
    
    def __init__(self, config: StrategyConfig = None):
        """
        Initialize the momentum strategy
        
        Args:
            config: Strategy configuration (uses default if None)
        """
        self.config = config or StrategyConfig()
        self.results = {}
        self.metrics = {}
        self.data_manager = None
        self.visualizer = None

        # Initialize database and monitoring
        self.db_manager = None
        self.monitor = None
        self.strategy_monitor = None

        # Try to initialize database (optional)
        try:
            self.db_manager = DatabaseManager()
            self.db_manager.create_tables()
            logger.info("Database connection established")
        except Exception as e:
            logger.warning(f"Database not available: {e}")

        # Initialize monitoring
        try:
            self.monitor = setup_monitoring(email_alerts=False, slack_alerts=False)
            self.strategy_monitor = StrategyMonitor(self.monitor)
            logger.info("Monitoring system initialized")
        except Exception as e:
            logger.warning(f"Monitoring not available: {e}")

        logger.info("Momentum Strategy initialized by Rishabh Ashok Patil")
        logger.info(f"Configuration: {self.config.__class__.__name__}")
    
    def run_backtest(self, 
                    assets: list = None,
                    start_date: str = None,
                    end_date: str = None,
                    universe_name: str = 'diversified_etf') -> Dict:
        """
        Run complete momentum strategy backtest
        
        Args:
            assets: List of asset tickers (if None, uses predefined universe)
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            universe_name: Name of predefined universe to use
            
        Returns:
            Dictionary containing backtest results
        """
        logger.info("="*80)
        logger.info("STARTING MOMENTUM STRATEGY BACKTEST")
        logger.info("="*80)
        
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        
        # Set asset universe
        if assets is None:
            assets = self._get_asset_universe(universe_name)
        
        logger.info(f"Universe: {universe_name} ({len(assets)} assets)")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Assets: {', '.join(assets)}")
        
        # Step 1: Data Management
        logger.info("\n" + "="*60)
        logger.info("STEP 1: DATA ACQUISITION AND PREPROCESSING")
        logger.info("="*60)
        
        self.data_manager = DataManager(assets, start_date, end_date)
        prices, returns = self.data_manager.get_clean_data()
        
        if prices is None or prices.empty:
            logger.error("Failed to acquire data. Exiting.")
            return {}
        
        # Print data summary
        data_summary = self.data_manager.get_data_summary()
        logger.info(f"Data quality: {data_summary['data_quality']['status']}")
        logger.info(f"Date range: {data_summary['date_range']['start']} to {data_summary['date_range']['end']}")
        logger.info(f"Total observations: {data_summary['date_range']['total_days']}")
        
        # Step 2: Run Backtest
        logger.info("\n" + "="*60)
        logger.info("STEP 2: RUNNING STRATEGY BACKTEST")
        logger.info("="*60)
        
        backtest_engine = BacktestEngine(self.config)
        self.results = backtest_engine.run_backtest(prices, returns)
        
        # Step 3: Performance Analytics
        logger.info("\n" + "="*60)
        logger.info("STEP 3: CALCULATING PERFORMANCE METRICS")
        logger.info("="*60)
        
        analytics = PerformanceAnalytics(self.results, self.config)
        self.metrics = analytics.calculate_comprehensive_metrics()

        # Step 3.5: Save to Database and Monitor
        self._save_to_database(universe_name, start_date, end_date)
        self._monitor_performance()

        # Step 4: Display Results
        self._display_results()
        
        # Step 5: Create Visualizations
        logger.info("\n" + "="*60)
        logger.info("STEP 4: GENERATING VISUALIZATIONS")
        logger.info("="*60)
        
        self.visualizer = Visualizer(self.results, self.metrics)
        
        # Create and show performance summary
        fig = self.visualizer.plot_performance_summary()
        
        logger.info("Backtest completed successfully!")
        
        return self.results
    
    def _get_asset_universe(self, universe_name: str) -> list:
        """Get predefined asset universe"""
        
        universe_map = {
            'diversified_etf': AssetUniverses.diversified_etf,
            'sector_rotation': AssetUniverses.sector_rotation,
            'global_equity': AssetUniverses.global_equity,
            'factor_investing': AssetUniverses.factor_investing
        }
        
        if universe_name in universe_map:
            return universe_map[universe_name]()
        else:
            logger.warning(f"Unknown universe '{universe_name}', using diversified_etf")
            return AssetUniverses.diversified_etf()
    
    def _display_results(self):
        """Display formatted results summary"""
        
        logger.info("\n" + "="*80)
        logger.info("PERFORMANCE METRICS SUMMARY")
        logger.info("="*80)
        
        # Format metrics for display
        formatted_metrics = StrategyUtils.format_metrics_for_display(self.metrics)
        
        # Group metrics by category
        basic_metrics = {k: v for k, v in formatted_metrics.items() 
                        if any(term in k.lower() for term in ['return', 'volatility', 'excess'])}
        
        risk_metrics = {k: v for k, v in formatted_metrics.items() 
                       if any(term in k.lower() for term in ['sharpe', 'ratio', 'drawdown', 'var'])}
        
        portfolio_metrics = {k: v for k, v in formatted_metrics.items() 
                            if any(term in k.lower() for term in ['turnover', 'position', 'concentration'])}
        
        # Display basic performance
        logger.info("\n📈 PERFORMANCE SUMMARY:")
        logger.info("-" * 50)
        for metric, value in basic_metrics.items():
            logger.info(f"{metric:<35}: {value}")
        
        # Display risk metrics
        logger.info("\n⚠️  RISK METRICS:")
        logger.info("-" * 50)
        for metric, value in risk_metrics.items():
            logger.info(f"{metric:<35}: {value}")
        
        # Display portfolio metrics
        if portfolio_metrics:
            logger.info("\n📊 PORTFOLIO METRICS:")
            logger.info("-" * 50)
            for metric, value in portfolio_metrics.items():
                logger.info(f"{metric:<35}: {value}")
        
        # Strategy insights
        self._display_strategy_insights()
    
    def _display_strategy_insights(self):
        """Display strategy-specific insights"""
        
        logger.info("\n" + "="*80)
        logger.info("STRATEGY INSIGHTS")
        logger.info("="*80)
        
        # Portfolio characteristics
        if 'positions' in self.results:
            positions = self.results['positions']
            avg_positions = (positions > 0).sum(axis=1).mean()
            max_position = positions.max().max()
            
            logger.info(f"\n🎯 PORTFOLIO CHARACTERISTICS:")
            logger.info(f"Average number of positions: {avg_positions:.1f}")
            logger.info(f"Maximum single position: {max_position:.2%}")
            
            # Most held assets
            total_exposure = positions.sum(axis=0).sort_values(ascending=False)
            logger.info(f"\n📋 TOP 5 MOST HELD ASSETS:")
            for i, (asset, exposure) in enumerate(total_exposure.head().items(), 1):
                pct_time = (positions[asset] > 0).mean()
                logger.info(f"{i}. {asset}: {exposure:.1f}% total exposure ({pct_time:.1%} of time)")
        
        # Turnover analysis
        if 'turnover' in self.results:
            turnover = self.results['turnover']
            avg_monthly_turnover = turnover.mean()
            logger.info(f"\n🔄 TURNOVER ANALYSIS:")
            logger.info(f"Average monthly turnover: {avg_monthly_turnover:.2%}")
            logger.info(f"Estimated annual trading cost: {avg_monthly_turnover * 12 * self.config.transaction_cost:.3%}")
        
        # Signal analysis
        if 'momentum_scores' in self.results:
            momentum_scores = self.results['momentum_scores']
            signal_stability = momentum_scores.std(axis=1).mean()
            logger.info(f"\n📡 SIGNAL ANALYSIS:")
            logger.info(f"Average signal dispersion: {signal_stability:.3f}")
            logger.info(f"Signal range: [{momentum_scores.min().min():.3f}, {momentum_scores.max().max():.3f}]")
    
    def run_sensitivity_analysis(self) -> Dict:
        """Run sensitivity analysis on key parameters"""
        
        logger.info("\n" + "="*60)
        logger.info("RUNNING SENSITIVITY ANALYSIS")
        logger.info("="*60)
        
        base_results = self.results
        sensitivity_results = {}
        
        # Test different lookback periods
        lookback_periods = [126, 189, 252, 315, 378]  # 6M, 9M, 12M, 15M, 18M
        
        for lookback in lookback_periods:
            logger.info(f"Testing lookback period: {lookback} days")
            
            # Create modified config
            test_config = self.config.copy()
            test_config.lookback_period = lookback
            
            try:
                # Run backtest with modified config
                test_strategy = MomentumStrategy(test_config)
                test_results = test_strategy.run_backtest(
                    assets=self.data_manager.assets,
                    start_date=self.data_manager.start_date,
                    end_date=self.data_manager.end_date
                )
                
                # Extract key metrics
                if test_results:
                    test_analytics = PerformanceAnalytics(test_results, test_config)
                    test_metrics = test_analytics.calculate_comprehensive_metrics()
                    
                    sensitivity_results[f'lookback_{lookback}'] = {
                        'sharpe_ratio': test_metrics.get('Sharpe Ratio (Strategy)', 0),
                        'total_return': test_metrics.get('Total Return (Strategy)', 0),
                        'max_drawdown': test_metrics.get('Maximum Drawdown', 0),
                        'turnover': test_metrics.get('Average Turnover', 0)
                    }
                
            except Exception as e:
                logger.warning(f"Sensitivity test failed for lookback {lookback}: {e}")
                continue
        
        # Display sensitivity results
        if sensitivity_results:
            logger.info("\n📊 SENSITIVITY ANALYSIS RESULTS:")
            logger.info("-" * 70)
            logger.info(f"{'Parameter':<15} {'Sharpe':<8} {'Return':<8} {'MaxDD':<8} {'Turnover':<10}")
            logger.info("-" * 70)
            
            for param, metrics in sensitivity_results.items():
                logger.info(f"{param:<15} {metrics['sharpe_ratio']:<8.3f} "
                          f"{metrics['total_return']:<8.2%} {metrics['max_drawdown']:<8.2%} "
                          f"{metrics['turnover']:<10.2%}")
        
        return sensitivity_results
    
    def generate_report(self, output_dir: str = "results"):
        """Generate comprehensive strategy report"""
        
        logger.info(f"\n" + "="*60)
        logger.info("GENERATING COMPREHENSIVE REPORT")
        logger.info("="*60)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{output_dir}/momentum_strategy_{timestamp}"
        
        try:
            # Save results to Excel
            save_results_to_excel(self.results, self.metrics, base_filename)
            
            # Generate visualizations
            if self.visualizer:
                # Performance summary
                self.visualizer.plot_performance_summary(save_path=base_filename)
                
                # Risk analysis
                self.visualizer.plot_risk_analysis(save_path=base_filename)
                
                # Portfolio analysis
                self.visualizer.plot_portfolio_analysis(save_path=base_filename)
                
                # Signal analysis
                self.visualizer.plot_signal_analysis(save_path=base_filename)
                
                # Interactive dashboard
                dashboard = self.visualizer.create_performance_dashboard(save_path=base_filename)
                
                # Create tearsheet
                self.visualizer.create_tearsheet(save_path=base_filename)
            
            # Export individual analytics
            if self.results and self.metrics:
                analytics = PerformanceAnalytics(self.results, self.config)
                analytics.export_analytics(base_filename)
            
            logger.info(f"Report generated successfully!")
            logger.info(f"Files saved with prefix: {base_filename}")
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
    
    def get_summary(self) -> Dict:
        """Get a concise summary of strategy performance"""
        
        if not self.metrics:
            return {"status": "No backtest results available"}
        
        strategy_returns = self.results.get('strategy_returns', pd.Series())
        
        summary = {
            "strategy_name": "Multi-Asset Momentum Strategy",
            "backtest_period": {
                "start": strategy_returns.index[0].strftime('%Y-%m-%d') if len(strategy_returns) > 0 else "N/A",
                "end": strategy_returns.index[-1].strftime('%Y-%m-%d') if len(strategy_returns) > 0 else "N/A",
                "duration_years": len(strategy_returns) / 252 if len(strategy_returns) > 0 else 0
            },
            "performance": {
                "total_return": self.metrics.get('Total Return (Strategy)', 0),
                "annual_return": self.metrics.get('Annualized Return (Strategy)', 0),
                "volatility": self.metrics.get('Annualized Volatility (Strategy)', 0),
                "sharpe_ratio": self.metrics.get('Sharpe Ratio (Strategy)', 0),
                "max_drawdown": self.metrics.get('Maximum Drawdown', 0)
            },
            "risk_metrics": {
                "var_95": self.metrics.get('VaR (95%)', 0),
                "information_ratio": self.metrics.get('Information Ratio', 0),
                "beta": self.metrics.get('Beta vs Benchmark', 1),
                "tracking_error": self.metrics.get('Tracking Error', 0)
            },
            "portfolio_characteristics": {
                "avg_turnover": self.metrics.get('Average Turnover', 0),
                "avg_positions": self.metrics.get('Average Active Positions', 0),
                "max_position": self.metrics.get('Maximum Position Size', 0)
            }
        }
        
        return summary

    def _save_to_database(self, universe_name: str, start_date: str, end_date: str):
        """Save strategy results to database"""
        if not self.db_manager:
            return

        try:
            # Generate unique run ID
            run_id = f"momentum_{universe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Save strategy run metadata
            config_dict = {
                'lookback_periods': getattr(self.config, 'lookback_periods', [252]),
                'rebalance_frequency': getattr(self.config, 'rebalance_frequency', 'monthly'),
                'transaction_cost': getattr(self.config, 'transaction_cost', 0.001),
                'max_position_size': getattr(self.config, 'max_position_size', 0.2)
            }

            self.db_manager.save_strategy_run(
                run_id=run_id,
                strategy_name="Momentum Strategy",
                universe_name=universe_name,
                start_date=start_date,
                end_date=end_date,
                config=config_dict
            )

            # Save performance metrics
            if self.metrics:
                performance = self.metrics.get('performance', {})
                risk_metrics = self.metrics.get('risk_metrics', {})

                metrics_dict = {
                    'total_return': performance.get('total_return'),
                    'annual_return': performance.get('annual_return'),
                    'volatility': performance.get('volatility'),
                    'sharpe_ratio': performance.get('sharpe_ratio'),
                    'max_drawdown': performance.get('max_drawdown'),
                    'var_95': risk_metrics.get('var_95'),
                    'beta': risk_metrics.get('beta'),
                    'alpha': risk_metrics.get('alpha'),
                    'information_ratio': risk_metrics.get('information_ratio'),
                    'tracking_error': risk_metrics.get('tracking_error')
                }

                self.db_manager.save_performance_metrics(run_id, metrics_dict)

            # Save portfolio positions
            if 'positions' in self.results:
                self.db_manager.save_portfolio_positions(run_id, self.results['positions'])

            # Save strategy returns
            if 'strategy_returns' in self.results:
                benchmark_returns = self.results.get('benchmark_returns')
                self.db_manager.save_strategy_returns(
                    run_id,
                    self.results['strategy_returns'],
                    benchmark_returns
                )

            # Update status to completed
            self.db_manager.update_strategy_status(run_id, 'completed')

            logger.info(f"Strategy results saved to database with run_id: {run_id}")

        except Exception as e:
            logger.error(f"Failed to save to database: {e}")

    def _monitor_performance(self):
        """Monitor strategy performance and generate alerts"""
        if not self.strategy_monitor:
            return

        try:
            # Monitor overall strategy performance
            if self.metrics:
                self.strategy_monitor.monitor_strategy_performance(self.metrics)

            # Monitor portfolio positions
            if 'positions' in self.results:
                self.strategy_monitor.monitor_portfolio_positions(self.results['positions'])

            logger.info("Performance monitoring completed")

        except Exception as e:
            logger.error(f"Failed to monitor performance: {e}")

def main():
    """Main execution function with command line interface"""
    
    parser = argparse.ArgumentParser(description='Multi-Asset Momentum Trading Strategy')
    parser.add_argument('--config', type=str, default='default', 
                       choices=['default', 'conservative', 'aggressive', 'low_turnover'],
                       help='Strategy configuration to use')
    parser.add_argument('--universe', type=str, default='diversified_etf',
                       choices=['diversified_etf', 'sector_rotation', 'global_equity', 'factor_investing'],
                       help='Asset universe to use')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--report', action='store_true', help='Generate comprehensive report')
    parser.add_argument('--sensitivity', action='store_true', help='Run sensitivity analysis')
    parser.add_argument('--output', type=str, default='results', help='Output directory for reports')
    
    args = parser.parse_args()
    
    # Set up configuration
    config_map = {
        'default': StrategyConfig(),
        'conservative': StrategyPresets.conservative(),
        'aggressive': StrategyPresets.aggressive(),
        'low_turnover': StrategyPresets.low_turnover()
    }
    
    config = config_map[args.config]
    
    # Initialize and run strategy
    strategy = MomentumStrategy(config)
    
    # Run backtest
    results = strategy.run_backtest(
        universe_name=args.universe,
        start_date=args.start,
        end_date=args.end
    )
    
    if not results:
        logger.error("Backtest failed. Exiting.")
        return
    
    # Run sensitivity analysis if requested
    if args.sensitivity:
        strategy.run_sensitivity_analysis()
    
    # Generate report if requested
    if args.report:
        strategy.generate_report(args.output)
    
    # Print final summary
    summary = strategy.get_summary()
    
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    logger.info(f"Strategy: {summary['strategy_name']}")
    logger.info(f"Period: {summary['backtest_period']['start']} to {summary['backtest_period']['end']}")
    logger.info(f"Total Return: {summary['performance']['total_return']:.2%}")
    logger.info(f"Sharpe Ratio: {summary['performance']['sharpe_ratio']:.3f}")
    logger.info(f"Max Drawdown: {summary['performance']['max_drawdown']:.2%}")
    logger.info("="*80)

if __name__ == "__main__":
    main()