"""
Database Management Module
=========================

PostgreSQL database integration for momentum strategy data persistence.
Handles strategy results, performance metrics, portfolio positions, and historical data.

Author: Rishabh Ashok Patil
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import json

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database manager for momentum strategy data persistence"""
    
    def __init__(self, connection_string: str = None):
        """Initialize database connection"""
        self.connection_string = connection_string or self._get_connection_string()
        self.engine = None
        self.session_factory = None
        self._initialize_connection()
    
    def _get_connection_string(self) -> str:
        """Get database connection string from environment or defaults"""
        # Try environment variables first
        db_url = os.getenv('DATABASE_URL')
        if db_url:
            return db_url
        
        # Default local PostgreSQL connection
        host = os.getenv('DB_HOST', 'localhost')
        port = os.getenv('DB_PORT', '5432')
        database = os.getenv('DB_NAME', 'momentum_strategy')
        username = os.getenv('DB_USER', 'postgres')
        password = os.getenv('DB_PASSWORD', 'password')
        
        return f"postgresql://{username}:{password}@{host}:{port}/{database}"
    
    def _initialize_connection(self):
        """Initialize SQLAlchemy engine and session factory"""
        try:
            self.engine = create_engine(
                self.connection_string,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=False
            )
            
            self.session_factory = sessionmaker(bind=self.engine)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info("Database connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise
    
    def create_tables(self):
        """Create all required database tables"""
        logger.info("Creating database tables...")
        
        tables_sql = """
        -- Strategy runs table
        CREATE TABLE IF NOT EXISTS strategy_runs (
            id SERIAL PRIMARY KEY,
            run_id VARCHAR(50) UNIQUE NOT NULL,
            strategy_name VARCHAR(100) NOT NULL,
            universe_name VARCHAR(50) NOT NULL,
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status VARCHAR(20) DEFAULT 'running',
            config JSONB,
            author VARCHAR(100) DEFAULT 'Rishabh Ashok Patil'
        );
        
        -- Performance metrics table
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id SERIAL PRIMARY KEY,
            run_id VARCHAR(50) REFERENCES strategy_runs(run_id),
            metric_date DATE NOT NULL,
            total_return DECIMAL(10,6),
            annual_return DECIMAL(10,6),
            volatility DECIMAL(10,6),
            sharpe_ratio DECIMAL(10,6),
            max_drawdown DECIMAL(10,6),
            var_95 DECIMAL(10,6),
            beta DECIMAL(10,6),
            alpha DECIMAL(10,6),
            information_ratio DECIMAL(10,6),
            tracking_error DECIMAL(10,6),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Portfolio positions table
        CREATE TABLE IF NOT EXISTS portfolio_positions (
            id SERIAL PRIMARY KEY,
            run_id VARCHAR(50) REFERENCES strategy_runs(run_id),
            position_date DATE NOT NULL,
            asset_symbol VARCHAR(20) NOT NULL,
            weight DECIMAL(10,6) NOT NULL,
            market_value DECIMAL(15,2),
            sector VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Strategy returns table
        CREATE TABLE IF NOT EXISTS strategy_returns (
            id SERIAL PRIMARY KEY,
            run_id VARCHAR(50) REFERENCES strategy_runs(run_id),
            return_date DATE NOT NULL,
            strategy_return DECIMAL(10,6) NOT NULL,
            benchmark_return DECIMAL(10,6),
            excess_return DECIMAL(10,6),
            cumulative_return DECIMAL(10,6),
            drawdown DECIMAL(10,6),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Market data table
        CREATE TABLE IF NOT EXISTS market_data (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            data_date DATE NOT NULL,
            open_price DECIMAL(12,4),
            high_price DECIMAL(12,4),
            low_price DECIMAL(12,4),
            close_price DECIMAL(12,4),
            adj_close_price DECIMAL(12,4),
            volume BIGINT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, data_date)
        );
        
        -- Risk metrics table
        CREATE TABLE IF NOT EXISTS risk_metrics (
            id SERIAL PRIMARY KEY,
            run_id VARCHAR(50) REFERENCES strategy_runs(run_id),
            metric_date DATE NOT NULL,
            portfolio_volatility DECIMAL(10,6),
            var_1d DECIMAL(10,6),
            var_5d DECIMAL(10,6),
            cvar_95 DECIMAL(10,6),
            max_position_weight DECIMAL(10,6),
            concentration_ratio DECIMAL(10,6),
            turnover DECIMAL(10,6),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Analytics results table
        CREATE TABLE IF NOT EXISTS analytics_results (
            id SERIAL PRIMARY KEY,
            run_id VARCHAR(50) REFERENCES strategy_runs(run_id),
            analysis_type VARCHAR(50) NOT NULL,
            analysis_date DATE NOT NULL,
            results JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_strategy_runs_run_id ON strategy_runs(run_id);
        CREATE INDEX IF NOT EXISTS idx_performance_metrics_run_id ON performance_metrics(run_id);
        CREATE INDEX IF NOT EXISTS idx_portfolio_positions_run_id ON portfolio_positions(run_id);
        CREATE INDEX IF NOT EXISTS idx_strategy_returns_run_id ON strategy_returns(run_id);
        CREATE INDEX IF NOT EXISTS idx_market_data_symbol_date ON market_data(symbol, data_date);
        CREATE INDEX IF NOT EXISTS idx_risk_metrics_run_id ON risk_metrics(run_id);
        CREATE INDEX IF NOT EXISTS idx_analytics_results_run_id ON analytics_results(run_id);
        """
        
        try:
            with self.engine.connect() as conn:
                # Execute each statement separately
                for statement in tables_sql.split(';'):
                    if statement.strip():
                        conn.execute(text(statement))
                conn.commit()
            
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def save_strategy_run(self, run_id: str, strategy_name: str, universe_name: str,
                         start_date: str, end_date: str, config: Dict) -> bool:
        """Save strategy run metadata"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO strategy_runs 
                    (run_id, strategy_name, universe_name, start_date, end_date, config, status)
                    VALUES (:run_id, :strategy_name, :universe_name, :start_date, :end_date, :config, 'running')
                    ON CONFLICT (run_id) DO UPDATE SET
                    updated_at = CURRENT_TIMESTAMP,
                    status = 'running'
                """), {
                    'run_id': run_id,
                    'strategy_name': strategy_name,
                    'universe_name': universe_name,
                    'start_date': start_date,
                    'end_date': end_date,
                    'config': json.dumps(config)
                })
                conn.commit()
            
            logger.info(f"Strategy run {run_id} saved to database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save strategy run: {e}")
            return False
    
    def save_performance_metrics(self, run_id: str, metrics: Dict) -> bool:
        """Save performance metrics to database"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO performance_metrics 
                    (run_id, metric_date, total_return, annual_return, volatility, sharpe_ratio,
                     max_drawdown, var_95, beta, alpha, information_ratio, tracking_error)
                    VALUES (:run_id, :metric_date, :total_return, :annual_return, :volatility,
                            :sharpe_ratio, :max_drawdown, :var_95, :beta, :alpha, 
                            :information_ratio, :tracking_error)
                """), {
                    'run_id': run_id,
                    'metric_date': datetime.now().date(),
                    'total_return': metrics.get('total_return'),
                    'annual_return': metrics.get('annual_return'),
                    'volatility': metrics.get('volatility'),
                    'sharpe_ratio': metrics.get('sharpe_ratio'),
                    'max_drawdown': metrics.get('max_drawdown'),
                    'var_95': metrics.get('var_95'),
                    'beta': metrics.get('beta'),
                    'alpha': metrics.get('alpha'),
                    'information_ratio': metrics.get('information_ratio'),
                    'tracking_error': metrics.get('tracking_error')
                })
                conn.commit()
            
            logger.info(f"Performance metrics saved for run {run_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save performance metrics: {e}")
            return False
    
    def save_portfolio_positions(self, run_id: str, positions: pd.DataFrame) -> bool:
        """Save portfolio positions to database"""
        try:
            # Prepare data for bulk insert
            position_data = []
            for date, row in positions.iterrows():
                for asset, weight in row.items():
                    if weight > 0.001:  # Only save significant positions
                        position_data.append({
                            'run_id': run_id,
                            'position_date': date.date() if hasattr(date, 'date') else date,
                            'asset_symbol': asset,
                            'weight': float(weight),
                            'market_value': None,  # Can be calculated later
                            'sector': self._get_asset_sector(asset)
                        })
            
            if position_data:
                df = pd.DataFrame(position_data)
                df.to_sql('portfolio_positions', self.engine, if_exists='append', index=False)
                
                logger.info(f"Saved {len(position_data)} portfolio positions for run {run_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save portfolio positions: {e}")
            return False
    
    def save_strategy_returns(self, run_id: str, returns: pd.Series, 
                            benchmark_returns: pd.Series = None) -> bool:
        """Save strategy returns to database"""
        try:
            returns_data = []
            cumulative_return = 1.0
            peak = 1.0
            
            for date, ret in returns.items():
                cumulative_return *= (1 + ret)
                peak = max(peak, cumulative_return)
                drawdown = (cumulative_return - peak) / peak
                
                benchmark_ret = None
                excess_ret = None
                if benchmark_returns is not None and date in benchmark_returns.index:
                    benchmark_ret = float(benchmark_returns[date])
                    excess_ret = float(ret - benchmark_ret)
                
                returns_data.append({
                    'run_id': run_id,
                    'return_date': date.date() if hasattr(date, 'date') else date,
                    'strategy_return': float(ret),
                    'benchmark_return': benchmark_ret,
                    'excess_return': excess_ret,
                    'cumulative_return': float(cumulative_return - 1),
                    'drawdown': float(drawdown)
                })
            
            if returns_data:
                df = pd.DataFrame(returns_data)
                df.to_sql('strategy_returns', self.engine, if_exists='append', index=False)
                
                logger.info(f"Saved {len(returns_data)} strategy returns for run {run_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save strategy returns: {e}")
            return False
    
    def save_analytics_results(self, run_id: str, analysis_type: str, results: Dict) -> bool:
        """Save analytics results to database"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO analytics_results 
                    (run_id, analysis_type, analysis_date, results)
                    VALUES (:run_id, :analysis_type, :analysis_date, :results)
                """), {
                    'run_id': run_id,
                    'analysis_type': analysis_type,
                    'analysis_date': datetime.now().date(),
                    'results': json.dumps(results, default=str)
                })
                conn.commit()
            
            logger.info(f"Analytics results saved for run {run_id}, type: {analysis_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save analytics results: {e}")
            return False
    
    def get_strategy_runs(self, limit: int = 100) -> List[Dict]:
        """Get recent strategy runs"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT * FROM strategy_runs 
                    ORDER BY created_at DESC 
                    LIMIT :limit
                """), {'limit': limit})
                
                return [dict(row._mapping) for row in result]
                
        except Exception as e:
            logger.error(f"Failed to get strategy runs: {e}")
            return []
    
    def get_performance_history(self, run_id: str) -> pd.DataFrame:
        """Get performance history for a strategy run"""
        try:
            query = """
                SELECT * FROM performance_metrics 
                WHERE run_id = :run_id 
                ORDER BY metric_date
            """
            
            return pd.read_sql(query, self.engine, params={'run_id': run_id})
            
        except Exception as e:
            logger.error(f"Failed to get performance history: {e}")
            return pd.DataFrame()
    
    def _get_asset_sector(self, asset: str) -> str:
        """Map asset to sector (simplified)"""
        sector_map = {
            'SPY': 'US Large Cap', 'QQQ': 'US Tech', 'IWM': 'US Small Cap',
            'VTI': 'US Total Market', 'VEA': 'International Developed',
            'VWO': 'Emerging Markets', 'TLT': 'Long-Term Bonds',
            'GLD': 'Commodities', 'VNQ': 'Real Estate'
        }
        return sector_map.get(asset, 'Other')
    
    def update_strategy_status(self, run_id: str, status: str) -> bool:
        """Update strategy run status"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    UPDATE strategy_runs 
                    SET status = :status, updated_at = CURRENT_TIMESTAMP
                    WHERE run_id = :run_id
                """), {'run_id': run_id, 'status': status})
                conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update strategy status: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")
