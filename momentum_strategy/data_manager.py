"""
Data Management Module
=====================

Handles data acquisition, preprocessing, and validation for the momentum strategy.
Supports multiple data sources and implements robust data cleaning procedures.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    """
    Handles market data acquisition and preprocessing
    
    This class manages all data-related operations including:
    - Fetching market data from various sources
    - Data cleaning and preprocessing
    - Missing data handling
    - Data validation and quality checks
    - Corporate action adjustments
    """
    
    def __init__(self, assets: List[str], start_date: str, end_date: str, 
                 data_source: str = 'yahoo', cache_data: bool = True):
        """
        Initialize DataManager
        
        Args:
            assets: List of asset tickers/symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format  
            data_source: Data source ('yahoo', 'alpha_vantage', 'quandl')
            cache_data: Whether to cache downloaded data
        """
        self.assets = assets
        self.start_date = start_date
        self.end_date = end_date
        self.data_source = data_source
        self.cache_data = cache_data
        
        # Data storage
        self.prices = None
        self.returns = None
        self.volumes = None
        self.market_caps = None
        
        # Data quality metrics
        self.data_quality = {}
        
        # Suppress pandas warnings
        warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
        
    def fetch_data(self, include_volume: bool = True, 
                   include_market_cap: bool = False) -> pd.DataFrame:
        """
        Fetch price data for all assets
        
        Args:
            include_volume: Whether to fetch volume data
            include_market_cap: Whether to fetch market cap data
            
        Returns:
            DataFrame with adjusted close prices
        """
        logger.info(f"Fetching market data for {len(self.assets)} assets...")
        
        # Add buffer for momentum calculation
        buffer_days = 400  # ~1.5 years buffer
        buffer_start = (datetime.strptime(self.start_date, '%Y-%m-%d') - 
                       timedelta(days=buffer_days)).strftime('%Y-%m-%d')
        
        try:
            if self.data_source == 'yahoo':
                data = self._fetch_yahoo_data(buffer_start, include_volume)
            else:
                raise NotImplementedError(f"Data source '{self.data_source}' not implemented")
            
            if data is None or not data or 'Adj Close' not in data:
                raise ValueError("No data fetched")
            
            # Store data
            self.prices = data['Adj Close'].dropna()
            self.returns = self.prices.pct_change().dropna()

            # Debug information
            logger.info(f"Prices shape: {self.prices.shape}")
            logger.info(f"Returns shape: {self.returns.shape}")
            if not self.prices.empty:
                logger.info(f"Date range: {self.prices.index[0]} to {self.prices.index[-1]}")
            else:
                logger.warning("Prices DataFrame is empty!")
            
            if include_volume and 'Volume' in data:
                self.volumes = data['Volume']
            
            # Perform data quality checks
            self._validate_data()
            
            logger.info(f"Successfully fetched data for {len(self.assets)} assets")
            logger.info(f"Date range: {self.prices.index[0].date()} to {self.prices.index[-1].date()}")
            
            return self.prices
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None
    
    def _fetch_yahoo_data(self, buffer_start: str, include_volume: bool) -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        
        # Fetch data using yfinance
        logger.info(f"Downloading data from {buffer_start} to {self.end_date}")
        data = yf.download(
            self.assets,
            start=buffer_start,
            end=self.end_date,
            progress=False,
            threads=True,  # Enable multi-threading
            group_by='ticker' if len(self.assets) > 1 else None
        )

        logger.info(f"Raw data shape: {data.shape if hasattr(data, 'shape') else 'No shape'}")
        logger.info(f"Raw data columns: {data.columns.tolist() if hasattr(data, 'columns') else 'No columns'}")
        
        # Handle single vs multiple asset data structure
        if len(self.assets) == 1:
            # Single asset - data comes as simple DataFrame
            data.columns = pd.MultiIndex.from_product([[self.assets[0]], data.columns])
        
        # Restructure data for consistent access
        price_data = {}
        volume_data = {}
        
        for asset in self.assets:
            try:
                if (asset, 'Adj Close') in data.columns:
                    price_data[asset] = data[(asset, 'Adj Close')]
                elif (asset, 'Close') in data.columns:
                    price_data[asset] = data[(asset, 'Close')]
                elif 'Adj Close' in data.columns:
                    price_data[asset] = data['Adj Close']
                elif 'Close' in data.columns:
                    price_data[asset] = data['Close']
                
                if include_volume:
                    if (asset, 'Volume') in data.columns:
                        volume_data[asset] = data[(asset, 'Volume')]
                    elif 'Volume' in data.columns:
                        volume_data[asset] = data['Volume']
            
            except KeyError:
                logger.warning(f"Could not find data for asset: {asset}")
                continue
        
        # Create clean DataFrames
        clean_data = {}
        clean_data['Adj Close'] = pd.DataFrame(price_data)
        
        if include_volume and volume_data:
            clean_data['Volume'] = pd.DataFrame(volume_data)
        
        return clean_data
    
    def get_clean_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return clean prices and returns data for the strategy period
        
        Returns:
            Tuple of (clean_prices, clean_returns) for strategy period
        """
        if self.prices is None:
            logger.info("No data found, fetching data...")
            result = self.fetch_data()
            if result is None or self.prices is None:
                logger.error("Failed to fetch data")
                return None, None

        # Filter to actual strategy period
        price_start_idx = self.prices.index >= self.start_date
        returns_start_idx = self.returns.index >= self.start_date
        clean_prices = self.prices[price_start_idx].copy()
        clean_returns = self.returns[returns_start_idx].copy()
        
        # Final data cleaning
        clean_prices = self._clean_price_data(clean_prices)
        clean_returns = self._clean_return_data(clean_returns)
        
        return clean_prices, clean_returns
    
    def _clean_price_data(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate price data"""
        
        # Remove assets with insufficient data
        min_observations = 100  # Require at least ~4 months of data
        valid_assets = []
        
        for asset in prices.columns:
            valid_data = prices[asset].dropna()
            if len(valid_data) >= min_observations:
                valid_assets.append(asset)
            else:
                logger.warning(f"Removing {asset}: insufficient data ({len(valid_data)} obs)")
        
        prices = prices[valid_assets].copy()
        
        # Handle missing values (forward fill, then backward fill)
        prices = prices.fillna(method='ffill').fillna(method='bfill')
        
        # Remove extreme price movements (potential errors)
        for asset in prices.columns:
            returns = prices[asset].pct_change()
            extreme_moves = np.abs(returns) > 0.5  # 50% daily moves
            
            if extreme_moves.any():
                logger.warning(f"Found {extreme_moves.sum()} extreme moves in {asset}")
                # Replace extreme moves with previous day's price
                prices.loc[extreme_moves, asset] = prices[asset].shift(1)[extreme_moves]
        
        return prices
    
    def _clean_return_data(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate return data"""
        
        # Cap extreme returns at +/- 25% daily
        returns = returns.clip(lower=-0.25, upper=0.25)
        
        # Replace infinite values with NaN, then forward fill
        returns = returns.replace([np.inf, -np.inf], np.nan)
        returns = returns.fillna(method='ffill')
        
        return returns
    
    def _validate_data(self):
        """Perform comprehensive data quality validation"""
        
        if self.prices is None or self.prices.empty:
            self.data_quality['status'] = 'FAILED'
            self.data_quality['error'] = 'No price data available'
            return
        
        validation_results = {}
        
        for asset in self.prices.columns:
            asset_data = self.prices[asset].dropna()
            
            # Calculate data quality metrics
            if len(self.prices.index) == 0:
                total_days = 0
                observations = 0
                coverage = 0
            else:
                total_days = (self.prices.index[-1] - self.prices.index[0]).days
                observations = len(asset_data)
                coverage = observations / total_days if total_days > 0 else 0
            
            # Check for price continuity
            price_jumps = asset_data.pct_change().abs() > 0.2  # 20% daily moves
            suspicious_moves = price_jumps.sum()
            
            # Check for flat periods (potential stale data)
            flat_periods = (asset_data.diff() == 0).sum()
            
            validation_results[asset] = {
                'observations': observations,
                'coverage': coverage,
                'suspicious_moves': suspicious_moves,
                'flat_periods': flat_periods,
                'quality_score': self._calculate_quality_score(coverage, suspicious_moves, flat_periods)
            }
        
        self.data_quality = {
            'status': 'PASSED',
            'assets': validation_results,
            'overall_quality': np.mean([v['quality_score'] for v in validation_results.values()])
        }
        
        # Log data quality summary
        avg_quality = self.data_quality['overall_quality']
        logger.info(f"Data quality validation complete. Average quality score: {avg_quality:.2f}")
        
        # Warn about low-quality assets
        for asset, metrics in validation_results.items():
            if metrics['quality_score'] < 0.7:
                logger.warning(f"Low quality data for {asset}: score = {metrics['quality_score']:.2f}")
    
    def _calculate_quality_score(self, coverage: float, suspicious_moves: int, 
                                flat_periods: int) -> float:
        """Calculate overall data quality score (0-1)"""
        
        # Coverage score (0-1)
        coverage_score = min(coverage * 1.2, 1.0)  # Boost good coverage
        
        # Penalty for suspicious moves
        move_penalty = min(suspicious_moves * 0.02, 0.3)  # Cap penalty at 30%
        
        # Penalty for flat periods  
        flat_penalty = min(flat_periods * 0.001, 0.2)  # Cap penalty at 20%
        
        # Combined score
        quality_score = coverage_score - move_penalty - flat_penalty
        
        return max(quality_score, 0.0)  # Ensure non-negative
    
    def get_data_summary(self) -> Dict:
        """Get comprehensive data summary"""
        
        if self.prices is None:
            return {'status': 'No data loaded'}
        
        summary = {
            'assets': list(self.prices.columns),
            'date_range': {
                'start': self.prices.index[0].strftime('%Y-%m-%d'),
                'end': self.prices.index[-1].strftime('%Y-%m-%d'),
                'total_days': len(self.prices)
            },
            'data_quality': self.data_quality,
            'statistics': {
                'avg_daily_return': self.returns.mean().mean(),
                'avg_daily_volatility': self.returns.std().mean(),
                'correlation_range': {
                    'min': self.returns.corr().min().min(),
                    'max': self.returns.corr().max().max()
                }
            }
        }
        
        return summary
    
    def export_data(self, file_path: str, format: str = 'csv'):
        """Export data to file"""
        
        if format.lower() == 'csv':
            self.prices.to_csv(f"{file_path}_prices.csv")
            self.returns.to_csv(f"{file_path}_returns.csv")
            logger.info(f"Data exported to {file_path}_prices.csv and {file_path}_returns.csv")
        
        elif format.lower() == 'excel':
            with pd.ExcelWriter(f"{file_path}.xlsx") as writer:
                self.prices.to_excel(writer, sheet_name='Prices')
                self.returns.to_excel(writer, sheet_name='Returns')
                if self.volumes is not None:
                    self.volumes.to_excel(writer, sheet_name='Volumes')
            logger.info(f"Data exported to {file_path}.xlsx")
        
        else:
            raise ValueError(f"Unsupported export format: {format}")

# Asset Universe Definitions
class AssetUniverses:
    """Predefined asset universes for different strategies"""
    
    @staticmethod
    def diversified_etf() -> List[str]:
        """Diversified ETF universe across asset classes"""
        return [
            # US Equities
            'SPY', 'QQQ', 'IWM', 'VTI',
            # International Equities  
            'EFA', 'EEM', 'VEA', 'IEFA',
            # Fixed Income
            'TLT', 'IEF', 'LQD', 'HYG', 'TIP',
            # Commodities
            'GLD', 'SLV', 'DBC', 'USO',
            # REITs
            'VNQ', 'VNQI',
            # Currencies/Alternatives
            'UUP', 'FXE'
        ]
    
    @staticmethod
    def sector_rotation() -> List[str]:
        """Sector ETFs for sector rotation strategies"""
        return [
            'XLK',  # Technology
            'XLF',  # Financials
            'XLV',  # Healthcare
            'XLE',  # Energy
            'XLI',  # Industrials
            'XLY',  # Consumer Discretionary
            'XLP',  # Consumer Staples
            'XLB',  # Materials
            'XLU',  # Utilities
            'XLRE'  # Real Estate
        ]
    
    @staticmethod
    def global_equity() -> List[str]:
        """Global equity ETFs"""
        return [
            'VTI',  # US Total Market
            'VEA',  # Developed Markets
            'VWO',  # Emerging Markets
            'VGK',  # Europe
            'VPL',  # Pacific
            'IEMG', # Core Emerging Markets
            'IEFA', # Core Developed Markets
            'ACWI'  # All Country World
        ]
    
    @staticmethod
    def factor_investing() -> List[str]:
        """Factor-based ETFs"""
        return [
            'MTUM', # Momentum
            'QUAL', # Quality  
            'USMV', # Low Volatility
            'VLUE', # Value
            'SIZE', # Size
            'SPHQ', # High Quality
            'VMOT', # Alpha Momentum
            'FNDX'  # Fundamental Index
        ]