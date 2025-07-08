"""
Advanced Analytics Module
========================

Advanced analytics and machine learning features for momentum strategy.
Includes factor analysis, regime detection, and predictive modeling.

Author: Rishabh Ashok Patil
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedAnalytics:
    """Advanced analytics for momentum strategy"""
    
    def __init__(self, results: Dict, config):
        self.results = results
        self.config = config
        self.scaler = StandardScaler()
        
    def perform_factor_analysis(self) -> Dict:
        """Perform factor analysis on returns"""
        logger.info("Performing factor analysis...")
        
        try:
            returns = self.results.get('returns')
            if returns is None or returns.empty:
                return {"error": "No returns data available"}
            
            # Standardize returns
            returns_scaled = self.scaler.fit_transform(returns.fillna(0))
            
            # Principal Component Analysis
            pca = PCA(n_components=min(5, returns.shape[1]))
            factors = pca.fit_transform(returns_scaled)
            
            # Create factor loadings
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=[f'Factor_{i+1}' for i in range(pca.n_components_)],
                index=returns.columns
            )
            
            # Factor returns
            factor_returns = pd.DataFrame(
                factors,
                columns=[f'Factor_{i+1}' for i in range(pca.n_components_)],
                index=returns.index
            )
            
            # Calculate factor statistics
            factor_stats = {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'factor_loadings': loadings.to_dict(),
                'factor_returns': factor_returns.to_dict('records')[:100]  # Limit for JSON
            }
            
            return factor_stats
            
        except Exception as e:
            logger.error(f"Error in factor analysis: {e}")
            return {"error": str(e)}
    
    def detect_market_regimes(self) -> Dict:
        """Detect market regimes using clustering"""
        logger.info("Detecting market regimes...")
        
        try:
            strategy_returns = self.results.get('strategy_returns')
            if strategy_returns is None or strategy_returns.empty:
                return {"error": "No strategy returns available"}
            
            # Create features for regime detection
            features = self._create_regime_features(strategy_returns)
            
            # Standardize features
            features_scaled = self.scaler.fit_transform(features)
            
            # K-means clustering for regime detection
            n_regimes = 3  # Bull, Bear, Sideways
            kmeans = KMeans(n_clusters=n_regimes, random_state=42)
            regimes = kmeans.fit_predict(features_scaled)
            
            # Analyze regimes
            regime_analysis = self._analyze_regimes(strategy_returns, regimes, features)
            
            return regime_analysis
            
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return {"error": str(e)}
    
    def _create_regime_features(self, returns: pd.Series) -> pd.DataFrame:
        """Create features for regime detection"""
        features = pd.DataFrame(index=returns.index)
        
        # Rolling statistics
        windows = [20, 60, 120]
        for window in windows:
            features[f'volatility_{window}'] = returns.rolling(window).std()
            features[f'return_{window}'] = returns.rolling(window).mean()
            features[f'skewness_{window}'] = returns.rolling(window).skew()
            features[f'kurtosis_{window}'] = returns.rolling(window).kurt()
        
        # Technical indicators
        features['momentum_12_1'] = returns.rolling(252).sum() / returns.rolling(21).sum()
        features['volatility_ratio'] = (returns.rolling(20).std() / 
                                       returns.rolling(60).std())
        
        return features.dropna()
    
    def _analyze_regimes(self, returns: pd.Series, regimes: np.ndarray, 
                        features: pd.DataFrame) -> Dict:
        """Analyze detected regimes"""
        regime_stats = {}
        
        for regime in np.unique(regimes):
            mask = regimes == regime
            regime_returns = returns.iloc[features.index[mask]]
            
            regime_stats[f'regime_{regime}'] = {
                'count': int(mask.sum()),
                'percentage': float(mask.mean() * 100),
                'avg_return': float(regime_returns.mean() * 252),  # Annualized
                'volatility': float(regime_returns.std() * np.sqrt(252)),
                'sharpe_ratio': float((regime_returns.mean() * 252) / 
                                    (regime_returns.std() * np.sqrt(252))),
                'max_drawdown': float(self._calculate_max_drawdown(regime_returns))
            }
        
        return {
            'regimes': regime_stats,
            'regime_labels': regimes.tolist()[:100],  # Limit for JSON
            'total_periods': len(regimes)
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    def build_predictive_model(self) -> Dict:
        """Build predictive model for returns"""
        logger.info("Building predictive model...")
        
        try:
            returns = self.results.get('returns')
            if returns is None or returns.empty:
                return {"error": "No returns data available"}
            
            # Create features and targets
            features, targets = self._create_ml_features(returns)
            
            if features.empty or targets.empty:
                return {"error": "Insufficient data for modeling"}
            
            # Split data
            split_idx = int(len(features) * 0.8)
            X_train, X_test = features[:split_idx], features[split_idx:]
            y_train, y_test = targets[:split_idx], targets[split_idx:]
            
            # Train Random Forest model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return {
                'model_performance': {
                    'train_r2': float(train_r2),
                    'test_r2': float(test_r2),
                    'train_mse': float(train_mse),
                    'test_mse': float(test_mse)
                },
                'feature_importance': feature_importance.head(10).to_dict('records'),
                'predictions': {
                    'dates': X_test.index.strftime('%Y-%m-%d').tolist()[:50],
                    'actual': y_test.tolist()[:50],
                    'predicted': y_pred_test.tolist()[:50]
                }
            }
            
        except Exception as e:
            logger.error(f"Error in predictive modeling: {e}")
            return {"error": str(e)}
    
    def _create_ml_features(self, returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Create features for machine learning"""
        features = pd.DataFrame(index=returns.index)
        
        # Technical features
        for col in returns.columns[:5]:  # Limit to first 5 assets
            col_returns = returns[col]
            
            # Momentum features
            features[f'{col}_momentum_5'] = col_returns.rolling(5).mean()
            features[f'{col}_momentum_20'] = col_returns.rolling(20).mean()
            features[f'{col}_momentum_60'] = col_returns.rolling(60).mean()
            
            # Volatility features
            features[f'{col}_vol_20'] = col_returns.rolling(20).std()
            features[f'{col}_vol_60'] = col_returns.rolling(60).std()
            
            # RSI-like feature
            delta = col_returns.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            features[f'{col}_rsi'] = 100 - (100 / (1 + gain / loss))
        
        # Market-wide features
        market_returns = returns.mean(axis=1)
        features['market_momentum'] = market_returns.rolling(20).mean()
        features['market_volatility'] = market_returns.rolling(20).std()
        
        # Target: next period strategy return
        strategy_returns = self.results.get('strategy_returns', market_returns)
        targets = strategy_returns.shift(-1)  # Next period return
        
        # Align and clean
        common_index = features.index.intersection(targets.index)
        features = features.loc[common_index].dropna()
        targets = targets.loc[features.index].dropna()
        
        return features, targets
    
    def calculate_risk_attribution(self) -> Dict:
        """Calculate risk attribution analysis"""
        logger.info("Calculating risk attribution...")
        
        try:
            positions = self.results.get('positions')
            returns = self.results.get('returns')
            
            if positions is None or returns is None:
                return {"error": "Missing positions or returns data"}
            
            # Calculate portfolio risk contributions
            risk_contributions = {}
            
            # Get latest positions
            latest_positions = positions.iloc[-1]
            
            # Calculate individual asset volatilities
            asset_vols = returns.std() * np.sqrt(252)  # Annualized
            
            # Calculate correlations
            correlation_matrix = returns.corr()
            
            # Risk contribution calculation
            portfolio_vol = (latest_positions @ correlation_matrix @ latest_positions) ** 0.5
            
            for asset in latest_positions.index:
                if latest_positions[asset] > 0.001:  # Only significant positions
                    # Marginal contribution to risk
                    marginal_contrib = (correlation_matrix.loc[asset] @ latest_positions) / portfolio_vol
                    risk_contrib = latest_positions[asset] * marginal_contrib
                    
                    risk_contributions[asset] = {
                        'position_weight': float(latest_positions[asset]),
                        'individual_volatility': float(asset_vols[asset]),
                        'risk_contribution': float(risk_contrib),
                        'risk_contribution_pct': float(risk_contrib / portfolio_vol * 100)
                    }
            
            return {
                'portfolio_volatility': float(portfolio_vol * np.sqrt(252)),
                'risk_contributions': risk_contributions,
                'diversification_ratio': float(
                    (latest_positions * asset_vols).sum() / (portfolio_vol * np.sqrt(252))
                )
            }
            
        except Exception as e:
            logger.error(f"Error in risk attribution: {e}")
            return {"error": str(e)}
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive advanced analytics report"""
        logger.info("Generating comprehensive advanced analytics report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'author': 'Rishabh Ashok Patil',
            'analytics': {}
        }
        
        # Run all analytics
        analytics_functions = [
            ('factor_analysis', self.perform_factor_analysis),
            ('regime_detection', self.detect_market_regimes),
            ('predictive_model', self.build_predictive_model),
            ('risk_attribution', self.calculate_risk_attribution)
        ]
        
        for name, func in analytics_functions:
            try:
                logger.info(f"Running {name}...")
                result = func()
                report['analytics'][name] = result
            except Exception as e:
                logger.error(f"Error in {name}: {e}")
                report['analytics'][name] = {"error": str(e)}
        
        return report
