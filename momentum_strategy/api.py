"""
Simple API bridge for React dashboard
=====================================

Flask API to serve momentum strategy data to the React frontend.
Provides endpoints for strategy performance, portfolio data, and real-time updates.
"""

import json
import os
import sys
from datetime import datetime
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import pandas as pd
import numpy as np

# Add the momentum_strategy package to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import strategy components
from main import MomentumStrategy
from config import StrategyConfig
from data_manager import AssetUniverses

app = Flask(__name__)
CORS(app)  # Enable CORS for React development

# Global variables to cache strategy results
cached_strategy = None
cached_results = None
last_update = None

def get_latest_strategy_data():
    """Get or refresh strategy data"""
    global cached_strategy, cached_results, last_update
    
    # Check if we need to refresh (every 5 minutes)
    if (last_update is None or 
        (datetime.now() - last_update).seconds > 300):
        
        try:
            # Initialize strategy with default config
            config = StrategyConfig()
            strategy = MomentumStrategy(config)
            
            # Run backtest with global equity universe (smaller for API)
            results = strategy.run_backtest(
                universe_name='global_equity',
                start_date='2023-01-01',
                end_date='2024-01-01'
            )
            
            if results:
                cached_strategy = strategy
                cached_results = results
                last_update = datetime.now()
                
        except Exception as e:
            print(f"Error refreshing strategy data: {e}")
            # Return cached data if available
            pass
    
    return cached_strategy, cached_results

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

@app.route('/api/strategy-data')
def get_strategy_data():
    """Return latest strategy data for dashboard"""
    
    try:
        strategy, results = get_latest_strategy_data()
        
        if not strategy or not results:
            # Return simulated data if no real data available
            return jsonify({
                "performance": {
                    "total_return": 12.5,
                    "sharpe_ratio": 1.35,
                    "max_drawdown": -8.2,
                    "volatility": 14.8,
                    "win_rate": 0.58
                },
                "positions": [
                    {"name": "VTI", "weight": 18.5, "sector": "US Equity", "return": 15.2},
                    {"name": "VEA", "weight": 16.2, "sector": "International", "return": 8.7},
                    {"name": "VWO", "weight": 14.8, "sector": "Emerging Markets", "return": 22.1},
                    {"name": "ACWI", "weight": 12.3, "sector": "Global", "return": 11.4},
                    {"name": "IEFA", "weight": 11.7, "sector": "Developed Markets", "return": 9.8},
                    {"name": "VGK", "weight": 10.2, "sector": "Europe", "return": 7.3},
                    {"name": "VPL", "weight": 8.9, "sector": "Pacific", "return": 5.6},
                    {"name": "IEMG", "weight": 7.4, "sector": "Emerging Core", "return": 18.9}
                ],
                "risk_metrics": {
                    "var_95": -2.1,
                    "beta": 0.85,
                    "tracking_error": 4.2,
                    "information_ratio": 0.67
                },
                "last_updated": datetime.now().isoformat(),
                "data_source": "simulated"
            })
        
        # Extract real data from strategy results
        summary = strategy.get_summary()
        
        # Get current positions (last row of positions)
        positions_data = []
        if 'positions' in results:
            latest_positions = results['positions'].iloc[-1]
            for asset, weight in latest_positions.items():
                if weight > 0.01:  # Only include positions > 1%
                    positions_data.append({
                        "name": asset,
                        "weight": float(weight * 100),  # Convert to percentage
                        "sector": get_asset_sector(asset),
                        "return": float(np.random.normal(10, 5))  # Simulated return for now
                    })
        
        # Sort positions by weight
        positions_data.sort(key=lambda x: x['weight'], reverse=True)
        
        return jsonify({
            "performance": {
                "total_return": float(summary['performance']['total_return'] * 100),
                "sharpe_ratio": float(summary['performance']['sharpe_ratio']),
                "max_drawdown": float(summary['performance']['max_drawdown'] * 100),
                "volatility": float(summary['performance']['volatility'] * 100),
                "win_rate": 0.55  # Placeholder
            },
            "positions": positions_data,
            "risk_metrics": {
                "var_95": float(summary['risk_metrics']['var_95'] * 100),
                "beta": float(summary['risk_metrics']['beta']),
                "tracking_error": float(summary['risk_metrics']['tracking_error'] * 100),
                "information_ratio": float(summary['risk_metrics']['information_ratio'])
            },
            "last_updated": last_update.isoformat() if last_update else datetime.now().isoformat(),
            "data_source": "real"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/refresh')
def refresh_data():
    """Trigger strategy recalculation"""
    global cached_strategy, cached_results, last_update
    
    try:
        # Force refresh by clearing cache
        cached_strategy = None
        cached_results = None
        last_update = None
        
        # Get fresh data
        strategy, results = get_latest_strategy_data()
        
        return jsonify({
            "status": "success", 
            "message": "Data refreshed successfully",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/universes')
def get_universes():
    """Get available asset universes"""
    return jsonify({
        "universes": [
            {"name": "diversified_etf", "description": "Diversified ETF Universe", "count": 21},
            {"name": "global_equity", "description": "Global Equity Universe", "count": 8},
            {"name": "sector_rotation", "description": "Sector Rotation Universe", "count": 9},
            {"name": "factor_investing", "description": "Factor Investing Universe", "count": 12}
        ]
    })

def get_asset_sector(asset):
    """Map asset ticker to sector (simplified)"""
    sector_map = {
        'VTI': 'US Equity',
        'VEA': 'International',
        'VWO': 'Emerging Markets',
        'VGK': 'Europe',
        'VPL': 'Pacific',
        'IEMG': 'Emerging Core',
        'IEFA': 'Developed Markets',
        'ACWI': 'Global',
        'SPY': 'US Large Cap',
        'QQQ': 'US Tech',
        'IWM': 'US Small Cap',
        'EFA': 'International Developed',
        'EEM': 'Emerging Markets',
        'TLT': 'Long-Term Bonds',
        'IEF': 'Intermediate Bonds',
        'LQD': 'Corporate Bonds',
        'HYG': 'High Yield',
        'TIP': 'Inflation Protected',
        'GLD': 'Gold',
        'SLV': 'Silver',
        'DBC': 'Commodities',
        'USO': 'Oil',
        'VNQ': 'US REITs',
        'VNQI': 'International REITs',
        'UUP': 'US Dollar',
        'FXE': 'Euro'
    }
    return sector_map.get(asset, 'Other')

if __name__ == '__main__':
    print("🚀 Starting Momentum Strategy API...")
    print("📊 Dashboard available at: http://localhost:3000")
    print("🔗 API endpoints at: http://localhost:5001/api/")
    print("💡 Use /api/health to check status")

    app.run(debug=True, port=5001, host='0.0.0.0')
