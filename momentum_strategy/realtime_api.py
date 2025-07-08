"""
Real-time Data Streaming API
============================

WebSocket-based real-time data streaming for momentum strategy.
Provides live updates of strategy performance, positions, and market data.

Author: Rishabh Ashok Patil
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set
import websockets
from flask import Flask
from flask_socketio import SocketIO, emit, disconnect
from flask_cors import CORS
import threading
import time
import numpy as np
import pandas as pd

# Import strategy components
from main import MomentumStrategy
from config import StrategyConfig
from data_manager import AssetUniverses

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeDataStreamer:
    """Real-time data streaming service for momentum strategy"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'momentum_strategy_secret_key'
        CORS(self.app)
        
        self.socketio = SocketIO(
            self.app, 
            cors_allowed_origins="*",
            async_mode='threading'
        )
        
        # Data cache
        self.strategy_cache = {}
        self.connected_clients: Set[str] = set()
        self.streaming_active = False
        
        # Strategy instance
        self.strategy = None
        self.last_update = None
        
        # Set up event handlers
        self._setup_event_handlers()
        
        # Start background tasks
        self.background_thread = None
        
    def _setup_event_handlers(self):
        """Set up WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            client_id = str(id(threading.current_thread()))
            self.connected_clients.add(client_id)
            logger.info(f"Client {client_id} connected. Total clients: {len(self.connected_clients)}")
            
            # Send initial data
            if self.strategy_cache:
                emit('strategy_data', self.strategy_cache)
            
            # Start streaming if first client
            if len(self.connected_clients) == 1:
                self._start_streaming()
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            client_id = str(id(threading.current_thread()))
            self.connected_clients.discard(client_id)
            logger.info(f"Client {client_id} disconnected. Total clients: {len(self.connected_clients)}")
            
            # Stop streaming if no clients
            if len(self.connected_clients) == 0:
                self._stop_streaming()
        
        @self.socketio.on('request_refresh')
        def handle_refresh_request():
            """Handle manual refresh request from client"""
            logger.info("Manual refresh requested")
            self._refresh_strategy_data()
            
        @self.socketio.on('subscribe_to_updates')
        def handle_subscription(data):
            """Handle subscription to specific data types"""
            update_types = data.get('types', ['all'])
            logger.info(f"Client subscribed to updates: {update_types}")
            # Store subscription preferences (can be extended)
    
    def _start_streaming(self):
        """Start real-time data streaming"""
        if not self.streaming_active:
            self.streaming_active = True
            self.background_thread = self.socketio.start_background_task(self._streaming_worker)
            logger.info("Real-time streaming started")
    
    def _stop_streaming(self):
        """Stop real-time data streaming"""
        self.streaming_active = False
        logger.info("Real-time streaming stopped")
    
    def _streaming_worker(self):
        """Background worker for streaming data updates"""
        while self.streaming_active:
            try:
                # Update strategy data every 30 seconds
                self._refresh_strategy_data()
                
                # Simulate real-time price updates every 5 seconds
                self._simulate_price_updates()
                
                # Sleep for 5 seconds
                self.socketio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in streaming worker: {e}")
                self.socketio.sleep(10)  # Wait longer on error
    
    def _refresh_strategy_data(self):
        """Refresh strategy data and broadcast to clients"""
        try:
            # Check if we need to refresh (every 5 minutes for full refresh)
            if (self.last_update is None or 
                (datetime.now() - self.last_update).seconds > 300):
                
                logger.info("Refreshing strategy data...")
                
                # Initialize strategy if needed
                if self.strategy is None:
                    config = StrategyConfig()
                    self.strategy = MomentumStrategy(config)
                
                # Run quick backtest with smaller universe
                results = self.strategy.run_backtest(
                    universe_name='global_equity',
                    start_date='2023-01-01',
                    end_date='2024-01-01'
                )
                
                if results:
                    summary = self.strategy.get_summary()
                    
                    # Update cache
                    self.strategy_cache = {
                        'timestamp': datetime.now().isoformat(),
                        'performance': {
                            'total_return': float(summary['performance']['total_return'] * 100),
                            'sharpe_ratio': float(summary['performance']['sharpe_ratio']),
                            'max_drawdown': float(summary['performance']['max_drawdown'] * 100),
                            'volatility': float(summary['performance']['volatility'] * 100),
                            'annual_return': float(summary['performance']['annual_return'] * 100)
                        },
                        'risk_metrics': {
                            'var_95': float(summary['risk_metrics']['var_95'] * 100),
                            'beta': float(summary['risk_metrics']['beta']),
                            'tracking_error': float(summary['risk_metrics']['tracking_error'] * 100),
                            'information_ratio': float(summary['risk_metrics']['information_ratio'])
                        },
                        'portfolio_stats': {
                            'avg_turnover': float(summary['portfolio_characteristics']['avg_turnover'] * 100),
                            'avg_positions': float(summary['portfolio_characteristics']['avg_positions']),
                            'max_position': float(summary['portfolio_characteristics']['max_position'] * 100)
                        }
                    }
                    
                    # Add current positions
                    if 'positions' in results:
                        latest_positions = results['positions'].iloc[-1]
                        positions_data = []
                        for asset, weight in latest_positions.items():
                            if weight > 0.01:  # Only positions > 1%
                                positions_data.append({
                                    'name': asset,
                                    'weight': float(weight * 100),
                                    'sector': self._get_asset_sector(asset)
                                })
                        
                        positions_data.sort(key=lambda x: x['weight'], reverse=True)
                        self.strategy_cache['positions'] = positions_data
                    
                    self.last_update = datetime.now()
                    
                    # Broadcast to all clients
                    self.socketio.emit('strategy_data', self.strategy_cache)
                    logger.info("Strategy data refreshed and broadcasted")
                
        except Exception as e:
            logger.error(f"Error refreshing strategy data: {e}")
    
    def _simulate_price_updates(self):
        """Simulate real-time price updates for demonstration"""
        try:
            # Generate simulated price changes
            price_updates = {}
            
            if 'positions' in self.strategy_cache:
                for position in self.strategy_cache['positions'][:5]:  # Top 5 positions
                    # Simulate price change (-2% to +2%)
                    price_change = np.random.normal(0, 0.005)  # 0.5% std dev
                    price_updates[position['name']] = {
                        'price_change': float(price_change * 100),
                        'timestamp': datetime.now().isoformat()
                    }
            
            if price_updates:
                self.socketio.emit('price_updates', price_updates)
                
        except Exception as e:
            logger.error(f"Error simulating price updates: {e}")
    
    def _get_asset_sector(self, asset):
        """Map asset ticker to sector"""
        sector_map = {
            'VTI': 'US Equity', 'VEA': 'International', 'VWO': 'Emerging Markets',
            'VGK': 'Europe', 'VPL': 'Pacific', 'IEMG': 'Emerging Core',
            'IEFA': 'Developed Markets', 'ACWI': 'Global', 'SPY': 'US Large Cap',
            'QQQ': 'US Tech', 'IWM': 'US Small Cap', 'EFA': 'International Developed',
            'EEM': 'Emerging Markets', 'TLT': 'Long-Term Bonds', 'IEF': 'Intermediate Bonds',
            'LQD': 'Corporate Bonds', 'HYG': 'High Yield', 'TIP': 'Inflation Protected',
            'GLD': 'Gold', 'SLV': 'Silver', 'DBC': 'Commodities', 'USO': 'Oil',
            'VNQ': 'US REITs', 'VNQI': 'International REITs', 'UUP': 'US Dollar', 'FXE': 'Euro'
        }
        return sector_map.get(asset, 'Other')
    
    def run(self, host='0.0.0.0', port=9001, debug=False):
        """Run the real-time streaming server"""
        logger.info(f"🚀 Starting Real-time Data Streaming Server...")
        logger.info(f"📡 WebSocket server: ws://{host}:{port}")
        logger.info(f"👤 Author: Rishabh Ashok Patil")
        
        self.socketio.run(
            self.app,
            host=host,
            port=port,
            debug=debug,
            allow_unsafe_werkzeug=True
        )

if __name__ == '__main__':
    streamer = RealTimeDataStreamer()
    streamer.run(debug=True)
