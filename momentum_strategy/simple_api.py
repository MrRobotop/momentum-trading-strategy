"""
Simple API for testing React connection
"""

from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

@app.route('/api/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

@app.route('/api/strategy-data')
def get_strategy_data():
    return jsonify({
        "performance": {
            "total_return": 24.8,
            "sharpe_ratio": 1.42,
            "max_drawdown": -8.5,
            "volatility": 14.2,
            "win_rate": 0.58
        },
        "positions": [
            {"name": "QQQ", "weight": 18.5, "sector": "US Tech", "return": 28.3},
            {"name": "SPY", "weight": 16.2, "sector": "US Equity", "return": 15.7},
            {"name": "VTI", "weight": 14.8, "sector": "US Total Market", "return": 16.1},
            {"name": "EFA", "weight": 12.3, "sector": "International", "return": 8.9},
            {"name": "VEA", "weight": 11.7, "sector": "Developed Markets", "return": 9.2},
            {"name": "EEM", "weight": 10.2, "sector": "Emerging Markets", "return": 22.4},
            {"name": "IWM", "weight": 8.9, "sector": "US Small Cap", "return": 12.6},
            {"name": "TLT", "weight": 7.4, "sector": "Long-Term Bonds", "return": -5.2}
        ],
        "risk_metrics": {
            "var_95": -2.1,
            "beta": 0.92,
            "tracking_error": 3.8,
            "information_ratio": 0.74
        },
        "last_updated": datetime.now().isoformat(),
        "data_source": "simulated"
    })

@app.route('/api/refresh')
def refresh_data():
    return jsonify({
        "status": "success", 
        "message": "Data refreshed successfully",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Starting Simple API on port 9000...")
    app.run(debug=True, port=9000, host='0.0.0.0')
