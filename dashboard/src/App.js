/**
 * Main Application Component
 * =========================
 * 
 * Root component for the Momentum Strategy Dashboard.
 * Handles application-level state, routing, and layout.
 */

import React, { useState, useEffect, useCallback } from 'react';
import MomentumDashboard from './components/MomentumDashboard';
import AdvancedAnalytics from './components/AdvancedAnalytics';
import useWebSocket from './hooks/useWebSocket';
import './App.css';

// App-level configuration
const APP_CONFIG = {
  title: 'Momentum Strategy Dashboard',
  version: '1.0.0',
  refreshInterval: 30000, // 30 seconds
  maxRetries: 3
};

function App() {
  // Application state
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('connected');
  const [lastUpdated, setLastUpdated] = useState(new Date());
  const [activeTab, setActiveTab] = useState('dashboard');

  // WebSocket connection
  const {
    isConnected: wsConnected,
    strategyData: wsStrategyData,
    connectionError: wsError,
    requestRefresh
  } = useWebSocket('http://localhost:9001', {
    autoConnect: true,
    maxReconnectAttempts: 5
  });

  // Initialize application
  useEffect(() => {
    const initializeApp = async () => {
      try {
        // Simulate initialization delay
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Check system requirements
        checkSystemRequirements();
        
        // Set up performance monitoring
        setupPerformanceMonitoring();
        
        // Mark as loaded
        setIsLoading(false);
        setLastUpdated(new Date());
        
        console.log('✅ Dashboard initialized successfully');
        
      } catch (err) {
        console.error('❌ Failed to initialize dashboard:', err);
        setError(err.message);
        setIsLoading(false);
      }
    };

    initializeApp();
  }, []);

  // Check browser compatibility and system requirements
  const checkSystemRequirements = useCallback(() => {
    const requirements = {
      localStorage: typeof(Storage) !== 'undefined',
      fetch: typeof fetch !== 'undefined',
      promises: typeof Promise !== 'undefined',
      es6: typeof Symbol !== 'undefined'
    };

    const missingFeatures = Object.entries(requirements)
      .filter(([feature, supported]) => !supported)
      .map(([feature]) => feature);

    if (missingFeatures.length > 0) {
      throw new Error(`Unsupported browser. Missing features: ${missingFeatures.join(', ')}`);
    }

    // Check for minimum screen size
    if (window.innerWidth < 320) {
      console.warn('⚠️ Small screen detected. Dashboard may have limited functionality.');
    }
  }, []);

  // Set up performance monitoring
  const setupPerformanceMonitoring = useCallback(() => {
    // Monitor memory usage
    if ('memory' in performance) {
      const checkMemory = () => {
        const memory = performance.memory;
        const memoryUsage = {
          used: Math.round(memory.usedJSHeapSize / 1048576), // MB
          total: Math.round(memory.totalJSHeapSize / 1048576), // MB
          limit: Math.round(memory.jsHeapSizeLimit / 1048576) // MB
        };
        
        // Warn if memory usage is high
        if (memoryUsage.used / memoryUsage.limit > 0.8) {
          console.warn('⚠️ High memory usage detected:', memoryUsage);
        }
      };

      // Check memory every 5 minutes
      const memoryInterval = setInterval(checkMemory, 300000);
      
      // Cleanup on unmount
      return () => clearInterval(memoryInterval);
    }

    // Monitor network status
    const updateOnlineStatus = () => {
      setConnectionStatus(navigator.onLine ? 'connected' : 'disconnected');
    };

    window.addEventListener('online', updateOnlineStatus);
    window.addEventListener('offline', updateOnlineStatus);

    // Cleanup event listeners
    return () => {
      window.removeEventListener('online', updateOnlineStatus);
      window.removeEventListener('offline', updateOnlineStatus);
    };
  }, []);

  // Handle application errors
  const handleError = useCallback((error, errorInfo) => {
    console.error('Application Error:', error, errorInfo);
    setError(error.message);
    
    // In production, you might want to send this to an error reporting service
    // errorReportingService.captureException(error, { extra: errorInfo });
  }, []);

  // Retry functionality for failed operations
  const handleRetry = useCallback(() => {
    setError(null);
    setIsLoading(true);
    window.location.reload();
  }, []);

  // Loading screen
  if (isLoading) {
    return (
      <div className="app-loading">
        <div className="loading-content">
          <div className="loading-spinner"></div>
          <h2 className="loading-title">Loading Strategy Dashboard</h2>
          <p className="loading-subtitle">Initializing quantitative analytics...</p>
          <div className="loading-progress">
            <div className="loading-bar"></div>
          </div>
        </div>
      </div>
    );
  }

  // Error screen
  if (error) {
    return (
      <div className="app-error">
        <div className="error-content">
          <div className="error-icon">⚠️</div>
          <h2 className="error-title">Dashboard Error</h2>
          <p className="error-message">{error}</p>
          <div className="error-actions">
            <button 
              className="btn btn-primary" 
              onClick={handleRetry}
            >
              Retry
            </button>
            <button 
              className="btn btn-secondary" 
              onClick={() => window.location.href = 'mailto:support@example.com'}
            >
              Contact Support
            </button>
          </div>
          <details className="error-details">
            <summary>Technical Information</summary>
            <pre className="error-stack">
              User Agent: {navigator.userAgent}
              {'\n'}Timestamp: {new Date().toISOString()}
              {'\n'}Screen: {window.screen.width}x{window.screen.height}
              {'\n'}Viewport: {window.innerWidth}x{window.innerHeight}
            </pre>
          </details>
        </div>
      </div>
    );
  }

  // Main application
  return (
    <div className="app">
      {/* Status bar for development */}
      {process.env.NODE_ENV === 'development' && (
        <div className="status-bar">
          <span className={`status-indicator ${connectionStatus}`}>
            {connectionStatus === 'connected' ? '🟢' : '🔴'} {connectionStatus}
          </span>
          <span className="last-updated">
            Last updated: {lastUpdated.toLocaleTimeString()}
          </span>
          <span className="version">
            v{APP_CONFIG.version}
          </span>
        </div>
      )}

      {/* Navigation Tabs */}
      <nav className="app-nav">
        <div className="nav-container">
          <button
            onClick={() => setActiveTab('dashboard')}
            className={`nav-tab ${activeTab === 'dashboard' ? 'active' : ''}`}
          >
            📊 Dashboard
          </button>
          <button
            onClick={() => setActiveTab('analytics')}
            className={`nav-tab ${activeTab === 'analytics' ? 'active' : ''}`}
          >
            🧠 Advanced Analytics
          </button>
          <div className="nav-status">
            <span className={`status-indicator ${wsConnected ? 'connected' : 'disconnected'}`}>
              {wsConnected ? '🟢 Live' : '🔴 Offline'}
            </span>
          </div>
        </div>
      </nav>

      {/* Main dashboard */}
      <main className="app-main">
        {activeTab === 'dashboard' && (
          <MomentumDashboard
            onError={handleError}
            connectionStatus={wsConnected ? 'connected' : 'disconnected'}
            lastUpdated={lastUpdated}
            strategyData={wsStrategyData}
          />
        )}
        {activeTab === 'analytics' && (
          <AdvancedAnalytics
            strategyData={wsStrategyData}
            isConnected={wsConnected}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <div className="footer-content">
          <p className="footer-text">
            © 2024 Momentum Strategy Dashboard by Rishabh Ashok Patil. Built with React & Recharts for quantitative finance.
          </p>
          <div className="footer-links">
            <a 
              href="https://github.com/MrRobotop/momentum-trading-strategy"
              target="_blank" 
              rel="noopener noreferrer"
              className="footer-link"
            >
              GitHub
            </a>
            <a 
              href="mailto:support@example.com" 
              className="footer-link"
            >
              Support
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;