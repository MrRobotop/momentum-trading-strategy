/**
 * React Application Entry Point
 * ============================
 * 
 * Main entry point for the Momentum Strategy Dashboard React application.
 * Handles React 18 root rendering and performance monitoring.
 */

import React from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';

// Error Boundary Component
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    // Log error details
    console.error('Dashboard Error Boundary caught an error:', error, errorInfo);
    
    this.setState({
      error: error,
      errorInfo: errorInfo
    });
  }

  render() {
    if (this.state.hasError) {
      // Fallback UI
      return (
        <div className="error-boundary-container">
          <div className="error-boundary-content">
            <h1 className="error-boundary-title">
              📊 Dashboard Error
            </h1>
            <p className="error-boundary-message">
              Something went wrong while loading the momentum strategy dashboard.
            </p>
            <details className="error-boundary-details">
              <summary>Technical Details</summary>
              <pre className="error-boundary-stack">
                {this.state.error && this.state.error.toString()}
                <br />
                {this.state.errorInfo.componentStack}
              </pre>
            </details>
            <button 
              className="error-boundary-retry"
              onClick={() => window.location.reload()}
            >
              Reload Dashboard
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Get the root element
const container = document.getElementById('root');

// Create React 18 root
const root = createRoot(container);

// Remove loading screen
const loadingElement = document.getElementById('loading');
if (loadingElement) {
  loadingElement.style.display = 'none';
}

// Render the application with error boundary
root.render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>
);

// Performance monitoring function
function sendToAnalytics(metric) {
  // Log performance metrics (in production, send to analytics service)
  console.log('Performance Metric:', metric);
  
  // Example: Send to Google Analytics, Mixpanel, etc.
  // analytics.track('web_vital', metric);
}

// Report web vitals for performance monitoring
reportWebVitals(sendToAnalytics);

// Service Worker Registration (optional)
if ('serviceWorker' in navigator && process.env.NODE_ENV === 'production') {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js')
      .then((registration) => {
        console.log('SW registered: ', registration);
      })
      .catch((registrationError) => {
        console.log('SW registration failed: ', registrationError);
      });
  });
}

// Global application configuration
window.DASHBOARD_CONFIG = {
  version: '1.0.0',
  buildDate: new Date().toISOString(),
  features: {
    realTimeData: true,
    exportFunctionality: true,
    advancedCharts: true,
    performanceMetrics: true
  }
};

// Console welcome message
console.log(`
%c🚀 Momentum Strategy Dashboard v${window.DASHBOARD_CONFIG.version}
%cBuilt with React 18 + Recharts for quantitative finance analytics
%cFor support: https://github.com/yourusername/momentum-trading-strategy

Performance Features Enabled:
✅ Real-time data visualization
✅ Interactive performance charts  
✅ Risk analytics dashboard
✅ Portfolio composition analysis
`, 
'color: #2563eb; font-size: 16px; font-weight: bold;',
'color: #64748b; font-size: 12px;',
'color: #16a34a; font-size: 12px;'
);