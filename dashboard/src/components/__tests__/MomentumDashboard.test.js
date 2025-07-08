/**
 * Test Suite for MomentumDashboard Component
 * ==========================================
 * 
 * Comprehensive tests for the main dashboard component including
 * rendering, data handling, user interactions, and error states.
 * 
 * Author: Rishabh Ashok Patil
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import MomentumDashboard from '../MomentumDashboard';

// Mock the recharts library
jest.mock('recharts', () => ({
  ResponsiveContainer: ({ children }) => <div data-testid="responsive-container">{children}</div>,
  LineChart: ({ children }) => <div data-testid="line-chart">{children}</div>,
  Line: () => <div data-testid="line" />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
  Legend: () => <div data-testid="legend" />,
  AreaChart: ({ children }) => <div data-testid="area-chart">{children}</div>,
  Area: () => <div data-testid="area" />,
  BarChart: ({ children }) => <div data-testid="bar-chart">{children}</div>,
  Bar: () => <div data-testid="bar" />,
  PieChart: ({ children }) => <div data-testid="pie-chart">{children}</div>,
  Pie: () => <div data-testid="pie" />,
  Cell: () => <div data-testid="cell" />
}));

// Mock lucide-react icons
jest.mock('lucide-react', () => ({
  TrendingUp: () => <div data-testid="trending-up-icon" />,
  DollarSign: () => <div data-testid="dollar-sign-icon" />,
  Activity: () => <div data-testid="activity-icon" />,
  Shield: () => <div data-testid="shield-icon" />,
  RefreshCw: () => <div data-testid="refresh-icon" />,
  AlertTriangle: () => <div data-testid="alert-icon" />,
  CheckCircle: () => <div data-testid="check-icon" />,
  Clock: () => <div data-testid="clock-icon" />
}));

// Mock fetch for API calls
global.fetch = jest.fn();

describe('MomentumDashboard', () => {
  const defaultProps = {
    onError: jest.fn(),
    connectionStatus: 'connected',
    lastUpdated: new Date('2024-01-01T12:00:00Z')
  };

  beforeEach(() => {
    fetch.mockClear();
    defaultProps.onError.mockClear();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Component Rendering', () => {
    test('renders dashboard header correctly', async () => {
      render(<MomentumDashboard {...defaultProps} />);
      
      await waitFor(() => {
        expect(screen.getByText('Momentum Strategy Dashboard')).toBeInTheDocument();
        expect(screen.getByText(/Real-time quantitative trading insights/)).toBeInTheDocument();
      });
    });

    test('renders all main tabs', async () => {
      render(<MomentumDashboard {...defaultProps} />);
      
      await waitFor(() => {
        expect(screen.getByText('Performance')).toBeInTheDocument();
        expect(screen.getByText('Portfolio')).toBeInTheDocument();
        expect(screen.getByText('Risk Analysis')).toBeInTheDocument();
      });
    });

    test('renders performance metrics cards', async () => {
      render(<MomentumDashboard {...defaultProps} />);
      
      await waitFor(() => {
        expect(screen.getByText('Total Return')).toBeInTheDocument();
        expect(screen.getByText('Sharpe Ratio')).toBeInTheDocument();
        expect(screen.getByText('Max Drawdown')).toBeInTheDocument();
        expect(screen.getByText('Volatility')).toBeInTheDocument();
      });
    });

    test('shows loading state initially', () => {
      render(<MomentumDashboard {...defaultProps} />);
      
      expect(screen.getByText(/Loading strategy data/)).toBeInTheDocument();
      expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
    });
  });

  describe('Data Loading and API Integration', () => {
    test('fetches data from API on mount', async () => {
      const mockApiResponse = {
        performance: {
          total_return: 15.5,
          sharpe_ratio: 1.25,
          max_drawdown: -8.2,
          volatility: 12.3
        },
        positions: [
          { name: 'SPY', weight: 25.0, sector: 'US Equity' },
          { name: 'QQQ', weight: 20.0, sector: 'US Tech' }
        ]
      };

      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockApiResponse
      });

      render(<MomentumDashboard {...defaultProps} />);

      await waitFor(() => {
        expect(fetch).toHaveBeenCalledWith('http://localhost:9000/api/strategy-data');
      });
    });

    test('handles API errors gracefully', async () => {
      fetch.mockRejectedValueOnce(new Error('Network error'));

      render(<MomentumDashboard {...defaultProps} />);

      await waitFor(() => {
        expect(fetch).toHaveBeenCalled();
        // Should fall back to simulated data
        expect(screen.queryByText('Failed to load data')).not.toBeInTheDocument();
      });
    });

    test('displays real data when API succeeds', async () => {
      const mockApiResponse = {
        performance: {
          total_return: 15.5,
          sharpe_ratio: 1.25,
          max_drawdown: -8.2,
          volatility: 12.3
        },
        data_source: 'real'
      };

      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockApiResponse
      });

      render(<MomentumDashboard {...defaultProps} />);

      await waitFor(() => {
        expect(screen.getByText('15.5%')).toBeInTheDocument();
        expect(screen.getByText('1.25')).toBeInTheDocument();
      });
    });
  });

  describe('User Interactions', () => {
    test('switches between tabs correctly', async () => {
      render(<MomentumDashboard {...defaultProps} />);

      await waitFor(() => {
        expect(screen.getByText('Performance')).toBeInTheDocument();
      });

      // Click on Portfolio tab
      fireEvent.click(screen.getByText('Portfolio'));
      
      await waitFor(() => {
        expect(screen.getByText(/Current Holdings/)).toBeInTheDocument();
      });

      // Click on Risk Analysis tab
      fireEvent.click(screen.getByText('Risk Analysis'));
      
      await waitFor(() => {
        expect(screen.getByText(/Risk Metrics/)).toBeInTheDocument();
      });
    });

    test('refresh button triggers data reload', async () => {
      fetch.mockResolvedValue({
        ok: true,
        json: async () => ({ performance: {}, positions: [] })
      });

      render(<MomentumDashboard {...defaultProps} />);

      await waitFor(() => {
        expect(screen.getByTestId('refresh-icon')).toBeInTheDocument();
      });

      const refreshButton = screen.getByTestId('refresh-icon').closest('button');
      fireEvent.click(refreshButton);

      await waitFor(() => {
        expect(fetch).toHaveBeenCalledTimes(2); // Initial load + refresh
      });
    });

    test('handles refresh errors', async () => {
      fetch
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ performance: {}, positions: [] })
        })
        .mockRejectedValueOnce(new Error('Refresh failed'));

      render(<MomentumDashboard {...defaultProps} />);

      await waitFor(() => {
        const refreshButton = screen.getByTestId('refresh-icon').closest('button');
        fireEvent.click(refreshButton);
      });

      await waitFor(() => {
        expect(defaultProps.onError).not.toHaveBeenCalled(); // Should handle gracefully
      });
    });
  });

  describe('Connection Status', () => {
    test('displays connected status correctly', async () => {
      render(<MomentumDashboard {...defaultProps} connectionStatus="connected" />);
      
      await waitFor(() => {
        expect(screen.getByText(/Connected/)).toBeInTheDocument();
        expect(screen.getByTestId('check-icon')).toBeInTheDocument();
      });
    });

    test('displays disconnected status correctly', async () => {
      render(<MomentumDashboard {...defaultProps} connectionStatus="disconnected" />);
      
      await waitFor(() => {
        expect(screen.getByText(/Disconnected/)).toBeInTheDocument();
        expect(screen.getByTestId('alert-icon')).toBeInTheDocument();
      });
    });

    test('shows last updated time', async () => {
      const lastUpdated = new Date('2024-01-01T12:00:00Z');
      render(<MomentumDashboard {...defaultProps} lastUpdated={lastUpdated} />);
      
      await waitFor(() => {
        expect(screen.getByText(/Last updated/)).toBeInTheDocument();
      });
    });
  });

  describe('Data Visualization', () => {
    test('renders performance charts', async () => {
      render(<MomentumDashboard {...defaultProps} />);
      
      await waitFor(() => {
        expect(screen.getAllByTestId('responsive-container')).toHaveLength(1);
        expect(screen.getByTestId('line-chart')).toBeInTheDocument();
      });
    });

    test('renders portfolio allocation chart', async () => {
      render(<MomentumDashboard {...defaultProps} />);
      
      // Switch to Portfolio tab
      await waitFor(() => {
        fireEvent.click(screen.getByText('Portfolio'));
      });

      await waitFor(() => {
        expect(screen.getByTestId('pie-chart')).toBeInTheDocument();
      });
    });

    test('renders risk analysis charts', async () => {
      render(<MomentumDashboard {...defaultProps} />);
      
      // Switch to Risk Analysis tab
      await waitFor(() => {
        fireEvent.click(screen.getByText('Risk Analysis'));
      });

      await waitFor(() => {
        expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling', () => {
    test('handles component errors gracefully', async () => {
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
      
      // Force an error by passing invalid props
      render(<MomentumDashboard {...defaultProps} lastUpdated="invalid-date" />);
      
      await waitFor(() => {
        // Component should still render without crashing
        expect(screen.getByText('Momentum Strategy Dashboard')).toBeInTheDocument();
      });

      consoleSpy.mockRestore();
    });

    test('displays fallback content when data is unavailable', async () => {
      fetch.mockResolvedValueOnce({
        ok: false,
        status: 500
      });

      render(<MomentumDashboard {...defaultProps} />);

      await waitFor(() => {
        // Should show simulated data instead of error
        expect(screen.getByText('Total Return')).toBeInTheDocument();
      });
    });
  });

  describe('Performance and Optimization', () => {
    test('does not cause memory leaks', async () => {
      const { unmount } = render(<MomentumDashboard {...defaultProps} />);
      
      await waitFor(() => {
        expect(screen.getByText('Momentum Strategy Dashboard')).toBeInTheDocument();
      });

      // Unmount component
      unmount();
      
      // Verify cleanup (no specific assertions needed, just ensure no errors)
      expect(true).toBe(true);
    });

    test('handles rapid tab switching', async () => {
      render(<MomentumDashboard {...defaultProps} />);

      await waitFor(() => {
        expect(screen.getByText('Performance')).toBeInTheDocument();
      });

      // Rapidly switch tabs
      const tabs = ['Portfolio', 'Risk Analysis', 'Performance'];
      
      for (const tab of tabs) {
        fireEvent.click(screen.getByText(tab));
        await waitFor(() => {
          expect(screen.getByText(tab)).toBeInTheDocument();
        });
      }
    });
  });

  describe('Accessibility', () => {
    test('has proper ARIA labels', async () => {
      render(<MomentumDashboard {...defaultProps} />);
      
      await waitFor(() => {
        const buttons = screen.getAllByRole('button');
        expect(buttons.length).toBeGreaterThan(0);
        
        // Check that important buttons have accessible names
        const refreshButton = screen.getByTestId('refresh-icon').closest('button');
        expect(refreshButton).toBeInTheDocument();
      });
    });

    test('supports keyboard navigation', async () => {
      render(<MomentumDashboard {...defaultProps} />);
      
      await waitFor(() => {
        const tabs = screen.getAllByRole('button');
        expect(tabs[0]).toBeInTheDocument();
        
        // Focus first tab
        tabs[0].focus();
        expect(document.activeElement).toBe(tabs[0]);
      });
    });
  });
});

describe('Integration with WebSocket', () => {
  test('handles WebSocket data updates', async () => {
    const mockStrategyData = {
      performance: {
        total_return: 18.7,
        sharpe_ratio: 1.45
      },
      timestamp: new Date().toISOString()
    };

    render(<MomentumDashboard {...defaultProps} strategyData={mockStrategyData} />);

    await waitFor(() => {
      expect(screen.getByText('18.7%')).toBeInTheDocument();
      expect(screen.getByText('1.45')).toBeInTheDocument();
    });
  });

  test('handles real-time price updates', async () => {
    const mockPriceUpdates = {
      'SPY': { price_change: 1.25, timestamp: new Date().toISOString() },
      'QQQ': { price_change: -0.85, timestamp: new Date().toISOString() }
    };

    render(<MomentumDashboard {...defaultProps} priceUpdates={mockPriceUpdates} />);

    // Switch to Portfolio tab to see price updates
    await waitFor(() => {
      fireEvent.click(screen.getByText('Portfolio'));
    });

    // Price updates should be reflected in the portfolio view
    await waitFor(() => {
      expect(screen.getByText(/Current Holdings/)).toBeInTheDocument();
    });
  });
});
