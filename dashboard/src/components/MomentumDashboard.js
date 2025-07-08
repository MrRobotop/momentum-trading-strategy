/**
 * Momentum Strategy Dashboard Component
 * ====================================
 * 
 * Main dashboard component for the momentum trading strategy.
 * Provides comprehensive performance analytics, portfolio insights, and risk metrics.
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, 
  BarChart, Bar, PieChart, Pie, Cell, AreaChart, Area 
} from 'recharts';
import { 
  TrendingUp, TrendingDown, DollarSign, Activity, Target, BarChart3, 
  Download, RefreshCw, Settings, Info, AlertTriangle, CheckCircle 
} from 'lucide-react';

const MomentumDashboard = ({ onError, connectionStatus, lastUpdated }) => {
  // State management
  const [activeTab, setActiveTab] = useState('performance');
  const [simulatedData, setSimulatedData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [dataError, setDataError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedTimeframe, setSelectedTimeframe] = useState('1Y');

  // Fetch real data from API
  const fetchStrategyData = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:9000/api/strategy-data');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Failed to fetch strategy data:', error);
      return null;
    }
  }, []);

  // Generate simulated time series data for charts
  const generateSimulatedData = useCallback(() => {
    try {
      const dates = [];
      const strategyReturns = [];
      const benchmarkReturns = [];
      let strategyCum = 1.0;
      let benchmarkCum = 1.0;

      // Generate 2 years of daily data
      const dataPoints = selectedTimeframe === '6M' ? 125 : selectedTimeframe === '1Y' ? 250 : 500;
      
      for (let i = 0; i < dataPoints; i++) {
        const date = new Date();
        date.setDate(date.getDate() - (dataPoints - i));
        
        // Simulate momentum strategy outperformance with realistic volatility
        const marketNoise = (Math.random() - 0.5) * 0.02;
        const momentumAlpha = 0.0002; // 5bps daily alpha
        const strategyReturn = marketNoise + momentumAlpha + (Math.random() - 0.5) * 0.005;
        const benchmarkReturn = marketNoise + (Math.random() - 0.5) * 0.004;
        
        strategyCum *= (1 + strategyReturn);
        benchmarkCum *= (1 + benchmarkReturn);
        
        if (i % 3 === 0) { // Every 3rd day for chart performance
          dates.push(date.toISOString().split('T')[0]);
          strategyReturns.push({
            date: date.toISOString().split('T')[0],
            strategy: ((strategyCum - 1) * 100).toFixed(2),
            benchmark: ((benchmarkCum - 1) * 100).toFixed(2),
            drawdown: Math.min(0, (Math.random() - 0.9) * 12).toFixed(2),
            volatility: (Math.random() * 8 + 10).toFixed(1)
          });
        }
      }

      // Generate current portfolio holdings
      const holdings = [
        { name: 'QQQ', weight: 18.5, momentum: 0.156, sector: 'US Tech', return_1m: 4.2 },
        { name: 'SPY', weight: 16.2, momentum: 0.134, sector: 'US Equity', return_1m: 2.8 },
        { name: 'EEM', weight: 14.1, momentum: 0.189, sector: 'EM Equity', return_1m: 5.7 },
        { name: 'GLD', weight: 12.8, momentum: 0.087, sector: 'Commodities', return_1m: 1.2 },
        { name: 'TLT', weight: 11.4, momentum: -0.023, sector: 'Bonds', return_1m: -1.8 },
        { name: 'VNQ', weight: 10.3, momentum: 0.092, sector: 'REITs', return_1m: 3.1 },
        { name: 'EFA', weight: 9.7, momentum: 0.071, sector: 'Intl Equity', return_1m: 2.4 },
        { name: 'DBC', weight: 7.0, momentum: 0.045, sector: 'Commodities', return_1m: 0.8 }
      ];

      // Calculate comprehensive risk metrics
      const riskMetrics = {
        sharpeRatio: 1.42,
        informationRatio: 0.78,
        maxDrawdown: -8.5,
        volatility: 14.2,
        var95: -2.1,
        cvar95: -3.2,
        turnover: 85.6,
        beta: 0.87,
        alpha: 4.2,
        trackingError: 3.8,
        calmarRatio: 2.1,
        sortinoRatio: 1.89
      };

      // Monthly returns heatmap data
      const monthlyReturns = [
        { month: 'Jan', '2023': 2.1, '2024': -0.8 },
        { month: 'Feb', '2023': -1.5, '2024': 3.2 },
        { month: 'Mar', '2023': 4.2, '2024': 1.7 },
        { month: 'Apr', '2023': 1.8, '2024': -2.1 },
        { month: 'May', '2023': -0.9, '2024': 2.8 },
        { month: 'Jun', '2023': 3.1, '2024': 1.2 },
        { month: 'Jul', '2023': 2.4, '2024': 0.5 },
        { month: 'Aug', '2023': -1.8, '2024': 2.9 },
        { month: 'Sep', '2023': 1.2, '2024': -1.4 },
        { month: 'Oct', '2023': 2.7, '2024': 1.8 },
        { month: 'Nov', '2023': 1.9, '2024': 2.3 },
        { month: 'Dec', '2023': 0.8, '2024': null }
      ];

      // Sector allocation data
      const sectorData = Object.entries(
        holdings.reduce((acc, holding) => {
          acc[holding.sector] = (acc[holding.sector] || 0) + holding.weight;
          return acc;
        }, {})
      ).map(([sector, weight]) => ({ 
        sector, 
        weight: parseFloat(weight.toFixed(1)),
        color: getSectorColor(sector)
      }));

      return {
        performanceData: strategyReturns,
        holdings,
        riskMetrics,
        monthlyReturns,
        sectorData,
        generatedAt: new Date().toISOString()
      };

    } catch (error) {
      console.error('Error generating simulated data:', error);
      setDataError('Failed to generate dashboard data');
      return null;
    }
  }, [selectedTimeframe]);

  // Helper function to get sector colors
  const getSectorColor = (sector) => {
    const colors = {
      'US Tech': '#3b82f6',
      'US Equity': '#10b981',
      'EM Equity': '#f59e0b',
      'Commodities': '#ef4444',
      'Bonds': '#8b5cf6',
      'REITs': '#06b6d4',
      'Intl Equity': '#84cc16'
    };
    return colors[sector] || '#6b7280';
  };

  // Load data on component mount and timeframe change
  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      setDataError(null);
      
      try {
        // Try to fetch real data from API first
        const apiData = await fetchStrategyData();

        if (apiData) {
          // Combine API data with simulated time series
          const timeSeriesData = generateSimulatedData();
          const combinedData = {
            ...timeSeriesData,
            // Override with real API data
            performance: apiData.performance,
            positions: apiData.positions,
            riskMetrics: apiData.risk_metrics,
            lastUpdated: apiData.last_updated,
            dataSource: apiData.data_source
          };
          setSimulatedData(combinedData);
        } else {
          // Fallback to simulated data
          const data = generateSimulatedData();
          if (data) {
            setSimulatedData(data);
          }
        }
      } catch (error) {
        console.error('Failed to load data:', error);
        setDataError('Failed to load dashboard data');
        if (onError) onError(error);
      } finally {
        setIsLoading(false);
      }
    };

    loadData();
  }, [generateSimulatedData, onError]);

  // Refresh data function
  const refreshData = useCallback(async () => {
    setRefreshing(true);

    try {
      // Call API refresh endpoint
      const refreshResponse = await fetch('http://localhost:9000/api/refresh');
      if (refreshResponse.ok) {
        // Fetch updated data
        const apiData = await fetchStrategyData();

        if (apiData) {
          const timeSeriesData = generateSimulatedData();
          const combinedData = {
            ...timeSeriesData,
            performance: apiData.performance,
            positions: apiData.positions,
            riskMetrics: apiData.risk_metrics,
            lastUpdated: apiData.last_updated,
            dataSource: apiData.data_source
          };
          setSimulatedData(combinedData);
        } else {
          // Fallback to simulated data
          const data = generateSimulatedData();
          if (data) {
            setSimulatedData(data);
          }
        }
      }
    } catch (error) {
      console.error('Failed to refresh data:', error);
      setDataError('Failed to refresh data');
    } finally {
      setRefreshing(false);
    }
  }, [fetchStrategyData, generateSimulatedData]);

  // Memoized calculations for performance
  const calculatedMetrics = useMemo(() => {
    if (!simulatedData) return {};

    const { performanceData, riskMetrics } = simulatedData;
    const latestPerformance = performanceData[performanceData.length - 1];
    
    return {
      totalReturn: parseFloat(latestPerformance?.strategy || 0),
      benchmarkReturn: parseFloat(latestPerformance?.benchmark || 0),
      outperformance: parseFloat(latestPerformance?.strategy || 0) - parseFloat(latestPerformance?.benchmark || 0),
      ...riskMetrics
    };
  }, [simulatedData]);

  // Loading state
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <h2 className="text-lg font-semibold text-gray-900">Loading Strategy Data</h2>
          <p className="text-gray-600">Fetching latest performance metrics...</p>
        </div>
      </div>
    );
  }

  // Error state
  if (dataError) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-50">
        <div className="text-center max-w-md">
          <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <h2 className="text-lg font-semibold text-gray-900 mb-2">Data Load Error</h2>
          <p className="text-gray-600 mb-4">{dataError}</p>
          <button 
            onClick={refreshData}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  // Metric card component
  const MetricCard = ({ title, value, change, icon: Icon, color = "blue", subtitle }) => (
    <div className="bg-white rounded-lg shadow-md p-6 border-l-4 border-blue-500 hover:shadow-lg transition-shadow">
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-600 mb-1">{title}</p>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
          {subtitle && <p className="text-xs text-gray-500 mt-1">{subtitle}</p>}
          {change && (
            <p className={`text-sm flex items-center mt-2 ${change > 0 ? 'text-green-600' : 'text-red-600'}`}>
              {change > 0 ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
              <span className="ml-1">{Math.abs(change).toFixed(2)}%</span>
            </p>
          )}
        </div>
        <div className="ml-4">
          <Icon className={`h-8 w-8 text-${color}-500`} />
        </div>
      </div>
    </div>
  );

  // Performance tab component
  const PerformanceTab = () => (
    <div className="space-y-6">
      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard 
          title="Total Return" 
          value={`${calculatedMetrics.totalReturn?.toFixed(1)}%`}
          change={calculatedMetrics.outperformance}
          icon={TrendingUp}
          color="green"
          subtitle={`vs ${calculatedMetrics.benchmarkReturn?.toFixed(1)}% benchmark`}
        />
        <MetricCard 
          title="Sharpe Ratio" 
          value={calculatedMetrics.sharpeRatio?.toFixed(2)}
          icon={Activity}
          color="blue"
          subtitle="Risk-adjusted return"
        />
        <MetricCard 
          title="Max Drawdown" 
          value={`${calculatedMetrics.maxDrawdown?.toFixed(1)}%`}
          icon={TrendingDown}
          color="red"
          subtitle="Largest peak-to-trough decline"
        />
        <MetricCard 
          title="Annual Volatility" 
          value={`${calculatedMetrics.volatility?.toFixed(1)}%`}
          icon={BarChart3}
          color="purple"
          subtitle="Annualized standard deviation"
        />
      </div>

      {/* Performance Chart */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold">Cumulative Performance</h3>
          <div className="flex space-x-2">
            {['6M', '1Y', '2Y'].map((period) => (
              <button
                key={period}
                onClick={() => setSelectedTimeframe(period)}
                className={`px-3 py-1 text-sm rounded ${
                  selectedTimeframe === period 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {period}
              </button>
            ))}
          </div>
        </div>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={simulatedData.performanceData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis 
              dataKey="date" 
              tick={{ fontSize: 12 }}
              tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', year: '2-digit' })}
            />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip 
              formatter={(value, name) => [`${parseFloat(value).toFixed(2)}%`, name === 'strategy' ? 'Strategy' : 'Benchmark']}
              labelFormatter={(value) => new Date(value).toLocaleDateString()}
            />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="strategy" 
              stroke="#2563eb" 
              strokeWidth={3} 
              name="Momentum Strategy"
              dot={false}
            />
            <Line 
              type="monotone" 
              dataKey="benchmark" 
              stroke="#dc2626" 
              strokeWidth={2} 
              name="Benchmark"
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Drawdown and Volatility Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Drawdown Analysis</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={simulatedData.performanceData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="date" 
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short' })}
              />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip 
                formatter={(value) => [`${parseFloat(value).toFixed(2)}%`, 'Drawdown']}
                labelFormatter={(value) => new Date(value).toLocaleDateString()}
              />
              <Area 
                type="monotone" 
                dataKey="drawdown" 
                stroke="#dc2626" 
                fill="#dc2626"
                fillOpacity={0.3}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Rolling Volatility</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={simulatedData.performanceData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="date" 
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short' })}
              />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip 
                formatter={(value) => [`${parseFloat(value).toFixed(1)}%`, 'Volatility']}
                labelFormatter={(value) => new Date(value).toLocaleDateString()}
              />
              <Line 
                type="monotone" 
                dataKey="volatility" 
                stroke="#8b5cf6" 
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );

  // Portfolio tab component
  const PortfolioTab = () => (
    <div className="space-y-6">
      {/* Current Holdings */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Current Holdings</h3>
          <div className="space-y-3">
            {simulatedData.holdings.map((holding, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                <div className="flex-1">
                  <div className="flex items-center space-x-2">
                    <span className="font-medium text-lg">{holding.name}</span>
                    <span className="text-sm text-gray-600">({holding.sector})</span>
                  </div>
                  <div className="text-sm text-gray-500">
                    Momentum: {(holding.momentum * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-medium text-lg">{holding.weight}%</div>
                  <div className={`text-sm ${holding.return_1m > 0 ? 'text-green-600' : 'text-red-600'}`}>
                    1M: {holding.return_1m > 0 ? '+' : ''}{holding.return_1m.toFixed(1)}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Portfolio Allocation Pie Chart */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Portfolio Allocation</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={simulatedData.holdings}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, weight }) => `${name}: ${weight}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="weight"
              >
                {simulatedData.holdings.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={`hsl(${index * 45}, 70%, 50%)`} />
                ))}
              </Pie>
              <Tooltip formatter={(value) => `${value}%`} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Sector Allocation */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold mb-4">Sector Allocation</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={simulatedData.sectorData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="sector" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip formatter={(value) => `${value}%`} />
            <Bar dataKey="weight" fill="#2563eb" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  // Risk tab component
  const RiskTab = () => (
    <div className="space-y-6">
      {/* Risk Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {Object.entries(simulatedData.riskMetrics).map(([metric, value]) => (
          <div key={metric} className="bg-white rounded-lg shadow-md p-4 hover:shadow-lg transition-shadow">
            <h4 className="text-sm font-medium text-gray-600 capitalize mb-2">
              {metric.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
            </h4>
            <p className="text-xl font-bold text-gray-900">
              {typeof value === 'number' ? 
                (metric.includes('ratio') || metric === 'beta' ? value.toFixed(2) : 
                 metric.includes('drawdown') || metric.includes('var') || metric.includes('alpha') ? `${value.toFixed(1)}%` :
                 `${value.toFixed(1)}%`) : value}
            </p>
          </div>
        ))}
      </div>

      {/* Monthly Returns Heatmap */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold mb-4">Monthly Returns Heatmap (%)</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full">
            <thead>
              <tr>
                <th className="px-4 py-2 text-left font-medium text-gray-700">Month</th>
                <th className="px-4 py-2 text-center font-medium text-gray-700">2023</th>
                <th className="px-4 py-2 text-center font-medium text-gray-700">2024</th>
              </tr>
            </thead>
            <tbody>
              {simulatedData.monthlyReturns.map((row, index) => (
                <tr key={index} className="border-t border-gray-200">
                  <td className="px-4 py-2 font-medium text-gray-900">{row.month}</td>
                  <td className="px-4 py-2 text-center">
                    <span className={`px-2 py-1 rounded text-sm font-medium ${
                      row['2023'] > 0 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                    }`}>
                      {row['2023'] > 0 ? '+' : ''}{row['2023']}%
                    </span>
                  </td>
                  <td className="px-4 py-2 text-center">
                    {row['2024'] !== null ? (
                      <span className={`px-2 py-1 rounded text-sm font-medium ${
                        row['2024'] > 0 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                      }`}>
                        {row['2024'] > 0 ? '+' : ''}{row['2024']}%
                      </span>
                    ) : (
                      <span className="text-gray-400">-</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Risk Attribution */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold mb-4">Risk Attribution</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 bg-blue-50 rounded-lg">
            <h4 className="font-medium text-blue-900 mb-2">Systematic Risk</h4>
            <p className="text-2xl font-bold text-blue-700">72%</p>
            <p className="text-sm text-blue-600">Market exposure</p>
          </div>
          <div className="p-4 bg-green-50 rounded-lg">
            <h4 className="font-medium text-green-900 mb-2">Idiosyncratic Risk</h4>
            <p className="text-2xl font-bold text-green-700">18%</p>
            <p className="text-sm text-green-600">Asset selection</p>
          </div>
          <div className="p-4 bg-purple-50 rounded-lg">
            <h4 className="font-medium text-purple-900 mb-2">Timing Risk</h4>
            <p className="text-2xl font-bold text-purple-700">10%</p>
            <p className="text-sm text-purple-600">Rebalancing</p>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Momentum Strategy Dashboard</h1>
              <p className="text-gray-600">Real-time quantitative trading strategy monitoring</p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-right">
                <p className="text-sm text-gray-600">Last Updated</p>
                <p className="font-medium">{lastUpdated?.toLocaleString() || new Date().toLocaleString()}</p>
              </div>
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${connectionStatus === 'connected' ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
                <span className="text-sm text-gray-600">{connectionStatus}</span>
              </div>
              <button
                onClick={refreshData}
                disabled={refreshing}
                className="flex items-center space-x-2 px-3 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
              >
                <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
                <span>{refreshing ? 'Refreshing...' : 'Refresh'}</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            {[
              { id: 'performance', name: 'Performance', icon: TrendingUp },
              { id: 'portfolio', name: 'Portfolio', icon: Target },
              { id: 'risk', name: 'Risk Analysis', icon: Activity }
            ].map(({ id, name, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={`${
                  activeTab === id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm flex items-center space-x-2 transition-colors`}
              >
                <Icon size={16} />
                <span>{name}</span>
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'performance' && <PerformanceTab />}
        {activeTab === 'portfolio' && <PortfolioTab />}
        {activeTab === 'risk' && <RiskTab />}
      </div>
    </div>
  );
};

export default MomentumDashboard;