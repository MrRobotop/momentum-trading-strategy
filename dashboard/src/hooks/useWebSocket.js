/**
 * WebSocket Hook for Real-time Data
 * =================================
 * 
 * Custom React hook for managing WebSocket connections and real-time data streaming.
 * Provides automatic reconnection, error handling, and data synchronization.
 * 
 * Author: Rishabh Ashok Patil
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import io from 'socket.io-client';

const useWebSocket = (url = 'http://localhost:9001', options = {}) => {
  // State management
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState(null);
  const [lastMessage, setLastMessage] = useState(null);
  const [strategyData, setStrategyData] = useState(null);
  const [priceUpdates, setPriceUpdates] = useState({});
  
  // Refs
  const socketRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const reconnectAttempts = useRef(0);
  
  // Configuration
  const maxReconnectAttempts = options.maxReconnectAttempts || 5;
  const reconnectInterval = options.reconnectInterval || 3000;
  
  // Connect to WebSocket
  const connect = useCallback(() => {
    try {
      console.log('🔌 Connecting to WebSocket:', url);
      
      socketRef.current = io(url, {
        transports: ['websocket', 'polling'],
        timeout: 10000,
        forceNew: true,
        ...options.socketOptions
      });
      
      // Connection event handlers
      socketRef.current.on('connect', () => {
        console.log('✅ WebSocket connected');
        setIsConnected(true);
        setConnectionError(null);
        reconnectAttempts.current = 0;
        
        // Subscribe to updates
        socketRef.current.emit('subscribe_to_updates', {
          types: ['strategy_data', 'price_updates', 'alerts']
        });
      });
      
      socketRef.current.on('disconnect', (reason) => {
        console.log('❌ WebSocket disconnected:', reason);
        setIsConnected(false);
        
        // Attempt reconnection if not manually disconnected
        if (reason !== 'io client disconnect') {
          scheduleReconnect();
        }
      });
      
      socketRef.current.on('connect_error', (error) => {
        console.error('🚫 WebSocket connection error:', error);
        setConnectionError(error.message);
        setIsConnected(false);
        scheduleReconnect();
      });
      
      // Data event handlers
      socketRef.current.on('strategy_data', (data) => {
        console.log('📊 Received strategy data:', data);
        setStrategyData(data);
        setLastMessage({
          type: 'strategy_data',
          data,
          timestamp: new Date().toISOString()
        });
      });
      
      socketRef.current.on('price_updates', (data) => {
        console.log('💰 Received price updates:', data);
        setPriceUpdates(prevUpdates => ({
          ...prevUpdates,
          ...data
        }));
        setLastMessage({
          type: 'price_updates',
          data,
          timestamp: new Date().toISOString()
        });
      });
      
      socketRef.current.on('alert', (data) => {
        console.log('🚨 Received alert:', data);
        setLastMessage({
          type: 'alert',
          data,
          timestamp: new Date().toISOString()
        });
      });
      
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setConnectionError(error.message);
    }
  }, [url, options.socketOptions]);
  
  // Schedule reconnection
  const scheduleReconnect = useCallback(() => {
    if (reconnectAttempts.current < maxReconnectAttempts) {
      reconnectAttempts.current += 1;
      console.log(`🔄 Scheduling reconnection attempt ${reconnectAttempts.current}/${maxReconnectAttempts}`);
      
      reconnectTimeoutRef.current = setTimeout(() => {
        connect();
      }, reconnectInterval * reconnectAttempts.current);
    } else {
      console.error('❌ Max reconnection attempts reached');
      setConnectionError('Failed to reconnect after multiple attempts');
    }
  }, [connect, maxReconnectAttempts, reconnectInterval]);
  
  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }
    
    setIsConnected(false);
    reconnectAttempts.current = 0;
  }, []);
  
  // Send message
  const sendMessage = useCallback((event, data) => {
    if (socketRef.current && isConnected) {
      socketRef.current.emit(event, data);
      return true;
    } else {
      console.warn('Cannot send message: WebSocket not connected');
      return false;
    }
  }, [isConnected]);
  
  // Request data refresh
  const requestRefresh = useCallback(() => {
    return sendMessage('request_refresh', { timestamp: new Date().toISOString() });
  }, [sendMessage]);
  
  // Initialize connection on mount
  useEffect(() => {
    if (options.autoConnect !== false) {
      connect();
    }
    
    // Cleanup on unmount
    return () => {
      disconnect();
    };
  }, [connect, disconnect, options.autoConnect]);
  
  // Heartbeat to maintain connection
  useEffect(() => {
    if (!isConnected) return;
    
    const heartbeatInterval = setInterval(() => {
      if (socketRef.current) {
        socketRef.current.emit('ping');
      }
    }, 30000); // 30 seconds
    
    return () => clearInterval(heartbeatInterval);
  }, [isConnected]);
  
  return {
    // Connection state
    isConnected,
    connectionError,
    
    // Data
    strategyData,
    priceUpdates,
    lastMessage,
    
    // Actions
    connect,
    disconnect,
    sendMessage,
    requestRefresh,
    
    // Connection info
    reconnectAttempts: reconnectAttempts.current,
    maxReconnectAttempts
  };
};

export default useWebSocket;
