import { useState, useEffect, useCallback, useRef } from 'react';

const WS_URL = 'ws://localhost:8080/ws';
const RECONNECT_DELAY = 2000;

export function useWebSocket() {
    // Only keep minimal state for UI indicators (connection status, errors)
    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState(null);

    // Ref-based data storage - does NOT trigger React re-renders
    const dataRef = useRef(null);
    const wsRef = useRef(null);
    const reconnectTimeoutRef = useRef(null);

    // Callback registry for direct canvas updates
    const subscribersRef = useRef(new Set());

    // Subscribe to raw data updates (bypasses React render cycle)
    const subscribe = useCallback((callback) => {
        subscribersRef.current.add(callback);
        return () => subscribersRef.current.delete(callback);
    }, []);

    // Get current data ref (for components that need it)
    const getDataRef = useCallback(() => dataRef, []);

    const connect = useCallback(() => {
        try {
            console.log('[WS] Connecting to', WS_URL);
            const ws = new WebSocket(WS_URL);

            ws.onopen = () => {
                console.log('[WS] Connected');
                setIsConnected(true);
                setError(null);
            };

            ws.onmessage = (event) => {
                try {
                    const parsed = JSON.parse(event.data);
                    // Store in ref (no React re-render)
                    dataRef.current = parsed;
                    // Notify all subscribers directly
                    subscribersRef.current.forEach(callback => callback(parsed));
                } catch (e) {
                    console.error('[WS] Parse error:', e);
                }
            };

            ws.onerror = (event) => {
                console.error('[WS] Error:', event);
                setError('Connection error');
            };

            ws.onclose = () => {
                console.log('[WS] Disconnected');
                setIsConnected(false);
                wsRef.current = null;

                // Auto-reconnect
                reconnectTimeoutRef.current = setTimeout(() => {
                    console.log('[WS] Reconnecting...');
                    connect();
                }, RECONNECT_DELAY);
            };

            wsRef.current = ws;
        } catch (e) {
            console.error('[WS] Connection failed:', e);
            setError('Failed to connect');
        }
    }, []);

    const disconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
        }
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
    }, []);

    const sendMessage = useCallback((message) => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(message);
        }
    }, []);

    useEffect(() => {
        connect();
        return () => disconnect();
    }, [connect, disconnect]);

    return {
        isConnected,
        error,
        sendMessage,
        reconnect: connect,
        // New APIs for direct canvas access
        subscribe,
        getDataRef
    };
}

export default useWebSocket;
