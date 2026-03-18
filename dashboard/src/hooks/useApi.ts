import { useState, useEffect } from 'react';

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

export interface PredictionData {
  crash_probability: number;
  risk_level: 'Critical' | 'High' | 'Medium' | 'Low';
  risk_score: number;
  num_vehicles: number;
  latency_ms: number;
  timestamp: number;
  alert?: {
    type: string;
    message: string;
    risk_level: string;
  };
}

export interface SystemMetrics {
  total_predictions: number;
  total_alerts: number;
  avg_latency_ms: number;
  avg_crash_probability: number;
  avg_uncertainty: number;
}

export function useLiveMetrics(pollingInterval = 1000) {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    let mounted = true;

    async function fetchMetrics() {
      try {
        const response = await fetch(`${API_BASE_URL}/metrics`);
        if (!response.ok) throw new Error('Failed to fetch metrics');
        
        const data = await response.json();
        if (mounted) {
          setMetrics(data);
          setIsConnected(true);
          setError(null);
        }
      } catch (err) {
        if (mounted) {
          setError(err instanceof Error ? err.message : 'Unknown error');
          setIsConnected(false);
        }
      }
    }

    // Initial fetch
    fetchMetrics();

    // Poll for updates
    const interval = setInterval(fetchMetrics, pollingInterval);
    
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [pollingInterval]);

  return { metrics, isConnected, error };
}

export function useLiveAlerts(pollingInterval = 2000) {
  const [alerts, setAlerts] = useState<any[]>([]);
  
  useEffect(() => {
    let mounted = true;

    async function fetchAlerts() {
      try {
        const response = await fetch(`${API_BASE_URL}/alerts?limit=10`);
        if (!response.ok) return;
        
        const data = await response.json();
        if (mounted && data.alerts) {
          setAlerts(data.alerts.reverse()); // Show newest first
        }
      } catch (err) {
        // Silent fail for alerts poll
      }
    }

    fetchAlerts();
    const interval = setInterval(fetchAlerts, pollingInterval);
    
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [pollingInterval]);

  return { alerts };
}
