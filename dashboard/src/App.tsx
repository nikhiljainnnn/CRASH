import { useState, useEffect } from 'react';
import { BrainCircuit, Activity, Settings2 } from 'lucide-react';
import { LiveFeed } from './components/dashboard/LiveFeed';
import { RiskGauge } from './components/dashboard/RiskGauge';
import { MetricsChart } from './components/dashboard/MetricsChart';
import { AlertsFeed } from './components/dashboard/AlertsFeed';
import { SettingsModal } from './components/dashboard/SettingsModal';
import { useLiveMetrics, useLiveAlerts } from './hooks/useApi';
import type { PredictionData } from './hooks/useApi';

// Extend PredictionData locally if the hook doesn't have it yet
interface ExtendedPredictionData extends PredictionData {
  uncertainty?: number;
}

// Initial chart data point
const createEmptyChartData = () => Array.from({ length: 20 }, (_, i) => ({ 
  time: new Date(Date.now() - (19 - i) * 1000).toLocaleTimeString([], { hour12: false }), 
  latency: 0 
}));

function App() {
  const { metrics, isConnected } = useLiveMetrics(1000);
  const { alerts } = useLiveAlerts(2000);
  
  const [chartData, setChartData] = useState(createEmptyChartData());
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  
  // Real-time Mock Prediction Data (Used to drive the UI until the actual webcam/video feed is connected)
  const [currentPrediction, setCurrentPrediction] = useState<ExtendedPredictionData>({
    crash_probability: metrics?.avg_crash_probability || 0,
    risk_level: 'Low',
    risk_score: 0,
    num_vehicles: 0,
    latency_ms: metrics?.avg_latency_ms || 0,
    timestamp: Date.now() / 1000,
    uncertainty: metrics?.avg_uncertainty || 0
  });

  // Track latency for the chart
  useEffect(() => {
    if (metrics) {
      setChartData(prev => {
        const newData = [...prev.slice(1), {
          time: new Date().toLocaleTimeString([], { hour12: false }),
          latency: metrics.avg_latency_ms || Math.random() * 20 + 40 // fallback if 0
        }];
        return newData;
      });
      
      setCurrentPrediction(prev => ({
        ...prev,
        crash_probability: metrics.avg_crash_probability,
        latency_ms: metrics.avg_latency_ms,
        risk_level: metrics.avg_crash_probability > 0.7 ? 'Critical' : metrics.avg_crash_probability > 0.4 ? 'Medium' : 'Low',
        uncertainty: metrics.avg_uncertainty || 0
      }));
    }
  }, [metrics]);

  const isHighRisk = currentPrediction.crash_probability > 0.7;

  return (
    <div className="min-h-screen bg-background flex flex-col font-sans selection:bg-primary/30">
      {/* Top Navbar */}
      <nav className="h-16 border-b border-white/5 bg-black/60 backdrop-blur-xl flex items-center justify-between px-4 lg:px-6 sticky top-0 z-50 shadow-2xl">
        <div className="flex items-center gap-3">
          <div className="bg-primary/10 p-2 rounded-xl border border-primary/20 relative group cursor-pointer transition-all hover:bg-primary/20">
            <div className="absolute inset-0 bg-primary/20 blur-md rounded-xl z-0 group-hover:blur-lg transition-all" />
            <BrainCircuit className="text-primary relative z-10" size={24} />
          </div>
          <div>
            <h1 className="text-lg lg:text-xl font-extrabold tracking-tight text-white m-0 leading-tight">CRASH AI</h1>
            <p className="text-[10px] lg:text-xs font-mono text-muted tracking-widest uppercase opacity-80">Visual Intelligence Network</p>
          </div>
        </div>
        
        <div className="flex items-center gap-4 lg:gap-6">
          <div className="flex items-center gap-2">
            <span className="hidden lg:block text-xs font-mono text-muted uppercase">System Status</span>
            <div className={`px-3 py-1.5 rounded-full text-[10px] lg:text-xs font-bold font-mono tracking-wider flex items-center gap-2 transition-colors ${isConnected ? 'bg-success/10 text-success border border-success/20 shadow-[0_0_15px_rgba(16,185,129,0.15)]' : 'bg-danger/10 text-danger border border-danger/20 shadow-[0_0_15px_rgba(244,63,94,0.15)]'}`}>
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-success animate-pulse' : 'bg-danger'}`} />
              {isConnected ? 'ONLINE' : 'OFFLINE'}
            </div>
          </div>
          
          <button 
            onClick={() => setIsSettingsOpen(true)}
            className="text-muted hover:text-white transition-all bg-white/5 p-2 rounded-xl border border-white/10 hover:border-white/30 hover:bg-white/10 active:scale-95"
          >
            <Settings2 size={20} />
          </button>
        </div>
      </nav>

      <main className="flex-1 p-4 lg:p-6 overflow-x-hidden">
        <div className="max-w-[1800px] mx-auto grid grid-cols-1 xl:grid-cols-4 gap-4 lg:gap-6 lg:h-[calc(100vh-8rem)]">
          
          {/* Left/Main Column - Live Feed & Key Metrics */}
          <div className="xl:col-span-3 flex flex-col gap-4 lg:gap-6 h-full">
            {/* Live Camera Feed */}
            <div className="flex-none lg:flex-1 min-h-[300px] lg:min-h-0">
              <LiveFeed 
                streamUrl="http://localhost:8000/video_feed" 
                isHighRisk={isHighRisk} 
              />
            </div>
            
            {/* Bottom Row - Metrics & Gauges */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 lg:gap-6 h-auto lg:h-[35%] shrink-0">
              {/* Real-time Risk Gauge */}
              <div className="col-span-1 rounded-2xl bg-black/40 border border-white/5 shadow-2xl backdrop-blur-md overflow-hidden flex flex-col relative">
                <RiskGauge 
                  probability={currentPrediction.crash_probability} 
                  level={currentPrediction.risk_level}
                  uncertainty={currentPrediction.uncertainty}
                />
              </div>
              
              {/* Quick Stats Panel */}
              <div className="col-span-1 rounded-2xl bg-black/40 border border-white/5 shadow-2xl backdrop-blur-md p-5 lg:p-6 flex flex-col justify-center relative overflow-hidden group">
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-primary/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                <div className="flex items-center gap-2 text-muted text-xs lg:text-sm font-medium tracking-wider uppercase mb-6">
                  <Activity size={16} className="text-primary" />
                  Pipeline Stats
                </div>
                
                <div className="space-y-4 lg:space-y-5">
                  <div className="flex justify-between items-center border-b border-white/5 pb-2">
                    <span className="text-muted text-sm">Total Predictions</span>
                    <span className="font-mono text-white text-base lg:text-lg">{metrics?.total_predictions.toLocaleString() || '0'}</span>
                  </div>
                  <div className="flex justify-between items-center border-b border-white/5 pb-2">
                    <span className="text-muted text-sm">Inference Latency</span>
                    <span className="font-mono text-primary text-base lg:text-lg drop-shadow-[0_0_8px_rgba(56,189,248,0.5)]">{metrics?.avg_latency_ms.toFixed(1) || '0'} ms</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-muted text-sm">Active Alerts</span>
                    <span className={`font-mono font-bold text-base lg:text-lg ${metrics?.total_alerts ? 'text-danger drop-shadow-[0_0_8px_rgba(244,63,94,0.5)]' : 'text-success'}`}>
                      {metrics?.total_alerts || '0'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Bottom Chart */}
              <div className="col-span-1 md:col-span-3 xl:col-span-1 rounded-2xl bg-black/40 border border-white/5 shadow-2xl backdrop-blur-md overflow-hidden relative min-h-[200px]">
                <MetricsChart data={chartData} />
              </div>
            </div>
          </div>

          {/* Right Column - Alert Feed */}
          <div className="xl:col-span-1 h-[500px] xl:h-full rounded-2xl bg-black/40 border border-white/5 shadow-2xl backdrop-blur-md flex flex-col overflow-hidden">
            <AlertsFeed alerts={alerts} />
          </div>
          
        </div>
      </main>

      {/* Modals & Overlays */}
      <SettingsModal 
        isOpen={isSettingsOpen} 
        onClose={() => setIsSettingsOpen(false)} 
      />
    </div>
  );
}

export default App;
