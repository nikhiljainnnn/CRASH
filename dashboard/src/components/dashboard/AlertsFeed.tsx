import React from 'react';
import { ShieldCheck, AlertTriangle, Clock, MapPin, Video } from 'lucide-react';

interface Alert {
  risk_level: string;
  crash_probability: number;
  message?: string;
  source?: string;
  timestamp?: number;
}

interface AlertsFeedProps {
  alerts: Alert[];
}

export const AlertsFeed: React.FC<AlertsFeedProps> = ({ alerts }) => {
  return (
    <div className="panel col-span-1 row-span-2 flex flex-col h-full overflow-hidden">
      <div className="p-4 border-b border-border bg-black/20 flex items-center justify-between sticky top-0 z-10">
        <div className="flex items-center gap-2 font-mono text-sm tracking-wider uppercase text-muted">
          <AlertTriangle size={16} className="text-warning" />
          Event Log
        </div>
        <span className="bg-primary/20 text-primary text-xs px-2 py-1 rounded-full font-bold">
          {alerts.length} EVENTS
        </span>
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-2 relative">
        {alerts.length === 0 ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-muted/50 p-6 text-center">
             <ShieldCheck size={48} className="mb-4 opacity-50" />
             <p className="font-mono text-sm uppercase tracking-wider mb-2">No Anomalies Detected</p>
             <p className="text-xs">System operating within normal parameters</p>
          </div>
        ) : (
          alerts.map((alert, i) => {
            const isHighRisk = alert.risk_level.toLowerCase() === 'critical' || alert.risk_level.toLowerCase() === 'high';
            const riskColor = isHighRisk ? 'text-danger' : 'text-warning';
            const riskBg = isHighRisk ? 'bg-danger/10 border-danger/30' : 'bg-warning/10 border-warning/30';
            
            // Format timestamp if available, otherwise just mock a time for the visual feed
            const timeString = alert.timestamp 
              ? new Date(alert.timestamp * 1000).toLocaleTimeString([], { hour12: false })
              : new Date().toLocaleTimeString([], { hour12: false });

            return (
              <div 
                key={i} 
                className={`p-4 rounded-lg border ${riskBg} transition-all duration-300 hover:bg-white/5 animate-fade-in`}
                style={{ animationFillMode: 'both', animationDelay: `${i * 50}ms` }}
              >
                <div className="flex justify-between items-start mb-2">
                  <div className={`font-bold tracking-wide flex items-center gap-2 ${riskColor}`}>
                    <AlertTriangle size={16} />
                    {alert.risk_level.toUpperCase()}
                  </div>
                  <div className="text-xs text-muted font-mono flex items-center gap-1">
                    <Clock size={12} />
                    {timeString}
                  </div>
                </div>
                
                <p className="text-sm text-text/90 mb-3">
                  {alert.message || `Crash probability reached ${(alert.crash_probability * 100).toFixed(1)}%`}
                </p>

                <div className="flex items-center gap-3 text-xs text-muted/80">
                  <span className="flex items-center gap-1">
                    <Video size={12} />
                    {alert.source || 'CAM_01'}
                  </span>
                  <span className="flex items-center gap-1">
                    <MapPin size={12} />
                    Sector 7G
                  </span>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
