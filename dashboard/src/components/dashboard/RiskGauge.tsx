import React from 'react';
import { ShieldAlert, ShieldCheck, Activity } from 'lucide-react';

interface RiskGaugeProps {
  probability: number; // 0.0 to 1.0
  level: string;
  uncertainty?: number; // 0.0 to 1.0
}

export const RiskGauge: React.FC<RiskGaugeProps> = ({ probability, level, uncertainty = 0.0 }) => {
  const percentage = Math.round(probability * 100);
  const uncPercentage = Math.round(uncertainty * 100);
  
  // Determine colors based on risk level
  const isHighRisk = probability > 0.7;
  const isWarning = probability > 0.4 && probability <= 0.7;
  
  const ringColor = isHighRisk 
    ? 'text-danger' 
    : isWarning 
      ? 'text-warning' 
      : 'text-success';
      
  const glowShadow = isHighRisk 
    ? 'drop-shadow-glow-danger' 
    : isWarning 
      ? 'drop-shadow-[0_0_15px_rgba(245,158,11,0.3)]' 
      : 'drop-shadow-[0_0_15px_rgba(16,185,129,0.3)]';

  const circumference = 2 * Math.PI * 45; // r=45
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  return (
    <div className="panel p-6 flex flex-col items-center justify-center relative min-h-[250px]">
      <div className="absolute top-4 left-4 flex items-center gap-2 text-muted text-sm font-medium tracking-wider uppercase">
        <Activity size={16} className={isHighRisk ? 'text-danger animate-pulse' : 'text-primary'} />
        Current Risk
      </div>

      <div className={`relative w-full max-w-[200px] aspect-square ${glowShadow} transition-all duration-500`}>
        <svg className="w-full h-full transform -rotate-90" viewBox="0 0 160 160">
          <circle
            cx="80"
            cy="80"
            r="45"
            className="stroke-white/5"
            strokeWidth="12"
            fill="none"
          />
          <circle
            cx="80"
            cy="80"
            r="45"
            className={`${ringColor} transition-all duration-1000 ease-in-out`}
            strokeWidth="12"
            fill="none"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
          />
        </svg>
        
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-3xl lg:text-4xl xl:text-5xl font-mono font-bold font-tracking-tighter text-white drop-shadow-md">
            {percentage}%
          </span>
          {uncPercentage > 0 && (
            <span className="text-xs font-mono text-muted/80 mt-1 whitespace-nowrap">
              ±{uncPercentage}% Unc.
            </span>
          )}
        </div>
      </div>

      <div className="mt-4 flex items-center gap-2">
        {isHighRisk ? (
          <ShieldAlert className="text-danger animate-bounce" size={24} />
        ) : (
          <ShieldCheck className="text-success" size={24} />
        )}
        <span className={`text-xl font-bold tracking-wide ${isHighRisk ? 'text-danger' : 'text-white'}`}>
          {level?.toUpperCase() || (isHighRisk ? 'CRITICAL' : 'SAFE')}
        </span>
      </div>
    </div>
  );
};
