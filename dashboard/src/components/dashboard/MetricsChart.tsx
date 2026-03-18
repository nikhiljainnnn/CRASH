import React from 'react';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer 
} from 'recharts';
import { Clock } from 'lucide-react';

interface MetricsChartProps {
  data: Array<{ time: string; latency: number }>;
}

export const MetricsChart: React.FC<MetricsChartProps> = ({ data }) => {
  return (
    <div className="panel p-6 flex flex-col min-h-[300px]">
      <div className="flex items-center gap-2 text-muted text-sm font-medium tracking-wider uppercase mb-6">
        <Clock size={16} className="text-primary" />
        Inference Latency (ms)
      </div>

      <div className="flex-1 w-full relative">
        {data.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="colorLatency" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
              <XAxis 
                dataKey="time" 
                stroke="#52525b" 
                fontSize={12} 
                tickLine={false} 
                axisLine={false}
              />
              <YAxis 
                stroke="#52525b" 
                fontSize={12} 
                tickLine={false} 
                axisLine={false} 
                domain={[0, 'dataMax + 20']}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#111214', 
                  borderColor: '#27272a',
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.5)'
                }}
                itemStyle={{ color: '#f4f4f5' }}
              />
              <Area 
                type="monotone" 
                dataKey="latency" 
                stroke="#3b82f6" 
                strokeWidth={2}
                fillOpacity={1} 
                fill="url(#colorLatency)" 
                isAnimationActive={false} // Disable to make real-time updates smoother
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="absolute inset-0 flex items-center justify-center text-muted/50 font-mono text-sm">
            WAITING FOR METRICS...
          </div>
        )}
      </div>
    </div>
  );
};
