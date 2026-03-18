import React, { useRef, useState, useEffect } from 'react';
import { Camera, Maximize2, Minimize2, AlertTriangle } from 'lucide-react';

interface LiveFeedProps {
  streamUrl?: string;
  isHighRisk: boolean;
}

export const LiveFeed: React.FC<LiveFeedProps> = ({ streamUrl, isHighRisk }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      containerRef.current?.requestFullscreen().catch(err => {
        console.error(`Error attempting to enable fullscreen: ${err.message}`);
      });
    } else {
      document.exitFullscreen();
    }
  };

  return (
    <div 
      ref={containerRef}
      className={`panel ${isFullscreen ? 'fixed inset-0 z-50 rounded-none bg-black' : 'col-span-2 flex-1 rounded-2xl relative bg-black/40'} overflow-hidden group transition-all duration-500 ${isHighRisk ? 'ring-2 ring-danger shadow-[0_0_30px_rgba(244,63,94,0.4)]' : 'border border-white/10 shadow-2xl backdrop-blur-md'}`}
    >
      
      {/* Feed Header */}
      <div className="absolute top-0 left-0 right-0 z-10 p-4 lg:p-6 bg-gradient-to-b from-black/90 via-black/50 to-transparent flex justify-between items-start pointer-events-none">
        <div className="flex items-center gap-3">
          <div className="relative flex h-3 w-3 mt-1">
            <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${isHighRisk ? 'bg-danger' : 'bg-primary'}`}></span>
            <span className={`relative inline-flex rounded-full h-3 w-3 ${isHighRisk ? 'bg-danger' : 'bg-primary'}`}></span>
          </div>
          <div>
            <div className="text-white font-mono text-sm tracking-wider font-bold drop-shadow-md">LIVE FEED :: CAM_01</div>
            <div className="text-[10px] text-white/50 font-mono tracking-widest mt-1">MAIN INTERSECTION</div>
          </div>
        </div>
        
        <button 
          onClick={toggleFullscreen}
          className="p-2.5 bg-black/40 hover:bg-black/80 rounded-xl text-white/70 hover:text-white transition-all backdrop-blur-md border border-white/10 hover:border-white/30 pointer-events-auto active:scale-95"
        >
          {isFullscreen ? <Minimize2 size={20} /> : <Maximize2 size={20} />}
        </button>
      </div>

      {/* Actual Feed Content */}
      <div className="w-full h-full bg-black flex items-center justify-center relative">
        {streamUrl ? (
          <img 
            src={streamUrl} 
            alt="Live Stream" 
            className="w-full h-full object-cover opacity-90"
          />
        ) : (
          <div className="text-center text-muted/50 font-mono flex flex-col items-center">
            <Camera size={48} className="mb-4 opacity-50" />
            <p>AWAITING VIDEO STREAM</p>
            <p className="text-xs mt-2">Connecting to Inference API...</p>
          </div>
        )}

        {/* Danger Overlay */}
        {isHighRisk && (
          <div className="absolute inset-0 border-4 border-danger pointer-events-none">
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-danger/90 text-white px-6 py-2 rounded-full font-bold tracking-widest flex items-center gap-2 animate-pulse">
              <AlertTriangle size={20} />
              CRASH IMMINENT
            </div>
          </div>
        )}
      </div>

      {/* Decorative corners for a high-tech "HUD" feel */}
      <div className="absolute top-4 left-4 w-12 h-12 border-t-2 border-l-2 border-white/20 group-hover:border-primary/50 transition-colors pointer-events-none z-10 rounded-tl-lg" />
      <div className="absolute top-4 right-4 w-12 h-12 border-t-2 border-r-2 border-white/20 group-hover:border-primary/50 transition-colors pointer-events-none z-10 rounded-tr-lg" />
      <div className="absolute bottom-4 left-4 w-12 h-12 border-b-2 border-l-2 border-white/20 group-hover:border-primary/50 transition-colors pointer-events-none z-10 rounded-bl-lg" />
      <div className="absolute bottom-4 right-4 w-12 h-12 border-b-2 border-r-2 border-white/20 group-hover:border-primary/50 transition-colors pointer-events-none z-10 rounded-br-lg" />
      
      {/* Center Reticle (Subtle) */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 pointer-events-none z-10 opacity-20 group-hover:opacity-40 transition-opacity flex items-center justify-center">
        <div className="w-full h-[1px] bg-primary absolute" />
        <div className="h-full w-[1px] bg-primary absolute" />
        <div className="w-2 h-2 rounded-full border border-primary" />
      </div>
    </div>
  );
};
