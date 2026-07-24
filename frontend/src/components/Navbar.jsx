import React from 'react';
import { Camera, Zap } from 'lucide-react';

const Navbar = ({ isConnected }) => {
  return (
    <header className="sticky top-0 z-50 w-full glass-panel border-b border-slate-800/80 px-4 lg:px-8 py-3.5">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        
        {/* Brand Logo & Title */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-500 to-teal-400 flex items-center justify-center shadow-glow">
            <Camera className="w-5 h-5 text-slate-950 font-bold" />
          </div>
          <div>
            <h1 className="text-lg lg:text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white via-slate-200 to-emerald-400 tracking-tight">
              ASL Vision AI
            </h1>
            <p className="text-xs text-slate-400 flex items-center gap-1.5 font-medium">
              <Zap className="w-3.5 h-3.5 text-emerald-400" /> Real-Time Sign Language Recognition Engine
            </p>
          </div>
        </div>

        {/* System Online Status Indicator */}
        <div className="flex items-center gap-3">
          <div className={`flex items-center gap-2 px-3.5 py-1.5 rounded-full text-xs font-semibold border transition-all ${
            isConnected
              ? 'bg-emerald-950/60 text-emerald-400 border-emerald-500/30 shadow-sm'
              : 'bg-rose-950/60 text-rose-400 border-rose-500/30'
          }`}>
            <span className="relative flex h-2 w-2">
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                isConnected ? 'bg-emerald-400' : 'bg-rose-400'
              }`}></span>
              <span className={`relative inline-flex rounded-full h-2 w-2 ${
                isConnected ? 'bg-emerald-500' : 'bg-rose-500'
              }`}></span>
            </span>
            <span>{isConnected ? 'System Ready' : 'Connecting Engine...'}</span>
          </div>
        </div>

      </div>
    </header>
  );
};

export default Navbar;
