import React from 'react';
import apiClient from '../api/apiClient';
import { Sparkles, ShieldCheck, Timer, Lock, PlusCircle } from 'lucide-react';

const PredictionBox = ({ letter, confidence, handDetected, handStable, stabilityProgress, onSentenceChange }) => {
  const getConfidenceColor = (conf) => {
    if (conf >= 85) return 'from-emerald-500 to-teal-400 text-emerald-400';
    if (conf >= 60) return 'from-amber-500 to-yellow-400 text-amber-400';
    return 'from-rose-500 to-red-400 text-rose-400';
  };

  const progressVal = stabilityProgress || 0;

  const handleManualAppend = async () => {
    if (!letter) return;
    try {
      const response = await apiClient.post('/api/v1/sentence/add', { letter });
      if (response.data?.sentence !== undefined && onSentenceChange) {
        onSentenceChange(response.data.sentence);
      }
    } catch (err) {
      console.error('Failed to append letter manually:', err);
    }
  };

  return (
    <div className="glass-panel rounded-2xl p-6 border border-slate-800 shadow-xl flex flex-col justify-between items-center text-center relative overflow-hidden">
      
      {/* Top Header & Status Pill */}
      <div className="w-full flex items-center justify-between border-b border-slate-800/80 pb-3 mb-4">
        <div className="flex items-center gap-2 text-slate-300 text-sm font-semibold">
          <Sparkles className="w-4 h-4 text-emerald-400" />
          <span>Real-time Prediction</span>
        </div>
        <span className={`text-xs font-semibold px-3 py-1 rounded-full border flex items-center gap-1.5 ${
          handStable && letter
            ? 'bg-emerald-950/80 text-emerald-400 border-emerald-500/30'
            : handDetected
            ? 'bg-emerald-950/60 text-emerald-300 border-emerald-500/30'
            : 'bg-slate-800 text-slate-400 border-slate-700'
        }`}>
          {handStable && letter ? (
            <>
              <Lock className="w-3 h-3 text-emerald-400" /> Appended
            </>
          ) : handDetected ? (
            <>
              <Sparkles className="w-3 h-3 text-emerald-400" /> Live Predicting
            </>
          ) : (
            'No Hand'
          )}
        </span>
      </div>

      {/* Hero Prediction Display Box */}
      <div className="my-3 flex flex-col items-center justify-center">
        <div className="relative group">
          <div className="w-36 h-36 lg:w-44 lg:h-44 rounded-3xl bg-gradient-to-tr from-slate-950 to-slate-900 border-2 border-emerald-500/40 flex items-center justify-center shadow-glow transition-all duration-300 group-hover:border-emerald-400">
            {handDetected && letter ? (
              <span className="text-7xl lg:text-8xl font-extrabold bg-clip-text text-transparent bg-gradient-to-tr from-white via-emerald-200 to-emerald-400 animate-pulse">
                {letter}
              </span>
            ) : (
              <span className="text-5xl font-extrabold text-slate-700">--</span>
            )}
          </div>
        </div>

        <p className="text-xs text-slate-400 font-medium mt-3">
          {handDetected && letter
            ? `Recognized ASL Gesture: Letter '${letter}'`
            : 'Position hand inside camera frame'}
        </p>

        {/* Manual Append Button */}
        {handDetected && letter && (
          <button
            onClick={handleManualAppend}
            className="mt-3 flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 border border-emerald-500/40 text-xs font-bold transition-all active:scale-95 shadow-glow"
          >
            <PlusCircle className="w-4 h-4 text-emerald-400" /> Append '{letter}' to Sentence
          </button>
        )}
      </div>

      {/* Stability Auto-Buffer Timer Bar */}
      {handDetected && (
        <div className="w-full bg-slate-950/80 p-3.5 rounded-xl border border-emerald-500/20 mb-3 flex flex-col gap-1.5">
          <div className="flex justify-between items-center text-xs font-semibold text-emerald-400">
            <span className="flex items-center gap-1.5">
              <Timer className="w-3.5 h-3.5" /> Sentence Auto-Append Buffer
            </span>
            <span>{progressVal.toFixed(1)}%</span>
          </div>
          <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden p-0.5 border border-slate-700/50">
            <div
              className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-150 ease-linear"
              style={{ width: `${Math.min(100, Math.max(0, progressVal))}%` }}
            ></div>
          </div>
        </div>
      )}

      {/* Confidence Percentage Bar */}
      <div className="w-full bg-slate-950/80 p-4 rounded-xl border border-slate-800 flex flex-col gap-2">
        <div className="flex justify-between items-center text-xs font-semibold">
          <span className="text-slate-400 flex items-center gap-1.5">
            <ShieldCheck className="w-4 h-4 text-emerald-400" /> Model Confidence
          </span>
          <span className={`text-sm font-bold ${getConfidenceColor(confidence).split(' ').pop()}`}>
            {confidence ? `${confidence.toFixed(1)}%` : '0.0%'}
          </span>
        </div>

        <div className="w-full h-2.5 bg-slate-800 rounded-full overflow-hidden p-0.5 border border-slate-700/50">
          <div
            className={`h-full rounded-full bg-gradient-to-r ${getConfidenceColor(confidence)} transition-all duration-300 ease-out`}
            style={{ width: `${Math.min(100, Math.max(0, confidence || 0))}%` }}
          ></div>
        </div>
      </div>

    </div>
  );
};

export default PredictionBox;
