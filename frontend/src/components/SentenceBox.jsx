import React, { useState } from 'react';
import API from '../services/api';
import { Volume2, Trash2, Delete, Space, Type } from 'lucide-react';

const SentenceBox = ({ sentence, onSentenceChange }) => {
  const [isSpeaking, setIsSpeaking] = useState(false);

  // Browser SpeechSynthesis API Handler
  const handleSpeak = () => {
    if (!sentence || sentence.trim() === '') return;

    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel(); // Stop active speech
      const utterance = new SpeechSynthesisUtterance(sentence);
      utterance.rate = 0.9;
      utterance.pitch = 1.0;

      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => setIsSpeaking(false);
      utterance.onerror = () => setIsSpeaking(false);

      window.speechSynthesis.speak(utterance);
    } else {
      alert('Browser SpeechSynthesis API is not supported in your browser.');
    }
  };

  const handleSpace = async () => {
    try {
      console.log('[SentenceBox] Appending space via API.post("/api/v1/sentence/space")...');
      const response = await API.post('/api/v1/sentence/space');
      if (response.data?.sentence !== undefined) {
        onSentenceChange(response.data.sentence);
      }
    } catch (err) {
      console.error('[SentenceBox] Failed to append space:', err?.message, err?.response?.data);
    }
  };

  const handleDelete = async () => {
    try {
      console.log('[SentenceBox] Deleting character via API.post("/api/v1/sentence/delete")...');
      const response = await API.post('/api/v1/sentence/delete');
      if (response.data?.sentence !== undefined) {
        onSentenceChange(response.data.sentence);
      }
    } catch (err) {
      console.error('[SentenceBox] Failed to delete character:', err?.message, err?.response?.data);
    }
  };

  const handleClear = async () => {
    try {
      console.log('[SentenceBox] Clearing sentence via API.post("/api/v1/sentence/clear")...');
      const response = await API.post('/api/v1/sentence/clear');
      if (response.data?.sentence !== undefined) {
        onSentenceChange(response.data.sentence);
      }
    } catch (err) {
      console.error('[SentenceBox] Failed to clear sentence:', err?.message, err?.response?.data);
    }
  };

  return (
    <div className="glass-panel rounded-2xl p-6 border border-slate-800 shadow-xl flex flex-col gap-4">
      
      {/* Box Header */}
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
        <div className="flex items-center gap-2 text-slate-300 text-sm font-semibold">
          <Type className="w-4 h-4 text-emerald-400" />
          <span>Buffered Sentence Builder</span>
        </div>
        <span className="text-xs text-slate-500 font-medium">
          {sentence ? `${sentence.length} characters` : 'Empty buffer'}
        </span>
      </div>

      {/* Main Text Display Area */}
      <div className="w-full min-h-[100px] bg-slate-950/90 rounded-xl border border-slate-800 p-4 flex items-center justify-between gap-4">
        <p className="text-lg lg:text-xl font-semibold text-slate-100 tracking-wide break-all">
          {sentence || <span className="text-slate-600 italic font-normal">Buffered sign letters will appear here...</span>}
        </p>

        {/* Speak Button */}
        <button
          onClick={handleSpeak}
          disabled={!sentence || sentence.trim() === ''}
          className={`flex items-center gap-2 px-4 py-2.5 rounded-xl font-bold text-xs transition-all ${
            sentence && sentence.trim() !== ''
              ? 'bg-gradient-to-r from-emerald-500 to-teal-600 text-slate-950 hover:from-emerald-400 hover:to-teal-500 shadow-glow cursor-pointer'
              : 'bg-slate-800 text-slate-600 cursor-not-allowed border border-slate-700/50'
          }`}
        >
          <Volume2 className={`w-4 h-4 ${isSpeaking ? 'animate-bounce' : ''}`} />
          <span>{isSpeaking ? 'Speaking...' : 'Speak'}</span>
        </button>
      </div>

      {/* Control Buttons Toolbar */}
      <div className="grid grid-cols-3 gap-3">
        <button
          onClick={handleSpace}
          className="flex items-center justify-center gap-2 py-2.5 px-3 rounded-xl bg-slate-800/80 hover:bg-slate-700/80 text-slate-200 text-xs font-semibold border border-slate-700/60 transition-all active:scale-95"
        >
          <Space className="w-4 h-4 text-brand-500" /> Space
        </button>

        <button
          onClick={handleDelete}
          className="flex items-center justify-center gap-2 py-2.5 px-3 rounded-xl bg-slate-800/80 hover:bg-slate-700/80 text-slate-200 text-xs font-semibold border border-slate-700/60 transition-all active:scale-95"
        >
          <Delete className="w-4 h-4 text-amber-400" /> Delete
        </button>

        <button
          onClick={handleClear}
          className="flex items-center justify-center gap-2 py-2.5 px-3 rounded-xl bg-rose-950/40 hover:bg-rose-900/50 text-rose-300 text-xs font-semibold border border-rose-500/30 transition-all active:scale-95"
        >
          <Trash2 className="w-4 h-4 text-rose-400" /> Clear
        </button>
      </div>

    </div>
  );
};

export default SentenceBox;
