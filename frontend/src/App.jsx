import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Navbar from './components/Navbar';
import WebcamFeed from './components/WebcamFeed';
import PredictionBox from './components/PredictionBox';
import SentenceBox from './components/SentenceBox';

const App = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [predictionData, setPredictionData] = useState({
    letter: '',
    confidence: 0,
    sentence: '',
    hand_detected: false,
    hand_stable: false,
    stability_progress: 0
  });

  // Check Backend Health Endpoint on Mount
  useEffect(() => {
    const checkBackendHealth = async () => {
      try {
        const response = await axios.get('/api/v1/health');
        if (response.data?.status === 'healthy') {
          setIsConnected(true);
        } else {
          setIsConnected(false);
        }
      } catch (error) {
        console.warn('Backend API connection check failed:', error.message);
        setIsConnected(false);
      }
    };

    checkBackendHealth();
    const interval = setInterval(checkBackendHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  const handlePrediction = (data) => {
    setPredictionData((prev) => ({
      letter: data.letter,
      confidence: data.confidence,
      sentence: data.sentence !== undefined ? data.sentence : prev.sentence,
      hand_detected: data.hand_detected,
      hand_stable: data.hand_stable,
      stability_progress: data.stability_progress
    }));
  };

  const handleSentenceChange = (newSentence) => {
    setPredictionData((prev) => ({
      ...prev,
      sentence: newSentence
    }));
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col font-sans selection:bg-brand-500 selection:text-slate-950">
      {/* Top Header Bar */}
      <Navbar isConnected={isConnected} />

      {/* Main Dashboard Layout */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 lg:p-8 flex flex-col gap-6">
        
        {/* Main 2-Column Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
          
          {/* Left 2 Columns: Webcam Stream & Skeleton Synthesis Preview */}
          <div className="lg:col-span-2">
            <WebcamFeed
              onPrediction={handlePrediction}
              isConnected={isConnected}
            />
          </div>

          {/* Right 1 Column: Prediction Result Box */}
          <div className="lg:col-span-1">
            <PredictionBox
              letter={predictionData.letter}
              confidence={predictionData.confidence}
              handDetected={predictionData.hand_detected}
              handStable={predictionData.hand_stable}
              stabilityProgress={predictionData.stability_progress}
              onSentenceChange={handleSentenceChange}
            />
          </div>

        </div>

        {/* Bottom Full-Width Row: Buffered Sentence Builder & Text-to-Speech */}
        <div className="w-full">
          <SentenceBox
            sentence={predictionData.sentence}
            onSentenceChange={handleSentenceChange}
          />
        </div>

      </main>

      {/* Footer */}
      <footer className="w-full py-4 px-8 border-t border-slate-800/80 text-center text-xs text-slate-500 font-medium">
        <p>&copy; 2026 ASL Vision AI &bull; Deep Learning Hand Gesture Recognition System</p>
      </footer>
    </div>
  );
};

export default App;
