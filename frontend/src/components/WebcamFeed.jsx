import React, { useRef, useEffect, useState, useCallback } from 'react';
import Webcam from 'react-webcam';
import apiClient from '../api/apiClient';
import { Camera, Play, Square, Eye } from 'lucide-react';

const WebcamFeed = ({ onPrediction, isConnected }) => {
  const webcamRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(true);
  const [skeletonPreview, setSkeletonPreview] = useState(null);
  const [handDetected, setHandDetected] = useState(false);
  const isRequestingRef = useRef(false);

  // Frame Capture and Prediction Callback
  const captureFrame = useCallback(async () => {
    if (!isStreaming || !webcamRef.current || isRequestingRef.current) return;

    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) return;

    isRequestingRef.current = true;

    try {
      const response = await apiClient.post('/api/v1/predict/base64', {
        image: imageSrc
      });

      if (response.data) {
        const { letter, confidence, sentence, skeleton_image, hand_detected, hand_stable, stability_progress } = response.data;
        setSkeletonPreview(skeleton_image);
        setHandDetected(hand_detected);
        onPrediction({ letter, confidence, sentence, hand_detected, hand_stable, stability_progress });
      }
    } catch (error) {
      console.warn('Prediction API call skipped/failed:', error?.message);
    } finally {
      isRequestingRef.current = false;
    }
  }, [isStreaming, onPrediction]);

  // Real-time Frame Capture Loop (Every 150ms)
  useEffect(() => {
    let intervalId = null;
    if (isStreaming) {
      intervalId = setInterval(() => {
        captureFrame();
      }, 150);
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isStreaming, captureFrame]);

  const toggleStreaming = () => {
    setIsStreaming(!isStreaming);
  };

  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: 'user',
  };

  return (
    <div className="glass-panel rounded-2xl p-4 lg:p-6 border border-slate-800 shadow-xl flex flex-col gap-4">
      {/* Card Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <Camera className="w-5 h-5 text-brand-500" />
          <h2 className="text-base font-bold text-slate-100">Webcam Input Feed</h2>
        </div>
        <button
          onClick={toggleStreaming}
          className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
            isStreaming
              ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30 hover:bg-rose-500/30'
              : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 hover:bg-emerald-500/30'
          }`}
        >
          {isStreaming ? (
            <>
              <Square className="w-3.5 h-3.5 fill-current" /> Stop Streaming
            </>
          ) : (
            <>
              <Play className="w-3.5 h-3.5 fill-current" /> Start Streaming
            </>
          )}
        </button>
      </div>

      {/* Main Grid: Webcam Feed + Generated Skeleton Canvas Preview */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        
        {/* Live Webcam Container */}
        <div className="relative rounded-xl overflow-hidden bg-slate-950 border border-slate-800 aspect-video flex items-center justify-center">
          {isStreaming ? (
            <Webcam
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              videoConstraints={videoConstraints}
              className="w-full h-full object-cover transform -scale-x-100"
            />
          ) : (
            <div className="flex flex-col items-center gap-2 text-slate-500">
              <Camera className="w-10 h-10 stroke-1" />
              <span className="text-xs font-medium">Webcam Stream Paused</span>
            </div>
          )}

          {/* Overlay Status Badge */}
          <div className="absolute top-3 left-3 px-2.5 py-1 rounded-md bg-slate-950/80 backdrop-blur-sm border border-slate-800 text-[11px] font-semibold flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${handDetected ? 'bg-emerald-400' : 'bg-amber-400'}`}></span>
            <span className={handDetected ? 'text-emerald-400' : 'text-amber-400'}>
              {handDetected ? 'Hand Tracked' : 'Searching for Hand...'}
            </span>
          </div>
        </div>

        {/* Synthesized Skeleton Canvas Preview */}
        <div className="relative rounded-xl overflow-hidden bg-slate-950 border border-slate-800 aspect-video flex items-center justify-center p-2">
          {skeletonPreview && isStreaming ? (
            <img
              src={skeletonPreview}
              alt="Synthesized Skeleton Canvas"
              className="w-full h-full object-contain rounded-lg"
            />
          ) : (
            <div className="flex flex-col items-center gap-2 text-slate-500">
              <Eye className="w-10 h-10 stroke-1" />
              <span className="text-xs font-medium text-center">
                Generated Skeleton Image<br />(Sent to CNN)
              </span>
            </div>
          )}

          <div className="absolute top-3 left-3 px-2.5 py-1 rounded-md bg-slate-950/80 backdrop-blur-sm border border-slate-800 text-[11px] font-semibold text-slate-300">
            Synthesized Skeleton
          </div>
        </div>

      </div>
    </div>
  );
};

export default WebcamFeed;
