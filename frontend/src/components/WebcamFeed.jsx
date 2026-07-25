import React, { useRef, useEffect, useState, useCallback } from 'react';
import Webcam from 'react-webcam';
import API from '../services/api';
import { Camera, Play, Square, Eye } from 'lucide-react';

const WebcamFeed = ({ onPrediction, isConnected }) => {
  const webcamRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(true);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [skeletonPreview, setSkeletonPreview] = useState(null);
  const [handDetected, setHandDetected] = useState(false);
  const isRequestingRef = useRef(false);

  useEffect(() => {
    console.log('[Prediction Trace] Task 2 Environment Verification: VITE_API_URL =', import.meta.env.VITE_API_URL);
  }, []);

  const handleUserMedia = useCallback(() => {
    console.log('[Prediction Trace] Camera Ready event fired via react-webcam onUserMedia.');
    setIsCameraReady(true);
  }, []);

  const handleUserMediaError = useCallback((error) => {
    console.error('[Prediction Trace] Camera access error:', error);
    setIsCameraReady(false);
  }, []);

  // Task 4 & 5: Trace Prediction Flow from Frame Capture to API Response
  const captureFrame = useCallback(async () => {
    if (!isStreaming) return;

    if (!webcamRef.current) {
      // console.warn('[Prediction Trace] Skip: webcamRef.current is null');
      return;
    }

    if (isRequestingRef.current) {
      // Skip if previous API call is still in flight to avoid request stacking
      return;
    }

    let imageSrc = null;
    try {
      imageSrc = webcamRef.current.getScreenshot();
    } catch (err) {
      console.warn('[Prediction Trace] getScreenshot exception:', err);
      return;
    }

    if (!imageSrc) {
      // Camera is still initializing or video stream not yet rendering frames
      return;
    }

    console.log('[Prediction Trace] Frame Captured successfully. Base64 payload length:', imageSrc.length);
    console.log('[Prediction Trace] Preparing Payload and Sending Request to /api/v1/predict/base64...');

    isRequestingRef.current = true;

    try {
      const response = await API.post('/api/v1/predict/base64', {
        image: imageSrc
      });

      console.log('[Prediction Trace] Backend Response Received:', response.status, response.data);

      if (response.data) {
        const { letter, confidence, sentence, skeleton_image, hand_detected, hand_stable, stability_progress } = response.data;
        
        console.log(`[Prediction Trace] Hand Detected: ${hand_detected}, Hand Stable: ${hand_stable}`);
        if (hand_detected) {
          console.log(`[Prediction Trace] Skeleton Generated: preview present=${Boolean(skeleton_image)}`);
          console.log(`[Prediction Trace] Prediction Received: Letter = '${letter}', Confidence = ${confidence}%`);
          console.log(`[Prediction Trace] Buffered Sentence Updated: "${sentence}"`);
        }

        setSkeletonPreview(skeleton_image);
        setHandDetected(hand_detected);
        onPrediction({ letter, confidence, sentence, hand_detected, hand_stable, stability_progress });
      }
    } catch (error) {
      console.error('[Prediction Trace] Prediction Request Failed:', {
        message: error.message,
        response: error.response?.data,
        status: error.response?.status,
        stack: error.stack
      });
    } finally {
      isRequestingRef.current = false;
    }
  }, [isStreaming, onPrediction]);

  // Real-time Frame Capture Loop (Runs every 200ms when streaming)
  useEffect(() => {
    let intervalId = null;
    if (isStreaming) {
      console.log('[Prediction Trace] Starting frame capture loop (every 200ms)...');
      intervalId = setInterval(() => {
        captureFrame();
      }, 200);
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
              onUserMedia={handleUserMedia}
              onUserMediaError={handleUserMediaError}
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
              {handDetected ? 'Hand Tracked' : isCameraReady ? 'Searching for Hand...' : 'Initializing Camera...'}
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
