import { useEffect, useRef, useState } from 'react';
import Webcam from 'react-webcam';
import '../../styles/translate.css';

const API_BASE_URL = 'http://127.0.0.1:5000';

export default function Translate() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [statusText, setStatusText] = useState('Camera is off. Ready to start.');
  const [translatedText, setTranslatedText] = useState('');
  const [confidence, setConfidence] = useState(0);

  const webcamRef = useRef<Webcam>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);

  const processPrediction = async (videoBlob: Blob) => {
    setIsProcessing(true);
    setStatusText('Recognizing gesture...');

    const formData = new FormData();
    formData.append('video', videoBlob, 'sign_clip.webm');

    try {
      const response = await fetch(`${API_BASE_URL}/api/sign/predict`, {
        method: 'POST',
        body: formData,
      });
      const result = await response.json();

      if (!response.ok || result.code !== 200) {
        const message = result.message || 'Prediction failed.';
        setTranslatedText('');
        setConfidence(0);
        setStatusText(message);
        return;
      }

      const wordName = String(result.data?.word_name || '').toUpperCase();
      const nextConfidence = Number(result.data?.confidence || 0);

      setTranslatedText(wordName);
      setConfidence(Math.round(nextConfidence * 100));
      setStatusText('Recognition complete.');
    } catch (error) {
      console.error('Prediction Error:', error);
      setTranslatedText('');
      setConfidence(0);
      setStatusText('Network error occurred.');
    } finally {
      setIsProcessing(false);
    }
  };

  useEffect(() => {
    if (isStreaming) {
      chunksRef.current = [];
      setTranslatedText('');
      setConfidence(0);
      setStatusText('Camera turning on...');
      return;
    }

    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state === 'recording') {
      setStatusText('Stopping recording...');
      recorder.stop();
    }
  }, [isStreaming]);

  const handleUserMedia = (stream: MediaStream) => {
    if (!isStreaming) {
      return;
    }

    setStatusText('Recording...');
    chunksRef.current = [];

    const mimeType = MediaRecorder.isTypeSupported('video/webm') ? 'video/webm' : 'video/mp4';
    const mediaRecorder = new MediaRecorder(stream, { mimeType });

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        chunksRef.current.push(event.data);
      }
    };

    mediaRecorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: mimeType });
      chunksRef.current = [];

      if (blob.size === 0) {
        setStatusText('No video captured.');
        return;
      }

      void processPrediction(blob);
    };

    mediaRecorder.start();
    mediaRecorderRef.current = mediaRecorder;
  };

  const headerLabel = isStreaming ? 'LIVE CAPTURE' : isProcessing ? 'PROCESSING' : 'READY';
  const primaryText = translatedText || 'Camera is off. Ready to start.';

  return (
    <div className="translate-page">
      <section className="translate-shell">
        <div className="translate-shell__glow translate-shell__glow--left" />
        <div className="translate-shell__glow translate-shell__glow--right" />

        <div className="translate-panel">
          <div className="translate-panel__status">
            <div className="translate-panel__eyebrow">
              <span className={`translate-panel__dot ${isStreaming ? 'is-live' : ''}`} />
              {headerLabel}
            </div>

            <div className="translate-panel__content">
              <div className="translate-panel__result-row">
                <p className="translate-panel__title">{primaryText}</p>
                {confidence > 0 ? (
                  <p className="translate-panel__confidence">Confidence {confidence}%</p>
                ) : null}
              </div>

              <p className="translate-panel__status-copy">{statusText}</p>

              <button
                className={`translate-camera-card__button ${isStreaming ? 'is-live' : ''}`}
                onClick={() => setIsStreaming(!isStreaming)}
                disabled={isProcessing}
              >
                {isStreaming ? 'Stop & Predict' : 'Start Camera'}
              </button>
            </div>
          </div>

          <div className="translate-camera-card">
            <div className="translate-camera-card__frame">
              {isStreaming ? (
                <>
                  <Webcam
                    ref={webcamRef}
                    audio={false}
                    className="translate-camera-card__video"
                    videoConstraints={{ facingMode: 'user' }}
                    onUserMedia={handleUserMedia}
                  />
                  <div className="translate-camera-card__overlay">
                    <HandSkeletonSVG />
                  </div>
                </>
              ) : (
                <div className="translate-camera-card__placeholder">
                  <div className="translate-camera-card__placeholder-icon">📷</div>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

const HandSkeletonSVG = () => (
  <svg viewBox="0 0 200 200" className="translate-hand-svg" aria-hidden="true">
    <circle cx="100" cy="150" r="4" fill="#62f0d8" />
    <path
      d="M100 150 L60 140 L40 120 L30 100"
      stroke="rgba(98, 240, 216, 0.6)"
      strokeWidth="2"
      fill="none"
    />
    <circle cx="60" cy="140" r="3" fill="#62f0d8" />
    <circle cx="40" cy="120" r="3" fill="#62f0d8" />
    <circle cx="30" cy="100" r="3" fill="#62f0d8" />
    <path
      d="M100 150 L90 110 L85 80 L80 50"
      stroke="rgba(98, 240, 216, 0.6)"
      strokeWidth="2"
      fill="none"
    />
    <circle cx="90" cy="110" r="3" fill="#62f0d8" />
    <circle cx="85" cy="80" r="3" fill="#62f0d8" />
    <circle cx="80" cy="50" r="3" fill="#62f0d8" />
    <path
      d="M100 150 L105 105 L108 70 L110 40"
      stroke="rgba(98, 240, 216, 0.6)"
      strokeWidth="2"
      fill="none"
    />
    <circle cx="105" cy="105" r="3" fill="#62f0d8" />
    <circle cx="108" cy="70" r="3" fill="#62f0d8" />
    <circle cx="110" cy="40" r="3" fill="#62f0d8" />
    <circle
      cx="100"
      cy="150"
      r="10"
      stroke="#62f0d8"
      strokeWidth="1"
      fill="none"
      className="translate-hand-svg__pulse"
    />
  </svg>
);
