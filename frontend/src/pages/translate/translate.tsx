import { useState, useEffect, useRef } from "react";
import Webcam from "react-webcam";
import ModelSequencePlayer from "../../shared/components/ModelSequencePlayer.tsx";
import "../../styles/translate.css";

export default function Translate() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [translatedText, setTranslatedText] = useState("Waiting for input...");
  const [confidence, setConfidence] = useState(0);

  // 增加摄像头引用，用于获取底层视频流
  const webcamRef = useRef<Webcam>(null);
  // 控制循环录制的开关
  const isRecordingRef = useRef(false);

  const [glbFrames] = useState<string[]>([
    "/models/frame_000.glb",
    // ... (省略你原有的 glb 数组，保持不变)
    "/models/frame_028.glb",
  ]);

  // 视频上传与接口调用逻辑
  const uploadVideo = async (videoBlob: Blob) => {
    const formData = new FormData();
    // 对应接口文档中的 video 字段，后缀用 .webm
    formData.append("video", videoBlob, "sign_clip.webm");

    try {
      const response = await fetch("/api/sign/predict", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      // 处理成功响应
      if (result.code === 200) {
        setTranslatedText(result.data.word_name);
        // 后端返回的是 0.98，前端展示 98%
        setConfidence(Math.round(result.data.confidence * 100));
      }
      // 处理错误响应（如视频太短 < 5帧）
      else if (result.code === 400) {
        console.warn("Prediction Error:", result.message);
      }
    } catch (error) {
      console.error("API Error:", error);
    }
  };

  // 核心录制循环逻辑
  useEffect(() => {
    if (!isStreaming) {
      setTranslatedText("Camera is off.");
      setConfidence(0);
      isRecordingRef.current = false;
      return;
    }

    isRecordingRef.current = true;
    setTranslatedText("Listening to gestures...");

    // 递归执行录制-上传流程
    const captureCycle = async () => {
      if (!isRecordingRef.current || !webcamRef.current?.video?.srcObject) {
        // 如果没有获取到视频流，延迟重试
        if (isRecordingRef.current) setTimeout(captureCycle, 500);
        return;
      }

      const stream = webcamRef.current.video.srcObject as MediaStream;

      // 兼容性处理：优先使用 webm
      const mimeType = MediaRecorder.isTypeSupported("video/webm")
        ? "video/webm"
        : "video/mp4";

      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      const chunks: BlobPart[] = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        // 停止录制时，打包生成视频 Blob
        const blob = new Blob(chunks, { type: mimeType });

        // 发送给后端
        await uploadVideo(blob);

        // 上传完毕后，如果还在直播状态，立即开始下一轮 2 秒的录制
        if (isRecordingRef.current) {
          captureCycle();
        }
      };

      // 开始录制
      mediaRecorder.start();

      // 录制 2 秒后停止（2秒足够产生 60 帧，远大于后端的 5 帧最低要求）
      setTimeout(() => {
        if (mediaRecorder.state === "recording") {
          mediaRecorder.stop();
        }
      }, 2000);
    };

    captureCycle();

    return () => {
      isRecordingRef.current = false; // 组件卸载或关闭摄像头时终止循环
    };
  }, [isStreaming]);

  return (
    <div className="sign-container">
      <div className="sign-card">
        {/* 左侧：翻译结果展示 */}
        <div className="text-section">
          <div className="text-header">
            <span className="live-dot"></span>
            {isStreaming ? "LIVE TRANSLATION" : "OFFLINE"}
          </div>

          <div className="text-display">
            <h1 className="main-text">{translatedText}</h1>
            {isStreaming && confidence > 0 && (
              <div className="confidence-badge">Confidence: {confidence}%</div>
            )}
          </div>

          <ModelSequencePlayer isStreaming={isStreaming} glbUrls={glbFrames} />

          <div className="instruction">
            <p>Perform sign language gestures in front of the camera.</p>
          </div>
        </div>

        {/* 右侧：摄像头输入 */}
        <div className="camera-section">
          <div className="camera-wrapper">
            {isStreaming ? (
              <>
                <Webcam
                  ref={webcamRef} // 绑定 ref 以获取流
                  audio={false}
                  className="webcam-video"
                  videoConstraints={{ facingMode: "user" }}
                />
                <div className="skeleton-overlay">
                  <HandSkeletonSVG />
                </div>
              </>
            ) : (
              <div className="camera-placeholder">
                <div className="placeholder-icon">📷</div>
              </div>
            )}
          </div>

          <div className="control-bar">
            <button
              className={`toggle-btn ${isStreaming ? "active" : ""}`}
              onClick={() => setIsStreaming(!isStreaming)}
            >
              {isStreaming ? "Stop Translation" : "Start Camera"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// 一个简单的 SVG 组件，模拟手部骨架连接点
const HandSkeletonSVG = () => (
  <svg viewBox="0 0 200 200" className="hand-svg">
    {/* 手掌中心 */}
    <circle cx="100" cy="150" r="4" fill="#00f0ff" />

    {/* 大拇指 */}
    <path
      d="M100 150 L60 140 L40 120 L30 100"
      stroke="rgba(0, 240, 255, 0.6)"
      strokeWidth="2"
      fill="none"
    />
    <circle cx="60" cy="140" r="3" fill="#00f0ff" />
    <circle cx="40" cy="120" r="3" fill="#00f0ff" />
    <circle cx="30" cy="100" r="3" fill="#00f0ff" />

    {/* 食指 */}
    <path
      d="M100 150 L90 110 L85 80 L80 50"
      stroke="rgba(0, 240, 255, 0.6)"
      strokeWidth="2"
      fill="none"
    />
    <circle cx="90" cy="110" r="3" fill="#00f0ff" />
    <circle cx="85" cy="80" r="3" fill="#00f0ff" />
    <circle cx="80" cy="50" r="3" fill="#00f0ff" />

    {/* 中指 */}
    <path
      d="M100 150 L105 105 L108 70 L110 40"
      stroke="rgba(0, 240, 255, 0.6)"
      strokeWidth="2"
      fill="none"
    />
    <circle cx="105" cy="105" r="3" fill="#00f0ff" />
    <circle cx="108" cy="70" r="3" fill="#00f0ff" />
    <circle cx="110" cy="40" r="3" fill="#00f0ff" />

    {/* 动画光圈 */}
    <circle
      cx="100"
      cy="150"
      r="10"
      stroke="#00f0ff"
      strokeWidth="1"
      fill="none"
      className="pulse-circle"
    />
  </svg>
);
