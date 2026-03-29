import cv2
import mindspore as ms
import mindspore.nn as nn
import numpy as np
import mediapipe as mp
import os
import sys

# 将当前目录加入系统路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
import json
import pickle
from model import BiLSTMAttentionModel
from core_preprocess import to_double_relative_with_velocity

class LandmarkSmoother:
    def __init__(self, alpha=config.ALPHA):
        self.alpha = alpha
        self.prev_landmarks = None

    def smooth(self, current_landmarks):
        if current_landmarks is None:
            self.prev_landmarks = None
            return None
        if self.prev_landmarks is None:
            self.prev_landmarks = current_landmarks
            return current_landmarks
        smoothed = self.alpha * current_landmarks + (1 - self.alpha) * self.prev_landmarks
        self.prev_landmarks = smoothed
        return smoothed

class SignLanguageInferencePipeline:
    def __init__(self, model_path, mean_path=None, std_path=None, json_path=None):
        # 设置 MindSpore 上下文 - 优先使用 Ascend，无 Ascend 时回退到 CPU
        try:
            ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
            print(f"Using MindSpore with device: Ascend")
        except Exception:
            ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
            print(f"Using MindSpore with device: CPU")

        # 初始化 MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True
        )

        # 2. 加载全量模型数据
        self.seq_len = config.SEQ_LEN
        self.num_classes = config.NUM_CLASSES
        self.input_size = config.INPUT_SIZE #268
        self.hidden_size=config.HIDDEN_SIZE# 268
        self.num_layers=config.NUM_LAYERS

        self.model = BiLSTMAttentionModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            dropout=0.0
        )

        # 加载 MindSpore .ckpt 模型
        ms.load_checkpoint(model_path, self.model)
        self.model.set_train(False)
        print(f"Loaded model weights from {model_path}")

        # 3. 加载归一化统计量
        self.mean = None
        self.std = None

        if mean_path is None and hasattr(config, 'MEAN_PATH'):
            mean_path = config.MEAN_PATH
        if std_path is None and hasattr(config, 'STD_PATH'):
            std_path = config.STD_PATH

        if mean_path and os.path.exists(mean_path):
            self.mean = np.load(mean_path).astype(np.float32)
            self.std = np.load(std_path).astype(np.float32)
            print(f"Loaded normalization stats from {mean_path}")
        else:
            print("🔴 WARNING: Normalization stats not found. Inference might be inaccurate if the model was trained with normalized data 🔴")

        # 4. 加载标签映射
        self.label_map = {}
        if json_path is None and hasattr(config, 'IDX2NAME_PATH'):
            json_path = config.IDX2NAME_PATH

        if json_path and os.path.exists(json_path):
            try:
                if json_path.endswith('.txt'):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue

                            parts = line.split()

                            if len(parts) >= 2:
                                try:
                                    lbl_id = int(parts[0])
                                    word = parts[1]
                                    self.label_map[lbl_id] = word
                                except ValueError:
                                    continue
                    print(f"✅ Successfully loaded {len(self.label_map)} classes from TXT.")
                elif json_path.endswith('.pkl'):
                    with open(json_path, 'rb') as f:
                        self.label_map = pickle.load(f)
                    print(f"✅ Loaded label map from PKL.")
                else:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        split_data = json.load(f)
                        for word, info in split_data.items():
                            if "action" in info:
                                lbl_id = info["action"][0]
                                self.label_map[lbl_id] = word
                    print(f"Loaded label map from {json_path}")
            except Exception as e:
                print(f"🔴 WARNING:Failed to load label map: {e}")

        self.smoother = LandmarkSmoother(alpha=config.ALPHA)
        
    def extract_features(self, frame):
        """单帧提取134维特征"""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image)
        
        row = []
        # Pose (25 点 -> 50维)
        if results.pose_landmarks:
            for i in range(25):
                lm = results.pose_landmarks.landmark[i]
                row.extend([lm.x, lm.y])
        else:
            row.extend([0.0] * 50)

        # Left Hand (21 点 -> 42维)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                row.extend([lm.x, lm.y])
        else:
            row.extend([0.0] * 42)

        # Right Hand (21 点 -> 42维)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                row.extend([lm.x, lm.y])
        else:
            row.extend([0.0] * 42)
            
        return np.array(row, dtype=np.float32)
        
    def preprocess_sequence(self, raw_seq):
        """处理特征序列：转换双重相对+速度 -> 归一化 -> 线性抽样"""
        if len(raw_seq) < 5:
            return None
            
        raw_seq = np.array(raw_seq, dtype=np.float32)
        
        # 提取双相对坐标与速度 -> 维度变为 (T, 268)
        data = to_double_relative_with_velocity(raw_seq)
        
        # 归一化
        if self.mean is not None:
            data = (data - self.mean) / self.std
            
        # 线性索引抽样
        idx = np.linspace(0, len(data) - 1, self.seq_len)
        idx = np.round(idx).astype(int)
        data = data[idx]
        
        return data
        
    def predict(self, raw_seq):
        """模型推理"""
        # 预处理
        processed_data = self.preprocess_sequence(raw_seq)
        if processed_data is None:
            return "Too Short", 0.0

        # 转 MindSpore Tensor
        tensor_data = ms.Tensor(processed_data, dtype=ms.float32).expand_dims(0)  # [1, 64, 268]

        out = self.model(tensor_data)
        probs = nn.Softmax(axis=1)(out)
        conf, pred_class = ms.ops.max(probs, axis=1)

        conf = float(conf.asnumpy())
        pred_class = int(pred_class.asnumpy())

        # 获取文本标签
        label_text = self.label_map.get(pred_class, f"Class {pred_class}")
        return label_text, conf

def run_camera_inference():
    # 路径配置
    MODEL_PATH = config.MODEL_PATH
    
    pipeline = SignLanguageInferencePipeline(model_path=MODEL_PATH)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera could not be opened.")
        return
        
    # 定义手势识别窗口参数
    frame_buffer = []
    
    current_prediction = "Waiting..."
    current_confidence = 0.0
    recording = False
    
    print("🎬 Press 'q', 'ESC' or close window to quit.")
    print("🎬 Click screen to record.")
    
    cv2.namedWindow('Sign Language Real-time Inference', cv2.WINDOW_NORMAL)
    
    def mouse_click(event, x, y, flags, param):
        nonlocal recording, frame_buffer, current_prediction, current_confidence
        if event == cv2.EVENT_LBUTTONDOWN:
            if not recording:
                recording = True
                frame_buffer = []
                current_prediction = "..."
                current_confidence = 0.0
            else:
                recording = False
                pred_text, conf = pipeline.predict(frame_buffer)
                current_prediction = pred_text
                current_confidence = conf

    cv2.setMouseCallback('Sign Language Real-time Inference', mouse_click)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)  # 镜像显示更自然
        
        # 提取特征 & 绘制关键点
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pipeline.holistic.process(rgb)
        
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, pipeline.mp_holistic.POSE_CONNECTIONS)
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.left_hand_landmarks, pipeline.mp_holistic.HAND_CONNECTIONS)
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.right_hand_landmarks, pipeline.mp_holistic.HAND_CONNECTIONS)
        
        if recording:
            raw_feat = pipeline.extract_features(frame)
            smooth_feat = pipeline.smoother.smooth(raw_feat)
            if smooth_feat is not None:
                frame_buffer.append(smooth_feat)
            cv2.putText(frame, f"REC {len(frame_buffer)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            pipeline.smoother.prev_landmarks = None
            cv2.putText(frame, "Click to Record", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 4. 在画面上绘制预测结果
        cv2.rectangle(frame, (0, 400), (640, 480), (40, 40, 40), -1)
        cv2.putText(frame, f"Result: {current_prediction}", (20, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, f"Conf: {current_confidence:.2%}", (20, 465),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, f"Frames: {len(frame_buffer)}", (230, 465),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        
        cv2.imshow('Sign Language Real-time Inference', frame)
        
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q') or cv2.getWindowProperty("Sign Language Real-time Inference", 0) < 0:
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera_inference()