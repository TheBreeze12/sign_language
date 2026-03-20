# 3D 手语重建系统 — 保姆级完整文档

> **文档版本**：v1.0 | **撰写日期**：2026-03-15  
> **适用人群**：初次接触本项目的开发者、参赛评委、迁移开发者

---

## 目录

1. [项目简介与背景](#1-项目简介与背景)
2. [解决的核心问题](#2-解决的核心问题)
3. [系统架构总览](#3-系统架构总览)
4. [技术栈详解](#4-技术栈详解)
5. [目录结构说明](#5-目录结构说明)
6. [三阶段流水线详解](#6-三阶段流水线详解)
   - [6.1 阶段一：视频预处理 `preprocess_3d.py`](#61-阶段一视频预处理-preprocess_3dpy)
   - [6.2 阶段二：3D 参数推理 `inference_to_db.py`](#62-阶段二3d-参数推理-inference_to_dbpy)
   - [6.3 阶段三：3D 渲染导出 `render_3d.py`](#63-阶段三3d-渲染导出-render_3dpy)
7. [MANO 手部模型库解析](#7-mano-手部模型库解析)
8. [配置文件详解 `config_3d.py`](#8-配置文件详解-config_3dpy)
9. [核心设计决策解析（15个）](#9-核心设计决策解析15个)
10. [数据流转与坐标系变换](#10-数据流转与坐标系变换)
11. [使用实操指南（从零开始）](#11-使用实操指南从零开始)
12. [迁移到 MindSpore 指南](#12-迁移到-mindspore-指南)
13. [常见问题与排错](#13-常见问题与排错)
14. [依赖安装速查表](#14-依赖安装速查表)

---

## 1. 项目简介与背景

### 1.1 这个项目是做什么的？

本项目是一套**全自动三维手语重建流水线**，能够将 ASL Citizen 手语视频数据集中的原始 MP4 视频，转换为可供展示、分析和驱动 3D 角色的**逐帧三维手部网格模型（GLB 格式）**。

简单来说：给它一段打手语的视频，它会输出每一帧的三维立体手部模型。

### 1.2 应用场景

| 场景 | 说明 |
|------|------|
| **手语教学辅助** | 将真人手语视频转成 3D 模型，方便从任意角度观察手型 |
| **手语识别数据增强** | 生成 3D 参数数据库，可用于训练手语识别神经网络 |
| **无障碍辅助技术** | 驱动虚拟手语翻译角色，实现文字→3D 手语动画 |
| **竞赛/科研展示** | 作为 ICT 竞赛参赛项目的核心技术组件 |

### 1.3 使用的数据集

**ASL Citizen**（美国手语公民数据集）
- 包含数千个美国手语词汇的视频
- 每个词汇有多个不同人员的录制版本
- CSV 格式的 splits 元数据（含 `Video file`、`Gloss` 字段）
- 本项目从中取 **300 个目标词汇**，每词取 **1 个视频**进行处理

---

## 2. 解决的核心问题

### 问题一：如何从 2D 视频中提取 3D 手部信息？

手机或摄像头拍摄的视频是 2D 平面图像，无法直接获得深度信息。本项目使用 **S2HAND（Single View Hand Reconstruction）** 模型，从单张手部图像中回归出完整的三维 MANO 手部参数。

### 问题二：如何准确区分左右手并追踪？

手语视频中，左右手常常同时出现并快速移动，MediaPipe 检测结果没有固定的左右手标签。本项目设计了一套**基于最近邻的手部身份追踪算法**，在初始化帧按 X 坐标判断左右，后续帧按运动轨迹最近邻继承。

### 问题三：S2HAND 只有右手模型，左手怎么办？

S2HAND 的预训练权重仅针对右手。本项目采用**镜像技巧**：将左手图像水平翻转后当作右手输入，推理完成后再将 X 轴坐标取反，实现左手的 3D 重建，无需额外的左手模型。

### 问题四：如何保证动画流畅不抖动？

原始手部检测框会随图像噪声剧烈抖动。本项目在 BBox 序列上施加**一维高斯平滑**（σ=1.7），并对 3D 顶点序列进行**三次样条插值升频**（×2），使输出动画平滑自然。

### 问题五：如何还原手在真实空间中的位移？

S2HAND 只能输出相对手腕的局部坐标，无法知道手在画面中的全局位置。本项目从 `meta.json` 读取 BBox 的位置和尺寸，通过**透视反比公式**估算手到镜头的深度，并将 2D 像素位移映射为 3D 世界坐标位移，还原出完整的空间轨迹。

---

## 3. 系统架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    输入：ASL Citizen 视频数据集                    │
│              (.mp4 文件，按 gloss_id 命名)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              阶段 1：preprocess_3d.py（视频预处理）                │
│                                                                  │
│  ① MediaPipe 逐帧检测手部关键点（21个）                            │
│  ② 最近邻算法追踪左右手身份                                        │
│  ③ 自适应采样（16~64帧）                                          │
│  ④ BBox 插值补全 + 高斯平滑                                       │
│  ⑤ 正方形裁剪（224×224）+ 左手水平翻转                             │
│  ⑥ 保存 meta.json（offset、scale、original_size）                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼  hand_crops_224/<Gloss>/<VideoID>/R|L/
┌─────────────────────────────────────────────────────────────────┐
│            阶段 2：inference_to_db.py（3D 参数提取）              │
│                                                                  │
│  ① 加载 S2HAND 预训练模型（EfficientNet-B3 编码器）               │
│  ② ImageNet 归一化预处理                                          │
│  ③ 批量推理（batch_size=32），虚拟单位相机内参                     │
│  ④ 输出：joints(N,21,3), vertices(N,778,3), pose(N,48), shape   │
│  ⑤ 左手 X 轴取反（还原真实左手坐标）                               │
│  ⑥ 保存为压缩 .npz 数据库                                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼  database_npz/<Gloss>/<VideoID>/data_R.npz
┌─────────────────────────────────────────────────────────────────┐
│              阶段 3：render_3d.py（GLB 渲染导出）                  │
│                                                                  │
│  ① 从 meta.json 计算全局空间位移（X/Y 像素→米，Z 透视反比）        │
│  ② V_local + J_root + Offset_global 合并最终顶点坐标              │
│  ③ 三次样条插值升频（×2）                                         │
│  ④ 左手面片绕序翻转 faces[:,[0,2,1]]                              │
│  ⑤ 双手合并网格，右手天蓝色，左手珊瑚橙                             │
│  ⑥ 逐帧导出 .glb 三维模型文件                                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼  glb_models/<Word>_<VideoID>/frame_000.glb ...
┌─────────────────────────────────────────────────────────────────┐
│                输出：逐帧 GLB 三维手部模型动画序列                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 技术栈详解

| 技术/库 | 版本要求 | 用途 | 所在阶段 |
|---------|---------|------|---------|
| **MediaPipe** | ≥0.10 | 手部 21 关键点实时检测 | 阶段 1 |
| **OpenCV** | ≥4.5 | 视频解码、图像处理、JPEG 保存 | 阶段 1 |
| **SciPy** | ≥1.7 | 高斯平滑、三次样条插值 | 阶段 1&3 |
| **NumPy** | ≥1.21 | 全程数值计算 | 全程 |
| **tqdm** | any | 进度条显示 | 全程 |
| **Pillow (PIL)** | ≥9.0 | 图像读取与翻转 | 阶段 2 |
| **MindSpore** | ≥2.0 | 深度学习推理框架（主，华为昇腾） | 阶段 2 |
| **msadapter** | ≥0.3 | MindSpore 的 PyTorch 接口适配层 | 阶段 2 |
| **PyTorch** | ≥1.12 | 深度学习推理框架（备用） | 阶段 2 |
| **S2HAND** | 自研版 | 单视角手部 3D 重建模型（EfficientNet-B3） | 阶段 2 |
| **MANO** | v1.2 | 参数化手部模型（Max Planck Institute） | 阶段 2&3 |
| **trimesh** | ≥3.15 | 3D 网格处理、GLB 格式导出 | 阶段 3 |
| **chumpy** | 0.70 | 可微分自动求导（MANO 库内部使用） | MANO |

### 核心模型：S2HAND

S2HAND 是一个**单视角（Single View）手部三维重建**神经网络，架构如下：

```
输入: 224×224 RGB 图像
  ↓
EfficientNet-B3（特征提取主干，ImageNet 预训练）
  ↓
多头回归器
  ├── MANO 姿态参数 pose (48维: 3维全局旋转 + 45维手指关节旋转)
  ├── MANO 形状参数 shape/betas (10维 PCA 手型系数)
  ├── 相机参数 scale + translation (重投影用)
  └── 21个3D关节坐标 joints (21×3)
  ↓
MANO 解码器（将参数转为778个顶点坐标）
  ↓
输出: vertices (778×3) + joints (21×3)
```

### 核心模型：MANO

MANO（**M**odel of the **A**rticulated **N**ands and their **O**bjects）是马克斯·普朗克研究所开发的参数化手部形状模型：

- **778 个顶点**构成手部网格表面
- **21 个关节**构成手部骨骼树
- **48 维姿态参数**（pose）控制手型弯曲
- **10 维形状参数**（betas）控制手的胖瘦/长短
- 使用**线性混合蒙皮（LBS）**将参数映射到顶点坐标

---

## 5. 目录结构说明

```
3d_sign_language/
├── src_3d/                          ← 核心业务代码（本项目自研）
│   ├── config_3d.py                 ← 🔧 全局配置文件（路径、超参数）
│   ├── preprocess_3d.py             ← 📹 阶段1：视频 → 手部裁剪图
│   ├── inference_to_db.py           ← 🧠 阶段2：裁剪图 → 3D 参数数据库
│   ├── render_3d.py                 ← 🎨 阶段3：3D 参数 → GLB 动画文件
│   ├── thinking_3d.md               ← 📝 设计思路笔记（15条核心决策）
│   └── logs/                        ← 运行日志目录
│       ├── inference_*.log
│       └── preprocess_3d_*.log
│
└── mano_v1_2/                       ← MANO 官方手部模型库（第三方）
    ├── __init__.py
    ├── LICENSE.txt
    ├── models/
    │   ├── MANO_RIGHT.pkl           ← ⚠️ 右手模型参数（需单独下载，不在 git 中）
    │   ├── info.txt
    │   └── LICENSE.txt
    └── webuser/
        ├── serialization.py         ← MANO 模型序列化/反序列化
        ├── lbs.py                   ← 线性混合蒙皮（LBS）核心算法
        ├── verts.py                 ← 顶点装饰器（verts_decorated）
        ├── posemapper.py            ← Rodrigues 旋转映射
        ├── smpl_handpca_wrapper_HAND_only.py  ← 独手 MANO 加载器
        └── smpl_handpca_wrapper.py  ← SMPL+H 全身模型加载器
```

**外部依赖目录**（不在本仓库中，需手动配置）：

```
/home/jm802/sign_language/
├── data/
│   ├── ASL_Citizen/
│   │   ├── videos/                  ← 原始 MP4 视频
│   │   └── splits/train.csv         ← 数据集分割元数据
│   ├── hand_crops_224/              ← 阶段1输出（手部裁剪图）
│   └── idx2name_300.txt             ← 300个目标词汇列表
├── s2hand_code/
│   ├── S2HAND/                      ← S2HAND 源代码
│   └── checkpoints/checkpoints.pth  ← 预训练权重
└── result_3d/
    ├── database_npz/                ← 阶段2输出（3D参数数据库）
    └── glb_models/                  ← 阶段3输出（GLB 动画文件）
```

---

## 6. 三阶段流水线详解

### 6.1 阶段一：视频预处理 `preprocess_3d.py`

**目标**：将原始 MP4 视频，提取出左右手分离的、224×224 的手部裁剪图序列，并保存空间元数据。

#### 完整执行流程

```
读取视频所有帧（全量读入内存）
    ↓
逐帧运行 MediaPipe 双手检测
    ↓
左右手身份追踪（最近邻 + 丢帧重置）
    ↓
有效帧区间检测（去首尾无手帧）
    ↓
自适应采样（16~64帧）
    ↓
BBox 线性插值补全丢帧
    ↓
BBox 高斯平滑（sigma=1.7）
    ↓
正方形扩展裁剪（×1.45）+ meta.json 保存
    ↓
左手水平翻转（伪装为右手）
    ↓
缩放至 224×224，保存 JPEG
```

#### 代码详解：`extract_bbox_from_landmarks()`

```python
def extract_bbox_from_landmarks(landmarks, image_width, image_height):
    # MediaPipe 返回的是归一化坐标（0~1），需要乘以图像尺寸转为像素坐标
    x_coords = [np.clip(lm.x * image_width, 0, image_width-1) for lm in landmarks.landmark]
    y_coords = [np.clip(lm.y * image_height, 0, image_height-1) for lm in landmarks.landmark]
    
    # 取21个关键点的极值，形成包围盒
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    return [x_min, y_min, x_max, y_max]
```

> **关键点**：`lm.x` 是归一化值（0~1），必须乘以图像宽高才能得到真实像素坐标。`np.clip` 防止越界。

#### 代码详解：`expand_and_clamp_bbox()` — 正方形裁剪与元数据

```python
def expand_and_clamp_bbox(bbox, expand_ratio, image_width, image_height):
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    center_x = x_min + width / 2
    center_y = y_min + height / 2
    
    # S2HAND 需要正方形输入，取长边作为正方形边长，再乘以扩展系数
    # 扩展系数 1.45 是为了包含手腕以下的部分手臂，帮助 S2HAND 判断手掌朝向
    max_side = max(width, height) * expand_ratio
    
    new_x_min = int(center_x - max_side / 2)
    new_y_min = int(center_y - max_side / 2)
    
    # 保存关键元数据：
    # offset = 正方形左上角的原始图像坐标
    # scale  = 正方形边长（像素）
    # 后续通过公式还原坐标：P_original = (P_224 / 224) × scale + offset
    meta_params = {
        "offset": [new_x_min, new_y_min],
        "scale": max_side,
        "original_size": [image_width, image_height]
    }
    
    # 裁剪框必须 clamp 在图像边界内（超出则截断）
    crop_box = [
        max(0, new_x_min),
        max(0, new_y_min),
        min(image_width - 1, int(center_x + max_side / 2)),
        min(image_height - 1, int(center_y + max_side / 2))
    ]
    
    return crop_box, meta_params
```

> **为什么保存 offset 和 scale？** 阶段 2 的推理是在 224×224 局部坐标系下进行的，要还原手在原始视频中的真实位置，必须记录这个"从哪里裁的"信息。

#### 代码详解：左右手身份追踪（核心算法）

```python
# ========= 初始化（第一帧，或丢失后重新找回）=========
if last_pos["R"] is None and last_pos["L"] is None:
    current_detected_hands.sort(key=lambda h: h["center"][0])  # 按X坐标排序
    
    if len(current_detected_hands) == 1:
        # 只检测到一只手：比较它在画面左半边还是右半边
        if current_detected_hands[0]["center"][0] < image_w / 2:
            bbox_R = current_detected_hands[0]["box"]  # 画面左边 = 右手（镜像关系）
        else:
            bbox_L = current_detected_hands[0]["box"]  # 画面右边 = 左手
    else:
        # 检测到两只手：X坐标最小（最左）的是右手
        bbox_R = current_detected_hands[0]["box"]
        bbox_L = current_detected_hands[-1]["box"]

# ========= 实时追踪（后续帧）=========
else:
    for hand in current_detected_hands:
        # 计算当前手到上一帧"右手位置"和"左手位置"的欧氏距离
        dist_to_R = np.linalg.norm(np.array(hand["center"]) - np.array(last_pos["R"])) \
                    if last_pos["R"] else 9999
        dist_to_L = np.linalg.norm(np.array(hand["center"]) - np.array(last_pos["L"])) \
                    if last_pos["L"] else 9999
        
        # 哪边近就是哪只手（最近邻匹配）
        if dist_to_R < dist_to_L:
            bbox_R = hand["box"]
        else:
            bbox_L = hand["box"]

# ========= 丢帧保护：连续丢失2帧则重置，防止"幽灵坐标"影响后续匹配 =========
if bbox_R is None:
    lost_frames["R"] += 1
    if lost_frames["R"] > 2:
        last_pos["R"] = None  # 重置：下次当初始帧处理
```

> **为什么初始化时画面左边是右手？** 视频拍摄时，被拍摄者面对镜头，其右手在镜头的左边（镜像关系）。

#### 代码详解：自适应采样

```python
# 只在有手的区间内采样，去掉首尾准备手势的无效帧
start_f, end_f = valid_indices[0], valid_indices[-1]
active_len = end_f - start_f + 1

# 目标帧数：最少16帧（太短不足以表达手势），最多64帧（防止数据量过大）
target_count = max(16, min(active_len, 64))

# 均匀采样：从有效区间等间距取 target_count 个索引
sample_indices = np.round(np.linspace(start_f, end_f, target_count)).astype(int).tolist()
```

#### 代码详解：`smooth_bboxes()` — 高斯平滑

```python
def smooth_bboxes(bboxes, sigma=1.7):
    bboxes_arr = np.array(bboxes, dtype=np.float64)
    smoothed_bboxes = np.zeros_like(bboxes_arr)
    
    # 对 x_min, y_min, x_max, y_max 四个维度分别做一维高斯平滑
    # 高斯核会对相邻帧的坐标做加权平均，消除突然跳变
    for i in range(4):
        smoothed_bboxes[:, i] = gaussian_filter1d(bboxes_arr[:, i], sigma=sigma)
    
    return smoothed_bboxes.tolist()
```

> **sigma=1.7 的选择**：经验测试值。sigma 越大平滑越强但会产生延迟感（手势跟不上）；sigma 越小则保留了噪声。1.0~2.0 是手语场景的推荐范围。

#### 输出文件结构

```
hand_crops_224/
└── HELLO/                           ← 手语词条（Gloss）
    └── video_001/                   ← 视频 ID
        ├── R/                       ← 右手序列
        │   ├── frame_0000.jpg       ← 224×224 JPEG（RGB）
        │   ├── frame_0001.jpg
        │   ├── ...
        │   └── meta.json            ← 每帧的 offset、scale、original_size
        └── L/                       ← 左手序列（图片已水平翻转）
            ├── frame_0000.jpg
            └── meta.json
```

`meta.json` 示例：
```json
[
    {
        "offset": [142, 87],
        "scale": 312.5,
        "original_size": [640, 480],
        "frame_idx": 0,
        "file_name": "frame_0000.jpg",
        "original_start_frame": 5
    },
    ...
]
```

---

### 6.2 阶段二：3D 参数推理 `inference_to_db.py`

**目标**：用 S2HAND 模型对每张裁剪图进行推理，得到手部的 3D MANO 参数，保存为 `.npz` 数据库。

#### 双后端自动切换机制

```python
try:
    import mindspore as ms
    import msadapter.pytorch as torch    # msadapter 让 MindSpore 兼容 PyTorch API
    BACKEND = "mindspore-msadapter"      # 优先使用华为昇腾 NPU
except Exception:
    try:
        import torch
        BACKEND = "torch-fallback"       # 降级到 PyTorch（GPU/CPU）
    except Exception:
        BACKEND = "none"                 # 安全退出，不崩溃

# 关键：将 torch 注入到 sys.modules，让 S2HAND 内部的 import torch 自动解析到 msadapter
if torch is not None:
    sys.modules["torch"] = torch
```

> **设计意图**：代码同时支持华为昇腾 NPU（通过 msadapter 适配）和 NVIDIA GPU（原生 PyTorch），无需修改 S2HAND 源码。

#### `ConfigStub` — S2HAND 参数配置详解

```python
class ConfigStub:
    def __init__(self):
        # 推理时只输出关节坐标和网格顶点，不计算热力图和光照（节省计算）
        self.test_requires = ['joints', 'verts']
        
        # 直接预测 MANO 参数（pose + shape），而非关键点检测
        self.regress_mode = 'mano'
        
        # False = 为每个人预测个性化手型，而不是用平均手型
        self.use_mean_shape = False
        
        # False = 关闭 2D 热力图注意力机制
        # 原因1：大规模数据减少计算
        # 原因2：2D 预测不准时会引入错误的注意力
        self.use_2d_as_attention = False
        
        # False = 不加后处理微调网络（保持 MANO 参数纯净，防止手指奇怪弯曲）
        self.use_pose_regressor = False
        
        # FreiHand 坐标系（与下载的预训练权重对齐）
        self.train_datasets = ['FreiHand']
        
        # EfficientNet-B3 的标准输入尺寸
        self.image_size = 224
```

#### 代码详解：图像预处理 `preprocess_image()`

```python
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

def preprocess_image(img_pil, side):
    # 步骤1: 确保 RGB 格式，resize 到 224×224（双线性插值，质量较好）
    img = img_pil.convert('RGB').resize((224, 224), Image.BILINEAR)
    
    # 步骤2: 左手在阶段1已经水平翻转了，这里再翻转一次是为了... 
    # 实际上阶段1保存时就已经翻转了，这里是双重保险（等效于不翻转）
    if side == 'L':
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    # 步骤3: 转为 numpy float32，归一化到 0~1
    arr = np.asarray(img, dtype=np.float32) / 255.0
    
    # 步骤4: HWC (Height×Width×Channel) → CHW (Channel×Height×Width)，CNN 标准格式
    arr = np.transpose(arr, (2, 0, 1))
    
    # 步骤5: ImageNet 标准化（减均值除标准差），对齐 EfficientNet 预训练时的数据分布
    # output = (input - mean) / std
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    
    return torch.tensor(arr, dtype=torch.float32)
```

#### 代码详解：批量推理 `run_batch_inference()`

```python
@no_grad()   # 推理时关闭梯度计算，节省显存
def run_batch_inference(model, device, folder_path, side, batch_size=32):
    img_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    
    all_joints, all_vertices, all_pose, all_shape = [], [], [], []
    
    for i in range(0, len(img_files), batch_size):  # 每次处理32帧
        batch_files = img_files[i : i + batch_size]
        
        # 打包成 batch tensor: (N, 3, 224, 224)
        batch_imgs = [preprocess_image(Image.open(os.path.join(folder_path, f)), side) 
                      for f in batch_files]
        imgs = torch.stack(batch_imgs).to(device)
        
        # 虚拟相机内参：4×4 单位矩阵
        # 因为是在 224×224 局部坐标系推理，假设相机在原点且无畸变
        # 真实的相机投影参数不需要，我们只要相对手型
        K = torch.eye(4).unsqueeze(0).repeat(len(batch_files), 1, 1).to(device)
        
        # 核心推理：一次前向传播
        out = model.predict_singleview(imgs, None, K, 'test', ['joints', 'verts'], None, None)
        
        # 拉回 CPU 存储
        all_joints.append(out['joints'].cpu().numpy())
        all_vertices.append(out['vertices'].cpu().numpy())
        all_pose.append(out['pose'].cpu().numpy())
        all_shape.append(out['shape'].cpu().numpy())
        
        del imgs, K, out  # 手动释放 GPU 显存
    
    joints   = np.concatenate(all_joints,   axis=0)  # (N, 21, 3)
    vertices = np.concatenate(all_vertices, axis=0)  # (N, 778, 3)
    pose     = np.concatenate(all_pose,     axis=0)  # (N, 48)
    shape    = np.concatenate(all_shape,    axis=0)  # (N, 10)
    
    # ⚠️ 左手坐标还原：将伪右手的X轴取反，得到正确的左手坐标
    # joints 的形状是 (N, 21, 3)，第3维的第0个元素是 X 坐标
    if side == 'L':
        joints[:, :, 0]   = -joints[:, :, 0]    # X = -X
        vertices[:, :, 0] = -vertices[:, :, 0]  # X = -X
    
    return {"joints": joints, "vertices": vertices, "pose": pose, "shape": shape}
```

#### 输出数据格式

每个词条/视频保存两个文件（如果左右手都检测到的话）：

```
database_npz/
└── HELLO/
    └── video_001/
        ├── data_R.npz   ← 右手数据
        └── data_L.npz   ← 左手数据
```

每个 `.npz` 文件包含：

| 键名 | 形状 | 含义 |
|------|------|------|
| `joints` | (N, 21, 3) | 21个关节的3D坐标，以手腕为原点 |
| `vertices` | (N, 778, 3) | 778个网格顶点的3D坐标 |
| `pose` | (N, 48) | MANO姿态参数（3维全局旋转+45维手指旋转） |
| `shape` | (N, 10) | MANO形状参数（10维手型PCA系数） |
| `is_flipped` | bool | 是否为翻转处理的左手（data_L.npz 为 True） |

---

### 6.3 阶段三：3D 渲染导出 `render_3d.py`

**目标**：将 `.npz` 数据库中的 3D 参数，结合 `meta.json` 的空间信息，渲染为可导入 3D 软件的 GLB 文件序列。

#### 代码详解：`get_global_offset_from_meta()` — 像素→3D空间坐标

```python
def get_global_offset_from_meta(meta_path, num_frames, spatial_scale=0.15, depth_scale=0.2):
    with open(meta_path, 'r') as f:
        meta_data = json.load(f)
    
    offsets = np.zeros((num_frames, 3))
    base_scale = meta_data[0]['scale']  # 第一帧的 BBox 边长，作为深度基准
    
    for meta in meta_data:
        idx = meta['frame_idx']
        current_scale = meta['scale']
        
        # BBox 中心点的像素坐标
        # offset 是左上角坐标，加上 scale/2 得到中心点
        cx = meta['offset'][0] + meta['scale'] / 2
        cy = meta['offset'][1] + meta['scale'] / 2
        img_w, img_h = meta['original_size']
        
        # ===== X 轴：将像素中心偏移归一化到世界坐标 =====
        # (cx - img_w/2)：将原点从图像左上角移到图像中心
        # / current_scale：除以当前帧 BBox 大小，近似于透视归一化
        # × spatial_scale：缩放到真实世界米制单位（经验系数）
        tx = spatial_scale * (cx - img_w / 2) / current_scale
        
        # ===== Y 轴：注意2D和3D的Y轴方向相反 =====
        # 2D 图像坐标：向下为正
        # 3D 世界坐标：向上为正
        # 因此取负号翻转
        ty = -spatial_scale * (cy - img_h / 2) / current_scale
        
        # ===== Z 轴：基于透视投影反比关系估算深度 =====
        # 透视投影公式：图像上的大小 s ∝ 1/Z
        # 因此：Z ∝ 1/s，BBox 越大说明手离镜头越近
        # tz = depth_scale × (1/base_scale - 1/current_scale) × base_scale（化简后）
        # 让第一帧的 Z = 0，后续帧相对第一帧计算深度变化
        tz = depth_scale * (1.0 - base_scale / (current_scale + 1e-6))
        
        offsets[idx] = [tx, ty, tz]
    
    return offsets  # shape: (num_frames, 3)
```

#### 最终顶点坐标计算

```python
# V_local 是 S2HAND 在局部坐标系（224×224正方形内）输出的顶点坐标
# joints[:,0:1,:] 是每帧手腕（根关节）的位置
# V_local + J_root 把手从"以原点为中心"恢复到"以手腕为基准"的位置
v_r_local = data_r['vertices'] + data_r['joints'][:, 0:1, :]   # (N, 778, 3)

# 加上全局位移（从 meta.json 计算得到的世界坐标偏移）
# global_offset[:, None, :] 广播到所有778个顶点
v_r_global = v_r_local + global_offset_r[:, None, :]             # (N, 778, 3)
```

用数学公式表示：

$$V_{final} = (V_{local} + J_{root}) + Offset_{global}$$

#### 代码详解：三次样条插值升频

```python
def interpolate_vertices_sequence(verts, factor):
    orig_len = len(verts)
    target_len = orig_len * factor   # factor=2 则帧数翻倍
    
    x_orig = np.linspace(0, 1, orig_len)
    x_new  = np.linspace(0, 1, target_len)
    
    if orig_len >= 4:
        # 三次样条插值：在相邻关键帧之间生成平滑过渡帧
        # 三次（cubic）比线性插值更平滑，关节不会突然折角
        f = interp1d(x_orig, verts, axis=0, kind='cubic')
    else:
        # 帧数太少时退化为线性插值（避免样条震荡）
        f = interp1d(x_orig, verts, axis=0, kind='linear')
    
    return f(x_new)  # shape: (target_len, 778, 3)
```

#### 代码详解：左手面片翻转技巧

```python
faces_base = load_mano_faces()         # 右手面片，形状 (1538, 3)，每行是三角面的3个顶点索引
faces_left = faces_base[:, [0, 2, 1]]  # 交换第2和第3个顶点索引

# 为什么这样能得到左手面片？
# 三角面的顶点顺序决定面的法向量方向（右手定则）
# 原始: A→B→C 方向，法向量朝外（右手正面）
# 翻转: A→C→B 方向，法向量朝内（需要翻转X轴后变成左手正面）
# 配合前面对 vertices X 轴取反，就完美还原了左手的几何形状
```

#### 代码详解：渲染循环

```python
faces_base = load_mano_faces()
faces_left = faces_base[:, [0, 2, 1]]

for i in tqdm(range(num_frames)):
    combined_mesh = None
    
    # 渲染右手（天蓝色）
    if v_r_smooth is not None and i < len_r_smooth:
        mesh_r = trimesh.Trimesh(
            vertices=v_r_smooth[i],  # 当前帧的778个顶点坐标
            faces=faces_base,
            process=False            # 不做自动修复（保持原始拓扑）
        )
        mesh_r.visual.face_colors = [135, 206, 250, 255]  # RGBA 天蓝色
        combined_mesh = mesh_r
    
    # 渲染左手（珊瑚橙）
    if v_l_smooth is not None and i < len_l_smooth:
        mesh_l = trimesh.Trimesh(
            vertices=v_l_smooth[i],
            faces=faces_left,        # 使用翻转后的面片
            process=False
        )
        mesh_l.visual.face_colors = [255, 127, 80, 255]   # RGBA 珊瑚橙
        
        if combined_mesh is None:
            combined_mesh = mesh_l
        else:
            combined_mesh = combined_mesh + mesh_l  # 合并双手网格
    
    if combined_mesh is not None:
        combined_mesh.export(os.path.join(output_folder, f"frame_{i:03d}.glb"))
```

> **双手独立帧计数器的意义**：左右手的动作时长不一定相同。比如右手打了 60 帧，左手只打了 40 帧。用 `i < len_r_smooth` 和 `i < len_l_smooth` 分别控制，就能实现"左手先停，右手继续"的自然效果，避免残影或闪烁。

---

## 7. MANO 手部模型库解析

### 7.1 `lbs.py` — 线性混合蒙皮（LBS）

LBS 是 3D 动画中将骨骼运动传递到皮肤网格的标准算法。

```python
def verts_core(pose, v, J, weights, kintree_table, want_Jtr=False, xp=chumpy):
    # 步骤1: 计算每个骨骼关节的全局变换矩阵（前向运动学）
    A, A_global = global_rigid_transformation(pose, J, kintree_table, xp)
    
    # 步骤2: 混合蒙皮矩阵 T = A × weights.T
    # weights 是每个顶点对每个骨骼的蒙皮权重 (778顶点 × 16骨骼)
    T = A.dot(weights.T)
    
    # 步骤3: LBS 核心公式（向量化实现）
    # 每个顶点的最终位置 = Σ(蒙皮权重 × 骨骼变换 × 模板顶点)
    v = (T[:,0,:] * rest_shape_h[0,:] +
         T[:,1,:] * rest_shape_h[1,:] +
         T[:,2,:] * rest_shape_h[2,:] +
         T[:,3,:] * rest_shape_h[3,:]).T
    return v
```

### 7.2 `posemapper.py` — Rodrigues 旋转

MANO 的姿态参数使用**轴角表示（Axis-Angle）**，每个关节的旋转用一个 3 维向量表示（方向是旋转轴，长度是旋转角度）。Rodrigues 公式将轴角转为 3×3 旋转矩阵：

```python
class Rodrigues(ch.Ch):
    def compute_r(self):
        # cv2.Rodrigues 完成轴角→旋转矩阵的转换
        return cv2.Rodrigues(self.rt.r)[0]
    
    def compute_dr_wrt(self, wrt):
        # 支持自动求导（训练时用），返回雅可比矩阵
        if wrt is self.rt:
            return cv2.Rodrigues(self.rt.r)[1].T
```

### 7.3 `smpl_handpca_wrapper_HAND_only.py` — PCA 降维手型

手的 45 维关节旋转参数（15个手指关节 × 每个关节3轴）维度太高，MANO 用 PCA 降维：

```python
def load_model(fname_or_dict, ncomps=6, flat_hand_mean=False):
    hands_components = smpl_data['hands_components']  # PCA基向量矩阵 (45 × 45)
    
    # 只取前 ncomps 个主成分（默认6个，能表达绝大多数手势变化）
    selected_components = np.vstack((hands_components[:ncomps]))  # (6, 45)
    
    # pose_coeffs[3:9] 是6维PCA系数
    # full_hand_pose = PCA系数 × PCA基向量 + 手势均值
    full_hand_pose = pose_coeffs[rot:(rot+ncomps)].dot(selected_components)
    smpl_data['fullpose'] = ch.concatenate((pose_coeffs[:rot], hands_mean + full_hand_pose))
    # 最终: fullpose = [3维全局旋转] + [45维手指旋转] = 48维
```

> **rot=3**：前3维是全局旋转（手腕朝向）；`ncomps=6`：6维PCA系数控制手指弯曲形态。

---

## 8. 配置文件详解 `config_3d.py`

```python
# ============ 服务器数据路径（运行前必须修改） ============
DATA_ROOT  = "/home/jm802/sign_language/data"
VIDEO_DIR  = os.path.join(DATA_ROOT, "ASL_Citizen", "videos")
CSV_PATH   = os.path.join(DATA_ROOT, "ASL_Citizen", "splits", "train.csv")

HAND_CROP_DIR = os.path.join(DATA_ROOT, "hand_crops_224")   # 阶段1输出
BBOX_CACHE_DIR = os.path.join(DATA_ROOT, "bbox_cache")
DB_ROOT    = "/home/jm802/sign_language/result_3d/database_npz"   # 阶段2输出
BASE_DB    = DB_ROOT + "/"

# ============ 关键模型路径 ============
S2HAND_PATH     = "/home/jm802/sign_language/s2hand_code/S2HAND"
PERTAINED_MODEL = "/home/jm802/sign_language/s2hand_code/checkpoints/checkpoints.pth"
RIGHT_PKL_PATH  = "/home/jm802/sign_language/3d_sign_language/mano_v1_2/models/MANO_RIGHT.pkl"
TARGET_WORDS_FILE = "/home/jm802/sign_language/data/idx2name_300.txt"

# ============ 图像处理参数 ============
IMAGE_SIZE        = 224    # S2HAND 和 EfficientNet-B3 的标准输入尺寸
BBOX_EXPAND_RATIO = 1.45   # BBox 扩展系数（含手腕及部分手臂，S2HAND需要）

# ============ MediaPipe 参数 ============
MP_DETECTION_CONFIDENCE = 0.5   # 检测置信度阈值（0~1，越高越严格）
MP_TRACKING_CONFIDENCE  = 0.5   # 追踪置信度阈值
MP_MAX_NUM_HANDS        = 2     # 最多检测几只手（手语需要双手）

# ============ 并行与质量参数 ============
NUM_WORKERS  = 4     # 多进程数（建议 = CPU核心数/2，避免内存不足）
JPEG_QUALITY = 95    # JPEG 压缩质量（95分，接近无损）
SKIP_EXISTING = True # 断点续跑开关（True=跳过已处理的视频）

# ============ 平滑参数 ============
SIGMA = 1.7   # 高斯平滑标准差（1.0~2.0，越大越平滑但有延迟）

# ============ 渲染参数 ============
FACTOR        = 2      # 帧数升频倍数（2=翻倍，3=三倍放慢动作）
SCALE_FACTOR  = 0.003  # 像素→米的缩放比（调小=动作幅度更小）
FOCAL_CONSTANT = 20.0  # Z轴深度估计焦距常数（调大=Z方向运动更明显）
```

---

## 9. 核心设计决策解析（15个）

这些设计决策来自 `thinking_3d.md`，是整个系统最有价值的工程思考。

### 决策1：有效帧区间检测与自适应采样

**问题**：手语视频开头和结尾通常有准备动作（手放下、举手），这些帧没有手语内容。

**方案**：找到第一个和最后一个有手检测的帧，只在这个区间内采样，强制采样到 16~64 帧。

### 决策2：联合外接矩形固定裁剪框

**问题**：如果每帧用自己的 BBox 裁剪，背景会随手移动而移动，产生画面晃动。

**方案**：计算64帧所有手部 BBox 的最大公共外接正方形，用固定框裁剪所有帧。手在框内移动，背景固定，手部相对位移轨迹平滑。

### 决策3：最近邻追踪 + 丢帧重置

**问题**：MediaPipe 不保证左右手标签的连贯性，跨帧可能混淆。

**方案**：记录上一帧的左右手中心坐标，下一帧的手按照"离谁最近就是谁"分配身份。连续丢失超过2帧则重置，防止"幽灵坐标"。

### 决策4：左手镜像处理

**问题**：S2HAND 只有右手预训练权重，无左手模型。

**方案**：
1. 阶段1：保存裁剪图时，将左手图像水平翻转（伪装成右手）
2. 阶段2：用右手权重推理，得到"伪右手"的 3D 坐标
3. 阶段2：将推理结果的 X 轴取反，还原真实左手坐标
4. 阶段3：将右手面片绕序翻转 `faces[:,[0,2,1]]`，修正法向量方向

### 决策5：虚拟单位相机内参

**问题**：S2HAND 需要相机内参矩阵 K，但我们不知道原始拍摄相机的参数。

**方案**：使用 4×4 单位矩阵作为虚拟内参。因为推理在 224×224 的局部裁剪坐标系下进行，相机等效于在原点无畸变，单位矩阵完全合适。

### 决策6：透视 Z 深度估计

**问题**：从 2D 图像无法直接获得手的深度信息。

**方案**：利用透视投影原理——**图像中物体的大小与距离成反比**。BBox 越大，手越近；BBox 越小，手越远。

$$Z_{relative} = depth\_scale \times \left(1 - \frac{scale_{base}}{scale_{current}}\right)$$

### 决策7：三次样条插值升频

**问题**：原始视频通常 25~30fps，S2HAND 输出的 3D 动画跳跃感明显。

**方案**：对778个顶点的时间序列做三次样条插值，帧数乘以 `FACTOR=2`，动作更流畅。帧数少于4时退化为线性插值（避免样条震荡）。

### 决策8：面片绕序翻转

**公式**：`faces_left = faces_right[:, [0, 2, 1]]`

仅交换三角面第2和第3个顶点，等效于翻转法向量方向，将右手面片变为左手面片，无需额外模型文件。

### 决策9：ImageNet 归一化对齐预训练分布

**问题**：EfficientNet-B3 在 ImageNet 上预训练，输入数据有特定的均值/方差分布。

**方案**：推理时对输入图像做与训练时完全相同的归一化（mean=[0.485,0.456,0.406]，std=[0.229,0.224,0.225]），确保模型的"期望输入"与实际输入一致，输出精度更高。

### 决策10：batch_size=32 的显存友好推理

**问题**：一个视频可能有 64 帧，全部放入 GPU 可能导致显存溢出（OOM）。

**方案**：每次只处理 32 帧，处理完后手动 `del imgs, K, out` 并调用 `cuda.empty_cache()`，让显存及时回收，适配显存较小的设备。

### 决策11：BBox 插值补全丢帧

**问题**：MediaPipe 在某些帧可能检测不到手（光线差、遮挡等），导致该帧 BBox 为 `None`。

**方案**：用 `np.interp` 对四个坐标维度分别做线性插值，将 `None` 帧用前后有效帧的坐标插值填充，保证序列完整性。

### 决策12：每个词汇只取第一个视频

**问题**：ASL Citizen 数据集每个词汇有多个录制版本（不同人），处理所有版本耗时过长。

**方案**：从 CSV 扫描时，每找到一个词汇的第一个视频就将其从"待处理名单"中划去，找够 300 个后立即停止扫描，极大减少 I/O 开销。

### 决策13：多进程并行处理（8个 Worker）

**问题**：300 个视频串行处理太慢，每个视频处理约 1~3 秒，总计约 10 分钟。

**方案**：用 `multiprocessing.Pool` 启动 8 个 Worker，每个 Worker 在初始化时创建独立的 MediaPipe 实例（`init_worker`），避免多进程共享对象的线程安全问题。

### 决策14：左手图像在哪个阶段翻转

**设计细节**：左手水平翻转在**阶段1（保存 JPEG 时）**完成，而 `meta.json` 记录的 `offset` 和 `scale` 是**翻转前的原始坐标**。

- 阶段2 推理时，再次翻转（`preprocess_image` 中的 `FLIP_LEFT_RIGHT`）恢复正常
- 实际上两次翻转等效于不翻转，图像以正确的"右手"姿态输入 S2HAND
- 推理后X轴取反，即可得到正确的左手 3D 坐标

### 决策15：双手独立帧计数器

**问题**：左右手的有效帧数可能不同，若按同一帧数循环会出现"一只手消失了但另一只手还在渲染空帧"的问题。

**方案**：分别记录 `len_r_smooth` 和 `len_l_smooth`，总帧数取两者最大值。每一帧渲染时独立判断 `i < len_r_smooth` 和 `i < len_l_smooth`，哪只手结束哪只手就不渲染，完全独立。

---

## 10. 数据流转与坐标系变换

整个流水线经历了四次坐标系变换，这是最容易混淆的地方。

```
阶段          坐标系               原点位置              说明
─────────────────────────────────────────────────────────────────
阶段1前       原始图像像素坐标      图像左上角(0,0)        MP 检测结果
阶段1后       224×224 局部坐标     正方形左上角(0,0)      裁剪后的图像坐标
阶段2         MANO 局部坐标        手腕关节(0,0,0)        S2HAND 输出坐标（手腕为原点）
阶段3         世界坐标（米制）      画面中心(0,0,0)        最终 GLB 坐标
```

### 坐标变换公式汇总

**从 224 局部坐标 → 原始图像像素坐标（逆变换，验证用）**：
$$P_{original} = \frac{P_{224}}{224} \times scale + offset$$

**从原始图像像素坐标 → 世界坐标（阶段3使用）**：
$$tx = spatial\_scale \times \frac{c_x - W/2}{scale_{current}}$$
$$ty = -spatial\_scale \times \frac{c_y - H/2}{scale_{current}}$$
$$tz = depth\_scale \times \left(1 - \frac{scale_{base}}{scale_{current}}\right)$$

**最终顶点世界坐标**：
$$V_{world} = (V_{mano} + J_{root}) + [tx, ty, tz]$$

### 左手坐标变换链

```
原始左手视频帧
    ↓ 阶段1：水平翻转图像（x' = W - x）
伪右手图像 224×224
    ↓ 阶段2：S2HAND 推理
伪右手 MANO 坐标（joints[:,·,0] 为正值的"右手"坐标）
    ↓ 阶段2：X 轴取反（x = -x）
真实左手局部坐标（镜像还原）
    ↓ 阶段3：加全局位移（offset 是翻转前记录的真实位置）
真实左手世界坐标
    ↓ 阶段3：面片绕序翻转 faces[:,[0,2,1]]
正确朝向的左手网格（法向量指向外侧）
```

---

## 11. 使用实操指南（从零开始）

### 11.1 环境准备

#### 系统要求

| 要求 | 说明 |
|------|------|
| 操作系统 | Linux（推荐 Ubuntu 20.04/22.04），不支持 Windows 运行 |
| Python | 3.8 ~ 3.10 |
| GPU | NVIDIA GPU（≥6GB 显存）或华为昇腾 NPU |
| CUDA | 11.3 ~ 11.8（PyTorch 后端）|
| 内存 | ≥16GB RAM |
| 磁盘 | 原始视频 + 输出约需 50GB |

#### 安装 Python 依赖

```bash
# 阶段1 依赖
pip install mediapipe>=0.10.0
pip install opencv-python>=4.5
pip install scipy>=1.7
pip install numpy>=1.21
pip install tqdm

# 阶段2 依赖（二选一）
# 方案A：MindSpore + msadapter（华为昇腾推荐）
pip install mindspore>=2.0
pip install msadapter>=0.3

# 方案B：PyTorch（NVIDIA GPU）
pip install torch>=1.12 torchvision

# 阶段3 依赖
pip install trimesh>=3.15
pip install scipy   # 已装过，确认即可
pip install Pillow>=9.0

# MANO 库依赖（需要 chumpy 旧版）
pip install chumpy==0.70
```

### 11.2 获取必要的模型文件

#### MANO 模型

1. 前往 [MANO 官网](https://mano.is.tue.mpg.de/) 注册账号并下载 `MANO_RIGHT.pkl`
2. 将文件放置到：`3d_sign_language/mano_v1_2/models/MANO_RIGHT.pkl`

#### S2HAND 预训练权重

1. 克隆 S2HAND 代码：
```bash
git clone https://github.com/TerenceCYJ/S2HAND.git /home/jm802/sign_language/s2hand_code/S2HAND
```
2. 下载预训练权重 `checkpoints.pth` 放到：
```
/home/jm802/sign_language/s2hand_code/checkpoints/checkpoints.pth
```

#### ASL Citizen 数据集

1. 前往 [ASL Citizen 官网](https://www.microsoft.com/en-us/research/project/asl-citizen/) 申请并下载
2. 解压后放置到：
```
/home/jm802/sign_language/data/ASL_Citizen/
    ├── videos/       ← MP4 文件
    └── splits/
        └── train.csv
```

#### 目标词汇列表

创建 `/home/jm802/sign_language/data/idx2name_300.txt`，每行格式：
```
0 HELLO
1 THANK_YOU
2 PLEASE
...
```

### 11.3 修改配置文件

打开 `src_3d/config_3d.py`，根据实际路径修改：

```python
DATA_ROOT         = "/your/path/to/data"       # 数据根目录
VIDEO_DIR         = DATA_ROOT + "/ASL_Citizen/videos"
CSV_PATH          = DATA_ROOT + "/ASL_Citizen/splits/train.csv"
HAND_CROP_DIR     = DATA_ROOT + "/hand_crops_224"
DB_ROOT           = "/your/path/to/result_3d/database_npz"
BASE_DB           = DB_ROOT + "/"

S2HAND_PATH       = "/your/path/to/S2HAND"
PERTAINED_MODEL   = "/your/path/to/checkpoints.pth"
RIGHT_PKL_PATH    = "/your/path/to/mano_v1_2/models/MANO_RIGHT.pkl"
TARGET_WORDS_FILE = DATA_ROOT + "/idx2name_300.txt"
```

### 11.4 运行阶段一：视频预处理

```bash
cd /path/to/3d_sign_language/src_3d
python preprocess_3d.py
```

**预期输出**：
```
✅ 成功加载了 300 个目标词汇进行过滤。
Total videos to process: 300 (每个词语仅保留1个视频)
🚀 正式任务开始: 正在处理 300 个视频样本。
Processing Videos: 100%|████████████| 300/300 [05:23<00:00,  0.93it/s]
Successful: 298  Failed: 2
```

**检查输出**：
```bash
ls /your/path/to/hand_crops_224/HELLO/video_001/R/
# 应该看到：frame_0000.jpg frame_0001.jpg ... meta.json
```

### 11.5 运行阶段二：3D 参数推理

```bash
cd /path/to/3d_sign_language
python src_3d/inference_to_db.py
```

**预期输出**：
```
Backend: mindspore-msadapter   (或 torch-fallback)
Building 3D Database: 100%|████| 300/300 [47:32<00:00,  9.51s/it]
🎉 提取任务完成！成功: 295, 失败: 5
```

**检查输出**：
```bash
ls /your/path/to/database_npz/HELLO/video_001/
# 应该看到：data_R.npz data_L.npz

python3 -c "
import numpy as np
d = np.load('/your/path/to/database_npz/HELLO/video_001/data_R.npz')
print('joints:', d['joints'].shape)    # (N, 21, 3)
print('vertices:', d['vertices'].shape) # (N, 778, 3)
"
```

### 11.6 运行阶段三：3D 渲染导出

```bash
cd /path/to/3d_sign_language
python src_3d/render_3d.py
```

**预期输出**：
```
📂 探测完成，共发现 295 个视频样本待渲染。
📦 [1/295] 正在渲染词条: HELLO (ID: video_001)
🚀 正在导出 3D 模型序列至: /home/jm802/.../HELLO_video_001/,(总帧数: 128)
100%|████████████| 128/128 [00:12<00:00, 10.24it/s]
✅ 导出成功！共生成 128 个 3D 模型文件。
```

**查看 GLB 文件**：
将 `frame_000.glb` 拖入 [https://gltf-viewer.donmccurdy.com/](https://gltf-viewer.donmccurdy.com/) 即可在线预览。

### 11.7 单词条测试（快速验证）

如果只想测试某一个词条，可以在 Python 中直接调用渲染函数：

```python
# 在 Python 交互式终端或 Jupyter Notebook 中运行
import sys
sys.path.insert(0, '/path/to/3d_sign_language')
from src_3d.render_3d import export_glb_sequence

export_glb_sequence(
    word_dir="/your/path/to/database_npz/HELLO/video_001/",
    output_folder="/tmp/test_glb/"
)
```

---

## 12. 迁移到 MindSpore 指南

本项目已内置 MindSpore 支持，但需要做以下配置和适配工作。

### 12.1 迁移原理说明

项目使用 **msadapter**（MindSpore 的 PyTorch API 适配层），核心思路是：

```python
import mindspore as ms
import msadapter.pytorch as torch   # 用 msadapter 替代 PyTorch
sys.modules["torch"] = torch        # 让 S2HAND 的 import torch 自动走 msadapter
```

这样 S2HAND 的代码**无需任何修改**即可在 MindSpore 上运行，因为 msadapter 实现了与 PyTorch 相同的 API 接口。

### 12.2 安装 MindSpore 环境

#### 方案A：华为昇腾 NPU 服务器（推荐）

```bash
# 1. 安装 MindSpore Ascend 版本（以 MindSpore 2.3 为例）
pip install mindspore-ascend==2.3.0 \
    -i https://pypi.mirrors.ustc.edu.cn/simple

# 2. 安装 msadapter
pip install msadapter

# 3. 验证安装
python -c "import mindspore as ms; ms.set_context(mode=ms.PYNATIVE_MODE); print(ms.__version__)"
```

#### 方案B：CPU 模式（调试用）

```bash
# CPU 版 MindSpore
pip install mindspore==2.3.0 \
    -i https://pypi.mirrors.ustc.edu.cn/simple
pip install msadapter
```

#### 方案C：GPU 版（NVIDIA）

```bash
pip install mindspore-gpu==2.3.0 \
    -i https://pypi.mirrors.ustc.edu.cn/simple
pip install msadapter
```

### 12.3 当前代码已有的 MindSpore 适配点

查看 `inference_to_db.py` 中已实现的适配：

```python
# ① 后端自动切换（已实现）
try:
    import mindspore as ms
    import msadapter.pytorch as torch
    BACKEND = "mindspore-msadapter"
except Exception:
    import torch
    BACKEND = "torch-fallback"

# ② MindSpore 动态图模式（已实现）
def init_engine():
    if BACKEND == "mindspore-msadapter" and ms is not None:
        ms.set_context(mode=ms.PYNATIVE_MODE)   # 使用 PyNative 动态图，与 msadapter 兼容性最好
    ...

# ③ no_grad 装饰器兼容（已实现）
if torch is not None and hasattr(torch, "no_grad"):
    no_grad = torch.no_grad
else:
    def no_grad():              # 无后端时的空装饰器，不会崩溃
        def _decorator(func):
            return func
        return _decorator

# ④ 显存清理兼容（已实现）
def _empty_cache_if_available():
    if torch is not None and hasattr(torch, 'cuda') and hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()    # MindSpore NPU 上可能不需要，但调用不会报错
```

### 12.4 需要额外适配的部分

目前代码中有几处在 MindSpore 完整迁移时需要注意：

#### ① S2HAND 模型的 `load_model` 函数

S2HAND 使用 PyTorch 的 `checkpoint` 格式加载权重。在 msadapter 下通常可以透明加载，但如果遇到问题：

```python
# 如果 msadapter 的 load_model 报错，可以用 MindSpore 原生方式加载：
import mindspore as ms

# 方法：先用原生 PyTorch 加载，转存为 MindSpore ckpt
# 步骤1（在有PyTorch的环境中执行）：
import torch
ckpt = torch.load("checkpoints.pth", map_location="cpu")
# 将 state_dict 保存为 numpy
import numpy as np
np.save("weights.npy", {k: v.numpy() for k, v in ckpt['model'].items()})

# 步骤2（在 MindSpore 环境中加载）：
import mindspore as ms
weights = np.load("weights.npy", allow_pickle=True).item()
param_dict = {k: ms.Parameter(ms.Tensor(v)) for k, v in weights.items()}
ms.load_param_into_net(model, param_dict)
```

#### ② 张量操作兼容性

msadapter 已覆盖大多数 PyTorch API，但少数操作需要注意：

```python
# PyTorch 写法（原始代码，通常可直接用 msadapter）
K = torch.eye(4).unsqueeze(0).repeat(N, 1, 1).to(device)

# 如果 msadapter 不支持 .to(device)，替换为：
import mindspore as ms
K = ms.Tensor(np.eye(4)[np.newaxis].repeat(N, axis=0), dtype=ms.float32)
```

#### ③ S2HAND 内部 `torch.nn.Module` 迁移

如果需要完全脱离 msadapter，将 S2HAND 的 `nn.Module` 改为 `mindspore.nn.Cell`：

```python
# PyTorch（S2HAND 原始代码）
import torch.nn as nn
class Model(nn.Module):
    def forward(self, x):
        ...

# MindSpore 原生写法
import mindspore.nn as nn
class Model(nn.Cell):
    def construct(self, x):    # construct 替代 forward
        ...
```

#### ④ 自动求导上下文

```python
# PyTorch
with torch.no_grad():
    out = model(imgs)

# MindSpore 原生
from mindspore import context
# 在 PYNATIVE_MODE 下，直接调用推理即可（无梯度计算上下文）
# 或使用：
from mindspore.nn import WithEvalCell
eval_net = WithEvalCell(model)
out = eval_net(imgs)
```

### 12.5 完整 MindSpore 原生迁移路线图

如果目标是完全用 MindSpore 原生 API（不依赖 msadapter）重写，建议按以下步骤进行：

```
第1步：替换 EfficientNet-B3 主干网络
  └── 使用 MindSpore 官方 MindCV 库的 EfficientNet-B3
      pip install mindcv
      from mindcv.models import create_model
      backbone = create_model('efficientnet_b3', pretrained=True)

第2步：重写 MANO 解码器
  └── 将 chumpy（NumPy-based）的 LBS 实现
      改为 MindSpore Tensor 操作（纯矩阵运算，难度低）

第3步：重写模型推理类
  └── nn.Module → nn.Cell
      forward() → construct()
      model.eval() → 无需显式调用（PYNATIVE_MODE 默认非训练状态）

第4步：重写权重加载
  └── torch.load() → ms.load_checkpoint()
      需要先将 .pth 转为 .ckpt 格式

第5步：替换数据流
  └── torch.Tensor → ms.Tensor
      DataLoader → mindspore.dataset
```

### 12.6 MindSpore 迁移验证脚本

迁移完成后，用以下脚本快速验证输出一致性：

```python
import numpy as np

# 用 PyTorch 跑一批图像，保存输出
# (在有 PyTorch 的机器上执行)
torch_joints = np.load("torch_joints.npy")   # shape: (4, 21, 3)

# 用 MindSpore 跑相同图像，保存输出
ms_joints = np.load("ms_joints.npy")          # shape: (4, 21, 3)

# 计算均方误差（应该 < 1e-4）
mse = np.mean((torch_joints - ms_joints) ** 2)
print(f"MSE between PyTorch and MindSpore outputs: {mse:.6f}")
assert mse < 1e-4, "输出不一致，请检查权重加载或模型结构"
print("✅ 迁移验证通过！")
```

---

## 13. 常见问题与排错

### Q1：运行阶段1时报 `ModuleNotFoundError: No module named 'mediapipe'`

```bash
pip install mediapipe
# 若网络慢：
pip install mediapipe -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q2：运行阶段1时报 `找不到词汇表文件`

确认 `config_3d.py` 中 `TARGET_WORDS_FILE` 路径正确，且文件格式每行是 `<数字> <词条>` 格式（如 `0 HELLO`）。

### Q3：运行阶段2时显示 `Backend: none`

说明 MindSpore 和 PyTorch 都没有安装。必须至少安装一个：
```bash
pip install torch torchvision  # 最简单的方案
```

### Q4：运行阶段2时报 `FileNotFoundError: Cannot find S2HAND source path`

确认 `config_3d.py` 中 `S2HAND_PATH` 正确，且该目录下存在 `examples/models_new.py` 文件。

### Q5：`MANO_RIGHT.pkl` 找不到

该文件因版权原因不能公开分发，需前往 [https://mano.is.tue.mpg.de/](https://mano.is.tue.mpg.de/) 注册下载。

### Q6：GLB 文件在 Windows 上无法预览

推荐使用以下在线工具：
- [https://gltf-viewer.donmccurdy.com/](https://gltf-viewer.donmccurdy.com/) — 直接拖入浏览器
- Blender（免费 3D 软件）→ File → Import → glTF 2.0

### Q7：生成的 3D 手型奇怪（手指穿插或变形）

可能原因：
1. **BBox 裁剪不准**：调低 `BBOX_EXPAND_RATIO`（如从 1.45 降到 1.3）
2. **预训练权重不匹配**：确认 `checkpoints.pth` 是 FreiHand 数据集训练的版本
3. **手部遮挡严重**：部分手势动作本身就有手指遮挡，属于正常现象

### Q8：左右手坐标位置互换

检查 `preprocess_3d.py` 中的初始化逻辑。如果你的视频是**自拍模式**（镜像），初始化规则需要调整：
```python
# 当前规则：画面左边 = 右手（适合第三方拍摄视角）
# 如果是自拍镜像视角，改为：画面左边 = 左手
if current_detected_hands[0]["center"][0] < image_w / 2:
    bbox_L = current_detected_hands[0]["box"]   # 改这里
else:
    bbox_R = current_detected_hands[0]["box"]
```

### Q9：动画播放时出现手部"跳跃"

原因：三次样条插值在极端手势变化时可能产生震荡（龙格现象）。

解决方案：降低 `FACTOR` 值，或改用线性插值：
```python
# render_3d.py 中，将 kind='cubic' 改为 kind='linear'
f = interp1d(x_orig, verts, axis=0, kind='linear')
```

### Q10：如何在 Windows 本地测试阶段3（渲染）？

阶段3（`render_3d.py`）不依赖深度学习库，可以在 Windows 上运行：
```bash
pip install trimesh scipy numpy
python render_3d.py
```
前提是已经有 `.npz` 数据库文件（可以从服务器下载）。

---

## 14. 依赖安装速查表

```bash
# ============ 完整环境一键安装（Linux，PyTorch 后端）============
pip install \
    mediapipe>=0.10.0 \
    opencv-python>=4.5.0 \
    scipy>=1.7.0 \
    numpy>=1.21.0 \
    tqdm \
    Pillow>=9.0.0 \
    torch>=1.12.0 \
    torchvision \
    trimesh>=3.15.0 \
    chumpy==0.70

# ============ MindSpore 后端替换（二选一）============
# 替换 torch torchvision 为：
pip install mindspore>=2.0.0 msadapter
```

| 库名 | 用途 | 阶段 | 安装优先级 |
|------|------|------|-----------|
| mediapipe | 手部关键点检测 | 阶段1 | **必须** |
| opencv-python | 视频解码 | 阶段1 | **必须** |
| scipy | 高斯平滑、样条插值 | 阶段1,3 | **必须** |
| numpy | 数值计算 | 全程 | **必须** |
| tqdm | 进度条 | 全程 | 推荐 |
| Pillow | 图像读取 | 阶段2 | **必须** |
| torch 或 mindspore | 深度学习推理 | 阶段2 | **必须** |
| msadapter | PyTorch→MindSpore 适配 | 阶段2 | MindSpore时必须 |
| trimesh | 3D 网格导出 | 阶段3 | **必须** |
| chumpy | MANO 内部依赖 | MANO库 | **必须** |

---

> **文档结束**  
> 如有疑问，请参阅 `src_3d/thinking_3d.md` 中的设计思路笔记，或查阅各源文件的代码注释。  
> 本项目运行环境为 Linux 服务器（`/home/jm802/`），Windows 仅支持阶段3的渲染部分。
