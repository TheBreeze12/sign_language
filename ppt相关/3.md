# S²HAND 项目保姆级详解文档

## 目录

1. [项目背景](#1-项目背景)
2. [解决的核心问题](#2-解决的核心问题)
3. [系统架构总体说明](#3-系统架构总体说明)
4. [技术栈](#4-技术栈)
5. [核心代码详解](#5-核心代码详解)
6. [如何使用](#6-如何使用)
7. [迁移到 MindSpore 指南](#7-迁移到-mindspore-指南)

---

## 1. 项目背景

### 1.1 研究领域

S²HAND（Self-Supervised Hand reconstruction）是一个基于自监督学习的 3D 手部重建项目，发表于 CVPR 2021。论文全称：

> **Model-based 3D Hand Reconstruction via Self-Supervised Learning**
> 作者：Yujin Chen, Zhigang Tu, Di Kang, Linchao Bao, Ying Zhang, Xuefei Zhe, Ruizhi Chen, Junsong Yuan

### 1.2 问题背景

3D 手部重建是计算机视觉中的重要任务，广泛应用于：
- **手语识别**：将手部动作转化为语义信息
- **人机交互**：VR/AR 中的手势控制
- **机器人操作**：理解人手抓取姿态
- **动作捕捉**：影视和游戏中的手部动画

传统方法依赖大量 **人工标注的 3D 关节点数据**，这些标注成本极高且容易出错。S²HAND 的核心创新在于：**首次证明了无需人工 3D 标注即可训练出精确的 3D 手部重建网络**。

### 1.3 MANO 手部模型

项目基于 **MANO（hand Model with Articulated and Non-rigid deformations）** 参数化手部模型：
- 用 **10 个形状参数（shape/beta）** 控制手的大小、粗细等
- 用 **30 个姿态参数（pose/theta）**（PCA 压缩后）控制 16 个关节的旋转
- 输出 **778 个顶点** 和 **1538 个三角面片** 的手部网格
- 输出 **21 个关键点**（16 个关节 + 5 个指尖）

---

## 2. 解决的核心问题

### 2.1 核心挑战

| 挑战 | 传统方法 | S²HAND 方案 |
|------|---------|------------|
| 3D 标注昂贵 | 需要大量 3D GT | 自监督，无需 3D 标注 |
| 2D 检测噪声大 | 直接使用，精度受限 | 多种一致性损失过滤噪声 |
| 手部姿态不合理 | 后处理约束 | TSA 姿态先验损失 |
| 纹理/光照未建模 | 仅估计几何 | 联合估计纹理和光照 |

### 2.2 核心思路

```
输入: 单张 RGB 手部图像
  ↓
2D 关键点检测 (OpenPose, 离线)  →  提供 2D 监督信号
  ↓
网络联合估计:
  - 3D 手部姿态 (pose)
  - 手部形状 (shape)
  - 表面纹理 (texture)
  - 光照参数 (lighting)
  - 相机视角 (camera)
  ↓
通过 2D-3D 一致性 + 渲染一致性 进行自监督训练
  ↓
输出: 3D 手部网格 + 关节点位置
```

### 2.3 自监督信号来源

1. **2D 关键点一致性**：预测的 3D 关节投影到 2D 后应与 OpenPose 检测结果一致
2. **骨骼方向一致性**：预测的骨骼方向应与 2D 检测的骨骼方向一致
3. **渲染一致性**：用预测的网格+纹理渲染出的图像应与输入图像一致
4. **姿态先验**：手指关节角度应在解剖学合理范围内
5. **形状先验**：手部形状应接近平均形状
6. **网格平滑性**：Laplacian 正则化保证网格光滑

---

## 3. 系统架构总体说明

### 3.1 整体架构图

```
                        ┌─────────────────────────────────────────────┐
                        │              S²HAND Model                   │
                        │                                             │
  RGB Image ──────────► │  ┌──────────┐    ┌──────────────────────┐   │
  [B,3,224,224]         │  │  RGB2HM  │    │      Encoder         │   │
                        │  │(Hourglass│    │  (EfficientNet-B3)   │   │
                        │  │ Network) │    │                      │   │
                        │  └────┬─────┘    └──────┬───────────────┘   │
                        │       │                  │                   │
                        │  Heatmaps           features    low_features│
                        │  [B,21,64,64]      [B,1536]    [B,32,56,56]│
                        │       │                  │           │      │
                        │       │           ┌──────┴──────┐    │      │
                        │       │           │ HandDecoder  │    │      │
                        │       │           │  (MANO)     │    │      │
                        │       │           └──────┬──────┘    │      │
                        │       │                  │           │      │
                        │       │          joints, verts  ┌────┴────┐ │
                        │       │          pose, shape    │Texture &│ │
                        │       │          scale, trans   │ Light   │ │
                        │       │          rot            │Estimator│ │
                        │       │                  │      └────┬────┘ │
                        │       │                  │           │      │
                        │       ▼                  ▼           ▼      │
                        │   2D Keypoints    3D Mesh      Texture+Light│
                        └─────────────────────────────────────────────┘
                                    │              │           │
                                    ▼              ▼           ▼
                              ┌─────────────────────────────────────┐
                              │         Loss Functions              │
                              │  · 2D Keypoint Loss (OpenPose)     │
                              │  · Bone Direction Loss             │
                              │  · TSA Pose Prior Loss             │
                              │  · Shape Regularization            │
                              │  · Scale Regularization            │
                              │  · Photometric Loss (渲染一致性)    │
                              │  · SSIM Loss                       │
                              │  · Laplacian Smoothness            │
                              │  · 2D-3D Keypoint Consistency      │
                              └─────────────────────────────────────┘
```

### 3.2 目录结构

```
S2HAND/
├── README.md                              # 项目说明
├── efficientnet_pt/                       # EfficientNet 骨干网络
│   ├── __init__.py
│   ├── model.py                           # EfficientNet 模型定义
│   └── utils.py                           # 辅助工具
├── examples/
│   ├── train.py                           # 【入口】训练/评估主脚本
│   ├── models_new.py                      # 【核心】完整模型架构
│   ├── losses.py                          # 【核心】损失函数定义
│   ├── traineval_util.py                  # 【核心】训练评估工具
│   ├── train_utils.py                     # 模型加载/冻结工具
│   ├── util.py                            # 可视化工具
│   ├── draw_util.py                       # 误差曲线绘制
│   ├── options/
│   │   └── train_options.py               # 命令行参数定义
│   ├── config/                            # JSON 配置文件
│   │   ├── FreiHAND/
│   │   │   ├── SSL-e2e.json               # 端到端训练配置
│   │   │   └── evaluation.json            # 评估配置
│   │   └── HO3D/
│   │       ├── SSL-shape.json             # 阶段1: 形状学习
│   │       ├── SSL-kp.json                # 阶段2: 关键点学习
│   │       ├── SSL-finetune.json          # 阶段3: 微调
│   │       ├── SSL-e2e.json               # 端到端训练
│   │       └── evaluation.json            # 评估配置
│   ├── utils/
│   │   ├── fh_utils.py                    # FreiHAND 工具函数
│   │   ├── freihandnet.py                 # MANO 解码器网络
│   │   ├── hand_3d_model.py               # MANO 模型核心实现
│   │   ├── hand_model.py                  # 手部姿态工具
│   │   ├── net_hg.py                      # Hourglass 网络
│   │   ├── laplacianloss.py               # Laplacian 正则化
│   │   └── mano_core/                     # MANO 底层实现
│   │       ├── lbs.py                     # 线性混合蒙皮
│   │       ├── mano_loader.py             # MANO 模型加载
│   │       ├── posemapper.py              # 姿态映射
│   │       └── verts.py                   # 顶点处理
│   ├── openpose_detector/                 # OpenPose 2D 检测
│   │   ├── hand_detect.py                 # 手部检测脚本
│   │   └── src/
│   │       ├── hand.py
│   │       ├── model.py
│   │       └── util.py
│   └── pytorch_ssim/                      # SSIM 损失
│       └── __init__.py
```

### 3.3 数据流

```
1. 数据加载 (data/dataset.py)
   → 读取图像、相机内参、OpenPose 2D 关键点

2. 数据预处理 (traineval_util.py::data_dic)
   → 归一化、坐标转换、构建训练字典

3. 前向传播 (models_new.py::Model)
   → RGB2HM: 图像 → 2D 热力图 → 2D 关键点
   → Encoder: 图像 → 特征向量 [B,1536]
   → HandDecoder: 特征 → MANO 参数 → 3D 网格
   → TextureLightEstimator: 低层特征 → 纹理 + 光照

4. 投影 (traineval_util.py::trans_proj)
   → 3D 关节点 → 相机投影 → 2D 关节点

5. 损失计算 (traineval_util.py::loss_func)
   → 计算所有自监督损失

6. 反向传播 + 优化
```

---

## 4. 技术栈

### 4.1 核心依赖

| 组件 | 版本/说明 |
|------|----------|
| Python | 3.6+ |
| PyTorch | 1.1+ |
| CUDA | GPU 训练必需 |
| tqdm | 进度条 |
| tensorboardX | 训练可视化 |
| transforms3d | 3D 变换工具 |
| chumpy | MANO 模型依赖 |
| scikit-image | 图像处理 |
| neural_renderer | 可微分渲染器（自定义 fork） |
| pytorch-openpose | 2D 关键点检测（离线） |

### 4.2 关键技术

| 技术 | 用途 | 对应文件 |
|------|------|---------|
| EfficientNet-B3 | 图像特征提取 | `efficientnet_pt/model.py` |
| Stacked Hourglass | 2D 热力图估计 | `utils/net_hg.py` |
| MANO 参数化模型 | 3D 手部表示 | `utils/hand_3d_model.py` |
| Neural Renderer | 可微分渲染 | 外部依赖 |
| Rodrigues 旋转 | 轴角→旋转矩阵 | `utils/hand_3d_model.py` |
| Linear Blend Skinning | 网格变形 | `utils/hand_3d_model.py` |
| Cotangent Laplacian | 网格平滑正则化 | `utils/laplacianloss.py` |
| Integral Regression | 热力图→坐标（可微） | `util.py` |

---

## 5. 核心代码详解

### 5.1 模型主体 (`models_new.py::Model`)

这是整个系统的核心类，组合了所有子模块：

```python
class Model(nn.Module):
    def __init__(self, filename_obj, args):
        # 1. 2D 热力图估计模块（可选）
        self.rgb2hm = RGB2HM()          # Stacked Hourglass Network

        # 2. 图像编码器
        self.encoder = Encoder()          # EfficientNet-B3, 输出 [B,1536]

        # 3. MANO 手部解码器
        self.hand_decoder = MyHandDecoder(inp_neurons=1536)

        # 4. 纹理+光照估计器
        self.texture_light_from_low = texture_light_estimator(mode='surf')

        # 5. 可选：2D 注意力机制
        self.heatmap_attention = heatmap_attention(out_len=1536)
```

前向传播流程 (`predict_singleview` 方法)：

```
Step 1: 热力图估计
  images [B,3,224,224] → pad到256 → RGB2HM → heatmaps [B,21,64,64]
  → integral regression → 2D关键点 [B,21,2]

Step 2: 特征提取
  images → EfficientNet-B3 → features [B,1536] + low_features [B,32,56,56]
  (可选) features = features * heatmap_attention(encoding)

Step 3: MANO 解码
  features → HandDecoder → joints[B,21,3], verts[B,778,3], faces[B,1538,3]
                            pose[B,30], shape[B,10], scale, trans, rot

Step 4: 纹理光照估计
  low_features → texture_light_estimator → textures[B,1538,3], lights[B,11]

Step 5: 渲染（Neural Renderer）
  verts + faces + textures → rendered_image, depth, silhouette
```

### 5.2 MANO 手部模型 (`utils/hand_3d_model.py::rot_pose_beta_to_mesh`)

这是将 MANO 参数转换为 3D 网格的核心函数：

```python
def rot_pose_beta_to_mesh(rots, poses, betas):
    """
    输入:
        rots:  [B, 3]  全局旋转（轴角表示）
        poses: [B, 30] PCA 压缩的姿态参数
        betas: [B, 10] 形状参数
    输出:
        jv:    [B, 21+778, 3]  关节点+顶点坐标
        faces: [B, 1538, 3]    三角面片索引
        poses: [B, 16, 3]      TSA 姿态角度
    """
```

计算步骤详解：

```
1. 加载 MANO_RIGHT.pkl（预计算的手部模型参数）
   - v_template: 平均手部网格 [778, 3]
   - shapedirs:  形状基 [778, 3, 10]
   - posedirs:   姿态修正基 [778, 3, 135]
   - J_regressor: 关节回归矩阵 [16, 778]
   - weights:    蒙皮权重 [778, 16]
   - hands_components: PCA 基 [30, 45]
   - hands_mean: 姿态均值 [45]

2. PCA 解压缩姿态
   full_poses = hands_mean + poses @ hands_components  → [B, 15, 3]
   加上根关节旋转 → [B, 16, 3]

3. 形状变形
   v_shaped = v_template + betas @ shapedirs  → [B, 778, 3]

4. 姿态变形
   pose_weights = rodrigues(poses) - I  → [B, 135]
   v_posed = v_shaped + posedirs @ pose_weights  → [B, 778, 3]

5. 关节位置计算
   J_posed = v_shaped @ J_regressor  → [B, 16, 3]

6. 前向运动学（Forward Kinematics）
   对每个关节计算 4x4 变换矩阵，沿运动链传播

7. 线性混合蒙皮（Linear Blend Skinning）
   v_final = Σ(weight_i * Transform_i) @ v_posed

8. 全局旋转
   v_final = Rots @ v_final

9. 添加指尖关节
   从网格顶点中取 5 个指尖位置，插入关节列表
   最终: 21 个关节 + 778 个顶点
```

### 5.3 Stacked Hourglass 网络 (`utils/net_hg.py`)

用于从 RGB 图像估计 2D 关键点热力图：

```python
class Net_HM_HG(nn.Module):
    """
    2 阶段堆叠沙漏网络
    输入: [B, 3, 256, 256]
    输出: heatmaps [B, 21, 64, 64] × 2 阶段
    """
    # 初始卷积: 3→64, stride=2
    # 残差块: 64→128→256
    # 最大池化: stride=2
    # 2 个 Hourglass 模块（4 层深度）
```

Hourglass 模块结构（4 层递归）：

```
输入 [256, 64, 64]
  ├── 上路径: Residual blocks（保持分辨率）
  └── 下路径: MaxPool → Residual → 递归Hourglass → Residual → Upsample
最终: 上路径 + 下路径（跳跃连接）
```

Integral Regression（可微分关键点提取）：
将热力图转换为坐标（soft-argmax），heatmap [B,21,64,64] → softmax → 加权求和 → [B,21,2]，比 argmax 可微分，支持端到端训练。

### 5.4 MANO 解码器 (`utils/freihandnet.py::MyPoseHand`)

将特征向量回归为 MANO 参数的网络：

```python
class MyPoseHand(nn.Module):
    """
    输入: features [B, 1536]
    输出: joints, verts, faces, pose, shape, scale, trans, rot, tsa_poses
    """
    def __init__(self, inp_neurons=1536):
        # 共享基础层
        self.base_layers:  1536 → 1024 → BN → ReLU → 512 → BN → ReLU

        # 多头回归（从 512 维特征分别回归各参数）
        self.pose_reg:   512 → 128 → ReLU → 30   # MANO 姿态参数
        self.shape_reg:  512 → 128 → ReLU → 10   # MANO 形状参数
        self.scale_reg:  512 → 128 → ReLU → 32 → 1    # 缩放因子
        self.trans_reg:  512 → 128 → ReLU → 32 → 3    # 平移向量
        self.rot_reg:    512 → 128 → ReLU → 32 → 3    # 全局旋转
```

关键设计：scale 和 trans 的初始化非常精心：
- `scale` 偏置初始化为 0.95（手部在相机坐标系中的典型尺度）
- `trans` 的 z 分量偏置初始化为 0.65（手部到相机的典型距离）

### 5.5 纹理与光照估计器 (`models_new.py::texture_light_estimator`)

```python
class texture_light_estimator(nn.Module):
    """
    从低层特征估计表面纹理和光照参数
    输入: low_features [B, 32, 56, 56]  (EfficientNet 低层特征)
    输出:
        textures [B, 1538, 3]  每个面片的 RGB 颜色
        lights   [B, 11]       光照参数
    """
    # 卷积降维: Conv(32→48,k=10,s=4) → ReLU → MaxPool → Conv(48→64,k=3) → ReLU → MaxPool
    # → [B,64,2,2] → flatten → [B,256]
    # 纹理回归: 256 → 64 → ReLU → 1538*3 (初始化接近肤色均值)
    # 光照回归: 256 → 64 → ReLU → 11
    # 11个参数: [环境光强度, 方向光强度, 环境光RGB(3), 方向光RGB(3), 光照方向(3)]
```

### 5.6 损失函数详解 (`traineval_util.py::loss_func` + `losses.py`)

#### 5.6.1 2D 关键点损失 (`open_2dj`)

```python
# 预测的 3D 关节投影到 2D 后，与 OpenPose 检测结果对比
open_2dj_distance = sqrt(sum((open_2dj - j2d)², dim=2))  # [B, 21]
# Huber-like 损失（减少离群点影响）
distance = where(d < 5, d²/10, d - 2.5)
# 置信度加权（OpenPose 检测置信度）
# 手腕和指尖权重更高: [2,1,1,1,1.5,1,1,1,1.5,...]
loss = sum(distance * confidence²) / sum(confidence²)
```

#### 5.6.2 TSA 姿态先验损失 (`tsa_pose_loss`)

```python
def tsa_pose_loss(tsaposes):
    """
    Tilt-Swing-Azimuth 姿态先验, tsaposes: [B, 16, 3]
    为每根手指的每个关节设定解剖学合理范围:
    - 食指/中指/无名指/小指: 弯曲角 0~100°, 侧摆 ±5~10°
    - 拇指: 特殊范围（更大的活动空间）
    超出范围的角度产生惩罚，方位角(azimuth)权重×2
    """
    error = max(0, pose - max_limit) + max(0, min_limit - pose)
    loss = mean(error * [1, 1, 2])  # 方位角权重加倍
```

#### 5.6.3 骨骼方向一致性损失 (`bone_direction_loss`)

```python
def bone_direction_loss(j2d, open_2dj, open_2dj_con):
    """
    确保预测的 2D 骨骼方向与 OpenPose 检测一致
    通过 20×21 的邻接矩阵计算 20 根骨骼的方向向量
    归一化后计算方向差异，用置信度加权
    """
    bone_vec_nm = normalize(j2d @ adjacency_matrix)
    bone_open_nm = normalize(open_2dj @ adjacency_matrix)
    loss = mean(||bone_vec_nm - bone_open_nm||² * confidence)
```

#### 5.6.4 其他关键损失

- **2D-3D 关键点一致性 (`kp_cons`)**：热力图分支的 2D 关键点 vs 3D 分支投影的 2D 关键点应一致
- **光度一致性 (`texture`)**：渲染图像与输入图像手部区域的 L1 距离
- **SSIM 损失 (`ssim_tex`)**：渲染图像与输入的结构相似性
- **Laplacian 网格平滑 (`triangle`)**：基于余切权重的 Laplacian 正则化，鼓励最小平均曲率
- **形状正则化 (`mshape`)**：shape 参数的 L2 正则化，趋向平均形状
- **尺度正则化 (`mscale`)**：骨骼长度应接近先验值 0.0282

#### 5.6.5 损失权重汇总

| 损失名称 | 权重 | 作用 |
|---------|------|------|
| `open_2dj` | λ=1e-3 | 2D 关键点对齐 |
| `tsa_poses` | λ=1e-1 | 姿态先验约束 |
| `open_bone_direc` | λ=0.1 | 骨骼方向一致性 |
| `mscale` | λ=0.1 | 骨骼长度先验 |
| `mshape` | λ=1e-2 | 形状正则化 |
| `hm_integral` | λ=1e-3 | 热力图关键点损失 |
| `kp_cons` | λ=2e-4 | 2D-3D 一致性 |
| `texture` | λ=5e-3 | 光度一致性 |
| `mtex` | λ=5e-3 | 纹理正则化 |
| `triangle` | λ=0.1 | 网格平滑 |
| `ssim_tex` | λ=1e-3 | SSIM 纹理损失 |
| `mrgb` | λ=1e-3 | 平均 RGB 损失 |

### 5.7 训练流程 (`train.py`)

```python
# 主入口
if __name__ == '__main__':
    args = train_options.parse()           # 解析命令行参数
    args = load_from_json(args)            # 从 JSON 配置覆盖参数
    model = models.Model(obj_path, args)   # 创建模型
    model = load_pretrained(model, args)   # 加载预训练权重
    model = nn.DataParallel(model.cuda())  # 多 GPU 并行
    freeze_model_modules(model, args)      # 冻结指定模块
    main(args.base_path, set_name=args.mode)  # 开始训练/评估
```

训练循环 (`TrainVal`)：

```python
for idx, sample in enumerate(train_loader):
    # 1. 数据预处理
    examples = data_dic(sample, dat_name, set_name, args)

    # 2. 前向传播
    outputs = model(images=examples['imgs'], P=examples['Ps'],
                    task=args.task, requires=requires)

    # 3. 3D→2D 投影
    outputs = trans_proj(outputs, examples['Ks'], dat_name, ...)

    # 4. 计算损失
    loss_dic = loss_func(examples, outputs, dat_name, args)

    # 5. 汇总损失并反向传播
    loss = sum(loss_dic[key] for key in args.losses)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 5.8 分阶段训练策略（以 HO3D 为例）

```
阶段 1: SSL-shape（形状学习）— 120 epochs
  损失: tsa_poses + open_2dj + mscale + open_bone_direc
  模块: joints + verts（仅 3D 分支）
  目标: 学习合理的手部形状和姿态

阶段 2: SSL-kp（关键点学习）— 120 epochs
  损失: hm_integral
  模块: heatmaps（仅 2D 分支）
  目标: 训练热力图估计网络

阶段 3: SSL-finetune（联合微调）— 60 epochs
  损失: 全部损失
  模块: heatmaps + joints + verts + textures + lights
  目标: 联合优化所有模块

或者: SSL-e2e（端到端训练）— 180 epochs
  一次性训练所有模块，使用全部损失
```

学习率调度：
- 初始 LR: 0.001
- Milestones: [30, 60, 90, 120, 150]
- Gamma: 0.5（每到 milestone 乘以 0.5）

---

## 6. 如何使用

### 6.1 环境搭建

```bash
# 1. 创建 Python 环境
conda create -n s2hand python=3.6
conda activate s2hand

# 2. 安装 PyTorch（根据 CUDA 版本选择）
pip install torch==1.1.0 torchvision==0.3.0

# 3. 安装依赖
pip install tqdm tensorboardX transforms3d chumpy scikit-image

# 4. 安装自定义 neural_renderer（注意：必须用作者 fork 的版本）
git clone https://github.com/TerenceCYJ/neural_renderer.git
cd neural_renderer
python setup.py install
cd .. && rm -rf neural_renderer
```

### 6.2 数据准备

#### FreiHAND 数据集

1. 从 [FreiHAND 官网](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html) 下载数据集
2. 修改 `examples/config/FreiHAND/*.json` 中的路径：
   - `freihand_base_path`: 数据集根目录
   - `base_out_path`: 输出目录

#### HO3D 数据集

1. 从 [HO3D 官网](https://www.tugraz.at/index.php?id=40231) 下载数据集
2. 修改 `examples/config/HO3D/*.json` 中的路径：
   - `ho3d_base_path`: 数据集根目录
   - `base_out_path`: 输出目录

#### MANO 模型文件

需要将 `MANO_RIGHT.pkl` 放到 `examples/data/` 目录下。从 [MANO 官网](https://mano.is.tue.mpg.de/) 下载。

#### 离线 2D 关键点检测

```bash
# 方式一：下载作者提供的预检测结果（FreiHAND）
# 下载 freihand-train.json，修改 dataset.py 中的 self.open_2dj_lists 路径

# 方式二：自行检测
# 下载 hand_pose_model.pth 放到 examples/openpose_detector/src/
python examples/openpose_detector/hand_detect.py
```

### 6.3 训练

#### 分阶段训练（推荐，以 HO3D 为例）

```bash
cd S2HAND

# 阶段 1: 形状学习（120 epochs）
python3 ./examples/train.py --config_json examples/config/HO3D/SSL-shape.json

# 阶段 2: 关键点学习（120 epochs）
# 需要修改 SSL-kp.json 中的 pretrain_model 指向阶段 1 的输出
python3 ./examples/train.py --config_json examples/config/HO3D/SSL-kp.json

# 阶段 3: 联合微调（60 epochs）
# 需要修改 SSL-finetune.json 中的 pretrain_model 指向阶段 2 的输出
python3 ./examples/train.py --config_json examples/config/HO3D/SSL-finetune.json
```

#### 端到端训练

```bash
python3 ./examples/train.py --config_json examples/config/HO3D/SSL-e2e.json
```

### 6.4 评估

```bash
# 下载预训练模型
# HO3D: texturehand_ho3d.t7
# FreiHAND: texturehand_freihand.t7

# 修改 evaluation.json 中的 pretrain_model 路径
python3 ./examples/train.py --config_json examples/config/HO3D/evaluation.json
```

### 6.5 输出说明

训练/评估后的输出目录结构：

```
base_out_path/
├── pic/                # 可视化图像
│   ├── train/          # 训练过程可视化
│   └── test/           # 测试结果可视化
├── model/              # 模型检查点 (texturehand_*.t7)
├── obj/                # 3D 网格 OBJ 文件
├── json/               # 预测结果 JSON 文件
│   └── test/epoch/pred.json  # [关节点坐标, 顶点坐标]
└── train.log           # 训练日志
```

### 6.6 JSON 配置文件关键字段说明

```json
{
    "train_datasets": ["HO3D"],          // 训练数据集
    "val_datasets": ["HO3D"],            // 验证数据集
    "ho3d_base_path": "/path/to/HO3D",   // 数据集路径
    "base_out_path": "/path/to/output",   // 输出路径

    "total_epochs": 180,                  // 总训练轮数
    "init_lr": 0.001,                     // 初始学习率
    "lr_steps": [30, 60, 90, 120, 150],   // 学习率衰减节点
    "lr_gamma": 0.5,                      // 衰减系数
    "train_batch": 32,                    // 训练批大小
    "val_batch": 8,                       // 验证批大小

    "losses": ["tsa_poses", "open_2dj", "mscale", ...],  // 使用的损失
    "train_requires": ["heatmaps", "joints", "verts", "textures", "lights"],
    "test_requires": ["heatmaps", "joints", "verts", "textures", "lights"],
    "task": "train",                      // 任务类型: train/test/hm_train
    "mode": ["training"]                  // 模式: training/evaluation
}
```

---

## 7. 迁移到 MindSpore 指南

### 7.1 迁移总览

MindSpore 是华为开源的深度学习框架，与 PyTorch 在 API 设计上有较多相似之处，但也存在关键差异。以下是 S²HAND 迁移的完整路线图。

迁移难度评估：

| 模块 | 难度 | 说明 |
|------|------|------|
| EfficientNet 编码器 | ★★☆ | MindSpore 有官方实现可复用 |
| Hourglass 网络 | ★★☆ | 标准卷积操作，直接映射 |
| MANO 手部模型 | ★★★ | 涉及 numpy/scipy 混合计算，需重写 |
| 损失函数 | ★★☆ | 大部分可直接映射 |
| Neural Renderer | ★★★★★ | 最大难点，需找替代方案或重写 |
| 训练循环 | ★★☆ | API 差异较大但有规律 |
| 数据加载 | ★★☆ | Dataset/DataLoader 映射 |
| Laplacian Loss | ★★★★ | 依赖 scipy sparse，需重写 |

### 7.2 基础 API 映射表

#### 7.2.1 模块与张量

```python
# ============ PyTorch ============          # ============ MindSpore ============
import torch                                 import mindspore as ms
import torch.nn as nn                        import mindspore.nn as nn
from torch.autograd import Variable          # MindSpore 不需要 Variable，直接用 Tensor
torch.tensor([1,2,3])                        ms.Tensor([1,2,3])
torch.zeros(3,4)                             ms.ops.zeros((3,4), ms.float32)
torch.ones(3,4)                              ms.ops.ones((3,4), ms.float32)
x.cuda()                                     # MindSpore 自动管理设备，或用 .to('Ascend')
x.cpu()                                      x.asnumpy()  # 转为 numpy
x.detach()                                   ms.ops.stop_gradient(x)
x.requires_grad = True                       # 通过 Parameter 或 GradOperation 控制
x.view(B, -1)                                x.view(B, -1)  # 或 ms.ops.reshape(x, (B,-1))
x.permute(0,2,1)                             ms.ops.transpose(x, (0,2,1))
x.contiguous()                               # MindSpore 不需要
x.unsqueeze(0)                               ms.ops.expand_dims(x, 0)
torch.cat([a,b], dim=1)                      ms.ops.concat((a,b), axis=1)
torch.stack([a,b], dim=0)                    ms.ops.stack((a,b), axis=0)
torch.split(x, 1, dim=0)                     ms.ops.split(x, axis=0, output_num=N)
torch.matmul(a, b)                           ms.ops.matmul(a, b)
torch.bmm(a, b)                              ms.ops.bmm(a, b)  # 或 ms.ops.matmul
torch.where(cond, a, b)                      ms.ops.where(cond, a, b)
torch.clamp(x, min=0)                        ms.ops.clamp(x, min=0)
torch.sqrt(x)                                ms.ops.sqrt(x)
torch.abs(x)                                 ms.ops.abs(x)
torch.sum(x, dim=1)                          ms.ops.reduce_sum(x, axis=1)
torch.mean(x, dim=1)                         ms.ops.reduce_mean(x, axis=1)
torch.min(x, dim=1)                          ms.ops.min(x, axis=1)  # 返回 (values, indices)
```

#### 7.2.2 网络层

```python
# ============ PyTorch ============          # ============ MindSpore ============
nn.Conv2d(in, out, k, s, p)                  nn.Conv2d(in, out, k, stride=s, pad_mode='pad', padding=p)
nn.ConvTranspose2d(in, out, k, s, p)         nn.Conv2dTranspose(in, out, k, stride=s, pad_mode='pad', padding=p)
nn.BatchNorm2d(n)                            nn.BatchNorm2d(n)
nn.BatchNorm1d(n)                            nn.BatchNorm1d(n)
nn.Linear(in, out)                           nn.Dense(in, out)
nn.ReLU(inplace=True)                        nn.ReLU()
nn.Sigmoid()                                 nn.Sigmoid()
nn.Dropout(p)                                nn.Dropout(keep_prob=1-p)
nn.MaxPool2d(k, stride=s)                    nn.MaxPool2d(k, stride=s)
nn.AvgPool2d(k, stride=s)                    nn.AvgPool2d(k, stride=s)
nn.Upsample(scale_factor=2)                  nn.ResizeBilinear() 或 ops.interpolate
nn.Sequential(*layers)                       nn.SequentialCell(*layers)
nn.ModuleList(modules)                       nn.CellList(modules)
nn.Parameter(data)                           ms.Parameter(data)
nn.functional.interpolate(x, size)           ms.ops.interpolate(x, sizes=size)
nn.functional.mse_loss(a, b)                 nn.MSELoss()(a, b)
nn.functional.l1_loss(a, b)                  nn.L1Loss()(a, b)
nn.functional.relu(x)                        ms.ops.relu(x)
```

#### 7.2.3 训练相关

```python
# ============ PyTorch ============          # ============ MindSpore ============
optim.Adam(params, lr=0.001)                 nn.Adam(params, learning_rate=0.001)
optim.lr_scheduler.MultiStepLR(             nn.piecewise_constant_lr(milestones, lr_values)
    optimizer, milestones, gamma)             # 或自定义 LearningRateSchedule
model.train()                                model.set_train(True)
model.eval()                                 model.set_train(False)
model.parameters()                           model.trainable_params()
torch.save(state, path)                      ms.save_checkpoint(model, path)
torch.load(path)                             ms.load_checkpoint(path)
model.load_state_dict(state)                 ms.load_param_into_net(model, param_dict)
nn.DataParallel(model)                       # MindSpore 用 context.set_auto_parallel_context()
loss.backward()                              # MindSpore 用 GradOperation 或 TrainOneStepCell
optimizer.zero_grad()                        # MindSpore 自动处理
optimizer.step()                             # 包含在 TrainOneStepCell 中
```

### 7.3 逐模块迁移指南

#### 7.3.1 EfficientNet 编码器

```python
# PyTorch 原版
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = EfficientNet.from_name('efficientnet-b3')
        self.pool = nn.AvgPool2d(7, stride=1)
    def forward(self, x):
        features, low_features = self.encoder.extract_features(x)
        features = self.pool(features).view(features.shape[0], -1)
        return features, low_features

# MindSpore 迁移版
# 方案 1: 使用 MindSpore Model Zoo 中的 EfficientNet
# 方案 2: 手动迁移（推荐，因为需要提取中间特征）
import mindspore.nn as nn
import mindspore.ops as ops

class Encoder(nn.Cell):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = EfficientNetB3_MindSpore()  # 需自行迁移或从 model zoo 获取
        self.pool = nn.AvgPool2d(7, stride=1)
    def construct(self, x):
        features, low_features = self.encoder.extract_features(x)
        features = self.pool(features)
        features = features.view(features.shape[0], -1)
        return features, low_features
```

关键注意点：
- `nn.Module` → `nn.Cell`
- `forward()` → `construct()`
- 需要修改 EfficientNet 内部的 `extract_features` 方法以返回中间层特征

#### 7.3.2 MANO 手部模型（最复杂的部分）

```python
# PyTorch 原版核心问题：
# 1. 使用 pickle 加载 MANO_RIGHT.pkl
# 2. 大量 numpy 操作混合 torch 操作
# 3. Variable 包装（MindSpore 不需要）
# 4. scipy.sparse 用于 Laplacian

# MindSpore 迁移策略：
import mindspore as ms
import mindspore.ops as ops
import numpy as np
import pickle

def rot_pose_beta_to_mesh(rots, poses, betas):
    # 1. 加载 MANO 参数（保持 pickle + numpy，仅在初始化时执行一次）
    dd = pickle.load(open(MANO_file, 'rb'), encoding='latin1')

    # 2. 将 numpy 数组转为 MindSpore Tensor（初始化时做一次）
    mesh_mu = ms.Tensor(dd['v_template'], ms.float32).expand_dims(0)
    mesh_pca = ms.Tensor(dd['shapedirs'], ms.float32).expand_dims(0)
    # ... 其他参数类似

    # 3. Rodrigues 旋转公式迁移
    # torch.sum → ops.reduce_sum
    # torch.sin → ops.sin
    # torch.eye → ops.eye
    # torch.matmul → ops.matmul

    # 4. 关键差异：MindSpore 的动态索引
    # PyTorch: R[idx,:,:] = R2[idx,:,:]
    # MindSpore: 需要用 ops.select 或 ops.where 替代
    # 因为 MindSpore 图模式不支持动态索引赋值

    # 5. np.argwhere 需要移到 construct 外部或用 ops.where 替代
```

MANO 模型迁移的核心难点和解决方案：

| 难点 | PyTorch 写法 | MindSpore 解决方案 |
|------|-------------|-------------------|
| `Variable` 包装 | `Variable(torch.from_numpy(...))` | 直接用 `ms.Tensor(...)` |
| 动态索引赋值 | `R[idx,:,:] = R2[idx,:,:]` | `ops.where(mask, R2, R)` |
| `np.argwhere` | 在 forward 中调用 | 用 `ops.where` 替代，或移到预处理 |
| `.cuda()` | 显式 GPU 迁移 | MindSpore 自动管理，通过 `context.set_context(device_target="GPU")` |
| `torch.split` + 列表操作 | 动态列表拼接 | 预分配 Tensor，用 `ops.concat` |

#### 7.3.3 Hourglass 网络

```python
# PyTorch 原版
class Residual(nn.Module):
    def __init__(self, numIn, numOut):
        super().__init__()
        self.conv1 = nn.Conv2d(numIn, numOut//2, kernel_size=1)
        # ...

# MindSpore 迁移
class Residual(nn.Cell):
    def __init__(self, numIn, numOut):
        super().__init__()
        self.conv1 = nn.Conv2d(numIn, numOut//2, kernel_size=1, has_bias=True)
        # 注意: MindSpore 的 Conv2d 默认 has_bias=False
        # PyTorch 的 bias=True 对应 MindSpore 的 has_bias=True
        self.bn = nn.BatchNorm2d(numIn)
        self.relu = nn.ReLU()
        # ...

    def construct(self, x):  # forward → construct
        residual = x
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv1(out)
        # ...
        return out + residual
```

Hourglass 递归结构注意事项：
- MindSpore 图模式下递归深度有限制，但 4 层递归通常没问题
- `nn.ModuleList` → `nn.CellList`
- `nn.Upsample(scale_factor=2)` → `ops.ResizeBilinear()` 或自定义上采样

#### 7.3.4 损失函数迁移

```python
# PyTorch 原版 tsa_pose_loss
def tsa_pose_loss(tsaposes):
    max_nonloss = torch.tensor([...]).float().to(tsaposes.device)
    min_nonloss = torch.tensor([...]).float().to(tsaposes.device)
    errors = torch.where(tsaposes > max_nonloss.unsqueeze(0), ...)
    return torch.mean(errors)

# MindSpore 迁移
def tsa_pose_loss(tsaposes):
    max_nonloss = ms.Tensor([...], ms.float32)  # 不需要 .to(device)
    min_nonloss = ms.Tensor([...], ms.float32)
    errors = ops.where(tsaposes > ops.expand_dims(max_nonloss, 0), ...)
    return ops.reduce_mean(errors)
```

```python
# bone_direction_loss 迁移关键点：
# torch.bmm → ops.bmm 或 ops.matmul
# torch.transpose → ops.transpose
# .mul() → ops.mul 或 *
# .to(torch.uint8) → .astype(ms.uint8)
```

#### 7.3.5 Laplacian Loss（需要重写）

这是迁移难度最大的损失之一，因为原版依赖 `scipy.sparse`：

```python
# 方案 1: 预计算 Laplacian 矩阵，转为稠密 Tensor
# 适用于固定拓扑（MANO 的面片拓扑是固定的）
class LaplacianLoss:
    def __init__(self, faces, vertices):
        # 在初始化时用 numpy/scipy 计算 Laplacian 矩阵
        L_sparse = compute_laplacian(faces, vertices)  # scipy sparse
        # 转为稠密矩阵（778×778 不算大，可以接受）
        self.L_dense = ms.Tensor(L_sparse.toarray(), ms.float32)

    def __call__(self, verts):
        # verts: [B, 778, 3]
        # L_dense: [778, 778]
        Lx = ops.matmul(self.L_dense, verts)  # [B, 778, 3]
        loss = ops.reduce_mean(ops.norm(Lx.view(-1, 3), dim=1))
        return loss

# 方案 2: 用 MindSpore 的 COOTensor 实现稀疏矩阵运算
# MindSpore 支持 COOTensor 和 CSRTensor
```

#### 7.3.6 Neural Renderer（最大挑战）

Neural Renderer 是一个 CUDA 自定义算子库，迁移到 MindSpore 有以下方案：

```
方案 1（推荐）: 使用 MindSpore 生态中的可微分渲染器
  - MindSpore 官方或社区可能有类似实现
  - 或使用 nvdiffrast 等跨框架渲染器

方案 2: 将渲染部分保留为 numpy 操作
  - 渲染结果不参与梯度计算时可行
  - 但光度损失需要渲染梯度，此方案受限

方案 3: 用 MindSpore 自定义算子重写
  - 使用 MindSpore 的 Custom 算子接口包装 CUDA kernel
  - 工作量最大但最灵活

方案 4: 简化方案——去掉渲染相关损失
  - 仅保留几何损失（2D关键点、骨骼方向、姿态先验等）
  - 去掉 texture、ssim_tex、mrgb 等渲染损失
  - 这样完全不需要 Neural Renderer
  - 精度会有一定下降，但大幅降低迁移难度
```

### 7.4 训练循环迁移

```python
# ============ PyTorch 训练循环 ============
model.train()
for data in dataloader:
    outputs = model(data)
    loss = compute_loss(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ============ MindSpore 训练循环 ============
# 方式 1: 使用 Model.train() 高级 API
from mindspore import Model as MSModel
from mindspore.train.callback import LossMonitor

loss_fn = CustomLoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=0.001)
ms_model = MSModel(model, loss_fn, optimizer)
ms_model.train(epoch, dataset, callbacks=[LossMonitor()])

# 方式 2: 自定义训练循环（更灵活，推荐用于本项目）
import mindspore.ops as ops

# 定义前向函数
def forward_fn(images, Ps, Ks, open_2dj, open_2dj_con):
    outputs = model(images, Ps)
    loss = compute_all_losses(outputs, Ks, open_2dj, open_2dj_con)
    return loss

# 创建梯度函数
grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

# 训练步
def train_step(images, Ps, Ks, open_2dj, open_2dj_con):
    loss, grads = grad_fn(images, Ps, Ks, open_2dj, open_2dj_con)
    optimizer(grads)
    return loss

# 训练循环
model.set_train(True)
for epoch in range(total_epochs):
    for data in dataset.create_dict_iterator():
        loss = train_step(data['images'], data['Ps'], ...)
```

### 7.5 数据加载迁移

```python
# ============ PyTorch ============
from torch.utils.data import Dataset, DataLoader

class HandDataset(Dataset):
    def __getitem__(self, idx):
        return sample
    def __len__(self):
        return self.length

loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# ============ MindSpore ============
import mindspore.dataset as ds

# 方式 1: 使用 GeneratorDataset 包装已有 Dataset
class HandDataset:
    def __getitem__(self, idx):
        return sample_tuple  # 返回 tuple 而非 dict
    def __len__(self):
        return self.length

dataset = ds.GeneratorDataset(
    source=HandDataset(),
    column_names=["images", "Ks", "joints", "open_2dj"],
    shuffle=True,
    num_parallel_workers=4
)
dataset = dataset.batch(32, drop_remainder=True)

# 方式 2: 使用 MindRecord 格式（性能更好）
# 先将数据转为 MindRecord，再用 MindDataset 读取
```

### 7.6 迁移检查清单

```
□ 环境准备
  □ 安装 MindSpore（GPU 版本）
  □ 确认 CUDA 版本兼容性

□ 模型迁移
  □ EfficientNet-B3 编码器 (nn.Module → nn.Cell)
  □ Stacked Hourglass 网络
  □ MANO 手部解码器 (MyPoseHand)
  □ rot_pose_beta_to_mesh 函数
  □ texture_light_estimator
  □ heatmap_attention

□ 损失函数迁移
  □ tsa_pose_loss
  □ bone_direction_loss
  □ LaplacianLoss（重写为稠密矩阵版本）
  □ VGGPerceptualLoss / EffiPerceptualLoss
  □ SSIM Loss
  □ 其他标准损失（MSE, L1 等）

□ 渲染器处理
  □ 选择方案（替代渲染器 / 去掉渲染损失 / 自定义算子）
  □ 实现或集成

□ 训练流程迁移
  □ 数据加载 (Dataset → GeneratorDataset)
  □ 优化器 (Adam)
  □ 学习率调度 (MultiStepLR)
  □ 训练循环 (value_and_grad)
  □ 模型保存/加载
  □ TensorBoard 日志（可用 MindInsight 替代）

□ 验证
  □ 单模块输出对齐（逐层对比 PyTorch 和 MindSpore 输出）
  □ 损失值对齐
  □ 训练收敛性验证
  □ 最终精度对比
```

### 7.7 常见坑与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `construct` 中不能用 Python 动态控制流 | MindSpore 图模式限制 | 用 `ops.where` 替代 `if`，或切换到 PYNATIVE 模式 |
| Tensor 不能动态索引赋值 | 图模式不支持 | 用 `ops.scatter_nd_update` 或 `ops.where` |
| `numpy()` 不能在 `construct` 中调用 | 图模式限制 | 将 numpy 操作移到数据预处理中 |
| Conv2d 默认无偏置 | API 差异 | 显式设置 `has_bias=True` |
| `nn.Linear` 不存在 | API 命名差异 | 使用 `nn.Dense` |
| `inplace=True` 不支持 | MindSpore 不支持原地操作 | 去掉 `inplace` 参数 |
| `model.eval()` 行为不同 | BN/Dropout 行为 | 使用 `model.set_train(False)` |
| pickle 加载 MANO 模型 | 不影响，在 `__init__` 中执行 | 保持不变，仅将结果转为 `ms.Tensor` |
| `torch.autograd.Function` | 自定义反向传播 | 使用 `ms.ops.Custom` 或 `bprop` 装饰器 |

### 7.8 推荐迁移顺序

```
第 1 步: 基础设施（1-2 天）
  → 数据加载、配置解析、工具函数

第 2 步: MANO 模型（2-3 天）
  → rot_pose_beta_to_mesh、rodrigues
  → 单独测试：给定相同输入，对比 PyTorch 和 MindSpore 输出

第 3 步: 网络模块（3-5 天）
  → EfficientNet、Hourglass、HandDecoder、TextureEstimator
  → 逐模块对齐测试

第 4 步: 损失函数（2-3 天）
  → 所有损失函数迁移和测试
  → Laplacian Loss 重写

第 5 步: 训练循环（1-2 天）
  → 优化器、学习率调度、训练/评估循环

第 6 步: 渲染器（3-7 天，视方案而定）
  → 如果选择去掉渲染损失，可跳过此步

第 7 步: 集成测试与调优（3-5 天）
  → 端到端训练、精度对齐、性能优化

预计总工时: 15-27 天（1 人）
```
