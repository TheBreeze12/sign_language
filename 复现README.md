# 手语项目总复现 README

本文档用于拿到代码后，按步骤完整复现：

1. 启动前端
2. 启动后端
3. 训练 backend 中的识别模型
4. 使用 S2HAND 权重，通过 3d_sign_language 流程产出 3D GLB 文件

适用系统：Windows（PowerShell）

---

## 0. 代码与目录说明

设项目根目录为 `<PROJECT_ROOT>`（即本 README 所在目录的上层三级目录）

核心目录：

- `frontend/`：前端（Vite + React）
- `backend/`：后端接口与手语识别模型（MindSpore）
- `3d_sign_language/src_3d/`：3D 重建流水线（预处理 -> 3D 参数提取 -> GLB 导出）
- `s2hand_code/`：S2HAND 代码与预训练权重目录（默认权重路径在 `s2hand_code/checkpoints/`）
- `data/`：存储部分数据集
- `result/`：存储bilstm+attention模型的权重
- `日志`：存储部分日志和输出结果信息
---

## 1. 环境准备（建议先做）

### 1.1 Node.js（前端）

建议 Node.js >= 18。

验证：

```powershell
node -v
npm -v
```

### 1.2 Python（后端与3D）

建议 Python 3.9~3.11。可使用 Conda。

示例（可选）：

```powershell
conda create -n signlang python=3.10 -y
conda activate signlang
```

### 1.3 后端 Python 依赖安装

在 `<PROJECT_ROOT>` 目录执行：
```
pip install flask==3.1.3 flask-cors==6.0.2 python-dotenv==1.2.2 pymysql==1.1.0 opencv-python==4.12.0.88 mediapipe==0.10.10 numpy==1.26.4 tqdm==4.67.1 scipy==1.15.3 pytest==7.4.2 mindspore==2.8.0

说明：

- `backend/train.py`、`backend/inference_camera.py` 使用 MindSpore。
- 若评测机器无 Ascend，脚本会自动回退 CPU（速度会慢）。

### 1.4 3D 流程 Python 依赖安装

```powershell
pip install pillow==11.1.0 onnx== 1.16.2 onnxruntime==1.23.2 trimesh
```

---

## 2. 复现启动前端

### 2.1 安装依赖

```powershell
cd <PROJECT_ROOT>/frontend
npm install
```

### 2.2 启动开发服务器

```powershell
npm run dev
```

默认地址：

- `http://localhost:5173`

说明：

- 后端 CORS 已允许 5173 端口。

---

## 3. 复现启动后端

### 3.1 准备数据库（MySQL）

后端依赖 MySQL。请先确保本机 MySQL 服务已启动。

编辑 `backend/.env`（按你机器实际信息填写）：

```env
DB_HOST=localhost
DB_USER=你的数据库用户名
DB_PORT=3306
DB_PASSWORD=你的数据库密码
DB_NAME=sign_language_db

FLASK_PORT=5000
```

说明：

- 启动时 `backend/app.py` 会自动创建数据库与 `sign_assets` 表。
- 启动时也会自动扫描 `result_3d/glb_models/` 并同步词条映射到数据库。

### 3.2 检查关键模型与资源是否存在

至少确认以下文件/目录存在：

- `result/checkpoints/best_model_300.ckpt`
- `data/global_mean_300_double_vel.npy`
- `data/global_std_300_double_vel.npy`
- `data/idx2name_300.txt`

如果你已跑通过完整训练流程，这些文件会自动生成。

### 3.3 启动后端服务

```powershell
cd <PROJECT_ROOT>/backend
python app.py
```

默认地址：

- `http://127.0.0.1:5000`

核心接口：

- `POST /api/sign/predict`（上传视频，返回词条与置信度）
- `GET /api/sign/downloads?name=HELLO`（返回对应 GLB 帧 URL）

---

## 4. 复现训练 backend 里的模型

下面是从原始 WLASL-300 数据重新训练的完整步骤。

### 4.1 准备数据目录

请把数据放到以下结构（与 `backend/config.py` 一致）：

```text
data/
  wlasl-complete/
    videos/
      *.mp4
    nslt_300.json
```

### 4.2 第一步：抽取 134 维特征并生成 map 文件

```powershell
cd <PROJECT_ROOT>/backend
python preprocess.py
```

该步骤会生成：

- `data/processed_features_300/*.npy`
- `data/train_map_300.txt`
- `data/val_map_300.txt`
- `data/test_map_300.txt`

### 4.3 第二步：生成 268 维双相对+速度统计量

```powershell
python core_preprocess.py
```

该步骤会生成：

- `data/global_mean_300_double_vel.npy`
- `data/global_std_300_double_vel.npy`

### 4.4 第三步：训练 MindSpore 模型

```powershell
python train.py
```

默认输出：

- 最优模型：`result/checkpoints/best_model_300.ckpt`
- 最新模型：`result/checkpoints/last_model_300.ckpt`

训练超参数在 `backend/config.py` 中可改：

- `BATCH_SIZE`
- `EPOCHS`
- `LEARNING_RATE`
- `SEQ_LEN`

---

## 5. 使用 S2HAND 权重，通过 3d_sign_language 产出 GLB

这是第4个复现目标的关键部分。完整链路：

1. 手部裁剪预处理（生成 224x224 手部序列）
2. S2HAND 推理提取 3D 参数（joints/vertices/pose/shape）
3. 渲染导出 GLB 序列

### 5.1 准备 S2HAND 权重

默认期望路径（在 `3d_sign_language/src_3d/config_3d.py`）：

- `s2hand_code/checkpoints/checkpoints.onnx`

可通过环境变量覆盖：

- `SIGN_PERTAINED_MODEL`

### 5.3 运行手部预处理

```powershell
python preprocess_3d.py
```

默认输入（可在 `config_3d.py` 或环境变量中改）：

- 视频目录：`data/ASL_Citizen/videos`
- 标注 CSV：`data/ASL_Citizen/splits/train.csv`
- 目标词表：`data/idx2name_300.txt`

默认输出：

- `data/hand_crops_224/<GLOSS>/<VIDEO_ID>/R|L/frame_XXXX.jpg`
- `meta.json`（记录 offset/scale/original_size）

### 5.4 运行 S2HAND 推理入库

```powershell
python inference_to_db.py
```

默认输出：

- `result_3d/database_npz/<GLOSS>/<VIDEO_ID>/data_R.npz`
- `result_3d/database_npz/<GLOSS>/<VIDEO_ID>/data_L.npz`

### 5.5 渲染导出 GLB

```powershell
python render_3d.py
```

默认会按样本逐帧导出 `.glb`。

建议你检查 `render_3d.py` 中 `out_path` 是否是你当前机器有效目录；若不是，请改成项目内路径，例如：

- `<PROJECT_ROOT>/result_3d/glb_models/{word}_{vid_id}/`

导出后用于后端接口的目录为：

- `<PROJECT_ROOT>/result_3d/glb_models/`

后端启动时会自动扫描该目录并同步数据库映射。

---

## 6. 评委快速复现清单（最短路径）

如果你只想快速看到系统跑通（不重新训练）：

1. 启动 MySQL，配置好 `backend/.env`
2. 确认已有：
   - `result/checkpoints/best_model_300.ckpt`
   - `result_3d/glb_models/` 下已有 GLB
3. 终端 A（前端）：

```powershell
cd <PROJECT_ROOT>/frontend
npm install
npm run dev
```

4. 终端 B（后端）：

```powershell
cd <PROJECT_ROOT>/backend
python app.py
```

5. 打开 `http://localhost:5173` 进行上传/识别/3D 播放。

---

## 7. 常见问题（复现高频）

### 7.1 启动后端时报数据库连接失败

请检查：

- MySQL 服务是否已启动
- `backend/.env` 的 `DB_HOST/DB_PORT/DB_USER/DB_PASSWORD`

### 7.2 后端能启动但识别结果异常

请检查：

- `best_model_300.ckpt` 是否是当前特征版本训练出的模型
- `global_mean_300_double_vel.npy` 与 `global_std_300_double_vel.npy` 是否存在且匹配


### 7.4 运行 Python 单行脚本出现模块导入错误

在本仓库里做临时 Python 片段执行时，建议先设置：

```powershell
$env:PYTHONPATH="."
```

---

## 8. 交付说明

本 README 对应的复现目标已经覆盖：

1. 前端启动复现
2. 后端启动复现
3. backend 模型训练复现
4. S2HAND 权重 -> 3D GLB 文件生成复现

评委可按第 6 节最短路径直接演示，也可按第 4/5 节进行完整重跑。
