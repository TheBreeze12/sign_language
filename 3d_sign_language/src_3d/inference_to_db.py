import numpy as np
import os
import json
from tqdm import tqdm
from PIL import Image
import logging
import sys
import datetime

# 优先使用 MindSpore + msadapter，支持 ONNX 推理
BACKEND = None
ms = None
torch = None

try:
    import mindspore as ms
    import msadapter.pytorch as torch
    BACKEND = "mindspore-msadapter"
except Exception:
    try:
        import torch
        BACKEND = "torch-fallback"
    except Exception:
        BACKEND = "none"

# Ensure downstream S2HAND modules importing `torch` resolve to msadapter.
if torch is not None:
    sys.modules["torch"] = torch

# 获取当前脚本所在目录的上一级目录（即 3d_sign_language 文件夹）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 将 3d_sign_language 加入搜索路径
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from src_3d import config_3d

# 1. 路径与环境配置
DEFAULT_S2HAND_PATH = os.path.abspath(os.path.join(parent_dir, "..", "s2hand_code", "S2HAND"))


def resolve_existing_path(*candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return os.path.abspath(p)
    return None


S2HAND_PATH = resolve_existing_path(getattr(config_3d, "S2HAND_PATH", None), DEFAULT_S2HAND_PATH)


def _add_s2hand_to_syspath(base_path):
    for p in [base_path, os.path.join(base_path, "examples"), os.path.join(base_path, "utils")]:
        if p not in sys.path:
            sys.path.append(p)

# 配置日志系统
def setup_logging():
    log_dir = config_3d.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    # 以当前时间命名日志文件
    log_filename = datetime.datetime.now().strftime("inference_%Y%m%d_%H%M%S.log")
    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout) # 同时输出到终端
        ]
    )
    return log_path

# 2. 模拟参数配置
class ConfigStub:
    def __init__(self):
        self.train_requires = ['joints', 'verts', 'heatmaps', 'lights']
        self.test_requires = ['joints', 'verts']
        self.regress_mode = 'mano'
        self.use_mean_shape = False
        self.use_2d_as_attention = False
        self.renderer_mode = 'NR'
        self.texture_mode = 'surf'
        self.image_size = 224
        self.train_datasets = ['FreiHand']
        self.use_pose_regressor = False
        self.pretrain_model = config_3d.PERTAINED_MODEL
        self.pretrain_segmnet = self.pretrain_texture_model = self.pretrain_rgb2hm = None

# 3. 图像预处理
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def preprocess_image(img_pil, side):
    img = img_pil.convert('RGB').resize((224, 224), Image.BILINEAR)
    if side == 'L':
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.tensor(arr, dtype=torch.float32)


def _empty_cache_if_available():
    if torch is not None and hasattr(torch, 'cuda') and hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()


if torch is not None and hasattr(torch, "no_grad"):
    no_grad = torch.no_grad
else:
    def no_grad():
        def _decorator(func):
            return func
        return _decorator

def init_engine():
    # Keep MindSpore in eager mode for better compatibility with msadapter path.
    if BACKEND == "mindspore-msadapter" and ms is not None:
        ms.set_context(mode=ms.PYNATIVE_MODE)

    if S2HAND_PATH is None:
        raise FileNotFoundError(
            "Cannot find S2HAND source path. Please set config_3d.S2HAND_PATH "
            "or place code at ../s2hand_code/S2HAND."
        )

    _add_s2hand_to_syspath(S2HAND_PATH)

    # 检查是否使用 ONNX 模型
    if config_3d.PERTAINED_MODEL.endswith('.onnx'):
        logging.info(f"Loading ONNX model from: {config_3d.PERTAINED_MODEL}")
        from examples.train_utils_ms import load_onnx_model
        model = load_onnx_model(config_3d.PERTAINED_MODEL)
        device = None  # ONNX 不需要显式设备管理
        return model, device

    # 否则加载 PyTorch/MindSpore 模型
    from examples.models_new_ms import Model
    from examples.train_utils_ms import load_model

    # 根据后端选择最优设备
    if BACKEND == "mindspore-msadapter" and ms is not None:
        # 优先尝试 Ascend NPU
        try:
            ms.set_context(device_target="Ascend")
            device = torch.device("npu:0")  # msadapter 中 Ascend 对应 npu
            logging.info("✓ 使用 Ascend NPU 设备")
        except Exception:
            # Ascend 不可用，尝试 GPU
            try:
                ms.set_context(device_target="GPU")
                device = torch.device("cuda:0")
                logging.info("✓ 使用 GPU 设备（MindSpore 后端）")
            except Exception:
                # 降级到 CPU
                ms.set_context(device_target="CPU")
                device = torch.device("cpu")
                logging.info("✓ 使用 CPU 设备（MindSpore 后端）")
    else:
        # PyTorch 原生后端
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"✓ 使用 {device} 设备（PyTorch 后端）")

    args = ConfigStub()
    model = Model(filename_obj=None, args=args).to(device)
    model, _ = load_model(model, args)
    model.eval()
    return model, device

@no_grad()
def run_batch_inference(model, device, folder_path,side,batch_size=32):
    img_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    if not img_files: return None

    # 用于暂存每个 mini-batch 的结果
    all_joints, all_vertices, all_pose, all_shape = [], [], [], []

    for i in range(0, len(img_files), batch_size):
        batch_files = img_files[i : i + batch_size]

        # 打包当前块
        batch_imgs = []
        for f in batch_files:
            img_pil = Image.open(os.path.join(folder_path, f))
            batch_imgs.append(preprocess_image(img_pil, side))

        imgs = torch.stack(batch_imgs)

        # 如果使用 ONNX 模型，device 为 None
        if device is not None:
            imgs = imgs.to(device)

        K = torch.eye(4).unsqueeze(0).repeat(len(batch_files), 1, 1)
        if device is not None:
            K = K.to(device)

        # 核心推理
        out = model.predict_singleview(imgs, None, K, 'test', ['joints', 'verts'], None, None)

        # 将结果拉回 CPU 并追加到列表中
        if isinstance(out['joints'], np.ndarray):
            # ONNX 输出已经是 numpy
            all_joints.append(out['joints'])
            all_vertices.append(out['vertices'])
            all_pose.append(out['pose'])
            all_shape.append(out['shape'])
        else:
            # PyTorch/MindSpore tensor
            all_joints.append(out['joints'].cpu().numpy())
            all_vertices.append(out['vertices'].cpu().numpy())
            all_pose.append(out['pose'].cpu().numpy())
            all_shape.append(out['shape'].cpu().numpy())

        # 手动释放当前 batch 的 GPU 显存引用
        del imgs, K, out
        _empty_cache_if_available()

    joints = np.concatenate(all_joints, axis=0)     # (N, 21, 3)
    vertices = np.concatenate(all_vertices, axis=0) # (N, 778, 3)
    pose = np.concatenate(all_pose, axis=0)         # (N, 48)
    shape = np.concatenate(all_shape, axis=0)       # (N, 10)

    # 如果是左手，将推理出的伪右手 3D 坐标在 X 轴上翻转回左手
    if side == 'L':
        joints[:, :, 0] = -joints[:, :, 0]
        vertices[:, :, 0] = -vertices[:, :, 0]
        # 后续在 render_3d.py 中渲染左手时，必须使用 MANO_LEFT.pkl 的面片，或者对 MANO_RIGHT 的面片做 [0, 2, 1] 绕序反转。

    return {"joints": joints, "vertices": vertices, "pose": pose, "shape": shape, "camera": None}

def build_database():
    log_path = setup_logging()
    logging.info(f"🚀 开始全量 3D 数据库构建任务。日志文件: {log_path}")

    logging.info(f"Backend: {BACKEND}")

    if BACKEND == "none" or torch is None:
        logging.warning(
            "❌ 未检测到可用后端：请安装 mindspore+msadapter 或 torch。当前任务已安全退出。"
        )
        return

    if BACKEND == "torch-fallback":
        logging.warning("MindSpore/msadapter not found, using torch fallback backend.")

    # 先检查输入与权重路径，避免无意义初始化模型
    if not os.path.exists(config_3d.HAND_CROP_DIR):
        logging.warning(f"❌ 错误：找不到源数据目录 {config_3d.HAND_CROP_DIR}")
        return

    pretrained_model_path = resolve_existing_path(
        getattr(config_3d, "PERTAINED_MODEL", None),
        os.path.abspath(os.path.join(parent_dir, "..", "s2hand_code", "checkpoints", "checkpoints.pth"))
    )
    if pretrained_model_path is None:
        logging.warning(
            "❌ 错误：找不到预训练模型 checkpoints.pth。请检查 config_3d.PERTAINED_MODEL 或默认路径。"
        )
        return
    config_3d.PERTAINED_MODEL = pretrained_model_path

    if S2HAND_PATH is None:
        logging.warning(
            "❌ 错误：找不到 S2HAND 代码目录。请检查 config_3d.S2HAND_PATH 或默认路径 ../s2hand_code/S2HAND。"
        )
        return

    # 1. 初始化引擎与设备
    model, device = init_engine()

    # 2. 结果保存路径配置
    db_root = config_3d.DB_ROOT
    os.makedirs(db_root, exist_ok=True)

    # 3. 初始化计数器
    processed_count = 0
    error_count=0

    gloss_list = sorted(os.listdir(config_3d.HAND_CROP_DIR))

    # 外层循环：遍历词条 (如 8HOUR, 1DOLLAR 等)
    for gloss in tqdm(gloss_list, desc="Building 3D Database"):

        gloss_path = os.path.join(config_3d.HAND_CROP_DIR, gloss)
        if not os.path.isdir(gloss_path):
            continue

        # 中层循环：遍历该词条下的具体视频 ID 文件夹
        vid_folders = sorted(os.listdir(gloss_path))
        for vid_id in vid_folders:

            vid_path = os.path.join(gloss_path, vid_id)
            # 标记当前视频是否成功产生数据
            # 内层循环：处理左手(L)和右手(R)
            has_data_in_this_vid = False
            for side in ['R', 'L']:
                target_dir = os.path.join(vid_path, side)

                # 检查 R/L 文件夹及 meta.json 是否存在
                meta_path = os.path.join(target_dir, "meta.json")
                if not os.path.exists(target_dir) or not os.path.exists(meta_path):
                    continue

                try:
                    # 核心推理：获取 3D 坐标
                    data = run_batch_inference(model, device, target_dir, side)

                    if data is not None:
                        # 创建存储目录：db_root/词条/视频ID/
                        save_dir = os.path.join(db_root, gloss, vid_id)
                        os.makedirs(save_dir, exist_ok=True)

                        # 存储为压缩格式 .npz
                        save_path = os.path.join(save_dir, f"data_{side}.npz")

                        is_flipped_flag = True if side == 'L' else False

                        np.savez_compressed(
                            save_path,
                            joints=data['joints'],
                            pose=data['pose'],
                            shape=data['shape'],
                            vertices=data['vertices'],
                            is_flipped=is_flipped_flag
                        )
                        has_data_in_this_vid=True
                    else:
                        error_count+=1
                except Exception as e:
                    logging.warning(f"跳过视频 {gloss}/{vid_id}/{side}: {str(e)}")
                    error_count += 1

            # 成功处理一个视频 ID 后自增计数
            if has_data_in_this_vid:
                processed_count += 1

            # 只有当该视频 ID 下至少有一个手(L或R)成功处理后，才增加总计数
    logging.info(f"🎉 提取任务完成！成功: {processed_count}, 失败: {error_count}")

if __name__ == "__main__":
    build_database()
