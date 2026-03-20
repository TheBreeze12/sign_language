import numpy as np
import os
import json
from tqdm import tqdm
from PIL import Image
import logging
import sys
import datetime

# 使用 ONNX 推理
BACKEND = "onnx"
ms = None

# 获取当前脚本所在目录的上一级目录（即 3d_sign_language 文件夹）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 将 3d_sign_language 加入搜索路径
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from src_3d import config_3d

# 1. 路径与环境配置
DEFAULT_ONNX_MODEL = os.path.join(os.path.dirname(__file__), "output.onnx")

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
            logging.FileHandler(log_path, encoding='utf-8'),  # UTF-8 编码支持 emoji
            logging.StreamHandler(sys.stdout)  # 同时输出到终端
        ]
    )
    return log_path

# 图像预处理常量
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def resolve_existing_path(*paths):
    """返回第一个存在的路径，否则返回最后一个路径"""
    for path in paths:
        if path and os.path.exists(path):
            return path
    return paths[-1] if paths else None


def preprocess_image_numpy(img_pil, side):
    """返回 numpy 数组（用于 ONNX 推理）"""
    img = img_pil.convert('RGB').resize((224, 224), Image.BILINEAR)
    if side == 'L':
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr


def _empty_cache_if_available():
    """无需缓存清理（纯 ONNX 推理）"""
    pass

def init_engine_onnx():
    """初始化 ONNX 推理引擎"""
    import onnxruntime as ort

    model_path = config_3d.PERTAINED_MODEL
    logging.info(f"Loading ONNX model from: {model_path}")

    try:
        # 根据本机可用 provider 自动选择，避免无效 provider 警告
        available_providers = ort.get_available_providers()
        providers = []
        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
        if 'CPUExecutionProvider' in available_providers:
            providers.append('CPUExecutionProvider')
        if not providers:
            providers = available_providers

        session = ort.InferenceSession(model_path, providers=providers)
        logging.info(f"ONNX Runtime providers: {providers}")
        logging.info(f"✓ ONNX 模型加载成功 (路径: {model_path})")

        # 获取输入输出信息
        input_names = [input_info.name for input_info in session.get_inputs()]
        output_names = [output_info.name for output_info in session.get_outputs()]

        return {
            'session': session,
            'input_names': input_names,
            'output_names': output_names,
            'model_type': 'onnx'
        }
    except Exception as e:
        logging.error(f"❌ ONNX 模型加载失败: {e}")
        raise


def init_engine():
    """初始化 ONNX 推理引擎"""
    return init_engine_onnx()

def run_batch_inference_onnx(engine_info, folder_path, side, batch_size=32):
    """ONNX 模型推理（使用 batch_size=1 逐帧处理以避免形状不匹配）"""
    session = engine_info['session']
    input_name = engine_info['input_names'][0]
    output_names = engine_info['output_names']

    img_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    if not img_files:
        return None

    all_joints, all_vertices = [], []
    all_pose, all_shape = [], []
    all_scale, all_trans, all_rot, all_tsa_poses = [], [], [], []

    # 逐帧处理（batch_size=1）以避免 ONNX 模型中间层的形状不匹配问题
    for img_file in img_files:
        try:
            img_pil = Image.open(os.path.join(folder_path, img_file))
            img_arr = preprocess_image_numpy(img_pil, side)

            # 添加批维度: (3, 224, 224) -> (1, 3, 224, 224)
            imgs = np.expand_dims(img_arr, axis=0).astype(np.float32)

            # ONNX 推理（输出 8 个张量）
            outputs = session.run(output_names, {input_name: imgs})

            # 输出顺序（对应 export_onnx.py 中的 output_names）：
            # [joints, vertices, pose, shape, scale, trans, rot, tsa_poses]
            joints_batch = outputs[0]      # (1, 21, 3)
            vertices_batch = outputs[1]    # (1, 778, 3)
            pose_batch = outputs[2] if len(outputs) > 2 else np.zeros((1, 48), dtype=np.float32)       # (1, 48)
            shape_batch = outputs[3] if len(outputs) > 3 else np.zeros((1, 10), dtype=np.float32)      # (1, 10)
            scale_batch = outputs[4] if len(outputs) > 4 else np.ones((1, 1), dtype=np.float32)        # (1, 1)
            trans_batch = outputs[5] if len(outputs) > 5 else np.zeros((1, 2), dtype=np.float32)       # (1, 2)
            rot_batch = outputs[6] if len(outputs) > 6 else np.zeros((1, 3), dtype=np.float32)         # (1, 3)
            tsa_poses_batch = outputs[7] if len(outputs) > 7 else np.zeros((1, 16), dtype=np.float32)  # (1, 16)

            all_joints.append(joints_batch[0])
            all_vertices.append(vertices_batch[0])
            all_pose.append(pose_batch[0])
            all_shape.append(shape_batch[0])
            all_scale.append(scale_batch[0])
            all_trans.append(trans_batch[0])
            all_rot.append(rot_batch[0])
            all_tsa_poses.append(tsa_poses_batch[0])

            del imgs, outputs
        except Exception as e:
            logging.warning(f"ONNX 推理失败 ({img_file}): {e}")
            continue

    if not all_joints:
        return None

    joints = np.stack(all_joints, axis=0)         # (N, 21, 3)
    vertices = np.stack(all_vertices, axis=0)     # (N, 778, 3)
    pose = np.stack(all_pose, axis=0)             # (N, 48)
    shape = np.stack(all_shape, axis=0)           # (N, 10)
    scale = np.stack(all_scale, axis=0)           # (N, 1)
    trans = np.stack(all_trans, axis=0)           # (N, 2)
    rot = np.stack(all_rot, axis=0)               # (N, 3)
    tsa_poses = np.stack(all_tsa_poses, axis=0)   # (N, 16)

    # 若是左手，在 X 轴翻转
    if side == 'L':
        joints[:, :, 0] = -joints[:, :, 0]
        vertices[:, :, 0] = -vertices[:, :, 0]

    # 返回格式：与数据库存储格式兼容（包含完整 MANO 参数）
    return {
        "joints": joints,
        "vertices": vertices,
        "pose": pose,
        "shape": shape,
        "scale": scale,
        "trans": trans,
        "rot": rot,
        "tsa_poses": tsa_poses,
    }


def run_batch_inference(engine_info, folder_path, side, batch_size=32):
    """执行 ONNX 推理"""
    return run_batch_inference_onnx(engine_info, folder_path, side, batch_size)

def build_database():
    log_path = setup_logging()
    logging.info(f"🚀 开始全量 3D 数据库构建任务。日志文件: {log_path}")

    logging.info(f"Backend: {BACKEND} (ONNX 推理)")

    # 先检查输入与权重路径，避免无意义初始化模型
    if not os.path.exists(config_3d.HAND_CROP_DIR):
        logging.warning(f"❌ 错误：找不到源数据目录 {config_3d.HAND_CROP_DIR}")
        return

    pretrained_model_path = resolve_existing_path(
        getattr(config_3d, "PERTAINED_MODEL", None),
        os.path.abspath(os.path.join(parent_dir, "..", "s2hand_code", "checkpoints", "checkpoints.pth"))
    )

    # 对于 ONNX，寻找 .onnx 文件；如果没有，则转换 .pth 到 .onnx
    if pretrained_model_path and not pretrained_model_path.endswith('.onnx'):
        onnx_path = pretrained_model_path.replace('.pth', '.onnx')
        if os.path.exists(onnx_path):
            pretrained_model_path = onnx_path
        else:
            logging.warning(f"⚠️ 找不到 ONNX 模型 {onnx_path}，请先将 .pth 导出成 .onnx")
            logging.warning(f"   可以使用: python export_onnx.py --input {pretrained_model_path} --output {onnx_path}")
            return

    if pretrained_model_path is None:
        logging.warning(
            "❌ 错误：找不到预训练模型（.onnx 格式）。请检查 config_3d.PERTAINED_MODEL 或默认路径。"
        )
        return
    config_3d.PERTAINED_MODEL = pretrained_model_path

    # 1. 初始化引擎
    engine_info = init_engine()

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
                    data = run_batch_inference(engine_info, target_dir, side)

                    if data is not None:
                        # 创建存储目录：db_root/词条/视频ID/
                        save_dir = os.path.join(db_root, gloss, vid_id)
                        os.makedirs(save_dir, exist_ok=True)

                        # 存储为压缩格式 .npz（包含完整 MANO 参数，防止权重丢失）
                        save_path = os.path.join(save_dir, f"data_{side}.npz")

                        is_flipped_flag = True if side == 'L' else False

                        np.savez_compressed(
                            save_path,
                            joints=data['joints'],
                            vertices=data['vertices'],
                            pose=data['pose'],
                            shape=data['shape'],
                            scale=data['scale'],
                            trans=data['trans'],
                            rot=data['rot'],
                            tsa_poses=data['tsa_poses'],
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
