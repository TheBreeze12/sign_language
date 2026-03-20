"""
config_3d.py — 三维手部重建流水线配置
阶段：Stage 1 手部裁剪预处理 (Hand Cropping for S2HAND)
"""
import os


def _path_from_env(name, default):
	"""Return normalized absolute path from env var or default."""
	raw = os.getenv(name)
	value = raw if raw else default
	value = os.path.expandvars(os.path.expanduser(value))
	if not os.path.isabs(value):
		value = os.path.join(PROJECT_ROOT, value)
	return os.path.normpath(value)


def _bool_from_env(name, default):
	raw = os.getenv(name)
	if raw is None:
		return default
	return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int_from_env(name, default):
	raw = os.getenv(name)
	if raw is None:
		return default
	return int(raw)


def _float_from_env(name, default):
	raw = os.getenv(name)
	if raw is None:
		return default
	return float(raw)


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Workspace root (e.g. .../sign_language)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

# ========================= 数据路径 =========================
DATA_ROOT = _path_from_env("SIGN_DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))

# 原始视频目录（按 video_id 命名的 .mp4 文件）
VIDEO_DIR = _path_from_env("SIGN_VIDEO_DIR", os.path.join(DATA_ROOT, "ASL_Citizen", "videos"))

# 更新后的 CSV 路径（先测试 train.csv）
CSV_PATH = _path_from_env("SIGN_CSV_PATH", os.path.join(DATA_ROOT, "ASL_Citizen", "splits", "train.csv"))

# ========================= 输出路径 =========================
# 裁剪后的手部图片根目录
# 结构：HAND_CROP_DIR/<gloss_id>/<split>/<video_id>/frame_0000.jpg
HAND_CROP_DIR = _path_from_env("SIGN_HAND_CROP_DIR", os.path.join(DATA_ROOT, "hand_crops_224"))

# BBox 缓存（每个视频每帧的检测框，.npy，方便调试/复用）
BBOX_CACHE_DIR = _path_from_env("SIGN_BBOX_CACHE_DIR", os.path.join(DATA_ROOT, "bbox_cache"))

# 处理日志目录（同脚本文件夹下）
LOG_DIR = _path_from_env("SIGN_LOG_DIR", os.path.join(CURRENT_DIR, "logs"))

# ========================= 图像参数 =========================
IMAGE_SIZE = _int_from_env("SIGN_IMAGE_SIZE", 224)           # S2HAND 标准输入 224×224
BBOX_EXPAND_RATIO = _float_from_env("SIGN_BBOX_EXPAND_RATIO", 1.45)  # BBox 扩展系数（1.2~1.5 均可）


# ========================= MediaPipe 参数 =========================
MP_DETECTION_CONFIDENCE = _float_from_env("SIGN_MP_DETECTION_CONFIDENCE", 0.5)
MP_TRACKING_CONFIDENCE = _float_from_env("SIGN_MP_TRACKING_CONFIDENCE", 0.5)
MP_MAX_NUM_HANDS = _int_from_env("SIGN_MP_MAX_NUM_HANDS", 2)   # 手语通常需要双手

# ========================= 并行 / 保存参数 =========================
NUM_WORKERS = _int_from_env("SIGN_NUM_WORKERS", 4)    # 多进程并行（0 = 自动=CPU核心数）
JPEG_QUALITY = _int_from_env("SIGN_JPEG_QUALITY", 95)   # JPEG 保存质量

# ========================= 跳过控制 =========================
# True  → 已处理过的视频直接跳过（断点续跑）
# False → 强制重新处理所有视频
SKIP_EXISTING = _bool_from_env("SIGN_SKIP_EXISTING", True)

#高斯平滑
SIGMA = _float_from_env("SIGN_SIGMA", 1.7)

#debug目录
DEBUG_OUT_DIR = _path_from_env("SIGN_DEBUG_OUT_DIR", os.path.join(PROJECT_ROOT, "debug_results"))

#SHaMeR路径配置
S2HAND_PATH = _path_from_env("SIGN_S2HAND_PATH", os.path.join(PROJECT_ROOT, "s2hand_code", "S2HAND"))

PERTAINED_MODEL = _path_from_env("SIGN_PERTAINED_MODEL", os.path.join(PROJECT_ROOT, "s2hand_code", "checkpoints", "checkpoints.pth"))

DB_ROOT = _path_from_env("SIGN_DB_ROOT", os.path.join(PROJECT_ROOT, "result_3d", "database_npz"))

RIGHT_PKL_PATH = _path_from_env("SIGN_RIGHT_PKL_PATH", os.path.join(PROJECT_ROOT, "3d_sign_language", "mano_v1_2", "models", "MANO_RIGHT.pkl"))

TARGET_WORDS_FILE = _path_from_env("SIGN_TARGET_WORDS_FILE", os.path.join(DATA_ROOT, "idx2name_300.txt"))

BASE_DB = DB_ROOT + os.sep


FACTOR = _int_from_env("SIGN_FACTOR", 2)
SCALE_FACTOR = _float_from_env("SIGN_SCALE_FACTOR", 0.003)
FOCAL_CONSTANT = _float_from_env("SIGN_FOCAL_CONSTANT", 20.0)
