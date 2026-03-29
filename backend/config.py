import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据根目录
DATA_ROOT = os.path.join(BASE_DIR, "data")
# 原始视频目录
VIDEO_DIR = os.path.join(DATA_ROOT, "wlasl-complete", "videos")

SPLIT_JSON_PATH = os.path.join(DATA_ROOT, "wlasl-complete", "nslt_300.json")

# 输出目录：存放提取好的 .npy 文件
SAVE_NPY_DIR = os.path.join(DATA_ROOT, "processed_features_300")

# 结果目录
RESULT_DIR = os.path.join(BASE_DIR, "result")
MODEL_SAVE_PATH = os.path.join(RESULT_DIR, "checkpoints")
MODEL_PATH = os.path.join(RESULT_DIR, "checkpoints", "best_model_300.ckpt")

MEAN_PATH = os.path.join(DATA_ROOT, "global_mean_300_double_vel.npy")
STD_PATH = os.path.join(DATA_ROOT, "global_std_300_double_vel.npy")

IDX2NAME_PATH = os.path.join(DATA_ROOT, "idx2name_300.txt")

GLB_ROOT = os.path.join(BASE_DIR, "result_3d", "glb_models")

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

# 数据划分文件
TRAIN_MAP_PATH = os.path.join(DATA_ROOT, "train_map_300.txt")
VAL_MAP_PATH = os.path.join(DATA_ROOT, "val_map_300.txt")
TEST_MAP_PATH = os.path.join(DATA_ROOT, "test_map_300.txt")
GLOBAL_MEAN_PATH = os.path.join(DATA_ROOT, "global_mean_300_double_vel.npy")
GLOBAL_STD_PATH = os.path.join(DATA_ROOT, "global_std_300_double_vel.npy")


BEST_MODEL_PATH = os.path.join(RESULT_DIR, "checkpoints", "best_model_300.ckpt")
LAST_MODEL_PATH = os.path.join(RESULT_DIR, "checkpoints", "last_model_300.ckpt")

ASL_JSON_ROOT = os.getenv("ASL_JSON_ROOT", os.path.join(DATA_ROOT, "ASL_300_JSON"))
ASL_300_VIDEO = os.getenv("ASL_300_VIDEO", os.path.join(DATA_ROOT, "ASL_300_VIDEO"))
ASL_JSON_ROUTE = os.getenv("ASL_JSON_ROUTE", "/asl_json")
ASL_VIDEO_ROUTE = os.getenv("ASL_VIDEO_ROUTE", "/asl_video")
# 数据参数
# MediaPipe特征维度计算:
# Pose(只取上半身0-24点=25个) * 2(x,y) = 50
# Left Hand(21个) * 2(x,y) = 42
# Right Hand(21个) * 2(x,y) = 42
# 加速度 Δx, Δy
#268维
INPUT_SIZE = 268
HIDDEN_SIZE= 256

SEQ_LEN = 64         # 序列统一长度
NUM_CLASSES = 300    # 类别数
NUM_LAYERS=2
DROP_OUT=0.3

# 训练参数
BATCH_SIZE = 64
EPOCHS = 80
LEARNING_RATE = 1e-3

#推理参数
ALPHA=0.6