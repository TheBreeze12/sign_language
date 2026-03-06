import torch
import numpy as np
import os
import json
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import sys
import config_3d

# 1. 路径与环境配置
S2HAND_SDK_PATH = config_3d.S2HAND_SDK_PATH
sys.path.append(S2HAND_SDK_PATH)
sys.path.append(os.path.join(S2HAND_SDK_PATH, "examples")) 
sys.path.append(os.path.join(S2HAND_SDK_PATH, "utils"))

from examples.models_new import Model
from examples.train_utils import load_model

# 获取当前脚本所在目录的上一级目录（即 3d_sign_language 文件夹）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 将 3d_sign_language 加入搜索路径
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from src_3d import config_3d  

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
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def init_engine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = ConfigStub()
    model = Model(filename_obj=None, args=args).to(device)
    model, _ = load_model(model, args)
    model.eval()
    return model, device

@torch.no_grad()
def run_batch_inference(model, device, folder_path, is_flipped):
    img_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    if not img_files: return None

    # 打包 Batch
    imgs = torch.stack([transform(Image.open(os.path.join(folder_path, f)).convert('RGB')) for f in img_files]).to(device)
    K = torch.eye(4).unsqueeze(0).repeat(len(img_files), 1, 1).to(device)
    
    # 核心推理：获取所有 MANO 参数
    out = model.predict_singleview(imgs, None, K, 'test', ['joints', 'verts'], None, None)
    
    # 转为 Numpy 方便存储
    joints = out['joints'].cpu().numpy()     # (N, 21, 3)
    vertices = out['vertices'].cpu().numpy() # (N, 778, 3)
    pose = out['pose'].cpu().numpy()         # (N, 48)
    shape = out['shape'].cpu().numpy()       # (N, 10)
    camera = out['cam_params'].cpu().numpy() if 'cam_params' in out else None

    # =============================
    # 左手镜像翻转逻辑
    # =============================
    if is_flipped:
        # 1. 3D 关节坐标：X 轴取反
        joints[:, :, 0] = -joints[:, :, 0]
        # 2. 3D 网格顶点：X 轴取反 (保证渲染出来是左手)，最后渲染时要修正法向量反向
        vertices[:, :, 0] = -vertices[:, :, 0]

    
    return {"joints": joints, "vertices": vertices, "pose": pose, "shape": shape, "camera": camera}

def build_database():
    # 1. 初始化引擎与设备
    model, device = init_engine()
    
    # 2. 结果保存路径配置
    db_root = config_3d.DB_ROOT
    os.makedirs(db_root, exist_ok=True)

    # 3. 测试计数器与阈值
    processed_count = 0
    MAX_TEST_VIDEOS = 10  # 严格限制只测试10个视频

    # 4. 获取词条列表
    if not os.path.exists(config_3d.HAND_CROP_DIR):
        print(f"❌ 错误：找不到源数据目录 {config_3d.HAND_CROP_DIR}")
        return

    gloss_list = sorted(os.listdir(config_3d.HAND_CROP_DIR))
    
    # 外层循环：遍历词条 (如 8HOUR, 1DOLLAR 等)
    for gloss in tqdm(gloss_list, desc="Building 3D Database"):
        
        # 检查是否已达到测试上限，若是则彻底退出
        if processed_count >= MAX_TEST_VIDEOS:
            break

        gloss_path = os.path.join(config_3d.HAND_CROP_DIR, gloss)
        if not os.path.isdir(gloss_path): 
            continue

        # 中层循环：遍历该词条下的具体视频 ID 文件夹
        vid_folders = sorted(os.listdir(gloss_path))
        for vid_id in vid_folders:
            
            # 再次检查计数器，确保内层循环也能及时中断
            if processed_count >= MAX_TEST_VIDEOS:
                break

            vid_path = os.path.join(gloss_path, vid_id)
            has_data_in_this_vid = False # 标记当前视频是否成功产生数据

            # 内层循环：处理左手(L)和右手(R)
            for side in ['R', 'L']:
                target_dir = os.path.join(vid_path, side)
                
                # 检查 R/L 文件夹及 meta.json 是否存在
                meta_path = os.path.join(target_dir, "meta.json")
                if not os.path.exists(target_dir) or not os.path.exists(meta_path):
                    continue
                
                # 读取 meta 确定是否需要翻转
                try:
                    with open(meta_path, 'r') as f:
                        meta_data = json.load(f)
                        is_flipped = meta_data[0]['is_flipped']
                except Exception as e:
                    print(f"⚠️ 跳过视频 {vid_id} {side}: 读取 meta 失败 - {e}")
                    continue
                
                # 核心推理：获取 3D 坐标
                data = run_batch_inference(model, device, target_dir, is_flipped)
                
                if data is not None:
                    # 创建存储目录：db_root/词条/视频ID/
                    save_dir = os.path.join(db_root, gloss, vid_id)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # 存储为压缩格式 .npz
                    save_path = os.path.join(save_dir, f"data_{side}.npz")
                    np.savez_compressed(
                        save_path,
                        joints=data['joints'],
                        pose=data['pose'],
                        shape=data['shape'],
                        vertices=data['vertices'],
                        is_flipped=is_flipped
                    )
                    has_data_in_this_vid = True

            # 只有当该视频 ID 下至少有一个手(L或R)成功处理后，才增加总计数
            if has_data_in_this_vid:
                processed_count += 1
                print(f"✅ [{processed_count}/{MAX_TEST_VIDEOS}] 已成功提取: {gloss}/{vid_id}")

    print(f"\n🎉 测试任务结束！共提取了 {processed_count} 个视频的 3D 数据到: {db_root}")

if __name__ == "__main__":
    build_database()