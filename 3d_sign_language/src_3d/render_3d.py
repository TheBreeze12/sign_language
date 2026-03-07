import os
import sys
import torch
import numpy as np
import cv2
import trimesh
import pickle
from tqdm import tqdm
import config_3d

# 1. 强制无窗口渲染 (WSL2 必须在导入 pyrender 前设置)
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender

# 2. 注入路径以确保能找到 S2HAND 相关模块
S2HAND_PATH = config_3d.S2HAND_PATH
if S2HAND_PATH not in sys.path:
    sys.path.append(S2HAND_PATH)
    sys.path.append(os.path.join(S2HAND_PATH, "examples"))

# 3. 加载 MANO 面片 (Faces) 方案
def load_mano_faces():
    # MANO 权重路径
    right_pkl_path = config_3d.RIGHT_PKL_PATH
    if not os.path.exists(right_pkl_path):
        raise FileNotFoundError(f"❌ 找不到 MANO 权重文件: {right_pkl_path}")
    
    with open(right_pkl_path, 'rb') as f:
        mano_data = pickle.load(f, encoding='latin1')
    return mano_data['f']#加载face拓扑

def render_npz_to_video(npz_path, output_video):
    # 加载数据
    if not os.path.exists(npz_path):
        print(f"❌ 找不到数据文件: {npz_path}")
        return

    data = np.load(npz_path)
    verts_seq = data['vertices']  # (N, 778, 3)
    #检查左手
    is_left=data.get('is_flipped',False)
    
    #  获取面片拓扑
    try:
        faces = load_mano_faces()
        if is_left:
            faces=faces[:,[0,2,1]]
    except Exception as e:
        print(f"❌ 加载面片失败: {e}")
        return

    # 初始化Pyrender 场景 
    # 背景设为白色，环境光设为中等
    scene = pyrender.Scene(ambient_light=[0.6, 0.6, 0.6], bg_color=[1.0, 1.0, 1.0])

    # 设置相机：yfov决定视角宽度
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    
    #首先计算第一帧vertices中心位置，相机指向第一帧中心，防止手部在画面中位置过小
    mean_vert=np.mean(verts_seq[0], axis=0)
    cam_pose = np.eye(4)
    #z轴高度为0.22
    cam_pose[:3, 3] = mean_vert + np.array([0, 0, 0.22])
    scene.add(camera, pose=cam_pose)
    
    # 添加主光源
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=cam_pose)
    
    # 初始化离线渲染器
    renderer = pyrender.OffscreenRenderer(1024, 1024)
    
    # 初始化视频写入器
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    # 使用 avc1 编码，这类视频在 Web 和大多数播放器兼容性更好
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(output_video, fourcc, 5, (1024, 1024))

    print(f"🚀 开始渲染视频: {npz_path}")
    for i in tqdm(range(len(verts_seq))):
        # 1. 创建网格对象
        mesh = trimesh.Trimesh(vertices=verts_seq[i], faces=faces,process=True)

        # 制计算面法线和顶点法线，这能显著解决“糊”的问题，让肌肉线条清晰
        mesh.fix_normals()
        
        # 2. 坐标系对齐：绕 Y 轴旋转 180 度，使手背朝向镜头（符合图像视觉）
        rot_matrix = trimesh.transformations.rotation_matrix(np.radians(180), [0, 1, 0])
        mesh.apply_transform(rot_matrix)
        
        # 3. 材质调节，将网格加入场景
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            roughnessFactor=0.6,
            baseColorFactor=[0.5, 0.5, 0.5, 1.0]
        )
        
        render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        mesh_node = scene.add(render_mesh)

        # 4. 渲染并写入视频帧
        color, _ = renderer.render(scene)
        video_writer.write(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        
        # 5. 清理节点准备下一帧（及其重要）
        scene.remove_node(mesh_node)

    # 资源清理 
    renderer.delete()
    video_writer.release()
    print(f"✅ 渲染成功！视频保存至: {output_video}")

if __name__ == "__main__":
    # 使用你测试视频中的一个路径
    target_npz = "/home/jm802/sign_language/result_3d/database_npz/1DOLLAR/0719792557216079-1 DOLLAR/data_R.npz"
    out_video = "/home/jm802/sign_language/result_3d/video/test_render_result.mp4"
    
    render_npz_to_video(target_npz, out_video)