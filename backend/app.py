import os
import sys
import cv2
import tempfile
import numpy as np
import pymysql
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv 
import config
from inference_camera import SignLanguageInferencePipeline

env_path=config.ENV_PATH

load_dotenv(env_path)

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173", "http://127.0.0.1:5173"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# 配置
GLB_ROOT = config.GLB_ROOT

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 3307))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

# 供后续请求使用的连接池配置
DB_CONFIG = {
    'host': DB_HOST,
    'port': DB_PORT,
    'user': DB_USER,
    'password': DB_PASSWORD,
    'db': DB_NAME,
    'charset': 'utf8mb4'
}


# 数据库自动化初始化
def init_db():
    """在应用启动前，检查并自动创建数据库和表"""
    try:
        # 此时不指定 db，以便创建尚未存在的数据库
        conn = pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            charset='utf8mb4'
        )
        with conn.cursor() as cursor:
            # 创建数据库
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
            cursor.execute(f"USE `{DB_NAME}`;")
            
            # 创建映射表
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS `sign_assets` (
              `id` INT(11) UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键ID',
              `word_name` VARCHAR(100) NOT NULL COMMENT '手语词条名称',
              `folder_name` VARCHAR(255) NOT NULL COMMENT '服务器上真实的文件夹全名',
              `total_frames` INT(11) DEFAULT 0 COMMENT '动作对应的模型总帧数',
              `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
              PRIMARY KEY (`id`),
              UNIQUE KEY `uk_word_name` (`word_name`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
            cursor.execute(create_table_sql)
        conn.commit()
        conn.close()
        print("✅ 数据库与表结构自动初始化完成！")
    except Exception as e:
        print(f"❌ 数据库初始化失败: {e}")
        sys.exit(1) # 如果数据库连不上，直接终止启动

def sync_glb_to_db():
    """在应用启动时，扫描 GLB_ROOT 目录并将映射关系同步到数据库"""
    if not os.path.exists(GLB_ROOT):
        print(f"⚠️ GLB 根目录不存在: {GLB_ROOT}，请检查配置，跳过同步。")
        return

    print(f"🔍 开始扫描 GLB 模型目录: {GLB_ROOT} ...")
    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            # 扫描目录下所有文件夹
            folders = [f for f in os.listdir(GLB_ROOT) if os.path.isdir(os.path.join(GLB_ROOT, f))]
            sync_count = 0

            for folder in folders:
                try:
                    # 提取词条名称：ACCIDENT_xxx-ACCIDENT)
                    word_name = folder.split('-')[-1].upper()
                    if not word_name:
                        continue

                    # 统计该文件夹下的总帧数 (.glb 文件数量)
                    folder_path = os.path.join(GLB_ROOT, folder)
                    total_frames = len([f for f in os.listdir(folder_path) if f.endswith('.glb')])

                    # 插入或更新数据库 (利用 uk_word_name 唯一索引)
                    # 如果词条已存在，则更新文件夹路径和总帧数
                    sql = """
                        INSERT INTO `sign_assets` (`word_name`, `folder_name`, `total_frames`)
                        VALUES (%s, %s, %s)
                        ON DUPLICATE KEY UPDATE 
                            `folder_name` = VALUES(`folder_name`),
                            `total_frames` = VALUES(`total_frames`);
                    """
                    cursor.execute(sql, (word_name, folder, total_frames))
                    sync_count += 1
                except Exception as e:
                    print(f"⚠️ 解析文件夹 {folder} 时出错，已跳过: {e}")

            conn.commit()
        conn.close()
        print(f"✅ 成功同步 {sync_count} 个手语模型映射到数据库！")
    except Exception as e:
        print(f"❌ 同步模型到数据库失败: {e}")

# 执行数据库初始化
init_db()

# 自动扫描磁盘并同步映射数据到 MySQL
sync_glb_to_db()


# 复用 inference_camera.py 的完整推理流水线
# 包含：模型加载、归一化统计量、标签映射、平滑器
pipeline = SignLanguageInferencePipeline(model_path=config.BEST_MODEL_PATH)


@app.route('/api/sign/predict', methods=['POST'])
def predict():
    """
    POST /api/sign/predict  (multipart/form-data)
    字段: video — 用户录制的手语视频 (mp4/webm/avi)

    后端处理流程:
      1. OpenCV 逐帧解码视频
      2. pipeline.extract_features(frame) → 134 维坐标
      3. pipeline.smoother.smooth(raw_feat) → 平滑
      4. pipeline.predict(frame_buffer)
         内部: preprocess_sequence → 双重相对坐标 + 线性抽样 → 模型推理
    """
    # 接收视频文件
    if 'video' not in request.files:
        return jsonify({"code": 400, "message": "Missing 'video' field"}), 400

    video_file = request.files['video']

    # 保存到临时文件 (OpenCV 需要文件路径)
    suffix = os.path.splitext(video_file.filename)[1] or '.mp4'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    video_file.save(tmp_path)
    tmp.close()

    try:
        # OpenCV 逐帧解码
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return jsonify({"code": 400, "message": "Cannot decode video file"}), 400

        frame_buffer = []
        pipeline.smoother.prev_landmarks = None  # 每次请求重置平滑器

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 提取 134 维特征 + 平滑 (与摄像头推理完全一致)
            raw_feat = pipeline.extract_features(frame)
            smooth_feat = pipeline.smoother.smooth(raw_feat)
            if smooth_feat is not None:
                frame_buffer.append(smooth_feat)

        cap.release()

        # 帧数检查
        if len(frame_buffer) < 5:
            return jsonify({"code": 400, "message": "Video too short, minimum 5 frames required"}), 400

        # 推理 (内部: 双重相对坐标 + 归一化 + 线性插值到64帧 + 模型前向)
        word_name, confidence = pipeline.predict(frame_buffer)

        return jsonify({
            "code": 200,
            "data": {
                "word_name": word_name.lower(),
                "confidence": round(float(confidence), 4)
            }
        })

    finally:
        os.unlink(tmp_path)  # 清理临时文件

@app.route('/api/sign/downloads', methods=['GET'])
def downloads():
    word = request.args.get('name')

    # 在这里转为大写，去匹配数据库
    word_query = word.strip().upper()

    # 从 MySQL 查文件夹名
    conn = pymysql.connect(**DB_CONFIG, cursorclass=pymysql.cursors.DictCursor)
    with conn.cursor() as cursor:
        cursor.execute("SELECT folder_name FROM sign_assets WHERE word_name = %s", (word_query,))
        row = cursor.fetchone()
    conn.close()

    if not row:
        return jsonify({"code": 404, "msg": "Resource not found"}), 404

    folder = row['folder_name']
    abs_path = os.path.join(GLB_ROOT, folder)
    
    # 递归/遍历所有 .glb
    urls = []
    for root, _, files in os.walk(abs_path):
        for f in files:
            if f.endswith('.glb'):
                rel = os.path.relpath(os.path.join(root, f), GLB_ROOT)
                urls.append(f"http://{request.host}/result_3d/glb_models/{rel}")
    
    urls.sort() # 确保帧顺序
    return jsonify({"code": 200, "data": {"urls": urls}})

@app.route('/result_3d/glb_models/<path:filename>')
def serve_glb(filename):
    return send_from_directory(GLB_ROOT, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)