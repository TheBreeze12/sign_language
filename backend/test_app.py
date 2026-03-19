import os
import sys
import io
import pytest
import pymysql

from backend.app import app, pipeline, DB_CONFIG, GLB_ROOT, init_db, sync_glb_to_db
import config


@pytest.fixture(scope="session")
def client():
    """Flask 测试客户端 (整个测试会话复用)"""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture(scope="session")
def db_conn():
    """真实数据库连接 (用于验证数据)"""
    conn = pymysql.connect(**DB_CONFIG, cursorclass=pymysql.cursors.DictCursor)
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def test_video_bytes():
    """读取项目中真实的测试视频文件"""
    video_path = os.path.join(os.path.dirname(__file__), '..', 'frontend','src', 'assets', 'video', 'banana.mp4')
    video_path = os.path.abspath(video_path)
    assert os.path.exists(video_path), f"测试视频不存在: {video_path}"
    with open(video_path, "rb") as f:
        return f.read()


class TestStartupFunctions:
    """测试 init_db 和 sync_glb_to_db 的真实执行"""

    def test_init_db_creates_table(self, db_conn):
        """init_db 应成功创建 sign_assets 表"""
        with db_conn.cursor() as cursor:
            cursor.execute("SHOW TABLES LIKE 'sign_assets'")
            result = cursor.fetchone()
        assert result is not None, "sign_assets 表不存在，init_db 未正确执行"

    def test_sign_assets_table_structure(self, db_conn):
        """验证 sign_assets 表结构包含必要字段"""
        with db_conn.cursor() as cursor:
            cursor.execute("DESCRIBE sign_assets")
            columns = {row["Field"] for row in cursor.fetchall()}

        assert "id" in columns
        assert "word_name" in columns
        assert "folder_name" in columns
        assert "total_frames" in columns

    def test_sync_glb_to_db_has_data(self, db_conn):
        """sync_glb_to_db 应已将 GLB 目录中的文件夹同步到数据库"""
        if not os.path.exists(GLB_ROOT):
            pytest.skip(f"GLB_ROOT 目录不存在: {GLB_ROOT}")

        with db_conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS cnt FROM sign_assets")
            result = cursor.fetchone()

        assert result["cnt"] > 0, "sign_assets 表中没有数据，sync_glb_to_db 可能未执行"

    def test_sync_data_matches_disk(self, db_conn):
        """验证数据库中至少一条记录的 folder_name 在磁盘上真实存在"""
        if not os.path.exists(GLB_ROOT):
            pytest.skip(f"GLB_ROOT 目录不存在: {GLB_ROOT}")

        with db_conn.cursor() as cursor:
            cursor.execute("SELECT word_name, folder_name, total_frames FROM sign_assets LIMIT 5")
            rows = cursor.fetchall()

        assert len(rows) > 0, "数据库中没有数据"

        for row in rows:
            folder_path = os.path.join(GLB_ROOT, row["folder_name"])
            assert os.path.isdir(folder_path), f"数据库记录的文件夹不存在: {folder_path}"
            # 验证帧数与磁盘上 .glb 文件数一致
            glb_count = len([f for f in os.listdir(folder_path) if f.endswith('.glb')])
            assert row["total_frames"] == glb_count, \
                f"{row['word_name']}: 数据库帧数 {row['total_frames']} != 磁盘帧数 {glb_count}"


class TestPredictEndpoint:
    """测试模型推理接口 (真实模型 + 真实视频)"""

    def test_predict_success(self, client, test_video_bytes):
        """上传真实视频 → 应返回 code=200 + word_name + confidence"""
        resp = client.post(
            "/api/sign/predict",
            content_type="multipart/form-data",
            data={"video": (io.BytesIO(test_video_bytes), "banana.mp4")},
        )

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["code"] == 200
        assert "data" in body
        assert "word_name" in body["data"], f"响应缺少 word_name: {body}"
        assert "confidence" in body["data"], f"响应缺少 confidence: {body}"
        assert isinstance(body["data"]["confidence"], float)
        assert 0.0 <= body["data"]["confidence"] <= 1.0
        # word_name 应为全小写
        assert body["data"]["word_name"] == body["data"]["word_name"].lower(), \
            f"word_name 不是小写: {body['data']['word_name']}"
        print(f"\n  ✅ 推理结果: {body['data']['word_name']} (置信度: {body['data']['confidence']:.4f})")

    def test_predict_missing_video_field(self, client):
        """不传 video 字段 → 应返回 400"""
        resp = client.post(
            "/api/sign/predict",
            content_type="multipart/form-data",
            data={},
        )

        assert resp.status_code == 400
        body = resp.get_json()
        assert body["code"] == 400
        assert "message" in body
        print(f"\n  ✅ 缺少 video 字段: {body['message']}")

    def test_predict_empty_file(self, client):
        """上传空文件 → 应返回 400"""
        resp = client.post(
            "/api/sign/predict",
            content_type="multipart/form-data",
            data={"video": (io.BytesIO(b""), "empty.mp4")},
        )

        assert resp.status_code == 400
        body = resp.get_json()
        assert body["code"] == 400
        print(f"\n  ✅ 空文件: {body['message']}")

    def test_predict_response_format(self, client, test_video_bytes):
        """验证响应格式严格符合 interface.md 定义"""
        resp = client.post(
            "/api/sign/predict",
            content_type="multipart/form-data",
            data={"video": (io.BytesIO(test_video_bytes), "banana.mp4")},
        )

        body = resp.get_json()
        # 顶层必须有 code 和 data
        assert "code" in body
        assert "data" in body
        # data 内部必须有 word_name 和 confidence
        assert set(body["data"].keys()) == {"word_name", "confidence"}


# ============================================================
# 3. 接口二: GET /api/sign/downloads
# ============================================================

class TestDownloadsEndpoint:
    """测试资源索引接口 (真实数据库 + 真实文件系统)"""

    def _get_one_word_from_db(self):
        """从数据库取一个真实存在的 word_name"""
        conn = pymysql.connect(**DB_CONFIG, cursorclass=pymysql.cursors.DictCursor)
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT word_name FROM sign_assets LIMIT 1")
                row = cursor.fetchone()
        finally:
            conn.close()
        return row["word_name"] if row else None

    def test_downloads_success(self, client):
        """用真实的 word_name 请求 → 应返回 code=200 + urls 列表"""
        word = self._get_one_word_from_db()
        if word is None:
            pytest.skip("数据库中没有数据，无法测试")

        resp = client.get(f"/api/sign/downloads?name={word}")

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["code"] == 200
        assert "urls" in body["data"]
        assert isinstance(body["data"]["urls"], list)
        assert len(body["data"]["urls"]) > 0, f"词条 {word} 对应的 .glb 文件列表为空"
        # 每个 URL 应以 .glb 结尾
        for url in body["data"]["urls"]:
            assert url.endswith(".glb"), f"URL 不以 .glb 结尾: {url}"
        # URL 应有序
        assert body["data"]["urls"] == sorted(body["data"]["urls"]), "URL 列表未排序"
        print(f"\n  ✅ {word}: 返回 {len(body['data']['urls'])} 个 .glb URL")

    def test_downloads_case_insensitive(self, client):
        """用小写的 word_name 请求 → 应返回与大写一致的结果"""
        word = self._get_one_word_from_db()
        if word is None:
            pytest.skip("数据库中没有数据")

        resp_upper = client.get(f"/api/sign/downloads?name={word.upper()}")
        resp_lower = client.get(f"/api/sign/downloads?name={word.lower()}")

        body_upper = resp_upper.get_json()
        body_lower = resp_lower.get_json()
        assert body_upper == body_lower, "大小写查询结果不一致"
        print(f"\n  ✅ 大小写无关: {word.upper()} == {word.lower()}")

    def test_downloads_not_found(self, client):
        """查询不存在的词条 → 应返回 404"""
        resp = client.get("/api/sign/downloads?name=ZZZZZ_NOT_EXIST_12345")

        assert resp.status_code == 404
        body = resp.get_json()
        assert body["code"] == 404
        print(f"\n  ✅ 不存在的词条: 返回 404")


# ============================================================
# 4. 接口三: GET /result_3d/glb_models/<path>
# ============================================================

class TestServeGlbEndpoint:
    """测试静态 GLB 文件服务接口 (真实文件系统)"""

    def _get_one_glb_path(self):
        """从 GLB_ROOT 中找到一个真实的 .glb 文件相对路径"""
        if not os.path.exists(GLB_ROOT):
            return None
        for root, _, files in os.walk(GLB_ROOT):
            for f in files:
                if f.endswith('.glb'):
                    return os.path.relpath(os.path.join(root, f), GLB_ROOT)
        return None

    def test_serve_glb_returns_file(self, client):
        """请求真实的 .glb 文件 → 应返回 200 + 二进制内容"""
        rel_path = self._get_one_glb_path()
        if rel_path is None:
            pytest.skip(f"GLB_ROOT ({GLB_ROOT}) 中找不到 .glb 文件")

        # Windows 路径需要转为 URL 格式
        url_path = rel_path.replace("\\", "/")
        resp = client.get(f"/result_3d/glb_models/{url_path}")

        assert resp.status_code == 200
        assert len(resp.data) > 0, "返回的文件内容为空"
        print(f"\n  ✅ 获取文件成功: {url_path} ({len(resp.data)} bytes)")

    def test_serve_glb_not_found(self, client):
        """请求不存在的文件 → 应返回 404"""
        resp = client.get("/result_3d/glb_models/nonexistent_folder/fake_frame.glb")
        assert resp.status_code == 404
        print(f"\n  ✅ 不存在的文件: 返回 404")


# ============================================================
# 5. Pipeline 完整性验证
# ============================================================

class TestPipelineIntegrity:
    """验证推理流水线加载状态"""

    def test_model_loaded(self):
        """模型应已成功加载"""
        assert pipeline is not None
        assert pipeline.model is not None
        print(f"\n  ✅ 模型已加载到设备: {pipeline.device}")

    def test_label_map_loaded(self):
        """标签映射应已加载且非空"""
        assert len(pipeline.label_map) > 0, "label_map 为空"
        print(f"\n  ✅ 标签映射: {len(pipeline.label_map)} 个类别")

    def test_normalization_stats_loaded(self):
        """归一化统计量应已加载"""
        assert pipeline.mean is not None, "mean 未加载"
        assert pipeline.std is not None, "std 未加载"
        assert pipeline.mean.shape == (config.INPUT_SIZE,), f"mean 维度错误: {pipeline.mean.shape}"
        assert pipeline.std.shape == (config.INPUT_SIZE,), f"std 维度错误: {pipeline.std.shape}"
        print(f"\n  ✅ 归一化统计量: mean/std shape = {pipeline.mean.shape}")

    def test_smoother_initialized(self):
        """平滑器应已初始化"""
        assert pipeline.smoother is not None
        assert pipeline.smoother.alpha == config.ALPHA
        print(f"\n  ✅ 平滑器 alpha = {pipeline.smoother.alpha}")
