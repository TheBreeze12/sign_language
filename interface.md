后端在localhost:5000端口

接口一：模型推理接口
路由： POST /api/sign/predict
Content-Type： multipart/form-data
Request Body：
字段类型必填说明

| videoFile | ✅用户录制的手语片段（mp4 / webm / avi） |
| --------- | ---------------------------------------- |

后端处理流程（对应你的代码）：

- 用 OpenCV VideoCapture 逐帧解码上传的视频文件
- 每帧调用 pipeline.extract_features(frame) → 提取 134 维坐标每帧经过 pipeline.smoother.smooth(raw_feat) → 平滑处理
- 攒成 frame_buffer 后调用 pipeline.predict(frame_buffer)
- 内部走 preprocess_sequence → 双重相对坐标 + 线性抽样 → 模型推理

```
Response Body：
json{
  "code": 200,
  "data": {
    "word_name": "ACCIDENT",
    "confidence": 0.98
  }
}
错误响应：
json{
  "code": 400,
  "message": "Video too short, minimum 5 frames required"
}
```

对应代码中 preprocess_sequence 的 if len(raw_seq) < 5: return None 分支。

接口二：资源索引接口
路由： GET /api/sign/downloads
Query Params： name=ACCIDENT
功能： 查 MySQL sign_assets 表 → 找到物理文件夹 → 扫描并返回 Web URLs。

表名sign_language_db

| **字段名**           | **类型** | **与接口的关联说明**                                                                                                                                                                                       |
| -------------------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`word_name`**    | `VARCHAR`    | **核心枢纽** 。它接收**接口一**预测出的 `data.word_name`（如 "ACCIDENT"），并作为**接口二** `Query Params: name=ACCIDENT`的搜索条件。加了 `UNIQUE KEY`保证查询是**$O(1)$**的极速响应。 |
| **`folder_name`**  | `VARCHAR`    | **底层映射** 。接口二查到这行数据后，取出这个字段拼接到 `GLB_ROOT`后面，然后用 `os.walk()`去遍历这个文件夹。                                                                                           |
| **`total_frames`** | `INT`        | **优化预留** 。虽然接口二目前是靠 `os.walk`查出来的 `urls`数组长度来确定帧数，但在数据库里存一下总帧数，可以在数据校验或者前端做加载进度条时提供很大便利。                                             |
