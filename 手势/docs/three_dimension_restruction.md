你这套链路，核心上是 3 段：

1. Python 用 MediaPipe 从视频逐帧提取 21 个手部 landmark。
2. Python 把每一帧整理成 Unity 能直接反序列化的 **json/jsonl**。
3. Unity 读取这些帧数据，用 landmark 先驱动 21 个调试点，再把点之间的方向转换成 FBX 手骨的旋转和手掌位姿。

**一、你是怎么提取 MediaPipe 数据的**

在 [offline_hand_pipeline.py](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/) 里，真正做提取的是 **extract_frames_from_video()**，入口在大约 [173](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/) 行。

它的流程是：

* 用 **cv2.VideoCapture** 打开视频。
* 用 **mp.solutions.hands.Hands(...)** 初始化 MediaPipe Hands，参数里允许最多两只手，检测和跟踪阈值都是 **0.5**，见 [188-195 附近](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)。
* 每读一帧，就转成 RGB，然后 **hands.process(rgb)** 做检测，见 [203-209](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)。
* 如果检测到手，就遍历 **result.multi_hand_landmarks**，把 21 个点都存成：

  * **id**
  * **x**
  * **y**
  * **z**

  这部分在 [209-235](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)。
* 同时保存 **hand_type**（Left/Right）、**bound_area**（手在画面中的包围面积，用来粗略表达远近/大小），**hand_gesture** 先写成 **"unknown"**，见 [231-234](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)。
* 每帧最终会形成一个 packet：

  * **frame_index**
  * **timestamp_ms**
  * **frame_time_sec**
  * **hand_count**
  * **hands**

  见 [243-251](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)。

这一步提取出来的坐标，还是 MediaPipe 的归一化坐标系，不是 Unity 世界坐标。

**二、你是怎么生成 JSON 的**

提取完以后，Python 还做了两步后处理：

* **_interpolate_missing_hands()**：如果某只手短暂丢帧，会在前后两帧之间插值补点，见 [82](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)。
* **_smooth_landmarks_savgol()**：对每个 landmark 的 **x/y/z** 做 Savitzky-Golay 平滑，减少抖动，见 [119](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)。

然后有两种导出方式：

* **export_unity_json()**：导出一个完整 JSON 文件，见 [276](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)
* **export_unity_gesture_stream()**：导出 JSONL，一行一帧，见 [314](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)

你 Unity 现在实际在用的是第二种，也就是 **jsonl** 流式格式，因为 [Test.cs](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/) 里读的是 **offlineGestureStreamPath**，并且按“每一行一个 frame”来解析，见 [88](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)、[163-198](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)。

也就是说，虽然文件里写了 **export_unity_json()**，但当前命令行参数并没有把 **export-json** 模式开放出来。你现在主要走的是 **jsonl**。

如果你单文件导出，实际命令思路就是：

`<span><span>python offline_hand_pipeline.py \   --mode export-gesture-stream \   --video your_video.mp4 \   --output-jsonl unity_gesture_stream_xxx.jsonl </span></span>`

**三、Unity 是怎么用这些数据重建手部的**

Unity 侧的结构定义在 [Test.cs](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/) 开头：

* **HandLandmarkData**
* **HandData**
* **GestureData**

见 [14-40](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)。

也就是说，Python 导出的字段名，就是按 Unity **JsonUtility.FromJson`<T>`()** 的字段名来对齐的。

运行流程是：

* **Start()** 里如果 **useOfflineGestureStream = true**，先 **PreloadGestureFrames()** 预加载 JSONL，再让 **VideoPlayer** 播放视频，见 [132-157](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)。
* **PreloadGestureFrames()** 会把每一行 JSONL 读成一个 **GestureData**，同时单独取出 **timestamp_ms** 建时间索引，见 [163-198](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)。
* **Update()** 里根据 **videoPlayer.time * 1000** 找到最接近当前视频时间的那一帧，再调用 **ProcessGestureData()**，见 [282-306](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)。

真正“重建手”的核心都在 [ProcessGestureData()](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)。

它做了 4 件事：

1. **先更新 21 个 landmark 可视化点**
   左手和右手都会把每个 **landmark.id** 对应到一个 sphere：

   * 左手见 [821-826](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)
   * 右手见 [883-888](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)

   坐标写法是：

   * **x = landmark.x**
   * **y = landmark.y**
   * **z = landmark.z * 4**

   说明你把 MediaPipe 的 z 做了一个放大，方便 Unity 里观察深度变化。
2. **先算手掌根骨/手腕的整体朝向**
   你拿了 3 个关键点：

   * 0: wrist
   * 5: index MCP
   * 9: middle MCP

   用 **wrist->middle** 和 **wrist->index** 两个向量做正交化，再叉乘出法向量，然后用 **Quaternion.LookRotation()** 生成手掌根节点朝向：

   * 左手见 [829-843](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)
   * 右手见 [892-907](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)

   这一步本质上是在用 3 个 landmark 建一个“手掌局部坐标系”。
3. **再算手掌整体位置**
   你没有把手直接放到 landmark 原坐标，而是做了一个经验映射：

   * 左手：**(mark0.x / 2 - 0.15, -mark0.y / 2 - 0.3, mark0.z + bound_area)**，见 [845-846](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)
   * 右手：**(mark0.x / 2 + 0.15, -mark0.y / 2 - 0.3, mark0.z + bound_area)**，见 [908-909](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)

   这里有几个明显意图：

   * **/2**：把 MediaPipe 归一化坐标缩到 Unity 场景里合适的范围
   * **-y**：把图像坐标系翻成 Unity 更自然的方向
   * 左右手分别加 **-0.15 / +0.15**：让两只手在场景里分开
   * **z + bound_area**：用手面积补偿深度感，手越大越靠前
4. **最后驱动每根手指骨骼**
   你不是直接拿每个点去摆网格，而是让骨骼的 **forward** 指向“当前关节到下一个关节”的方向：

   * 左手见 [848-868](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)
   * 右手见 [911-931](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)

   核心公式就是：

   * **direction = nextSphere.position - currentSphere.position**
   * **bone.transform.forward = direction.normalized**

   然后你又把每根骨头的 **z** 轴欧拉角强行改回建模初始值：

   * 左手：**euler.z = origin_leftRotations[index].z**
   * 右手：**euler.z = origin_rightRotations[index].z**

   这说明你已经发现：只靠 landmark 算出来的方向，容易把骨骼“拧麻花”；所以保留原始 rig 的 z 轴扭转，只让主要弯曲方向跟随 landmark。

**四、**LeftHand.fbx** / **RightHand.fbx** 在这里扮演什么角色**

这两个 FBX 文件：

* [LeftHand.fbx](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)
* [RightHand.fbx](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)

在这套系统里本质上是“被驱动的手部骨骼模型”。

虽然我没有直接展开 FBX 二进制内部骨骼树，但从场景 [Test.unity](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/) 可以确认：

* 场景里确实有 **RightHand** 对象，见 [3601](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)
* 场景里确实有 **LeftHand** 对象，见 [6616](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)
* **Test.cs** 挂在场景对象上，并且已经配置了 **left_points/right_points**、**left_spheres/right_spheres** 这些引用，见 [1467-1533 附近](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/17849/.vscode/extensions/openai.chatgpt-26.325.21211-win32-x64/webview/)

所以完整关系是：

* **LeftHand.fbx / RightHand.fbx** 提供手模型和骨骼层级
* **left_points / right_points** 指向这些骨骼节点
* **left_spheres / right_spheres** 是 21 个 landmark 对应的中间可视化点
* **ProcessGestureData()** 先更新 spheres，再用 spheres 之间的方向更新 points，也就是更新 FBX 骨骼

**五、把整条链路用一句话串起来**

你现在的系统其实是：

**视频 -> MediaPipe Hands -> 逐帧 21 点 landmark -> 插值补帧 + Savitzky-Golay 平滑 -> 导出 Unity 可读 JSONL -> Unity 按 timestamp 对齐视频 -> landmark 驱动调试点 -> 调试点方向驱动 FBX 手骨 -> 重建左右手动画**

**六、这套方案的几个关键特点**

* 它不是“直接还原真实 3D 手”，而是“用 MediaPipe 的归一化 landmark 去近似驱动一个现成的手 rig”。



# 可能的问题


### 1. 动态包围盒补偿（`bound_area`）是怎么实现的？

**背景痛点：** MediaPipe 提取的 21 个关键点的 `z` 值，是一个 **相对深度** （以手腕点为原点，相对于手掌大小的深度），它并不能告诉你“整只手距离摄像头有多远”。如果不做处理，无论视频里的人手离镜头近还是远，Unity 里的 3D 手都在同一个固定平面上。

**实现原理：**
在你们的 Python 提取阶段，计算了 `bound_area`（手部关键点在 2D 画面中所占的包围盒面积）。根据透视原理“近大远小”，手离镜头越近，`bound_area` 越大；离得越远，面积越小。
在 Unity 的 `Test.cs` 中，你们将这个 2D 面积直接作为 Z 轴的深度偏移量叠加到了手腕根节点上：
`right_points[0].transform.localPosition = new Vector3(..., ..., mark0.z + hand.bound_area);`

**最终效果：**
当视频里的人把手伸向镜头时（面积变大），Unity 里的 3D 手部模型也会随着 `bound_area` 的增大而在 Z 轴上向屏幕外凸出，产生真实的“前后空间位移感”，物理透视更加真实。

---

### 2. Z 轴空间放大（`landmark.z * 4`）是怎么实现的？

**背景痛点：**
MediaPipe 给出的相对深度（z 值）通常数值非常小。如果直接把 `(x, y, z)` 1:1 映射到 Unity 空间中，你会发现生成的手非常“扁平”，就像一张纸片，侧面看手指的厚度不够。

**实现原理：**
在 Unity 更新 21 个虚拟球体（Spheres）位置时，你们对 Z 轴坐标做了一个显式的乘法增益：
`right_spheres[landmark.id].transform.localPosition = new Vector3(landmark.x, landmark.y, landmark.z * 4);`

**最终效果：**
这相当于把手沿着深度方向（前后方向）“拉伸”了 4 倍。因为 X 和 Y 没有放大，所以从正面看手的大小不变，但当手掌发生倾斜、手指重叠时，手指与手指之间的前后距离被放大了。这极大增强了三维立体感，让原本在二维里重叠的动作在三维里拉开了层次。

---

### 3. 锁定欧拉角是什么意思？（防扭曲骨骼驱动）

要理解这个，首先要理解 3D 空间中的 **欧拉角（Euler Angles）** 。
欧拉角就是物体绕着 X、Y、Z 三个坐标轴的旋转角度（通常对应 Pitch 俯仰、Yaw 偏航、Roll 翻滚）。

**背景痛点：**
在你们的代码中，为了让 3D 骨骼对齐 MediaPipe 的点，你们计算了两个球体之间的方向向量（`direction`），然后用 `transform.forward = direction.normalized;` 让手指骨骼指向下一个关节。
**问题在于：** 当你仅仅告诉 Unity “让这根骨头指向前方那个点”时，Unity 只保证了“指的方向（Z轴前向）”是对的，但它 **不知道这根骨头自身应该怎么扭转（Roll）** 。默认的计算往往会导致骨骼沿着自身的长轴发生随机扭转，映射到 3D 模型上，就是手指的网格皮肤像“拧毛巾”一样畸变、扭在一起（术语叫万向节死锁或轴向丢失）。

**“锁定欧拉角”的实现原理：**
正常人的手指弯曲，主要是沿着关节的一个轴（比如 X 轴）做屈伸， **手指本身是不能沿着指骨自转的** 。
所以你们在代码里做了非常精妙的一步拦截：

1. 先让骨骼指向目标（拿到一个带有畸变的旋转 `currentRotation`）。
2. 把这个旋转拆解成 X、Y、Z 三个欧拉角（`Vector3 euler = currentRotation.eulerAngles;`）。
3. **关键点：** 把代表手指“自转/扭转”的 Z 轴角度，强行替换回模型初始状态下最完美的 Z 轴角度（`euler.z = origin_leftRotations[index].z;`）。
4. 再把修改后的旋转应用回去（`left_points[index].transform.localRotation = Quaternion.Euler(euler);`）。

**最终效果：**
这就叫“锁定欧拉角的 Z 轴（扭转角）”。它允许手指自由地跟随坐标点进行上下左右的屈伸弯曲，但 **绝对禁止手指发生自转** 。从根本上保护了 FBX 模型的拓扑结构，让渲染出来的手语动作既精准又自然，不会出现反人类的惊悚扭曲。



### 为什么会发生“全局抖动”与“拧麻花”？

#### 1. 为什么手掌整体姿态容易抖动或偏转？

MediaPipe 输出的是 21 个在笛卡尔坐标系下的绝对坐标（或者相对于摄像机的坐标）。但在 3D 引擎中，FBX 模型是一棵标准的 **多叉树数据结构** ，它的运动是基于父子层级（Parent-Child Hierarchy）的局部变换驱动的。

手掌根部（Wrist）通常是这棵树的根节点（Root）。要确定根节点的空间姿态，我们需要构建一个正交基（三个互相垂直的轴 **$\vec{X}, \vec{Y}, \vec{Z}$**）。如果仅仅依靠手腕、食指根部、小指根部这几个散点连线来计算法向量， **单目视觉在 **$Z$** 轴（深度）上的估算极不稳定** 。只要这几个点中有一个点在 **$Z$** 轴上出现几毫米的像素级闪烁（高频噪声），通过叉乘计算出的法向量就会发生剧烈的角度偏移。这就导致了整棵“骨骼树”的根节点发生偏转，宏观表现就是手掌在疯狂抖动。

#### 2. 为什么手指会“拧麻花”（局部拓扑畸变）？

这是最典型的自由度（Degree of Freedom, DOF）约束失效问题。

人手的手指关节大多是铰链关节（Hinge Joint），比如指间关节只能绕着单一轴（如局部 **$X$** 轴）做屈伸，自由度基本为 1。**手指是不能沿着自身长轴发生“自转”的。**

但是，当我们用两点之间的向量（例如从点 5 指向点 6 的 **$\vec{v}$**）去驱动骨骼时，我们实际上是命令引擎：“让这根骨头的正前方对准目标点”。这里存在一个数学上的无数组：**有无数种旋转姿态都能让骨头指向同一个目标。** 3D 引擎底层的四元数或 LookRotation 算法在计算时，为了走最短路径，极容易引入一个绕着骨骼纵轴的 **滚动角（Roll）** 。因为网格（Mesh）是蒙皮绑定在骨骼上的，骨骼一旦发生未经约束的 Roll 旋转，表皮就会像拧糖果纸一样扭曲。此外，当手指向上或向下弯曲接近 90 度时，欧拉角的计算极易陷入**万向节死锁（Gimbal Lock）** ，导致轴向瞬间翻转，原本用于屈伸的轴突然变成了自转轴，瞬间引发严重的畸变。
