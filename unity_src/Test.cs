using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Android;
using System.Linq;
using Unity.VisualScripting;

[System.Serializable]
public class HandLandmarkData
{
    public int id;
    public float x;
    public float y;
    public float z;
}

[System.Serializable]
public class HandData
{
    public int hand_index;
    public string hand_type;
    public float bound_area;
    public string hand_gesture;
    public HandLandmarkData[] landmarks;
}

[System.Serializable]
public class GestureData
{
    public int hand_count;
    public HandData[] hands;
}

public class Test : MonoBehaviour
{
    public GameObject left_hand;
    public GameObject right_hand;
    public GameObject target;
    public Hold left_hold;
    public Hold right_hold;
    public Text left_result;
    public Text right_result;

    // 缂╂斁鎿嶇旱
    public Vector3 LastLeftHandPosition;
    public Vector3 LastRightHandPosition;
    public Quaternion LastLeftHandRotation;
    public Quaternion LastRightHandRotation;
    private float lastHandDistance;

    public List<GameObject> left_points = new List<GameObject>();
    public List<GameObject> right_points = new List<GameObject>();
    public List<GameObject> left_spheres = new List<GameObject>(); 
    public List<GameObject> right_spheres = new List<GameObject>(); 
    public GameObject sphere;

    private List<Vector3> origin_leftRotations = new List<Vector3>();
    private List<Vector3> origin_rightRotations = new List<Vector3>();

    // 手势滤波：滑动窗口多数表决
    [SerializeField] private int gestureWindowSize = 8; // 绐楀彛澶у皬锛堝彲璋冿紝3-7杈冨悎閫傦級
    private Queue<string> leftGestureHistory = new Queue<string>(); // 宸︽墜鎵嬪娍鍘嗗彶
    private Queue<string> rightGestureHistory = new Queue<string>(); // 鍙虫墜鎵嬪娍鍘嗗彶
    private string filteredLeftGesture = ""; // 杩囨护鍚庣殑宸︽墜鎵嬪娍
    private string filteredRightGesture = ""; // 杩囨护鍚庣殑鍙虫墜鎵嬪娍

    // 卡尔曼滤波
    private HandLandmarkFilter landmarkFilter;

    // 缃戠粶閰嶇疆
    [SerializeField] private string serverIP = "117.50.46.14";
    [SerializeField] private int serverPort = 8080;
    [SerializeField] private bool useOfflineGestureStream = false;
    [SerializeField] private string offlineGestureStreamPath = "";
    [SerializeField] private bool invertOfflineHandedness = true;

    // 鎽勫儚澶村拰缃戠粶鐩稿叧
    private WebCamTexture webcamTexture;
    public RGBCameraExample arCamera;
    public RawImage arCameraImage;
    public RenderTexture arCameraTexture;
    public ComputeShader YUV2RGBShader;
    private TcpClient client;
    private NetworkStream stream;
    private Thread sendThread;
    private bool isRunning = false;
    private bool disconnecting = false;
    private readonly Queue<byte[]> frameQueue = new Queue<byte[]>();
    private readonly object queueLock = new object();
    [Range(10, 100)]
    [SerializeField] private int jpegQuality = 100;

    // 帧数据队列（线程安全）

    // 鍥惧儚鍘嬬缉璐ㄩ噺

    // 帧处理间隔（毫秒）
    [SerializeField] private int frameInterval = 33;
    private float lastFrameTime = 0;
    private bool captureThisFrame = false;
    private Texture2D texture;
    [SerializeField] private Texture2D testImage;
    private bool permissionGranted = false;
    private bool permissionRequested = false;
    private Thread receiveThread;
    private bool isReceiving = false;

    // 鏍囪鏄惁闇€瑕佹崟鑾峰抚

    // 鏉冮檺璇锋眰鐩稿叧

    // 鏂板锛氭帴鏀剁嚎绋嬪拰鐩稿叧鍙橀噺
    private Coroutine offlinePlaybackCoroutine;
    private bool runtimeInitialized = false;
    private bool offlinePlaybackEnabled = false;
    private bool liveCaptureEnabled = false;

    void Awake()
    {
        EnsureRuntimeInitialized();
    }

    void Start()
    {
        // 确保主线程调度器存在
        EnsureRuntimeInitialized();
        if (UnityMainThreadDispatcher.Instance == null)
        {
            new GameObject("UnityMainThreadDispatcher").AddComponent<UnityMainThreadDispatcher>();
        }

        // 初始化滤波器，可调整噪声参数
        landmarkFilter = landmarkFilter ?? new HandLandmarkFilter(0.008f, 0.03f);

        if (origin_leftRotations.Count == 0 && origin_rightRotations.Count == 0)
        {
            for (int i = 0; i < Mathf.Min(left_points.Count, right_points.Count); i++)
            {
                if (left_points[i] != null)
                {
                    origin_leftRotations.Add(left_points[i].transform.localEulerAngles);
                }

                if (right_points[i] != null)
                {
                    origin_rightRotations.Add(right_points[i].transform.localEulerAngles);
                }
            }
        }

        offlinePlaybackEnabled = ShouldEnableOfflineGestureStream();

        if (offlinePlaybackEnabled)
        {
            StartOfflineGesturePlayback();
        }
    }

    void StartOfflineGesturePlayback()
    {
        if (string.IsNullOrWhiteSpace(offlineGestureStreamPath))
        {
            Debug.LogError("Offline gesture stream path is empty.");
            return;
        }

        if (!File.Exists(offlineGestureStreamPath))
        {
            Debug.LogError("Offline gesture stream file not found: " + offlineGestureStreamPath);
            return;
        }

        string[] lines = File.ReadAllLines(offlineGestureStreamPath, Encoding.UTF8);
        offlinePlaybackCoroutine = StartCoroutine(PlayOfflineGestureStream(lines));
        Debug.Log("Started offline gesture playback: " + offlineGestureStreamPath);
    }

    IEnumerator PlayOfflineGestureStream(string[] lines)
    {
        float delaySeconds = Mathf.Max(frameInterval, 1) / 1000f;

        foreach (string rawLine in lines)
        {
            if (disconnecting)
            {
                yield break;
            }

            if (string.IsNullOrWhiteSpace(rawLine))
            {
                yield return new WaitForSeconds(delaySeconds);
                continue;
            }

            GestureData gestureData = JsonUtility.FromJson<GestureData>(rawLine);
            if (gestureData == null)
            {
                Debug.LogWarning("Failed to parse offline gesture line.");
                yield return new WaitForSeconds(delaySeconds);
                continue;
            }

            if (gestureData.hands == null)
            {
                gestureData.hands = new HandData[0];
            }

            ProcessGestureData(gestureData);
            yield return new WaitForSeconds(delaySeconds);
        }

        Debug.Log("Offline gesture playback finished.");
    }

    private bool ShouldEnableOfflineGestureStream()
    {
        if (!useOfflineGestureStream)
        {
            return false;
        }

#if UNITY_WEBGL && !UNITY_EDITOR
        Debug.LogWarning("Offline gesture stream playback is disabled in WebGL builds. Use GestureSyncController with an HTTP(S) JSON URL instead.");
        return false;
#else
        return true;
#endif
    }

    private bool ShouldEnableLiveCapture()
    {
#if UNITY_WEBGL && !UNITY_EDITOR
        return false;
#else
        return true;
#endif
    }

    void CheckCameraPermission()
    {
        if (offlinePlaybackEnabled || !liveCaptureEnabled)
        {
            return;
        }

        // 检查是否有摄像头权限
        if (Permission.HasUserAuthorizedPermission(Permission.Camera))
        {
            permissionGranted = true;
            InitializeWebcam();
            ConnectToServer();
        }
        else
        {
            // 请求摄像头权限
            if (!permissionRequested)
            {
                permissionRequested = true;
                Permission.RequestUserPermission(Permission.Camera);
            }
        }
    }

    void Update()
    {
        if (offlinePlaybackEnabled || !liveCaptureEnabled)
        {
            return;
        }

        // 检查权限请求结果
        if (!permissionGranted && permissionRequested)
        {
            if (Permission.HasUserAuthorizedPermission(Permission.Camera))
            {
                permissionGranted = true;
                InitializeWebcam();
                ConnectToServer();
            }
        }

        // 浠呭湪涓荤嚎绋嬩腑澶勭悊鎽勫儚澶村拰绾圭悊
        if (isRunning && !disconnecting)
        {
            // 鎺у埗甯х巼
            if (Time.time - lastFrameTime >= frameInterval / 1000f)
            {
                lastFrameTime = Time.time;

                // 鏍囪闇€瑕佹崟鑾峰綋鍓嶅抚
                captureThisFrame = true;
            }
        }

        if (captureThisFrame && texture != null)
        {
            captureThisFrame = false;
            CaptureFrame();
        }
    }

    void InitializeWebcam()
    {
        // 在 Android 上，需要确保在权限获取后执行
        if (!permissionGranted)
        {
            return;
        }

        // 获取可用摄像头列表
        WebCamDevice[] devices = WebCamTexture.devices;
        if (devices.Length == 0)
        {
            return;
        }

        // 浣跨敤绗竴涓憚鍍忓ご
        webcamTexture = new WebCamTexture(devices[0].name, 800, 400, 30);
        webcamTexture.Play();

        // 鍒涘缓涓€娆exture2D瀹炰緥
        texture = new Texture2D(arCameraImage.texture.width, arCameraImage.texture.height, ((Texture2D)arCameraImage.texture).format, true);

        Debug.Log(texture.width + " " + texture.height);
    }

    void CaptureFrame()
    {
        //Debug.Log("capture");
        //byte[] frameData = testImage.EncodeToJPG(jpegQuality);
        // 缂栫爜涓篔PEG锛堝繀椤诲湪涓荤嚎绋嬶級
        if (arCamera.y_texture == null) { return; }

        int width = arCamera.y_texture.width;
        int height = arCamera.y_texture.height;
        //int width = 640;
        //int height = 480;

        // 纭繚鎴戜滑鏈変竴涓悎閫傜殑Texture2D鐢ㄤ簬鎹曡幏
        // 纭繚鎴戜滑鏈変竴涓悎閫傜殑RenderTexture鐢ㄤ簬Compute Shader杈撳嚭
        if (arCameraTexture == null ||
            arCameraTexture.width != width ||
            arCameraTexture.height != height)
        {
            // 释放旧纹理
            if (arCameraTexture != null)
            {
                RenderTexture.ReleaseTemporary(arCameraTexture);
            }

            // 创建支持 UAV 的渲染纹理
            arCameraTexture = RenderTexture.GetTemporary(
                width,
                height,
                0,
                RenderTextureFormat.ARGB32,
                RenderTextureReadWrite.Linear
            );
            arCameraTexture.enableRandomWrite = true; // 鍏佽闅忔満鍐欏叆
            //arCameraTexture.creationFlags = RenderTextureCreationFlags.UnorderedAccess; // 鍚敤UAV
        }

        int kernelHandle = YUV2RGBShader.FindKernel("CSMain");

        YUV2RGBShader.SetTexture(kernelHandle, "YTexture", arCamera.y_texture);
        YUV2RGBShader.SetTexture(kernelHandle, "UTexture", arCamera.u_texture);
        YUV2RGBShader.SetTexture(kernelHandle, "VTexture", arCamera.v_texture);
        YUV2RGBShader.SetTexture(kernelHandle, "Result", arCameraTexture);

        // 计算线程组数量 (8x8 线程每组)
        int threadGroupsX = Mathf.CeilToInt(width / 8f);
        int threadGroupsY = Mathf.CeilToInt(height / 8f);

        // 鎵цCompute Shader
        YUV2RGBShader.Dispatch(kernelHandle, threadGroupsX, threadGroupsY, 1);

        // 鍒涘缓涓存椂Texture2D鐢ㄤ簬璇诲彇鍍忕礌
        Texture2D tempTexture = new Texture2D(arCameraTexture.width, arCameraTexture.height, TextureFormat.RGBA32, false);

        // 婵€娲籖enderTexture浠ヤ究璇诲彇
        RenderTexture.active = arCameraTexture;

        // 璇诲彇鍍忕礌鍒癟exture2D
        tempTexture.ReadPixels(new Rect(0, 0, arCameraTexture.width, arCameraTexture.height), 0, 0);
        tempTexture.Apply();

        // 恢复 RenderTexture 状态
        RenderTexture.active = null;

        // 缂栫爜涓篔PG
        byte[] frameData = tempTexture.EncodeToJPG();

        Destroy(tempTexture); // 销毁临时 Texture2D
        tempTexture = null;

        // 将帧数据加入队列（线程安全）
        if (frameData != null && frameData.Length > 0)
        {
            lock (queueLock)
            {
                // 限制队列大小，防止内存溢出
                while (frameQueue.Count > 3)
                {
                    frameQueue.Dequeue();
                }

                frameQueue.Enqueue(frameData);
            }
        }
        try
        {
           
        }
        catch (Exception e)
        {
            Debug.LogError("Capture frame failed: " + e.Message);
        }
    }

    void ConnectToServer()
    {
        try
        {
            // 鍒涘缓TCP瀹㈡埛绔苟杩炴帴鍒版湇鍔″櫒
            client = new TcpClient();
            IAsyncResult result = client.BeginConnect(serverIP, serverPort, null, null);

            // 设置连接超时时间（5 秒）
            bool success = result.AsyncWaitHandle.WaitOne(TimeSpan.FromSeconds(5));

            if (client.Connected)
            {
                client.EndConnect(result);
                stream = client.GetStream();
                isRunning = true;

                // 启动发送线程
                sendThread = new Thread(SendFrames);
                sendThread.IsBackground = true; // 设置为后台线程
                sendThread.Start();

                // 启动接收线程
                isReceiving = true;
                receiveThread = new Thread(ReceiveGestureData);
                receiveThread.IsBackground = true;
                receiveThread.Start();
            }
            else
            {
                client.Close();
            }
        }
        catch (Exception e)
        {

        }
    }

    void SendFrames()
    {
        try
        {
            while (isRunning && !disconnecting)
            {
                try
                {
                    // 检查客户端连接状态
                    if (client == null || !client.Connected)
                    {
                        Debug.LogWarning("Server connection lost.");
                        break;
                    }

                    // 从队列中获取帧数据（线程安全）
                    byte[] frameData = null;
                    lock (queueLock)
                    {
                        if (frameQueue.Count > 0)
                        {
                            frameData = frameQueue.Dequeue();
                        }
                    }

                    // 鍙戦€佸抚鏁版嵁
                    if (frameData != null && frameData.Length > 0)
                    {
                        // 鍙戦€佸抚鏁版嵁闀垮害 (4瀛楄妭鏁存暟)
                        byte[] lengthBytes = BitConverter.GetBytes(frameData.Length);

                        if (stream != null && client.Connected)
                        {
                            stream.Write(lengthBytes, 0, lengthBytes.Length);
                            stream.Write(frameData, 0, frameData.Length);
                        }
                    }
                    else
                    {
                        // 没有帧数据时短暂休眠
                        Thread.Sleep(10);
                    }
                }
                catch (Exception e)
                {
                    if (!disconnecting)
                    {
                        Debug.LogError("Send frame failed: " + e.Message);
                    }
                    break;
                }
            }
        }
        finally
        {
            if (!disconnecting)
            {
                Debug.Log("Send thread exited.");
                Disconnect();
            }
        }
    }

    // 鏂板锛氭帴鏀舵墜鍔挎暟鎹殑绾跨▼鍑芥暟
    void ReceiveGestureData()
    {
        try
        {
            while (isReceiving && !disconnecting)
            {
                if (stream == null || !client.Connected)
                {
                    Debug.LogWarning("Connection lost, cannot receive data.");
                    break;
                }

                // 先接收数据长度（4 字节）
                byte[] lengthBytes = new byte[4];
                int bytesRead = 0;
                try
                {
                    bytesRead = stream.Read(lengthBytes, 0, 4);
                }
                catch (Exception e)
                {
                    Debug.LogError("Read data length failed: " + e.Message);
                    break;
                }

                if (bytesRead != 4)
                {
                    Debug.LogWarning("Failed to receive data length; connection may be closed.");
                    break;
                }

                int dataLength = BitConverter.ToInt32(lengthBytes, 0);

                // 鎺ユ敹瀹為檯鏁版嵁
                byte[] dataBytes = new byte[dataLength];
                int totalRead = 0;
                while (totalRead < dataLength)
                {
                    try
                    {
                        int read = stream.Read(dataBytes, totalRead, dataLength - totalRead);
                        if (read == 0)
                        {
                            Debug.LogWarning("Receive data failed; connection may be closed.");
                            break;
                        }
                        totalRead += read;
                    }
                    catch (Exception e)
                    {
                        Debug.LogError("Read payload failed: " + e.Message);
                        break;
                    }
                }

                if (totalRead == dataLength)
                {
                    // 杞崲涓哄瓧绗︿覆
                    string jsonData = Encoding.UTF8.GetString(dataBytes);

                    // 在主线程中处理 JSON
                    UnityMainThreadDispatcher.Instance.Enqueue(() =>
                    {
                        // 鍙嶅簭鍒楀寲 JSON 鏁版嵁
                        GestureData gestureData = JsonUtility.FromJson<GestureData>(jsonData);

                        // 澶勭悊鎵嬪娍鏁版嵁
                        ProcessGestureData(gestureData);
                    });
                }
                else
                {
                    // 数据接收不完整
                    Debug.LogWarning($"Incomplete payload: expected {dataLength} bytes, received {totalRead} bytes");
                }
            }
        }
        catch (Exception e)
        {
            if (!disconnecting)
            {
                Debug.LogError("Receive gesture data failed: " + e.Message);
            }
        }
    }

    void OnApplicationQuit()
    {
        Disconnect();
    }

    void OnDestroy()
    {
        Disconnect();
    }

    void Disconnect()
    {
        // 闃叉閲嶅璋冪敤
        if (disconnecting) return;

        disconnecting = true;
        isRunning = false;
        isReceiving = false; // 停止接收线程

        try
        {
            Debug.Log("Disconnecting...");

            // 清空帧队列
            lock (queueLock)
            {
                frameQueue.Clear();
            }

            // 鍏抽棴缃戠粶娴佸拰瀹㈡埛绔紙瀛愮嚎绋嬫搷浣滐級
            if (stream != null)
            {
                try { stream.Close(); } catch { }
                stream = null;
            }

            if (client != null)
            {
                try { client.Close(); } catch { }
                client = null;
            }

            // 鍦ㄤ富绾跨▼涓仠姝㈡憚鍍忓ご
            if (webcamTexture != null)
            {
                webcamTexture.Stop();
                webcamTexture = null;
            }

            // 閲婃斁Texture2D璧勬簮
            if (texture != null)
            {
                Destroy(texture);
                texture = null;
            }

            if (offlinePlaybackCoroutine != null)
            {
                StopCoroutine(offlinePlaybackCoroutine);
                offlinePlaybackCoroutine = null;
            }

            // 等待发送线程退出（带超时）
            if (sendThread != null && sendThread.IsAlive)
            {
                Debug.Log("Waiting for send thread to exit...");
                if (!sendThread.Join(2000))
                {
                    Debug.LogWarning("Send thread did not exit within 2 seconds; aborting.");
                    try { sendThread.Abort(); } catch { }
                }
                sendThread = null;
            }

            // 等待接收线程退出（带超时）
            if (receiveThread != null && receiveThread.IsAlive)
            {
                Debug.Log("Waiting for receive thread to exit...");
                if (!receiveThread.Join(2000))
                {
                    Debug.LogWarning("Receive thread did not exit within 2 seconds; aborting.");
                    try { receiveThread.Abort(); } catch { }
                }
                receiveThread = null;
            }

            Debug.Log("Disconnected.");
        }
        catch (Exception e)
        {
            Debug.LogError("Disconnect failed: " + e.Message);
        }
    }

    /// <summary>
    /// 滑动窗口多数表决，过滤离散手势结果
    /// </summary>
    /// <param name="history">鎵嬪娍鍘嗗彶闃熷垪</param>
    /// <param name="newGesture">鏂拌瘑鍒殑鎵嬪娍</param>
    /// <param name="windowSize">绐楀彛澶у皬</param>
    /// <returns>杩囨护鍚庣殑鎵嬪娍缁撴灉</returns>
    private string FilterGesture(Queue<string> history, string newGesture, int windowSize)
    {
        // 1. 灏嗘柊缁撴灉鍔犲叆鍘嗗彶闃熷垪
        history.Enqueue(newGesture);

        // 2. 保证队列长度不超过窗口大小
        while (history.Count > windowSize)
        {
            history.Dequeue();
        }

        // 3. 缁熻闃熷垪涓瘡涓墜鍔跨殑鍑虹幇娆℃暟
        Dictionary<string, int> gestureCount = new Dictionary<string, int>();
        foreach (string gesture in history)
        {
            if (gestureCount.ContainsKey(gesture))
                gestureCount[gesture]++;
            else
                gestureCount[gesture] = 1;
        }

        // 4. 鎵惧埌鍑虹幇娆℃暟鏈€澶氱殑鎵嬪娍锛堝鏁拌〃鍐筹級
        int maxCount = 0;
        string majorityGesture = newGesture; // 榛樿鐢ㄦ柊缁撴灉
        foreach (var pair in gestureCount)
        {
            // 优先选择出现次数更多的手势
            if (pair.Value > maxCount)
            {
                maxCount = pair.Value;
                majorityGesture = pair.Key;
            }
            // 若次数相同，优先保留最近出现的结果
            else if (pair.Value == maxCount)
            {
                foreach (string g in history.Reverse()) // 浠庡悗寰€鍓嶆壘
                {
                    if (gestureCount[g] == maxCount)
                    {
                        majorityGesture = g;
                        break;
                    }
                }
            }
        }

        return majorityGesture;
    }


    public void ApplyGestureData(GestureData gestureData)
    {
        EnsureRuntimeInitialized();
        ProcessGestureData(gestureData);
    }

    private void EnsureRuntimeInitialized()
    {
        if (runtimeInitialized)
        {
            return;
        }

        landmarkFilter = landmarkFilter ?? new HandLandmarkFilter(0.008f, 0.03f);

        if (origin_leftRotations.Count == 0 && origin_rightRotations.Count == 0)
        {
            for (int i = 0; i < Mathf.Min(left_points.Count, right_points.Count); i++)
            {
                if (left_points[i] != null)
                {
                    origin_leftRotations.Add(left_points[i].transform.localEulerAngles);
                }

                if (right_points[i] != null)
                {
                    origin_rightRotations.Add(right_points[i].transform.localEulerAngles);
                }
            }
        }

        runtimeInitialized = true;
    }

    void ProcessGestureData(GestureData gestureData)
    {
        if (gestureData == null || gestureData.hands == null)
        {
            left_hand.SetActive(false);
            right_hand.SetActive(false);
            return;
        }

        bool is_left_detected = false;
        bool is_right_detected = false;
        // 遍历每只手
        foreach (HandData hand in gestureData.hands)
        {
            string effectiveHandType = hand.hand_type;
            if (hand.hand_type == "Left")
            {
                is_left_detected = true;

                // 鎵嬪娍婊戝姩绐楀彛杩囨护
                filteredLeftGesture = FilterGesture(leftGestureHistory, hand.hand_gesture, gestureWindowSize);
                Debug.Log("Left filtered: " + filteredLeftGesture);
                left_result.text = filteredLeftGesture;

                // 手势滑动窗口过滤
                landmarkFilter.FilterLeftHandLandmarks(hand.landmarks);

                foreach (HandLandmarkData landmark in hand.landmarks)
                {
                    // 【关键修改1：Z轴取反 (-landmark.z)，解决左手变成右手的镜像问题】
                    left_spheres[landmark.id].transform.localPosition = new Vector3(landmark.x, landmark.y, -landmark.z * 4);
                }

                // 0
                // 【注意这里的 Z 也要取反】
                Vector3 wristPosition = new Vector3(hand.landmarks[0].x, hand.landmarks[0].y, -hand.landmarks[0].z * 4);
                Vector3 indexPosition = new Vector3(hand.landmarks[5].x, hand.landmarks[5].y, -hand.landmarks[5].z * 4);
                Vector3 middlePosition = new Vector3(hand.landmarks[9].x, hand.landmarks[9].y, -hand.landmarks[9].z * 4);
                Vector3 vectorToMiddle = middlePosition - wristPosition;
                Vector3 vectorToIndex = indexPosition - wristPosition;

                Vector3.OrthoNormalize(ref vectorToMiddle, ref vectorToIndex);
                
                // 【关键修改2：左手的叉乘顺序必须与右手相反】
                // 刚才右手改成了 Cross(Middle, Index)，那么左手这里必须是 Cross(Index, Middle)
                Vector3 normalVector = Vector3.Cross(vectorToIndex, vectorToMiddle);
                
                left_points[0].transform.localRotation = Quaternion.LookRotation(normalVector, vectorToIndex) * Quaternion.Euler(0, 90, 90);

                left_points[0].transform.localRotation = Quaternion.Euler(
                    left_points[0].transform.localRotation.eulerAngles.x,
                    -left_points[0].transform.localRotation.eulerAngles.y,
                    -left_points[0].transform.localRotation.eulerAngles.z
                );

                HandLandmarkData mark0 = hand.landmarks[0];
                // 【位置的 Z 轴同样取反】
                left_points[0].transform.localPosition = new Vector3(mark0.x / 2 - 0.15f, -mark0.y / 2 - 0.3f, -mark0.z + hand.bound_area);

                // 1-5
                for (int i = 0; i < 5; i++)
                {
                    for (int j = 1; j <= 4; j++)
                    {
                        if (j < 4)
                        {
                            int index = i * 4 + j;
                            HandLandmarkData nextMark = hand.landmarks[index];
                            
                            // direction 会根据前面已经取反的 left_spheres 自动得出正确的左手向量
                            Vector3 direction = left_spheres[index + 1].transform.position - left_spheres[index].transform.position;
                            left_points[index].transform.forward = direction.normalized;

                            // 保持当前旋转，只修改 Z 轴
                            Quaternion currentRotation = left_points[index].transform.localRotation;

                            // 提取当前的 X/Y 旋转，固定 Z 轴
                            Vector3 euler = currentRotation.eulerAngles;
                            euler.z = origin_leftRotations[index].z;

                            // 搴旂敤鏂扮殑鏃嬭浆
                            left_points[index].transform.localRotation = Quaternion.Euler(euler);
                        }
                    }
                }
            }
            else if (hand.hand_type == "Right")
            {
                is_right_detected = true;

                // 2. 鎵嬪娍婊戝姩绐楀彛杩囨护
                filteredRightGesture = FilterGesture(rightGestureHistory, hand.hand_gesture, gestureWindowSize);
                Debug.Log("Right filtered: " + filteredRightGesture);
                right_result.text = filteredRightGesture;

                // 手势滑动窗口过滤
                landmarkFilter.FilterRightHandLandmarks(hand.landmarks);

                foreach (HandLandmarkData landmark in hand.landmarks)
                {
                    // 【关键修改1：将 Z 轴取反 (-landmark.z)，解决右手变成左手的镜像问题】
                    right_spheres[landmark.id].transform.localPosition = new Vector3(landmark.x, landmark.y, -landmark.z * 4);
                }

                // 0
                // 【注意这里的 Z 也要取反】
                Vector3 wristPosition = new Vector3(hand.landmarks[0].x, hand.landmarks[0].y, -hand.landmarks[0].z * 4);
                Vector3 indexPosition = new Vector3(hand.landmarks[5].x, hand.landmarks[5].y, -hand.landmarks[5].z * 4);
                Vector3 middlePosition = new Vector3(hand.landmarks[9].x, hand.landmarks[9].y, -hand.landmarks[9].z * 4);
                Vector3 vectorToMiddle = middlePosition - wristPosition;
                Vector3 vectorToIndex = indexPosition - wristPosition;

                Vector3.OrthoNormalize(ref vectorToMiddle, ref vectorToIndex);
                
                // 【关键修改2：右手叉乘顺序必须与左手相反 (中指在前，食指在后)】
                Vector3 normalVector = Vector3.Cross(vectorToMiddle, vectorToIndex); 
                
                right_points[0].transform.localRotation = Quaternion.LookRotation(normalVector, vectorToIndex) * Quaternion.Euler(0, 90, -90);
                
                // 建议保留原有的欧拉角翻转，除非模型再次发生扭曲（如果依然扭曲，尝试注释掉下面这三行）
                right_points[0].transform.localRotation = Quaternion.Euler(
                    right_points[0].transform.localRotation.eulerAngles.x,
                    -right_points[0].transform.localRotation.eulerAngles.y,
                    -right_points[0].transform.localRotation.eulerAngles.z
                );

                HandLandmarkData mark0 = hand.landmarks[0];
                // 【注意此处的 Z 轴最好也保持翻转逻辑，如果深度不对可以把负号去掉】
                right_points[0].transform.localPosition = new Vector3(mark0.x / 2 + 0.15f, -mark0.y / 2 - 0.3f, -mark0.z + hand.bound_area);

                // 1-5
                for (int i = 0; i < 5; i++)
                {
                    for (int j = 1; j <= 4; j++)
                    {
                        if (j < 4)
                        {
                            int index = i * 4 + j;
                            HandLandmarkData nextMark = hand.landmarks[index];
                            
                            // 因为前面 right_spheres 的 Z 已经取反，这里的 direction 会自动计算出正确的右手骨骼方向
                            Vector3 direction = right_spheres[index + 1].transform.position - right_spheres[index].transform.position;
                            right_points[index].transform.forward = direction.normalized;

                            // 保持当前旋转，只修改 Z 轴
                            Quaternion currentRotation = right_points[index].transform.localRotation;

                            // 提取当前的 X/Y 旋转，固定 Z 轴
                            Vector3 euler = currentRotation.eulerAngles;
                            euler.z = origin_rightRotations[index].z;

                            // 搴旂敤鏂扮殑鏃嬭浆
                            right_points[index].transform.localRotation = Quaternion.Euler(euler);
                        }
                    }
                }
            }
        }
        left_hand.SetActive(is_left_detected);
        right_hand.SetActive(is_right_detected);

        // 鎵嬪娍浜や簰
        left_hold.hold = right_hold.hold = false;
        if (filteredLeftGesture.Equals("7") && is_left_detected)
        {
            left_hold.hold = true;
        }
        if (filteredRightGesture.Equals("7") && is_right_detected)
        {
            right_hold.hold = true;
        }
        if(left_hold.hold && right_hold.hold)
        {
            left_hold.hold = right_hold.hold = false;

            float currentHandDistance = Vector3.Distance(left_hold.transform.position, right_hold.transform.position);
            float scaleFactor = currentHandDistance / lastHandDistance;
            target.transform.localScale *= scaleFactor;
        }
        LastLeftHandPosition = left_hold.transform.position;
        LastRightHandPosition = right_hold.transform.position;
        lastHandDistance = Vector3.Distance(LastLeftHandPosition, LastRightHandPosition);
    }
}
