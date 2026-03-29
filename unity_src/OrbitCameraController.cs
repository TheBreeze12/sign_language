using UnityEngine;

public class OrbitCameraController : MonoBehaviour
{
    [SerializeField] private float lookSensitivity = 3f;
    [SerializeField] private float panSpeed = 0.01f;
    [SerializeField] private float zoomSpeed = 3f;
    [SerializeField] private float minPitch = -80f;
    [SerializeField] private float maxPitch = 80f;
    [SerializeField] private float roll = 180f; // 你想要设置的 Z 轴角度
    [SerializeField] private int panMouseButton = 2;
    
    private float yaw;
    private float pitch;

    private void Start()
    {
        Vector3 euler = transform.localEulerAngles;
        yaw = euler.y;
        pitch = NormalizeAngle(euler.x);
        
        // 初始化时应用 roll
        transform.rotation = Quaternion.Euler(pitch, yaw, roll);
    }

    private void Update()
    {
        bool leftMousePressed = Input.GetMouseButton(0);
        bool isRotating = leftMousePressed && (Input.GetKey(KeyCode.LeftAlt) || Input.GetKey(KeyCode.RightAlt));
        bool isPanning = Input.GetMouseButton(panMouseButton);

        if (isRotating)
        {
            float mouseX = Input.GetAxis("Mouse X");
            float mouseY = Input.GetAxis("Mouse Y");

            // --- 核心逻辑：处理反转 ---
            // 检查当前的 roll 是否让摄像机倒置了（即绝对值大于 90 度）
            // 如果倒置了，鼠标向右移动时，我们应该减去 yaw 而不是加上它
            float normalizedRoll = NormalizeAngle(roll);
            if (Mathf.Abs(normalizedRoll) > 90f)
            {
                yaw -= mouseX * lookSensitivity; // 倒置时反向增加偏航角
            }
            else
            {
                yaw += mouseX * lookSensitivity; // 正常操作
            }

            pitch -= mouseY * lookSensitivity;
            pitch = Mathf.Clamp(pitch, minPitch, maxPitch);
            
            // 应用旋转，包含你设定的 roll
            transform.rotation = Quaternion.Euler(pitch, yaw, roll);
        }

        // 平移逻辑也需要根据是否倒置进行修正（可选）
        if (isPanning)
        {
            float deltaX = -Input.GetAxis("Mouse X");
            float deltaY = -Input.GetAxis("Mouse Y");
            // transform.right 和 transform.up 会根据当前旋转自动调整，所以通常不需要额外修改
            Vector3 translation = (transform.right * deltaX + transform.up * deltaY) * panSpeed;
            transform.position += translation;
        }

        float scroll = Input.GetAxis("Mouse ScrollWheel");
        if (Mathf.Abs(scroll) > Mathf.Epsilon)
        {
            transform.position += transform.forward * (scroll * zoomSpeed);
        }
    }

    private static float NormalizeAngle(float angle)
    {
        while (angle > 180f) angle -= 360f;
        while (angle < -180f) angle += 360f;
        return angle;
    }
}