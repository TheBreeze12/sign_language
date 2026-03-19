"""
ONNX 模型推理测试脚本
检查模型输入输出格式，模拟输入数据进行推理测试
"""
import os
import numpy as np
import onnxruntime as ort
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

ONNX_MODEL_PATH = os.path.join(os.path.dirname(__file__), "output.onnx")

def inspect_onnx_model(model_path):
    """检查 ONNX 模型的输入输出格式"""
    print("=" * 80)
    print(f" 检查 ONNX 模型: {model_path}")
    print("=" * 80)

    if not os.path.exists(model_path):
        logging.error(f"模型文件不存在: {model_path}")
        return None

    try:
        # 创建推理会话
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
        logging.info("✓ 模型加载成功")

        # 获取输入信息
        print("\n 输入信息:")
        print("-" * 80)
        input_names = []
        input_shapes = {}
        input_types = {}

        for idx, input_info in enumerate(session.get_inputs()):
            name = input_info.name
            shape = input_info.shape
            dtype = input_info.type
            input_names.append(name)
            input_shapes[name] = shape
            input_types[name] = dtype

            print(f"  Input[{idx}]:")
            print(f"    • 名称: {name}")
            print(f"    • 形状: {shape}")
            print(f"    • 数据类型: {dtype}")

        # 获取输出信息
        print("\n 输出信息:")
        print("-" * 80)
        output_names = []
        output_shapes = {}
        output_types = {}

        for idx, output_info in enumerate(session.get_outputs()):
            name = output_info.name
            shape = output_info.shape
            dtype = output_info.type
            output_names.append(name)
            output_shapes[name] = shape
            output_types[name] = dtype

            print(f"  Output[{idx}]:")
            print(f"    • 名称: {name}")
            print(f"    • 形状: {shape}")
            print(f"    • 数据类型: {dtype}")

        return {
            'session': session,
            'input_names': input_names,
            'input_shapes': input_shapes,
            'input_types': input_types,
            'output_names': output_names,
            'output_shapes': output_shapes,
            'output_types': output_types
        }

    except Exception as e:
        logging.error(f"❌ 模型加载失败: {e}")
        return None


def create_mock_input(input_shapes, input_names):
    """根据输入形状创建模拟输入数据"""
    print("\n生成模拟输入数据:")
    print("-" * 80)

    mock_inputs = {}

    for name in input_names:
        shape = input_shapes[name]
        shape_list = list(shape)

        # 处理动态维度：把字符串维度替换成可推理的固定值
        fixed_shape = []
        for idx, dim in enumerate(shape_list):
            if isinstance(dim, str):
                fixed_shape.append(1 if idx == 0 else 224)
            elif dim is None:
                fixed_shape.append(1 if idx == 0 else 224)
            else:
                fixed_shape.append(int(dim))

        shape = tuple(fixed_shape)

        # 创建随机数据
        mock_data = np.random.randn(*shape).astype(np.float32)
        mock_inputs[name] = mock_data

        print(f"  {name}:")
        print(f"    • 形状: {mock_data.shape}")
        print(f"    • 数据类型: {mock_data.dtype}")
        print(f"    • 范围: [{mock_data.min():.3f}, {mock_data.max():.3f}]")

    return mock_inputs


def run_inference(session, input_names, output_names, mock_inputs):
    """运行模型推理"""
    print("\n 执行推理:")
    print("-" * 80)

    try:
        # 准备输入字典
        input_feed = {name: mock_inputs[name] for name in input_names}

        # 运行推理
        outputs = session.run(output_names, input_feed)

        logging.info("✓ 推理完成")

        print("\n 推理结果:")
        print("-" * 80)

        for idx, (name, data) in enumerate(zip(output_names, outputs)):
            print(f"  Output[{idx}] - {name}:")
            print(f"    • 形状: {data.shape}")
            print(f"    • 数据类型: {data.dtype}")
            print(f"    • 范围: [{data.min():.6f}, {data.max():.6f}]")
            print(f"    • 均值: {data.mean():.6f}, 方差: {data.std():.6f}")

        return outputs

    except Exception as e:
        logging.error(f" 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  ONNX 模型推理测试".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")

    # 1. 检查模型结构
    model_info = inspect_onnx_model(ONNX_MODEL_PATH)
    if model_info is None:
        logging.error(" 无法继续测试")
        return

    # 2. 生成模拟输入
    mock_inputs = create_mock_input(
        model_info['input_shapes'],
        model_info['input_names']
    )

    # 3. 运行推理
    outputs = run_inference(
        model_info['session'],
        model_info['input_names'],
        model_info['output_names'],
        mock_inputs
    )

    # 总结
    print("\n" + "=" * 80)
    if outputs is not None:
        print(" 测试成功！ONNX 模型可以正常推理")
    else:
        print(" 测试失败！请检查模型或输入数据")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
