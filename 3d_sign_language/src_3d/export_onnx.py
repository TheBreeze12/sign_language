"""
PyTorch ONNX 模型导出脚本
将训练好的 S2HAND PyTorch 模型 (.pth) 导出为 ONNX

使用方法:
python export_onnx.py --input checkpoints.pth --output model.onnx
"""

import os
import sys
import argparse
import numpy as np
import warnings

# Some third-party deps in S2HAND may load duplicate OpenMP runtimes on Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import torch
except (ImportError, OSError) as e:
    print(f"[ERROR] Failed to import PyTorch: {e}")
    sys.exit(1)

# 添加 S2HAND 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
s2hand_path = os.path.join(parent_dir, "..", "s2hand_code", "S2HAND")
examples_path = os.path.join(s2hand_path, "examples")
sys.path.insert(0, s2hand_path)
sys.path.insert(0, examples_path)


class ConfigStub:
    """最小配置，满足 S2HAND PyTorch Model 初始化。"""

    def __init__(self):
        self.train_requires = ["joints", "verts"]
        self.test_requires = ["joints", "verts"]
        self.regress_mode = "mano"
        self.use_mean_shape = False
        self.use_2d_as_attention = False
        # 使用 'NR' 让 models_new.Model 在 __init__ 中创建 self.renderer_NR 属性
        # 在当前配置下该属性会是 None，不会真正调用神经渲染器。
        self.renderer_mode = "NR"
        self.texture_mode = "surf"
        self.image_size = 224
        self.train_datasets = ["FreiHand"]
        self.use_pose_regressor = False
        self.pretrain_model = None
        self.pretrain_segmnet = None
        self.pretrain_texture_model = None
        self.pretrain_rgb2hm = None


def load_pytorch_weights(model, checkpoint_path):
    """按原 train_utils.py 逻辑加载 .pth 权重，但固定 map_location=cpu。"""
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "encoder" in state_dict and hasattr(model, "encoder"):
        model.encoder.load_state_dict(state_dict["encoder"])
        print("load encoder")
    if "decoder" in state_dict and hasattr(model, "hand_decoder"):
        model.hand_decoder.load_state_dict(state_dict["decoder"])
        print("load hand_decoder")
    if "heatmap_attention" in state_dict and hasattr(model, "heatmap_attention"):
        model.heatmap_attention.load_state_dict(state_dict["heatmap_attention"])
        print("load heatmap_attention")
    if "rgb2hm" in state_dict and hasattr(model, "rgb2hm"):
        model.rgb2hm.load_state_dict(state_dict["rgb2hm"])
        print("load rgb2hm")
    if "hm2hand" in state_dict and hasattr(model, "hm2hand"):
        model.hm2hand.load_state_dict(state_dict["hm2hand"])
        print("load hm2hand")
    if "mesh2pose" in state_dict and hasattr(model, "mesh2pose"):
        model.mesh2pose.load_state_dict(state_dict["mesh2pose"])
        print("load mesh2pose")


def patch_numpy_for_chumpy():
    """Provide deprecated numpy aliases required by old chumpy."""
    alias_pairs = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
    }
    for alias, target in alias_pairs.items():
        if not hasattr(np, alias):
            setattr(np, alias, target)
    if not hasattr(np, "unicode"):
        setattr(np, "unicode", str)


class OnnxExportWrapper(torch.nn.Module):
    """包装 S2HAND 前向，导出关键的 3D 手部参数。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        # models_new.forward(task='test') 会走 evaluate_iou 分支（该仓库缺失该方法）
        # 导出时应走 predict_singleview 分支，因此使用 task='train'。
        outputs = self.model(images=images, task="train", requires=["joints", "verts"])

        joints = outputs["joints"]           # (B, 21, 3)
        vertices = outputs["vertices"]       # (B, 778, 3)

        # 返回最核心的两个输出
        return joints, vertices


def main():
    parser = argparse.ArgumentParser(description="Export S2HAND PyTorch model to ONNX")
    parser.add_argument("--input", type=str, required=True, help="Input model path (.pth)")
    parser.add_argument("--output", type=str, required=True, help="Output ONNX model path (.onnx)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size (default: 224)")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version (default: 13)")
    args = parser.parse_args()

    print("Backend: pytorch")
    print(f"Input model: {args.input}")
    print(f"Output ONNX: {args.output}")

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    print("\n[1/3] Importing PyTorch model definition...")
    # 兼容旧 chumpy 别名时抑制 numpy FutureWarning，避免污染导出日志。
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        patch_numpy_for_chumpy()
    from examples.models_new import Model

    print("\n[2/3] Building model and loading .pth weights...")
    config = ConfigStub()
    model = Model(filename_obj=None, args=config)
    load_pytorch_weights(model, args.input)
    model.eval()

    print("\n[3/3] Exporting ONNX...")
    wrapper = OnnxExportWrapper(model).eval()
    dummy_input = torch.randn(args.batch_size, 3, args.image_size, args.image_size, dtype=torch.float32)

    # 对应 OnnxExportWrapper 的 2 个输出：joints, vertices
    output_names = ["joints", "vertices"]

    dynamic_axes = {
        "input": {0: "batch"},
        "joints": {0: "batch"},
        "vertices": {0: "batch"},
    }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            args.output,
            dynamo=False,
            input_names=["input"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
        )

    print(f"✓ ONNX model exported to: {args.output}")


if __name__ == "__main__":
    main()
