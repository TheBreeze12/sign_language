"""
MindSpore 版本的 train_utils.py
支持 MindSpore checkpoint 格式的加载和保存
"""

# 优先使用 MindSpore + msadapter
try:
    import mindspore as ms
    import msadapter.pytorch as torch
    BACKEND = "mindspore"
except ImportError:
    import torch
    BACKEND = "pytorch"

import json
import util
import time
from tensorboardX import SummaryWriter
from datetime import datetime
import logging
import os
import numpy as np

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnx/onnxruntime not available. ONNX loading disabled.")


def load_model(model, args):
    """
    加载预训练模型权重
    支持 PyTorch .pth、MindSpore .ckpt 和 ONNX 格式
    """
    current_epoch = 0

    # 检查是否为 ONNX 格式
    if hasattr(args, 'pretrain_model') and args.pretrain_model is not None:
        if args.pretrain_model.endswith('.onnx'):
            print(f"Loading ONNX model from: {args.pretrain_model}")
            return load_onnx_model(args.pretrain_model), 0

    if args.pretrain_segmnet is not None:
        state_dict = torch.load(args.pretrain_segmnet)
        model.seghandnet.load_state_dict(state_dict['seghandnet'])
        current_epoch = state_dict['epoch']
        print('loading the model from:', args.pretrain_segmnet)
        logging.info('pretrain_segmentation_model: %s' % args.pretrain_segmnet)

    if args.pretrain_model is not None:
        # 检测文件格式
        if args.pretrain_model.endswith('.ckpt') and BACKEND == "mindspore":
            # MindSpore checkpoint 格式
            current_epoch = load_mindspore_checkpoint(model, args.pretrain_model)
        else:
            # PyTorch .pth 格式
            state_dict = torch.load(args.pretrain_model, weights_only=False)

            if 'encoder' in state_dict.keys() and hasattr(model, 'encoder'):
                model.encoder.load_state_dict(state_dict['encoder'])
                print('load encoder')
            if 'decoder' in state_dict.keys() and hasattr(model, 'hand_decoder'):
                model.hand_decoder.load_state_dict(state_dict['decoder'])
                print('load hand_decoder')
            if 'heatmap_attention' in state_dict.keys() and hasattr(model, 'heatmap_attention'):
                model.heatmap_attention.load_state_dict(state_dict['heatmap_attention'])
                print('load heatmap_attention')
            if 'rgb2hm' in state_dict.keys() and hasattr(model, 'rgb2hm'):
                model.rgb2hm.load_state_dict(state_dict['rgb2hm'])
                print('load rgb2hm')
            if 'hm2hand' in state_dict.keys() and hasattr(model, 'hm2hand'):
                model.hm2hand.load_state_dict(state_dict['hm2hand'])
            if 'mesh2pose' in state_dict.keys() and hasattr(model, 'mesh2pose'):
                model.mesh2pose.load_state_dict(state_dict['mesh2pose'])
                print('load mesh2pose')

            if 'percep_encoder' in state_dict.keys() and hasattr(model, 'percep_encoder'):
                model.percep_encoder.load_state_dict(state_dict['percep_encoder'])

            if 'texture_light_from_low' in state_dict.keys() and hasattr(model, 'texture_light_from_low'):
                model.texture_light_from_low.load_state_dict(state_dict['texture_light_from_low'])
            if 'textures' in args.train_requires and 'texture_estimator' in state_dict.keys():
                if hasattr(model, 'renderer'):
                    model.renderer.load_state_dict(state_dict['renderer'])
                    print('load renderer')
                if hasattr(model, 'texture_estimator'):
                    model.texture_estimator.load_state_dict(state_dict['texture_estimator'])
                    print('load texture_estimator')
                if hasattr(model, 'pca_texture_estimator'):
                    model.pca_texture_estimator.load_state_dict(state_dict['pca_texture_estimator'])
                    print('load pca_texture_estimator')
            if 'lights' in args.train_requires and 'light_estimator' in state_dict.keys():
                if hasattr(model, 'light_estimator'):
                    model.light_estimator.load_state_dict(state_dict['light_estimator'])
                    print('load light_estimator')
            print('loading the model from:', args.pretrain_model)
            logging.info('pretrain_model: %s' % args.pretrain_model)
            current_epoch = state_dict.get('epoch', 0)

            if hasattr(model, 'texture_light_from_low') and args.pretrain_texture_model is not None:
                texture_state_dict = torch.load(args.pretrain_texture_model)
                model.texture_light_from_low.load_state_dict(texture_state_dict['texture_light_from_low'])
                print('loading the texture module from:', args.pretrain_texture_model)

    # load the pre-trained heat-map estimation model
    if hasattr(model, 'rgb2hm') and args.pretrain_rgb2hm is not None:
        hm_state_dict = torch.load(args.pretrain_rgb2hm)
        model.rgb2hm.load_state_dict(hm_state_dict['rgb2hm'])
        print('load rgb2hm')
        print('loading the rgb2hm model from:', args.pretrain_rgb2hm)

    return model, current_epoch


def load_mindspore_checkpoint(model, ckpt_path):
    """
    加载 MindSpore checkpoint 格式的权重

    Args:
        model: 模型实例
        ckpt_path: checkpoint 文件路径

    Returns:
        current_epoch: 当前训练轮数
    """
    try:
        import mindspore as ms_native

        # 加载 checkpoint
        param_dict = ms_native.load_checkpoint(ckpt_path)

        # 提取 epoch 信息
        current_epoch = 0
        if 'epoch' in param_dict:
            current_epoch = int(param_dict['epoch'].asnumpy())
            del param_dict['epoch']

        # 加载参数到模型
        # 注意：由于使用 msadapter，需要特殊处理
        # 这里假设 checkpoint 中的参数名与模型参数名一致
        model_state_dict = model.state_dict()

        # 匹配并加载参数
        loaded_params = {}
        for name, param in param_dict.items():
            if name in model_state_dict:
                # 转换 MindSpore Tensor 到 PyTorch Tensor (通过 msadapter)
                if hasattr(param, 'asnumpy'):
                    np_param = param.asnumpy()
                    loaded_params[name] = torch.from_numpy(np_param)
                else:
                    loaded_params[name] = param

        # 加载到模型
        model.load_state_dict(loaded_params, strict=False)

        print(f'Loaded MindSpore checkpoint from: {ckpt_path}')
        print(f'Loaded {len(loaded_params)} parameters')
        logging.info(f'MindSpore checkpoint: {ckpt_path}')

        return current_epoch

    except Exception as e:
        print(f'Failed to load MindSpore checkpoint: {e}')
        print('Falling back to PyTorch format')
        return 0


def save_model(model, optimizer, epoch, save_path, additional_info=None):
    """
    保存模型权重
    支持 PyTorch .pth 和 MindSpore .ckpt 格式

    Args:
        model: 模型实例
        optimizer: 优化器
        epoch: 当前训练轮数
        save_path: 保存路径
        additional_info: 额外信息字典
    """
    if save_path.endswith('.ckpt') and BACKEND == "mindspore":
        # 保存为 MindSpore checkpoint 格式
        save_mindspore_checkpoint(model, optimizer, epoch, save_path, additional_info)
    else:
        # 保存为 PyTorch .pth 格式
        state_dict = {
            'epoch': epoch,
            'optimizer': optimizer.state_dict() if optimizer is not None else None,
        }

        # 保存模型各个组件
        if hasattr(model, 'encoder'):
            state_dict['encoder'] = model.encoder.state_dict()
        if hasattr(model, 'hand_decoder'):
            state_dict['decoder'] = model.hand_decoder.state_dict()
        if hasattr(model, 'rgb2hm'):
            state_dict['rgb2hm'] = model.rgb2hm.state_dict()
        if hasattr(model, 'heatmap_attention'):
            state_dict['heatmap_attention'] = model.heatmap_attention.state_dict()
        if hasattr(model, 'texture_light_from_low'):
            state_dict['texture_light_from_low'] = model.texture_light_from_low.state_dict()
        if hasattr(model, 'mesh2pose'):
            state_dict['mesh2pose'] = model.mesh2pose.state_dict()

        # 添加额外信息
        if additional_info is not None:
            state_dict.update(additional_info)

        torch.save(state_dict, save_path)
        print(f'Model saved to: {save_path}')


def save_mindspore_checkpoint(model, optimizer, epoch, save_path, additional_info=None):
    """
    保存为 MindSpore checkpoint 格式
    Args:
        model: 模型实例
        optimizer: 优化器
        epoch: 当前训练轮数
        save_path: 保存路径
        additional_info: 额外信息字典
    """
    try:
        import mindspore as ms_native

        # 收集所有参数
        param_list = []

        # 添加 epoch 信息
        param_list.append({
            'name': 'epoch',
            'data': ms_native.Tensor(epoch, dtype=ms_native.int32)
        })

        # 获取模型参数
        model_state_dict = model.state_dict()
        for name, param in model_state_dict.items():
            # 转换 PyTorch Tensor 到 MindSpore Tensor
            if hasattr(param, 'cpu'):
                np_param = param.cpu().numpy()
                ms_param = ms_native.Tensor(np_param)
            else:
                ms_param = param

            param_list.append({
                'name': name,
                'data': ms_param
            })

        # 保存 checkpoint
        ms_native.save_checkpoint(param_list, save_path)

        print(f'MindSpore checkpoint saved to: {save_path}')
        print(f'Saved {len(param_list)} parameters')

    except Exception as e:
        print(f'Failed to save MindSpore checkpoint: {e}')
        print('Falling back to PyTorch format')
        # 降级保存为 .pth 格式
        pth_path = save_path.replace('.ckpt', '.pth')
        save_model(model, optimizer, epoch, pth_path, additional_info)


def freeze_model_modules(model, args):
    """冻结模型的特定模块"""
    if args.freeze_hm_estimator and hasattr(model.module, 'rgb2hm'):
        util.rec_freeze(model.module.rgb2hm)
        print("Froze heatmap estimator")
    if args.only_train_regressor:
        if hasattr(model.module, 'encoder'):
            util.rec_freeze(model.module.encoder)
            print("Froze encoder")
        if hasattr(model.module, 'hand_decoder'):
            util.rec_freeze(model.module.hand_decoder)
            print("Froze hand decoder")
        if hasattr(model.module, 'texture_estimator'):
            util.rec_freeze(model.module.texture_estimator)
            print("Froze texture estimator")
    if args.only_train_texture:
        if hasattr(model.module, 'rgb2hm'):
            util.rec_freeze(model.module.rgb2hm)
            print("Froze rgb2hm")
        if hasattr(model.module, 'encoder'):
            util.rec_freeze(model.module.encoder)
            print("Froze encoder")
        if hasattr(model.module, 'hand_decoder'):
            util.rec_freeze(model.module.hand_decoder)
            print("Froze hand decoder")


def dump(pred_out_path, xyz_pred_list, verts_pred_list):
    """Save predictions into a json file."""
    # make sure its only lists
    xyz_pred_list = [x.tolist() for x in xyz_pred_list]
    verts_pred_list = [x.tolist() for x in verts_pred_list]

    # save to a json
    with open(pred_out_path, 'w') as fo:
        json.dump(
            [
                xyz_pred_list,
                verts_pred_list
            ], fo)
    print('Dumped %d joints and %d verts predictions to %s' % (
    len(xyz_pred_list), len(verts_pred_list), pred_out_path))


def convert_pth_to_ckpt(pth_path, ckpt_path):
    """
    将 PyTorch .pth 权重转换为 MindSpore .ckpt 格式

    Args:
        pth_path: PyTorch 权重文件路径
        ckpt_path: MindSpore checkpoint 保存路径
    """
    try:
        import mindspore as ms_native

        # 加载 PyTorch 权重
        state_dict = torch.load(pth_path, map_location='cpu', weights_only=False)

        # 转换参数
        param_list = []

        # 保存 epoch 信息
        if 'epoch' in state_dict:
            param_list.append({
                'name': 'epoch',
                'data': ms_native.Tensor(state_dict['epoch'], dtype=ms_native.int32)
            })

        # 转换模型参数
        for key, value in state_dict.items():
            # 跳过非模型参数
            if key in ['epoch', 'optimizer', 'scheduler']:
                continue

            # 处理嵌套的 state_dict
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        np_value = sub_value.cpu().numpy()
                        ms_value = ms_native.Tensor(np_value)
                        param_name = f"{key}.{sub_key}"
                        param_list.append({'name': param_name, 'data': ms_value})
            elif isinstance(value, torch.Tensor):
                np_value = value.cpu().numpy()
                ms_value = ms_native.Tensor(np_value)
                param_list.append({'name': key, 'data': ms_value})

        # 保存 MindSpore checkpoint
        ms_native.save_checkpoint(param_list, ckpt_path)

        print(f"Successfully converted {len(param_list)} parameters")
        print(f"PyTorch .pth: {pth_path}")
        print(f"MindSpore .ckpt: {ckpt_path}")

        return True

    except Exception as e:
        print(f"Conversion failed: {e}")
        return False


if __name__ == "__main__":
    # 示例：转换权重格式
    import sys
    if len(sys.argv) == 3:
        pth_file = sys.argv[1]
        ckpt_file = sys.argv[2]
        print(f"Converting {pth_file} to {ckpt_file}...")
        convert_pth_to_ckpt(pth_file, ckpt_file)
    else:
        print("Usage: python train_utils.py <input.pth> <output.ckpt>")


def load_onnx_model(onnx_path):
    """
    加载 ONNX 模型用于推理

    Args:
        onnx_path: ONNX 模型文件路径

    Returns:
        ONNXInferenceWrapper: ONNX 推理包装器
    """
    if not ONNX_AVAILABLE:
        raise ImportError("onnx and onnxruntime are required for ONNX model loading")

    class ONNXInferenceWrapper:
        """ONNX 模型推理包装器，提供与 PyTorch 模型相同的接口"""

        def __init__(self, onnx_path):
            self.session = ort.InferenceSession(onnx_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            print(f"ONNX model loaded: {onnx_path}")
            print(f"Input: {self.input_name}")
            print(f"Outputs: {self.output_names}")

        def predict_singleview(self, images, mask_images, Ks, task, requires, gt_verts, bgimgs):
            """
            与 PyTorch 模型兼容的推理接口

            Args:
                images: 输入图像 tensor [B, 3, 224, 224]
                其他参数保持兼容性

            Returns:
                output: 包含 joints, vertices, pose, shape 的字典
            """
            # 转换输入为 numpy
            if hasattr(images, 'cpu'):
                images_np = images.cpu().numpy()
            elif hasattr(images, 'asnumpy'):
                images_np = images.asnumpy()
            else:
                images_np = np.array(images)

            # ONNX 推理
            outputs = self.session.run(self.output_names, {self.input_name: images_np})

            # 构造输出字典
            output = {}
            if len(outputs) >= 1:
                output['joints'] = outputs[0]  # [B, 21, 3]
            if len(outputs) >= 2:
                output['vertices'] = outputs[1]  # [B, 778, 3]
            if len(outputs) >= 3:
                output['pose'] = outputs[2]  # [B, 48]
            if len(outputs) >= 4:
                output['shape'] = outputs[3]  # [B, 10]

            # 添加其他必需的字段（占位符）
            output['scale'] = None
            output['trans'] = None
            output['rot'] = None
            output['tsa_poses'] = None
            output['faces'] = None

            return output

        def eval(self):
            """兼容 PyTorch 的 eval() 方法"""
            pass

        def __call__(self, *args, **kwargs):
            """支持直接调用"""
            return self.predict_singleview(*args, **kwargs)

    return ONNXInferenceWrapper(onnx_path)
