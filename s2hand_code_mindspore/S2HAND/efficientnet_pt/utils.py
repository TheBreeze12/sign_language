"""
MindSpore helper functions for EfficientNet.
"""

import collections
import math
import re

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


GlobalParams = collections.namedtuple(
    "GlobalParams",
    [
        "batch_norm_momentum",
        "batch_norm_epsilon",
        "dropout_rate",
        "num_classes",
        "width_coefficient",
        "depth_coefficient",
        "depth_divisor",
        "min_depth",
        "drop_connect_rate",
        "image_size",
    ],
)

BlockArgs = collections.namedtuple(
    "BlockArgs",
    [
        "kernel_size",
        "num_repeat",
        "input_filters",
        "output_filters",
        "expand_ratio",
        "id_skip",
        "stride",
        "se_ratio",
    ],
)

GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


class MemoryEfficientSwish(nn.Cell):
    def construct(self, x):
        return x * ops.sigmoid(x)


class Swish(nn.Cell):
    def construct(self, x):
        return x * ops.sigmoid(x)


def round_filters(filters, global_params):
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    if (not training) or (p <= 0.0):
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1.0 - p
    random_tensor = keep_prob + ops.rand((batch_size, 1, 1, 1), dtype=inputs.dtype)
    binary_tensor = ops.floor(random_tensor)
    return inputs / keep_prob * binary_tensor


class Conv2dDynamicSamePadding(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode="same",
            group=groups,
            has_bias=bias,
        )

    def construct(self, x):
        return self.conv(x)


class Conv2dStaticSamePadding(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__()
        stride = kwargs.get("stride", 1)
        groups = kwargs.get("groups", 1)
        bias = kwargs.get("bias", True)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode="same",
            group=groups,
            has_bias=bias,
        )

    def construct(self, x):
        return self.conv(x)


class Identity(nn.Cell):
    def construct(self, x):
        return x


def get_same_padding_conv2d(image_size=None):
    del image_size

    def _conv(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        del dilation
        return Conv2dDynamicSamePadding(in_channels, out_channels, kernel_size, stride=stride, groups=groups, bias=bias)

    return _conv


def efficientnet_params(model_name):
    params_dict = {
        "efficientnet-b0": (1.0, 1.0, 224, 0.2),
        "efficientnet-b1": (1.0, 1.1, 240, 0.2),
        "efficientnet-b2": (1.1, 1.2, 260, 0.3),
        "efficientnet-b3": (1.2, 1.4, 300, 0.3),
        "efficientnet-b4": (1.4, 1.8, 380, 0.4),
        "efficientnet-b5": (1.6, 2.2, 456, 0.4),
        "efficientnet-b6": (1.8, 2.6, 528, 0.5),
        "efficientnet-b7": (2.0, 3.1, 600, 0.5),
    }
    return params_dict[model_name]


class BlockDecoder(object):
    @staticmethod
    def _decode_block_string(block_string):
        assert isinstance(block_string, str)
        ops_str = block_string.split("_")
        options = {}
        for op in ops_str:
            splits = re.split(r"(\d.*)", op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        assert (("s" in options and len(options["s"]) == 1) or (len(options["s"]) == 2 and options["s"][0] == options["s"][1]))

        return BlockArgs(
            kernel_size=int(options["k"]),
            num_repeat=int(options["r"]),
            input_filters=int(options["i"]),
            output_filters=int(options["o"]),
            expand_ratio=int(options["e"]),
            id_skip=("noskip" not in block_string),
            se_ratio=float(options["se"]) if "se" in options else None,
            stride=[int(options["s"][0])],
        )

    @staticmethod
    def decode(string_list):
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args


def efficientnet(
    width_coefficient=None,
    depth_coefficient=None,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    image_size=None,
    num_classes=1000,
):
    blocks_args = [
        "r1_k3_s11_e1_i32_o16_se0.25",
        "r2_k3_s22_e6_i16_o24_se0.25",
        "r2_k5_s22_e6_i24_o40_se0.25",
        "r3_k3_s22_e6_i40_o80_se0.25",
        "r3_k5_s11_e6_i80_o112_se0.25",
        "r4_k5_s22_e6_i112_o192_se0.25",
        "r1_k3_s11_e6_i192_o320_se0.25",
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        num_classes=num_classes,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        image_size=image_size,
    )
    return blocks_args, global_params


def get_model_params(model_name, override_params):
    if model_name.startswith("efficientnet"):
        w, d, s, p = efficientnet_params(model_name)
        blocks_args, global_params = efficientnet(
            width_coefficient=w,
            depth_coefficient=d,
            dropout_rate=p,
            image_size=s,
        )
    else:
        raise NotImplementedError("model name is not pre-defined: %s" % model_name)

    if override_params:
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


def load_pretrained_weights(model, model_name, load_fc=True):
    del model, model_name, load_fc
    raise NotImplementedError("MindSpore path does not support loading PyTorch pretrained weights directly.")
