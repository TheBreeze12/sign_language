import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Constant, Normal, initializer

from utils.hand_3d_model import rot_pose_beta_to_mesh
from utils.net_hg import Residual


def normal_init(module, mean=0.0, std=1.0, bias=0.0):
    if hasattr(module, "weight") and module.weight is not None:
        module.weight.set_data(initializer(Normal(sigma=std, mean=mean), module.weight.shape, module.weight.dtype))
    if hasattr(module, "bias") and module.bias is not None:
        module.bias.set_data(initializer(Constant(bias), module.bias.shape, module.bias.dtype))


class PoseLiftNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.net = nn.SequentialCell(
            nn.Dense(42, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Dense(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Dense(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Dense(256, 63),
        )

    def construct(self, j2d):
        batch_size = j2d.shape[0]
        j2d = ops.reshape(j2d, (batch_size, -1))
        j3d = self.net(j2d)
        j3d = ops.reshape(j3d, (batch_size, -1, 3))
        return j3d


class mesh2poseNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.net = nn.SequentialCell(
            nn.Dense(778 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Dense(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Dense(128, 63),
        )
        self.init_weights()

    def init_weights(self):
        normal_init(self.net[0], std=0.0001)
        normal_init(self.net[4], std=0.0001)
        normal_init(self.net[8], std=0.0001)

    def construct(self, mesh_coords):
        batch_size = mesh_coords.shape[0]
        mesh_coords = ops.reshape(mesh_coords, (batch_size, -1))
        j3d = self.net(mesh_coords)
        j3d = ops.reshape(j3d, (batch_size, -1, 3))
        return j3d


class MyPoseHand(nn.Cell):
    def __init__(self, ncomps=6, inp_neurons=1536, use_pca=True, dropout=0, use_mean_shape=False):
        super().__init__()
        self.use_mean_shape = use_mean_shape

        self.base_layers = nn.SequentialCell(
            nn.Dense(inp_neurons, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dense(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.pose_reg = nn.SequentialCell(nn.Dense(512, 128), nn.ReLU(), nn.Dense(128, 30))
        self.shape_reg = nn.SequentialCell(nn.Dense(512, 128), nn.ReLU(), nn.Dense(128, 10))
        self.trans_reg = nn.SequentialCell(nn.Dense(512, 128), nn.ReLU(), nn.Dense(128, 32), nn.Dense(32, 3))
        self.rot_reg = nn.SequentialCell(nn.Dense(512, 128), nn.ReLU(), nn.Dense(128, 32), nn.Dense(32, 3))
        self.scale_reg = nn.SequentialCell(nn.Dense(512, 128), nn.ReLU(), nn.Dense(128, 32), nn.Dense(32, 1))

        self.init_weights()

    def init_weights(self):
        normal_init(self.scale_reg[0], std=0.001)
        normal_init(self.scale_reg[2], std=0.001)
        normal_init(self.scale_reg[3], std=0.001, bias=0.95)

        normal_init(self.trans_reg[0], std=0.001)
        normal_init(self.trans_reg[2], std=0.001)
        normal_init(self.trans_reg[3], std=0.001)
        bias_np = np.zeros((3,), dtype=np.float32)
        bias_np[2] = 0.65
        self.trans_reg[3].bias.set_data(ms.Tensor(bias_np, ms.float32))

    def construct(self, features):
        base_features = self.base_layers(features)
        theta = self.pose_reg(base_features)
        beta = self.shape_reg(base_features)
        scale = self.scale_reg(base_features)
        trans = self.trans_reg(base_features)
        rot = self.rot_reg(base_features)

        if self.use_mean_shape:
            beta = ops.zeros_like(beta)

        jv, faces, tsa_poses = rot_pose_beta_to_mesh(rot, theta, beta)
        jv_ts = ops.expand_dims(trans, 1) + ops.abs(ops.expand_dims(scale, 2)) * jv
        joints = jv_ts[:, 0:21]
        verts = jv_ts[:, 21:]
        return joints, verts, faces, theta, beta, scale, trans, rot, tsa_poses


class PoseHand(nn.Cell):
    def __init__(self, inp_neurons=1536, use_mean_shape=False, trans_dim=2):
        super().__init__()
        self.use_mean_shape = use_mean_shape

        self.base_layers = nn.SequentialCell(
            nn.Dense(inp_neurons, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dense(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.pose_reg = nn.SequentialCell(nn.Dense(512, 128), nn.ReLU(), nn.Dense(128, 30))
        self.shape_reg = nn.SequentialCell(nn.Dense(512, 128), nn.ReLU(), nn.Dense(128, 10))
        self.trans_reg = nn.SequentialCell(nn.Dense(512, 128), nn.ReLU(), nn.Dense(128, 32), nn.Dense(32, trans_dim))
        self.rot_reg = nn.SequentialCell(nn.Dense(512, 128), nn.ReLU(), nn.Dense(128, 32), nn.Dense(32, 3))
        self.scale_reg = nn.SequentialCell(nn.Dense(512, 128), nn.ReLU(), nn.Dense(128, 32), nn.Dense(32, 1))

        self.init_weights()

    def init_weights(self):
        normal_init(self.scale_reg[0], std=0.001)
        normal_init(self.scale_reg[2], std=0.001)
        normal_init(self.scale_reg[3], std=0.001, bias=1.0)

        normal_init(self.trans_reg[0], std=0.001)
        normal_init(self.trans_reg[2], std=0.001)
        normal_init(self.trans_reg[3], std=0.001)
        self.trans_reg[3].bias.set_data(initializer(Constant(0.0), self.trans_reg[3].bias.shape, self.trans_reg[3].bias.dtype))

    def construct(self, features):
        base_features = self.base_layers(features)
        theta = self.pose_reg(base_features)
        beta = self.shape_reg(base_features)
        scale = self.scale_reg(base_features)
        trans = self.trans_reg(base_features)
        rot = self.rot_reg(base_features)

        if self.use_mean_shape:
            beta = ops.zeros_like(beta)

        jv, faces, tsa_poses = rot_pose_beta_to_mesh(rot, theta, beta)
        verts = jv[:, 21:]
        joints = jv[:, 0:21]
        return joints, verts, faces, theta, beta, scale, trans, rot, tsa_poses


class Net_HM_Feat(nn.Cell):
    def __init__(self, num_heatmap_chan, num_feat_chan, size_input_feature=(64, 64)):
        super().__init__()
        self.num_heatmap_chan = num_heatmap_chan
        self.num_feat_chan = num_feat_chan
        self.size_input_feature = size_input_feature
        self.nRegBlock = 4
        self.nRegModules = 2

        self.heatmap_conv = nn.Conv2d(self.num_heatmap_chan, self.num_feat_chan, kernel_size=1, stride=1, has_bias=True)
        self.encoding_conv = nn.Conv2d(self.num_feat_chan, self.num_feat_chan, kernel_size=1, stride=1, has_bias=True)
        reg_cells = []
        for _ in range(self.nRegBlock):
            for _ in range(self.nRegModules):
                reg_cells.append(Residual(self.num_feat_chan, self.num_feat_chan))
        self.reg_ = nn.CellList(reg_cells)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downsample_scale = 2 ** self.nRegBlock

        self.num_feat_out = self.num_feat_chan * (
            size_input_feature[0] * size_input_feature[1] // (self.downsample_scale ** 2)
        )

    def construct(self, hm_list, encoding_list):
        x = self.heatmap_conv(hm_list[-1]) + self.encoding_conv(encoding_list[-1])
        if len(encoding_list) > 1:
            x = x + encoding_list[-2]

        for i in range(self.nRegBlock):
            for j in range(self.nRegModules):
                x = self.reg_[i * self.nRegModules + j](x)
            x = self.maxpool(x)
        return ops.reshape(x, (x.shape[0], -1))


class HM2Mano(nn.Cell):
    def __init__(self, num_heatmap_chan, num_feat_chan, num_mesh_output_chan, graph_L, size_input_feature=(64, 64)):
        super().__init__()
        self.feat_net = Net_HM_Feat(num_heatmap_chan, num_feat_chan, size_input_feature)
        self.hand_decoder = MyPoseHand(inp_neurons=4096)

    def construct(self, hm_list, encoding_list):
        feat = self.feat_net(hm_list, encoding_list)
        joints, verts, faces, theta, beta, _, _, _, _ = self.hand_decoder(feat)
        return joints, verts, faces, theta, beta, feat
