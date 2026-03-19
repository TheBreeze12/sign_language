"""
Pure MindSpore implementation of S2HAND main model graph.
"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from utils.freihandnet import HM2Mano, MyPoseHand, mesh2poseNet, normal_init
from utils.net_hg import Net_HM_HG
from efficientnet_pt.model import EfficientNet


def normalize_image(im):
    return im - 0.5


def compute_uv_from_integral_ms(hm, resize_dim):
    b, k, h, w = hm.shape
    hm_up = ops.interpolate(hm, size=resize_dim, mode="bilinear", align_corners=True)
    flat = ops.reshape(hm_up, (b, k, -1))
    idx = ops.argmax(flat, axis=2)
    idx_f = ops.cast(idx, ms.float32)
    width = ms.Tensor(float(resize_dim[1]), ms.float32)
    x = ops.floor_mod(idx_f, width)
    y = ops.floor(idx_f / width)
    conf = ops.reduce_max(flat, axis=2)
    return ops.stack((x, y, conf), axis=2)


class Encoder(nn.Cell):
    def __init__(self, version="b3"):
        super().__init__()
        self.version = version
        if self.version == "b3":
            self.encoder = EfficientNet.from_name("efficientnet-b3")
            self.pool = nn.AvgPool2d(kernel_size=7, stride=1)
        else:
            raise ValueError("Only efficientnet-b3 is supported in this implementation.")

    def construct(self, x):
        features, low_features = self.encoder.extract_features(x)
        features = self.pool(features)
        features = ops.reshape(features, (features.shape[0], -1))
        return features, low_features


class RGB2HM(nn.Cell):
    def __init__(self):
        super().__init__()
        num_joints = 21
        self.net_hm = Net_HM_HG(num_joints, num_stages=2, num_modules=2, num_feats=256)

    def construct(self, images):
        images = normalize_image(images)
        est_hm_list, encoding = self.net_hm(images)
        return est_hm_list, encoding


class MyHandDecoder(nn.Cell):
    def __init__(self, inp_neurons=1536, use_mean_shape=False):
        super().__init__()
        self.hand_decode = MyPoseHand(inp_neurons=inp_neurons, use_mean_shape=use_mean_shape)

    def construct(self, features):
        return self.hand_decode(features)


class light_estimator(nn.Cell):
    def __init__(self, dim_in=1536):
        super().__init__()
        self.fc1 = nn.Dense(dim_in, 256)
        self.fc2 = nn.Dense(256, 11)

    def construct(self, x):
        x = ops.relu(self.fc1(x))
        lights = ops.sigmoid(self.fc2(x))
        return lights


class texture_light_estimator(nn.Cell):
    def __init__(self, num_channel=32, dim_in=56, mode="surf"):
        super().__init__()
        del num_channel, dim_in
        self.base_layers = nn.SequentialCell(
            nn.Conv2d(32, 48, kernel_size=10, stride=4, padding=1, pad_mode="pad", has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 64, kernel_size=3, has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.texture_reg = nn.SequentialCell(nn.Dense(256, 64), nn.ReLU(), nn.Dense(64, 1538 * 3))
        self.light_reg = nn.SequentialCell(nn.Dense(256, 64), nn.ReLU(), nn.Dense(64, 11))
        self.mode = mode
        self.texture_mean = ms.Tensor([200 / 256, 150 / 256, 150 / 256], ms.float32)

        normal_init(self.texture_reg[0], std=0.001)
        normal_init(self.texture_reg[2], std=0.001)
        normal_init(self.light_reg[0], std=0.001)
        normal_init(self.light_reg[2], mean=1.0, std=0.001)

    def construct(self, low_features):
        base_features = self.base_layers(low_features)
        base_features = ops.reshape(base_features, (base_features.shape[0], -1))

        bias = self.texture_reg(base_features)
        mean_t = self.texture_mean
        if self.mode == "surf":
            bias = ops.reshape(bias, (-1, 1538, 3))
            mean_t = ops.tile(ops.expand_dims(ops.expand_dims(mean_t, 0), 0), (1, bias.shape[1], 1))
        textures = mean_t + bias

        lights = self.light_reg(base_features)
        return textures, lights


class heatmap_attention(nn.Cell):
    def __init__(self, num_channel=256, dim_in=64, out_len=1536, mode="surf"):
        super().__init__()
        del dim_in, mode
        self.base_layers = nn.SequentialCell(
            nn.BatchNorm2d(num_channel),
            nn.Conv2d(num_channel, 64, kernel_size=10, stride=7, padding=1, pad_mode="pad", has_bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=3, padding=1, pad_mode="pad", has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
        )
        self.reg = nn.SequentialCell(nn.Dense(64, out_len))

    def construct(self, x):
        x0 = self.base_layers(x)
        x0 = ops.reshape(x0, (x.shape[0], -1))
        return self.reg(x0)


class Model(nn.Cell):
    def __init__(self, filename_obj, args):
        super().__init__()
        del filename_obj

        self.rgb2hm = None
        self.encoder = None
        self.hand_decoder = None
        self.heatmap_attention = None
        self.texture_light_from_low = None
        self.hm2hand = None
        self.mesh2pose = None

        if "heatmaps" in args.train_requires or "heatmaps" in args.test_requires:
            self.rgb2hm = RGB2HM()

        if (
            "joints" in args.train_requires
            or "verts" in args.train_requires
            or "joints" in args.test_requires
            or "verts" in args.test_requires
        ):
            self.regress_mode = args.regress_mode
            self.use_mean_shape = args.use_mean_shape
            self.use_2d_as_attention = args.use_2d_as_attention

            if self.regress_mode == "mano":
                self.encoder = Encoder()
                self.dim_in = 1536
                self.hand_decoder = MyHandDecoder(inp_neurons=self.dim_in, use_mean_shape=self.use_mean_shape)
                if self.use_2d_as_attention:
                    self.heatmap_attention = heatmap_attention(out_len=self.dim_in)
            elif self.regress_mode == "hm2mano":
                self.hm2hand = HM2Mano(21, 256, 0, None)

            self.render_choice = args.renderer_mode
            self.texture_choice = args.texture_mode

            if self.render_choice == "NR":
                self.texture_light_from_low = texture_light_estimator(mode="surf")

            self.use_pose_regressor = args.use_pose_regressor
            if (args.train_datasets)[0] == "FreiHand":
                self.get_gt_depth = True
                self.dataset = "FreiHand"
            elif (args.train_datasets)[0] == "RHD":
                self.get_gt_depth = False
                self.dataset = "RHD"
                if self.use_pose_regressor:
                    self.mesh2pose = mesh2poseNet()
            elif (args.train_datasets)[0] == "Obman":
                self.get_gt_depth = False
                self.dataset = "Obman"
            elif (args.train_datasets)[0] == "HO3D":
                self.get_gt_depth = True
                self.dataset = "HO3D"
            else:
                self.get_gt_depth = False
                self.dataset = "Unknown"
        else:
            self.regress_mode = None

    def predict_singleview(self, images, mask_images, ks, task, requires, gt_verts, bgimgs):
        del mask_images, ks, gt_verts, bgimgs
        output = {}

        est_hm_list = None
        encoding = None

        if self.regress_mode == "hm2mano" or task == "hm_train" or "heatmaps" in requires:
            images_this = images
            if images_this.shape[3] != 256:
                images_this = ops.pad(images_this, ((0, 0), (0, 0), (0, 32), (0, 32)))
            est_hm_list, encoding = self.rgb2hm(images_this)

            est_pose_uv_list = []
            for est_hm in est_hm_list:
                est_pose_uv = compute_uv_from_integral_ms(est_hm, images_this.shape[2:4])
                est_pose_uv_list.append(est_pose_uv)

            output["hm_list"] = est_hm_list
            output["hm_pose_uv_list"] = est_pose_uv_list
            output["hm_j2d_list"] = [hm_pose_uv[:, :, :2] for hm_pose_uv in est_pose_uv_list]

        if task == "hm_train":
            return output

        if self.regress_mode == "hm2mano":
            joints, vertices, faces, pose, shape, features = self.hm2hand(est_hm_list, encoding)
            scale = None
            trans = None
            rot = None
            tsa_poses = None
            low_features = None
        elif self.regress_mode in ("mano", "mano1"):
            features, low_features = self.encoder(images)

            if self.use_2d_as_attention and encoding is not None:
                attention_2d = self.heatmap_attention(encoding[-1])
                features = features * attention_2d

            joints, vertices, faces, pose, shape, scale, trans, rot, tsa_poses = self.hand_decoder(features)
            if self.dataset == "RHD" and self.use_pose_regressor and self.mesh2pose is not None:
                joints_res = self.mesh2pose(vertices)
                joints = joints + joints_res
        else:
            return output

        output["joints"] = joints
        output["vertices"] = vertices
        output["pose"] = pose
        output["shape"] = shape
        output["scale"] = scale
        output["trans"] = trans
        output["rot"] = rot
        output["tsa_poses"] = tsa_poses

        if self.texture_light_from_low is not None and low_features is not None:
            textures, lights = self.texture_light_from_low(low_features)
            output["textures"] = textures
            output["lights"] = lights

        output["faces"] = ops.cast(faces, ms.int32)
        return output

    def construct(
        self,
        images=None,
        mask_images=None,
        viewpoints=None,
        p=None,
        voxels=None,
        mano_para=None,
        task="train",
        requires=("joints",),
        gt_verts=None,
        gt_2d_joints=None,
        bgimgs=None,
    ):
        del viewpoints, voxels, mano_para, gt_2d_joints
        if task in ("train", "hm_train"):
            return self.predict_singleview(images, mask_images, p, task, requires, gt_verts, bgimgs)
        if task == "stacked_train":
            return self.predict_singleview(images, mask_images, p, task, requires, gt_verts, bgimgs)
        return self.predict_singleview(images, mask_images, p, task, requires, gt_verts, bgimgs)
