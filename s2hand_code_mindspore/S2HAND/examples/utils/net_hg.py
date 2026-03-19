# Copyright (c) Liuhao Ge. All Rights Reserved.
"""
MindSpore implementation of stacked hourglass for hand heatmap estimation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import mindspore.nn as nn
import mindspore.ops as ops


class Residual(nn.Cell):
    def __init__(self, numIn, numOut):
        super().__init__()
        self.numIn = numIn
        self.numOut = numOut
        self.bn = nn.BatchNorm2d(self.numIn)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(self.numIn, self.numOut // 2, kernel_size=1, has_bias=True)
        self.bn1 = nn.BatchNorm2d(self.numOut // 2)
        self.conv2 = nn.Conv2d(
            self.numOut // 2,
            self.numOut // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            pad_mode="pad",
            has_bias=True,
        )
        self.bn2 = nn.BatchNorm2d(self.numOut // 2)
        self.conv3 = nn.Conv2d(self.numOut // 2, self.numOut, kernel_size=1, has_bias=True)

        if self.numIn != self.numOut:
            self.conv4 = nn.Conv2d(self.numIn, self.numOut, kernel_size=1, has_bias=True)
        else:
            self.conv4 = None

    def construct(self, x):
        residual = x
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.conv4 is not None:
            residual = self.conv4(x)

        return out + residual


class Hourglass(nn.Cell):
    def __init__(self, n, nModules, nFeats):
        super().__init__()
        self.n = n
        self.nModules = nModules
        self.nFeats = nFeats

        up1_cells = [Residual(self.nFeats, self.nFeats) for _ in range(self.nModules)]
        low1_cells = [Residual(self.nFeats, self.nFeats) for _ in range(self.nModules)]
        low3_cells = [Residual(self.nFeats, self.nFeats) for _ in range(self.nModules)]

        self.low1 = nn.MaxPool2d(kernel_size=2, stride=2)

        if self.n > 1:
            self.low2 = Hourglass(n - 1, self.nModules, self.nFeats)
            self.low2_ = None
        else:
            self.low2_ = nn.CellList([Residual(self.nFeats, self.nFeats) for _ in range(self.nModules)])
            self.low2 = None

        self.up1_ = nn.CellList(up1_cells)
        self.low1_ = nn.CellList(low1_cells)
        self.low3_ = nn.CellList(low3_cells)

    def construct(self, x):
        up1 = x
        for j in range(self.nModules):
            up1 = self.up1_[j](up1)

        low1 = self.low1(x)
        for j in range(self.nModules):
            low1 = self.low1_[j](low1)

        if self.n > 1:
            low2 = self.low2(low1)
        else:
            low2 = low1
            for j in range(self.nModules):
                low2 = self.low2_[j](low2)

        low3 = low2
        for j in range(self.nModules):
            low3 = self.low3_[j](low3)

        up2 = ops.interpolate(low3, scale_factor=2.0, mode="nearest")
        return up1 + up2


class Net_HM_HG(nn.Cell):
    def __init__(self, num_joints, num_stages=2, num_modules=2, num_feats=256):
        super().__init__()

        self.numOutput = num_joints
        self.nStack = num_stages

        self.nModules = num_modules
        self.nFeats = num_feats

        self.conv1_ = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, pad_mode="pad", has_bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.r1 = Residual(64, 128)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.r4 = Residual(128, 128)
        self.r5 = Residual(128, self.nFeats)

        hourglass_cells = []
        residual_cells = []
        lin_cells = []
        tmpout_cells = []
        ll_cells = []
        tmpout_skip_cells = []

        for i in range(self.nStack):
            hourglass_cells.append(Hourglass(4, self.nModules, self.nFeats))
            for _ in range(self.nModules):
                residual_cells.append(Residual(self.nFeats, self.nFeats))
            lin_cells.append(
                nn.SequentialCell(
                    nn.Conv2d(self.nFeats, self.nFeats, kernel_size=1, stride=1, has_bias=True),
                    nn.BatchNorm2d(self.nFeats),
                    nn.ReLU(),
                )
            )
            tmpout_cells.append(
                nn.Conv2d(self.nFeats, self.numOutput, kernel_size=1, stride=1, has_bias=True)
            )
            if i < self.nStack - 1:
                ll_cells.append(nn.Conv2d(self.nFeats, self.nFeats, kernel_size=1, stride=1, has_bias=True))
                tmpout_skip_cells.append(
                    nn.Conv2d(self.numOutput, self.nFeats, kernel_size=1, stride=1, has_bias=True)
                )

        self.hourglass = nn.CellList(hourglass_cells)
        self.Residual = nn.CellList(residual_cells)
        self.lin_ = nn.CellList(lin_cells)
        self.tmpOut = nn.CellList(tmpout_cells)
        self.ll_ = nn.CellList(ll_cells)
        self.tmpOut_ = nn.CellList(tmpout_skip_cells)

    def construct(self, x):
        x = self.conv1_(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.r1(x)
        x = self.maxpool(x)
        x = self.r4(x)
        x = self.r5(x)

        out = []
        encoding = []

        for i in range(self.nStack):
            hg = self.hourglass[i](x)
            ll = hg
            for j in range(self.nModules):
                ll = self.Residual[i * self.nModules + j](ll)
            ll = self.lin_[i](ll)
            tmp_out = self.tmpOut[i](ll)
            out.append(tmp_out)
            if i < self.nStack - 1:
                ll_ = self.ll_[i](ll)
                tmp_out_ = self.tmpOut_[i](tmp_out)
                x = x + ll_ + tmp_out_
                encoding.append(x)
            else:
                encoding.append(ll)

        return out, encoding
