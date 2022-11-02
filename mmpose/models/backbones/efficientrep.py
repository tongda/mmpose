# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

import torch
import torch.nn as nn

from mmpose.registry import MODELS
from ..utils.repvggblock import RepBlock, RepVGGBlock
from .base_backbone import BaseBackbone


class Conv(nn.Module):
    """Normal Conv with SiLU activation."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class SimConv(nn.Module):
    """Normal Conv with ReLU activation."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class SimSPPF(nn.Module):
    """Simplified SPPF with ReLU activation."""

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = SimConv(in_channels, c_, 1, 1)
        self.cv2 = SimConv(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


def make_divisible(x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor


@MODELS.register_module()
class EfficientRep(BaseBackbone):
    """EfficientRep Backbone EfficientRep is handcrafted by hardware-aware
    neural network design.

    With rep-style struct, EfficientRep is friendly to high-computation
    hardware(e.g. GPU).
    """

    def __init__(
        self,
        in_channels=3,
        depth_mul=0.33,
        width_mul=0.25,
    ):
        super().__init__()

        num_repeat_neck = [12, 12, 12, 12]
        channels_list_neck = [256, 128, 128, 256, 256, 512]
        num_repeat_backbone = [1, 6, 12, 18, 6]
        channels_list_backbone = [64, 128, 256, 512, 1024]

        num_repeats = [(max(round(i * depth_mul), 1) if i > 1 else i)
                       for i in (num_repeat_backbone + num_repeat_neck)]
        channels_list = [
            make_divisible(i * width_mul, 8)
            for i in (channels_list_backbone + channels_list_neck)
        ]

        self.stem = RepVGGBlock(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2)

        self.ERBlock_2 = nn.Sequential(
            RepVGGBlock(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2),
            RepBlock(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1]))

        self.ERBlock_3 = nn.Sequential(
            RepVGGBlock(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2),
            RepBlock(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2]))

        self.ERBlock_4 = nn.Sequential(
            RepVGGBlock(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2),
            RepBlock(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3]))

        self.ERBlock_5 = nn.Sequential(
            RepVGGBlock(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2,
            ),
            RepBlock(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4]),
            SimSPPF(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                kernel_size=5))

    def forward(self, x):

        x = self.stem(x)
        x = self.ERBlock_2(x)
        x = self.ERBlock_3(x)
        # outputs.append(x)
        x = self.ERBlock_4(x)
        # outputs.append(x)
        x = self.ERBlock_5(x)
        # outputs.append(x)

        # return tuple(outputs)
        return [x]
