import torch
from torch import nn
import numpy as np
from collections import namedtuple
from typing import Callable, Any, Optional, List


__all__ = ['mobilenet_v2']


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, hidden_dim, use_conv1=True):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        norm_layer = nn.BatchNorm2d
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if use_conv1:
            # pw
            layers.append(ConvBNActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNActivation(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes


def get_group_stride_channel_expansion_list(inverted_residual_setting):
    gblocks, gchannels, gstrides, gexpansions, ghiddens = [], [], [], [], []
    for t, c, n, s in inverted_residual_setting:
        gblocks.append(n)
        gchannels.append(c)
        gstrides.append(s)
        gexpansions.append(t)
    for i in range(1, len(gchannels)):
        ch = gchannels[i-1]
        ep = gexpansions[i]
        ghiddens.append(int(round(ch * ep)))
    return gblocks, gchannels, gstrides, gexpansions, ghiddens


def group_to_block(gblocks, gchannels, gstrides, gexpansions):
    n = len(gchannels)
    assert(n == len(gblocks))
    bchannels, bstrides, bexpansions, bhiddens = [], [], [], []
    for i in range(n):
        ch = gchannels[i]
        bk = gblocks[i]
        sr = gstrides[i]
        ep = gexpansions[i]
        for j in range(bk):
            if i > 0:
                bhiddens.append(int(round(bchannels[-1] * ep)))
            bchannels.append(ch)
            bexpansions.append(ep)
            if j == 0:
                bstrides.append(sr)
            else:
                bstrides.append(1)
    return bchannels, bstrides, bexpansions, bhiddens


def create_mobilenet_v2_from_configurations(spec_confs):

    inverted_residual_setting =  [
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]

    gblocks, gchannels, gstrides, gexpansions, _ = get_group_stride_channel_expansion_list(inverted_residual_setting)
    _, bstrides, bexpansions, _ = group_to_block(gblocks, gchannels, gstrides, gexpansions)
    bstrides = [2] + bstrides + [1]
    bexpansions = [None] + bexpansions + [None]
    num_blocks = len(bstrides)
    assert(num_blocks == 19)

    layers = []
    k = 0
    for i in range(num_blocks):
        sr, ep = bstrides[i], bexpansions[i]
        if i == 0:
            mod = ConvBNActivation(3, spec_confs[k], stride=sr)
            k += 1
        elif i == num_blocks - 1:
            mod = ConvBNActivation(spec_confs[-2], spec_confs[-1], kernel_size=1)
            k += 1
        elif i == 1: # expansion == 1
            mod = InvertedResidual(spec_confs[k-1], spec_confs[k], stride=sr, hidden_dim=spec_confs[k-1], use_conv1=False)
            k += 1
        else:
            mod = InvertedResidual(spec_confs[k-1], spec_confs[k+1], stride=sr, hidden_dim=spec_confs[k])
            k += 2
        layers.append(mod)
    
    encoder = nn.Sequential(*layers)
    for m in encoder.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return encoder


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def mobilenet_v2(**kwargs: Any):
    width = 1.0
    mnv2_base_confs = [
        32, 16, 96, 24, 144, 24, 144, 32, 192, 32, 192, 32, 192, 
        64, 384, 64, 384, 64, 384, 64, 384, 96, 576, 96, 576, 96, 576, 
        160, 960, 160, 960, 160, 960, 320, 1280
    ]
    mnv2_confs = [_make_divisible(v * width, 8) for v in mnv2_base_confs]
    model = create_mobilenet_v2_from_configurations(spec_confs=mnv2_confs)
    return model