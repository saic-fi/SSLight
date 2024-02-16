import os.path as osp
import torch
from torch import nn, Tensor
from torch.hub import load_state_dict_from_url
from typing import Any, Callable, Dict, List, Optional, Sequence
from detectron2.modeling import Backbone, BACKBONE_REGISTRY, FPN
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

import backbones.architectures as arch
from .load_pretrained_model import *


__all__ = [
    "MobileNet",
    "build_mobilenet_backbone",
]


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


from detectron2.layers import (
    ShapeSpec,
    BatchNorm2d,
    FrozenBatchNorm2d, 
    NaiveSyncBatchNorm
)
from detectron2.utils import env


def get_norm(norm):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": BatchNorm2d,
            # Fixed in https://github.com/pytorch/pytorch/pull/36382
            "SyncBN": NaiveSyncBatchNorm if env.TORCH_VERSION <= (1, 5) else nn.SyncBatchNorm,
            "FrozenBN": FrozenBatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            # for debugging:
            "nnSyncBN": nn.SyncBatchNorm,
            "naiveSyncBN": NaiveSyncBatchNorm,
        }[norm]
    return norm


############################################################################
class MobileNet(Backbone):
    def __init__(self, 
            stem, stages, stride_per_stage, channel_per_stage,
            num_classes=None, out_features=None, freeze_at=0) -> None:
        super(MobileNet, self).__init__()

        self.stem = nn.Sequential(*stem)
        self.num_classes = num_classes
        self._out_feature_strides = stride_per_stage
        self._out_feature_channels = channel_per_stage
        if out_features is not None:
            # Avoid keeping unused layers in this module. They consume extra memory
            # and may cause allreduce to fail
            num_stages = max(
                [{"stage2": 1, "stage3": 2, "stage4": 3, "stage5": 4}.get(f, 0) for f in out_features]
            )
            stages = stages[:num_stages]

        self.stage_names, self.stages = [], []
        for i, blocks in enumerate(stages):
            assert len(blocks) > 0, len(blocks)
            name = "stage" + str(i + 2)
            stage = nn.Sequential(*blocks)

            self.add_module(name, stage)
            self.stage_names.append(name)
            self.stages.append(stage)
        self.stage_names = tuple(self.stage_names)  # Make it static for scripting

        if num_classes is not None:
            # building classifier
            self.dropout = nn.Dropout(0.2)
            self.linear = nn.Linear(channel_per_stage[-1], num_classes)
            nn.init.normal_(self.linear.weight, std=0.01)
            nn.init.zeros_(self.linear.bias)
            name = 'linear'

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))
        self.freeze(freeze_at)

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def _forward_impl(self, x: Tensor):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4, f"MobileNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def forward(self, x: Tensor):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        return self._forward_impl(x)
    
    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the MobileNet. 
        Commonly used in fine-tuning ResNets.

        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.

        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.

        Returns:
            nn.Module: this MobileNet itself
        """
        if freeze_at >= 1:
            for parameter in self.stem.parameters():
                parameter.requires_grad_(False)
            FrozenBatchNorm2d.convert_frozen_batchnorm(self.stem)
        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                for parameter in stage.parameters():
                    parameter.requires_grad_(False)
                FrozenBatchNorm2d.convert_frozen_batchnorm(stage)
        return self


@BACKBONE_REGISTRY.register()
def build_mobilenet_backbone(cfg, input_shape):
    kwargs = {
        "width_mult": cfg.MODEL.MOBILENET.WIDTH_MULT,
        "norm_layer": get_norm(cfg.MODEL.MOBILENET.NORM),
        "round_nearest": cfg.MODEL.MOBILENET.ROUND_NEAREST,
    }
    if cfg.MODEL.MOBILENET.USE_SPEC:
        if cfg.MODEL.MOBILENET.SHALLOW:
            kwargs['channel_configuration'] = cfg.MODEL.MOBILENET.SPEC_CHANNELS_SHALLOW
            kwargs['inverted_residual_setting'] = cfg.MODEL.MOBILENET.SHALLOW_SETTING
        else:
            kwargs['channel_configuration'] = cfg.MODEL.MOBILENET.SPEC_CHANNELS
        arch_name = 'slimmable_mobilenet_%s'%cfg.MODEL.MOBILENET.VERSION
    else:
        arch_name = 'mobilenet_%s'%cfg.MODEL.MOBILENET.VERSION
    wholenet = arch.__dict__[arch_name](**kwargs)

    if cfg.MODEL.MOBILENET.PRETRAIN_METHOD.lower() == 'supervised':
        state_dict = load_state_dict_from_url(model_urls[arch_name], progress=True)
        msg = wholenet.load_state_dict(state_dict, strict=False)
        print('Torchvision supervised pretrained weights loaded with msg: {}'.format(msg))
    elif cfg.MODEL.MOBILENET.PRETRAIN_METHOD != '':
        pretrain_method   = cfg.MODEL.MOBILENET.PRETRAIN_METHOD
        pretrained_weight = cfg.MODEL.MOBILENET.PRETRAINED_WEIGHT
        assert osp.exists(pretrained_weight)

        if cfg.MODEL.MOBILENET.USE_SPEC:
            state_dict = load_spec_mobilenet_v2(pretrained_weight, list(wholenet.state_dict().keys()))
        else:
            state_dict = get_model_loader(arch_name, pretrain_method)(pretrained_weight, list(wholenet.state_dict().keys()))
        msg = wholenet.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(cfg.MODEL.MOBILENET.PRETRAINED_WEIGHT, msg))

    backbone = wholenet.features
    strides, channels = {}, {}
    stage_idx = 1
    current_stride = 2
    current_block = []
    stages = []
    for i in range(len(backbone)):
        b = backbone[i]
        if getattr(b, "_is_cn", False):
            strides['stage%d'%stage_idx] = current_stride
            channels['stage%d'%stage_idx] = current_block[-1].out_channels
            stage_idx += 1
            current_stride *= 2
            stages.append(current_block)
            current_block = []
        current_block.append(b)
    # if 'v3' in cfg.MODEL.MOBILENET.VERSION:
    #     lastconv_output_channels = current_block[-1].out_channels
    #     last_channel = cfg.MODEL.MOBILENET.LAST_CHANNELS
    #     last_conv1x1 = nn.Sequential(
    #             nn.Conv2d(lastconv_output_channels, last_channel, kernel_size=(1,1)),
    #             nn.Hardswish(inplace=False)
    #     )
    #     last_state_dict = wholenet.classifier[0].state_dict()
    #     last_state_dict['weight'] = last_state_dict['weight'][:,:,None,None]
    #     last_conv1x1[0].load_state_dict(last_state_dict)
    #     current_block.append(
    #         last_conv1x1
    #     )
    strides['stage%d'%stage_idx] = current_stride
    channels['stage%d'%stage_idx] = current_block[-1].out_channels#cfg.MODEL.MOBILENET.LAST_CHANNELS
    stages.append(current_block)

    # rename
    stem = stages[0]; stages = stages[1:]
    strides['stem'] = strides['stage1']; del strides['stage1']
    channels['stem'] = channels['stage1']; del channels['stage1']

    out_features  = cfg.MODEL.MOBILENET.OUT_FEATURES
    freeze_at     = cfg.MODEL.BACKBONE.FREEZE_AT
    return MobileNet(stem, stages, strides, channels, out_features=out_features, freeze_at=freeze_at)


@BACKBONE_REGISTRY.register()
def build_mobilenet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_mobilenet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone