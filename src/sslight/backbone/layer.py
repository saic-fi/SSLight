import torch, torchvision
from torch import nn
from copy import deepcopy
from sslight.utils.pthflops import count_ops
from sslight.utils.param_utils import trunc_normal_
from sslight.backbone.mnv2 import mobilenet_v2
from sslight.backbone.resnet import *


class BACKBONE(nn.Module):
    def __init__(self, cfg):
        super(BACKBONE, self).__init__()
        self.cfg = cfg
        self.arch = cfg.MODEL.BACKBONE_ARCH
        if self.arch not in ['mobilenet_v2', 'resnet18', 'resnet34', 'resnet50']:
            assert False, f"Unsupported architecture: {self.arch}"
        if 'resnet' in self.arch:
            base_encoder = globals()[self.arch]()
            self.encoder = nn.Sequential(*list(base_encoder.children())[:-2])
            self.out_channels = base_encoder.fc.in_features
        else: # 'mobilenet' in self.arch:
            base_encoder = mobilenet_v2()
            self.encoder = base_encoder
            self.out_channels = base_encoder[-1].out_channels
            
        gflops, _ = count_ops(self.encoder, torch.rand(1,3,224,224), print_readable=False)
        print('GFLOPS of the backbone:', gflops/1e+9)

    def forward(self, x):
        x = self.encoder(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x


class TwoCropWrapper(nn.Module):
    def __init__(self, backbone, head):
        super(TwoCropWrapper, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, ignore_last_layer=False, extract_features_only=False):
        embeds = self.backbone(x)
        if extract_features_only:
            return embeds
        logits = self.head(embeds, ignore_last_layer)
        return embeds, logits


class MultiCropWrapper(nn.Module):
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, ignore_last_layer=False, extract_features_only=False):
        if extract_features_only:
            return self.backbone(x)


        if not isinstance(x, list):
            x = [x]
        
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1], 
            0
        )

        start_idx = 0
        
        for end_idx in idx_crops:
            res = self.backbone(torch.cat(x[start_idx: end_idx]))
            if start_idx == 0:
                embeds = res
            else:
                embeds = torch.cat((embeds, res))
            start_idx = end_idx
        

        logits = self.head(embeds, ignore_last_layer)
        
        return embeds, logits