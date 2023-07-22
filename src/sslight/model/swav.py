import torch
import torch.nn as nn
from copy import deepcopy
from sslight.backbone.layer import BACKBONE, MultiCropWrapper
from sslight.utils.param_utils import trunc_normal_


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, nlayers=3, hidden_dim=4096):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, out_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, out_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        return x


class SWAVHead(nn.Module):
    def __init__(self, cfg, encoder_channels, normalize=True):
        super(SWAVHead, self).__init__()
        # normalize output features
        self.l2norm = normalize
        self.projection = MLP(in_dim=encoder_channels, out_dim=cfg.SWAV.OUTPUT_DIM, use_bn=cfg.SWAV.USE_BN_IN_HEAD,  nlayers=cfg.SWAV.NUM_LAYERS, hidden_dim=cfg.SWAV.HIDDEN_SIZE)
        # prototype layer
        self.prototypes = torch.nn.Linear(cfg.SWAV.OUTPUT_DIM, cfg.SWAV.NMB_PROTOTYPES, bias=False)

    def forward(self, x, ignore_last_layer=False):
        x = self.projection(x)
        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)
        return x, self.prototypes(x)


class SWAV(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # multi-crop wrapper handles forward with inputs of different resolutions
        backbone = BACKBONE(cfg)
        self.embed_dim = backbone.out_channels
        self.network = MultiCropWrapper(backbone, SWAVHead(cfg, self.embed_dim))

        self.classifier = None
        if cfg.TRAIN.JOINT_LINEAR_PROBE:
            self.classifier = nn.Linear(self.embed_dim, cfg.TRAIN.NUM_CLASSES)
            
        ###################################################################
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm1d)):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    nn.init.ones_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # print(f"Network is built")

    def forward(self, images, extract_features_only=False):
        if extract_features_only:
            enc_feats = self.network(images, extract_features_only=True)
            sup_logits = self.classifier(enc_feats.detach()) if self.classifier is not None else None
            return enc_feats, sup_logits
        
        batch_size = images[0].size(0)
        enc_feats, swav_out = self.network(images)
        swav_feats, swav_logits = swav_out  
        sup_logits = self.classifier(enc_feats.detach()[:batch_size].detach()) if self.classifier is not None else None
        return enc_feats, sup_logits, swav_feats, swav_logits