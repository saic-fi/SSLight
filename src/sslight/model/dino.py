import torch
import torch.nn as nn

from copy import deepcopy
import sslight.backbone.vision_transformer as vits
from sslight.backbone.layer import BACKBONE, MultiCropWrapper
from sslight.utils.param_utils import trunc_normal_


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, ignore_last_layer=False):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        if ignore_last_layer:
            return x
        x = self.last_layer(x)
        return x


class DINO(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.arch = cfg.MODEL.BACKBONE_ARCH
        self.out_dim = cfg.DINOHead.OUTPUT_DIM
        self.use_bn_in_head = cfg.DINOHead.USE_BN_IN_HEAD
        self.norm_last_layer = cfg.DINOHead.NORM_LAST_YEAR

        if 'vit' in self.arch:
            self.student = vits.__dict__[self.arch](patch_size=cfg.VIT.PATCH_SIZE, drop_path_rate=cfg.VIT.DROP_PATH_RATE)
            self.teacher = vits.__dict__[self.arch](patch_size=cfg.VIT.PATCH_SIZE,)
            self.embed_dim = self.student.embed_dim
        elif self.arch in ['mobilenet_v2', 'resnet18', 'resnet34', 'resnet50']:
            encoder = BACKBONE(cfg)
            self.embed_dim = encoder.out_channels
            self.student = deepcopy(encoder)
            self.teacher = deepcopy(encoder)
        else:
            assert False, f"Unknow architecture: {self.arch}"

        self.student = MultiCropWrapper(
            self.student, 
            DINOHead(
                self.embed_dim,
                self.out_dim,
                use_bn=self.use_bn_in_head,
                norm_last_layer=self.norm_last_layer,
                nlayers=self.cfg.DINOHead.NUM_LAYERS,
                hidden_dim=self.cfg.DINOHead.HIDDEN_SIZE,
                bottleneck_dim=self.cfg.DINOHead.BOTTLENECK_DIM
            )
        )
        self.teacher = MultiCropWrapper(
            self.teacher,
            DINOHead(
                self.embed_dim, 
                self.out_dim, 
                use_bn=self.use_bn_in_head,
                norm_last_layer=self.norm_last_layer,
                nlayers = self.cfg.DINOHead.NUM_LAYERS,
                hidden_dim = self.cfg.DINOHead.HIDDEN_SIZE,
                bottleneck_dim=self.cfg.DINOHead.BOTTLENECK_DIM
            )
        )

        self.classifier = None
        if cfg.TRAIN.JOINT_LINEAR_PROBE:
            self.classifier = nn.Linear(self.embed_dim, cfg.TRAIN.NUM_CLASSES)

        self._initializes_teacher_network()

    @torch.no_grad()
    def _initializes_teacher_network(self):
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, images, extract_features_only=False):
        if extract_features_only:
            enc_feats = self.student(images, extract_features_only=True)
            sup_logits = self.classifier(enc_feats.detach()) if self.classifier is not None else None
            return enc_feats, sup_logits
        _, teacher_output = self.teacher(images[:self.cfg.MULTI_VIEWS_TRANSFORMS.NMB_CROPS[0]])  # only the global views pass through the teacher
        enc_feats, student_output = self.student(images)
        # Optionally train a linear classifier on top of the detached features
        # to estimate the quality of the representation during training
        batch_size = images[0].size(0)
        sup_logits = self.classifier(enc_feats.detach()[:batch_size].detach()) if self.classifier is not None else None
        return enc_feats, sup_logits, student_output, teacher_output
        
            