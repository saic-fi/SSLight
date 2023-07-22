import torch
import torch.nn as nn
from copy import deepcopy
from sslight.utils.param_utils import trunc_normal_
from sslight.backbone.layer import BACKBONE, TwoCropWrapper, MultiCropWrapper


class MOCOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, nlayers=2, hidden_dim=2048):
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

    def forward(self, x, ignore_last_layer=False):
        x = self.mlp(x)
        return x


class MOCO(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.K = cfg.MOCO.QUEUE_LENGTH
        self.m = cfg.MOCO.MOMENTUM
        self.T = cfg.MOCO.TEMPERATURE

        encoder = BACKBONE(cfg)
        self.embed_dim = encoder.out_channels
        if self.cfg.MOCO.GLOBAL_ONLY:
            self.encoder_q = TwoCropWrapper(
                deepcopy(encoder), 
                MOCOHead(
                    in_dim=self.embed_dim, 
                    out_dim=cfg.MOCO.OUTPUT_DIM, 
                    use_bn=cfg.MOCO.USE_BN_IN_HEAD, 
                    nlayers=cfg.MOCO.NUM_LAYERS, 
                    hidden_dim=cfg.MOCO.HIDDEN_SIZE
                )
            )
            self.encoder_k = TwoCropWrapper(
                deepcopy(encoder), 
                MOCOHead(
                    in_dim=self.embed_dim, 
                    out_dim=cfg.MOCO.OUTPUT_DIM, 
                    use_bn=cfg.MOCO.USE_BN_IN_HEAD, 
                    nlayers=cfg.MOCO.NUM_LAYERS, 
                    hidden_dim=cfg.MOCO.HIDDEN_SIZE
                )
            )
        else:
            self.encoder_q = MultiCropWrapper(
                deepcopy(encoder), 
                MOCOHead(
                    in_dim=self.embed_dim, 
                    out_dim=cfg.MOCO.OUTPUT_DIM, 
                    use_bn=cfg.MOCO.USE_BN_IN_HEAD, 
                    nlayers=cfg.MOCO.NUM_LAYERS, 
                    hidden_dim=cfg.MOCO.HIDDEN_SIZE
                )
            )
            self.encoder_k = MultiCropWrapper(
                deepcopy(encoder), 
                MOCOHead(
                    in_dim=self.embed_dim, 
                    out_dim=cfg.MOCO.OUTPUT_DIM, 
                    use_bn=cfg.MOCO.USE_BN_IN_HEAD, 
                    nlayers=cfg.MOCO.NUM_LAYERS, 
                    hidden_dim=cfg.MOCO.HIDDEN_SIZE
                )
            )

        self.classifier = None
        if cfg.TRAIN.JOINT_LINEAR_PROBE:
            self.classifier = nn.Linear(self.embed_dim, cfg.TRAIN.NUM_CLASSES)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(cfg.MOCO.OUTPUT_DIM, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, images, extract_features_only=False):
        if extract_features_only:
            enc_feats = self.encoder_q(images, extract_features_only=True)
            sup_logits = self.classifier(enc_feats.detach()) if self.classifier is not None else None
            return enc_feats, sup_logits

        batch_size = images[0].size(0)
        if self.cfg.MOCO.GLOBAL_ONLY:
            enc_feats, q = self.encoder_q(images[1])
        else:
            enc_feats, q = self.encoder_q(images[1:])
        q = nn.functional.normalize(q, dim=-1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            im_k = images[0]
            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            _, k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=-1)
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)


        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        # l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        if self.cfg.MOCO.GLOBAL_ONLY:
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        else:
            qs = q.chunk(sum(self.cfg.MULTI_VIEWS_TRANSFORMS.NMB_CROPS)-1)
            l_pos = torch.cat([torch.einsum('nc,nc->n', [u, k]).unsqueeze(-1) for u in qs], dim=0)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=-1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        sup_logits = self.classifier(enc_feats.detach()[:batch_size].detach()) if self.classifier is not None else None
        return enc_feats, sup_logits, q, k, logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
