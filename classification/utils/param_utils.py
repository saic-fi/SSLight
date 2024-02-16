from torch import nn
import numpy as np 


def num_of_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    return num_params


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def get_params(model, lr_factor):
    classifier = []
    regularized = []
    not_regularized = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ("fc." in name) or ("classifier." in name):
            classifier.append(param)
        elif name.endswith(".bias") or len(param.shape) == 1 or (len(param.shape) == 4 and param.shape[1] == 1):
            not_regularized.append(param)
        else:
            regularized.append(param)
        
    return [{'params': classifier, 'lr_factor': lr_factor}, {'params': regularized, 'lr_factor': 1.0}, {'params': not_regularized, 'weight_decay': 0., 'lr_factor': 1.0}]
