import torch


def load_dino_cnn(path, model_keys, use_head=False):
    state_dict = torch.load(path, map_location="cpu")
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    for k in list(state_dict.keys()):
        if use_head:
            if not ('module.student.backbone.encoder' in k or 'module.student.head.mlp.0' in k or 'module.student.head.mlp.1' in k or 'module.student.head.mlp.3' in k or 'module.student.head.mlp.4' in k):
                del state_dict[k]
        else:
            if not ('module.student.backbone.encoder' in k):
                del state_dict[k]
        
    for k_state, k_model in zip(list(state_dict.keys()), model_keys):
        state_dict[k_model] = state_dict[k_state]
        # print(k_model, k_state)
        del state_dict[k_state]

    return state_dict


def load_dino_vit(path, model_keys, use_head=False):
    state_dict = torch.load(path, map_location="cpu")
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    for k in list(state_dict.keys()):
        if use_head:
            if not ('module.student.backbone.encoder' in k or 'module.student.head.mlp.0' in k or 'module.student.head.mlp.1' in k):
                del state_dict[k]
        else:
            if not ('module.student.backbone' in k):
                del state_dict[k]
        
    for k_state, k_model in zip(list(state_dict.keys()), model_keys):
        state_dict[k_model] = state_dict[k_state]
        # print(k_model, k_state)
        del state_dict[k_state]

    return state_dict


def load_swav_cnn(path, model_keys, **kwargs):
    state_dict = torch.load(path, map_location="cpu")
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    for k in list(state_dict.keys()):
        if not ('module.network.backbone.encoder' in k):
            del state_dict[k]
        
    for k_state, k_model in zip(list(state_dict.keys()), model_keys):
        state_dict[k_model] = state_dict[k_state]
        del state_dict[k_state]

    return state_dict


def load_moco_cnn(path, model_keys, **kwargs):
    state_dict = torch.load(path, map_location="cpu")
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    for k in list(state_dict.keys()):
        if not ('module.encoder_q.backbone.encoder' in k):
            del state_dict[k]
        
    for k_state, k_model in zip(list(state_dict.keys()), model_keys):
        state_dict[k_model] = state_dict[k_state]
        del state_dict[k_state]

    return state_dict


model_to_loader = {
    'dino_mobilenet_v2': load_dino_cnn,
    'dino_resnet18': load_dino_cnn,
    'dino_resnet34': load_dino_cnn,
    'dino_resnet50': load_dino_cnn,
    'dino_vit_tiny': load_dino_vit,
    'dino_vit_small': load_dino_vit,
    'swav_mobilenet_v2': load_swav_cnn,
    'moco_mobilenet_v2': load_moco_cnn,
}


def get_model_loader(arch, method):
    model_name = '%s_%s'%(method.lower(), arch.lower())
    if model_name not in model_to_loader:
        raise NotImplementedError('Model %s not supported'%model_name)
    return model_to_loader[model_name]