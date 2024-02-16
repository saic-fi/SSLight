import torch


def get_model_loader(arch, method):
    func_name = 'load_%s_%s'%(method.lower(), arch)
    return globals()[func_name]


def load_simreg(path, model_keys):
    state_dict = torch.load(path, map_location="cpu")
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    for k in list(state_dict.keys()):
        if not ('encoder_q.module.' in k):
            del state_dict[k]
        if 'encoder_q.module.fc.' in k:
            del state_dict[k]
        
    for k_state, k_model in zip(list(state_dict.keys()), model_keys):
        state_dict[k_model] = state_dict[k_state]
        del state_dict[k_state]
    
    return state_dict


def load_simreg_resnet18(path, model_keys):
    return load_simreg(path, model_keys)


def load_simreg_mobilenet_v2(path, model_keys):
    return load_simreg(path, model_keys)


def load_disco(path, model_keys):
    state_dict = torch.load(path, map_location="cpu")
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    for k in list(state_dict.keys()):
        if not ('module.encoder_q.' in k):
            del state_dict[k]
        
    for k_state, k_model in zip(list(state_dict.keys()), model_keys):
        state_dict[k_model] = state_dict[k_state]
        del state_dict[k_state]
    
    return state_dict


def load_disco_resnet18(path, model_keys):
    return load_disco(path, model_keys)


def load_seed(path, model_keys):
    state_dict = torch.load(path, map_location="cpu")
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    for k in list(state_dict.keys()):
        if not ('module.encoder_q.' in k):
            del state_dict[k]
        
    for k_state, k_model in zip(list(state_dict.keys()), model_keys):
        state_dict[k_model] = state_dict[k_state]
        del state_dict[k_state]
    
    return state_dict


def load_seed_resnet18(path, model_keys):
    return load_seed(path, model_keys)


def load_paws_resnet50(path, model_keys):
    state_dict = torch.load(path, map_location="cpu")['encoder']

    for k in list(state_dict.keys()):
        if ('module.pred' in k) or ('module.fc.fc2' in k) or ('module.fc.bn2' in k) or ('module.fc.fc3' in k):
            del state_dict[k]
            
    for k_state, k_model in zip(list(state_dict.keys()), model_keys):
        state_dict[k_model] = state_dict[k_state]
        del state_dict[k_state]
    
    return state_dict 


def load_dino(path, model_keys):
    state_dict = torch.load(path, map_location="cpu")
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    for k in list(state_dict.keys()):
        if not ('module.student.backbone.encoder' in k):
            del state_dict[k]
        
    for k_state, k_model in zip(list(state_dict.keys()), model_keys):
        state_dict[k_model] = state_dict[k_state]
        del state_dict[k_state]

    return state_dict


def load_dino_mobilenet_v2(path, model_keys):
    return load_dino(path, model_keys)


def load_dino_mobilenet_v3_large(path, model_keys):
    return load_dino(path, model_keys)


def load_dino_resnet18(path, model_keys):
    return load_dino(path, model_keys)


def load_dino_resnet34(path, model_keys):
    return load_dino(path, model_keys)


def load_dino_resnet50(path, model_keys):
    return load_dino(path, model_keys)


def load_swav(path, model_keys):
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


def load_swav_mobilenet_v2(path, model_keys):
    return load_swav(path, model_keys)


def load_swav_mobilenet_v3_large(path, model_keys):
    return load_swav(path, model_keys)


def load_moco(path, model_keys):
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


def load_moco_mobilenet_v2(path, model_keys):
    return load_moco(path, model_keys)
