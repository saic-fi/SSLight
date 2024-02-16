from detectron2.config import CfgNode as CN


def add_mobilenet_config(cfg):
    """
    Add config for tridentnet.
    """
    _C = cfg

    _C.MODEL.MOBILENET = CN()
    _C.MODEL.MOBILENET.VERSION = 'v2'
    _C.MODEL.MOBILENET.OUT_FEATURES = ["stem", "stage2", "stage3", "stage4", "stage5"]
    _C.MODEL.MOBILENET.NORM = "SyncBN"
    _C.MODEL.MOBILENET.USE_SPEC = False
    _C.MODEL.MOBILENET.SHALLOW  = False
    _C.MODEL.MOBILENET.LAST_CHANNELS = 1280
    _C.MODEL.MOBILENET.WIDTH_MULT = 1.0
    _C.MODEL.MOBILENET.ROUND_NEAREST = 8
    _C.MODEL.MOBILENET.REDUCED_TAIL = False
    _C.MODEL.MOBILENET.DILATED = False
    _C.MODEL.MOBILENET.PRETRAIN_METHOD = ''
    _C.MODEL.MOBILENET.PRETRAINED_WEIGHT = ''
    _C.MODEL.MOBILENET.BASE_SETTING = [
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]
    _C.MODEL.MOBILENET.SHALLOW_SETTING = [
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 2, 2],
        [6, 64, 2, 2],
        [6, 96, 3, 1],
        [6, 160, 2, 2],
        [6, 320, 1, 1],
    ]
    _C.MODEL.MOBILENET.SPEC_CHANNELS = [
        32, 16, 96, 
        24, 144, 24, 144, 
        32, 192, 32, 192, 32, 192, 
        64, 384, 64, 384, 64, 384, 64, 384, 
        96, 576, 96, 576, 96, 576, 
        160, 960, 160, 960, 160, 960, 
        320, 1280
    ]
    _C.MODEL.MOBILENET.SPEC_CHANNELS_SHALLOW = [
        32, 16, 96, 
        24, 144, 24, 144, 
        32, 192, 32, 192, 
        64, 384, 64, 384, 
        96, 576, 96, 576, 96, 576, 
        160, 960, 160, 960, 
        320, 1280
    ]
