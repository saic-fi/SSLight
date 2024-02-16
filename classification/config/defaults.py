from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# Dataset.
_C.TRAIN.DATASET = "imagenet1k"
_C.TRAIN.FILE_NAME = 'train.h5'
_C.TRAIN.SUBSET_FILE_PATH = ''
_C.TRAIN.NUM_CLASSES = 1000

# Total mini-batch size to be fitted on each GPU
_C.TRAIN.BATCH_SIZE = 128

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 10

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# ---------------------------------------------------------------------------- #
# VAL options.
# ---------------------------------------------------------------------------- #
_C.VAL = CfgNode()

_C.VAL.DATASET = "imagenet1k"
_C.VAL.FILE_NAME = 'val.h5'
_C.VAL.SUBSET_FILE_PATH = ''

# Total mini-batch size to be fitted on each GPU
_C.VAL.BATCH_SIZE = 128

# ---------------------------------------------------------------------------- #
# DATA options.
# ---------------------------------------------------------------------------- #
_C.DATA = CfgNode()

# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = "./datasets/imagenet1k/"

# mean and std for image normalization
_C.DATA.MEAN = [0.485, 0.456, 0.406]
_C.DATA.STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------- #
# MODEL options.
# ---------------------------------------------------------------------------- #
_C.MODEL = CfgNode()

# choices=['vit_tiny', 'vit_small', 'vit_base'] + torchvision_archs
_C.MODEL.BACKBONE_ARCH = "resnet50" 
_C.MODEL.BACKBONE_GROUP_FACTOR = 1

# using supervised pre-trained imagenet weights
_C.MODEL.PRE_TRAINED_SUP = False

# width multiplier for the backbone
_C.MODEL.BACKBONE_WIDTH_MULT = 1.0

# using weights from a path
_C.MODEL.PRE_TRAINED_PATH = ""


_C.VIT = CfgNode()


_C.VIT.PATCH_SIZE = 16


_C.VIT.NUM_LAST_BLOCKS = 1


# ---------------------------------------------------------------------------- #
# SOLVER options. 
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()
_C.SOLVER.OPTIMIZING_METHOD = "LARS"
_C.SOLVER.NESTEROV = False
_C.SOLVER.WEIGHT_DECAY = 1.0e-6
_C.SOLVER.WEIGHT_DECAY_END = 0.0
_C.SOLVER.TOTAL_EPOCHS = 100
_C.SOLVER.WARMUP_EPOCHS = 0
_C.SOLVER.START_WARMUP = 0.0
_C.SOLVER.BASE_LR = 0.05
_C.SOLVER.MIN_LR = 1.0e-6
_C.SOLVER.LR_FACTOR = 1.0
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.CLIP_GRAD = 0.0
_C.SOLVER.SCHEDULER = "cos"
_C.SOLVER.MILESTONES = [60, 80]

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.LOG_STEP = 10
# stage from [TRAIN, VAL, TEST, FT]
_C.STAGE = "TRAIN" 
_C.DISTRIBUTED = True
_C.SEED = 0
_C.DIST_BACKEND = 'nccl'
_C.N_NODES = 1
_C.WORLD_SIZE = 1
_C.NODE_RANK = 0
_C.DIST_URL = "env://"
_C.WORKERS = 8
_C.SSL_METHOD  = "BYOL"
_C.EVAL_METHOD = "Linear" 
_C.NUM_GPUS_PER_NODE = 8

# this path works for mlp tensorboard only
_C.TENSORBOARD_PATH = "/tensorboard"


# Output basedir.
_C.OUTPUT_DIR = "./tmp"


def _assert_and_infer_cfg(cfg):
    # define assertions here!
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
