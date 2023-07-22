#!/usr/bin/env python3

"""Configs."""
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

_C.TRAIN.NUM_CLASSES = 1000

_C.TRAIN.FILE_NAME = 'train.h5'

# Subset for ablations
_C.TRAIN.SUBSET_FILE_PATH = ''

# Batch size per GPU
_C.TRAIN.BATCH_SIZE = 128

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 20

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# On-the-fly linear probe
_C.TRAIN.JOINT_LINEAR_PROBE = False

# ---------------------------------------------------------------------------- #
# VAL options.
# ---------------------------------------------------------------------------- #
_C.VAL = CfgNode()

_C.VAL.DATASET = "imagenet1k"

_C.VAL.FILE_NAME = 'val.h5'

_C.VAL.SUBSET_FILE_PATH = ''

_C.VAL.BATCH_SIZE = 128

# ---------------------------------------------------------------------------- #
# DATA options.
# ---------------------------------------------------------------------------- #
_C.DATA = CfgNode()

# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = "datasets/imagenet_ori/"

# Mean and std for image normalization
_C.DATA.MEAN = [0.485, 0.456, 0.406]

_C.DATA.STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------- #
# TWO_VIEWS_TRANSFORMS options. These transforms used for BYOL method
# ---------------------------------------------------------------------------- #
_C.TWO_VIEWS_TRANSFORMS = CfgNode()

# Different augmentation parameters needed to creat two views  
_C.TWO_VIEWS_TRANSFORMS.GLOBAL_CROPS_SCALE = [0.08, 1.0]

_C.TWO_VIEWS_TRANSFORMS.GLOBAL_CROP_SIZE = 224

_C.TWO_VIEWS_TRANSFORMS.RANDOM_HORIZONTAL_FLIP_PROB = [0.5, 0.5]

_C.TWO_VIEWS_TRANSFORMS.GAUSSIAN_BLUR_PROB = [1.0, 0.1]

_C.TWO_VIEWS_TRANSFORMS.COLOR_JITTER_PROB = [0.8, 0.8]

_C.TWO_VIEWS_TRANSFORMS.COLOR_JITTER_INTENSITY = [0.4, 0.4, 0.2, 0.1]

_C.TWO_VIEWS_TRANSFORMS.GREYSCALE_PROB = [0.2, 0.2]

_C.TWO_VIEWS_TRANSFORMS.CROP_PROB = [1.0, 1.0]

_C.TWO_VIEWS_TRANSFORMS.SOLARIZATION_PROB = [0.0, 0.2]

# ---------------------------------------------------------------------------- #
# MULTI_VIEWS_TRANSFORMS options. These transforms used for DINO method
# ---------------------------------------------------------------------------- #
_C.MULTI_VIEWS_TRANSFORMS = CfgNode()

# Different augmentation parameters needed to creat multiple views  (global and local)
_C.MULTI_VIEWS_TRANSFORMS.GLOBAL_CROPS_SCALE = [0.25, 1.0]

_C.MULTI_VIEWS_TRANSFORMS.LOCAL_CROPS_SCALE = [0.05, 0.25]

_C.MULTI_VIEWS_TRANSFORMS.LOCAL_CROPS_NUMBER = 6  

_C.MULTI_VIEWS_TRANSFORMS.RANDOM_HORIZONTAL_FLIP_PROB = [0.5, 0.5]

_C.MULTI_VIEWS_TRANSFORMS.GAUSSIAN_BLUR_PROB = [1.0, 0.1, 0.5] # global_1, global_2, local

_C.MULTI_VIEWS_TRANSFORMS.COLOR_JITTER_PROB = [0.8, 0.8]

_C.MULTI_VIEWS_TRANSFORMS.COLOR_JITTER_INTENSITY = [0.4, 0.4, 0.2, 0.1]

_C.MULTI_VIEWS_TRANSFORMS.GREYSCALE_PROB = [0.2, 0.2]

_C.MULTI_VIEWS_TRANSFORMS.CROP_PROB = [1.0, 1.0]

_C.MULTI_VIEWS_TRANSFORMS.LOCAL_CROP_SIZE = 96

_C.MULTI_VIEWS_TRANSFORMS.GLOBAL_CROP_SIZE = 224

_C.MULTI_VIEWS_TRANSFORMS.SOLARIZATION_PROB = [0.0, 0.2]

_C.MULTI_VIEWS_TRANSFORMS.CROPS_FOR_ASSIGN = [0, 1]

_C.MULTI_VIEWS_TRANSFORMS.NMB_CROPS = [2, 6]

# Hyper-parameters for the re-balanced loss (Sec. 3.3.4 in the paper)
_C.MULTI_VIEWS_TRANSFORMS.LAMBDAS = [0.142857, 0.857143]



# ---------------------------------------------------------------------------- #
# MODEL options.
# ---------------------------------------------------------------------------- #
_C.MODEL = CfgNode()

# choices=['vit_tiny', 'vit_small', 'vit_base'] + torchvision_archs
_C.MODEL.BACKBONE_ARCH = "vit_tiny" 

# Momentum for the teacher network in DINO
_C.MODEL.MODEL_MOMENTUM = 0.99

_C.MODEL.MODEL_MOMENTUM_END = 1.0


# ---------------------------------------------------------------------------- #
# SWAV options.
# ---------------------------------------------------------------------------- #
_C.SWAV = CfgNode()

_C.SWAV.FREEZE_PROTOTYPES_EPOCHS = 1

_C.SWAV.EPOCH_QUEUE_STARTS = 15 

_C.SWAV.TEMPERATURE = 0.1 

_C.SWAV.EPSILON = 0.05

_C.SWAV.SINKHORN_ITERATIONS = 3

_C.SWAV.NMB_PROTOTYPES = 3000 

_C.SWAV.QUEUE_LENGTH = 0

_C.SWAV.OUTPUT_DIM = 128

_C.SWAV.HIDDEN_SIZE = 4096

_C.SWAV.NUM_LAYERS = 2

_C.SWAV.USE_BN_IN_HEAD = True

# ---------------------------------------------------------------------------- #
# MOCO options.
# ---------------------------------------------------------------------------- #
_C.MOCO = CfgNode()

_C.MOCO.TEMPERATURE = 0.2

_C.MOCO.GLOBAL_ONLY = False 

_C.MOCO.QUEUE_LENGTH = 65536

_C.MOCO.MOMENTUM = 0.999

_C.MOCO.OUTPUT_DIM = 128

_C.MOCO.HIDDEN_SIZE = 2048

_C.MOCO.NUM_LAYERS = 2

_C.MOCO.USE_BN_IN_HEAD = False


# ---------------------------------------------------------------------------- #
# DINO options.
# ---------------------------------------------------------------------------- #
_C.DINO = CfgNode()

_C.DINO.GLOBAL_ONLY = False 

_C.DINO.STUDENT_TEMP = 0.1

# Initial value for the teacher temperature
_C.DINO.WARMUP_TEACHER_TEMP = 0.04

# Final value (after linear warmup) of the teacher temperature. 
_C.DINO.TEACHER_TEMP = 0.07

# Number of warmup epochs for the teacher temperature
_C.DINO.WARMUP_TEACHER_TEMP_EPOCHS = 30

_C.DINO.CENTER_MOMENTUM = 0.9

# ---------------------------------------------------------------------------- #
# DINOHead options. These options are used in DINO for the projection head
# ---------------------------------------------------------------------------- #
_C.DINOHead = CfgNode()

_C.DINOHead.OUTPUT_DIM = 4096

_C.DINOHead.HIDDEN_SIZE = 2048

_C.DINOHead.BOTTLENECK_DIM = 256

_C.DINOHead.NUM_LAYERS = 3

# Whether to use batch normalizations in projection head
_C.DINOHead.USE_BN_IN_HEAD = False

# Whether or not to weight normalize the last layer of the DINO head.
# Not normalizing leads to better performance but can make the training unstable.
# In our experiments, we typically set this paramater to False with vit_small and True with vit_base.
_C.DINOHead.NORM_LAST_YEAR = True

# ---------------------------------------------------------------------------- #
# VIT options. These are options related to transformer arch.
# ---------------------------------------------------------------------------- #
_C.VIT = CfgNode()

_C.VIT.PATCH_SIZE = 16

_C.VIT.DROP_PATH_RATE = 0.1

# ---------------------------------------------------------------------------- #
# SOLVER options. 
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

_C.SOLVER.OPTIMIZING_METHOD = "LARS"

_C.SOLVER.WD_FACTOR_FOR_BIAS = 0.0

_C.SOLVER.WD_FACTOR_FOR_DW = 0.0

_C.SOLVER.LARS_EXCLUDE_BIAS = True

_C.SOLVER.LARS_EXCLUDE_DW = False

_C.SOLVER.WEIGHT_DECAY = 1.0e-6

_C.SOLVER.WEIGHT_DECAY_END = 0.0

_C.SOLVER.TOTAL_EPOCHS = 300

_C.SOLVER.WARMUP_EPOCHS = 10

_C.SOLVER.START_WARMUP = 0.0

# Learning rate at the end of linear warmup (highest LR used during training). 
# The learning rate is linearly scaled
# with the batch size, and specified here for a reference batch size of 256.
_C.SOLVER.BASE_LR = 0.3

# Target LR at the end of optimization. 
# We use a cosine LR schedule with linear warmup.
_C.SOLVER.MIN_LR = 0.0

_C.SOLVER.MOMENTUM = 0.9

# Maximal parameter gradient norm if using gradient clipping. 
# Clipping with norm .3 ~ 1.0 can help optimization for larger ViT architectures. 
# 0 for disabling.
_C.SOLVER.CLIP_GRAD = 0.0

# Number of epochs during which we keep the output layer fixed. 
# Typically doing so during the first epoch helps training. 
# Try increasing this value if the loss does not decrease
_C.SOLVER.FREEZE_LAST_LAYER = 0

_C.SOLVER.SCHEDULER = "cos"

_C.SOLVER.MILESTONES = [60, 80]

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

_C.LOG_STEP = 10

_C.LOG_GRAD = True

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

_C.USE_FP16 = False

_C.SSL_METHOD = "DINO"

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

