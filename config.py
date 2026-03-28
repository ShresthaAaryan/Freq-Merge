"""
config.py  —  Central configuration for FreqMerge.
All hyper-parameters live here; no magic numbers in training scripts.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
HAM10000_DIR = os.path.join(_DATA_ROOT, "ham10000")
ISIC2019_DIR = os.path.join(_DATA_ROOT, "isic2019")
# Default dataset for CLI when --dataset is omitted
DEFAULT_SKIN_DATASET = "ham10000"
CKPT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
LOG_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
VIZ_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations")

# ---------------------------------------------------------------------------
# Dataset  (HAM10000 / ISIC 2019 — class count from ImageFolder at load time)
# ---------------------------------------------------------------------------
DATASET        = "skin_lesion"  # use --dataset ham10000 | isic2019
IMAGE_SIZE     = 224
PATCH_SIZE     = 16

# ImageNet normalization (ViT/ImageNet-pretrained backbones)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
BACKBONE       = "vit_small_patch16_224"
PRETRAINED     = True

MERGE_LAYERS   = [4, 6, 8, 10]
KEEP_RATE      = 0.7
ALPHA          = 0.7
HPF_RADIUS     = 2

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE     = 32
NUM_EPOCHS     = 20
LR             = 3e-4
WEIGHT_DECAY   = 0.01
LR_MIN         = 1e-6
WARMUP_EPOCHS  = 2
GRAD_CLIP      = 1.0

RANDOM_CROP_PADDING = 4
USE_MIXUP           = False

# ---------------------------------------------------------------------------
# Hardware & CUDA
# NOTE: PIN_MEMORY is intentionally a plain bool here.
#       cuda_utils.setup_cuda() handles the real device detection at runtime.
# ---------------------------------------------------------------------------
NUM_WORKERS    = 4
PIN_MEMORY     = True          # overridden to False at runtime when no GPU found
SEED           = 42

USE_AMP                      = True
CUDNN_BENCHMARK              = True
CUDNN_DETERMINISTIC          = False
USE_MULTI_GPU                = True
EMPTY_CACHE_EVERY_N_EPOCHS   = 5
USE_GRAD_CHECKPOINT          = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
PRINT_FREQ     = 100
SAVE_BEST_ONLY = True
