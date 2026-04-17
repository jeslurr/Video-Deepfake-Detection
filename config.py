"""
config.py — Centralised hyperparameters and path settings.

Edit the values in this file to customise dataset locations, model hyper-
parameters, and training behaviour without touching any other source file.
"""

import os
from pathlib import Path

# ── Project root (where this file lives) ─────────────────────────────────────
ROOT_DIR = Path(__file__).parent.resolve()

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR        = ROOT_DIR / "data"
FF_ROOT         = DATA_DIR / "FaceForensics"   # root of the FF++ dataset
DFDC_ROOT       = DATA_DIR / "DFDC"            # root of the DFDC dataset
CELEBDF_ROOT    = DATA_DIR / "Celeb-DF-v2"      # root of the Celeb-DF v2 dataset
DATASET_CSV     = DATA_DIR / "dataset.csv"     # unified CSV (video_path, label)

# ── Output / checkpoint paths ─────────────────────────────────────────────────
OUTPUT_DIR      = ROOT_DIR / "outputs"
CHECKPOINT_DIR  = OUTPUT_DIR / "checkpoints"
LOG_DIR         = OUTPUT_DIR / "logs"

for _d in (DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Frame sampling ────────────────────────────────────────────────────────────
FRAME_STEP  = 5    # Extract every N-th frame from the video
SEQ_LEN     = 12   # Number of consecutive face crops fed to the LSTM

# ── Image / face crop settings ────────────────────────────────────────────────
IMG_SIZE    = 160  # Height == Width of each face crop (pixels)
IMG_MEAN    = (0.485, 0.456, 0.406)   # ImageNet mean (RGB)
IMG_STD     = (0.229, 0.224, 0.225)   # ImageNet std  (RGB)

# ── Model architecture ────────────────────────────────────────────────────────
BACKBONE        = "efficientnet_v2_s"   # torchvision model name
SPATIAL_DIM     = 1280                  # Feature vector size from EfficientNetV2-S
LSTM_HIDDEN     = 512
LSTM_LAYERS     = 2
LSTM_DROPOUT    = 0.4
FC_HIDDEN       = 256
FREEZE_BACKBONE = True   # Freeze backbone weights for the first phase of training

# ── Training hyper-parameters ─────────────────────────────────────────────────
EPOCHS          = 30
BATCH_SIZE      = 2     # Keep small — each sample is SEQ_LEN frames stacked
LEARNING_RATE   = 3e-4
WEIGHT_DECAY    = 1e-4
EARLY_STOP_PAT  = 5      # Patience (epochs) for early stopping on val-AUC

# ── Dataset splits ────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

# ── Inference ─────────────────────────────────────────────────────────────────
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"
INFER_THRESHOLD = 0.5    # Probability threshold → FAKE if score >= threshold

# ── Misc ──────────────────────────────────────────────────────────────────────
NUM_WORKERS = 0    # 0 avoids multiprocessing pickle issues on Windows
PIN_MEMORY  = True
