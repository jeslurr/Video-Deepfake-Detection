"""
data_loader.py — Video frame sampling, face extraction, and dataset utilities.

Components
──────────
  VideoFrameSampler  — Iterates over a video and yields every N-th frame.
  FaceExtractor      — Detects and crops a face using MediaPipe (+ MTCNN fallback).
  DeepfakeDataset    — PyTorch Dataset: video → sequence of face tensors + label.
  get_dataloaders    — Creates train / val / test DataLoaders from a CSV.
  scaffold_ff_plus   — Builds dataset.csv from a FaceForensics++ directory tree.
  scaffold_dfdc      — Builds dataset.csv from a DFDC directory tree.
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

import config

# Silence noisy mediapipe / protobuf deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ── Optional imports (graceful degradation) ───────────────────────────────────

try:
    import mediapipe as mp
    # Verify mp.solutions is accessible (removed in some newer versions)
    _ = mp.solutions.face_detection
    _MP_AVAILABLE = True
except (ImportError, AttributeError):
    _MP_AVAILABLE = False
    print("[DataLoader] mediapipe face_detection not available — using MTCNN/Haar fallback.", file=sys.stderr)

try:
    from facenet_pytorch import MTCNN
    _MTCNN_AVAILABLE = True
except ImportError:
    _MTCNN_AVAILABLE = False
    print("[DataLoader] facenet-pytorch not found — using OpenCV Haar fallback.", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# VideoFrameSampler
# ─────────────────────────────────────────────────────────────────────────────

class VideoFrameSampler:
    """
    Iterate over a video file and yield every ``frame_step``-th frame as a
    BGR numpy array.

    Args:
        video_path: Path to the video file.
        frame_step: Sample every N-th frame (default from config).
    """

    def __init__(self, video_path: str | Path, frame_step: int = config.FRAME_STEP):
        self.video_path = Path(video_path)
        self.frame_step = frame_step

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % self.frame_step == 0:
                    yield frame          # BGR uint8 (H, W, 3)
                frame_idx += 1
        finally:
            cap.release()

    def count_frames(self) -> int:
        """Return the total number of frames that will be yielded."""
        cap = cv2.VideoCapture(str(self.video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return max(0, total // self.frame_step)


# ─────────────────────────────────────────────────────────────────────────────
# FaceExtractor
# ─────────────────────────────────────────────────────────────────────────────

class FaceExtractor:
    """
    Detect and crop a face from a BGR frame.

    Strategy (in priority order):
        1. MediaPipe Face Detection (fast, CPU-friendly)
        2. MTCNN from facenet-pytorch (GPU-optional)
        3. OpenCV Haar Cascade (always available as last resort)

    The returned crop is always resized to (IMG_SIZE, IMG_SIZE) in RGB.
    ``None`` is returned when no face is detected.
    """

    def __init__(self, device: str = "cpu"):
        self._detector = None
        self._mtcnn    = None
        self._haar     = None
        self._device   = device

        if _MP_AVAILABLE:
            _mp = mp.solutions.face_detection
            self._detector = _mp.FaceDetection(
                model_selection=1,   # 1 = full-range model
                min_detection_confidence=0.5,
            )
        if _MTCNN_AVAILABLE:
            self._mtcnn = MTCNN(
                keep_all=False,
                device=device,
                post_process=False,
            )
        # Haar is always available via OpenCV
        har_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._haar = cv2.CascadeClassifier(har_path)

    def extract(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract and return a face crop as RGB numpy array (IMG_SIZE, IMG_SIZE, 3).
        Returns None if no face is detected.
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # ── 1. MediaPipe ──────────────────────────────────────────────────────
        if self._detector is not None:
            results = self._detector.process(frame_rgb)
            if results.detections:
                det  = results.detections[0]
                bbox = det.location_data.relative_bounding_box
                x1   = max(0, int(bbox.xmin * w))
                y1   = max(0, int(bbox.ymin * h))
                x2   = min(w, int((bbox.xmin + bbox.width)  * w))
                y2   = min(h, int((bbox.ymin + bbox.height) * h))
                crop = frame_rgb[y1:y2, x1:x2]
                if crop.size > 0:
                    return cv2.resize(crop, (config.IMG_SIZE, config.IMG_SIZE))

        # ── 2. MTCNN ──────────────────────────────────────────────────────────
        if self._mtcnn is not None:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(frame_rgb)
            box, _ = self._mtcnn.detect(pil_img)
            if box is not None and len(box) > 0:
                x1, y1, x2, y2 = [int(v) for v in box[0]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = frame_rgb[y1:y2, x1:x2]
                if crop.size > 0:
                    return cv2.resize(crop, (config.IMG_SIZE, config.IMG_SIZE))

        # ── 3. Haar Cascade ───────────────────────────────────────────────────
        if self._haar is not None:
            gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) > 0:
                x, y, fw, fh = faces[0]
                crop = frame_rgb[y : y + fh, x : x + fw]
                if crop.size > 0:
                    return cv2.resize(crop, (config.IMG_SIZE, config.IMG_SIZE))

        return None  # No face detected in this frame


# ─────────────────────────────────────────────────────────────────────────────
# Albumentations augmentation pipelines
# ─────────────────────────────────────────────────────────────────────────────

def _build_train_transform() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CoarseDropout(max_holes=4, max_height=20, max_width=20, p=0.2),
        A.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
        ToTensorV2(),
    ])


def _build_val_transform() -> A.Compose:
    return A.Compose([
        A.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# DeepfakeDataset
# ─────────────────────────────────────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for deepfake video classification.

    Expects a CSV with columns ``video_path`` and ``label`` (0 = real, 1 = fake).
    Each item is a tensor of shape ``(SEQ_LEN, 3, IMG_SIZE, IMG_SIZE)`` and an
    integer label.

    Args:
        dataframe:  Pandas DataFrame with ``video_path`` and ``label`` columns.
        split:      One of ``'train'``, ``'val'``, ``'test'``.
        seq_len:    Number of consecutive face crops per sample.
        face_extractor: Shared FaceExtractor instance (avoids re-init overhead).
    """

    def __init__(
        self,
        dataframe:      pd.DataFrame,
        split:          str = "train",
        seq_len:        int = config.SEQ_LEN,
        face_extractor: Optional[FaceExtractor] = None,
    ):
        self.df            = dataframe.reset_index(drop=True)
        self.split         = split
        self.seq_len       = seq_len
        self.extractor     = face_extractor or FaceExtractor()
        self.transform     = (_build_train_transform() if split == "train"
                              else _build_val_transform())

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row        = self.df.iloc[idx]
        video_path = row["video_path"]
        label      = int(row["label"])

        frames = self._load_face_sequence(video_path)

        # Stack → (SEQ_LEN, 3, H, W)
        tensor_list = []
        for frame in frames:
            aug = self.transform(image=frame)["image"]  # (3, H, W) float32
            tensor_list.append(aug)

        video_tensor = torch.stack(tensor_list, dim=0)   # (SEQ_LEN, 3, H, W)
        return video_tensor, label

    def _load_face_sequence(self, video_path: str) -> list[np.ndarray]:
        """
        Sample SEQ_LEN face crops from the video.
        Crops that fail extraction are replaced with a black frame.
        """
        sampler   = VideoFrameSampler(video_path, frame_step=config.FRAME_STEP)
        all_crops = []

        for frame_bgr in sampler:
            crop = self.extractor.extract(frame_bgr)
            if crop is not None:
                all_crops.append(crop)
                if len(all_crops) >= self.seq_len * 3:   # collect a buffer
                    break

        # Ensure exactly SEQ_LEN crops
        if len(all_crops) == 0:
            # No faces detected — return black frames
            blank = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)
            return [blank] * self.seq_len

        # Evenly sample SEQ_LEN from available crops
        indices = np.linspace(0, len(all_crops) - 1, self.seq_len, dtype=int)
        return [all_crops[i] for i in indices]


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    csv_path:   str | Path = config.DATASET_CSV,
    batch_size: int        = config.BATCH_SIZE,
    num_workers: int       = config.NUM_WORKERS,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from a unified CSV.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    df = pd.read_csv(csv_path)

    # Shuffle deterministically
    df = df.sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)

    n_total = len(df)
    n_train = int(n_total * config.TRAIN_RATIO)
    n_val   = int(n_total * config.VAL_RATIO)

    train_df = df.iloc[:n_train]
    val_df   = df.iloc[n_train : n_train + n_val]
    test_df  = df.iloc[n_train + n_val :]

    extractor  = FaceExtractor()   # shared across datasets to avoid repeated init

    train_ds = DeepfakeDataset(train_df, split="train", face_extractor=extractor)
    val_ds   = DeepfakeDataset(val_df,   split="val",   face_extractor=extractor)
    test_ds  = DeepfakeDataset(test_df,  split="test",  face_extractor=extractor)

    make_loader = lambda ds, shuffle: DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = config.PIN_MEMORY,
    )

    return (
        make_loader(train_ds, shuffle=True),
        make_loader(val_ds,   shuffle=False),
        make_loader(test_ds,  shuffle=False),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset scaffolding helpers
# ─────────────────────────────────────────────────────────────────────────────

def scaffold_ff_plus_plus(
    ff_root: str | Path = config.FF_ROOT,
    output_csv: str | Path = config.DATASET_CSV,
) -> pd.DataFrame:
    """
    Walk a FaceForensics++ directory tree and build a unified dataset CSV.

    Expected layout::

        ff_root/
          original_sequences/
            youtube/
              c23/
                videos/
                  *.mp4  ← real (label 0)
          manipulated_sequences/
            Deepfakes/c23/videos/*.mp4    ← fake (label 1)
            Face2Face/c23/videos/*.mp4
            FaceSwap/c23/videos/*.mp4
            NeuralTextures/c23/videos/*.mp4

    Args:
        ff_root:    Root directory of the FF++ dataset.
        output_csv: Where to write the CSV.
    """
    ff_root = Path(ff_root)
    rows = []

    real_dir = ff_root / "original_sequences" / "youtube" / "c23" / "videos"
    if real_dir.exists():
        for f in real_dir.glob("*.mp4"):
            rows.append({"video_path": str(f), "label": 0})

    manip_root = ff_root / "manipulated_sequences"
    for method in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
        fake_dir = manip_root / method / "c23" / "videos"
        if fake_dir.exists():
            for f in fake_dir.glob("*.mp4"):
                rows.append({"video_path": str(f), "label": 1})

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"[Scaffold] FF++ CSV written: {len(df)} rows → {output_csv}")
    return df


def scaffold_dfdc(
    dfdc_root:  str | Path = config.DFDC_ROOT,
    output_csv: str | Path = config.DATASET_CSV,
) -> pd.DataFrame:
    """
    Walk a DFDC directory tree and build a unified dataset CSV.

    Expected layout::

        dfdc_root/
          train_part_XX/
            metadata.json   ← {"video.mp4": {"label": "REAL" or "FAKE"}, ...}
            *.mp4

    Args:
        dfdc_root:  Root directory of the DFDC dataset.
        output_csv: Where to write the CSV.
    """
    dfdc_root = Path(dfdc_root)
    rows = []

    for meta_file in sorted(dfdc_root.rglob("metadata.json")):
        part_dir = meta_file.parent
        with open(meta_file) as f:
            metadata = json.load(f)
        for fname, info in metadata.items():
            vid_path = part_dir / fname
            if not vid_path.exists():
                continue
            label = 0 if info["label"].upper() == "REAL" else 1
            rows.append({"video_path": str(vid_path), "label": label})

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"[Scaffold] DFDC CSV written: {len(df)} rows → {output_csv}")
    return df


def scaffold_celebdf(
    celebdf_root: str | Path = config.CELEBDF_ROOT,
    output_csv:   str | Path = config.DATASET_CSV,
) -> pd.DataFrame:
    """
    Walk a Celeb-DF v2 directory tree and build a unified dataset CSV.

    Expected layout::

        celebdf_root/
          Celeb-real/          ← real celebrity videos (label 0)
          Celeb-synthesis/     ← deepfake videos (label 1)
          YouTube-real/        ← real YouTube videos (label 0)

    Args:
        celebdf_root: Root directory of the Celeb-DF v2 dataset.
        output_csv:   Where to write the CSV.
    """
    celebdf_root = Path(celebdf_root)
    rows = []

    # Real videos
    for real_dir in ["Celeb-real", "YouTube-real"]:
        d = celebdf_root / real_dir
        if d.exists():
            for f in d.glob("*.mp4"):
                rows.append({"video_path": str(f), "label": 0})

    # Fake videos
    fake_dir = celebdf_root / "Celeb-synthesis"
    if fake_dir.exists():
        for f in fake_dir.glob("*.mp4"):
            rows.append({"video_path": str(f), "label": 1})

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"[Scaffold] Celeb-DF CSV written: {len(df)} rows → {output_csv}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("VideoFrameSampler, FaceExtractor, DeepfakeDataset — imports OK.")
    print("Run scaffold_ff_plus_plus(), scaffold_dfdc(), or scaffold_celebdf() after placing dataset files.")
