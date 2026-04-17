"""
evaluate.py — Comprehensive evaluation on Celeb-DF test set.

Computes accuracy, precision, recall, F1, AUC-ROC, and confusion matrix.

Usage
─────
  python evaluate.py --model outputs/checkpoints/best_model.pth
  python evaluate.py --model outputs/checkpoints/best_model.pth --threshold 0.5
  python evaluate.py --model outputs/checkpoints/best_model.pth --num-samples 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from tqdm import tqdm

import config
from data_loader import VideoFrameSampler, FaceExtractor, _build_val_transform
from model import DeepfakeDetector
from utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on Celeb-DF test set")
    parser.add_argument("--model",       type=str, default=str(config.BEST_MODEL_PATH),
                        help="Path to model checkpoint")
    parser.add_argument("--threshold",   type=float, default=0.5,
                        help="Decision threshold for FAKE classification")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Limit evaluation to N videos (for quick testing)")
    parser.add_argument("--stride",      type=int, default=10,
                        help="Window stride in frames")
    return parser.parse_args()


def load_model(ckpt_path: str | Path, device: torch.device) -> DeepfakeDetector:
    """Load trained model from checkpoint."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f"❌ Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    model = DeepfakeDetector(pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model.to(device)


def load_dataset() -> pd.DataFrame:
    """Load dataset.csv with video paths and true labels."""
    csv_path = config.DATASET_CSV
    if not csv_path.exists():
        print(f"❌ Dataset CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    # Filter to only existing files
    df = df[df['video_path'].apply(lambda p: Path(p).exists())]
    return df.reset_index(drop=True)


def run_inference_on_video(
    video_path: str,
    model: DeepfakeDetector,
    extractor: FaceExtractor,
    device: torch.device,
    stride: int = 10,
    seq_len: int = config.SEQ_LEN,
) -> float | None:
    """
    Run inference on a single video.
    Returns the mean fake probability, or None if face detection fails.
    """
    try:
        # Extract faces
        crops = []
        sampler = VideoFrameSampler(video_path, frame_step=config.FRAME_STEP)
        for frame_bgr in sampler:
            crop = extractor.extract(frame_bgr)
            if crop is not None:
                crops.append(crop)

        if not crops:
            return None

        # Build windows
        if len(crops) < seq_len:
            crops = crops + [crops[-1]] * (seq_len - len(crops))

        transform = _build_val_transform()
        windows = []
        for start in range(0, len(crops) - seq_len + 1, stride):
            window = crops[start : start + seq_len]
            tensor_list = [transform(image=f)["image"] for f in window]
            window_tensor = torch.stack(tensor_list, dim=0).unsqueeze(0)
            windows.append(window_tensor)

        if not windows:
            return None

        # Run inference
        probs = []
        with torch.no_grad():
            for window in windows:
                window = window.to(device)
                logit = model(window)
                prob = torch.sigmoid(logit).item()
                probs.append(prob)

        return float(np.mean(probs))

    except Exception as e:
        print(f"  ⚠️  Error processing {Path(video_path).name}: {e}")
        return None


def main():
    args = parse_args()
    device = get_device()

    print("\n" + "=" * 60)
    print("  CELEB-DF EVALUATION")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from: {args.model}")
    model = load_model(args.model, device)

    # Load dataset
    print("Loading dataset...")
    df = load_dataset()
    
    if args.num_samples:
        df = df.head(args.num_samples)
    
    print(f"Found {len(df)} test videos")
    
    # Initialize face extractor
    extractor = FaceExtractor(device=str(device))

    # Run inference on all videos
    print(f"\nRunning inference (threshold={args.threshold})...\n")
    
    predictions = []
    ground_truth = []
    video_names = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        video_path = row['video_path']
        true_label = int(row['label'])

        prob = run_inference_on_video(
            video_path, model, extractor, device, stride=args.stride
        )

        if prob is not None:
            pred_label = 1 if prob >= args.threshold else 0
            predictions.append(pred_label)
            ground_truth.append(true_label)
            video_names.append(Path(video_path).name)

    if not predictions:
        print("❌ No videos were successfully processed.", file=sys.stderr)
        sys.exit(1)

    # Compute metrics
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    acc = accuracy_score(ground_truth, predictions)
    prec = precision_score(ground_truth, predictions, zero_division=0)
    rec = recall_score(ground_truth, predictions, zero_division=0)
    f1 = f1_score(ground_truth, predictions, zero_division=0)
    auc = roc_auc_score(ground_truth, predictions) if len(set(ground_truth)) > 1 else 0.0
    cm = confusion_matrix(ground_truth, predictions)

    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:   {cm[0, 0]}")
    print(f"  False Positives:  {cm[0, 1]}")
    print(f"  False Negatives:  {cm[1, 0]}")
    print(f"  True Positives:   {cm[1, 1]}")

    print(f"\nVideos evaluated: {len(predictions)}/{len(df)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
