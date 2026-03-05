"""
inference.py — Run deepfake detection on a single video file.

Usage
─────
  python inference.py --video path/to/video.mp4
  python inference.py --video path/to/video.mp4 --model outputs/checkpoints/best_model.pth
  python inference.py --video path/to/video.mp4 --threshold 0.6

Output
──────
  ✅ REAL  (confidence: 87.3%)   ← if predicted probability < threshold
  🚨 FAKE  (confidence: 94.1%)   ← if predicted probability ≥ threshold
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import config
from data_loader import VideoFrameSampler, FaceExtractor, _build_val_transform
from model import DeepfakeDetector
from utils import get_device


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Temporal Deepfake Detector — Inference")
    parser.add_argument("--video",     required=True, type=str,
                        help="Path to the input video file")
    parser.add_argument("--model",     type=str, default=str(config.BEST_MODEL_PATH),
                        help="Path to the model checkpoint (.pth)")
    parser.add_argument("--threshold", type=float, default=config.INFER_THRESHOLD,
                        help="Probability threshold for FAKE classification (default: 0.5)")
    parser.add_argument("--seq_len",   type=int, default=config.SEQ_LEN,
                        help="Frames per inference window")
    parser.add_argument("--stride",    type=int, default=10,
                        help="Window stride in frames (overlap control)")
    return parser.parse_args()


# ── Face sequence loading ────────────────────────────────────────────────────

def load_face_crops(video_path: str, extractor: FaceExtractor) -> list[np.ndarray]:
    """
    Extract all detectable face crops from a video (every FRAME_STEP-th frame).
    Returns list of RGB numpy arrays (IMG_SIZE, IMG_SIZE, 3).
    """
    crops = []
    sampler = VideoFrameSampler(video_path, frame_step=config.FRAME_STEP)
    for frame_bgr in tqdm(sampler, desc="  Extracting faces", leave=False, dynamic_ncols=True):
        crop = extractor.extract(frame_bgr)
        if crop is not None:
            crops.append(crop)
    return crops


def crops_to_windows(
    crops:   list[np.ndarray],
    seq_len: int,
    stride:  int,
    transform,
) -> list[torch.Tensor]:
    """
    Slice the crop list into overlapping windows of length seq_len.
    Each window becomes a tensor of shape (1, seq_len, 3, H, W).
    """
    if len(crops) < seq_len:
        # Pad by repeating last frame
        crops = crops + [crops[-1]] * (seq_len - len(crops))

    windows = []
    for start in range(0, len(crops) - seq_len + 1, stride):
        window = crops[start : start + seq_len]
        tensor_list = [transform(image=f)["image"] for f in window]   # (3, H, W) each
        window_tensor = torch.stack(tensor_list, dim=0).unsqueeze(0)  # (1, SEQ, 3, H, W)
        windows.append(window_tensor)

    return windows


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: str | Path, device: torch.device) -> DeepfakeDetector:
    """Load a trained DeepfakeDetector from a checkpoint file."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f"[Inference] ⚠️  Checkpoint not found: {ckpt_path}")
        print("[Inference] Running with random weights (pipeline validation only).")
        model = DeepfakeDetector(pretrained=False)
    else:
        model = DeepfakeDetector(pretrained=False)
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[Inference] Loaded checkpoint: {ckpt_path}")

    model.eval()
    return model.to(device)


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model:     DeepfakeDetector,
    windows:   list[torch.Tensor],
    device:    torch.device,
) -> tuple[float, str]:
    """
    Run the model over all windows and aggregate results.

    Aggregation: mean probability across windows.
    Returns:
        (mean_prob, verdict)  — verdict is 'FAKE' or 'REAL'
    """
    probs = []
    for window in tqdm(windows, desc="  Running model", leave=False, dynamic_ncols=True):
        window = window.to(device)
        logit  = model(window)                        # (1, 1)
        prob   = torch.sigmoid(logit).item()
        probs.append(prob)

    mean_prob = float(np.mean(probs)) if probs else 0.5
    return mean_prob, probs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = get_device()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[Inference] ❌ Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[Inference] Video        : {video_path.name}")
    print(f"[Inference] Model        : {args.model}")
    print(f"[Inference] Threshold    : {args.threshold}")
    print(f"[Inference] Window size  : {args.seq_len} frames")
    print(f"[Inference] Window stride: {args.stride} frames")
    print()

    # Step 1 — extract faces
    extractor = FaceExtractor(device=str(device))
    crops     = load_face_crops(str(video_path), extractor)

    if not crops:
        print("[Inference] ⚠️  No faces detected in the video. Exiting.")
        sys.exit(1)

    print(f"[Inference] Detected {len(crops)} face crops.")

    # Step 2 — build sliding windows
    transform = _build_val_transform()
    windows   = crops_to_windows(crops, args.seq_len, args.stride, transform)
    print(f"[Inference] Built {len(windows)} inference window(s).")

    # Step 3 — load model
    model = load_model(args.model, device)

    # Step 4 — run inference
    mean_prob, all_probs = run_inference(model, windows, device)

    # Step 5 — display verdict
    is_fake   = mean_prob >= args.threshold
    confidence = mean_prob if is_fake else 1.0 - mean_prob
    verdict    = "FAKE" if is_fake else "REAL"
    icon       = "🚨" if is_fake else "✅"

    print("\n" + "═" * 50)
    print(f"  {icon}  {verdict}  (confidence: {confidence * 100:.1f}%)")
    print(f"      Raw fake probability: {mean_prob:.4f}")
    print(f"      Windows analysed    : {len(all_probs)}")
    if all_probs:
        print(f"      Per-window range    : [{min(all_probs):.3f} – {max(all_probs):.3f}]")
    print("═" * 50 + "\n")

    return verdict, mean_prob


if __name__ == "__main__":
    main()
