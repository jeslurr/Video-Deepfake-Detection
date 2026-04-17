"""
train.py — Training script for the Temporal Deepfake Detector.

Features
────────
  • AdamW optimiser + CosineAnnealingLR scheduler
  • BCEWithLogitsLoss with class-weight balancing
  • Early stopping on validation AUC (patience configurable in config.py)
  • Checkpointing: best_model.pth + last_model.pth after every epoch
  • TensorBoard logging (loss + AUC per epoch)
  • --smoke_test flag: synthetic data, 2-epoch dry-run — no real dataset needed

Usage
─────
  # Standard training (requires dataset.csv to be built first)
  python train.py

  # Smoke test (validates pipeline end-to-end with synthetic data)
  python train.py --epochs 2 --batch_size 2 --smoke_test

  # Resume from checkpoint
  python train.py --resume outputs/checkpoints/last_model.pth

  # Fine-tune with unfrozen backbone
  python train.py --freeze_backbone False
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import config
from model import DeepfakeDetector
from utils import (
    get_device,
    seed_everything,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_history,
    print_classification_report,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train the Temporal Deepfake Detector")
    parser.add_argument("--epochs",          type=int,   default=config.EPOCHS)
    parser.add_argument("--batch_size",      type=int,   default=config.BATCH_SIZE)
    parser.add_argument("--lr",              type=float, default=config.LEARNING_RATE)
    parser.add_argument("--weight_decay",    type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--early_stop_pat",  type=int,   default=config.EARLY_STOP_PAT)
    parser.add_argument("--freeze_backbone", type=lambda x: x.lower() != "false",
                        default=config.FREEZE_BACKBONE,
                        help="Freeze EfficientNetV2 backbone ('true'/'false')")
    parser.add_argument("--resume",          type=str,   default=None,
                        help="Path to a checkpoint (.pth) to resume from")
    parser.add_argument("--smoke_test",      action="store_true",
                        help="Use synthetic tensors for a quick pipeline sanity check")
    parser.add_argument("--seed",            type=int,   default=config.RANDOM_SEED)
    return parser.parse_args()


# ── Synthetic data (smoke test) ───────────────────────────────────────────────

def _make_smoke_loaders(batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Generate 3 tiny synthetic DataLoaders for pipeline validation."""
    def _ds(n: int):
        x = torch.randn(n, config.SEQ_LEN, 3, config.IMG_SIZE, config.IMG_SIZE)
        y = torch.randint(0, 2, (n,)).float()
        return TensorDataset(x, y)

    kw = dict(batch_size=batch_size, shuffle=False)
    return (
        DataLoader(_ds(10), **kw),
        DataLoader(_ds(4),  **kw),
        DataLoader(_ds(4),  **kw),
    )


# ── Training & evaluation helpers ─────────────────────────────────────────────

def train_one_epoch(
    model:     DeepfakeDetector,
    loader:    DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device:    torch.device,
    scaler:    torch.cuda.amp.GradScaler,
) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc="  train", leave=False, dynamic_ncols=True)
    for videos, labels in pbar:
        videos = videos.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(videos)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimiser)
        scaler.update()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model:     DeepfakeDetector,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float, list, list]:
    """
    Evaluate model on a DataLoader.

    Returns:
        (avg_loss, auc_score, all_labels, all_probs)
    """
    model.eval()
    total_loss  = 0.0
    all_labels  = []
    all_probs   = []

    for videos, labels in tqdm(loader, desc="  eval ", leave=False, dynamic_ncols=True):
        videos = videos.to(device, non_blocking=True)
        labels_d = labels.float().unsqueeze(1).to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(videos)
            loss   = criterion(logits, labels_d)

        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        total_loss  += loss.item()
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.tolist())

    avg_loss = total_loss / max(len(loader), 1)
    try:
        auc = roc_auc_score(all_labels, all_probs)
        if np.isnan(auc):
            auc = 0.5
    except ValueError:
        auc = 0.5   # only one class present (smoke test edge case)

    return avg_loss, auc, all_labels, all_probs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()

    # ── DataLoaders ───────────────────────────────────────────────────────────
    if args.smoke_test:
        print("\n[Train] 🧪  SMOKE TEST MODE — using synthetic data.")
        train_loader, val_loader, test_loader = _make_smoke_loaders(args.batch_size)
    else:
        from data_loader import get_dataloaders
        print("\n[Train] Loading real dataset…")
        train_loader, val_loader, test_loader = get_dataloaders(
            csv_path   = config.DATASET_CSV,
            batch_size = args.batch_size,
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DeepfakeDetector(
        pretrained      = not args.smoke_test,   # skip download in smoke test
        freeze_backbone = args.freeze_backbone,
    ).to(device)

    params = model.count_parameters()
    print(f"[Train] Parameters — total: {params['total']:,}  trainable: {params['trainable']:,}")

    # ── Loss (class-weighted for imbalanced datasets) ─────────────────────────
    # Read class ratio from CSV directly (avoids loading all videos)
    if args.smoke_test:
        all_train_labels = [int(y) for _, batch_y in train_loader for y in batch_y]
    else:
        import pandas as pd
        _df = pd.read_csv(config.DATASET_CSV)
        n_train = int(len(_df) * config.TRAIN_RATIO)
        all_train_labels = _df["label"].iloc[:n_train].tolist()
    n_fake  = sum(all_train_labels)
    n_real  = len(all_train_labels) - n_fake
    print(f"[Train] Class balance — real: {n_real}  fake: {n_fake}")
    pos_weight = torch.tensor([n_real / max(n_fake, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Optimiser + Scheduler ─────────────────────────────────────────────────
    optimiser = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    
    # ── Optional resume ───────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_auc  = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimiser.load_state_dict(ckpt["optim_state"])
        start_epoch  = ckpt.get("epoch", 0) + 1
        best_val_auc = ckpt.get("best_val_auc", 0.0)
        print(f"[Train] Resumed from epoch {start_epoch}  (best AUC so far: {best_val_auc:.4f})")

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(config.LOG_DIR))

    # ── Training loop ─────────────────────────────────────────────────────────
    train_losses: list[float] = []
    val_losses:   list[float] = []
    val_aucs:     list[float] = []
    no_improve     = 0
    best_ckpt_path = config.CHECKPOINT_DIR / "best_model.pth"
    last_ckpt_path = config.CHECKPOINT_DIR / "last_model.pth"

    print(f"\n[Train] Starting training for {args.epochs} epoch(s)…\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        print(f"Epoch [{epoch + 1:03d}/{args.epochs}]")

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimiser, device, scaler)

        # Validate
        val_loss, val_auc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)

        elapsed = time.time() - t0
        print(
            f"  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_auc={val_auc:.4f}  "
            f"({elapsed:.1f}s)"
        )

        # TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        writer.add_scalar("AUC/val",    val_auc,    epoch)

        # Save last checkpoint
        ckpt = {
            "epoch":         epoch,
            "model_state":   model.state_dict(),
            "optim_state":   optimiser.state_dict(),
            "best_val_auc":  best_val_auc,
        }
        torch.save(ckpt, last_ckpt_path)

        # Save best checkpoint
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(ckpt, best_ckpt_path)
            print(f"  ✅ New best model saved (AUC={best_val_auc:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.early_stop_pat:
                print(f"\n[Train] Early stopping triggered after {no_improve} epochs without improvement.")
                break

    writer.close()

    # ── Final evaluation on test set ──────────────────────────────────────────
    print("\n[Train] Evaluating on test set…")
    # Load best checkpoint for evaluation
    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state"])

    _, test_auc, test_labels, test_probs = evaluate(model, test_loader, criterion, device)
    test_preds = [1 if p >= config.INFER_THRESHOLD else 0 for p in test_probs]

    print(f"[Train] Test AUC: {test_auc:.4f}")
    print_classification_report(test_labels, test_preds)

    plot_confusion_matrix(test_labels, test_preds)
    plot_roc_curve(test_labels, test_probs)
    plot_training_history(train_losses, val_losses, val_aucs)

    print(f"\n[Train] ✅ Done. Outputs saved to: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
