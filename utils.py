"""
utils.py — Shared utility functions for metrics, visualisation, and reproducibility.
"""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
)
from config import OUTPUT_DIR


# ── Reproducibility ───────────────────────────────────────────────────────────

def seed_everything(seed: int = 42) -> None:
    """Fix all random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return the best available compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")
    return device


# ── Metrics & Visualisation ───────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: list,
    y_pred: list,
    save_path: Path | None = None,
    class_names: list[str] | None = None,
) -> None:
    """
    Generate and save a confusion matrix heatmap.

    Args:
        y_true:      Ground-truth binary labels (0 = real, 1 = fake).
        y_pred:      Predicted binary labels.
        save_path:   Where to save the PNG. Defaults to outputs/confusion_matrix.png.
        class_names: Label names for axes. Defaults to ['Real', 'Fake'].
    """
    if class_names is None:
        class_names = ["Real", "Fake"]
    if save_path is None:
        save_path = OUTPUT_DIR / "confusion_matrix.png"

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Utils] Confusion matrix saved → {save_path}")


def plot_roc_curve(
    y_true: list,
    y_scores: list,
    save_path: Path | None = None,
) -> float:
    """
    Generate and save an AUC-ROC curve.

    Args:
        y_true:    Ground-truth binary labels.
        y_scores:  Predicted probability scores for the positive class.
        save_path: Where to save the PNG. Defaults to outputs/roc_curve.png.

    Returns:
        roc_auc: Area under the ROC curve.
    """
    if save_path is None:
        save_path = OUTPUT_DIR / "roc_curve.png"

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4F8EF7", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Utils] ROC curve saved → {save_path}  |  AUC = {roc_auc:.4f}")
    return roc_auc


def print_classification_report(y_true: list, y_pred: list) -> None:
    """Print a full precision / recall / F1 report to stdout."""
    labels_present = sorted(set(int(v) for v in y_true) | set(int(v) for v in y_pred))
    all_names = ["Real", "Fake"]
    target_names = [all_names[i] for i in labels_present]
    report = classification_report(y_true, y_pred, labels=labels_present, target_names=target_names)
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(report)


def plot_training_history(
    train_losses: list[float],
    val_losses: list[float],
    val_aucs: list[float],
    save_path: Path | None = None,
) -> None:
    """
    Plot and save training/validation loss and validation AUC over epochs.

    Args:
        train_losses: Per-epoch training losses.
        val_losses:   Per-epoch validation losses.
        val_aucs:     Per-epoch validation AUC scores.
        save_path:    Output PNG path.
    """
    if save_path is None:
        save_path = OUTPUT_DIR / "training_history.png"

    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax1.plot(epochs, train_losses, label="Train Loss", color="#E74C3C", lw=2)
    ax1.plot(epochs, val_losses,   label="Val Loss",   color="#3498DB", lw=2)
    ax1.set_title("Loss over Epochs", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # AUC plot
    ax2.plot(epochs, val_aucs, label="Val AUC", color="#2ECC71", lw=2)
    ax2.set_title("Validation AUC over Epochs", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC")
    ax2.set_ylim([0.0, 1.05])
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Utils] Training history saved → {save_path}")
