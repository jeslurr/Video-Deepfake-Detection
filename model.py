"""
model.py — Temporal Deepfake Detector: EfficientNetV2-S (spatial) + Bi-LSTM (temporal).

Architecture overview
─────────────────────
Input  : (batch, seq_len, 3, H, W)   — a sequence of face-crop tensors
          ↓
SpatialEncoder   : EfficientNetV2-S backbone (ImageNet pre-trained)
                   → per-frame feature vector of size SPATIAL_DIM (1280)
          ↓  (batch × seq_len, SPATIAL_DIM)  →  reshape  →  (batch, seq_len, 1280)
TemporalClassifier : 2-layer Bidirectional LSTM
                   → mean-pool over time dimension
                   → FC(1024 → 256) → GELU → Dropout → FC(256 → 1) → Sigmoid
Output : (batch, 1)  — probability of being FAKE  [0 = Real, 1 = Fake]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm
from config import (
    BACKBONE,
    SPATIAL_DIM,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    LSTM_DROPOUT,
    FC_HIDDEN,
    FREEZE_BACKBONE,
    SEQ_LEN,
)


# ── Spatial Encoder ───────────────────────────────────────────────────────────

class SpatialEncoder(nn.Module):
    """
    EfficientNetV2-S backbone that returns a fixed-size feature vector for
    every input frame.

    Args:
        pretrained:      Load ImageNet-1K weights.
        freeze_backbone: Freeze all backbone parameters (fine-tune head only).
    """

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = FREEZE_BACKBONE):
        super().__init__()

        weights = tvm.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tvm.efficientnet_v2_s(weights=weights)

        # Remove the classification head — keep features + avgpool
        self.features = backbone.features
        self.avgpool  = backbone.avgpool
        # backbone.classifier[1].in_features == 1280 for EfficientNetV2-S
        self.out_dim  = backbone.classifier[1].in_features

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 3, H, W)
        Returns:
            features: (N, out_dim)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def unfreeze(self) -> None:
        """Un-freeze backbone weights (call before fine-tuning stage)."""
        for param in self.features.parameters():
            param.requires_grad = True


# ── Temporal Classifier ───────────────────────────────────────────────────────

class TemporalClassifier(nn.Module):
    """
    Bidirectional LSTM that consumes a sequence of spatial feature vectors and
    outputs a single fake-probability score.

    Args:
        input_size:  Dimensionality of each frame's spatial feature (SPATIAL_DIM).
        hidden_size: LSTM hidden units per direction.
        num_layers:  Number of stacked LSTM layers.
        dropout:     Dropout between LSTM layers (and before FC).
        fc_hidden:   Size of the intermediate fully-connected layer.
    """

    def __init__(
        self,
        input_size:  int = SPATIAL_DIM,
        hidden_size: int = LSTM_HIDDEN,
        num_layers:  int = LSTM_LAYERS,
        dropout:     float = LSTM_DROPOUT,
        fc_hidden:   int = FC_HIDDEN,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            bidirectional = True,
            dropout = dropout if num_layers > 1 else 0.0,
        )

        lstm_out_size = hidden_size * 2  # bidirectional → 2× hidden

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_size, fc_hidden),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            logits: (batch, 1)  — raw logits (no sigmoid applied here)
        """
        lstm_out, _ = self.lstm(x)           # (batch, seq_len, hidden*2)
        pooled = lstm_out.mean(dim=1)        # mean-pool over time → (batch, hidden*2)
        logits = self.classifier(pooled)     # (batch, 1)
        return logits


# ── Full Detector ─────────────────────────────────────────────────────────────

class DeepfakeDetector(nn.Module):
    """
    End-to-end Temporal Deepfake Detector.

    Combines SpatialEncoder (per-frame) with TemporalClassifier (sequence-level).

    Args:
        pretrained:      Pre-train EfficientNetV2-S on ImageNet.
        freeze_backbone: Freeze spatial encoder during initial training phase.

    Input shape  : (batch, seq_len, C, H, W)
    Output shape : (batch, 1)   — raw logit; apply sigmoid for probability
    """

    def __init__(
        self,
        pretrained:      bool = True,
        freeze_backbone: bool = FREEZE_BACKBONE,
    ):
        super().__init__()
        self.spatial  = SpatialEncoder(pretrained=pretrained, freeze_backbone=freeze_backbone)
        self.temporal = TemporalClassifier(input_size=self.spatial.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 3, H, W)
        Returns:
            logits: (batch, 1)
        """
        batch, seq_len, C, H, W = x.shape

        # Merge batch and time dims so SpatialEncoder sees (batch*seq_len, C, H, W)
        x_flat = x.view(batch * seq_len, C, H, W)
        feats  = self.spatial(x_flat)                          # (batch*seq_len, D)
        feats  = feats.view(batch, seq_len, -1)                # (batch, seq_len, D)

        logits = self.temporal(feats)                          # (batch, 1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability (0–1) rather than raw logit."""
        return torch.sigmoid(self.forward(x))

    def unfreeze_backbone(self) -> None:
        """Un-freeze spatial encoder for fine-tuning stage."""
        self.spatial.unfreeze()

    def count_parameters(self) -> dict[str, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepfakeDetector(pretrained=False).to(device)   # pretrained=False for speed
    dummy = torch.randn(2, SEQ_LEN, 3, config.IMG_SIZE, config.IMG_SIZE).to(device)

    with torch.no_grad():
        out = model(dummy)

    params = model.count_parameters()
    print(f"Input  shape : {tuple(dummy.shape)}")
    print(f"Output shape : {tuple(out.shape)}")
    print(f"Total params : {params['total']:,}")
    print(f"Trainable    : {params['trainable']:,}")
    print("✅ model.py sanity check passed.")
