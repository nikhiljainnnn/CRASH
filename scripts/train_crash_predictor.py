"""
Phase 3: MSTT-CA Crash Predictor Training
==========================================
Trains the Multi-Scale Temporal Transformer on CCD crash sequences.

Architecture:
  ResNet18 (backbone) → MSTT-CA (temporal transformer) → crash/no-crash

Data flow:
  .npy sequences (50, 224, 224, 3) → CNN features (50, 512) → MSTT-CA → logits (2)

Usage:
  python scripts/train_crash_predictor.py
  python scripts/train_crash_predictor.py --epochs 50 --batch 8 --lr 0.0003
  python scripts/train_crash_predictor.py --resume checkpoints/crash_predictor/best.pt
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import torchvision.models as models
import torchvision.transforms as T

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.temporal.mstt_transformer import MSTT_CA
from models.utils.losses import FocalLoss

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x


# ─── Dataset ─────────────────────────────────────────────────────────────────
class CrashSequenceDataset(Dataset):
    """
    Loads pre-built .npy sequences from manifest JSON files.
    Each .npy = (50, 224, 224, 3) uint8 array.
    Label encoded in filename: *_label0.npy (normal) or *_label1.npy (crash).
    """
    def __init__(self, manifest_path: str, max_seq_len: int = 50,
                 augment: bool = False):
        with open(manifest_path) as f:
            self.entries = json.load(f)

        self.max_seq_len = max_seq_len
        self.augment = augment

        # ImageNet normalization
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        path  = entry["path"]
        label = entry["label"]

        # Load sequence: (T, H, W, C) uint8
        frames = np.load(path)  # (50, 224, 224, 3)

        # Subsample if needed
        T_len = min(frames.shape[0], self.max_seq_len)
        frames = frames[:T_len]

        # Convert to float tensor: (T, C, H, W) normalized
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)  # (T, 3, 224, 224)

        # Apply augmentation
        if self.augment:
            # Random horizontal flip (consistent across all frames)
            if torch.rand(1).item() > 0.5:
                frames = torch.flip(frames, dims=[3])
            # Color jitter - slight brightness/contrast
            brightness = 0.8 + 0.4 * torch.rand(1).item()
            frames = (frames * brightness).clamp(0, 1)

        # Normalize each frame
        for t in range(frames.shape[0]):
            frames[t] = self.normalize(frames[t])

        return frames, torch.tensor(label, dtype=torch.long)


# ─── CNN Backbone ─────────────────────────────────────────────────────────────
class ResNetBackbone(nn.Module):
    """ResNet18 feature extractor — outputs 512-dim features per frame."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        # Remove classification head, keep up to avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.out_dim = 512

    def forward(self, x):
        """
        Args:  x: (batch, T, 3, 224, 224)
        Returns: (batch, T, 512)
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)        # flatten batch & time
        x = self.features(x)               # (B*T, 512, 1, 1)
        x = x.view(B, T, self.out_dim)     # (B, T, 512)
        return x


# ─── Combined Model ─────────────────────────────────────────────────────────
class CrashPredictor(nn.Module):
    """ResNet18 + MSTT-CA crash predictor."""
    def __init__(self, d_model=256, n_heads=8, n_layers=4, dropout=0.1,
                 freeze_backbone_epochs: int = 5):
        super().__init__()
        self.backbone = ResNetBackbone(pretrained=True)
        self.mstt = MSTT_CA(
            input_dim=self.backbone.out_dim,  # 512
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            num_classes=2,
            short_window=8,
            medium_window=16,
            long_window=32,
        )
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self._backbone_frozen = False

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._backbone_frozen = True

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        self._backbone_frozen = False

    def forward(self, x, mc_samples=1):
        features = self.backbone(x)            # (B, T, 512)
        output = self.mstt(features, mc_samples=mc_samples)
        return output


# ─── Training Loop ───────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for frames, labels in pbar:
        frames = frames.to(device)  # (B, T, 3, 224, 224)
        labels = labels.to(device)  # (B,)

        optimizer.zero_grad()

        with autocast('cuda'):
            output = model(frames)
            logits = output["logits"]
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         acc=f"{100*correct/total:.1f}%")

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, mc_samples=1):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    for frames, labels in tqdm(loader, desc="Validating"):
        frames = frames.to(device)
        labels = labels.to(device)

        output = model(frames, mc_samples=mc_samples)
        logits = output["logits"]
        probs  = output["probabilities"]
        loss   = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    all_probs  = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    # Compute metrics
    preds = all_probs.argmax(dim=1)
    tp = ((preds == 1) & (all_labels == 1)).sum().item()
    fp = ((preds == 1) & (all_labels == 0)).sum().item()
    fn = ((preds == 0) & (all_labels == 1)).sum().item()
    tn = ((preds == 0) & (all_labels == 0)).sum().item()

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy  = correct / total

    metrics = {
        "loss": total_loss / total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train MSTT-CA Crash Predictor")
    parser.add_argument("--epochs",   type=int, default=50)
    parser.add_argument("--batch",    type=int, default=4,
                        help="Batch size (small for 50-frame sequences)")
    parser.add_argument("--lr",       type=float, default=3e-4)
    parser.add_argument("--device",   type=str, default="0")
    parser.add_argument("--resume",   type=str, default=None)
    parser.add_argument("--d_model",  type=int, default=256)
    parser.add_argument("--n_heads",  type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--freeze_epochs", type=int, default=5,
                        help="Epochs to freeze backbone (transfer learning)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if args.device.isdigit()
                          and torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  MSTT-CA Crash Predictor Training")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch}")
    print(f"  LR      : {args.lr}")
    print(f"  d_model : {args.d_model}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    manifest_dir = ROOT / "data" / "manifests"
    train_ds = CrashSequenceDataset(manifest_dir / "train.json", augment=True)
    val_ds   = CrashSequenceDataset(manifest_dir / "val.json",   augment=False)

    print(f"  Train : {len(train_ds)} sequences")
    print(f"  Val   : {len(val_ds)} sequences\n")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = CrashPredictor(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        freeze_backbone_epochs=args.freeze_epochs,
    ).to(device)

    # Freeze backbone initially (train only MSTT head)
    model.freeze_backbone()
    print("  🔒  Backbone frozen for first", args.freeze_epochs, "epochs\n")

    # Resume if provided
    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"  ▶  Resumed from epoch {start_epoch}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters : {total_params:,} total, {trainable:,} trainable\n")

    # ── Loss / Optimizer ──────────────────────────────────────────────────
    # FocalLoss handles class imbalance (all samples are crash=1 currently)
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = GradScaler('cuda')

    # ── Checkpointing ─────────────────────────────────────────────────────
    ckpt_dir = ROOT / "checkpoints" / "crash_predictor"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    best_epoch = 0
    history = []

    # ── Training ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        # Unfreeze backbone after warmup
        if epoch == args.freeze_epochs and model._backbone_frozen:
            model.unfreeze_backbone()
            # Re-create optimizer with all params
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch, eta_min=1e-6
            )
            trainable = sum(p.numel() for p in model.parameters()
                            if p.requires_grad)
            print(f"\n  🔓  Backbone unfrozen! Trainable params: {trainable:,}\n")

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch + 1
        )

        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "lr": lr,
            "time": elapsed,
        }
        history.append(record)

        print(f"\n  Epoch {epoch+1}/{args.epochs}  ({elapsed:.0f}s)  lr={lr:.2e}")
        print(f"    Train  : loss={train_loss:.4f}  acc={100*train_acc:.1f}%")
        print(f"    Val    : loss={val_metrics['loss']:.4f}  "
              f"acc={100*val_metrics['accuracy']:.1f}%  "
              f"F1={val_metrics['f1']:.4f}  "
              f"P={val_metrics['precision']:.3f}  R={val_metrics['recall']:.3f}")
        print(f"    Confusion: TP={val_metrics['tp']} FP={val_metrics['fp']} "
              f"FN={val_metrics['fn']} TN={val_metrics['tn']}")

        # Save best
        if val_metrics["f1"] >= best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch + 1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_f1,
                "val_metrics": val_metrics,
            }, ckpt_dir / "best.pt")
            print(f"    ★ New best F1={best_f1:.4f} saved!")

        # Save latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, ckpt_dir / "last.pt")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best F1   : {best_f1:.4f} (epoch {best_epoch})")
    print(f"  Weights   : {ckpt_dir / 'best.pt'}")
    print(f"{'='*60}\n")

    # Save history
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  📊  Training history → {ckpt_dir / 'history.json'}\n")


if __name__ == "__main__":
    main()
