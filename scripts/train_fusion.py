"""
Phase 5: Multimodal Fusion & Bayesian Risk — CrashPredictionSystem
===================================================================
Combines all three sub-models into a unified crash prediction system:
  1. ResNet18 backbone → visual features
  2. MSTT-CA → temporal crash probability
  3. SimpleSTGNN → graph-based risk scores

Fusion via learned attention gate + MC Dropout for Bayesian uncertainty.

Usage:
  python scripts/train_fusion.py
  python scripts/train_fusion.py --epochs 40 --batch 4 --mc_samples 10
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.temporal.mstt_transformer import MSTT_CA
from scripts.train_st_gnn import SimpleSTGNN

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x


# ─── Fusion Dataset ─────────────────────────────────────────────────────────
class FusionDataset(Dataset):
    """
    Loads CCD sequences from manifests and generates synthetic vehicle
    graphs for each sequence. Provides both modalities for fusion training.
    """
    def __init__(self, manifest_path: str, num_vehicles: int = 6,
                 augment: bool = False, seed: int = 42):
        import torchvision.transforms as T

        with open(manifest_path) as f:
            self.entries = json.load(f)

        self.num_vehicles = num_vehicles
        self.augment = augment
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.entries)

    def _generate_vehicle_graph(self, is_crash: bool, seq_len: int = 10):
        """Generate synthetic vehicle interaction sequence matching the label."""
        N = self.num_vehicles
        T = seq_len

        positions = self.rng.randn(N, 2) * 20
        velocities = self.rng.randn(N, 2) * 5 + np.array([10.0, 0.0])
        accelerations = self.rng.randn(N, 2) * 0.5

        if is_crash:
            positions[1] = positions[0] + np.array([15.0, 3.0])
            velocities[0] = np.array([12.0, 0.5])
            velocities[1] = np.array([-8.0, -0.3])

        frames = []
        for t in range(T):
            positions = positions + velocities * 0.1
            velocities = velocities + accelerations * 0.1 + self.rng.randn(N, 2) * 0.1
            headings = np.arctan2(velocities[:, 1], velocities[:, 0])
            speed = np.linalg.norm(velocities, axis=1, keepdims=True)

            features = np.concatenate([
                positions, velocities, accelerations,
                headings[:, None], self.rng.uniform(3.5, 5.5, (N, 2)),
                self.rng.randint(0, 6, (N, 1)).astype(float),
                speed, self.rng.randn(N, 1) * 0.05,
                np.linalg.norm(positions, axis=1, keepdims=True) / 50.0,
                self.rng.randn(N, 3) * 0.1
            ], axis=1).astype(np.float32)

            frames.append(features)

        return np.stack(frames)  # (T, N, 16)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        label = entry["label"]

        # Load visual sequence
        frames = np.load(entry["path"])  # (50, 224, 224, 3)
        T_len = min(frames.shape[0], 50)
        frames = frames[:T_len]

        frames_t = torch.from_numpy(frames).float() / 255.0
        frames_t = frames_t.permute(0, 3, 1, 2)  # (T, 3, 224, 224)

        if self.augment:
            if torch.rand(1).item() > 0.5:
                frames_t = torch.flip(frames_t, dims=[3])
            brightness = 0.8 + 0.4 * torch.rand(1).item()
            frames_t = (frames_t * brightness).clamp(0, 1)

        for t in range(frames_t.shape[0]):
            frames_t[t] = self.normalize(frames_t[t])

        # Generate vehicle graph
        graph_seq = self._generate_vehicle_graph(is_crash=(label == 1))
        graph_t = torch.from_numpy(graph_seq)  # (10, N, 16)

        return frames_t, graph_t, torch.tensor(label, dtype=torch.long)


# ─── Fusion Model ───────────────────────────────────────────────────────────
class ResNetBackbone(nn.Module):
    """ResNet18 feature extractor."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.out_dim = 512

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.features(x)
        return x.view(B, T, self.out_dim)


class CrashPredictionSystem(nn.Module):
    """
    Unified Crash Prediction System — Multimodal Fusion with Bayesian Risk.

    Components:
      1. ResNet18 → frame-level visual features (512-dim)
      2. MSTT-CA  → temporal crash probability from visual stream
      3. SimpleSTGNN → graph-based vehicle interaction risk
      4. Attention fusion gate → combine all modality outputs
      5. MC Dropout → Bayesian uncertainty estimation

    Output: crash probability, risk score, uncertainty estimate
    """
    def __init__(self, d_model=256, n_heads=8, n_layers=4, dropout=0.15,
                 num_vehicles=6, freeze_backbone=True):
        super().__init__()

        # ── Sub-model 1: Visual backbone ──────────────────────────────
        self.backbone = ResNetBackbone(pretrained=True)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ── Sub-model 2: Temporal transformer ─────────────────────────
        self.mstt = MSTT_CA(
            input_dim=512, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, dropout=dropout, num_classes=2,
            short_window=8, medium_window=16, long_window=32,
        )

        # ── Sub-model 3: Graph neural network ─────────────────────────
        self.st_gnn = SimpleSTGNN(
            node_dim=16, edge_dim=8, hidden=128, heads=4,
            n_layers=3, dropout=dropout
        )

        # ── Fusion layers ─────────────────────────────────────────────
        # Inputs: MSTT logits (2) + GNN logits (2) + GNN risk (1) = 5
        self.fusion_gate = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3),
            nn.Softmax(dim=-1),
        )

        # Final classifier from fused features
        self.final_head = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

        # MC Dropout for uncertainty
        self.mc_dropout = nn.Dropout(0.2)

    def forward(self, visual_seq, graph_seq, mc_samples=1):
        """
        Args:
            visual_seq: (B, T, 3, 224, 224) - Video frames
            graph_seq:  (B, T_g, N, 16) - Vehicle graph sequence
            mc_samples: int - MC Dropout samples for uncertainty

        Returns: dict with logits, probabilities, uncertainty, fusion_weights
        """
        B = visual_seq.size(0)

        # ── Visual backbone → MSTT-CA ────────────────────────────────
        with torch.no_grad() if not any(p.requires_grad for p in self.backbone.parameters()) else torch.enable_grad():
            visual_features = self.backbone(visual_seq)  # (B, T, 512)

        mstt_out = self.mstt(visual_features)
        mstt_logits = mstt_out["logits"]  # (B, 2)

        # ── Graph GNN ─────────────────────────────────────────────────
        gnn_risks = []
        gnn_logits = []
        for b in range(B):
            risk, logits = self.st_gnn(graph_seq[b])  # risk: (N,1), logits: (2,)
            gnn_risks.append(risk.mean())  # mean risk across vehicles
            gnn_logits.append(logits)

        gnn_risk = torch.stack(gnn_risks).unsqueeze(-1)   # (B, 1)
        gnn_logits = torch.stack(gnn_logits)               # (B, 2)

        # ── Fusion ────────────────────────────────────────────────────
        # Combine all signals
        combined = torch.cat([mstt_logits, gnn_logits, gnn_risk], dim=-1)  # (B, 5)
        fusion_weights = self.fusion_gate(combined)  # (B, 3)

        # MC Dropout for Bayesian uncertainty
        if mc_samples > 1:
            all_probs = []
            for _ in range(mc_samples):
                dropped = self.mc_dropout(combined)
                logits = self.final_head(dropped)
                all_probs.append(F.softmax(logits, dim=-1))

            probs = torch.stack(all_probs)  # (S, B, 2)
            mean_probs = probs.mean(dim=0)
            uncertainty = probs.std(dim=0).mean(dim=-1)  # (B,)
            logits = torch.log(mean_probs + 1e-8)
        else:
            dropped = self.mc_dropout(combined)
            logits = self.final_head(dropped)
            mean_probs = F.softmax(logits, dim=-1)
            uncertainty = None

        output = {
            "logits": logits,
            "probabilities": mean_probs,
            "fusion_weights": fusion_weights,
            "mstt_logits": mstt_logits,
            "gnn_logits": gnn_logits,
            "gnn_risk": gnn_risk,
        }
        if uncertainty is not None:
            output["uncertainty"] = uncertainty

        return output


# ─── Training ────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = correct = total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for vis, graph, labels in pbar:
        vis    = vis.to(device)
        graph  = graph.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with autocast('cuda'):
            output = model(vis, graph)
            logits = output["logits"]
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.1f}%")

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, mc_samples=10):
    model.train()  # Keep dropout active for MC sampling
    tp = fp = fn = tn = 0
    total_loss = total = 0
    uncertainties = []

    for vis, graph, labels in tqdm(loader, desc="Validating"):
        vis    = vis.to(device)
        graph  = graph.to(device)
        labels = labels.to(device)

        output = model(vis, graph, mc_samples=mc_samples)
        logits = output["logits"]
        loss   = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(1)

        for p, l in zip(preds, labels):
            if p == 1 and l == 1: tp += 1
            elif p == 1 and l == 0: fp += 1
            elif p == 0 and l == 1: fn += 1
            else: tn += 1
        total += labels.size(0)

        if "uncertainty" in output:
            uncertainties.extend(output["uncertainty"].cpu().tolist())

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    metrics = {
        "loss": total_loss / total,
        "accuracy": (tp + tn) / total,
        "f1": f1, "precision": precision, "recall": recall,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    if uncertainties:
        metrics["mean_uncertainty"] = np.mean(uncertainties)
        metrics["std_uncertainty"]  = np.std(uncertainties)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Fusion CrashPredictionSystem")
    parser.add_argument("--epochs",     type=int, default=40)
    parser.add_argument("--batch",      type=int, default=4)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--device",     type=str, default="0")
    parser.add_argument("--mc_samples", type=int, default=10)
    parser.add_argument("--unfreeze_epoch", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if args.device.isdigit()
                          and torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  CrashPredictionSystem — Multimodal Fusion Training")
    print(f"  Device      : {device}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  MC Samples  : {args.mc_samples}")
    print(f"  Unfreeze BB : epoch {args.unfreeze_epoch}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    manifest_dir = ROOT / "data" / "manifests"
    train_ds = FusionDataset(manifest_dir / "train.json", augment=True, seed=42)
    val_ds   = FusionDataset(manifest_dir / "val.json", augment=False, seed=123)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}\n")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = CrashPredictionSystem(
        d_model=256, n_heads=8, n_layers=4, dropout=0.15, freeze_backbone=True
    ).to(device)

    # Load pretrained sub-model weights if available
    mstt_ckpt = ROOT / "checkpoints" / "crash_predictor" / "best.pt"
    if mstt_ckpt.exists():
        ckpt = torch.load(mstt_ckpt, map_location=device, weights_only=False)
        # Load only MSTT weights (skip backbone since we re-init it)
        mstt_state = {k.replace("mstt.", ""): v
                      for k, v in ckpt["model_state_dict"].items()
                      if k.startswith("mstt.")}
        model.mstt.load_state_dict(mstt_state, strict=False)
        print("  ✅  Loaded pretrained MSTT-CA weights\n")

    gnn_ckpt = ROOT / "checkpoints" / "st_gnn" / "best.pt"
    if gnn_ckpt.exists():
        ckpt = torch.load(gnn_ckpt, map_location=device, weights_only=False)
        model.st_gnn.load_state_dict(ckpt["model_state_dict"], strict=False)
        print("  ✅  Loaded pretrained ST-GNN weights\n")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total:,} total, {trainable:,} trainable\n")

    # ── Training setup ────────────────────────────────────────────────────
    from models.utils.losses import FocalLoss
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = GradScaler('cuda')

    ckpt_dir = ROOT / "checkpoints" / "fusion"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0
    history = []

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        # Unfreeze backbone after warmup
        if epoch == args.unfreeze_epoch:
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch, eta_min=1e-6
            )
            t = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n  🔓  Backbone unfrozen! Trainable: {t:,}\n")

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch + 1
        )
        val_m = validate(model, val_loader, criterion, device, args.mc_samples)
        scheduler.step()
        elapsed = time.time() - t0

        record = {
            "epoch": epoch + 1, "train_loss": train_loss,
            "train_acc": train_acc, **val_m, "time": elapsed,
        }
        history.append(record)

        unc_str = (f"  Unc={val_m['mean_uncertainty']:.4f}±{val_m['std_uncertainty']:.4f}"
                   if "mean_uncertainty" in val_m else "")

        print(f"\n  Epoch {epoch+1}/{args.epochs}  ({elapsed:.0f}s)")
        print(f"    Train : loss={train_loss:.4f}  acc={100*train_acc:.1f}%")
        print(f"    Val   : acc={100*val_m['accuracy']:.1f}%  F1={val_m['f1']:.4f}  "
              f"P={val_m['precision']:.3f}  R={val_m['recall']:.3f}{unc_str}")
        print(f"    TP={val_m['tp']} FP={val_m['fp']} FN={val_m['fn']} TN={val_m['tn']}")

        if val_m["f1"] >= best_f1:
            best_f1 = val_m["f1"]
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_f1": best_f1, "val_metrics": val_m,
            }, ckpt_dir / "best.pt")
            print(f"    ★ Best F1={best_f1:.4f} saved!")

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
                   ckpt_dir / "last.pt")

    # ── Latency benchmark ─────────────────────────────────────────────────
    model.eval()
    dummy_vis   = torch.randn(1, 50, 3, 224, 224).to(device)
    dummy_graph = torch.randn(1, 10, 6, 16).to(device)

    # Warmup
    for _ in range(5):
        model(dummy_vis, dummy_graph)

    latencies = []
    for _ in range(50):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(dummy_vis, dummy_graph, mc_samples=args.mc_samples)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    mean_lat = np.mean(latencies)
    p95_lat  = np.percentile(latencies, 95)

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best F1       : {best_f1:.4f}")
    print(f"  Latency (mean): {mean_lat:.1f} ms")
    print(f"  Latency (p95) : {p95_lat:.1f} ms")
    print(f"  Target (<100ms): {'✅ PASS' if mean_lat < 100 else '❌ FAIL'}")
    print(f"  Weights       : {ckpt_dir / 'best.pt'}")
    print(f"{'='*60}\n")

    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save latency results
    with open(ckpt_dir / "latency.json", "w") as f:
        json.dump({"mean_ms": mean_lat, "p95_ms": p95_lat,
                    "p99_ms": float(np.percentile(latencies, 99)),
                    "all_ms": latencies}, f, indent=2)


if __name__ == "__main__":
    main()
