"""
Phase 4 Ablation: Spatial Radius Impact on ST-GNN Performance
=============================================================
Tests radius = 15m, 30m, 50m with 20-epoch training each.
Saves results to checkpoints/st_gnn/ablation_results.json
"""
import sys, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_st_gnn import VehicleGraphDataset, SimpleSTGNN, validate

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x


def train_with_radius(radius, device, epochs=20):
    """Train ST-GNN with given spatial radius and return best metrics."""

    # Patch the model's radius
    class PatchedSTGNN(SimpleSTGNN):
        def _compute_edges(self, positions, velocities, headings, r=None):
            return super()._compute_edges(positions, velocities, headings,
                                          radius=radius)

    train_ds = VehicleGraphDataset(2000, 8, seed=42)
    val_ds   = VehicleGraphDataset(500, 8, seed=123)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    model = PatchedSTGNN(node_dim=16, edge_dim=8, hidden=128, heads=4, n_layers=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    best_f1 = 0
    best_metrics = {}

    for epoch in range(epochs):
        model.train()
        correct = total = 0
        for sequences, labels, risks in tqdm(train_loader,
                                             desc=f"R={radius}m E{epoch+1}",
                                             leave=False):
            for b in range(sequences.size(0)):
                seq = sequences[b].to(device)
                lbl = labels[b].to(device)
                risk_gt = risks[b].to(device)
                risk_pred, logits = model(seq)
                loss = F.cross_entropy(logits.unsqueeze(0), lbl.unsqueeze(0)) + \
                       0.5 * F.mse_loss(risk_pred.squeeze(), risk_gt)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                correct += (logits.argmax().item() == lbl.item())
                total += 1

        scheduler.step()
        val_m = validate(model, val_loader, device)

        if val_m["f1"] >= best_f1:
            best_f1 = val_m["f1"]
            best_metrics = {**val_m, "epoch": epoch + 1}

    return best_metrics


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    radii = [15, 30, 50]
    results = {}

    print(f"\n{'='*60}")
    print(f"  ST-GNN Spatial Radius Ablation Study")
    print(f"  Radii: {radii}m  |  20 epochs each  |  Device: {device}")
    print(f"{'='*60}\n")

    for r in radii:
        t0 = time.time()
        print(f"\n  ── Radius = {r}m ──────────────────────────────────")
        metrics = train_with_radius(r, device, epochs=20)
        elapsed = time.time() - t0
        results[f"{r}m"] = {**metrics, "time_min": elapsed / 60}
        print(f"    F1={metrics['f1']:.4f}  Acc={100*metrics['accuracy']:.1f}%  "
              f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
              f"RiskMSE={metrics['risk_mse']:.4f}  ({elapsed/60:.1f} min)")

    # Save results
    out = ROOT / "checkpoints" / "st_gnn" / "ablation_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Ablation Results Summary")
    print(f"{'='*60}")
    print(f"  {'Radius':<10} {'F1':<10} {'Acc':<10} {'P':<8} {'R':<8} {'RiskMSE':<10}")
    print(f"  {'─'*56}")
    for r, m in results.items():
        print(f"  {r:<10} {m['f1']:<10.4f} {100*m['accuracy']:<10.1f} "
              f"{m['precision']:<8.3f} {m['recall']:<8.3f} {m['risk_mse']:<10.4f}")
    print(f"\n  Results saved → {out}\n")


if __name__ == "__main__":
    main()
