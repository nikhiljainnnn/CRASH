"""
Phase 2: YOLOv8 Detection Training Script
==========================================
Trains YOLOv8s on KITTI + BDD100K for vehicle detection.
Then validates mAP on the validation set.

Usage:
  python scripts/train_detection.py
  python scripts/train_detection.py --epochs 50 --batch 16 --device cpu
  python scripts/train_detection.py --resume checkpoints/detection/last.pt
"""

import argparse
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
CKPT_DIR  = ROOT / "checkpoints" / "detection"
DATA_YAML = ROOT / "configs" / "dataset.yaml"

def train(args):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌  ultralytics not installed. Run: pip install ultralytics")
        return

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model (pretrained YOLOv8s)
    if args.resume and Path(args.resume).exists():
        print(f"▶  Resuming from {args.resume}")
        model = YOLO(args.resume)
    else:
        print("▶  Loading pretrained YOLOv8s weights")
        model = YOLO("yolov8s.pt")

    print(f"\n{'='*60}")
    print(f"  YOLOv8s Detection Training")
    print(f"  Dataset : {DATA_YAML}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch}")
    print(f"  Device  : {args.device}")
    print(f"  Output  : {CKPT_DIR}")
    print(f"{'='*60}\n")

    results = model.train(
        data       = str(DATA_YAML),
        epochs     = args.epochs,
        batch      = args.batch,
        imgsz      = 640,
        device     = args.device,
        project    = str(CKPT_DIR),
        name       = "run",
        exist_ok   = True,
        pretrained = True,
        optimizer  = "AdamW",
        lr0        = 0.001,
        lrf        = 0.01,
        momentum   = 0.937,
        weight_decay = 0.0005,
        warmup_epochs = 3,
        mosaic     = 1.0,
        mixup      = 0.15,
        flipud     = 0.0,
        fliplr     = 0.5,
        degrees    = 0.0,
        translate  = 0.1,
        scale      = 0.5,
        shear      = 0.0,
        perspective = 0.0,
        hsv_h      = 0.015,
        hsv_s      = 0.7,
        hsv_v      = 0.4,
        amp        = True,          # mixed precision
        save_period = 10,
        patience   = 20,
        seed       = 42,
        verbose    = True,
    )

    # ─── Validation ──────────────────────────────────────────────────────────
    print("\n📊  Running validation on best weights...")
    best_weights = CKPT_DIR / "run" / "weights" / "best.pt"
    if best_weights.exists():
        model = YOLO(str(best_weights))
        metrics = model.val(data=str(DATA_YAML), device=args.device, imgsz=640)
        print(f"\n  mAP@50       : {metrics.box.map50:.4f}")
        print(f"  mAP@50-95    : {metrics.box.map:.4f}")
        print(f"  Precision    : {metrics.box.mp:.4f}")
        print(f"  Recall       : {metrics.box.mr:.4f}")
    else:
        print(f"  ⚠️  Best weights not found at {best_weights}")

    print(f"\n  ✅  Detection training complete → {CKPT_DIR}/run/weights/best.pt\n")


def main():
    ap = argparse.ArgumentParser(description="Train YOLOv8 for vehicle detection")
    ap.add_argument("--epochs",  type=int,   default=100,  help="Training epochs")
    ap.add_argument("--batch",   type=int,   default=16,   help="Batch size")
    ap.add_argument("--device",  type=str,   default="0",  help="Device: 0 (GPU) or cpu")
    ap.add_argument("--resume",  type=str,   default=None, help="Resume from checkpoint")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
