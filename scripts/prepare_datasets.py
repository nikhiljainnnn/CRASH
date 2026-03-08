"""
Dataset Preparation Script for CRASH Detection System
======================================================
Handles all confirmed dataset paths found in Datasets/.

Dataset locations (actual paths verified):
  CCD crash frames  : Datasets/archive (10)/CrashBest/          (C_XXXXXX_XX.jpg)
  Normal-001 videos : Datasets/Normal-001/                       (.mp4)
  BDD100K images    : Datasets/archive (12)/bdd100k/bdd100k/images/
  BDD100K labels    : Datasets/archive (12)/labels/              (.json)
  KITTI images      : Datasets/data_object_image_2/training/image_2/
  KITTI labels      : Datasets/kitti_labels/training/            (KITTI format)
  KITTI YOLO labels : Datasets/yolo_format/                      (YOLO format)
  IDD-20K-II        : Datasets/idd-20k-II/idd20kII/
  US Accidents      : Datasets/archive (13)/US_Accidents_March23.csv
  DoTA metadata     : Datasets/Detection-of-Traffic-Anomaly-master/.../dataset/
  Extra videos      : Datasets/videos-20260206.../videos/        (1503 .mp4 clips)

Usage:
  python scripts/prepare_datasets.py --verify-only
  python scripts/prepare_datasets.py --stats
  python scripts/prepare_datasets.py --build-ccd
  python scripts/prepare_datasets.py --build-bdd
  python scripts/prepare_datasets.py --create-splits
  python scripts/prepare_datasets.py --all
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("  ⚠️  cv2 not installed — image reading disabled (pip install opencv-python)")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x   # no-op fallback

# ─── Actual confirmed dataset paths ──────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "Datasets"

# CCD – Car Crash Dataset (Kaggle: asefjamilajwad/car-crash-dataset-ccd)
CCD_CRASH_DIR  = DATASETS / "archive (10)" / "CrashBest"    # C_XXXXXX_XX.jpg
CCD_CSV        = DATASETS / "archive (10)" / "Crash_Table.csv"
NORMAL_VIDEOS  = DATASETS / "Normal-001"                     # 3000 normal .mp4

# BDD100K (Kaggle: awsaf49/bdd100k-dataset)
BDD_IMAGES     = DATASETS / "archive (12)" / "bdd100k" / "bdd100k" / "images"
BDD_LABELS_TR  = DATASETS / "archive (12)" / "labels" / "det_v2_train_release.json"
BDD_LABELS_VAL = DATASETS / "archive (12)" / "labels" / "det_v2_val_release.json"
BDD_TRAIN_CSV  = DATASETS / "archive (12)" / "train.csv"
BDD_VAL_CSV    = DATASETS / "archive (12)" / "val.csv"

# KITTI
KITTI_IMAGES   = DATASETS / "data_object_image_2" / "training" / "image_2"
KITTI_LABELS   = DATASETS / "data_object_label_2"
YOLO_LABELS    = DATASETS / "yolo_format"

# Others
IDD_DIR        = DATASETS / "idd-20k-II" / "idd20kII"
US_ACCIDENTS   = DATASETS / "archive (13)" / "US_Accidents_March23.csv"
DOTA_META_TR   = DATASETS / "Detection-of-Traffic-Anomaly-master" / \
                 "Detection-of-Traffic-Anomaly-master" / "dataset" / "metadata_train.json"
DOTA_META_VAL  = DATASETS / "Detection-of-Traffic-Anomaly-master" / \
                 "Detection-of-Traffic-Anomaly-master" / "dataset" / "metadata_val.json"
EXTRA_VIDEOS   = DATASETS / "videos-20260206T085121Z-1-002" / "videos"

# Output
PROCESSED      = ROOT / "data" / "processed"
SEQUENCES_DIR  = ROOT / "data" / "sequences"
TRAIN_DIR      = PROCESSED / "train"
VAL_DIR        = PROCESSED / "val"
TEST_DIR       = PROCESSED / "test"

# KITTI class names
CLASS_NAMES = ["car","truck","bus","motorcycle","bicycle","pedestrian",
               "person_sitting","cyclist","tram","misc"]


def _count(path: Path):
    if not path.exists():
        return None, None
    if path.is_file():
        return 1, path.stat().st_size
    n = sum(1 for f in path.rglob("*") if f.is_file())
    s = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return n, s


def _tag(path: Path, label: str):
    n, s = _count(path)
    if n is None:
        return f"  ❌  {label:<35}  NOT FOUND → {path.relative_to(ROOT)}"
    size_str = f"{s/1e9:.2f} GB" if s >= 1e8 else f"{s/1e6:.1f} MB"
    return f"  ✅  {label:<35}  [{n:>6,} files | {size_str}]"


# ─── Inventory ────────────────────────────────────────────────────────────────
def inventory():
    cols = [
        (CCD_CRASH_DIR,  "CCD Crash Frames"),
        (NORMAL_VIDEOS,  "Normal-001 Videos (MP4)"),
        (BDD_IMAGES,     "BDD100K Images"),
        (BDD_LABELS_TR,  "BDD100K Labels (train JSON)"),
        (KITTI_IMAGES,   "KITTI Images"),
        (YOLO_LABELS,    "KITTI YOLO Labels"),
        (IDD_DIR,        "IDD-20K-II"),
        (US_ACCIDENTS,   "US Accidents CSV"),
        (DOTA_META_TR,   "DoTA Metadata (train)"),
        (DOTA_META_VAL,  "DoTA Metadata (val)"),
        (EXTRA_VIDEOS,   "Extra Videos (~1503 clips)"),
    ]
    print("\n" + "=" * 68)
    print("  📦  CRASH PROJECT – CONFIRMED DATASET INVENTORY")
    print("=" * 68)
    missing = []
    for path, label in cols:
        line = _tag(path, label)
        print(line)
        if "❌" in line:
            missing.append(label)
    print("=" * 68)
    print(f"\n  Ready: {len(cols)-len(missing)}/{len(cols)}   Missing: {len(missing)}\n")
    return missing


# ─── KITTI stats ──────────────────────────────────────────────────────────────
def kitti_stats():
    counts = {n: 0 for n in CLASS_NAMES}
    label_files = list(YOLO_LABELS.rglob("*.txt")) if YOLO_LABELS.exists() else []
    for lf in label_files:
        for line in lf.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                cls = int(parts[0])
                if cls < len(CLASS_NAMES):
                    counts[CLASS_NAMES[cls]] += 1
    total = sum(counts.values())
    print(f"\n  📊  KITTI Class Distribution ({len(label_files):,} label files)")
    print("  " + "-" * 50)
    for name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(25 * cnt / max(total, 1))
        print(f"    {name:<18} {cnt:>7,}  {bar}")
    print(f"\n    Total objects : {total:,}\n")


# ─── CCD stats ────────────────────────────────────────────────────────────────
def ccd_stats():
    if not CCD_CRASH_DIR.exists():
        print("  ⚠️  CCD crash frames not found.\n")
        return None, None

    crash_frames = list(CCD_CRASH_DIR.glob("C_*.jpg"))
    crash_vids   = defaultdict(list)
    for f in crash_frames:
        vid_id = f.stem.split("_")[1]
        crash_vids[vid_id].append(f)

    print(f"\n  🎬  CCD Dataset")
    print(f"      Crash video sequences : {len(crash_vids):,}")
    print(f"      Crash frames total    : {len(crash_frames):,}")
    if crash_frames and CV2_AVAILABLE:
        img = cv2.imread(str(crash_frames[0]))
        if img is not None:
            print(f"      Frame resolution      : {img.shape[1]}×{img.shape[0]}")
    total_size = sum(f.stat().st_size for f in crash_frames)
    print(f"      Total size            : {total_size/1e9:.2f} GB\n")
    return crash_vids, None


# ─── BDD100K stats ────────────────────────────────────────────────────────────
def bdd_stats():
    if not BDD_LABELS_TR.exists():
        print("  ⚠️  BDD100K label JSON not found.\n")
        return

    print(f"\n  🚗  BDD100K Dataset")
    with open(BDD_LABELS_TR) as f:
        tr_data = json.load(f)
    print(f"      Train label entries   : {len(tr_data):,}")

    if BDD_LABELS_VAL.exists():
        with open(BDD_LABELS_VAL) as f:
            val_data = json.load(f)
        print(f"      Val   label entries   : {len(val_data):,}")

    # Count categories in train
    cats = defaultdict(int)
    for entry in tr_data:
        for lbl in entry.get("labels", []) or []:
            cats[lbl.get("category", "unknown")] += 1
    print(f"      Categories:")
    for c, n in sorted(cats.items(), key=lambda x: -x[1])[:8]:
        bar = "█" * min(int(n / max(cats.values()) * 20), 20)
        print(f"        {c:<20} {n:>7,}  {bar}")
    print()


# ─── DoTA stats ───────────────────────────────────────────────────────────────
def dota_stats():
    if not DOTA_META_TR.exists():
        return
    with open(DOTA_META_TR) as f:
        meta = json.load(f)
    cats = defaultdict(int)
    for v in meta.values():
        cats[v.get("accident_id", "unk")] += 1
    print(f"  🚦  DoTA Dataset")
    print(f"      Training sequences   : {len(meta):,}")
    if DOTA_META_VAL.exists():
        with open(DOTA_META_VAL) as f:
            mv = json.load(f)
        print(f"      Validation sequences : {len(mv):,}")
    for c, n in sorted(cats.items(), key=lambda x: -x[1])[:6]:
        print(f"      category {c}: {n}")
    print()


# ─── Build CCD temporal sequences ─────────────────────────────────────────────
def build_ccd_sequences(seq_len: int = 50, img_size: int = 224):
    """
    Group CCD frames by video ID → save as (T,H,W,C) numpy arrays.
    RESUME SAFE: skips video IDs that already have a saved .npy file.
    DISK GUARD : stops if < 500 MB free to avoid OSError mid-write.
    """
    if not CV2_AVAILABLE:
        print("  ❌  cv2 required. pip install opencv-python")
        return

    SEQUENCES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Collect already-saved video IDs (resume support) ──────────────────
    done_ids = set()
    for f in SEQUENCES_DIR.glob("crash_*_label1.npy"):
        # filename pattern: crash_000001_label1.npy
        parts = f.stem.split("_")   # ['crash', '000001', 'label1']
        if len(parts) >= 2:
            done_ids.add(parts[1])
    if done_ids:
        print(f"  ↩️   Resume: {len(done_ids)} sequences already saved, skipping them.\n")

    result = ccd_stats()
    if result[0] is None:
        return
    crash_vids = result[0]

    MB_PER_SEQ = (seq_len * img_size * img_size * 3) / 1e6   # uncompressed MB
    saved = skipped = 0

    for vid_id, frames in tqdm(crash_vids.items(), desc="CCD crash sequences"):
        # ── Skip already done ──────────────────────────────────────────────
        if vid_id in done_ids:
            skipped += 1
            continue

        # ── Disk space guard (warn if < 500 MB free) ──────────────────────
        free_mb = shutil.disk_usage(SEQUENCES_DIR).free / 1e6
        if free_mb < 500:
            print(f"\n  ⚠️   Only {free_mb:.0f} MB free — stopping to avoid disk full.")
            print(f"       Free more space and re-run; resume will pick up from here.")
            break

        frames_sorted = sorted(frames, key=lambda f: f.stem)
        imgs = []
        for fp in frames_sorted[:seq_len]:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
        if len(imgs) < 10:
            continue
        while len(imgs) < seq_len:
            imgs.append(imgs[-1])
        arr = np.array(imgs[:seq_len], dtype=np.uint8)
        out = SEQUENCES_DIR / f"crash_{vid_id}_label1.npy"
        np.save(str(out), arr)
        saved += 1

    total_done = len(done_ids) + saved
    print(f"\n  ✅  CCD crash sequences: {saved} new + {skipped} resumed = {total_done} total"
          f" → {SEQUENCES_DIR}")

    # ── Normal sequences from extra videos ────────────────────────────────
    if EXTRA_VIDEOS.exists():
        done_normal = {f.stem.split("_label")[0].replace("normal_","")
                       for f in SEQUENCES_DIR.glob("normal_*_label0.npy")}
        normal_vids = [v for v in list(EXTRA_VIDEOS.glob("*.mp4"))[:500]
                       if v.stem not in done_normal]

        if not normal_vids:
            print("  ↩️   All extra-video normal sequences already saved.")
        else:
            n_saved = 0
            for vid_path in tqdm(normal_vids, desc="Normal extra-video sequences"):
                free_mb = shutil.disk_usage(SEQUENCES_DIR).free / 1e6
                if free_mb < 500:
                    print(f"\n  ⚠️  Only {free_mb:.0f} MB free — stopping early.")
                    break
                cap = cv2.VideoCapture(str(vid_path))
                orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                skip = max(1, int(orig_fps / 10))
                frames = []
                idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if idx % skip == 0:
                        frame = cv2.resize(frame, (img_size, img_size))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)
                        if len(frames) >= seq_len:
                            break
                    idx += 1
                cap.release()
                if len(frames) < 10:
                    continue
                while len(frames) < seq_len:
                    frames.append(frames[-1])
                arr = np.array(frames[:seq_len], dtype=np.uint8)
                out = SEQUENCES_DIR / f"normal_{vid_path.stem}_label0.npy"
                np.save(str(out), arr)
                n_saved += 1
            print(f"  ✅  Extra Videos: {n_saved} normal sequences saved")


# ─── Build BDD100K YOLO labels ────────────────────────────────────────────────
def build_bdd_yolo(out_dir: Path = ROOT / "data" / "bdd_yolo"):
    """Convert BDD100K detection JSON to YOLO .txt label format."""
    if not BDD_LABELS_TR.exists():
        print("  ❌  BDD100K labels not found.\n")
        return

    BDD_CATEGORY_MAP = {
        "car": 0, "truck": 1, "bus": 2, "motor": 3, "bike": 4,
        "person": 5, "rider": 5, "traffic light": 6, "traffic sign": 7, "train": 8
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(BDD_LABELS_TR) as f:
        data = json.load(f)

    converted = 0
    for entry in tqdm(data, desc="BDD100K → YOLO"):
        name  = Path(entry.get("name", "")).stem
        labels = entry.get("labels") or []
        lines = []
        for lbl in labels:
            cat   = lbl.get("category", "").lower()
            cls   = BDD_CATEGORY_MAP.get(cat)
            if cls is None:
                continue
            box2d = lbl.get("box2d")
            if not box2d:
                continue
            x1, y1, x2, y2 = box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]
            W, H = 1280, 720   # BDD100K standard frame size
            cx = ((x1 + x2) / 2) / W
            cy = ((y1 + y2) / 2) / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if lines:
            (out_dir / f"{name}.txt").write_text("\n".join(lines))
            converted += 1

    print(f"\n  ✅  BDD100K: {converted:,} YOLO label files → {out_dir}")


# ─── Train/val/test splits (manifest-based, zero extra disk) ─────────────────
def create_splits(train=0.70, val=0.15, seed=42):
    """
    Write JSON manifest files instead of copying .npy data.
    Each manifest lists absolute paths the dataloader reads directly.
    Saves: data/manifests/train.json, val.json, test.json
    """
    MANIFEST_DIR = ROOT / "data" / "manifests"

    if not SEQUENCES_DIR.exists() or not any(SEQUENCES_DIR.glob("*.npy")):
        print("  ⚠️  No sequences found. Run --build-ccd first.\n")
        return

    np.random.seed(seed)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    all_entries = []
    for label in [0, 1]:
        seqs = sorted(SEQUENCES_DIR.glob(f"*_label{label}.npy"))
        for s in seqs:
            all_entries.append({"path": str(s.resolve()), "label": label})

    np.random.shuffle(all_entries)
    n = len(all_entries)
    n_tr = int(n * train)
    n_va = int(n * val)

    splits = {
        "train": all_entries[:n_tr],
        "val":   all_entries[n_tr:n_tr+n_va],
        "test":  all_entries[n_tr+n_va:],
    }

    print("\n  📑  Dataset splits (manifest files — no data copied):")
    for name, entries in splits.items():
        out = MANIFEST_DIR / f"{name}.json"
        import json as _json
        out.write_text(_json.dumps(entries, indent=2))
        c0 = sum(1 for e in entries if e["label"] == 0)
        c1 = sum(1 for e in entries if e["label"] == 1)
        print(f"      {name:<8}: {len(entries):4} samples  (normal={c0}, crash={c1})  → {out.name}")
    print(f"\n  ✅  Manifests saved in {MANIFEST_DIR}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="CRASH – Dataset Preparation")
    ap.add_argument("--verify-only",   action="store_true")
    ap.add_argument("--stats",         action="store_true")
    ap.add_argument("--build-ccd",     action="store_true", help="Build CCD sequences")
    ap.add_argument("--build-bdd",     action="store_true", help="Convert BDD100K to YOLO")
    ap.add_argument("--create-splits", action="store_true")
    ap.add_argument("--all",           action="store_true", help="Run full pipeline")
    args = ap.parse_args()

    missing = inventory()

    if args.verify_only:
        sys.exit(0 if not missing else 1)

    if args.stats or args.all:
        kitti_stats()
        ccd_stats()
        bdd_stats()
        dota_stats()

    if args.build_ccd or args.all:
        build_ccd_sequences()

    if args.build_bdd or args.all:
        build_bdd_yolo()

    if args.create_splits or args.all:
        create_splits()

    if not any([args.stats, args.build_ccd, args.build_bdd,
                args.create_splits, args.all]):
        print("  Tip: run with --all to execute full pipeline, or use:")
        print("       --stats | --build-ccd | --build-bdd | --create-splits\n")


if __name__ == "__main__":
    main()
