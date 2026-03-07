"""
Dataset Preparation Script for CRASH Detection System
Handles all existing datasets and CCD Kaggle format.

CCD Kaggle dataset (asefjamilajwad/car-crash-dataset-ccd):
  - Frames pre-extracted at 50 frames/video
  - Naming: C_000001_01.jpg (crash), N_000001_01.jpg (normal)
  - Place downloaded folder in: Datasets/ccd/

Download CCD via Kaggle API:
  pip install kaggle
  kaggle datasets download -d asefjamilajwad/car-crash-dataset-ccd -p Datasets/ccd --unzip

Usage:
  python scripts/prepare_datasets.py --verify-only   # inventory check
  python scripts/prepare_datasets.py --stats         # print stats
  python scripts/prepare_datasets.py --build-ccd     # build sequences from CCD frames
  python scripts/prepare_datasets.py --all           # full pipeline
"""

import os
import sys
import json
import shutil
import argparse
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "Datasets"

# Source dataset paths
KITTI_IMAGES  = DATASETS / "data_object_image_2" / "training" / "image_2"
YOLO_LABELS   = DATASETS / "yolo_format"
NORMAL_VIDEOS = DATASETS / "Normal-001"                 # 3000 normal .mp4 clips
IDD_DIR       = DATASETS / "idd-20k-II" / "idd20kII"
US_ACCIDENTS  = DATASETS / "archive (13)" / "US_Accidents_March23.csv"
DOTA_DIR      = DATASETS / "Detection-of-Traffic-Anomaly-master" / \
                "Detection-of-Traffic-Anomaly-master" / "dataset"
DOTA_META_TR  = DOTA_DIR / "metadata_train.json"
DOTA_META_VAL = DOTA_DIR / "metadata_val.json"

# CCD Kaggle dataset — frames extracted, C_*/N_* naming convention
CCD_DIR       = DATASETS / "ccd"        # place downloaded "ccd" folder here

# Output
PROCESSED     = ROOT / "data" / "processed"
SEQUENCES_DIR = ROOT / "data" / "sequences"    # .npy temporal sequences
TRAIN_DIR     = PROCESSED / "train"
VAL_DIR       = PROCESSED / "val"
TEST_DIR      = PROCESSED / "test"

CLASS_NAMES   = ["car","truck","bus","motorcycle","bicycle","pedestrian",
                 "person_sitting","cyclist","tram","misc"]


# ─── Dataset inventory ────────────────────────────────────────────────────────
def inventory():
    """Print complete dataset inventory and return list of missing items."""
    checks = {
        "KITTI Images"            : KITTI_IMAGES,
        "KITTI YOLO Labels"       : YOLO_LABELS,
        "Normal-001 videos"        : NORMAL_VIDEOS,
        "IDD-20K-II"              : IDD_DIR,
        "US Accidents CSV"        : US_ACCIDENTS,
        "DoTA Metadata (train)"   : DOTA_META_TR,
        "DoTA Metadata (val)"      : DOTA_META_VAL,
        "CCD Frames (Kaggle)"     : CCD_DIR,
    }

    print("\n" + "=" * 62)
    print("  📦  CRASH DATASET INVENTORY")
    print("=" * 62)

    found, missing = [], []
    for name, path in checks.items():
        if path.exists():
            if path.is_dir():
                count = sum(1 for f in path.rglob("*") if f.is_file())
                size  = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                tag   = f"  [{count:>6,} files | {size/1e9:.2f} GB]"
            else:
                size  = path.stat().st_size
                tag   = f"  [{size/1e9:.2f} GB]"
            print(f"  ✅  {name:<30}{tag}")
            found.append(name)
        else:
            print(f"  ❌  {name:<30}  NOT FOUND → {path.relative_to(ROOT)}")
            missing.append(name)

    print("=" * 62)
    print(f"\n  Ready: {len(found)}/8   Missing: {len(missing)}/8")

    if "CCD Frames (Kaggle)" in missing:
        print("\n  📥  To download CCD dataset:")
        print("      pip install kaggle")
        print("      # Place kaggle.json in ~/.kaggle/")
        print("      kaggle datasets download -d asefjamilajwad/car-crash-dataset-ccd \\")
        print(f"        -p \"{CCD_DIR}\" --unzip")
        print(f"\n  Or download manually: https://www.kaggle.com/datasets/asefjamilajwad/car-crash-dataset-ccd")
    print()
    return missing


# ─── CCD frame analysis ───────────────────────────────────────────────────────
def ccd_stats():
    """Analyse CCD frame naming convention and print stats."""
    if not CCD_DIR.exists():
        print("  ⚠️  CCD not found. Download first.\n")
        return

    crash_frames  = list(CCD_DIR.rglob("C_*.jpg"))
    normal_frames = list(CCD_DIR.rglob("N_*.jpg"))

    # Group by video ID
    crash_vids  = defaultdict(list)
    normal_vids = defaultdict(list)
    for f in crash_frames:
        vid_id = f.stem.split("_")[1]   # C_000001_01 → 000001
        crash_vids[vid_id].append(f)
    for f in normal_frames:
        vid_id = f.stem.split("_")[1]
        normal_vids[vid_id].append(f)

    print(f"\n  🎬  CCD Dataset (Kaggle)")
    print(f"      Crash  videos : {len(crash_vids):,}  ({len(crash_frames):,} frames)")
    print(f"      Normal videos : {len(normal_vids):,}  ({len(normal_frames):,} frames)")
    print(f"      Frames/video  : {len(crash_frames)//max(len(crash_vids),1)}")
    if crash_frames:
        img = cv2.imread(str(crash_frames[0]))
        if img is not None:
            print(f"      Frame size    : {img.shape[1]}×{img.shape[0]}")
    return crash_vids, normal_vids


# ─── KITTI stats ──────────────────────────────────────────────────────────────
def kitti_stats():
    counts = {n: 0 for n in CLASS_NAMES}
    label_files = list(YOLO_LABELS.rglob("*.txt"))
    for lf in label_files:
        for line in lf.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                cls = int(parts[0])
                if cls < len(CLASS_NAMES):
                    counts[CLASS_NAMES[cls]] += 1
    total = sum(counts.values())
    print(f"\n  📊  KITTI Class Distribution ({len(label_files):,} label files)")
    print("  " + "-" * 45)
    for name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(25 * cnt / max(total, 1))
        print(f"    {name:<18} {cnt:>6,}  {bar}")
    print(f"\n    Total objects : {total:,}\n")


# ─── DoTA stats ───────────────────────────────────────────────────────────────
def dota_stats():
    if not DOTA_META_TR.exists():
        return
    with open(DOTA_META_TR) as f:
        meta = json.load(f)
    cats = defaultdict(int)
    for v in meta.values():
        cats[v.get("accident_id", "unknown")] += 1
    print(f"\n  🚦  DoTA Dataset")
    print(f"      Training sequences   : {len(meta):,}")
    if DOTA_META_VAL.exists():
        with open(DOTA_META_VAL) as f:
            mv = json.load(f)
        print(f"      Validation sequences : {len(mv):,}")
    for c, n in sorted(cats.items(), key=lambda x: -x[1])[:10]:
        print(f"      category {c}: {n}")
    print()


# ─── Build CCD temporal sequences ─────────────────────────────────────────────
def build_ccd_sequences(seq_len: int = 50, img_size: int = 224):
    """
    Group CCD frames into temporal sequences per video.
    Saves as numpy arrays: (T, H, W, C).

    CCD has 50 frames per video → 1 sequence per video.
    """
    SEQUENCES_DIR.mkdir(parents=True, exist_ok=True)

    if not CCD_DIR.exists():
        print("  ❌  CCD not found. Cannot build sequences.\n")
        return

    result = ccd_stats()
    if not result:
        return
    crash_vids, normal_vids = result

    def save_sequences(vid_dict, label, desc):
        saved = 0
        for vid_id, frames in tqdm(vid_dict.items(), desc=desc):
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
            # Pad to seq_len
            while len(imgs) < seq_len:
                imgs.append(imgs[-1])
            arr = np.array(imgs[:seq_len], dtype=np.uint8)   # (T, H, W, C)
            out = SEQUENCES_DIR / f"{desc.lower()}_{vid_id}_label{label}.npy"
            np.save(str(out), arr)
            saved += 1
        print(f"      Saved {saved} sequences → {SEQUENCES_DIR}")

    save_sequences(crash_vids,  1, "crash")
    save_sequences(normal_vids, 0, "normal")


# ─── Train/val/test split ─────────────────────────────────────────────────────
def create_splits(train=0.70, val=0.15, seed=42):
    np.random.seed(seed)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    for label in [0, 1]:
        seqs = sorted(SEQUENCES_DIR.glob(f"*_label{label}.npy"))
        idxs = np.arange(len(seqs))
        np.random.shuffle(idxs)
        n_tr = int(len(seqs) * train)
        n_va = int(len(seqs) * val)
        splits = {
            TRAIN_DIR: idxs[:n_tr],
            VAL_DIR:   idxs[n_tr:n_tr+n_va],
            TEST_DIR:  idxs[n_tr+n_va:],
        }
        for out_dir, idx in splits.items():
            for i in idx:
                shutil.copy(seqs[i], out_dir / seqs[i].name)

    print(f"\n  📑  Splits created:")
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        print(f"      {d.name:<8}: {sum(1 for _ in d.glob('*.npy'))} sequences")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Prepare CRASH datasets")
    ap.add_argument("--verify-only",   action="store_true", help="Inventory check only")
    ap.add_argument("--stats",         action="store_true", help="Print dataset statistics")
    ap.add_argument("--build-ccd",     action="store_true", help="Build sequences from CCD frames")
    ap.add_argument("--create-splits", action="store_true", help="Train/val/test split")
    ap.add_argument("--all",           action="store_true", help="Full pipeline")
    args = ap.parse_args()

    missing = inventory()

    if args.verify_only:
        sys.exit(0 if not missing else 1)

    if args.stats or args.all:
        kitti_stats()
        dota_stats()
        ccd_stats()

    if args.build_ccd or args.all:
        build_ccd_sequences()

    if args.create_splits or args.all:
        create_splits()

    if not any([args.stats, args.build_ccd, args.create_splits, args.all]):
        print("  Tip: run with --all to execute full pipeline.")
        print("       Or use specific flags: --stats | --build-ccd | --create-splits")


if __name__ == "__main__":
    main()
