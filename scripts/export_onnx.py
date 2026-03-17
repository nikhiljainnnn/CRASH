"""
Export PyTorch CrashPredictionSystem to ONNX
=============================================
Converts the trained PyTorch fusion model to an optimized ONNX format.
ONNX serves as an intermediate step for TensorRT compilation.

Usage:
  python scripts/export_onnx.py --ckpt checkpoints/fusion/best.pt
"""

import os
import sys
import argparse
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_fusion import CrashPredictionSystem

def main():
    parser = argparse.ArgumentParser(description="Export Model to ONNX")
    parser.add_argument("--ckpt", type=str, default="checkpoints/fusion/best.pt")
    parser.add_argument("--output", type=str, default="checkpoints/fusion/model.onnx")
    parser.add_argument("--opset", type=int, default=16)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n==========================================")
    print(f"  ONNX Exporter")
    print(f"  Input  : {args.ckpt}")
    print(f"  Output : {args.output}")
    print(f"  Device : {device}")
    print(f"==========================================\n")

    # Load Model
    print("1. Loading PyTorch model...")
    model = CrashPredictionSystem().to(device)
    
    if not Path(args.ckpt).exists():
        print(f"❌ Error: Checkpoint not found at {args.ckpt}")
        sys.exit(1)
        
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    
    # Needs to be in eval mode for export
    model.eval()

    # Create dummy inputs that match what the model expects
    # visual_seq: (Batch, Frames, C, H, W)
    # graph_seq: (Batch, Frames, NumVehicles, Features)
    print("2. Generating dummy inputs (B=1, T=10, N=6)...")
    batch_size = 1
    seq_len = 10
    num_vehicles = 6
    
    dummy_vis = torch.randn(batch_size, seq_len, 3, 224, 224, device=device)
    dummy_graph = torch.randn(batch_size, seq_len, num_vehicles, 16, device=device)

    # Export configuration
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("3. Tracing and exporting to ONNX...")
    try:
        torch.onnx.export(
            model,
            (dummy_vis, dummy_graph),  # Model inputs
            str(output_path),          # Output file name
            export_params=True,        # Store trained weights
            opset_version=args.opset,  # ONNX opset version
            do_constant_folding=True,  # Optimization feature
            input_names=['visual_seq', 'graph_seq'],
            output_names=['logits', 'probabilities', 'fusion_weights', 'mstt_logits', 'gnn_logits', 'gnn_risk'],
            dynamic_axes={
                'visual_seq': {0: 'batch_size', 1: 'seq_len'},    # Variable batch & sequence length
                'graph_seq':  {0: 'batch_size', 1: 'seq_len', 2: 'num_vehicles'},
                'logits':        {0: 'batch_size'},
                'probabilities': {0: 'batch_size'}
            }
        )
        print(f"\n✅ Successfully exported ONNX model to: {output_path}")
        print(f"   Size: {output_path.stat().st_size / (1024*1024):.1f} MB")
        
    except Exception as e:
        print(f"\n❌ Export failed: {str(e)}")
        
if __name__ == "__main__":
    main()
