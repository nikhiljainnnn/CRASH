import torch
import time
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.train_fusion import CrashPredictionSystem

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CrashPredictionSystem(freeze_backbone=True).to(device)
    
    ckpt_path = ROOT / "checkpoints" / "fusion" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # Dummy data
    vis = torch.randn(1, 40, 3, 224, 224).to(device)
    graph = torch.randn(1, 10, 6, 16).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(vis, graph, mc_samples=1)
            
    latencies = []
    with torch.no_grad():
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(vis, graph, mc_samples=1) # Single pass for latency measurement
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)
            
    print(f"Mean Latency: {np.mean(latencies):.2f} ms")
    print(f"95th Pctl   : {np.percentile(latencies, 95):.2f} ms")

if __name__ == '__main__':
    main()
