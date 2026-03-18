import json
import cv2
import os
import glob

crash_videos = glob.glob(r'Datasets\videos-20260206T085121Z-1-002\videos\Crash-1500\*.mp4')[:5]
normal_videos = glob.glob(r'Datasets\Normal-001\*.mp4')[:5]

scenarios = []

def get_frame_count(path):
    cap = cv2.VideoCapture(path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count

for i, path in enumerate(crash_videos):
    frames = get_frame_count(path)
    # Estimate the actual crash happens at 80% through the clip duration
    crash_frame = int(frames * 0.8)
    scenarios.append({
        "name": f"Real Crash Video {i+1}",
        "video_path": path,
        "scenario_type": "head_on", 
        "weather": "clear",
        "ground_truth": {
            "crash_occurred": True,
            "crash_frame": crash_frame if crash_frame > 0 else 50
        }
    })

for i, path in enumerate(normal_videos):
    scenarios.append({
        "name": f"Real Normal Video {i+1}",
        "video_path": path,
        "scenario_type": "normal_traffic",
        "weather": "clear",
        "ground_truth": {
            "crash_occurred": False,
            "crash_frame": None
        }
    })

with open('tests/test_dataset_real.json', 'w') as f:
    json.dump(scenarios, f, indent=2)

print(f"Generated real dataset mapping with {len(scenarios)} videos.")
