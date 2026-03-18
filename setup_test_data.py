import json
import cv2
import numpy as np
import os

os.makedirs('tests/videos', exist_ok=True)
os.makedirs('tests/test_results', exist_ok=True)

scenarios = [
    # 5 rear-end
    *[{"name": f"Rear-End collision {i+1}", "video_path": f"tests/videos/rear_end_{i}.mp4", "scenario_type": "rear_end", "weather": "clear", "ground_truth": {"crash_occurred": True, "crash_frame": 120}} for i in range(5)],
    # 3 intersection
    *[{"name": f"Intersection crash {i+1}", "video_path": f"tests/videos/intersection_{i}.mp4", "scenario_type": "intersection", "weather": "rain", "ground_truth": {"crash_occurred": True, "crash_frame": 120}} for i in range(3)],
    # 2 pedestrian
    *[{"name": f"Pedestrian incident {i+1}", "video_path": f"tests/videos/pedestrian_{i}.mp4", "scenario_type": "pedestrian", "weather": "fog", "ground_truth": {"crash_occurred": True, "crash_frame": 120}} for i in range(2)],
    # 1 sideswipe
    {"name": "Sideswipe collision 1", "video_path": "tests/videos/sideswipe_0.mp4", "scenario_type": "sideswipe", "weather": "clear", "ground_truth": {"crash_occurred": True, "crash_frame": 120}},
    # 1 head-on
    {"name": "Head-on collision 1", "video_path": "tests/videos/head_on_0.mp4", "scenario_type": "head_on", "weather": "night", "ground_truth": {"crash_occurred": True, "crash_frame": 120}},
    # 2 multi-vehicle
    *[{"name": f"Multi-vehicle pileup {i+1}", "video_path": f"tests/videos/multi_{i}.mp4", "scenario_type": "multi_vehicle", "weather": "snow", "ground_truth": {"crash_occurred": True, "crash_frame": 120}} for i in range(2)],
    # 2 normal traffic
    *[{"name": f"Normal traffic {i+1}", "video_path": f"tests/videos/normal_{i}.mp4", "scenario_type": "normal_traffic", "weather": "clear", "ground_truth": {"crash_occurred": False, "crash_frame": None}} for i in range(2)]
]

with open('tests/test_dataset.json', 'w') as f:
    json.dump(scenarios, f, indent=2)

print("Creating dummy video files...")
for s in scenarios:
    path = s["video_path"]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 480))
    for i in range(150): # 5 seconds at 30fps
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add frame index to avoid totally blank compression issues
        cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    out.release()
print("Test dataset generation complete.")
