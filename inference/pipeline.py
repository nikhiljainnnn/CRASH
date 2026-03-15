"""
Real-time Inference Pipeline for Crash Detection
Processes video streams and predicts crashes in real-time
"""

import cv2
import torch
import numpy as np
import yaml
import time
import json
from collections import deque
from pathlib import Path
import argparse

from ultralytics import YOLO
from models.temporal.mstt_transformer import MSTT_CA
from scripts.train_st_gnn import SimpleSTGNN

ROOT = Path(__file__).resolve().parent.parent


def draw_detections(frame, detections):
    """Draw bounding boxes on frame."""
    for det in detections:
        bbox = det.get('bbox', [])
        if len(bbox) == 4:
            x, y, w, h = [int(v) for v in bbox]
            conf = det.get('confidence', 0)
            cls = det.get('class', '')
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls} {conf:.2f}", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame


def draw_risk_heatmap(frame, crash_prob, risk_level):
    """Overlay a risk color bar on top of the frame."""
    color = (0, 255, 0)
    if risk_level == 'high':     color = (0, 165, 255)
    if risk_level == 'critical': color = (0, 0, 255)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 8), color, -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    return frame


class CrashDetectionPipeline:
    """End-to-end crash detection pipeline"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to inference configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Device
        self.device = torch.device(
            self.config['edge']['device'] if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")
        
        # Initialize components
        self._init_edge_components()
        if self.config['cloud']['enabled']:
            self._init_cloud_components()
        
        # Frame buffer
        self.frame_buffer = deque(
            maxlen=self.config['data']['buffer_size']
        )
        
        # Feature buffer
        self.feature_buffer = deque(
            maxlen=self.config['data']['buffer_size']
        )
        
        # Performance tracking
        self.latency_tracker = {
            'detection': [],
            'tracking': [],
            'cloud': [],
            'total': []
        }
        
        # Alert history
        self.alert_history = []
        
    def _init_edge_components(self):
        """Initialize edge processing components using YOLOv8."""
        det_model_path = self.config.get('models', {}).get(
            'detection', str(ROOT / 'checkpoints/detection/run/weights/best.pt')
        )
        self.detector_model = YOLO(det_model_path) if Path(det_model_path).exists() else None
        self.track_history = {}  # track_id -> history
        self.next_track_id = 0

    def _detect(self, frame):
        """Run YOLOv8 detection on a frame."""
        if self.detector_model is None:
            return []
        results = self.detector_model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append({
                'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                'confidence': float(box.conf[0]),
                'class': results.names[int(box.cls[0])]
            })
        return detections

    def _track(self, detections):
        """Simple centroid-based tracker."""
        tracks = []
        for i, det in enumerate(detections):
            track_id = self.next_track_id + i
            bbox = det['bbox']
            tracks.append({**det, 'track_id': track_id,
                            'vx': 0.0, 'vy': 0.0, 'ax': 0.0, 'ay': 0.0,
                            'heading': 0.0})
        self.next_track_id += len(detections)
        return tracks

    def _init_cloud_components(self):
        """Initialize cloud processing components."""
        temporal_path = self.config.get('models', {}).get(
            'temporal', str(ROOT / 'checkpoints/crash_predictor/best.pt')
        )
        if Path(temporal_path).exists():
            self.temporal_model = MSTT_CA(
                input_dim=512, d_model=256, n_heads=8,
                n_layers=4, num_classes=2
            ).to(self.device)
            checkpoint = torch.load(temporal_path, map_location=self.device, weights_only=False)
            self.temporal_model.load_state_dict(
                checkpoint['model_state_dict'], strict=False)
            self.temporal_model.eval()
        else:
            self.temporal_model = None

        gnn_path = self.config.get('models', {}).get(
            'gnn', str(ROOT / 'checkpoints/st_gnn/best.pt')
        )
        if Path(gnn_path).exists():
            self.graph_model = SimpleSTGNN(
                node_dim=16, edge_dim=8, hidden=128, heads=4, n_layers=3
            ).to(self.device)
            checkpoint = torch.load(gnn_path, map_location=self.device, weights_only=False)
            self.graph_model.load_state_dict(
                checkpoint['model_state_dict'], strict=False)
            self.graph_model.eval()
        else:
            self.graph_model = None

        # Simple risk scorer
        self.risk_thresholds = self.config.get('alerts', {}).get(
            'thresholds', {'critical': 0.9, 'high': 0.7, 'medium': 0.5}
        )
        self.mc_samples = self.config.get('cloud', {}).get(
            'bayesian', {}).get('mc_samples', 5)

    def _compute_risk(self, crash_prob, uncertainty=0):
        """Map crash probability to risk level."""
        score = crash_prob * (1 + uncertainty)
        if score >= self.risk_thresholds.get('critical', 0.9):
            level = 'critical'
        elif score >= self.risk_thresholds.get('high', 0.7):
            level = 'high'
        elif score >= self.risk_thresholds.get('medium', 0.5):
            level = 'medium'
        elif score > 0.1:
            level = 'low'
        else:
            level = 'none'
        return {'level': level, 'score': float(score)}

    def process_frame(self, frame: np.ndarray) -> dict:
        start_time = time.time()

        # Detection + Tracking
        det_start = time.time()
        detections = self._detect(frame)
        det_time = (time.time() - det_start) * 1000

        track_start = time.time()
        tracks = self._track(detections)
        track_time = (time.time() - track_start) * 1000

        features = self._extract_features(frame, tracks)
        self.feature_buffer.append(features)

        result = {
            'detections': detections,
            'tracks': tracks,
            'crash_probability': 0.0,
            'risk_level': 'none',
            'time_to_collision': None,
            'alert': None
        }


        if len(self.feature_buffer) >= self.config['data']['sequence_length']:
            # Cloud processing
            cloud_start = time.time()
            
            if self.config.get('cloud', {}).get('enabled', True) and \
                    self.temporal_model is not None:
                feature_sequence = torch.stack(
                    list(self.feature_buffer)
                ).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    temporal_out = self.temporal_model(feature_sequence)

                vehicle_features = self._get_vehicle_features(tracks)

                if vehicle_features is not None and self.graph_model is not None:
                    with torch.no_grad():
                        risk_seq = vehicle_features.unsqueeze(0).unsqueeze(0)
                        g_risk, _ = self.graph_model(risk_seq[0])
                    risk_scores = g_risk
                else:
                    risk_scores = None

                crash_prob = temporal_out['probabilities'][0, 1].item()
                uncertainty = temporal_out.get(
                    'uncertainty', torch.tensor([0.0])
                )[0].item() if 'uncertainty' in temporal_out else 0.0

                risk_info = self._compute_risk(crash_prob, uncertainty)

                result.update({
                    'crash_probability': crash_prob,
                    'uncertainty': uncertainty,
                    'risk_level': risk_info['level'],
                    'risk_score': risk_info['score'],
                })

                if risk_info['level'] in ['high', 'critical']:
                    alert = self._generate_alert(result, tracks)
                    result['alert'] = alert
                    self.alert_history.append(alert)

            cloud_time = (time.time() - cloud_start) * 1000
            self.latency_tracker['cloud'].append(cloud_time)
        
        # Track latencies
        total_time = (time.time() - start_time) * 1000
        self.latency_tracker['detection'].append(det_time)
        self.latency_tracker['tracking'].append(track_time)
        self.latency_tracker['total'].append(total_time)
        
        result['latency'] = {
            'detection': det_time,
            'tracking': track_time,
            'total': total_time
        }
        
        return result
    
    def _extract_features(self, frame: np.ndarray, tracks: list) -> torch.Tensor:
        """
        Extract features from frame using CNN
        
        Args:
            frame: (H, W, 3)
            tracks: List of track objects
        
        Returns:
            features: (feature_dim,)
        """
        # Simple feature extraction (in production, use pretrained CNN)
        # Resize and normalize
        frame_resized = cv2.resize(frame, (224, 224))
        frame_norm = frame_resized.astype(np.float32) / 255.0
        
        # Convert to tensor
        frame_tensor = torch.from_numpy(frame_norm).permute(2, 0, 1)
        
        # Extract features (placeholder - use actual CNN in production)
        features = torch.randn(512)  # Placeholder
        
        return features
    
    def _get_vehicle_features(self, tracks: list) -> torch.Tensor:
        """
        Extract vehicle features for graph construction
        
        Args:
            tracks: List of track objects
        
        Returns:
            features: (num_vehicles, feature_dim)
        """
        if not tracks:
            return None
        
        features = []
        for track in tracks:
            # Extract position, velocity, etc.
            bbox = track['bbox']  # [x, y, w, h]
            x = bbox[0] + bbox[2] / 2
            y = bbox[1] + bbox[3] / 2
            
            # Velocity (from track history if available)
            vx = track.get('vx', 0.0)
            vy = track.get('vy', 0.0)
            
            # Acceleration
            ax = track.get('ax', 0.0)
            ay = track.get('ay', 0.0)
            
            # Other features
            heading = track.get('heading', 0.0)
            length = bbox[3]
            width = bbox[2]
            class_id = track.get('class', 0)
            
            feat = torch.tensor([
                x, y, vx, vy, ax, ay, heading, length, width,
                class_id, 0, 0, 0, 0, 0, 0  # Padding to 16
            ], dtype=torch.float32)
            
            features.append(feat)
        
        return torch.stack(features) if features else None
    
    def _generate_alert(self, result: dict, tracks: list) -> dict:
        """Generate alert information"""
        alert = {
            'timestamp': time.time(),
            'crash_probability': result['crash_probability'],
            'risk_level': result['risk_level'],
            'risk_score': result.get('risk_score', 0.0),
            'num_vehicles_involved': len(tracks),
            'uncertainty': result.get('uncertainty', 0.0),
            'recommended_action': self._get_recommended_action(result['risk_level'])
        }
        
        return alert
    
    def _get_recommended_action(self, risk_level: str) -> str:
        """Get recommended action based on risk level"""
        actions = {
            'critical': 'IMMEDIATE: Dispatch emergency services, activate traffic signals',
            'high': 'ALERT: Prepare emergency services, warn nearby vehicles',
            'medium': 'MONITOR: Increase surveillance, log event',
            'low': 'LOG: Record for analysis'
        }
        return actions.get(risk_level, 'NONE')
    
    def run_video(self, video_path: str, output_path: str = None):
        """
        Run pipeline on video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        print(f"Processing video at {fps} FPS...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            result = self.process_frame(frame)
            
            # Visualize
            if self.config['output']['draw_detections']:
                frame = draw_detections(frame, result['detections'])
            
            if self.config['output']['draw_risk_heatmap'] and result['crash_probability'] > 0:
                frame = draw_risk_heatmap(
                    frame, 
                    result['crash_probability'],
                    result.get('risk_level', 'none')
                )
            
            # Display alert
            if result['alert']:
                cv2.putText(
                    frame,
                    f"ALERT: {result['risk_level'].upper()}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2
                )
                cv2.putText(
                    frame,
                    f"Crash Prob: {result['crash_probability']:.2%}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
            
            # Write frame
            if output_path:
                out.write(frame)
            
            # Display
            cv2.imshow('Crash Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            if frame_count % 100 == 0:
                avg_latency = np.mean(self.latency_tracker['total'][-100:])
                print(f"Processed {frame_count} frames, Avg latency: {avg_latency:.2f}ms")
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        self.print_statistics()
    
    def print_statistics(self):
        """Print pipeline statistics"""
        print("\n" + "="*50)
        print("PIPELINE STATISTICS")
        print("="*50)
        
        for key in ['detection', 'tracking', 'cloud', 'total']:
            if self.latency_tracker[key]:
                times = self.latency_tracker[key]
                print(f"\n{key.upper()} Latency:")
                print(f"  Mean: {np.mean(times):.2f}ms")
                print(f"  Median: {np.median(times):.2f}ms")
                print(f"  P95: {np.percentile(times, 95):.2f}ms")
                print(f"  P99: {np.percentile(times, 99):.2f}ms")
        
        print(f"\nTotal Alerts Generated: {len(self.alert_history)}")
        
        if self.alert_history:
            risk_levels = [a['risk_level'] for a in self.alert_history]
            print("\nAlert Distribution:")
            for level in ['critical', 'high', 'medium', 'low']:
                count = risk_levels.count(level)
                print(f"  {level}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Crash Detection Inference')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--rtsp', type=str, help='RTSP stream URL')
    parser.add_argument('--config', type=str, default='configs/inference_config.yaml',
                       help='Path to config file')
    parser.add_argument('--output', type=str, help='Path to output video')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CrashDetectionPipeline(args.config)
    
    # Run on video or RTSP
    if args.video:
        pipeline.run_video(args.video, args.output)
    elif args.rtsp:
        pipeline.run_video(args.rtsp, args.output)
    else:
        print("Please specify --video or --rtsp")


if __name__ == "__main__":
    main()
