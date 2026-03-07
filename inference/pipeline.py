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

from models.temporal.mstt_transformer import MSTT_CA
from models.graph.st_gnn import ST_GNN
from inference.edge.edge_detector import EdgeDetector
from inference.edge.edge_tracker import EdgeTracker
from inference.cloud.risk_scorer import RiskScorer
from utils.visualization import draw_detections, draw_risk_heatmap


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
        """Initialize edge processing components"""
        # Object detector
        self.detector = EdgeDetector(
            model_path=self.config['models']['detection'],
            conf_threshold=self.config['edge']['detection']['confidence_threshold'],
            nms_threshold=self.config['edge']['detection']['nms_threshold'],
            device=self.device
        )
        
        # Object tracker
        self.tracker = EdgeTracker(
            max_age=self.config['edge']['tracking']['max_age'],
            min_hits=self.config['edge']['tracking']['min_hits'],
            iou_threshold=self.config['edge']['tracking']['iou_threshold']
        )
        
    def _init_cloud_components(self):
        """Initialize cloud processing components"""
        # Temporal transformer
        self.temporal_model = MSTT_CA(
            input_dim=512,
            d_model=256,
            n_heads=8,
            n_layers=4,
            num_classes=2
        ).to(self.device)
        
        # Load weights
        checkpoint = torch.load(
            self.config['models']['temporal'],
            map_location=self.device
        )
        self.temporal_model.load_state_dict(checkpoint['model_state_dict'])
        self.temporal_model.eval()
        
        # Graph neural network
        self.graph_model = ST_GNN(
            hidden_dim=128,
            num_layers=4
        ).to(self.device)
        
        checkpoint = torch.load(
            self.config['models']['gnn'],
            map_location=self.device
        )
        self.graph_model.load_state_dict(checkpoint['model_state_dict'])
        self.graph_model.eval()
        
        # Risk scorer
        self.risk_scorer = RiskScorer(
            thresholds=self.config['alerts']['thresholds'],
            mc_samples=self.config['cloud']['bayesian']['mc_samples']
        )
        
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single frame through the pipeline
        
        Args:
            frame: (H, W, 3) RGB frame
        
        Returns:
            dict with detection results and predictions
        """
        start_time = time.time()
        
        # Edge processing - Detection
        det_start = time.time()
        detections = self.detector.detect(frame)
        det_time = (time.time() - det_start) * 1000
        
        # Edge processing - Tracking
        track_start = time.time()
        tracks = self.tracker.update(detections)
        track_time = (time.time() - track_start) * 1000
        
        # Extract features
        features = self._extract_features(frame, tracks)
        self.feature_buffer.append(features)
        
        # Check if enough frames for cloud processing
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
            
            if self.config['cloud']['enabled']:
                # Prepare sequence
                feature_sequence = torch.stack(
                    list(self.feature_buffer)
                ).unsqueeze(0).to(self.device)
                
                # Temporal prediction
                with torch.no_grad():
                    temporal_out = self.temporal_model(
                        feature_sequence,
                        mc_samples=self.config['cloud']['bayesian']['mc_samples']
                    )
                
                # Extract vehicle features for graph
                vehicle_features = self._get_vehicle_features(tracks)
                
                # Graph prediction
                if vehicle_features is not None and len(vehicle_features) > 0:
                    with torch.no_grad():
                        graph_out = self.graph_model(
                            vehicle_features.unsqueeze(0).to(self.device)
                        )
                    risk_scores = graph_out['risk_scores']
                else:
                    risk_scores = None
                
                # Risk scoring
                crash_prob = temporal_out['probabilities'][0, 1].item()
                uncertainty = temporal_out.get('uncertainty', torch.tensor([0.0]))[0].item()
                
                # Compute final risk
                risk_info = self.risk_scorer.compute_risk(
                    crash_probability=crash_prob,
                    uncertainty=uncertainty,
                    graph_risk=risk_scores.mean().item() if risk_scores is not None else 0.0
                )
                
                result.update({
                    'crash_probability': crash_prob,
                    'uncertainty': uncertainty,
                    'risk_level': risk_info['level'],
                    'risk_score': risk_info['score'],
                    'fusion_weights': temporal_out['fusion_weights'][0].cpu().numpy()
                })
                
                # Generate alert if necessary
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
