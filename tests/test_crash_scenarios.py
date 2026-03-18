#!/usr/bin/env python3
"""
Comprehensive Testing Framework for Crash Detection System
Tests various crash scenarios and generates detailed metrics
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Add project root to Python path
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from inference.pipeline import CrashDetectionPipeline
from models.utils.metrics import (
    compute_metrics, 
    compute_time_to_detect,
    compute_calibration_metrics,
    compute_latency_metrics
)


class CrashScenarioTester:
    """
    Comprehensive testing framework for crash detection system
    Tests multiple scenarios and generates detailed performance reports
    """
    
    def __init__(self, model_config_path: str, output_dir: str = "test_results"):
        """
        Args:
            model_config_path: Path to model configuration
            output_dir: Directory to save test results
        """
        self.pipeline = CrashDetectionPipeline(model_config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.test_results = {
            'scenarios': [],
            'overall_metrics': {},
            'per_scenario_metrics': {},
            'latency_analysis': {},
            'failure_cases': []
        }
        
    def load_test_dataset(self, dataset_path: str) -> List[Dict]:
        """
        Load test dataset with ground truth annotations
        """
        with open(dataset_path, 'r') as f:
            test_data = json.load(f)
        
        print(f"Loaded {len(test_data)} test samples")
        return test_data
    
    def test_scenario(
        self, 
        video_path: str,
        ground_truth: Dict,
        scenario_name: str
    ) -> Dict:
        print(f"\nTesting scenario: {scenario_name}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Ground truth data
        crash_frame = ground_truth.get('crash_frame', None)
        crash_occurred = ground_truth.get('crash_occurred', False)
        crash_timestamp = crash_frame / fps if crash_frame else None
        
        # Predictions storage
        predictions = []
        frame_results = []
        crash_detected = False
        detection_frame = None
        
        import random
        # Process each frame
        for frame_idx in tqdm(range(total_frames), desc=f"Processing {scenario_name}"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run valid ML inference
            result = self.pipeline.process_frame(frame)
            
            # Store prediction
            crash_prob = result['crash_probability']
            predictions.append(crash_prob)
            
            frame_results.append({
                'frame': frame_idx,
                'timestamp': frame_idx / fps,
                'crash_probability': crash_prob,
                'risk_level': result['risk_level'],
                'num_vehicles': len(result['detections']),
                'latency_ms': result['latency']['total']
            })
            
            # Check if crash detected
            if not crash_detected and crash_prob > 0.7:  # Detection threshold
                crash_detected = True
                detection_frame = frame_idx
        
        cap.release()
        
        # Compute metrics
        detection_timestamp = detection_frame / fps if detection_frame else None
        
        # Time to detect (if crash actually occurred)
        time_to_detect = None
        early_detection = False
        if crash_occurred and crash_detected:
            time_to_detect = crash_timestamp - detection_timestamp
            early_detection = time_to_detect > 0
        
        # Classification metrics
        true_positive = crash_occurred and crash_detected
        false_positive = (not crash_occurred) and crash_detected
        false_negative = crash_occurred and (not crash_detected)
        true_negative = (not crash_occurred) and (not crash_detected)
        
        # Latency statistics
        latencies = [r['latency_ms'] for r in frame_results]
        
        result_dict = {
            'scenario_name': scenario_name,
            'video_path': video_path,
            'ground_truth': ground_truth,
            'predictions': {
                'crash_detected': crash_detected,
                'detection_frame': detection_frame,
                'detection_timestamp': detection_timestamp,
                'max_crash_probability': max(predictions),
                'mean_crash_probability': np.mean(predictions)
            },
            'metrics': {
                'true_positive': true_positive,
                'false_positive': false_positive,
                'false_negative': false_negative,
                'true_negative': true_negative,
                'time_to_detect_seconds': time_to_detect,
                'early_detection': early_detection
            },
            'latency': {
                'mean_ms': np.mean(latencies),
                'median_ms': np.median(latencies),
                'p95_ms': np.percentile(latencies, 95),
                'p99_ms': np.percentile(latencies, 99),
                'max_ms': np.max(latencies)
            },
            'frame_results': frame_results
        }
        
        return result_dict
    
    def test_scenario_types(self, test_data: List[Dict]) -> Dict:
        scenario_types = {
            'rear_end': [], 'intersection': [], 'pedestrian': [],
            'sideswipe': [], 'head_on': [], 'multi_vehicle': [],
            'near_miss': [], 'normal_traffic': []
        }
        
        for test_sample in test_data:
            scenario_type = test_sample.get('scenario_type', 'unknown')
            result = self.test_scenario(
                test_sample['video_path'],
                test_sample['ground_truth'],
                test_sample['name']
            )
            
            if scenario_type in scenario_types:
                scenario_types[scenario_type].append(result)
            
            self.test_results['scenarios'].append(result)
        
        return scenario_types
    
    def compute_overall_metrics(self) -> Dict:
        all_results = self.test_results['scenarios']
        
        # Aggregate classification results
        tp = sum(1 for r in all_results if r['metrics']['true_positive'])
        fp = sum(1 for r in all_results if r['metrics']['false_positive'])
        fn = sum(1 for r in all_results if r['metrics']['false_negative'])
        tn = sum(1 for r in all_results if r['metrics']['true_negative'])
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        time_to_detect_values = [
            r['metrics']['time_to_detect_seconds'] 
            for r in all_results 
            if r['metrics']['time_to_detect_seconds'] is not None 
            and r['metrics']['early_detection']
        ]
        
        all_latencies = []
        for r in all_results:
            all_latencies.extend([fr['latency_ms'] for fr in r['frame_results']])
        
        overall_metrics = {
            'classification': {
                'precision': precision, 'recall': recall, 'f1_score': f1,
                'accuracy': accuracy, 'true_positives': tp, 'false_positives': fp,
                'false_negatives': fn, 'true_negatives': tn
            },
            'time_to_detect': {
                'mean_seconds': np.mean(time_to_detect_values) if time_to_detect_values else None,
                'median_seconds': np.median(time_to_detect_values) if time_to_detect_values else None,
                'min_seconds': np.min(time_to_detect_values) if time_to_detect_values else None,
                'max_seconds': np.max(time_to_detect_values) if time_to_detect_values else None,
                'early_detection_rate': len(time_to_detect_values) / tp if tp > 0 else 0
            },
            'latency': {
                'mean_ms': np.mean(all_latencies),
                'median_ms': np.median(all_latencies),
                'p95_ms': np.percentile(all_latencies, 95),
                'p99_ms': np.percentile(all_latencies, 99),
                'max_ms': np.max(all_latencies),
                'below_100ms_rate': sum(1 for l in all_latencies if l < 100) / len(all_latencies)
            }
        }
        
        self.test_results['overall_metrics'] = overall_metrics
        return overall_metrics
    
    def generate_report(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"test_report_{timestamp}.md"
        metrics = self.test_results['overall_metrics']
        
        # Helper to safely format metrics that might be None
        def safe_fmt(val, fmt=".2f", default="N/A"):
            if val is None:
                return default
            return f"{val:{fmt}}"
            
        report = f"""# Crash Detection System - Test Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overview
- **Total Scenarios Tested**: {len(self.test_results['scenarios'])}

## Overall Performance Metrics

### Classification Metrics
- **Precision**: {metrics['classification']['precision']:.4f}
- **Recall**: {metrics['classification']['recall']:.4f}
- **F1-Score**: {metrics['classification']['f1_score']:.4f}

### Confusion Matrix
|                | Predicted Crash | Predicted Normal |
|----------------|----------------|------------------|
| **Actual Crash**  | {metrics['classification']['true_positives']} (TP) | {metrics['classification']['false_negatives']} (FN) |
| **Actual Normal** | {metrics['classification']['false_positives']} (FP) | {metrics['classification']['true_negatives']} (TN) |

### Time-to-Detect Performance
- **Mean Early Detection**: {safe_fmt(metrics['time_to_detect']['mean_seconds'])} seconds
- **Early Detection Rate**: {metrics['time_to_detect']['early_detection_rate']:.2%}

### Latency Performance
- **Mean Latency**: {metrics['latency']['mean_ms']:.2f} ms
- **P99 Latency**: {metrics['latency']['p99_ms']:.2f} ms
- **<100ms Compliance**: {metrics['latency']['below_100ms_rate']:.2%}
"""
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {report_path}")
        return str(report_path)
    
    def visualize_results(self):
        plt.figure(figsize=(8, 6))
        cm_data = [
            [self.test_results['overall_metrics']['classification']['true_positives'],
             self.test_results['overall_metrics']['classification']['false_negatives']],
            [self.test_results['overall_metrics']['classification']['false_positives'],
             self.test_results['overall_metrics']['classification']['true_negatives']]
        ]
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Predicted Crash', 'Predicted Normal'],
                   yticklabels=['Actual Crash', 'Actual Normal'])
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300)
        plt.close()
        
        all_latencies = []
        for r in self.test_results['scenarios']:
            all_latencies.extend([fr['latency_ms'] for fr in r['frame_results']])
        
        plt.figure(figsize=(10, 6))
        plt.hist(all_latencies, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(100, color='r', linestyle='--', label='100ms Target')
        plt.axvline(np.mean(all_latencies), color='g', linestyle='--', 
                   label=f'Mean: {np.mean(all_latencies):.1f}ms')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Frequency')
        plt.title('Latency Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'latency_distribution.png', dpi=300)
        plt.close()
        
        time_to_detect_values = [
            r['metrics']['time_to_detect_seconds'] 
            for r in self.test_results['scenarios']
            if r['metrics']['time_to_detect_seconds'] is not None
        ]
        
        if time_to_detect_values:
            plt.figure(figsize=(10, 6))
            plt.hist(time_to_detect_values, bins=20, edgecolor='black', alpha=0.7)
            plt.xlabel('Time to Detect (seconds before crash)')
            plt.ylabel('Frequency')
            plt.title('Early Detection Time Distribution')
            plt.axvline(np.mean(time_to_detect_values), color='r', linestyle='--',
                       label=f'Mean: {np.mean(time_to_detect_values):.2f}s')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / 'time_to_detect.png', dpi=300)
            plt.close()
    
    def save_results(self):
        output_file = self.output_dir / 'test_results.json'
        with open(output_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"Results saved to: {output_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test Crash Detection System')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to model configuration')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test dataset JSON')
    parser.add_argument('--output', type=str, default='test_results',
                       help='Output directory for results')
    args = parser.parse_args()
    
    tester = CrashScenarioTester(args.config, args.output)
    test_data = tester.load_test_dataset(args.test_data)
    
    scenario_results = tester.test_scenario_types(test_data)
    overall_metrics = tester.compute_overall_metrics()
    report_path = tester.generate_report()
    tester.visualize_results()
    tester.save_results()
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
    print(f"\nOverall F1-Score: {overall_metrics['classification']['f1_score']:.4f}")
    
    mean_ttd = overall_metrics['time_to_detect']['mean_seconds']
    ttd_str = f"{mean_ttd:.2f}s" if mean_ttd is not None else "N/A"
    print(f"Mean Time-to-Detect: {ttd_str}")
    
    print(f"Mean Latency: {overall_metrics['latency']['mean_ms']:.2f}ms")
    print(f"\nFull report: {report_path}")

if __name__ == "__main__":
    main()
