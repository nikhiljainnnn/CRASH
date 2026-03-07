"""
Metrics computation utilities for crash detection
"""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from typing import Dict, Tuple
import torch


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities (0 to 1)
        threshold: Classification threshold
    
    Returns:
        Dictionary of metrics
    """
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Basic metrics
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    # ROC AUC and PR AUC
    try:
        auroc = roc_auc_score(y_true, y_pred)
        auprc = average_precision_score(y_true, y_pred)
    except:
        auroc = 0.0
        auprc = 0.0
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # False Positive Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # False Negative Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    # Balanced accuracy
    balanced_acc = (recall + specificity) / 2
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auroc': float(auroc),
        'auprc': float(auprc),
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'specificity': float(specificity),
        'fpr': float(fpr),
        'fnr': float(fnr),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }


def compute_time_to_detect(
    crash_timestamps: np.ndarray,
    detection_timestamps: np.ndarray
) -> Dict[str, float]:
    """
    Compute time-to-detect metrics
    
    Args:
        crash_timestamps: Actual crash occurrence times
        detection_timestamps: Predicted crash times
    
    Returns:
        Dictionary with time-to-detect statistics
    """
    time_differences = crash_timestamps - detection_timestamps
    
    # Positive values = early detection (good)
    # Negative values = late detection (bad)
    
    early_detections = time_differences[time_differences > 0]
    late_detections = time_differences[time_differences < 0]
    
    return {
        'mean_time_to_detect': float(np.mean(time_differences)),
        'median_time_to_detect': float(np.median(time_differences)),
        'std_time_to_detect': float(np.std(time_differences)),
        'early_detection_rate': float(len(early_detections) / len(time_differences)),
        'mean_early_time': float(np.mean(early_detections)) if len(early_detections) > 0 else 0.0,
        'mean_late_time': float(np.abs(np.mean(late_detections))) if len(late_detections) > 0 else 0.0
    }


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Compute calibration metrics (Expected Calibration Error)
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        n_bins: Number of bins for calibration
    
    Returns:
        Calibration metrics
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    mce = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Get predictions in this bin
        in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_pred[in_bin])
            
            # ECE: weighted average of calibration error
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            # MCE: maximum calibration error
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
    
    return {
        'expected_calibration_error': float(ece),
        'maximum_calibration_error': float(mce)
    }


def compute_uncertainty_metrics(
    uncertainties: np.ndarray,
    errors: np.ndarray
) -> Dict[str, float]:
    """
    Compute metrics for uncertainty estimates
    
    Args:
        uncertainties: Model uncertainty estimates
        errors: Actual prediction errors
    
    Returns:
        Uncertainty quality metrics
    """
    # Correlation between uncertainty and error
    correlation = np.corrcoef(uncertainties, errors)[0, 1]
    
    # Rank correlation (Spearman)
    from scipy.stats import spearmanr
    rank_corr, p_value = spearmanr(uncertainties, errors)
    
    return {
        'uncertainty_error_correlation': float(correlation),
        'uncertainty_error_rank_correlation': float(rank_corr),
        'rank_correlation_pvalue': float(p_value),
        'mean_uncertainty': float(np.mean(uncertainties)),
        'std_uncertainty': float(np.std(uncertainties))
    }


def compute_trajectory_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int
) -> Dict[str, float]:
    """
    Compute trajectory prediction metrics
    
    Args:
        y_true: (N, horizon, 2) - Ground truth trajectories
        y_pred: (N, horizon, 2) - Predicted trajectories
        horizon: Prediction horizon
    
    Returns:
        Trajectory metrics
    """
    # Average Displacement Error (ADE)
    displacement = np.linalg.norm(y_true - y_pred, axis=-1)  # (N, horizon)
    ade = np.mean(displacement)
    
    # Final Displacement Error (FDE)
    fde = np.mean(displacement[:, -1])
    
    # Miss Rate (FDE > threshold)
    miss_threshold = 2.0  # meters
    miss_rate = np.mean(displacement[:, -1] > miss_threshold)
    
    # Per-timestep errors
    timestep_errors = np.mean(displacement, axis=0)
    
    return {
        'average_displacement_error': float(ade),
        'final_displacement_error': float(fde),
        'miss_rate': float(miss_rate),
        'timestep_errors': timestep_errors.tolist()
    }


def compute_latency_metrics(
    latencies: np.ndarray,
    target_latency: float = 100.0
) -> Dict[str, float]:
    """
    Compute latency performance metrics
    
    Args:
        latencies: Array of latency measurements (ms)
        target_latency: Target latency threshold (ms)
    
    Returns:
        Latency metrics
    """
    return {
        'mean_latency_ms': float(np.mean(latencies)),
        'median_latency_ms': float(np.median(latencies)),
        'std_latency_ms': float(np.std(latencies)),
        'p50_latency_ms': float(np.percentile(latencies, 50)),
        'p95_latency_ms': float(np.percentile(latencies, 95)),
        'p99_latency_ms': float(np.percentile(latencies, 99)),
        'max_latency_ms': float(np.max(latencies)),
        'min_latency_ms': float(np.min(latencies)),
        'target_compliance_rate': float(np.mean(latencies <= target_latency))
    }


class MetricsTracker:
    """Track and aggregate metrics over time"""
    
    def __init__(self):
        self.predictions = []
        self.labels = []
        self.uncertainties = []
        self.latencies = []
        self.timestamps = []
    
    def update(
        self,
        prediction: float,
        label: int,
        uncertainty: float = None,
        latency: float = None,
        timestamp: float = None
    ):
        """Add a new prediction"""
        self.predictions.append(prediction)
        self.labels.append(label)
        
        if uncertainty is not None:
            self.uncertainties.append(uncertainty)
        
        if latency is not None:
            self.latencies.append(latency)
        
        if timestamp is not None:
            self.timestamps.append(timestamp)
    
    def compute_all_metrics(self, threshold: float = 0.5) -> Dict:
        """Compute all available metrics"""
        metrics = {}
        
        if len(self.predictions) > 0:
            # Basic metrics
            metrics.update(compute_metrics(
                np.array(self.labels),
                np.array(self.predictions),
                threshold
            ))
            
            # Calibration
            if len(self.predictions) > 20:
                metrics.update(compute_calibration_metrics(
                    np.array(self.labels),
                    np.array(self.predictions)
                ))
        
        if len(self.uncertainties) > 0:
            errors = np.abs(np.array(self.predictions) - np.array(self.labels))
            metrics.update(compute_uncertainty_metrics(
                np.array(self.uncertainties),
                errors
            ))
        
        if len(self.latencies) > 0:
            metrics.update(compute_latency_metrics(
                np.array(self.latencies)
            ))
        
        return metrics
    
    def reset(self):
        """Reset all tracked data"""
        self.predictions = []
        self.labels = []
        self.uncertainties = []
        self.latencies = []
        self.timestamps = []


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_pred = np.random.random(1000)
    
    # Make predictions somewhat correlated with truth
    y_pred = 0.7 * y_true + 0.3 * y_pred
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, threshold=0.5)
    
    print("Classification Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Calibration
    calib_metrics = compute_calibration_metrics(y_true, y_pred)
    print("\nCalibration Metrics:")
    for key, value in calib_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Latency
    latencies = np.random.gamma(50, 1, 1000)  # Gamma distribution around 50ms
    latency_metrics = compute_latency_metrics(latencies, target_latency=100.0)
    print("\nLatency Metrics:")
    for key, value in latency_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
