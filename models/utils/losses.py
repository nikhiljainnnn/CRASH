"""
Custom loss functions for crash detection training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor for positive class (crash events)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class indices
        
        Returns:
            loss: scalar or (N,) depending on reduction
        """
        # Convert to probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get probability of true class
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        alpha_t = torch.where(targets == 1, self.alpha, 1.0)
        
        # Focal loss
        loss = alpha_t * focal_term * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class TemporalSmoothLoss(nn.Module):
    """
    Temporal smoothness loss for video predictions
    Encourages smooth predictions over time
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (N, T, C) predictions over time
        
        Returns:
            loss: Temporal smoothness penalty
        """
        # Compute differences between consecutive timesteps
        diff = predictions[:, 1:] - predictions[:, :-1]
        
        # L2 norm of differences
        loss = torch.norm(diff, p=2, dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class TrajectoryLoss(nn.Module):
    """
    Loss for trajectory prediction
    Combines displacement error with smoothness
    """
    
    def __init__(self, lambda_smooth: float = 0.1):
        super().__init__()
        self.lambda_smooth = lambda_smooth
    
    def forward(
        self,
        pred_traj: torch.Tensor,
        true_traj: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_traj: (N, T, 2) predicted trajectories
            true_traj: (N, T, 2) ground truth trajectories
        
        Returns:
            loss: Combined trajectory loss
        """
        # Displacement loss (L2)
        displacement_loss = F.mse_loss(pred_traj, true_traj)
        
        # Smoothness loss (penalize abrupt changes)
        pred_diff = pred_traj[:, 1:] - pred_traj[:, :-1]
        true_diff = true_traj[:, 1:] - true_traj[:, :-1]
        smoothness_loss = F.mse_loss(pred_diff, true_diff)
        
        # Combined loss
        total_loss = displacement_loss + self.lambda_smooth * smoothness_loss
        
        return total_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning discriminative features
    Useful for differentiating crash vs. non-crash scenarios
    """
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (N, D) feature embeddings
            labels: (N,) binary labels
        
        Returns:
            loss: Contrastive loss
        """
        # Compute pairwise distances
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        
        # Label matrix: 1 if same class, 0 otherwise
        label_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        # Positive pairs (same class)
        pos_loss = label_matrix * dist_matrix.pow(2)
        
        # Negative pairs (different class)
        neg_loss = (1 - label_matrix) * F.relu(self.margin - dist_matrix).pow(2)
        
        # Average over all pairs
        loss = (pos_loss + neg_loss).mean()
        
        return loss


class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal regression loss for severity classification
    Severity has natural ordering: minor < moderate < severe < critical
    """
    
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: (N, num_classes - 1) cumulative logits
            targets: (N,) ordinal targets (0 to num_classes-1)
        
        Returns:
            loss: Ordinal regression loss
        """
        # Create cumulative labels
        # Example: if target=2, cumulative labels = [1, 1, 0]
        cumulative_labels = torch.zeros(
            targets.size(0), self.num_classes - 1,
            device=targets.device
        )
        
        for i in range(self.num_classes - 1):
            cumulative_labels[:, i] = (targets > i).float()
        
        # Binary cross-entropy for each threshold
        loss = F.binary_cross_entropy_with_logits(
            logits, cumulative_labels, reduction='mean'
        )
        
        return loss


class UncertaintyLoss(nn.Module):
    """
    Loss for calibrated uncertainty estimation
    Combines task loss with uncertainty regularization
    """
    
    def __init__(self, lambda_reg: float = 0.01):
        super().__init__()
        self.lambda_reg = lambda_reg
    
    def forward(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: (N,) predicted probabilities
            uncertainties: (N,) predicted uncertainties
            targets: (N,) binary targets
        
        Returns:
            loss: Combined prediction and uncertainty loss
        """
        # Prediction loss (NLL)
        pred_loss = F.binary_cross_entropy(predictions, targets.float())
        
        # Uncertainty regularization
        # Higher uncertainty should correlate with errors
        errors = torch.abs(predictions - targets.float())
        
        # Penalize low uncertainty when error is high
        uncertainty_penalty = torch.mean(errors * (1 - uncertainties))
        
        # Penalize very high uncertainties (overconfident in uncertainty)
        uncertainty_reg = torch.mean(uncertainties)
        
        total_loss = pred_loss + self.lambda_reg * (uncertainty_penalty + uncertainty_reg)
        
        return total_loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining multiple objectives
    """
    
    def __init__(
        self,
        task_weights: dict,
        use_uncertainty_weighting: bool = False
    ):
        """
        Args:
            task_weights: Dictionary of task names to weights
            use_uncertainty_weighting: Learn task weights via uncertainty
        """
        super().__init__()
        self.task_names = list(task_weights.keys())
        
        if use_uncertainty_weighting:
            # Learnable log-variances for automatic weighting
            self.log_vars = nn.Parameter(
                torch.zeros(len(self.task_names))
            )
        else:
            self.register_buffer(
                'weights',
                torch.tensor([task_weights[name] for name in self.task_names])
            )
            self.log_vars = None
    
    def forward(self, losses: dict) -> torch.Tensor:
        """
        Args:
            losses: Dictionary of task losses
        
        Returns:
            total_loss: Weighted combination of losses
        """
        total_loss = 0
        
        for i, name in enumerate(self.task_names):
            if name in losses:
                if self.log_vars is not None:
                    # Uncertainty weighting: w = 1 / (2 * σ^2)
                    precision = torch.exp(-self.log_vars[i])
                    total_loss += precision * losses[name] + self.log_vars[i]
                else:
                    total_loss += self.weights[i] * losses[name]
        
        return total_loss


# Example usage
if __name__ == "__main__":
    # Test Focal Loss
    focal_loss = FocalLoss(alpha=10.0, gamma=2.0)
    
    # Simulate imbalanced data
    logits = torch.randn(100, 2)
    targets = torch.zeros(100, dtype=torch.long)
    targets[:5] = 1  # Only 5% positive class
    
    loss = focal_loss(logits, targets)
    print(f"Focal Loss: {loss.item():.4f}")
    
    # Test Temporal Smooth Loss
    smooth_loss = TemporalSmoothLoss()
    predictions = torch.randn(10, 30, 2)  # 10 sequences, 30 timesteps, 2 classes
    loss = smooth_loss(predictions)
    print(f"Temporal Smooth Loss: {loss.item():.4f}")
    
    # Test Trajectory Loss
    traj_loss = TrajectoryLoss(lambda_smooth=0.1)
    pred_traj = torch.randn(10, 20, 2)  # 10 trajectories, 20 timesteps, (x,y)
    true_traj = torch.randn(10, 20, 2)
    loss = traj_loss(pred_traj, true_traj)
    print(f"Trajectory Loss: {loss.item():.4f}")
    
    # Test Multi-Task Loss
    task_weights = {
        'detection': 1.0,
        'tracking': 0.5,
        'crash_pred': 2.0
    }
    multi_loss = MultiTaskLoss(task_weights)
    
    losses = {
        'detection': torch.tensor(0.5),
        'tracking': torch.tensor(0.3),
        'crash_pred': torch.tensor(0.8)
    }
    total = multi_loss(losses)
    print(f"Multi-Task Loss: {total.item():.4f}")
