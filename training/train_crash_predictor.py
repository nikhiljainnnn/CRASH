"""
Training script for Crash Prediction System
Trains the full multi-modal crash detection pipeline
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

try:
    import wandb
except ImportError:
    wandb = None

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.temporal.mstt_transformer import MSTT_CA
from scripts.train_st_gnn import SimpleSTGNN as ST_GNN
from models.utils.losses import FocalLoss, TemporalSmoothLoss
from models.utils.metrics import compute_metrics

# Optional modules (not yet implemented)
try:
    from models.detection.yolo import YOLODetector
except ImportError:
    YOLODetector = None

try:
    from models.fusion.multimodal_fusion import MultimodalFusion
except ImportError:
    MultimodalFusion = None

try:
    from data.crash_dataset import CrashDataset
except ImportError:
    CrashDataset = None


class CrashPredictionSystem(nn.Module):
    """Complete crash prediction system"""
    
    def __init__(self, config):
        super().__init__()
        
        # Feature extractor (CNN backbone)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            # Simplified ResNet-like blocks
            self._make_layer(64, 128, 2),
            self._make_layer(128, 256, 2),
            self._make_layer(256, 512, 2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Temporal transformer
        self.temporal_model = MSTT_CA(
            input_dim=config['model']['temporal']['d_model'],
            d_model=config['model']['temporal']['d_model'],
            n_heads=config['model']['temporal']['n_heads'],
            n_layers=config['model']['temporal']['n_layers'],
            num_classes=2
        )
        
        # Graph neural network
        self.graph_model = ST_GNN(
            node_feature_dim=16,
            edge_feature_dim=8,
            hidden_dim=config['model']['gnn']['hidden_dim'],
            num_layers=config['model']['gnn']['num_layers']
        )
        
        # Fusion module (if multimodal)
        if config['model']['fusion']['type'] != 'none':
            self.fusion = MultimodalFusion(
                fusion_type=config['model']['fusion']['type'],
                modalities=config['model']['fusion']['modalities']
            )
        else:
            self.fusion = None
        
        # Final classifier
        self.final_classifier = nn.Sequential(
            nn.Linear(512 + 64, 256),  # Temporal + Graph features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        
    def _make_layer(self, in_channels, out_channels, num_blocks):
        """Create ResNet-like layer"""
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        
        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)
    
    def forward(self, batch):
        """
        Args:
            batch: dict with keys:
                - frames: (B, T, C, H, W)
                - vehicle_features: (B, T, N, F)
                - labels: (B,)
        
        Returns:
            dict with predictions
        """
        frames = batch['frames']
        vehicle_features = batch.get('vehicle_features', None)
        
        B, T, C, H, W = frames.shape
        
        # Extract features from each frame
        frame_features = []
        for t in range(T):
            feat = self.feature_extractor(frames[:, t])
            feat = feat.view(B, -1)
            frame_features.append(feat)
        
        frame_features = torch.stack(frame_features, dim=1)  # (B, T, 512)
        
        # Temporal modeling
        temporal_out = self.temporal_model(frame_features, mc_samples=30)
        
        # Graph modeling (if vehicle features available)
        if vehicle_features is not None:
            graph_out = self.graph_model(vehicle_features[:, -1])  # Use last timestep
            graph_features = graph_out['collision_features'].mean(dim=0, keepdim=True).expand(B, -1)
        else:
            graph_features = torch.zeros(B, 64).to(frames.device)
        
        # Combine temporal and graph features
        combined = torch.cat([
            temporal_out['probabilities'][:, 1:2],  # Crash probability
            graph_features
        ], dim=-1)
        
        # Final prediction
        logits = self.final_classifier(combined)
        
        return {
            'logits': logits,
            'probabilities': torch.softmax(logits, dim=-1),
            'temporal_fusion_weights': temporal_out['fusion_weights'],
            'uncertainty': temporal_out.get('uncertainty', None)
        }


def train_epoch(model, dataloader, optimizer, criterion, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        frames = batch['frames'].to(device)
        labels = batch['labels'].to(device)
        vehicle_features = batch.get('vehicle_features', None)
        if vehicle_features is not None:
            vehicle_features = vehicle_features.to(device)
        
        batch_device = {
            'frames': frames,
            'vehicle_features': vehicle_features,
            'labels': labels
        }
        
        # Forward pass
        optimizer.zero_grad()
        output = model(batch_device)
        
        # Compute loss
        loss = criterion(output['logits'], labels)
        
        # Add temporal smoothness loss
        if config['training']['loss_weights'].get('temporal_smooth', 0) > 0:
            smooth_loss = TemporalSmoothLoss()(output['temporal_fusion_weights'])
            loss += config['training']['loss_weights']['temporal_smooth'] * smooth_loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config['training'].get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['training']['grad_clip']
            )
        
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        preds = output['probabilities'][:, 1].detach().cpu().numpy()
        labels_np = labels.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels_np)
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
        
        # Log to wandb
        if config['logging']['use_wandb'] and batch_idx % config['logging']['log_interval'] == 0:
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/step': batch_idx
            })
    
    # Compute epoch metrics
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    
    return avg_loss, metrics


def validate(model, dataloader, criterion, device, config):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_uncertainties = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            frames = batch['frames'].to(device)
            labels = batch['labels'].to(device)
            vehicle_features = batch.get('vehicle_features', None)
            if vehicle_features is not None:
                vehicle_features = vehicle_features.to(device)
            
            batch_device = {
                'frames': frames,
                'vehicle_features': vehicle_features,
                'labels': labels
            }
            
            # Forward pass
            output = model(batch_device)
            
            # Compute loss
            loss = criterion(output['logits'], labels)
            total_loss += loss.item()
            
            # Collect predictions
            preds = output['probabilities'][:, 1].cpu().numpy()
            labels_np = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_np)
            
            if output['uncertainty'] is not None:
                all_uncertainties.extend(output['uncertainty'].cpu().numpy())
    
    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    
    if all_uncertainties:
        metrics['mean_uncertainty'] = np.mean(all_uncertainties)
    
    return avg_loss, metrics


def main():
    # Load configuration
    with open('configs/train_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb
    if config['logging']['use_wandb']:
        wandb.init(
            project=config['logging']['wandb_project'],
            entity=config['logging']['wandb_entity'],
            config=config
        )
    
    # Create datasets
    train_dataset = CrashDataset(
        config['data']['train_path'],
        sequence_length=config['data']['sequence_length'],
        augmentation=config['data']['augmentation']
    )
    
    val_dataset = CrashDataset(
        config['data']['val_path'],
        sequence_length=config['data']['sequence_length'],
        augmentation=None
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    # Create model
    model = CrashPredictionSystem(config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss function
    criterion = FocalLoss(
        alpha=config['training']['focal_loss']['alpha'],
        gamma=config['training']['focal_loss']['gamma']
    )
    
    # Optimizer
    if config['training']['optimizer']['type'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['optimizer']['lr'],
            weight_decay=config['training']['optimizer']['weight_decay'],
            betas=config['training']['optimizer']['betas']
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['optimizer']['lr']
        )
    
    # Learning rate scheduler
    if config['training']['scheduler']['type'] == 'cosine_annealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['scheduler']['T_max'],
            eta_min=config['training']['scheduler']['eta_min']
        )
    
    # Training loop
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, config
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, config
        )
        
        # Step scheduler
        scheduler.step()
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_metrics['f1_score']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val F1: {val_metrics['f1_score']:.4f}")
        
        # Log to wandb
        if config['logging']['use_wandb']:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/f1': train_metrics['f1_score'],
                'train/precision': train_metrics['precision'],
                'train/recall': train_metrics['recall'],
                'val/loss': val_loss,
                'val/f1': val_metrics['f1_score'],
                'val/precision': val_metrics['precision'],
                'val/recall': val_metrics['recall'],
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Save best model
        if val_metrics['f1_score'] > best_f1:
            best_f1 = val_metrics['f1_score']
            checkpoint_path = os.path.join(
                config['checkpoint']['save_dir'], 
                'best_model.pth'
            )
            os.makedirs(config['checkpoint']['save_dir'], exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1_score': best_f1,
                'config': config
            }, checkpoint_path)
            print(f"Saved best model with F1: {best_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if config['early_stopping']['enabled']:
            if patience_counter >= config['early_stopping']['patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save periodic checkpoint
        if (epoch + 1) % config['checkpoint']['save_interval'] == 0:
            checkpoint_path = os.path.join(
                config['checkpoint']['save_dir'],
                f'checkpoint_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, checkpoint_path)
    
    print("\nTraining completed!")
    print(f"Best F1 Score: {best_f1:.4f}")
    
    if config['logging']['use_wandb']:
        wandb.finish()


if __name__ == "__main__":
    main()
