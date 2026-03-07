"""
Multi-Scale Temporal Transformer with Causal Attention (MSTT-CA)
For crash prediction using multiple temporal scales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class CausalMultiHeadAttention(nn.Module):
    """Multi-head attention with causal masking (no future leakage)"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(
        self, 
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, seq_len, d_model)
            key: (batch, seq_len, d_model)
            value: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len) or (seq_len, seq_len)
        
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, n_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.q_linear(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask (causal + padding)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        context = torch.matmul(attention, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear
        output = self.out_linear(context)
        
        return output, attention


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1)]


class TemporalTransformerBlock(nn.Module):
    """Single transformer block with causal attention"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = CausalMultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Causal mask
        
        Returns:
            output: (batch, seq_len, d_model)
            attention: Attention weights
        """
        # Self-attention with residual
        attn_out, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x, attn_weights


class TemporalTransformer(nn.Module):
    """Stack of temporal transformer blocks for single time scale"""
    
    def __init__(
        self, 
        window_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.window_size = window_size
        self.d_model = d_model
        
        self.pos_encoding = PositionalEncoding(d_model, max_len=window_size)
        
        self.layers = nn.ModuleList([
            TemporalTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Create causal mask
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(window_size, window_size)).unsqueeze(0).unsqueeze(0)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: List of attention weights from each layer
        """
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Get appropriate mask for sequence length
        seq_len = x.size(1)
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        
        # Pass through transformer layers
        attention_weights = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            attention_weights.append(attn)
        
        return x, attention_weights


class MSTT_CA(nn.Module):
    """
    Multi-Scale Temporal Transformer with Causal Attention
    
    Processes sequences at three temporal scales:
    - Short-term (8 frames ~ 0.27s): Sudden events
    - Medium-term (16 frames ~ 0.53s): Maneuvers
    - Long-term (32 frames ~ 1.07s): Traffic patterns
    """
    
    def __init__(
        self,
        input_dim: int = 512,  # Feature dimension from CNN backbone
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        num_classes: int = 2,  # crash/no-crash
        short_window: int = 8,
        medium_window: int = 16,
        long_window: int = 32
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Multi-scale transformers
        self.short_transformer = TemporalTransformer(
            short_window, d_model, n_heads, n_layers, d_ff, dropout
        )
        self.medium_transformer = TemporalTransformer(
            medium_window, d_model, n_heads, n_layers, d_ff, dropout
        )
        self.long_transformer = TemporalTransformer(
            long_window, d_model, n_heads, n_layers, d_ff, dropout
        )
        
        # Adaptive fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(3 * d_model, 3),
            nn.Softmax(dim=-1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # For Bayesian uncertainty (MC Dropout)
        self.mc_dropout = nn.Dropout(0.3)
        
    def extract_windows(
        self, 
        x: torch.Tensor, 
        window_size: int
    ) -> torch.Tensor:
        """
        Extract last window_size frames from sequence
        
        Args:
            x: (batch, seq_len, d_model)
            window_size: int
        
        Returns:
            windowed: (batch, window_size, d_model)
        """
        if x.size(1) >= window_size:
            return x[:, -window_size:, :]
        else:
            # Pad if sequence is shorter
            padding = torch.zeros(
                x.size(0), window_size - x.size(1), x.size(2),
                device=x.device, dtype=x.dtype
            )
            return torch.cat([padding, x], dim=1)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = False,
        mc_samples: int = 1
    ) -> dict:
        """
        Args:
            x: (batch, seq_len, input_dim) - Input sequence features
            return_attention: Whether to return attention weights
            mc_samples: Number of MC dropout samples for uncertainty
        
        Returns:
            dict with keys:
                - logits: (batch, num_classes)
                - probabilities: (batch, num_classes)
                - fusion_weights: (batch, 3) - α weights for [short, medium, long]
                - uncertainty: (batch,) if mc_samples > 1
                - attention: dict of attention weights if return_attention=True
        """
        batch_size = x.size(0)
        
        # Project input to d_model
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Extract windows for each scale
        x_short = self.extract_windows(x, self.short_transformer.window_size)
        x_medium = self.extract_windows(x, self.medium_transformer.window_size)
        x_long = self.extract_windows(x, self.long_transformer.window_size)
        
        # Process each scale
        h_short, attn_short = self.short_transformer(x_short)
        h_medium, attn_medium = self.medium_transformer(x_medium)
        h_long, attn_long = self.long_transformer(x_long)
        
        # Take last timestep from each scale
        h_short = h_short[:, -1, :]  # (batch, d_model)
        h_medium = h_medium[:, -1, :]
        h_long = h_long[:, -1, :]
        
        # Adaptive fusion
        h_concat = torch.cat([h_short, h_medium, h_long], dim=-1)
        fusion_weights = self.fusion_gate(h_concat)  # (batch, 3)
        
        # Weighted combination
        h_fused = (
            fusion_weights[:, 0:1] * h_short +
            fusion_weights[:, 1:2] * h_medium +
            fusion_weights[:, 2:3] * h_long
        )
        
        # MC Dropout for uncertainty estimation
        if mc_samples > 1:
            predictions = []
            for _ in range(mc_samples):
                h_dropped = self.mc_dropout(h_fused)
                logits = self.classifier(h_dropped)
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs)
            
            predictions = torch.stack(predictions)  # (mc_samples, batch, num_classes)
            mean_probs = predictions.mean(dim=0)
            uncertainty = predictions.std(dim=0).mean(dim=-1)  # (batch,)
            
            logits = torch.log(mean_probs + 1e-8)  # Back to logits
        else:
            logits = self.classifier(h_fused)
            mean_probs = F.softmax(logits, dim=-1)
            uncertainty = None
        
        # Prepare output
        output = {
            'logits': logits,
            'probabilities': mean_probs,
            'fusion_weights': fusion_weights,
        }
        
        if uncertainty is not None:
            output['uncertainty'] = uncertainty
        
        if return_attention:
            output['attention'] = {
                'short': attn_short,
                'medium': attn_medium,
                'long': attn_long
            }
        
        return output


# Example usage
if __name__ == "__main__":
    # Test the model
    batch_size = 4
    seq_len = 90  # 3 seconds at 30 FPS
    input_dim = 512  # From ResNet backbone
    
    model = MSTT_CA(
        input_dim=input_dim,
        d_model=256,
        n_heads=8,
        n_layers=4,
        num_classes=2
    )
    
    # Random input
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    output = model(x, return_attention=True, mc_samples=30)
    
    print("Logits shape:", output['logits'].shape)
    print("Probabilities shape:", output['probabilities'].shape)
    print("Fusion weights shape:", output['fusion_weights'].shape)
    print("Uncertainty shape:", output['uncertainty'].shape)
    print("\nSample fusion weights:", output['fusion_weights'][0])
    print("Sample crash probability:", output['probabilities'][0, 1].item())
    print("Sample uncertainty:", output['uncertainty'][0].item())
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
