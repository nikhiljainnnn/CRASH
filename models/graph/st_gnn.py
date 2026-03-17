"""
Spatio-Temporal Graph Neural Network for Vehicle Interaction Modeling (ST-GNN-VIM)
Models how crash risk propagates through vehicle interaction networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

try:
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import add_self_loops, softmax
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    # Provide a stub base class so the file can be imported without torch_geometric
    class MessagePassing(nn.Module):
        def __init__(self, **kwargs): super().__init__()
    def add_self_loops(*args, **kwargs): raise ImportError("torch_geometric is required for ST_GNN. Install via: pip install torch-geometric")
    def softmax(*args, **kwargs): raise ImportError("torch_geometric is required for ST_GNN. Install via: pip install torch-geometric")


class GraphAttentionLayer(MessagePassing):
    """Graph Attention Layer with edge features"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.1
    ):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        
        # Linear transformations
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_edge = nn.Linear(edge_dim, heads, bias=False)
        
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (num_nodes, in_channels)
            edge_index: (2, num_edges)
            edge_attr: (num_edges, edge_dim)
        
        Returns:
            out: (num_nodes, heads * out_channels) if concat else (num_nodes, out_channels)
        """
        # Linear transformation
        x = self.lin(x).view(-1, self.heads, self.out_channels)
        
        # Start propagating messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Concatenate or average heads
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        out = out + self.bias
        
        return out
    
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int]
    ) -> torch.Tensor:
        """
        Compute messages from j to i
        
        Args:
            x_i: (num_edges, heads, out_channels) - target nodes
            x_j: (num_edges, heads, out_channels) - source nodes
            edge_attr: (num_edges, edge_dim)
            index: Edge indices
        """
        # Compute attention scores
        alpha_src = (x_j * self.att_src).sum(dim=-1)  # (num_edges, heads)
        alpha_dst = (x_i * self.att_dst).sum(dim=-1)  # (num_edges, heads)
        
        # Add edge features to attention
        if edge_attr is not None:
            alpha_edge = self.att_edge(edge_attr)  # (num_edges, heads)
            alpha = alpha_src + alpha_dst + alpha_edge
        else:
            alpha = alpha_src + alpha_dst
        
        alpha = F.leaky_relu(alpha, 0.2)
        
        # Softmax over edges for each node
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Weight messages by attention
        return x_j * alpha.unsqueeze(-1)


class TemporalGRU(nn.Module):
    """GRU for temporal evolution of graph states"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (num_nodes, hidden_dim) - Current node features
            h_prev: (num_nodes, hidden_dim) - Previous hidden state
        
        Returns:
            h: (num_nodes, hidden_dim) - Updated hidden state
        """
        if h_prev is None:
            h_prev = torch.zeros_like(x)
        
        h = self.gru(x, h_prev)
        return h


class ST_GNN_Layer(nn.Module):
    """Single Spatio-Temporal GNN layer"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Spatial processing (GAT)
        self.spatial_conv = GraphAttentionLayer(
            in_channels, out_channels, edge_dim, heads, concat=True, dropout=dropout
        )
        
        # Temporal processing (GRU)
        self.temporal_gru = TemporalGRU(out_channels * heads)
        
        # Layer normalization
        self.norm = nn.LayerNorm(out_channels * heads)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (num_nodes, in_channels)
            edge_index: (2, num_edges)
            edge_attr: (num_edges, edge_dim)
            h_prev: (num_nodes, out_channels * heads) - Previous temporal state
        
        Returns:
            out: (num_nodes, out_channels * heads)
            h: (num_nodes, out_channels * heads) - New temporal state
        """
        # Spatial convolution
        x = self.spatial_conv(x, edge_index, edge_attr)
        x = self.dropout(x)
        
        # Temporal evolution
        h = self.temporal_gru(x, h_prev)
        
        # Normalization
        out = self.norm(h)
        
        return out, h


class ST_GNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network for Vehicle Interaction Modeling
    
    Models:
    - Spatial: How vehicles influence each other at current timestep
    - Temporal: How influence evolves over time
    - Risk propagation: How collision risk spreads through network
    """
    
    def __init__(
        self,
        node_feature_dim: int = 16,  # [position, velocity, acceleration, heading, size, class]
        edge_feature_dim: int = 8,   # [distance, relative_velocity, heading_diff, TTC]
        hidden_dim: int = 128,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        output_dim: int = 64
    ):
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_feature_dim, hidden_dim)
        
        # Spatio-temporal layers
        self.st_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * heads
            self.st_layers.append(
                ST_GNN_Layer(in_dim, hidden_dim, hidden_dim, heads, dropout)
            )
        
        # Risk propagation head
        self.risk_propagation = nn.Sequential(
            nn.Linear(hidden_dim * heads, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Collision prediction head
        self.collision_head = nn.Sequential(
            nn.Linear(hidden_dim * heads, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def build_graph(
        self,
        vehicle_features: torch.Tensor,
        spatial_radius: float = 30.0,
        heading_threshold: float = 45.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build interaction graph from vehicle features
        
        Args:
            vehicle_features: (batch, num_vehicles, feature_dim)
                Features: [x, y, vx, vy, ax, ay, heading, length, width, class, ...]
            spatial_radius: Maximum distance for edge creation (meters)
            heading_threshold: Maximum heading difference for same-direction edges (degrees)
        
        Returns:
            node_features: (num_nodes, node_feature_dim)
            edge_index: (2, num_edges)
            edge_features: (num_edges, edge_feature_dim)
        """
        batch_size, num_vehicles, _ = vehicle_features.shape
        
        # Flatten batch dimension
        node_features = vehicle_features.view(-1, vehicle_features.size(-1))
        
        edge_list = []
        edge_features_list = []
        
        for b in range(batch_size):
            offset = b * num_vehicles
            
            for i in range(num_vehicles):
                for j in range(num_vehicles):
                    if i == j:
                        continue
                    
                    # Extract vehicle positions and features
                    vi = vehicle_features[b, i]
                    vj = vehicle_features[b, j]
                    
                    # Calculate distance
                    pos_i = vi[:2]  # (x, y)
                    pos_j = vj[:2]
                    distance = torch.norm(pos_i - pos_j)
                    
                    # Calculate relative velocity
                    vel_i = vi[2:4]  # (vx, vy)
                    vel_j = vj[2:4]
                    rel_vel = torch.norm(vel_j - vel_i)
                    
                    # Calculate heading difference
                    heading_i = vi[6]
                    heading_j = vj[6]
                    heading_diff = torch.abs(heading_i - heading_j)
                    heading_diff = torch.min(heading_diff, 360 - heading_diff)
                    
                    # Check edge creation criteria
                    spatial_criteria = distance < spatial_radius
                    heading_criteria = heading_diff < heading_threshold
                    
                    if spatial_criteria or heading_criteria:
                        # Add edge
                        edge_list.append([offset + i, offset + j])
                        
                        # Calculate edge features
                        # Time-to-collision
                        if rel_vel > 1e-3:
                            ttc = distance / rel_vel
                        else:
                            ttc = torch.tensor(float('inf'))
                        
                        edge_feat = torch.stack([
                            distance,
                            rel_vel,
                            heading_diff,
                            torch.clamp(ttc, 0, 100),  # Clamp TTC
                            (heading_diff < 30).float(),  # Same lane indicator
                            (distance < 10).float(),  # Close proximity
                            torch.norm(vel_i),  # Speed of vehicle i
                            torch.norm(vel_j)   # Speed of vehicle j
                        ])
                        
                        edge_features_list.append(edge_feat)
        
        if len(edge_list) == 0:
            # No edges - create self-loops
            num_nodes = batch_size * num_vehicles
            edge_index = torch.arange(num_nodes).repeat(2, 1)
            edge_features = torch.zeros(num_nodes, self.edge_feature_dim)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            edge_features = torch.stack(edge_features_list)
        
        return node_features, edge_index, edge_features
    
    def forward(
        self,
        vehicle_features: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        h_prev: Optional[list] = None,
        return_risk: bool = True
    ) -> dict:
        """
        Args:
            vehicle_features: (batch, num_vehicles, feature_dim)
            edge_index: (2, num_edges) - Optional pre-computed edges
            edge_attr: (num_edges, edge_dim) - Optional pre-computed edge features
            h_prev: List of previous hidden states for each layer
            return_risk: Whether to compute per-node risk scores
        
        Returns:
            dict with keys:
                - node_embeddings: (num_nodes, hidden_dim * heads)
                - risk_scores: (num_nodes, 1) if return_risk=True
                - collision_features: (num_nodes, output_dim)
                - hidden_states: List of hidden states for each layer
        """
        # Build graph if not provided
        if edge_index is None or edge_attr is None:
            node_features, edge_index, edge_attr = self.build_graph(vehicle_features)
        else:
            batch_size, num_vehicles, _ = vehicle_features.shape
            node_features = vehicle_features.view(-1, vehicle_features.size(-1))
        
        # Encode features
        x = self.node_encoder(node_features)
        edge_attr_encoded = self.edge_encoder(edge_attr)
        
        # Initialize hidden states
        if h_prev is None:
            h_prev = [None] * self.num_layers
        
        # Process through ST-GNN layers
        hidden_states = []
        for i, layer in enumerate(self.st_layers):
            x, h = layer(x, edge_index, edge_attr_encoded, h_prev[i])
            hidden_states.append(h)
        
        # Compute outputs
        collision_features = self.collision_head(x)
        
        output = {
            'node_embeddings': x,
            'collision_features': collision_features,
            'hidden_states': hidden_states
        }
        
        if return_risk:
            risk_scores = self.risk_propagation(x)
            output['risk_scores'] = risk_scores
        
        return output


# Example usage
if __name__ == "__main__":
    batch_size = 2
    num_vehicles = 10
    feature_dim = 16
    
    # Random vehicle features
    # [x, y, vx, vy, ax, ay, heading, length, width, class, ...]
    vehicle_features = torch.randn(batch_size, num_vehicles, feature_dim)
    
    # Initialize model
    model = ST_GNN(
        node_feature_dim=feature_dim,
        edge_feature_dim=8,
        hidden_dim=128,
        num_layers=4,
        heads=4
    )
    
    # Forward pass
    output = model(vehicle_features, return_risk=True)
    
    print("Node embeddings shape:", output['node_embeddings'].shape)
    print("Risk scores shape:", output['risk_scores'].shape)
    print("Collision features shape:", output['collision_features'].shape)
    print("\nSample risk scores:", output['risk_scores'][:5].squeeze())
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
