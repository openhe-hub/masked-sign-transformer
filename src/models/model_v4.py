import torch
import torch.nn as nn
import math
from config_loader import config
from utils.constants import PART_KP_INDICES

class TemporalConv(nn.Module):
    """
    1D Temporal Convolutions with pooling to reduce sequence length,
    inspired by ssl_models_crossvideo.py.
    """
    def __init__(self, input_size, hidden_size, kernel_size=5, pool_size=2):
        super(TemporalConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size, ceil_mode=False)
        )

    def forward(self, x):
        """
        Args:
            x: [B, T, C] temporal features
        Returns:
            [B, T', C] downsampled features (T' < T)
        """
        # permute to [B, C, T] for Conv1d
        x = self.conv(x.permute(0, 2, 1))
        # permute back to [B, T', C]
        return x.permute(0, 2, 1)

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class PoseTransformerV4(nn.Module):
    """
    Part-aware Transformer with Temporal Convolutions for pose reconstruction.
    
    Architecture:
    1. Per-part Linear Projection: (T, V*2) -> (T, hidden_dim)
    2. Per-part Temporal Convolution: (B, T, hidden_dim) -> (B, T', hidden_dim)
    3. Concatenate part features -> (B, T', hidden_dim * num_parts)
    4. Project to Transformer dimension -> (B, T', hidden_dim)
    5. Global Transformer Encoder
    6. Per-part Reconstruction Head: (B, T', hidden_dim) -> (B, T', V*2)
    """
    def __init__(self):
        super().__init__()
        
        # --- Config ---
        model_cfg = config['model_v4']
        self.hidden_dim = model_cfg['hidden_dim']
        num_heads = model_cfg['num_heads']
        num_layers = model_cfg['num_layers']
        dropout = model_cfg['dropout']
        
        self.parts = list(PART_KP_INDICES.keys())
        self.part_kps = {part: len(indices) for part, indices in PART_KP_INDICES.items()}

        # --- Modules ---
        # 1. Per-part input projection
        self.proj_linear = nn.ModuleDict({
            part: nn.Linear(self.part_kps[part] * 2, self.hidden_dim)
            for part in self.parts
        })

        # 2. Per-part temporal convolution
        self.temporal_conv = nn.ModuleDict({
            part: TemporalConv(self.hidden_dim, self.hidden_dim)
            for part in self.parts
        })

        # 3. Projection from concatenated features to Transformer dimension
        concat_dim = self.hidden_dim * len(self.parts)
        self.to_transformer = nn.Linear(concat_dim, self.hidden_dim)

        # 4. Positional encoding and Transformer Encoder
        self.pos_encoder = PositionalEncoding(self.hidden_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. Per-part reconstruction head
        self.recon_head = nn.ModuleDict({
            part: nn.Linear(self.hidden_dim, self.part_kps[part] * 2)
            for part in self.parts
        })

    def forward(self, src_dict):
        """
        Args:
            src_dict: A dictionary with keys for each body part.
                      Each value is a tensor of shape (B, T, V*2).
        
        Returns:
            A dictionary with reconstructed poses for each part.
        """
        per_part_temporal = []
        
        # --- Per-Part Processing ---
        for part in self.parts:
            x = src_dict[part] # (B, T, V*2)
            
            # Reshape from (B, T, V, 2) to (B, T, V*2)
            B, T, V, C = x.shape
            x = x.reshape(B, T, -1)

            # 1. Linear projection
            x_proj = self.proj_linear[part](x) # (B, T, hidden_dim)
            
            # 2. Temporal convolution
            x_temp = self.temporal_conv[part](x_proj) # (B, T', hidden_dim)
            per_part_temporal.append(x_temp)

        # --- Global Processing ---
        # 3. Concatenate features from all parts
        # Shape: (B, T', hidden_dim * num_parts)
        concat_features = torch.cat(per_part_temporal, dim=-1)
        
        # 4. Project to Transformer's dimension
        # Shape: (B, T', hidden_dim)
        x = self.to_transformer(concat_features)

        # 5. Add positional encoding and pass through Transformer
        # Note: PositionalEncoding expects (T, B, D), so we permute
        x = x.permute(1, 0, 2) # (T', B, hidden_dim)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2) # (B, T', hidden_dim)
        
        encoded_features = self.transformer_encoder(x) # (B, T', hidden_dim)

        # --- Reconstruction ---
        recon_dict = {}
        for part in self.parts:
            # 6. Reconstruct each part from the global features
            recon_flat = self.recon_head[part](encoded_features) # (B, T', V*2)
            
            # Reshape back to (B, T', V, 2)
            B, T_prime, _ = recon_flat.shape
            V_part = self.part_kps[part]
            recon_dict[part] = recon_flat.reshape(B, T_prime, V_part, 2)
            
        return recon_dict