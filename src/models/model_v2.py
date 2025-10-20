import torch
import torch.nn as nn
import math

from config_loader import config

class SpatioTemporalPositionalEncoding(nn.Module):
    """
    Spatio-Temporal Positional Encoding for pose sequences.
    
    This module generates a combined positional encoding for a sequence of poses.
    It adds a learnable spatial embedding for each keypoint to a fixed temporal
    sine-cosine encoding for each frame.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000, n_kps=128):
        super(SpatioTemporalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Temporal Positional Encoding (fixed sine-cosine)
        pe_temporal = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe_temporal[:, 0::2] = torch.sin(position * div_term)
        pe_temporal[:, 1::2] = torch.cos(position * div_term)
        # Shape: (max_len, d_model) -> unsqueeze for broadcasting -> (1, max_len, 1, d_model)
        self.register_buffer('pe_temporal', pe_temporal.unsqueeze(0).unsqueeze(2))

        # Spatial Positional Encoding (learnable embedding)
        # One embedding for each keypoint
        self.pe_spatial = nn.Embedding(n_kps, d_model)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of keypoint embeddings.
                        Shape: (batch_size, seq_len, n_kps, d_model)
        """
        B, T, K, D = x.shape
        
        # Add temporal encoding: self.pe_temporal is (1, max_len, 1, d_model)
        # It will be broadcasted across batch and keypoint dimensions.
        x = x + self.pe_temporal[:, :T, :, :]
        
        # Add spatial encoding: self.pe_spatial.weight is (n_kps, d_model)
        # It will be broadcasted across batch and time dimensions.
        x = x + self.pe_spatial.weight.unsqueeze(0).unsqueeze(0)
        
        return self.dropout(x)

class PoseTransformerV2(nn.Module):
    def __init__(self):
        super(PoseTransformerV2, self).__init__()
        
        n_kps = config['data']['n_kps']
        # In V2, each keypoint (x, y) is a token, so features are 2
        features_per_kp = 2 
        d_model = config['model']['d_model']
        n_head = config['model']['n_head']
        num_encoder_layers = config['model']['num_encoder_layers']
        dim_feedforward = config['model']['dim_feedforward']
        dropout = config['model']['dropout']
        
        # 1. Per-Keypoint Embedding
        # Projects each keypoint's (x, y) coordinates into the d_model space
        self.keypoint_embedder = nn.Linear(features_per_kp, d_model)
        
        # 2. Spatio-Temporal Positional Encoding
        self.pos_encoder = SpatioTemporalPositionalEncoding(d_model, dropout, n_kps=n_kps)
        
        # 3. Transformer Encoder
        # batch_first=True is more intuitive with this tokenization
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # 4. Output Head
        # Projects from d_model back to the (x, y) coordinate space
        self.output_head = nn.Linear(d_model, features_per_kp)
        
        self.d_model = d_model

        # 5. Learnable Mask Token
        # This token will replace the embeddings of masked keypoints
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, src, mask):
        """
        Args:
            src (Tensor): The masked input sequence from the dataloader.
                          Shape: (batch_size, seq_len, n_kps * 2)
            mask (Tensor): The boolean mask indicating which keypoints were masked.
                           Shape: (batch_size, seq_len, n_kps)
        """
        B, T, _ = src.shape
        K = config['data']['n_kps']
        F = 2 # features per keypoint (x, y)

        # --- Input Preparation ---
        # 1. Reshape input to (B, T, K, F) to separate keypoints
        src = src.reshape(B, T, K, F)
        
        # 2. Embed each keypoint to d_model
        src_embedded = self.keypoint_embedder(src) # Shape: (B, T, K, d_model)
        
        # 3. Apply masking
        # mask is (B, T, K), we need to expand it for broadcasting
        expanded_mask = mask.unsqueeze(-1) # Shape: (B, T, K, 1)
        # Replace embeddings of masked keypoints with the learnable mask_token
        src_embedded = torch.where(expanded_mask, self.mask_token, src_embedded)
        
        # 4. Add spatio-temporal positional encodings
        src_with_pos = self.pos_encoder(src_embedded)
        
        # 5. Flatten the spatial and temporal dimensions for the Transformer
        # The Transformer expects a sequence of tokens: (B, T*K, d_model)
        transformer_input = src_with_pos.reshape(B, T * K, self.d_model)
        
        # --- Transformer Encoder ---
        # No src_key_padding_mask is needed as we use a learnable mask token
        output = self.transformer_encoder(transformer_input) # Shape: (B, T*K, d_model)
        
        # --- Output Processing ---
        # 1. Project back to the coordinate space
        output = self.output_head(output) # Shape: (B, T*K, F)
        
        # 2. Reshape back to the original flattened format (B, T, K*F)
        # This ensures compatibility with the existing loss functions
        output = output.reshape(B, T, K * F)
        
        return output
