"""
Stage-1 Pose Encoder with Cross-Video Contrastive Learning

Architecture:
1. Per-part Linear projection (V*3 -> hidden_dim)
2. Per-part Temporal Convolutions (reduce temporal dimension)
3. Concatenate part features
4. Transformer Encoder (global temporal attention)
5. Mean pooling -> block embeddings
6. Projection head -> cross-video contrastive learning

Key differences from SignCL:
- Cross-video contrastive loss (same gloss = positive, different gloss = negative)
- Designed for isolated sign datasets (one video = one gloss)
- SignCL's temporal contrasts don't work for isolated signs (all frames = same sign)
- No data augmentation needed
- Reduced temporal density via conv pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TemporalConv(nn.Module):
    """
    1D Temporal Convolutions with pooling to reduce sequence length.

    This is critical for preventing local overfitting and forcing
    the model to learn semantic structure.
    """
    def __init__(self, input_size, hidden_size, conv_type=2):
        super(TemporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_type = conv_type

        # conv_type controls depth: 0=shallow, 1=medium, 2=deep
        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                # Pooling layer reduces temporal dimension
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)

    def forward(self, x):
        """
        Args:
            x: [B, T, C] temporal features
        Returns:
            [B, T', C] downsampled features (T' < T)
        """
        x = self.temporal_conv(x.permute(0, 2, 1))  # [B, C, T] -> [B, C, T']
        return x.permute(0, 2, 1)  # [B, T', C]


class CrossVideoContrastiveLoss(nn.Module):
    """
    Cross-Video Contrastive Loss (InfoNCE) for isolated sign datasets.

    Key idea: Different videos of the same gloss should have similar embeddings (positive pairs),
    while videos of different glosses should be dissimilar (negative pairs).

    This is designed for isolated sign datasets where:
    - Each video contains exactly one sign/gloss
    - SignCL's temporal contrasts don't work (all frames = same sign)

    Args:
        temperature: Temperature parameter for softmax scaling (default: 0.07)
    """
    def __init__(self, temperature=0.07):
        super(CrossVideoContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        Compute InfoNCE loss for cross-video contrasts.

        Args:
            embeddings: [B, D] video embeddings (normalized inside)
            labels: [B] gloss labels for each video

        Returns:
            Scalar loss
        """
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Compute similarity matrix: [B, B]
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create positive pair mask: mask[i,j] = 1 if labels[i] == labels[j]
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()

        # Remove self-similarity (diagonal)
        mask = mask - torch.eye(mask.size(0), device=mask.device)

        # For each sample, compute InfoNCE loss
        # Numerator: sum of exp(sim) for positive pairs
        # Denominator: sum of exp(sim) for all pairs (excluding self)
        exp_sim = torch.exp(sim_matrix)

        # Mask out diagonal (self-similarity)
        exp_sim = exp_sim * (1 - torch.eye(mask.size(0), device=mask.device))

        # Sum of positive pairs
        pos_sum = (exp_sim * mask).sum(dim=1)

        # Sum of all pairs (excluding self)
        all_sum = exp_sim.sum(dim=1)

        # InfoNCE loss: -log(sum_pos / sum_all)
        # Add epsilon to avoid log(0)
        loss = -torch.log(pos_sum / (all_sum + 1e-8) + 1e-8)

        # Only compute loss for samples that have at least one positive pair
        valid_mask = (mask.sum(dim=1) > 0)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        return loss[valid_mask].mean()


class PoseBlockEncoderSignCL(nn.Module):
    """
    Pose block encoder with Transformer + Cross-Video Contrastive Learning.

    Architecture:
    1. Per-part linear projection + temporal conv (reduce density)
    2. Concatenate all parts
    3. Transformer encoder (global attention)
    4. Mean pooling -> block embeddings [B, embed_dim]
    5. Projection head -> [B, projection_dim] for contrastive learning

    For training: Uses cross-video contrastive loss (same gloss = positive)
    """
    def __init__(self,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 projection_dim: int = 128,
                 use_conf_weight: bool = True):
        super().__init__()

        self.modes = ['body', 'left', 'right', 'face_all']
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.use_conf_weight = use_conf_weight

        # Number of keypoints per part (extracted by load_part_kp in datasets.py)
        # body: [0] + [3-10] = 9 keypoints (nose + upper body)
        # left/right: [91:112] or [112:133] = 21 keypoints (hand keypoints)
        # face_all: selected face landmarks = 17 keypoints (simplified face representation)
        self.num_kp = {'body': 9, 'left': 21, 'right': 21, 'face_all': 18}

        # Per-part projection: flatten keypoints to hidden_dim
        self.proj_linear = nn.ModuleDict({
            part: nn.Linear(self.num_kp[part] * 3, hidden_dim)
            for part in self.modes
        })

        # Per-part temporal convolution (reduce temporal density)
        # conv_type=1: K5->P2 reduces by ~2.5x (20→8 frames)
        # Note: conv_type=2 would reduce too much for isolated signs (20→2 frames)
        self.temporal_conv = nn.ModuleDict({
            part: TemporalConv(hidden_dim, hidden_dim, conv_type=1)
            for part in self.modes
        })

        # Concatenated dimension
        self.concat_dim = hidden_dim * len(self.modes)

        # Projection to Transformer dimension
        self.to_transformer = nn.Linear(self.concat_dim, hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.embed_dim = hidden_dim

        # Reconstruction head (for reconstruction loss)
        self.recon_head = nn.ModuleDict({
            part: nn.Linear(hidden_dim, self.num_kp[part] * 3)
            for part in self.modes
        })

        # Projection head for cross-video contrastive learning
        # Two-layer MLP: hidden_dim -> hidden_dim -> projection_dim
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )

    def encode(self, inputs_by_part, return_sequence=False):
        """
        Encode pose block to embeddings.

        Args:
            inputs_by_part: dict with keys ['body','left','right','face_all']
                each tensor shaped [B, K, V, 3] with (x, y, conf)
            return_sequence: If True, return [B, T', hidden_dim] for SignCL
                            If False, return [B, hidden_dim] (mean pooled)

        Returns:
            If return_sequence=False: (block_embeddings [B, hidden_dim], per_part_features dict)
            If return_sequence=True: (sequence_embeddings [B, T', hidden_dim], per_part_features dict)
        """
        per_part_features = {}
        per_part_temporal = []

        for part in self.modes:
            x = inputs_by_part[part]  # [B, K, V, 3]
            B, K, V, C = x.shape

            # Flatten keypoints: [B, K, V*3]
            x_flat = x.reshape(B, K, -1)

            # Project to hidden_dim: [B, K, hidden_dim]
            x_proj = self.proj_linear[part](x_flat)

            # Temporal convolution (reduces K -> K')
            x_temp = self.temporal_conv[part](x_proj)  # [B, K', hidden_dim]

            per_part_features[part] = x_temp
            per_part_temporal.append(x_temp)

        # Concatenate all parts: [B, K', hidden_dim * 4]
        concat_features = torch.cat(per_part_temporal, dim=-1)

        # Project to transformer dim: [B, K', hidden_dim]
        x = self.to_transformer(concat_features)

        # Add positional encoding
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)  # PositionalEncoding expects [T, B, D]

        # Transformer encoder
        x = self.transformer_encoder(x)  # [B, K', hidden_dim]

        if return_sequence:
            return x, per_part_features
        else:
            # Mean pooling over time
            block_embeddings = x.mean(dim=1)  # [B, hidden_dim]
            return block_embeddings, per_part_features

    def reconstruct(self, per_part_features):
        """
        Reconstruct poses from per-part features.

        Args:
            per_part_features: dict of [B, K', hidden_dim]

        Returns:
            dict of reconstructed poses [B, K', V, 3]
        """
        recon_by_part = {}
        for part in self.modes:
            feat = per_part_features[part]  # [B, K', hidden_dim]
            pred_flat = self.recon_head[part](feat)  # [B, K', V*3]

            B, Kp, _ = pred_flat.shape
            V = self.num_kp[part]
            pred = pred_flat.reshape(B, Kp, V, 3)
            recon_by_part[part] = pred

        return recon_by_part

    def compute_reconstruction_loss(self,
                                   recon_by_part,
                                   inputs_by_part,
                                   masks_by_part,
                                   confs_by_part=None,
                                   reconstruct_xy_only=True,
                                   conf_threshold=0.3):
        """
        Compute masked reconstruction loss with HYBRID masking:
        1. Random masking (temporal/denoising) - from masks_by_part
        2. Confidence-based masking - only predict valid joints (conf > threshold)

        Note: After temporal convolution, sequence length changes K -> K'.
        We need to handle this mismatch.

        Args:
            conf_threshold: Confidence threshold for valid joints (default: 0.3)
                           Joints with conf ≤ threshold are masked out
        """
        total_loss = 0.0
        total_count = 0.0
        eps = 1e-8

        for part in self.modes:
            pred = recon_by_part[part]  # [B, K', V, 3]
            target_full = inputs_by_part[part]  # [B, K, V, 3]
            mask_random_full = masks_by_part[part]  # [B, K, V] - random mask (20-30%)

            B, Kp, V, C_pred = pred.shape
            B, K, V, C_target = target_full.shape

            # Downsample target and mask to match pred
            # Simple approach: take every stride-th frame
            stride = K // Kp if Kp > 0 else 1
            target = target_full[:, ::stride, :, :][:, :Kp, :, :]  # [B, K', V, 3]
            mask_random = mask_random_full[:, ::stride, :][:, :Kp, :]  # [B, K', V] - random mask

            # Extract confidence from target (3rd channel)
            conf = target[..., 2]  # [B, K', V]

            # Confidence-based mask: Only predict joints with conf > threshold
            mask_valid = (conf > conf_threshold).float()  # [B, K', V]

            # HYBRID MASK: Combine random mask AND confidence mask
            # Only predict joints that are:
            #   1. Randomly selected (denoising/temporal masking)
            #   2. AND have high confidence (valid joints)
            mask_combined = mask_random.float() * mask_valid  # [B, K', V]
            mask_b = mask_combined[..., None]  # [B, K', V, 1]

            # Compute squared error
            if reconstruct_xy_only:
                target_xy = target[..., :2]
                pred_xy = pred[..., :2]
                sqerr = (pred_xy - target_xy) ** 2
                num_channels = 2
            else:
                sqerr = (pred - target) ** 2
                num_channels = 3

            # Apply combined mask
            sqerr = sqerr * mask_b

            # Optional: Additional confidence weighting (if enabled)
            if self.use_conf_weight and (confs_by_part is not None) and (part in confs_by_part):
                # Use confidence from confs_by_part if provided, otherwise from target
                conf_weight = confs_by_part[part][:, ::stride, :][:, :Kp, :] if part in confs_by_part else conf
                w = conf_weight[..., None].float()
                weighted_sqerr = sqerr * w
                loss_part = weighted_sqerr.sum() / (mask_combined.sum() * num_channels + eps)
            else:
                loss_part = sqerr.sum() / (mask_combined.sum() * num_channels + eps)

            total_loss += loss_part
            total_count += 1

        return total_loss / max(1, total_count)

    def forward(self,
                inputs_by_part: dict,
                masks_by_part: dict,
                confs_by_part: dict = None,
                labels: torch.Tensor = None,
                num_videos: int = None,
                reconstruct_xy_only: bool = True,
                compute_cl_loss: bool = True,
                temperature: float = 0.07,
                conf_threshold: float = 0.3):
        """
        Forward pass with reconstruction + cross-video contrastive loss.

        Args:
            inputs_by_part: dict of [B, K, V, 3] where B = num_videos * num_blocks_per_video
            masks_by_part: dict of [B, K, V] random masks for temporal/denoising masking
            confs_by_part: dict of [B, K, V] confidence weights (optional)
            labels: [num_videos] gloss labels for cross-video CL (VIDEO-level, not block-level!)
            num_videos: Number of videos in batch (for aggregating blocks → videos)
            reconstruct_xy_only: Only reconstruct x,y (not confidence)
            compute_cl_loss: Whether to compute cross-video CL loss
            temperature: Temperature for InfoNCE loss
            conf_threshold: Confidence threshold for valid joints (default: 0.3)
                           Reconstruction only computed for joints with conf > threshold

        Returns:
            dict with:
                - loss: total loss
                - recon_loss: reconstruction loss (with hybrid masking)
                - cl_loss: contrastive loss (if compute_cl_loss=True)
                - block_embeddings: [B, hidden_dim]
        """
        # Get block embeddings (mean-pooled over time)
        block_embeds, per_part_features = self.encode(inputs_by_part, return_sequence=False)
        # block_embeds: [B, hidden_dim] where B = num_videos * num_blocks_per_video

        # Reconstruction loss (computed at BLOCK level with HYBRID masking)
        # Hybrid masking = random mask (temporal/denoising) + confidence mask (valid joints)
        recon_by_part = self.reconstruct(per_part_features)
        recon_loss = self.compute_reconstruction_loss(
            recon_by_part, inputs_by_part, masks_by_part, confs_by_part,
            reconstruct_xy_only, conf_threshold
        )

        # Cross-video contrastive loss (computed at VIDEO level - correct granularity!)
        if compute_cl_loss and labels is not None and num_videos is not None:
            # Aggregate blocks to video-level embeddings
            num_blocks_per_video = block_embeds.shape[0] // num_videos

            # Reshape: [num_videos * num_blocks, D] → [num_videos, num_blocks, D]
            video_block_embeds = block_embeds.view(num_videos, num_blocks_per_video, -1)

            # Mean-pool blocks within each video: [num_videos, num_blocks, D] → [num_videos, D]
            video_embeds = video_block_embeds.mean(dim=1)

            # Project video-level embeddings for contrastive learning
            projections = self.projection_head(video_embeds)  # [num_videos, projection_dim]

            # Compute InfoNCE loss on VIDEO-level embeddings with VIDEO-level labels
            cl_criterion = CrossVideoContrastiveLoss(temperature=temperature)
            cl_loss = cl_criterion(projections, labels)
        else:
            cl_loss = torch.tensor(0.0, device=recon_loss.device)

        # Total loss (CL weight will be applied in training script)
        total_loss = recon_loss + cl_loss

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'cl_loss': cl_loss,
            'block_embeddings': block_embeds  # Still return block-level for export
        }


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)