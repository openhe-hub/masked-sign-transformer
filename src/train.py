import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import argparse

from config_loader import config
# datasets
from datasets.dataset import PoseDataset as PoseDatasetV1
from datasets.dataset_v3 import PoseDatasetV3
# from datasets.dataset_v4 import PoseDatasetV4 # Old V4
from datasets.dataset_v5 import PoseDatasetV5 # New part-aware dataset
# models
from models.model import PoseTransformer as PoseTransformerV1
from models.model_v2 import PoseTransformerV2
from models.model_v3 import PoseTransformerV3
from models.model_v4 import PoseTransformerV4 # New part-aware model
# loss function
from losses.losses import (
    velocity_consistency_loss,
    acceleration_consistency_loss,
    total_variation_loss,
    body_bone_length_loss
)
from utils.constants import PART_KP_INDICES

def train():
    # --- 从配置中获取所有参数 ---
    device = config['training']['device']
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    num_epochs = config['training']['num_epochs']
    weight_decay = config['training']['weight_decay']
    grad_accum_steps = config['training']['gradient_accumulation_steps']
    
    experiment_name = config['training']['experiment']
    model_version = config['model'].get('version', 'v1')
    
    loss_weights = config['loss_weights']
    lambda_recon = loss_weights.get('lambda_recon', 1.0)
    lambda_vel = loss_weights.get('lambda_vel', loss_weights.get('lambda_velocity', 0.0))
    lambda_accel = loss_weights.get('lambda_accel', loss_weights.get('lambda_acceleration', 0.0))
    lambda_tv = loss_weights.get('lambda_tv', loss_weights.get('lambda_total_variation', 0.0))
    lambda_bone = loss_weights.get('lambda_bone', loss_weights.get('lambda_body_bone', 0.0))
    
    # --- Dynamic Model & Dataset Instantiation ---
    if model_version == 'v4':
        print("Instantiating Model Version: v4 (Part-Aware w/ TemporalConv)")
        model = PoseTransformerV4().to(device)
        dataset = PoseDatasetV5()
    elif model_version == 'v3':
        print("Instantiating Model Version: v3 (Asymmetric Encoder-Decoder)")
        model = PoseTransformerV4().to(device)
        dataset = PoseDatasetV3() # Assuming V3 can also be tested with new dataset logic
    elif model_version == 'v2':
        print("Instantiating Model Version: v2 (Per-Keypoint Tokenization)")
        model = PoseTransformerV2().to(device)
        dataset = PoseDatasetV1()
    else:
        print("Instantiating Model Version: v1 (Frame-level Tokenization)")
        model = PoseTransformerV1().to(device)
        dataset = PoseDatasetV1()
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params / 1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scheduler = None
    if config['lr_scheduler']['type'] == 'cosine_with_warmup':
        warmup_epochs = config['lr_scheduler']['warmup_epochs']
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6)
    
    print(f"Start training on device: {device}")
    print(f"Running experiment: {experiment_name}")
    print(f"Effective batch size: {batch_size * grad_accum_steps}")

    recon_criterion = nn.HuberLoss(reduction='none')

    for epoch in range(num_epochs):
        model.train()
        if config['lr_scheduler']['type'] == 'cosine_with_warmup' and epoch < warmup_epochs:
            warmup_lr = learning_rate * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        epoch_losses = {
            'total': 0.0,
            'recon': 0.0,
            'velocity': 0.0,
            'acceleration': 0.0,
            'tv': 0.0,
            'bone': 0.0,
        }
        
        optimizer.zero_grad()
        
        for i, (masked_sequence, input_mask, original_sequence, meta_info) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            
            # --- Move data to device (dictionary-wise) ---
            masked_sequence = {k: v.to(device) for k, v in masked_sequence.items()}
            input_mask = {k: v.to(device) for k, v in input_mask.items()}
            original_sequence = {k: v.to(device) for k, v in original_sequence.items()}
            
            subset_tensor = None
            if lambda_bone > 0:
                if isinstance(meta_info, dict):
                    meta_subset = meta_info.get('subset')
                elif isinstance(meta_info, list):
                    meta_subset = [m.get('subset') for m in meta_info if isinstance(m, dict)]
                else:
                    meta_subset = None

                if meta_subset:
                    subset_sequences = []
                    valid_subset = True
                    for seq in meta_subset:
                        if seq is None:
                            valid_subset = False
                            break
                        if isinstance(seq, torch.Tensor):
                            seq_tensor = seq.clone()
                        else:
                            frame_tensors = []
                            for frame_subset in seq:
                                if isinstance(frame_subset, torch.Tensor):
                                    frame_tensor = frame_subset.clone()
                                else:
                                    frame_tensor = torch.as_tensor(frame_subset)
                                if frame_tensor.dim() == 1:
                                    frame_tensor = frame_tensor.unsqueeze(0)
                                frame_tensors.append(frame_tensor)
                            if not frame_tensors:
                                valid_subset = False
                                break
                            seq_tensor = torch.stack(frame_tensors, dim=0)
                        subset_sequences.append(seq_tensor)
                    if valid_subset and subset_sequences:
                        subset_tensor = torch.stack(subset_sequences, dim=0)

            if subset_tensor is not None:
                if subset_tensor.dim() == 2:
                    subset_tensor = subset_tensor.unsqueeze(1)
                if subset_tensor.dim() == 3:
                    subset_tensor = subset_tensor.unsqueeze(2)

                batch_size = next(iter(masked_sequence.values())).shape[0]
                if subset_tensor.shape[0] != batch_size and subset_tensor.dim() >= 2 and subset_tensor.shape[1] == batch_size:
                    permute_dims = (1, 0) + tuple(range(2, subset_tensor.dim()))
                    subset_tensor = subset_tensor.permute(*permute_dims)

                subset_tensor = subset_tensor.contiguous().to(device=device, dtype=torch.long)

            predictions = model(masked_sequence)
            
            # --- Per-Part Loss Calculation ---
            total_recon_loss = torch.tensor(0.0, device=device)
            total_velocity_loss = torch.tensor(0.0, device=device)
            total_accel_loss = torch.tensor(0.0, device=device)
            total_tv_loss = torch.tensor(0.0, device=device)
            total_bone_loss = torch.tensor(0.0, device=device)
            num_parts = len(predictions)
            num_velocity_parts = 0
            num_accel_parts = 0
            num_tv_parts = 0
            num_bone_parts = 0

            for part, pred_part in predictions.items():
                target_part = original_sequence[part]
                mask_part = input_mask[part]
                
                # Align sequence lengths after temporal convolution
                T_pred = pred_part.shape[1]
                stride = max(1, target_part.shape[1] // T_pred)
                
                target_part_downsampled = target_part[:, ::stride, :, :][:, :T_pred, :, :]
                mask_part_downsampled = mask_part[:, ::stride, :][:, :T_pred, :]

                # Compute raw loss for all points
                loss_unreduced = recon_criterion(pred_part, target_part_downsampled)
                
                # Mask the loss: only penalize predictions for points that were masked
                # and had high confidence (which is what input_mask now represents).
                # We need to expand the mask's dimension to match the loss tensor.
                mask_expanded = mask_part_downsampled.unsqueeze(-1).expand_as(loss_unreduced)
                
                masked_loss = loss_unreduced * mask_expanded.float()
                
                # Normalize by the number of masked points to keep loss magnitude stable
                num_masked_points = mask_expanded.float().sum()
                if num_masked_points > 0:
                    part_loss = masked_loss.sum() / num_masked_points
                    total_recon_loss += part_loss

                part_indices = PART_KP_INDICES.get(part)
                n_keypoints = len(part_indices) if part_indices is not None else pred_part.shape[2]
                mask_float = mask_part_downsampled.float().unsqueeze(-1)
                pred_masked = pred_part * mask_float
                target_masked = target_part_downsampled * mask_float
                
                pred_masked_flat = pred_masked.reshape(pred_masked.shape[0], pred_masked.shape[1], -1).contiguous()
                target_masked_flat = target_masked.reshape(target_masked.shape[0], target_masked.shape[1], -1).contiguous()
                
                if lambda_vel > 0 and pred_part.shape[1] > 1:
                    total_velocity_loss += velocity_consistency_loss(
                        pred_masked_flat, target_masked_flat, n_keypoints
                    )
                    num_velocity_parts += 1
                
                if lambda_accel > 0 and pred_part.shape[1] > 2:
                    total_accel_loss += acceleration_consistency_loss(
                        pred_masked_flat, target_masked_flat, n_keypoints
                    )
                    num_accel_parts += 1
                
                if lambda_tv > 0 and pred_part.shape[1] > 1:
                    total_tv_loss += total_variation_loss(pred_masked_flat, n_keypoints)
                    num_tv_parts += 1

                if part == 'body' and lambda_bone > 0 and subset_tensor is not None:
                    subset_seq = subset_tensor
                    if subset_seq.shape[1] != target_part.shape[1]:
                        stride_subset = max(1, subset_seq.shape[1] // target_part.shape[1])
                    else:
                        stride_subset = max(1, stride)
                    subset_downsampled = subset_seq[:, ::stride_subset, ...][:, :T_pred, ...]
                    pred_body_flat = pred_part.reshape(pred_part.shape[0], pred_part.shape[1], -1)
                    target_body_flat = target_part_downsampled.reshape(target_part_downsampled.shape[0], target_part_downsampled.shape[1], -1)
                    bone_loss_val = body_bone_length_loss(
                        pred_body_flat, target_body_flat, subset_downsampled, n_keypoints
                    )
                    total_bone_loss += bone_loss_val
                    num_bone_parts += 1

            loss_recon = total_recon_loss / num_parts if num_parts > 0 else torch.tensor(0.0, device=device)
            loss_velocity = total_velocity_loss / num_velocity_parts if num_velocity_parts > 0 else torch.tensor(0.0, device=device)
            loss_accel = total_accel_loss / num_accel_parts if num_accel_parts > 0 else torch.tensor(0.0, device=device)
            loss_tv = total_tv_loss / num_tv_parts if num_tv_parts > 0 else torch.tensor(0.0, device=device)
            loss_bone = total_bone_loss / num_bone_parts if num_bone_parts > 0 else torch.tensor(0.0, device=device)
            
            total_loss = (
                lambda_recon * loss_recon
                + lambda_vel * loss_velocity
                + lambda_accel * loss_accel
                + lambda_tv * loss_tv
                + lambda_bone * loss_bone
            )
            
            total_loss = total_loss / grad_accum_steps
            total_loss.backward()
            
            if (i + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_losses['total'] += total_loss.item() * grad_accum_steps
            epoch_losses['recon'] += loss_recon.item() if isinstance(loss_recon, torch.Tensor) else loss_recon
            epoch_losses['velocity'] += loss_velocity.item() if isinstance(loss_velocity, torch.Tensor) else loss_velocity
            epoch_losses['acceleration'] += loss_accel.item() if isinstance(loss_accel, torch.Tensor) else loss_accel
            epoch_losses['tv'] += loss_tv.item() if isinstance(loss_tv, torch.Tensor) else loss_tv
            epoch_losses['bone'] += loss_bone.item() if isinstance(loss_bone, torch.Tensor) else loss_bone

        if scheduler is not None and epoch >= warmup_epochs:
            scheduler.step()

        num_batches = len(train_loader)
        print(f"Epoch {epoch+1} completed. Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        for loss_name, loss_val in epoch_losses.items():
            print(f"  - Avg {loss_name}: {loss_val / num_batches:.6f}")

        if (epoch + 1) % 10 == 0:
            checkpoints_dir = os.path.join("checkpoints", experiment_name)
            os.makedirs(checkpoints_dir, exist_ok=True)
            save_path = os.path.join(checkpoints_dir, f"{experiment_name}_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Pose Transformer model.")
    
    parser.add_argument('--experiment', type=str, help="Name of the experiment, used for saving checkpoints.")
    parser.add_argument('--model_version', type=str, help="Model version to train (e.g., 'v4_part_aware')")
    parser.add_argument('--temporal_mask_ratio', type=float, help="Override temporal masking ratio used by the dataset.")
    parser.add_argument('--spatial_mask_ratio', type=float, help="Override spatial masking ratio used by the dataset.")
    parser.add_argument('--confidence_threshold', type=float, help="Override confidence threshold used for masking low-quality keypoints.")
    # ... (other args remain the same)
    
    args, _ = parser.parse_known_args()

    if args.experiment:
        config['training']['experiment'] = args.experiment
    if args.model_version:
        config['model']['version'] = args.model_version
    if args.temporal_mask_ratio is not None:
        config['masking']['temporal_mask_ratio'] = args.temporal_mask_ratio
    if args.spatial_mask_ratio is not None:
        config['masking']['spatial_mask_ratio'] = args.spatial_mask_ratio
    if args.confidence_threshold is not None:
        config['masking']['confidence_threshold'] = args.confidence_threshold
    
    # Update config with any other command-line arguments
    # ...

    if config['training']['device'] == 'auto':
        config['training']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    train()
