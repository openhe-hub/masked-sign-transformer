import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

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
    total_variation_loss
)
from utils.constants import PART_KP_INDICES

def setup_ddp():
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

def train():
    # --- DDP Setup ---
    setup_ddp()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    is_main_process = local_rank == 0
    device = f'cuda:{local_rank}'

    # --- 从配置中获取所有参数 ---
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    num_epochs = config['training']['num_epochs']
    weight_decay = config['training']['weight_decay']
    grad_accum_steps = config['training']['gradient_accumulation_steps']
    
    experiment_name = config['training']['experiment']
    model_version = config['model'].get('version', 'v1')
    
    loss_weights = config['loss_weights']
    
    # --- Dynamic Model & Dataset Instantiation ---
    if is_main_process:
        print(f"Instantiating Model Version: {model_version}")

    if model_version == 'v4':
        model = PoseTransformerV4().to(device)
        dataset = PoseDatasetV5()
    elif model_version == 'v3':
        model = PoseTransformerV3().to(device)
        dataset = PoseDatasetV5()
    elif model_version == 'v2':
        model = PoseTransformerV2().to(device)
        dataset = PoseDatasetV1()
    else:
        model = PoseTransformerV1().to(device)
        dataset = PoseDatasetV1()
    
    model = DDP(model, device_ids=[local_rank])
    
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler)

    if is_main_process:
        num_params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {num_params / 1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scheduler = None
    if config['lr_scheduler']['type'] == 'cosine_with_warmup':
        warmup_epochs = config['lr_scheduler']['warmup_epochs']
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6)
    
    if is_main_process:
        print(f"Start training on {world_size} GPUs.")
        print(f"Running experiment: {experiment_name}")
        print(f"Effective batch size: {batch_size * grad_accum_steps * world_size}")

    recon_criterion = nn.HuberLoss(reduction='none')

    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch) # Important for shuffling

        if config['lr_scheduler']['type'] == 'cosine_with_warmup' and epoch < warmup_epochs:
            warmup_lr = learning_rate * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        epoch_losses = {'total': 0.0, 'recon': 0.0}
        
        optimizer.zero_grad()
        
        # Progress bar only on the main process
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not is_main_process)
        
        for i, (masked_sequence, input_mask, original_sequence) in enumerate(pbar):
            
            # --- Move data to device (dictionary-wise) ---
            masked_sequence = {k: v.to(device) for k, v in masked_sequence.items()}
            input_mask = {k: v.to(device) for k, v in input_mask.items()}
            original_sequence = {k: v.to(device) for k, v in original_sequence.items()}
            
            predictions = model(masked_sequence)
            
            # --- Per-Part Loss Calculation ---
            total_recon_loss = 0.0
            num_parts = len(predictions)

            for part, pred_part in predictions.items():
                target_part = original_sequence[part]
                mask_part = input_mask[part]
                
                T_pred = pred_part.shape[1]
                stride = target_part.shape[1] // T_pred
                
                target_part_downsampled = target_part[:, ::stride, :, :][:, :T_pred, :, :]
                mask_part_downsampled = mask_part[:, ::stride, :][:, :T_pred, :]

                loss_unreduced = recon_criterion(pred_part, target_part_downsampled)
                mask_expanded = mask_part_downsampled.unsqueeze(-1).expand_as(loss_unreduced)
                masked_loss = loss_unreduced * mask_expanded.float()
                
                num_masked_points = mask_expanded.float().sum()
                if num_masked_points > 0:
                    part_loss = masked_loss.sum() / num_masked_points
                    total_recon_loss += part_loss

            loss_recon = total_recon_loss / num_parts if num_parts > 0 else 0.0
            
            total_loss = loss_weights['lambda_recon'] * loss_recon
            
            total_loss = total_loss / grad_accum_steps
            total_loss.backward()
            
            if (i + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_losses['total'] += total_loss.item() * grad_accum_steps
            epoch_losses['recon'] += loss_recon.item() if isinstance(loss_recon, torch.Tensor) else loss_recon

        if scheduler is not None and epoch >= warmup_epochs:
            scheduler.step()

        if is_main_process:
            num_batches = len(train_loader)
            print(f"Epoch {epoch+1} completed. Current LR: {optimizer.param_groups[0]['lr']:.6f}")
            for loss_name, loss_val in epoch_losses.items():
                print(f"  - Avg {loss_name}: {loss_val / num_batches:.6f}")

            if (epoch + 1) % 10 == 0:
                checkpoints_dir = os.path.join("checkpoints", experiment_name)
                os.makedirs(checkpoints_dir, exist_ok=True)
                save_path = os.path.join(checkpoints_dir, f"{experiment_name}_epoch_{epoch+1}.pth")
                # Save the original model's state dict
                torch.save(model.module.state_dict(), save_path)
                print(f"Model saved to {save_path}")
    
    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Pose Transformer model with DDP.")
    
    parser.add_argument('--experiment', type=str, help="Name of the experiment, used for saving checkpoints.")
    parser.add_argument('--model_version', type=str, help="Model version to train (e.g., 'v4')")
    
    args, _ = parser.parse_known_args()

    if args.experiment:
        config['training']['experiment'] = args.experiment
    if args.model_version:
        config['model']['version'] = args.model_version
    
    train()
