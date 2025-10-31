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
# models
from models.model import PoseTransformer as PoseTransformerV1
from models.model_v2 import PoseTransformerV2
from models.model_v3 import PoseTransformerV3
# 导入我们新的损失函数
from losses.losses import (
    reconstruction_loss,
    full_sequence_reconstruction_loss,
    velocity_consistency_loss,
    acceleration_consistency_loss,
    body_bone_length_loss,
    total_variation_loss
)

def train():
    # --- 从配置中获取所有参数 ---
    device = config['training']['device']
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    num_epochs = config['training']['num_epochs']
    weight_decay = config['training']['weight_decay']
    grad_accum_steps = config['training']['gradient_accumulation_steps']
    
    n_kps = config['data']['n_kps']
    experiment_name = config['training']['experiment'] # 获取实验名称
    model_version = config['model'].get('version', 'v1')
    
    # 获取损失权重
    loss_weights = config['loss_weights']
    
    # --- Dynamic Model Instantiation ---
    if model_version == 'v5':
        print("Instantiating Model Version: v3 (Asymmetric Encoder-Decoder)")
        model = PoseTransformerV3().to(device)
        dataset = PoseDatasetV3()
    elif model_version == 'v4':
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
    
    # --- 优化器和学习率调度器 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scheduler = None
    if config['lr_scheduler']['type'] == 'cosine_with_warmup':
        warmup_epochs = config['lr_scheduler']['warmup_epochs']
        # Scheduler will start after the warmup phase
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6)
    
    print(f"Start training on device: {device}")
    print(f"Running experiment: {experiment_name}")
    print(f"Effective batch size: {batch_size * grad_accum_steps}")

    for epoch in range(num_epochs):
        model.train()
        # --- Warmup Logic ---
        if config['lr_scheduler']['type'] == 'cosine_with_warmup' and epoch < warmup_epochs:
            # Linearly increase learning rate during warmup
            warmup_lr = learning_rate * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        epoch_losses = {
            'total': 0.0, 'recon': 0.0, 'vel': 0.0, 
            'accel': 0.0, 'bone': 0.0, 'tv': 0.0
        }
        
        optimizer.zero_grad()
        
        for i, (masked_sequence, input_mask, original_sequence, subset) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            masked_sequence = masked_sequence.to(device)
            input_mask = input_mask.to(device)
            original_sequence = original_sequence.to(device)
            subset = subset.to(device)
            
            predictions = model(masked_sequence, input_mask)
            
            loss_recon = full_sequence_reconstruction_loss(predictions, original_sequence, nn.HuberLoss())
            loss_vel = velocity_consistency_loss(predictions, original_sequence, n_kps)
            loss_accel = acceleration_consistency_loss(predictions, original_sequence, n_kps)
            loss_bone = body_bone_length_loss(predictions, original_sequence, subset, n_kps)
            loss_tv = total_variation_loss(predictions, n_kps)

            total_loss = (loss_weights['lambda_recon'] * loss_recon +
                          loss_weights['lambda_vel'] * loss_vel +
                          loss_weights['lambda_accel'] * loss_accel +
                          loss_weights['lambda_bone'] * loss_bone +
                          loss_weights['lambda_tv'] * loss_tv)
            
            # --- Gradient Accumulation ---
            # Normalize loss to account for accumulation steps
            total_loss = total_loss / grad_accum_steps
            total_loss.backward()
            
            if (i + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Record losses (de-normalized)
            epoch_losses['total'] += total_loss.item() * grad_accum_steps
            epoch_losses['recon'] += loss_recon.item()
            epoch_losses['vel'] += loss_vel.item()
            epoch_losses['accel'] += loss_accel.item()
            epoch_losses['bone'] += loss_bone.item()
            epoch_losses['tv'] += loss_tv.item()

        # --- LR Scheduler Step ---
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
    parser.add_argument('--device', type=str, help="Device to train on ('cuda' or 'cpu')")
    parser.add_argument('--batch_size', type=int, help="Training batch size")
    parser.add_argument('--learning_rate', type=float, help="Optimizer learning rate")
    parser.add_argument('--num_epochs', type=int, help="Number of training epochs")
    parser.add_argument('--n_kps', type=int, help="Number of keypoints in the pose")
    
    parser.add_argument('--lambda_recon', type=float, help="Weight for reconstruction loss")
    parser.add_argument('--lambda_vel', type=float, help="Weight for velocity consistency loss")
    parser.add_argument('--lambda_accel', type=float, help="Weight for acceleration consistency loss")
    parser.add_argument('--lambda_bone', type=float, help="Weight for body bone length consistency loss")
    parser.add_argument('--lambda_tv', type=float, help="Weight for total variation loss")

    args = parser.parse_args()

    if args.experiment:
        config['training']['experiment'] = args.experiment
    else:
        if 'experiment' not in config['training']:
            config['training']['experiment'] = 'default_run'
            
    if args.device:
        config['training']['device'] = args.device
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    if args.n_kps:
        config['data']['n_kps'] = args.n_kps
        
    if args.lambda_recon is not None:
        config['loss_weights']['lambda_recon'] = args.lambda_recon
    if args.lambda_vel is not None:
        config['loss_weights']['lambda_vel'] = args.lambda_vel
    if args.lambda_accel is not None:
        config['loss_weights']['lambda_accel'] = args.lambda_accel
    if args.lambda_bone is not None:
        config['loss_weights']['lambda_bone'] = args.lambda_bone
    if args.lambda_tv is not None:
        config['loss_weights']['lambda_tv'] = args.lambda_tv

    if config['training']['device'] == 'auto':
        config['training']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    train()