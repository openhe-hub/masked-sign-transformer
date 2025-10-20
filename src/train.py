import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse

from config_loader import config
from dataset import PoseDataset
from model import PoseTransformer
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
    n_kps = config['data']['n_kps']
    experiment_name = config['training']['experiment'] # 获取实验名称
    
    # 获取损失权重
    loss_weights = config['loss_weights']
    
    # --- 数据和模型准备 (不变) ---
    dataset = PoseDataset()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = PoseTransformer().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params / 1e6:.2f}M")
    
    # --- 损失函数和优化器 (更新) ---
    # 使用导师建议的Huber Loss作为重建损失
    recon_criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print(f"Start training on device: {device}")
    print(f"Running experiment: {experiment_name}")
    for epoch in range(num_epochs):
        model.train()
        # 用于记录每个loss分量的字典
        epoch_losses = {
            'total': 0.0, 'recon': 0.0, 'vel': 0.0, 
            'accel': 0.0, 'bone': 0.0, 'tv': 0.0
        }
        
        # 假设你的Dataset现在返回 (masked_seq, input_mask, original_seq, loss_mask)
        for masked_sequence, input_mask, original_sequence, subset in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            masked_sequence = masked_sequence.to(device)
            input_mask = input_mask.to(device)
            original_sequence = original_sequence.to(device)
            subset = subset.to(device)

            optimizer.zero_grad()
            
            # --- 核心变化: 计算多目标损失 ---
            predictions = model(masked_sequence, input_mask)
            
            # 1. 重建损失
            loss_recon = full_sequence_reconstruction_loss(predictions, original_sequence, recon_criterion)
            
            # --- 以下损失在整个序列上计算，以保证全局平滑性和一致性 ---
            
            # 2. 速度一致性损失
            loss_vel = velocity_consistency_loss(predictions, original_sequence, n_kps)
            
            # 3. 加速度一致性损失
            loss_accel = acceleration_consistency_loss(predictions, original_sequence, n_kps)
            
            # 4. 骨骼长度一致性损失
            loss_bone = body_bone_length_loss(predictions, original_sequence, subset, n_kps)
            
            # 5. 总变差正则化 (只作用于预测)
            loss_tv = total_variation_loss(predictions, n_kps)

            # --- 组合总损失 ---
            total_loss = (loss_weights['lambda_recon'] * loss_recon +
                          loss_weights['lambda_vel'] * loss_vel +
                          loss_weights['lambda_accel'] * loss_accel +
                          loss_weights['lambda_bone'] * loss_bone +
                          loss_weights['lambda_tv'] * loss_tv)
            
            total_loss.backward()
            optimizer.step()
            
            # 记录各个损失的值
            epoch_losses['total'] += total_loss.item()
            epoch_losses['recon'] += loss_recon.item()
            epoch_losses['vel'] += loss_vel.item()
            epoch_losses['accel'] += loss_accel.item()
            epoch_losses['bone'] += loss_bone.item()
            epoch_losses['tv'] += loss_tv.item()

        # 打印每个epoch的平均损失
        num_batches = len(train_loader)
        print(f"Epoch {epoch+1} completed. Losses:")
        for loss_name, loss_val in epoch_losses.items():
            print(f"  - Avg {loss_name}: {loss_val / num_batches:.6f}")

        # --- 保存模型 (更新) ---
        if (epoch + 1) % 10 == 0:
            checkpoints_dir = os.path.join("checkpoints", experiment_name)
            os.makedirs(checkpoints_dir, exist_ok=True)
            save_path = os.path.join(checkpoints_dir, f"{experiment_name}_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Pose Transformer model.")
    
    # 添加命令行参数以覆盖config.toml中的设置
    parser.add_argument('--experiment', type=str, help="Name of the experiment, used for saving checkpoints.")
    parser.add_argument('--device', type=str, help="Device to train on ('cuda' or 'cpu')")
    parser.add_argument('--batch_size', type=int, help="Training batch size")
    parser.add_argument('--learning_rate', type=float, help="Optimizer learning rate")
    parser.add_argument('--num_epochs', type=int, help="Number of training epochs")
    parser.add_argument('--n_kps', type=int, help="Number of keypoints in the pose")
    
    # 添加用于覆盖损失权重的参数
    parser.add_argument('--lambda_recon', type=float, help="Weight for reconstruction loss")
    parser.add_argument('--lambda_vel', type=float, help="Weight for velocity consistency loss")
    parser.add_argument('--lambda_accel', type=float, help="Weight for acceleration consistency loss")
    parser.add_argument('--lambda_bone', type=float, help="Weight for body bone length consistency loss")
    parser.add_argument('--lambda_tv', type=float, help="Weight for total variation loss")

    args = parser.parse_args()

    # --- 使用命令行参数覆盖配置 ---
    # 只有当用户提供了命令行参数时，才覆盖config中的值
    if args.experiment:
        config['training']['experiment'] = args.experiment
    else:
        # 如果config.toml和命令行都没有提供，则设置一个默认值
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

    # 确保设备设置在torch中生效
    if config['training']['device'] == 'auto':
        config['training']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    train()