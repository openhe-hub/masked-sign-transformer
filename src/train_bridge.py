import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_loader import config
from datasets.dataset_v6_bridge import PoseBridgeDatasetV6
from models.model_v6_bridge import PoseBridgeTransformerV6
from utils.constants import get_part_kp_indices


def reshape_gap_tensor(tensor, n_kps):
    B, T, _ = tensor.shape
    return tensor.view(B, T, n_kps, 3)


def velocity_loss(pred, target, n_kps):
    if pred.size(1) < 2:
        return pred.sum() * 0.0
    pred_struct = reshape_gap_tensor(pred, n_kps)
    target_struct = reshape_gap_tensor(target, n_kps)
    pred_vel = pred_struct[:, 1:] - pred_struct[:, :-1]
    target_vel = target_struct[:, 1:] - target_struct[:, :-1]
    return F.mse_loss(pred_vel, target_vel)


def acceleration_loss(pred, target, n_kps):
    if pred.size(1) < 3:
        return pred.sum() * 0.0
    pred_struct = reshape_gap_tensor(pred, n_kps)
    target_struct = reshape_gap_tensor(target, n_kps)
    pred_vel = pred_struct[:, 1:] - pred_struct[:, :-1]
    target_vel = target_struct[:, 1:] - target_struct[:, :-1]
    pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
    target_acc = target_vel[:, 1:] - target_vel[:, :-1]
    return F.mse_loss(pred_acc, target_acc)


def total_variation(pred, n_kps):
    if pred.size(1) < 2:
        return pred.sum() * 0.0
    pred_struct = reshape_gap_tensor(pred, n_kps)
    variation = pred_struct[:, 1:] - pred_struct[:, :-1]
    return torch.mean(torch.abs(variation))


def train_bridge():
    device = config['training']['device']
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    num_epochs = config['training']['num_epochs']
    weight_decay = config['training']['weight_decay']
    grad_accum_steps = config['training']['gradient_accumulation_steps']
    experiment_name = config['training']['experiment'] + "_bridge"

    dataset = PoseBridgeDatasetV6()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PoseBridgeTransformerV6().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = None
    warmup_epochs = 0
    if config['lr_scheduler']['type'] == 'cosine_with_warmup':
        warmup_epochs = config['lr_scheduler']['warmup_epochs']
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - warmup_epochs), eta_min=1e-6)

    recon_criterion = nn.HuberLoss()

    loss_cfg = config['loss_weights']
    lambda_recon = loss_cfg.get('lambda_recon', 1.0)
    lambda_vel = loss_cfg.get('lambda_vel', 0.0)
    lambda_accel = loss_cfg.get('lambda_accel', 0.0)
    lambda_tv = loss_cfg.get('lambda_tv', 0.0)

    part_kps = {part: len(indices) for part, indices in get_part_kp_indices(include_face=True).items()}

    print(f"Bridge training on device {device}. Samples: {len(dataset)} | Gap length: {config['masking']['gap_length']}")

    for epoch in range(num_epochs):
        model.train()
        if scheduler and epoch < warmup_epochs:
            warmup_lr = learning_rate * (epoch + 1) / warmup_epochs
            for group in optimizer.param_groups:
                group['lr'] = warmup_lr

        epoch_loss = 0.0
        optimizer.zero_grad()

        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for step, batch in enumerate(progress):
            pre_dict, post_dict, pre_mask_dict, post_mask_dict, gap_target_dict = batch

            pre_dict = {k: v.to(device) for k, v in pre_dict.items()}
            post_dict = {k: v.to(device) for k, v in post_dict.items()}
            pre_mask_dict = {k: v.to(device) for k, v in pre_mask_dict.items()}
            post_mask_dict = {k: v.to(device) for k, v in post_mask_dict.items()}
            gap_target_dict = {k: v.to(device) for k, v in gap_target_dict.items()}

            predictions = model(pre_dict, post_dict, pre_mask_dict, post_mask_dict)

            total_recon = torch.tensor(0.0, device=device)
            total_vel = torch.tensor(0.0, device=device)
            total_accel = torch.tensor(0.0, device=device)
            total_tv = torch.tensor(0.0, device=device)

            num_parts = len(predictions)
            vel_parts = 0
            accel_parts = 0
            tv_parts = 0

            for part, pred_part in predictions.items():
                target_part = gap_target_dict[part]
                B, gap_len, V, C = pred_part.shape
                pred_flat = pred_part.view(B, gap_len, V * C)
                target_flat = target_part.view(B, gap_len, V * C)

                total_recon += recon_criterion(pred_flat, target_flat)

                n_kps = part_kps[part]

                if lambda_vel > 0 and gap_len > 1:
                    total_vel += velocity_loss(pred_flat, target_flat, n_kps)
                    vel_parts += 1

                if lambda_accel > 0 and gap_len > 2:
                    total_accel += acceleration_loss(pred_flat, target_flat, n_kps)
                    accel_parts += 1

                if lambda_tv > 0 and gap_len > 1:
                    total_tv += total_variation(pred_flat, n_kps)
                    tv_parts += 1

            loss_recon = total_recon / num_parts
            loss_vel = total_vel / vel_parts if vel_parts > 0 else torch.tensor(0.0, device=device)
            loss_accel = total_accel / accel_parts if accel_parts > 0 else torch.tensor(0.0, device=device)
            loss_tv = total_tv / tv_parts if tv_parts > 0 else torch.tensor(0.0, device=device)

            loss = (
                lambda_recon * loss_recon
                + lambda_vel * loss_vel
                + lambda_accel * loss_accel
                + lambda_tv * loss_tv
            )

            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            progress.set_postfix({'loss': epoch_loss / (step + 1)})

        if scheduler and epoch >= warmup_epochs:
            scheduler.step()

        checkpoints_dir = os.path.join("checkpoints", experiment_name)
        os.makedirs(checkpoints_dir, exist_ok=True)
        save_path = os.path.join(checkpoints_dir, f"{experiment_name}_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Epoch {epoch+1} finished. Avg loss: {epoch_loss / len(dataloader):.6f}. Saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the bridge transformer on contiguous gaps.")
    parser.add_argument('--experiment', type=str, help='Override experiment name suffix.')
    parser.add_argument('--gap_length', type=int, help='Override gap length for masking.')
    args, _ = parser.parse_known_args()

    if args.experiment:
        config['training']['experiment'] = args.experiment
    if args.gap_length:
        config['masking']['gap_length'] = args.gap_length

    if config['training']['device'] == 'auto':
        config['training']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_bridge()

# python src/train_bridge.py --experiment v6-gap --gap_length 6
