import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 辅助函数 ---
def _reshape_to_structured(tensor, n_kps, features_per_kp=2):
    """将扁平化的 (B, T, K*F) 张量重塑为结构化的 (B, T, K, F)"""
    batch_size, seq_len, _ = tensor.shape
    return tensor.view(batch_size, seq_len, n_kps, features_per_kp)

# --- 损失函数实现 ---

def reconstruction_loss(predictions, targets, mask, criterion=nn.HuberLoss()):
    """计算重建损失，只在被掩码的位置计算。"""
    # mask 形状: (B, T, K) -> expanded_mask 形状: (B, T, K*F)
    batch_size, seq_len, n_kps = mask.shape
    features_per_kp = predictions.shape[-1] // n_kps
    
    expanded_mask = mask.unsqueeze(-1).expand(-1, -1, -1, features_per_kp)
    expanded_mask = expanded_mask.reshape(batch_size, seq_len, n_kps * features_per_kp)
    
    pred_masked = torch.masked_select(predictions, expanded_mask)
    gt_masked = torch.masked_select(targets, expanded_mask)
    
    if pred_masked.numel() == 0:
        return torch.tensor(0.0, device=predictions.device)
        
    return criterion(pred_masked, gt_masked)

def full_sequence_reconstruction_loss(predictions, targets, criterion=nn.HuberLoss()):
    """
    【按需】计算整个序列的重建损失，不考虑掩码。
    这个函数不接收 mask 参数。
    """
    return criterion(predictions, targets)


def velocity_consistency_loss(predictions, targets, n_kps):
    """计算速度一致性损失 (相邻帧的位移)"""
    pred_struct = _reshape_to_structured(predictions, n_kps)
    target_struct = _reshape_to_structured(targets, n_kps)
    
    pred_vel = pred_struct[:, 1:] - pred_struct[:, :-1]
    target_vel = target_struct[:, 1:] - target_struct[:, :-1]
    
    return F.mse_loss(pred_vel, target_vel)


def acceleration_consistency_loss(predictions, targets, n_kps):
    """计算加速度一致性损失 (速度的变化率)"""
    pred_struct = _reshape_to_structured(predictions, n_kps)
    target_struct = _reshape_to_structured(targets, n_kps)

    pred_vel = pred_struct[:, 1:] - pred_struct[:, :-1]
    target_vel = target_struct[:, 1:] - target_struct[:, :-1]
    
    pred_accel = pred_vel[:, 1:] - pred_vel[:, :-1]
    target_accel = target_vel[:, 1:] - target_vel[:, :-1]

    return F.mse_loss(pred_accel, target_accel)


def body_bone_length_loss(predictions, targets, subset, n_kps):
    """
    计算身体骨骼长度的重建损失 (MSE)。
    这个损失函数利用 'subset' 信息来处理缺失的关键点，
    只在预测和目标中都存在的骨骼上计算损失。
    """
    # limbSeq from render.py, converted to 0-based indices for the 18 body keypoints
    body_limb_seq = [
        [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9],
        [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16],
        [0, 15], [15, 17], [2, 16], [5, 17]
    ]

    # Reshape inputs to be structured: (B, T, K, F)
    pred_struct = _reshape_to_structured(predictions, n_kps)
    target_struct = _reshape_to_structured(targets, n_kps)
    
    B, T, K, n_features = pred_struct.shape
    
    # Ensure subset is long type for indexing
    subset_long = subset.long()

    # For simplicity, we assume only one person per frame.
    if subset_long.shape[2] > 1:
        subset_long = subset_long[:, :, 0, :]
    else:
        subset_long = subset_long.squeeze(2) # -> (B, T, 20)

    all_pred_lens = []
    all_target_lens = []

    for p1_part_idx, p2_part_idx in body_limb_seq:
        # Get the actual keypoint indices from subset
        kp_indices_p1 = subset_long[..., p1_part_idx]  # Shape: (B, T)
        kp_indices_p2 = subset_long[..., p2_part_idx]  # Shape: (B, T)

        # Create a mask for valid bones where both keypoints are present (index != -1)
        valid_mask = (kp_indices_p1 != -1) & (kp_indices_p2 != -1) # Shape: (B, T)
        
        if not torch.any(valid_mask):
            continue

        valid_mask_flat = valid_mask.view(-1)
        
        kp_indices_p1_safe = torch.clamp(kp_indices_p1, min=0).view(-1)
        kp_indices_p2_safe = torch.clamp(kp_indices_p2, min=0).view(-1)

        # --- Gather coordinates for predicted poses ---
        pred_flat = pred_struct.reshape(B * T, K, n_features)
        idx1_gather = kp_indices_p1_safe.unsqueeze(1).unsqueeze(2).expand(-1, 1, n_features)
        coords1_pred = torch.gather(pred_flat, 1, idx1_gather).squeeze(1)
        idx2_gather = kp_indices_p2_safe.unsqueeze(1).unsqueeze(2).expand(-1, 1, n_features)
        coords2_pred = torch.gather(pred_flat, 1, idx2_gather).squeeze(1)

        # --- Gather coordinates for target poses ---
        target_flat = target_struct.reshape(B * T, K, n_features)
        coords1_target = torch.gather(target_flat, 1, idx1_gather).squeeze(1)
        coords2_target = torch.gather(target_flat, 1, idx2_gather).squeeze(1)

        # --- Calculate bone lengths ---
        pred_bone_len = torch.norm(coords1_pred - coords2_pred, dim=-1)
        target_bone_len = torch.norm(coords1_target - coords2_target, dim=-1)

        all_pred_lens.append(pred_bone_len[valid_mask_flat])
        all_target_lens.append(target_bone_len[valid_mask_flat])

    if not all_pred_lens:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)

    pred_lens_tensor = torch.cat(all_pred_lens)
    target_lens_tensor = torch.cat(all_target_lens)

    return F.mse_loss(pred_lens_tensor, target_lens_tensor)



def total_variation_loss(predictions, n_kps):
    """总变差正则化，惩罚预测结果中的抖动"""
    pred_struct = _reshape_to_structured(predictions, n_kps)
    # 计算相邻帧之间的位移（速度）
    variation = pred_struct[:, 1:] - pred_struct[:, :-1]
    # 使用L1范数来惩罚大的跳变
    tv_loss = torch.mean(torch.abs(variation))
    return tv_loss