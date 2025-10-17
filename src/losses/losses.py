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


def bone_length_consistency_loss(predictions, targets, skeleton_bones, n_kps):
    """计算骨骼长度一致性损失"""
    if not skeleton_bones:
        return torch.tensor(0.0, device=predictions.device)

    pred_struct = _reshape_to_structured(predictions, n_kps)
    target_struct = _reshape_to_structured(targets, n_kps)
    
    loss = 0
    for joint1_idx, joint2_idx in skeleton_bones:
        # 预测的骨骼长度
        pred_bone_vec = pred_struct[:, :, joint1_idx] - pred_struct[:, :, joint2_idx]
        pred_bone_len = torch.norm(pred_bone_vec, dim=-1)
        
        # 真实的骨骼长度
        target_bone_vec = target_struct[:, :, joint1_idx] - target_struct[:, :, joint2_idx]
        target_bone_len = torch.norm(target_bone_vec, dim=-1)
        
        loss += F.mse_loss(pred_bone_len, target_bone_len)
        
    return loss / len(skeleton_bones)


def total_variation_loss(predictions, n_kps):
    """总变差正则化，惩罚预测结果中的抖动"""
    pred_struct = _reshape_to_structured(predictions, n_kps)
    # 计算相邻帧之间的位移（速度）
    variation = pred_struct[:, 1:] - pred_struct[:, :-1]
    # 使用L1范数来惩罚大的跳变
    tv_loss = torch.mean(torch.abs(variation))
    return tv_loss