import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import json
from tqdm import tqdm

from config_loader import config
# 确保导入正确的模型和数据集
from models.model_v4 import PoseTransformerV4
from datasets.dataset_v3 import PoseDatasetV3
from datasets.dataset_v5 import PoseDatasetV5

def evaluate(checkpoint_path, output_name=None):
    """
    加载 v4 模型的检查点，在 v5 数据集上评估其在被遮挡关键点上的
    均方误差（MSE）和平均绝对误差（MAE），然后将结果保存为 JSON 文件。
    """
    # --- 配置 ---
    device = config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = config['training'].get('batch_size', 16)
    model_version = config['model'].get('version', 'v4')

    # --- 模型和数据加载 ---
    print("Instantiating Model Version: v4 (Part-Aware w/ TemporalConv)")
    model = PoseTransformerV4().to(device)
    
    # 使用与训练时相同的数据集
    # 注意：数据集内部的随机遮挡对于评估是有效的，因为它模拟了真实世界中的缺失数据
    dataset = PoseDatasetV3()
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
    # 加载模型权重
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval() # 切换到评估模式
    print(f"Model loaded successfully from {checkpoint_path}")

    # --- 指标累加器 ---
    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_masked_keypoints = 0

    print("Starting evaluation...")
    with torch.no_grad(): # 在评估期间禁用梯度计算
        for masked_sequence, input_mask, original_sequence in tqdm(eval_loader, desc="Evaluating"):
            
            # 将数据移动到指定设备
            masked_sequence = {k: v.to(device) for k, v in masked_sequence.items()}
            input_mask = {k: v.to(device) for k, v in input_mask.items()}
            original_sequence = {k: v.to(device) for k, v in original_sequence.items()}

            # --- 获取模型预测结果 ---
            predictions = model(masked_sequence)

            # --- 逐个身体部位计算误差 ---
            for part, pred_part in predictions.items():
                target_part = original_sequence[part]
                mask_part = input_mask[part]
                
                # --- 关键步骤: 对齐序列长度（与训练代码一致） ---
                T_pred = pred_part.shape[1]
                stride = target_part.shape[1] // T_pred
                target_part_downsampled = target_part[:, ::stride, :, :][:, :T_pred, :, :]
                mask_part_downsampled = mask_part[:, ::stride, :][:, :T_pred, :]

                # --- 筛选出被遮挡的关键点 ---
                # mask_part_downsampled 的形状是 [B, T, K]，我们需要将其扩展以匹配数据张量
                # 我们只关心被遮挡的部分（mask值为1）
                mask_expanded = mask_part_downsampled.unsqueeze(-1).expand_as(pred_part).bool()
                
                pred_masked = torch.masked_select(pred_part, mask_expanded)
                gt_masked = torch.masked_select(target_part_downsampled, mask_expanded)

                if gt_masked.numel() > 0:
                    # --- 累加误差 ---
                    # 计算平方误差 (for MSE)
                    total_squared_error += nn.functional.mse_loss(pred_masked, gt_masked, reduction='sum').item()
                    # 计算绝对误差 (for MAE/L1)
                    total_absolute_error += nn.functional.l1_loss(pred_masked, gt_masked, reduction='sum').item()
                    # 统计被遮挡的关键点数量（每个关键点有x,y两个坐标值）
                    total_masked_keypoints += gt_masked.numel()

    # --- 计算最终的平均指标 ---
    # 避免除以零
    if total_masked_keypoints == 0:
        print("Warning: No masked keypoints were found in the dataset. Cannot compute metrics.")
        avg_mse = 0
        avg_mae = 0
    else:
        avg_mse = total_squared_error / total_masked_keypoints
        avg_mae = total_absolute_error / total_masked_keypoints
    
    # --- 准备结果字典 ---
    results = {
        "checkpoint_path": checkpoint_path,
        "model_version": model_version,
        "overall_metrics": {
            "masked_mse": avg_mse,
            "masked_mae": avg_mae,
        }
    }

    # --- 将结果保存为 JSON 文件 ---
    output_dir = "output/metrics"
    os.makedirs(output_dir, exist_ok=True)

    if output_name:
        json_filename = f"{output_name}.json"
    else:
        # 默认使用检查点文件名
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        json_filename = f"metrics_{checkpoint_name}.json"
    
    output_path = os.path.join(output_dir, json_filename)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print("\n--- Evaluation Complete ---")
    print(f"Results saved to: {output_path}")
    print(json.dumps(results['overall_metrics'], indent=4))
    print("---------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for PoseTransformer v4")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint (e.g., checkpoints/experiment_name/experiment_name_epoch_10.pth)')
    parser.add_argument('--output_name', type=str, default=None,
                        help='Optional: Custom name for the output JSON file (without extension).')
    args = parser.parse_args()

    evaluate(args.checkpoint, args.output_name)