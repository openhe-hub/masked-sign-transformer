import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import json
from tqdm import tqdm

from config_loader import config
from models.model import PoseTransformer as PoseTransformerV1
from models.model_v2 import PoseTransformerV2
from models.model_v3 import PoseTransformerV3
from datasets.dataset import PoseDataset as PoseDatasetV1
from datasets.dataset_v3 import PoseDatasetV3
from losses.losses import body_bone_length_loss, velocity_consistency_loss

def evaluate(checkpoint_path, output_name=None):
    """
    Loads a model checkpoint, evaluates it on the dataset using a comprehensive
    set of metrics, and saves the results to a JSON file.
    """
    # Set a fixed seed for reproducibility of masking
    np.random.seed(42)

    # --- Configuration ---
    device = config['training']['device']
    batch_size = config['training']['batch_size']
    n_kps = config['data']['n_kps']
    features_per_kp = 2  # We evaluate on (x, y) coordinates
    model_version = config['model'].get('version', 'v1')

    # --- Model and Data Loading ---
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
    
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Model loaded from {checkpoint_path}")

    # --- Metrics Accumulators ---
    total_mse = 0
    total_mae = 0
    total_bone_error = 0
    total_velocity_error = 0
    
    # For per-joint MSE
    per_joint_error_sum = torch.zeros(n_kps, device=device)
    per_joint_mask_count = torch.zeros(n_kps, device=device)

    print("Starting evaluation...")
    with torch.no_grad():
        for masked_sequence, mask, original_sequence, subset, _ in tqdm(eval_loader, desc="Evaluating"):
            masked_sequence = masked_sequence.to(device)
            mask = mask.to(device)
            original_sequence = original_sequence.to(device)
            subset = subset.to(device)

            # --- Get Model Predictions ---
            predictions = model(masked_sequence, mask)

            # --- Prepare Tensors for Metric Calculation ---
            # Expand mask to match the feature dimension for selecting masked elements
            expanded_mask = mask.unsqueeze(-1).expand(-1, -1, -1, features_per_kp)
            expanded_mask = expanded_mask.reshape(
                predictions.shape[0], predictions.shape[1], n_kps * features_per_kp
            )
            
            # Select only the elements that were masked for basic reconstruction metrics
            pred_masked = torch.masked_select(predictions, expanded_mask)
            gt_masked = torch.masked_select(original_sequence, expanded_mask)

            # --- Calculate Metrics for the Batch ---
            # 1. Basic Reconstruction Metrics (on masked parts)
            total_mse += nn.functional.mse_loss(pred_masked, gt_masked).item()
            total_mae += nn.functional.l1_loss(pred_masked, gt_masked).item()

            # 2. Per-Joint MSE (on masked parts)
            pred_struct = predictions.reshape(-1, n_kps, features_per_kp)
            gt_struct = original_sequence.reshape(-1, n_kps, features_per_kp)
            mask_struct = mask.reshape(-1, n_kps)
            
            squared_error = (pred_struct - gt_struct)**2
            # Sum error over features (x, y)
            joint_sq_error = torch.sum(squared_error, dim=-1) # Shape: (B*T, K)
            
            # Accumulate error only for masked joints
            per_joint_error_sum += torch.sum(joint_sq_error * mask_struct, dim=0)
            per_joint_mask_count += torch.sum(mask_struct, dim=0)

            # 3. Body Bone Length Error (on the full pose)
            total_bone_error += body_bone_length_loss(predictions, original_sequence, subset, n_kps).item()
            
            # 4. Velocity Error (on the full sequence)
            total_velocity_error += velocity_consistency_loss(predictions, original_sequence, n_kps).item()

    # --- Finalize and Aggregate Metrics ---
    num_batches = len(eval_loader)
    avg_mse = total_mse / num_batches
    avg_mae = total_mae / num_batches
    avg_bone_error = total_bone_error / num_batches
    avg_velocity_error = total_velocity_error / num_batches
    
    # Avoid division by zero if a joint was never masked
    per_joint_mask_count[per_joint_mask_count == 0] = 1e-6
    avg_per_joint_mse = per_joint_error_sum / per_joint_mask_count
    
    # --- Prepare Results Dictionary ---
    results = {
        "checkpoint_path": checkpoint_path,
        "overall_metrics": {
            "masked_mse": avg_mse,
            "masked_l1": avg_mae,
            "body_bone_length_mae": avg_bone_error,
            "velocity_mse": avg_velocity_error
        },
        "per_joint_mse": avg_per_joint_mse.cpu().tolist()
    }

    # --- Save Results to JSON ---
    output_dir = "output/metrics"
    os.makedirs(output_dir, exist_ok=True)

    if output_name:
        # Use the custom name provided by the user
        json_filename = f"{output_name}.json"
    else:
        # Default behavior: create a filename from the checkpoint name
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        json_filename = f"{checkpoint_name}_metrics.json"
    
    output_path = os.path.join(output_dir, json_filename)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print("\n--- Evaluation Complete ---")
    print(f"Results saved to: {output_path}")
    print(json.dumps(results['overall_metrics'], indent=4))
    print("---------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for PoseTransformer")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint (e.g., checkpoints/v2/experiment_epoch_100.pth)')
    parser.add_argument('--output_name', type=str, default=None,
                        help='Optional: Custom name for the output JSON file (without extension).')
    args = parser.parse_args()

    evaluate(args.checkpoint, args.output_name)