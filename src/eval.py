import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse
from tqdm import tqdm

from config_loader import config
from model import PoseTransformer
from dataset import PoseDataset

def evaluate(checkpoint_path):
    """
    Loads a model checkpoint, evaluates it on the dataset, and prints key performance metrics.
    """
    # Set a fixed seed for reproducibility of masking
    np.random.seed(42)

    # Get config
    device = config['training']['device']
    batch_size = config['training']['batch_size']
    n_kps = config['data']['n_kps']
    features_per_kp = 2 # config['data']['features_per_kp']

    # Load model
    model = PoseTransformer().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Model loaded from {checkpoint_path}")

    # Prepare dataset and dataloader
    dataset = PoseDataset()
    # Disable shuffling to ensure consistent evaluation
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False) 

    # Loss functions for metrics
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()

    total_mse = 0
    total_mae = 0
    
    print("Starting evaluation...")
    with torch.no_grad():
        for masked_sequence, mask, original_sequence in tqdm(eval_loader, desc="Evaluating"):
            masked_sequence = masked_sequence.to(device)
            mask = mask.to(device)
            original_sequence = original_sequence.to(device)

            # Get model predictions
            predictions = model(masked_sequence, mask)
            
            # Expand mask to match the feature dimension of the predictions
            expanded_mask = mask.unsqueeze(-1).expand(-1, -1, -1, features_per_kp)
            expanded_mask = expanded_mask.reshape(
                predictions.shape[0], predictions.shape[1], n_kps * features_per_kp
            )
            
            # Select only the elements that were masked for evaluation
            pred_masked = torch.masked_select(predictions, expanded_mask)
            gt_masked = torch.masked_select(original_sequence, expanded_mask)
            
            # Calculate metrics for the masked parts
            mse = mse_criterion(pred_masked, gt_masked)
            mae = mae_criterion(pred_masked, gt_masked)
            
            total_mse += mse.item()
            total_mae += mae.item()

    # Calculate average metrics over all batches
    avg_mse = total_mse / len(eval_loader)
    avg_mae = total_mae / len(eval_loader)

    print("\n--- Evaluation Results ---")
    print(f"Average Masked MSE: {avg_mse:.6f}")
    print(f"Average Masked MAE: {avg_mae:.6f}")
    print("--------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for PoseTransformer")
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to the model checkpoint (e.g., checkpoints/pose_transformer_mae_epoch_80.pth)')
    args = parser.parse_args()

    evaluate(args.checkpoint)
