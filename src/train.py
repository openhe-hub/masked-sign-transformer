import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from config_loader import config
from dataset import PoseDataset
from model import PoseTransformer

def train():
    # 从配置中获取参数
    device = config['training']['device']
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    num_epochs = config['training']['num_epochs']
    n_kps = config['data']['n_kps']
    features_per_kp = config['data']['features_per_kp']
    
    # 准备数据
    dataset = PoseDataset()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    model = PoseTransformer().to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print(f"Start training on device: {device}")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # --- Key Change: Unpack three items from the dataloader ---
        for masked_sequence, mask, original_sequence in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            masked_sequence = masked_sequence.to(device)
            mask = mask.to(device)
            original_sequence = original_sequence.to(device)

            optimizer.zero_grad()
            
            # --- Key Change: Pass masked input and the mask to the model ---
            predictions = model(masked_sequence, mask)
            
            # --- Key Change: Calculate loss only on the masked parts ---
            # Expand mask to match the feature dimension of the predictions
            expanded_mask = mask.unsqueeze(-1).expand(-1, -1, -1, features_per_kp)
            expanded_mask = expanded_mask.reshape(
                predictions.shape[0], predictions.shape[1], n_kps * features_per_kp
            )
            
            # Select only the elements that were masked
            pred_masked = torch.masked_select(predictions, expanded_mask)
            gt_masked = torch.masked_select(original_sequence, expanded_mask)
            
            loss = criterion(pred_masked, gt_masked)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.6f}")
        
        # 保存模型
        if (epoch + 1) % 10 == 0:
            # Use a path relative to the project root
            checkpoints_dir = "checkpoints"
            if not os.path.exists(checkpoints_dir):
                os.makedirs(checkpoints_dir)
            save_path = os.path.join(checkpoints_dir, f"pose_transformer_mae_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()
