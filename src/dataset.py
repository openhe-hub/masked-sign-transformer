import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
from config_loader import config

class PoseDataset(Dataset):
    def __init__(self):
        self.sequence_length = config['data']['sequence_length']
        self.n_kps = config['data']['n_kps']
        self.features_per_kp = config['data']['features_per_kp']
        self.data_path = config['data']['data_dir']
        
        # 从配置中获取掩码比例
        self.spatial_mask_ratio = config['masking']['spatial_mask_ratio']
        self.temporal_mask_ratio = config['masking']['temporal_mask_ratio']
        
        self.pkl_files = [f for f in os.listdir(self.data_path) if f.endswith('.pkl')]
        self.samples = []
        self._prepare_samples()

    def _prepare_samples(self):
        """
        准备样本，每个样本都是一个完整的序列，而不是 (输入, 目标) 对。
        """
        print("Preparing samples for masked autoencoding...")
        for file_name in self.pkl_files:
            file_path = os.path.join(self.data_path, file_name)
            with open(file_path, 'rb') as f:
                keypoints_data = pickle.load(f)
            
            num_frames = keypoints_data.shape[0]
            
            # 每次取一个完整的 sequence_length 作为样本
            for i in range(num_frames - self.sequence_length + 1):
                sequence = keypoints_data[i : i + self.sequence_length]
                self.samples.append(sequence)
        print(f"Total samples created: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取一个样本，并对其应用时空掩码。
        返回:
            - masked_sequence (Tensor): 应用掩码后的序列。
            - mask (Tensor): 一个布尔张量，标记了哪些部分被掩盖 (True 表示被掩盖)。
            - original_sequence (Tensor): 原始的、未修改的序列。
        """
        original_sequence = self.samples[idx]
        
        # 1. 复制原始序列用于创建掩码版本
        masked_sequence = np.copy(original_sequence)
        
        # 2. 创建掩码张量 (初始化为全False)
        # 形状: (sequence_length, n_kps)
        mask = np.zeros((self.sequence_length, self.n_kps), dtype=bool)
        
        # 3. 时间掩码 (Temporal Masking)
        num_temporal_masks = int(self.sequence_length * self.temporal_mask_ratio)
        temporal_masked_indices = np.random.choice(self.sequence_length, num_temporal_masks, replace=False)
        
        # 将被选择进行时间掩码的帧的所有关键点标记为True
        mask[temporal_masked_indices, :] = True
        
        # 4. 空间掩码 (Spatial Masking)
        # 仅在未被时间掩码的帧上进行
        spatial_candidate_indices = np.where(np.all(mask == False, axis=1))[0]
        
        for frame_idx in spatial_candidate_indices:
            num_spatial_masks = int(self.n_kps * self.spatial_mask_ratio)
            spatial_masked_kps = np.random.choice(self.n_kps, num_spatial_masks, replace=False)
            
            # 标记被空间掩码的关键点
            mask[frame_idx, spatial_masked_kps] = True

        # 5. 应用掩码
        # 我们将掩码位置的坐标设置为0。在模型部分，我们会用一个可学习的 [MASK] token 替换这些0值。
        # mask[:, :, np.newaxis] 扩展维度以匹配 (L, K, 2) 的坐标数据
        masked_sequence[np.where(mask)] = 0
        
        # 6. 转换为Tensor并展平
        # 形状: (sequence_length, n_kps * features_per_kp)
        original_tensor = torch.tensor(original_sequence, dtype=torch.float32).view(self.sequence_length, -1)
        masked_tensor = torch.tensor(masked_sequence, dtype=torch.float32).view(self.sequence_length, -1)
        mask_tensor = torch.tensor(mask, dtype=torch.bool) # mask不需要展平，因为它作用于kps维度

        return masked_tensor, mask_tensor, original_tensor
