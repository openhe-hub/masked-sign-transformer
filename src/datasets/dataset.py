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
        self.conf_threshold = config['masking']['confidence_threshold']
        
        self.pkl_files = [f for f in os.listdir(self.data_path) if f.endswith('.pkl')]
        self.samples = []
        self._prepare_samples()

    def _prepare_samples(self):
        """
        准备样本，每个样本都是一个完整的序列，而不是 (输入, 目标) 对。
        现在样本包含关键点和'subset'信息。
        """
        print("Preparing samples for masked autoencoding...")
        for file_name in self.pkl_files:
            file_path = os.path.join(self.data_path, file_name)
            with open(file_path, 'rb') as f:
                # data is a list of dicts, each with 'keypoints' and 'subset'
                data = pickle.load(f)
            
            num_frames = len(data)
            
            # 每次取一个完整的 sequence_length 作为样本
            for i in range(num_frames - self.sequence_length + 1):
                sequence_dicts = data[i : i + self.sequence_length]
                
                # 从字典列表中提取关键点和子集信息
                keypoints_sequence = np.array([item['keypoints'] for item in sequence_dicts])
                subset_sequence = np.array([item['subset'] for item in sequence_dicts])

                if self.filter_low_hand_conf(keypoints_sequence):
                    # 将关键点和子集信息作为一个元组存储
                    self.samples.append((keypoints_sequence, subset_sequence))
        print(f"Total samples created: {len(self.samples)}")
    
    def filter_low_hand_conf(self, seq_data) -> bool:
        # seq_data (N_seq, N_kps, 3)
        conf1 = np.mean(seq_data[0, 86:107, 2])
        conf2 = np.mean(seq_data[0, 107:128, 2])
        if conf1 <= self.conf_threshold or conf2 <= self.conf_threshold:
            return False
        return True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取一个样本，并应用基于置信度和随机策略的混合时空掩码。
        返回:
            - masked_xy_tensor (Tensor): 应用掩码后的 (x,y) 序列，作为模型输入。
            - final_mask_tensor (Tensor): 一个布尔张量，标记了哪些部分被掩盖 (True=掩盖)。
            - original_xy_tensor (Tensor): 原始的、未修改的 (x,y) 序列，作为重建目标。
        """ 
        # self.samples 现在包含 (keypoints_sequence, subset_sequence) 的元组
        original_sequence, subset_info = self.samples[idx]
        
        # 1. 分离坐标 (x,y) 和置信度 (c)
        original_xy = original_sequence[:, :, :2]
        confidence_scores = original_sequence[:, :, 2]
        
        # 2. 创建“天然”掩码：所有低置信度的点都必须被修复
        # confidence_mask 的形状是 (seq_len, n_kps)，True代表低置信度
        confidence_mask = confidence_scores < self.conf_threshold
        
        # 3. 初始化最终掩码，它首先包含所有低置信度的点
        final_mask = np.copy(confidence_mask)
        
        # 4. 在“天然”掩码的基础上，添加随机的时空掩码
        
        # 4.1 时间掩码 (Temporal Masking)
        num_temporal_masks = int(self.sequence_length * self.temporal_mask_ratio)
        # 候选帧：那些还没有任何一个点被置信度mask的帧，我们只在这些“好”帧里随机挑选
        candidate_frames = np.where(np.all(final_mask == False, axis=1))[0]
        
        if len(candidate_frames) > num_temporal_masks:
            temporal_masked_indices = np.random.choice(candidate_frames, num_temporal_masks, replace=False)
            final_mask[temporal_masked_indices, :] = True

        # 4.2 空间掩码 (Spatial Masking)
        # 候选帧：那些还没有被时间掩码完全覆盖的帧
        spatial_candidate_frames = np.where(np.any(final_mask == False, axis=1))[0]
        
        for frame_idx in spatial_candidate_frames:
            # 候选关键点：在这一帧中，还没有被任何方式mask的点
            candidate_kps = np.where(final_mask[frame_idx, :] == False)[0]
            num_spatial_masks = int(len(candidate_kps) * self.spatial_mask_ratio)
            
            if num_spatial_masks > 0:
                spatial_masked_kps = np.random.choice(candidate_kps, num_spatial_masks, replace=False)
                final_mask[frame_idx, spatial_masked_kps] = True

        # 5. 准备给模型的输入：一个被掩码的 (x,y) 序列
        masked_xy_sequence = np.copy(original_xy)
        # 扩展 final_mask 以匹配 (L, K, 2) 的坐标数据，并将对应位置设为0
        # 模型内部会用可学习的 [MASK] token 替换这些0
        expanded_mask_for_input = np.expand_dims(final_mask, axis=-1)
        masked_xy_sequence[np.where(expanded_mask_for_input)] = 0
        
        # 6. 转换为Tensor并展平
        
        # 模型的输入 (masked x,y)，形状: (seq_len, n_kps * 2)
        masked_xy_tensor = torch.tensor(masked_xy_sequence, dtype=torch.float32).view(self.sequence_length, -1)
        
        # 损失计算的目标 (original x,y)，形状: (seq_len, n_kps * 2)
        original_xy_tensor = torch.tensor(original_xy, dtype=torch.float32).view(self.sequence_length, -1)

        # 最终的掩码，形状: (seq_len, n_kps)，模型和训练循环都需要它
        final_mask_tensor = torch.tensor(final_mask, dtype=torch.bool)

        return masked_xy_tensor, final_mask_tensor, original_xy_tensor, subset_info