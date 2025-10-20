import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
from config_loader import config

class PoseDatasetV3(Dataset):
    def __init__(self):
        self.sequence_length = config['data']['sequence_length']
        self.n_kps = config['data']['n_kps']
        self.features_per_kp = config['data']['features_per_kp']
        self.data_path = config['data']['data_dir']
        
        # 从配置中获取V3模型的掩码比例
        self.total_mask_ratio = config['masking']['total_mask_ratio']
        self.conf_threshold = config['masking']['confidence_threshold']
        
        self.pkl_files = [f for f in os.listdir(self.data_path) if f.endswith('.pkl')]
        self.samples = []
        self._prepare_samples()

    def _prepare_samples(self):
        """
        准备样本，每个样本都是一个完整的序列。
        """
        print("Preparing samples for V3 masked autoencoding...")
        for file_name in self.pkl_files:
            file_path = os.path.join(self.data_path, file_name)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            num_frames = len(data)
            
            for i in range(num_frames - self.sequence_length + 1):
                sequence_dicts = data[i : i + self.sequence_length]
                
                keypoints_sequence = np.array([item['keypoints'] for item in sequence_dicts])
                subset_sequence = np.array([item['subset'] for item in sequence_dicts])

                if self.filter_low_hand_conf(keypoints_sequence):
                    self.samples.append((keypoints_sequence, subset_sequence))
        print(f"Total samples created for V3: {len(self.samples)}")
    
    def filter_low_hand_conf(self, seq_data) -> bool:
        conf1 = np.mean(seq_data[0, 86:107, 2])
        conf2 = np.mean(seq_data[0, 107:128, 2])
        if conf1 <= self.conf_threshold or conf2 <= self.conf_threshold:
            return False
        return True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取一个样本，并应用固定比例的随机掩码策略。
        返回:
            - masked_xy_tensor (Tensor): 应用掩码后的 (x,y) 序列，作为模型输入。
            - final_mask_tensor (Tensor): 一个布尔张量，标记了哪些部分被掩盖 (True=掩盖)。
            - original_xy_tensor (Tensor): 原始的、未修改的 (x,y) 序列，作为重建目标。
        """ 
        original_sequence, subset_info = self.samples[idx]
        
        original_xy = original_sequence[:, :, :2]
        confidence_scores = original_sequence[:, :, 2]
        
        # 1. 创建“天然”掩码：所有低置信度的点都必须被修复
        confidence_mask = confidence_scores < self.conf_threshold
        
        # 2. 计算需要掩码的总Token数
        total_tokens = self.sequence_length * self.n_kps
        num_total_masks = int(total_tokens * self.total_mask_ratio)
        
        # 3. 创建最终掩码
        # 首先，所有低置信度的点都必须被mask
        final_mask = np.copy(confidence_mask)
        
        # 4. 计算还需要额外mask多少个点
        num_already_masked = np.sum(final_mask)
        num_additional_masks_needed = max(0, num_total_masks - num_already_masked)
        
        # 5. 随机选择额外的点进行mask
        # 找到所有可以被mask的候选位置 (即非低置信度的点)
        candidate_indices = np.where(final_mask == False)
        
        # 将 (行, 列) 索引转换为线性索引，以便于随机选择
        candidate_indices_flat = np.ravel_multi_index(candidate_indices, (self.sequence_length, self.n_kps))
        
        if len(candidate_indices_flat) > num_additional_masks_needed:
            # 随机选择需要补充的mask位置
            additional_masked_indices_flat = np.random.choice(
                candidate_indices_flat, 
                num_additional_masks_needed, 
                replace=False
            )
            
            # 将线性索引转换回 (行, 列) 索引
            additional_masked_indices = np.unravel_index(additional_masked_indices_flat, (self.sequence_length, self.n_kps))
            
            # 更新最终的mask
            final_mask[additional_masked_indices] = True

        # 6. 准备模型输入
        masked_xy_sequence = np.copy(original_xy)
        expanded_mask_for_input = np.expand_dims(final_mask, axis=-1)
        masked_xy_sequence[np.where(expanded_mask_for_input)] = 0
        
        # 7. 转换为Tensor
        masked_xy_tensor = torch.tensor(masked_xy_sequence, dtype=torch.float32).view(self.sequence_length, -1)
        original_xy_tensor = torch.tensor(original_xy, dtype=torch.float32).view(self.sequence_length, -1)
        final_mask_tensor = torch.tensor(final_mask, dtype=torch.bool)

        return masked_xy_tensor, final_mask_tensor, original_xy_tensor, subset_info
