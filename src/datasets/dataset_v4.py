import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
from config_loader import config

# ✨ 1. 定义我们唯一关心的关键点索引
# 共 7 (上半身) + 21 (左手) + 21 (右手) = 49 个点
INDICES_TO_USE = [
    # 上半身 (7个)
    0, 5, 6, 7, 8, 9, 10,
    # 左手 (21个) - 注意：根据您的列表，这里从91开始
    91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 
    # 右手 (21个)
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132
]

class PoseDatasetV4(Dataset):
    """
    一个只加载和处理指定关键点子集的数据集。
    """
    def __init__(self):
        self.sequence_length = config['data']['sequence_length']
        self.features_per_kp = config['data']['features_per_kp']
        self.data_path = config['data']['data_dir']
        self.limit = config['data']['limit']
        
        self.total_mask_ratio = config['masking']['total_mask_ratio']
        self.conf_threshold = config['masking']['confidence_threshold']
        
        # ✨ 2. 使用我们定义的索引列表，并覆盖配置中的 n_kps
        self.indices_to_use = INDICES_TO_USE
        self.n_kps = len(self.indices_to_use) # n_kps 现在是 49，而不是 133
        print(f"PoseDatasetV5 initialized. Processing {self.n_kps} selected keypoints.")
        
        self.pkl_files = [f for f in os.listdir(self.data_path) if f.endswith('.pkl')]
        self.samples = []
        self._prepare_samples()

    def _prepare_samples(self):
        """
        准备样本，只加载和存储指定索引的关键点数据。
        """
        print(f"Preparing samples using {self.n_kps} keypoints...")
        files_to_process = self.pkl_files
        if self.limit > 0 and self.limit <= len(self.pkl_files):
            files_to_process = self.pkl_files[:self.limit]
        
        for file_name in files_to_process:
            file_path = os.path.join(self.data_path, file_name)
            with open(file_path, 'rb') as f:
                data_dict = pickle.load(f)

            # 加载完整数据
            keypoints_full = np.squeeze(data_dict['keypoints'])
            scores_full = np.squeeze(data_dict['scores'])

            # ✨ 3. 立即进行筛选，只保留我们需要的点
            keypoints_subset = keypoints_full[:, self.indices_to_use, :]
            scores_subset = scores_full[:, self.indices_to_use]

            # 合并 [x, y, confidence]
            scores_expanded = np.expand_dims(scores_subset, axis=-1)
            combined_data = np.concatenate((keypoints_subset, scores_expanded), axis=-1)

            num_frames = combined_data.shape[0]
            
            # 创建滑窗序列
            for i in range(num_frames - self.sequence_length + 1):
                sequence_data = combined_data[i : i + self.sequence_length]
                if self.filter_low_hand_conf(sequence_data):
                    self.samples.append(sequence_data)

        print(f"Total samples created for V5: {len(self.samples)}")
    
    def filter_low_hand_conf(self, seq_data) -> bool:
        """
        过滤手部置信度低的序列。
        ✨ 4. 更新手部索引以匹配新的 49 点格式。
        """
        # 在我们的 49 点数组中：
        # - 上半身是索引 0-6 (7个点)
        # - 左手是索引 7-27 (21个点)
        # - 右手是索引 28-48 (21个点)
        left_hand_indices = slice(7, 28)
        right_hand_indices = slice(28, 49)
        
        # 检查第一帧的置信度
        conf1 = np.mean(seq_data[0, left_hand_indices, 2])
        conf2 = np.mean(seq_data[0, right_hand_indices, 2])
        if conf1 <= self.conf_threshold or conf2 <= self.conf_threshold:
            return False
        return True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取一个样本。由于所有数据都已预先筛选过，此处的逻辑无需更改。
        """ 
        original_sequence = self.samples[idx] # Shape: (seq_len, 49, 3)
        
        original_xy = original_sequence[:, :, :2]
        confidence_scores = original_sequence[:, :, 2]
        
        confidence_mask = confidence_scores < self.conf_threshold
        
        # 这里的 self.n_kps 已经是 49，所以计算是正确的
        total_tokens = self.sequence_length * self.n_kps
        num_total_masks = int(total_tokens * self.total_mask_ratio)
        
        final_mask = np.copy(confidence_mask)
        
        num_already_masked = np.sum(final_mask)
        num_additional_masks_needed = max(0, num_total_masks - num_already_masked)
        
        candidate_indices = np.where(final_mask == False)
        candidate_indices_flat = np.ravel_multi_index(candidate_indices, (self.sequence_length, self.n_kps))
        
        if len(candidate_indices_flat) > num_additional_masks_needed:
            additional_masked_indices_flat = np.random.choice(
                candidate_indices_flat, 
                num_additional_masks_needed, 
                replace=False
            )
            additional_masked_indices = np.unravel_index(additional_masked_indices_flat, (self.sequence_length, self.n_kps))
            final_mask[additional_masked_indices] = True

        masked_xy_sequence = np.copy(original_xy)
        masked_xy_sequence[final_mask] = 0
        
        masked_xy_tensor = torch.tensor(masked_xy_sequence, dtype=torch.float32).reshape(self.sequence_length, -1)
        original_xy_tensor = torch.tensor(original_xy, dtype=torch.float32).reshape(self.sequence_length, -1)
        final_mask_tensor = torch.tensor(final_mask, dtype=torch.bool)

        return masked_xy_tensor, final_mask_tensor, original_xy_tensor