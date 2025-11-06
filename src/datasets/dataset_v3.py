import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
from config_loader import config
from utils.constants import PART_KP_INDICES
from tqdm import tqdm

class PoseDatasetV3(Dataset):
    def __init__(self):
        self.sequence_length = config['data']['sequence_length']
        self.n_kps = config['data']['n_kps']
        self.features_per_kp = config['data']['features_per_kp']
        self.data_path = config['data']['data_dir']
        self.limit = config['data']['limit']
        
        self.total_mask_ratio = config['masking']['total_mask_ratio']
        self.spatial_mask_ratio = config['masking']['spatial_mask_ratio']
        self.temporal_mask_ratio = config['masking']['temporal_mask_ratio']
        self.conf_threshold = config['masking']['confidence_threshold']
        
        self.pkl_files = [f for f in os.listdir(self.data_path) if f.endswith('.pkl')]
        self.samples = []

        self.parts = list(PART_KP_INDICES.keys())
        self.part_kps = {part: len(indices) for part, indices in PART_KP_INDICES.items()}

        self._prepare_samples()

    def _prepare_samples(self):
        print("Preparing samples for V3 masked autoencoding...")
        if self.limit > 0 and self.limit <= len(self.pkl_files):
            self.pkl_files = self.pkl_files[:self.limit]
         
        for file_name in tqdm(self.pkl_files):
            file_path = os.path.join(self.data_path, file_name)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            num_frames = len(data)
            
            for i in range(num_frames - self.sequence_length + 1):
                sequence_dicts = data[i : i + self.sequence_length]
                # (N_frame, N_kps, 3) = (16, 128, 3) = (16, 128, (x, y, conf))
                keypoints_sequence = np.array([item['keypoints'] for item in sequence_dicts])
                keypoints_sequence_part = {
                    'body': keypoints_sequence[:, :18, :],
                    'left': keypoints_sequence[:, 107:128, :],
                    'right': keypoints_sequence[:, 86:107, :],
                }
                meta_info = {
                    'filename': file_name,
                    'norm_params': [item['norm_params'] for item in sequence_dicts],  # (N_frame, )
                    'subset': [item['subset'] for item in sequence_dicts],
                    'face_keypoints': [item['keypoints'][18:86] for item in sequence_dicts],
                }
                self.samples.append((keypoints_sequence_part, meta_info))
        print(f"Total samples created for V3: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence_data, meta_info = self.samples[idx]
        
        masked_sequence_dict = {}
        input_mask_dict = {}
        original_sequence_dict = {}

        for part in self.parts:
            part_data = sequence_data[part] # (T, V, 3)
            n_kps_part = self.part_kps[part]

            normalized_xy = part_data[:, :, :2]
            confidence_scores = part_data[:, :, 2]
            
            # 1. 基于置信度阈值的初始掩码
            confidence_mask = confidence_scores < self.conf_threshold
            final_mask = np.copy(confidence_mask)

            # 2. 时间掩码：抽取指定比例的整帧
            temporal_mask = np.zeros(self.sequence_length, dtype=bool)
            if self.temporal_mask_ratio > 0:
                num_temporal_frames = int(round(self.sequence_length * self.temporal_mask_ratio))
                num_temporal_frames = max(1, num_temporal_frames)
                num_temporal_frames = min(num_temporal_frames, self.sequence_length)

                frame_candidates = np.where(~final_mask.all(axis=1))[0]
                if frame_candidates.size == 0:
                    frame_candidates = np.arange(self.sequence_length)

                if frame_candidates.size < num_temporal_frames:
                    num_temporal_frames = frame_candidates.size

                if num_temporal_frames > 0:
                    chosen_frames = np.random.choice(frame_candidates, size=num_temporal_frames, replace=False)
                    temporal_mask[chosen_frames] = True
                    final_mask[temporal_mask, :] = True

            # 3. 空间掩码：在剩余帧上按节点级别抽取
            if self.spatial_mask_ratio > 0:
                total_tokens = self.sequence_length * n_kps_part
                num_spatial_masks = int(round(total_tokens * self.spatial_mask_ratio))
                num_spatial_masks = max(1, num_spatial_masks)

                available_rows, available_cols = np.where((~temporal_mask[:, None]) & (~final_mask))
                candidate_count = available_rows.size

                if candidate_count < num_spatial_masks:
                    num_spatial_masks = candidate_count

                if num_spatial_masks > 0:
                    candidate_flat = np.ravel_multi_index(
                        (available_rows, available_cols), (self.sequence_length, n_kps_part)
                    )
                    selected_flat = np.random.choice(candidate_flat, size=num_spatial_masks, replace=False)
                    selected_rows, selected_cols = np.unravel_index(
                        selected_flat, (self.sequence_length, n_kps_part)
                    )
                    final_mask[selected_rows, selected_cols] = True

            # 4. Create masked input sequence
            masked_xy_sequence = np.copy(normalized_xy)
            masked_xy_sequence[final_mask] = 0  # Apply mask
            
            # 5. Convert to tensors and store in dictionaries
            # Reshape to (Seq_Len, Features) for the model
            masked_sequence_dict[part] = torch.tensor(masked_xy_sequence, dtype=torch.float32)
            original_sequence_dict[part] = torch.tensor(normalized_xy, dtype=torch.float32)
            input_mask_dict[part] = torch.tensor(final_mask, dtype=torch.bool)

        return masked_sequence_dict, input_mask_dict, original_sequence_dict, meta_info
