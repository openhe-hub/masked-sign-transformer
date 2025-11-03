import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
from config_loader import config
from utils.constants import PART_KP_INDICES

class PoseDatasetV3(Dataset):
    def __init__(self):
        self.sequence_length = config['data']['sequence_length']
        self.n_kps = config['data']['n_kps']
        self.features_per_kp = config['data']['features_per_kp']
        self.data_path = config['data']['data_dir']
        self.limit = config['data']['limit']
        
        self.total_mask_ratio = config['masking']['total_mask_ratio']
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
         
        for file_name in self.pkl_files:
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
                self.samples.append(keypoints_sequence_part)
        print(f"Total samples created for V3: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence_data = self.samples[idx]
        
        masked_sequence_dict = {}
        input_mask_dict = {}
        original_sequence_dict = {}

        for part in self.parts:
            part_data = sequence_data[part] # (T, V, 3)
            n_kps_part = self.part_kps[part]

            normalized_xy = part_data[:, :, :2]
            confidence_scores = part_data[:, :, 2]
            
            # 1. Create mask based on low confidence
            confidence_mask = confidence_scores < self.conf_threshold
            
            # 2. Determine number of additional random masks needed for this part
            total_tokens = self.sequence_length * n_kps_part
            num_total_masks = int(total_tokens * self.total_mask_ratio)
            
            final_mask = np.copy(confidence_mask)
            num_already_masked = np.sum(final_mask)
            num_additional_masks_needed = max(0, num_total_masks - num_already_masked)
            
            # 3. Apply random masks
            candidate_indices = np.where(final_mask == False)
            candidate_indices_flat = np.ravel_multi_index(
                candidate_indices, (self.sequence_length, n_kps_part)
            )
            
            if len(candidate_indices_flat) > num_additional_masks_needed:
                additional_masked_indices_flat = np.random.choice(
                    candidate_indices_flat, 
                    num_additional_masks_needed, 
                    replace=False
                )
                additional_masked_indices = np.unravel_index(
                    additional_masked_indices_flat, (self.sequence_length, n_kps_part)
                )
                final_mask[additional_masked_indices] = True

            # 4. Create masked input sequence
            masked_xy_sequence = np.copy(normalized_xy)
            masked_xy_sequence[final_mask] = 0  # Apply mask
            
            # 5. Convert to tensors and store in dictionaries
            # Reshape to (Seq_Len, Features) for the model
            masked_sequence_dict[part] = torch.tensor(masked_xy_sequence, dtype=torch.float32)
            original_sequence_dict[part] = torch.tensor(normalized_xy, dtype=torch.float32)
            input_mask_dict[part] = torch.tensor(final_mask, dtype=torch.bool)

        return masked_sequence_dict, input_mask_dict, original_sequence_dict