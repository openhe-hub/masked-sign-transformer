import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import cv2
from config_loader import config

class PoseDataset(Dataset):
    def __init__(self):
        # --- 配置参数 ---
        self.sequence_length = config['data']['sequence_length']
        self.n_kps = config['data']['n_kps']
        self.features_per_kp = config['data']['features_per_kp']
        self.data_path = config['data']['data_dir']
        self.video_path = config['data']['video_dir']
        
        self.spatial_mask_ratio = config['masking']['spatial_mask_ratio']
        self.temporal_mask_ratio = config['masking']['temporal_mask_ratio']
        self.conf_threshold = config['masking']['confidence_threshold']
        
        # --- 数据准备 ---
        self.pkl_files = [f for f in os.listdir(self.data_path) if f.endswith('.pkl')]
        self.samples = []
        # 初始的 sample_stride，与get_video_pose中的参数对应
        self.base_sample_stride = 2 
        self._prepare_samples()

    def _prepare_samples(self):
        """
        准备样本，并为每个视频动态计算其采样步长，确保数据匹配。
        """
        print("Preparing samples with dynamic stride calculation...")
        for file_name in self.pkl_files:
            video_name = file_name.replace('_kps.pkl', '.mp4')
            video_file_path = os.path.join(self.video_path, video_name)

            if not os.path.exists(video_file_path):
                print(f"Warning: Video file not found for {file_name}, skipping.")
                continue

            # 加载 pkl 数据以获取其长度
            file_path = os.path.join(self.data_path, file_name)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            pkl_frame_count = len(data)

            # 打开视频，获取其帧数和FPS
            cap = cv2.VideoCapture(video_file_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video file {video_file_path}, skipping.")
                continue
            video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            # 【核心逻辑】复制 get_video_pose 中的步长计算方法
            if video_fps <= 0: # 防止除以零或负数
                print(f"Warning: Invalid FPS ({video_fps}) for video {video_name}, skipping.")
                continue
            
            # 计算这个特定视频的最终步长
            final_stride = self.base_sample_stride * max(1, int(video_fps / 24))

            # 使用动态计算的步长进行验证
            # 允许的误差范围也与步长相关，使其更鲁棒
            if abs(video_frame_count - pkl_frame_count * final_stride) > final_stride:
                print(f"Warning: Frame count mismatch for '{file_name}'. "
                      f"Pkl: {pkl_frame_count}, Video: {video_frame_count}, Calculated Stride: {final_stride}. "
                      f"This pair will be skipped.")
                continue

            # 验证通过，创建样本
            num_frames = pkl_frame_count
            for i in range(num_frames - self.sequence_length + 1):
                sequence_dicts = data[i : i + self.sequence_length]
                keypoints_sequence = np.array([item['keypoints'] for item in sequence_dicts])
                subset_sequence = np.array([item['subset'] for item in sequence_dicts])

                if self.filter_low_hand_conf(keypoints_sequence):
                    # 【重要】将计算出的 final_stride 存入每个样本中
                    self.samples.append((keypoints_sequence, subset_sequence, video_file_path, i, final_stride))
                    
        print(f"Total samples created: {len(self.samples)}")
    
    def filter_low_hand_conf(self, seq_data) -> bool:
        # ... 无改动 ...
        conf1 = np.mean(seq_data[0, 86:107, 2])
        conf2 = np.mean(seq_data[0, 107:128, 2])
        if conf1 <= self.conf_threshold or conf2 <= self.conf_threshold:
            return False
        return True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取样本，使用样本中存储的特定步长来读取视频帧。
        """ 
        # 从样本中解包出为该视频计算好的 final_stride
        original_sequence, subset_info, video_path, pkl_start_idx, final_stride = self.samples[idx]
        
        # --- 关键点和掩码处理 (无改动) ---
        original_xy = original_sequence[:, :, :2]
        confidence_scores = original_sequence[:, :, 2]
        confidence_mask = confidence_scores < self.conf_threshold
        final_mask = np.copy(confidence_mask)
        num_temporal_masks = int(self.sequence_length * self.temporal_mask_ratio)
        candidate_frames = np.where(np.all(final_mask == False, axis=1))[0]
        if len(candidate_frames) > num_temporal_masks:
            temporal_masked_indices = np.random.choice(candidate_frames, num_temporal_masks, replace=False)
            final_mask[temporal_masked_indices, :] = True
        spatial_candidate_frames = np.where(np.any(final_mask == False, axis=1))[0]
        for frame_idx in spatial_candidate_frames:
            candidate_kps = np.where(final_mask[frame_idx, :] == False)[0]
            num_spatial_masks = int(len(candidate_kps) * self.spatial_mask_ratio)
            if num_spatial_masks > 0:
                spatial_masked_kps = np.random.choice(candidate_kps, num_spatial_masks, replace=False)
                final_mask[frame_idx, spatial_masked_kps] = True
        masked_xy_sequence = np.copy(original_xy)
        expanded_mask_for_input = np.expand_dims(final_mask, axis=-1)
        masked_xy_sequence[np.where(expanded_mask_for_input)] = 0
        masked_xy_tensor = torch.tensor(masked_xy_sequence, dtype=torch.float32).view(self.sequence_length, -1)
        original_xy_tensor = torch.tensor(original_xy, dtype=torch.float32).view(self.sequence_length, -1)
        final_mask_tensor = torch.tensor(final_mask, dtype=torch.bool)

        return masked_xy_tensor, final_mask_tensor, original_xy_tensor, subset_info, (video_path, pkl_start_idx, final_stride)