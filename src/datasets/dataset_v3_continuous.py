import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from config_loader import config
from utils.constants import PART_KP_INDICES


class PoseDatasetV3Continuous(Dataset):
    """
    Variant of PoseDatasetV3 that samples non-overlapping windows.

    Each video is segmented into clips of length `sequence_length` with stride equal to
    `window_stride`. By default, `window_stride == sequence_length`, ensuring clips do not
    overlap. Remaining tail frames are covered by including the last possible window.
    """

    def __init__(self, window_stride: Optional[int] = None):
        self.sequence_length = config["data"]["sequence_length"]
        self.n_kps = config["data"]["n_kps"]
        self.features_per_kp = config["data"]["features_per_kp"]
        self.data_path = config["data"]["data_dir"]
        self.limit = config["data"]["limit"]

        self.conf_threshold = config["masking"]["confidence_threshold"]

        self.window_stride = max(1, window_stride or self.sequence_length)

        self.pkl_files = [f for f in os.listdir(self.data_path) if f.endswith(".pkl")]
        if 0 < self.limit <= len(self.pkl_files):
            self.pkl_files = self.pkl_files[: self.limit]

        self.parts: List[str] = list(PART_KP_INDICES.keys())
        self.part_kps: Dict[str, int] = {part: len(indices) for part, indices in PART_KP_INDICES.items()}

        self.samples: List = []
        self._prepare_samples()

    def _prepare_samples(self) -> None:
        print("Preparing samples for V3 continuous dataset (non-overlapping windows)...")
        for file_name in tqdm(self.pkl_files):
            file_path = os.path.join(self.data_path, file_name)
            with open(file_path, "rb") as f:
                data = pickle.load(f)

            num_frames = len(data)
            if num_frames < self.sequence_length:
                continue

            start_indices = list(range(0, num_frames - self.sequence_length + 1, self.window_stride))
            # last_possible_start = num_frames - self.sequence_length
            # if not start_indices or start_indices[-1] != last_possible_start:
            #     start_indices.append(last_possible_start)
            # print(f"{file_name}, {start_indices}")

            for start in start_indices:
                sequence_dicts = data[start : start + self.sequence_length]
                keypoints_sequence = np.array([item["keypoints"] for item in sequence_dicts])

                keypoints_sequence_part = {
                    "body": keypoints_sequence[:, :18, :],
                    "left": keypoints_sequence[:, 107:128, :],
                    "right": keypoints_sequence[:, 86:107, :],
                }
                confidence_sequence = keypoints_sequence[:, :, 2].astype(np.float32, copy=True)

                meta_info = {
                    "filename": file_name,
                    "norm_params": [item["norm_params"] for item in sequence_dicts],
                    "subset": [item["subset"] for item in sequence_dicts],
                    "face_keypoints": [item["keypoints"][18:86] for item in sequence_dicts],
                    "confidence": confidence_sequence,
                }

                self.samples.append((keypoints_sequence_part, meta_info))

        print(f"Total samples created for V3 continuous: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        sequence_data, meta_info = self.samples[idx]

        masked_sequence_dict: Dict[str, torch.Tensor] = {}
        input_mask_dict: Dict[str, torch.Tensor] = {}
        original_sequence_dict: Dict[str, torch.Tensor] = {}

        for part in self.parts:
            part_data = sequence_data[part]  # (T, V, 3)
            n_kps_part = self.part_kps[part]

            normalized_xy = part_data[:, :, :2]
            confidence_scores = part_data[:, :, 2]

            final_mask = confidence_scores < self.conf_threshold

            masked_xy_sequence = np.copy(normalized_xy)
            masked_xy_sequence[final_mask] = 0

            masked_sequence_dict[part] = torch.tensor(masked_xy_sequence, dtype=torch.float32)
            original_sequence_dict[part] = torch.tensor(normalized_xy, dtype=torch.float32)
            input_mask_dict[part] = torch.tensor(final_mask, dtype=torch.bool)

        return masked_sequence_dict, input_mask_dict, original_sequence_dict, meta_info
