import os
import numpy as np
import torch
from torch.utils.data import Dataset

from config_loader import config
from utils.constants import get_part_kp_indices
from utils.pickle_compat import load_pickle

PART_SLICE_FALLBACK = {
    'body': slice(0, 18),
    'right': slice(86, 107),
    'left': slice(107, 128),
    'face': slice(18, 86),
}

class PoseBridgeDatasetV6(Dataset):
    """
    Dataset that prepares contiguous gap-masking samples for bridge prediction.
    Each sample provides the frames before the gap, the frames after the gap,
    and the ground-truth gap segment for every body part.
    """

    def __init__(self, return_meta: bool = False):
        self.sequence_length = config['data']['sequence_length']
        self.data_path = config['data']['data_dir']
        self.limit = config['data']['limit']
        self.return_meta = return_meta

        masking_cfg = config['masking']
        self.gap_length = masking_cfg['gap_length']
        self.pre_context_length = masking_cfg['pre_context_length']
        self.post_context_length = masking_cfg['post_context_length']

        expected_total = self.pre_context_length + self.gap_length + self.post_context_length
        if expected_total != self.sequence_length:
            raise ValueError(
                "sequence_length must equal pre_context_length + gap_length + post_context_length"
            )

        part_indices = get_part_kp_indices(include_face=True)
        self.parts = list(part_indices.keys())
        self.part_kps = {part: len(indices) for part, indices in part_indices.items()}

        self.pkl_files = [f for f in os.listdir(self.data_path) if f.endswith('.pkl')]
        if self.limit > 0 and self.limit < len(self.pkl_files):
            self.pkl_files = self.pkl_files[: self.limit]

        self.samples = []
        self.metadata = []
        self._prepare_samples()

    def _prepare_samples(self):
        """
        Loads normalized per-part data and creates sliding-window samples.
        """
        print("Preparing samples for contiguous gap masking (bridge)...")
        for file_name in self.pkl_files:
            file_path = os.path.join(self.data_path, file_name)
            with open(file_path, 'rb') as f:
                data_dict = load_pickle(f)

            # support both per-part dicts and list of frames
            subset_list = None
            norm_params = None

            if isinstance(data_dict, dict):
                # Accept either explicit part keys or fallback-to-face naming.
                candidate = {}
                valid = True
                for part in self.parts:
                    key = part
                    if key == 'face' and key not in data_dict and 'face_all' in data_dict:
                        key = 'face_all'
                    if key not in data_dict:
                        valid = False
                        break
                    candidate[part] = data_dict[key]
                if valid:
                    num_frames = candidate[self.parts[0]].shape[0]
                    source = candidate
                else:
                    source = None
            elif isinstance(data_dict, list):
                keypoints = np.stack([frame['keypoints'] for frame in data_dict], axis=0)
                source = {}
                for part in self.parts:
                    if part not in PART_SLICE_FALLBACK:
                        continue
                    sl = PART_SLICE_FALLBACK[part]
                    source[part] = keypoints[:, sl, :]
                num_frames = keypoints.shape[0]
                subset_list = [frame.get('subset') for frame in data_dict]
                norm_params = [frame.get('norm_params') for frame in data_dict]
            else:
                print(f"Warning: Unsupported format for {file_name}; skipping.")
                continue

            if source is None:
                print(f"Warning: File {file_name} missing required parts; skipping.")
                continue

            if num_frames < self.sequence_length:
                continue

            for start in range(0, num_frames - self.sequence_length + 1):
                sample = {
                    part: source[part][start : start + self.sequence_length]
                    for part in self.parts
                }
                self.samples.append(sample)
                subset_chunk = None
                norm_chunk = None
                if subset_list is not None:
                    chunk = subset_list[start : start + self.sequence_length]
                    if any(item is not None for item in chunk):
                        subset_chunk = chunk
                if norm_params is not None:
                    chunk = norm_params[start : start + self.sequence_length]
                    if any(item is not None for item in chunk):
                        norm_chunk = chunk
                self.metadata.append(
                    {
                        'subset': subset_chunk,
                        'norm_params': norm_chunk,
                    }
                )

        print(f"Total bridge samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window = self.samples[idx]
        gap_start = self.pre_context_length
        gap_end = gap_start + self.gap_length

        pre_dict = {}
        post_dict = {}
        pre_pad_mask = {}
        post_pad_mask = {}
        gap_target = {}

        for part in self.parts:
            part_data = window[part]  # (T, V, 3)
            n_kps_part = self.part_kps[part]

            pre_segment = part_data[: self.pre_context_length]
            post_segment = part_data[gap_end: gap_end + self.post_context_length]
            gap_segment = part_data[gap_start:gap_end]

            pre_tensor = torch.tensor(pre_segment, dtype=torch.float32)
            post_tensor = torch.tensor(post_segment, dtype=torch.float32)

            pre_mask_tensor = torch.zeros(self.pre_context_length, dtype=torch.bool)
            post_mask_tensor = torch.zeros(self.post_context_length, dtype=torch.bool)

            pre_dict[part] = pre_tensor
            post_dict[part] = post_tensor
            pre_pad_mask[part] = pre_mask_tensor
            post_pad_mask[part] = post_mask_tensor
            gap_target[part] = torch.tensor(gap_segment, dtype=torch.float32)

        if self.return_meta:
            meta_info = self.metadata[idx] if idx < len(self.metadata) else None
            return pre_dict, post_dict, pre_pad_mask, post_pad_mask, gap_target, meta_info

        return pre_dict, post_dict, pre_pad_mask, post_pad_mask, gap_target
