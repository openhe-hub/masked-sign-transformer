#!/usr/bin/env python3
"""
Part-wise normalization script that SAVES the normalization parameters for
each frame, enabling a perfect denormalization later.

Input Format: List of T dictionaries [{'keypoints': (1, 128, 3), 'subset': ...}]
Output Format: List of T dictionaries [{'keypoints': (128, 3), 'subset': ..., 'norm_params': {...}}]
"""

import pickle
import numpy as np
import argparse
from tqdm import tqdm

def get_bbox_scale_offset(motion, thr):
    # (This function remains unchanged)
    valid_coords = motion[motion[..., 2] > thr][:, :2]
    if len(valid_coords) < 2:
        return 0, np.array([0.0, 0.0])
    xmin, ymin = valid_coords.min(axis=0)
    xmax, ymax = valid_coords.max(axis=0)
    scale = max(xmax - xmin, ymax - ymin)
    if scale == 0: return 0, np.array([0.0, 0.0])
    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    offset = np.array([xs, ys])
    return scale, offset

def normalize_pkl_and_save_params(input_path, output_path):
    print(f"Loading data from {input_path}...")
    with open(input_path, 'rb') as f:
        data_list = pickle.load(f)

    if not isinstance(data_list, list):
        raise TypeError("Input .pkl file must contain a list of dictionaries.")
    
    T = len(data_list)
    print(f"Found {T} frames to process.")

    BODY_INDICES = slice(0, 18)
    FACE_INDICES = slice(18, 86)
    LEFT_HAND_INDICES = slice(86, 107)
    RIGHT_HAND_INDICES = slice(107, 128)
    LEFT_WRIST_IDX_LOCAL, RIGHT_WRIST_IDX_LOCAL = 0, 0
    CONF_THRESHOLD = 0.3

    output_data_list = []
    print("Normalizing keypoints and saving transformation parameters...")

    for i in tqdm(range(T), desc="Processing Frames"):
        frame_data = data_list[i]
        keypoints = np.array(frame_data['keypoints'])
        
        # --- Store parameters in this dictionary ---
        norm_params = {}

        normalized_frame = np.zeros_like(keypoints)
        body_kps = keypoints[BODY_INDICES]
        body_scale, body_offset = get_bbox_scale_offset(body_kps, thr=CONF_THRESHOLD)

        # --- SAVE a. Body scale and offset ---
        norm_params['body_scale'] = body_scale
        norm_params['body_offset'] = body_offset

        if body_scale == 0:
            # Still save a zeroed-out frame dict for consistency
            output_data_list.append({
                'keypoints': normalized_frame, 
                'subset': frame_data['subset'],
                'norm_params': norm_params
            })
            continue

        # Normalize body
        norm_body_coords = ((body_kps[:, :2] - body_offset) / body_scale - 0.5) * 2
        normalized_frame[BODY_INDICES, :2] = np.clip(norm_body_coords, -1, 1)
        normalized_frame[BODY_INDICES, 2] = body_kps[:, 2]

        parts_to_process = {
            'face': (FACE_INDICES, None),
            'left_hand': (LEFT_HAND_INDICES, LEFT_WRIST_IDX_LOCAL),
            'right_hand': (RIGHT_HAND_INDICES, RIGHT_WRIST_IDX_LOCAL),
        }

        for part_name, (indices, center_idx_local) in parts_to_process.items():
            part_kps = keypoints[indices].copy()
            if center_idx_local is not None:
                center_point = part_kps[center_idx_local, :2].copy()
            else:
                valid_kps = part_kps[part_kps[:, 2] > CONF_THRESHOLD, :2]
                center_point = valid_kps.mean(axis=0) if len(valid_kps) > 0 else np.zeros(2)
            
            # --- SAVE b. Local center point for each part ---
            norm_params[f'{part_name}_center'] = center_point

            norm_part_coords = (part_kps[:, :2] - center_point) / body_scale
            normalized_frame[indices, :2] = np.clip(norm_part_coords, -1, 1)
            normalized_frame[indices, 2] = part_kps[:, 2]

        low_confidence_mask = normalized_frame[:, 2] <= CONF_THRESHOLD
        normalized_frame[low_confidence_mask] = 0
        
        output_data_list.append({
            'keypoints': normalized_frame,
            'subset': frame_data['subset'],
            'norm_params': norm_params # Add the saved parameters
        })

    print(f"\nSaving normalized data (with params) to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(output_data_list, f)

    print("Done!")
    print("  - Output now includes a 'norm_params' dictionary in each frame.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize pose data and save transformation parameters.")
    parser.add_argument("--input_path", type=str, help="Path to the input .pkl file.")
    parser.add_argument("--output_path", type=str, help="Path for the output normalized .pkl file.")
    args = parser.parse_args()
    normalize_pkl_and_save_params(args.input_path, args.output_path)