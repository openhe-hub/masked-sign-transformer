#!/usr/bin/env python3
"""
Batch normalization script to process all .pkl files in a folder.

This script applies part-wise normalization (saving transformation parameters)
to each .pkl file in an input directory and saves the results to an output
directory with the same filenames.

Input Format per file: List of T dictionaries [{'keypoints': (1, 128, 3), 'subset': ...}]
Output Format per file: List of T dictionaries [{'keypoints': (128, 3), 'subset': ..., 'norm_params': {...}}]

Usage:
    python batch_normalize.py /path/to/your/raw_data_folder /path/to/your/output_folder
"""

import os
import pickle
import numpy as np
import argparse
from tqdm import tqdm

def get_bbox_scale_offset(motion, thr):
    """Calculates normalization parameters based on the bounding box."""
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

def process_single_file(input_path, output_path):
    """
    Core logic to normalize one .pkl file and save the result.
    This function contains the part-wise normalization logic from previous scripts.
    """
    with open(input_path, 'rb') as f:
        data_list = pickle.load(f)

    if not isinstance(data_list, list):
        raise TypeError("Input .pkl file must contain a list of dictionaries.")
    
    T = len(data_list)
    
    # --- Define constants ---
    BODY_INDICES = slice(0, 18)
    FACE_INDICES = slice(18, 86)
    LEFT_HAND_INDICES = slice(86, 107)
    RIGHT_HAND_INDICES = slice(107, 128)
    LEFT_WRIST_IDX_LOCAL, RIGHT_WRIST_IDX_LOCAL = 0, 0
    CONF_THRESHOLD = 0.3

    output_data_list = []
    for frame_data in data_list:
        keypoints = np.array(frame_data['keypoints'])
        norm_params = {}
        normalized_frame = np.zeros_like(keypoints)
        
        body_kps = keypoints[BODY_INDICES]
        body_scale, body_offset = get_bbox_scale_offset(body_kps, thr=CONF_THRESHOLD)
        norm_params['body_scale'] = body_scale
        norm_params['body_offset'] = body_offset

        if body_scale != 0:
            # Normalize body
            norm_body_coords = ((body_kps[:, :2] - body_offset) / body_scale - 0.5) * 2
            normalized_frame[BODY_INDICES, :2] = np.clip(norm_body_coords, -1, 1)
            normalized_frame[BODY_INDICES, 2] = body_kps[:, 2]

            # Normalize other parts
            parts_to_process = {
                'face': (FACE_INDICES, None),
                'left_hand': (LEFT_HAND_INDICES, LEFT_WRIST_IDX_LOCAL),
                'right_hand': (RIGHT_HAND_INDICES, RIGHT_WRIST_IDX_LOCAL),
            }
            for part_name, (indices, center_idx) in parts_to_process.items():
                part_kps = keypoints[indices].copy()
                if center_idx is not None:
                    center_point = part_kps[center_idx, :2].copy()
                else:
                    valid_kps = part_kps[part_kps[:, 2] > CONF_THRESHOLD, :2]
                    center_point = valid_kps.mean(axis=0) if len(valid_kps) > 0 else np.zeros(2)
                
                norm_params[f'{part_name}_center'] = center_point
                norm_part_coords = (part_kps[:, :2] - center_point) / body_scale
                normalized_frame[indices, :2] = np.clip(norm_part_coords, -1, 1)
                normalized_frame[indices, 2] = part_kps[:, 2]

            low_conf_mask = normalized_frame[:, 2] <= CONF_THRESHOLD
            normalized_frame[low_conf_mask] = 0

        output_data_list.append({
            'keypoints': normalized_frame,
            'subset': frame_data['subset'],
            'norm_params': norm_params
        })

    with open(output_path, 'wb') as f:
        pickle.dump(output_data_list, f)

def main():
    parser = argparse.ArgumentParser(description="Batch normalize .pkl files in a directory.")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing raw .pkl files.")
    parser.add_argument("output_folder", type=str, help="Path to the folder where normalized files will be saved.")
    args = parser.parse_args()

    # --- 1. Setup Input and Output Folders ---
    input_dir = args.input_folder
    output_dir = args.output_folder
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # --- 2. Find all .pkl files to process ---
    print(f"Scanning for .pkl files in '{input_dir}'...")
    try:
        all_files = os.listdir(input_dir)
        pkl_files = [f for f in all_files if f.lower().endswith('.pkl')]
    except FileNotFoundError:
        print(f"Error: Input folder not found at '{input_dir}'")
        return

    if not pkl_files:
        print("No .pkl files found. Exiting.")
        return
        
    print(f"Found {len(pkl_files)} files to process.")

    # --- 3. Loop through files and process them ---
    for filename in tqdm(pkl_files, desc="Normalizing files"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            process_single_file(input_path, output_path)
        except Exception as e:
            # Print an error message but continue with the next file
            print(f"\n[ERROR] Failed to process {filename}: {e}")
            print("Skipping this file and continuing...")

    print("\nBatch processing complete!")
    print(f"Normalized files have been saved to '{output_dir}'.")


if __name__ == "__main__":
    # You might need to install tqdm if you haven't already:
    # pip install tqdm
    main()