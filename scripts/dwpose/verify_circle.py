#!/usr/bin/env python3
"""
Verifies the consistency of the normalization and denormalization cycle.

This script compares an original data file with a file that has been normalized
and then denormalized. It reports the maximum difference found, which should be
a very small number due to floating-point precision.

Usage:
    python verify_cycle.py original_data.pkl denormalized_data.pkl
"""

import pickle
import numpy as np
import argparse

def verify_normalization_cycle(original_path, denormalized_path):
    print("--- Normalization Cycle Verification ---")
    
    # --- Load both files ---
    print(f"Loading original data from: {original_path}")
    with open(original_path, 'rb') as f:
        original_list = pickle.load(f)

    print(f"Loading denormalized data from: {denormalized_path}")
    with open(denormalized_path, 'rb') as f:
        denormalized_list = pickle.load(f)

    # --- Basic Sanity Checks ---
    if len(original_list) != len(denormalized_list):
        print(f"Error: Frame count mismatch! Original={len(original_list)}, Denormalized={len(denormalized_list)}")
        return

    T = len(original_list)
    CONF_THRESHOLD = 0.3
    overall_max_diff = 0.0
    
    # --- Compare Frame by Frame ---
    for i in range(T):
        original_kps = np.array(original_list[i]['keypoints']).squeeze()
        denormalized_kps = np.array(denormalized_list[i]['keypoints']).squeeze()

        # --- CRITICAL: Only compare keypoints with high confidence ---
        # We create a mask based on the *original* confidence scores.
        valid_mask = original_kps[:, 2] > CONF_THRESHOLD
        
        # If there are no valid keypoints in this frame, skip comparison
        if not np.any(valid_mask):
            continue

        # Select the coordinates (x, y) of the valid keypoints from both arrays
        original_coords_valid = original_kps[valid_mask, :2]
        denormalized_coords_valid = denormalized_kps[valid_mask, :2]

        # Calculate the absolute difference for this frame
        frame_diff = np.abs(original_coords_valid - denormalized_coords_valid)
        max_frame_diff = np.max(frame_diff)
        
        # Keep track of the worst-case difference across all frames
        if max_frame_diff > overall_max_diff:
            overall_max_diff = max_frame_diff

    # --- Final Report ---
    print("\n--- Verification Report ---")
    print(f"Compared {T} frames.")
    print(f"Maximum coordinate difference found (for keypoints with conf > {CONF_THRESHOLD}): {overall_max_diff}")

    # Use np.allclose to check if arrays are "close enough"
    # We re-check the last valid frame for a clear True/False result
    is_close = np.allclose(original_coords_valid, denormalized_coords_valid, atol=1e-5)

    if is_close and overall_max_diff < 1e-5:
        print("\n✅ Verification PASSED.")
        print("The data is identical up to negligible floating-point precision differences.")
    else:
        print("\n❌ Verification FAILED.")
        print("The difference is larger than expected. Please check the logic.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify the norm -> denorm cycle.")
    parser.add_argument("original_path", type=str, help="Path to the original raw .pkl file.")
    parser.add_argument("denormalized_path", type=str, help="Path to the file after it has been normalized and then denormalized.")
    args = parser.parse_args()

    verify_normalization_cycle(args.original_path, args.denormalized_path)