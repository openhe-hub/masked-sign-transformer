#!/usr/bin/env python3
"""
CoSign-style denormalization script.
This script reverses the normalization process performed by norm_cosign.py.

To denormalize, it requires the original (unnormalized) data to recalculate
the 'scale' and 'offset' parameters that were used during the initial normalization.

Usage:
    # This script is intended to be used as a module.
    # See the example in the `if __name__ == "__main__"` block.
"""

import pickle
import numpy as np
import copy
import sys
import os

def get_normalization_params(original_skeletons, original_confs, thr=0.3):
    """
    Calculates the scale and offset from the original, unnormalized data.
    This is a partial re-implementation of the logic in norm_cosign.py,
    but it only returns the transformation parameters.

    Args:
        original_skeletons (list): List of original skeleton arrays [(1, 133, 2), ...]
        original_confs (list): List of original confidence arrays [(1, 133), ...]
        thr (float): Confidence threshold for valid keypoints.

    Returns:
        tuple: A tuple containing:
            - scale (float): The scaling factor.
            - offset (list): The [xs, ys] offset.
            - original_kps (dict): Dictionary of original keypoints for wrist/nose positions.
    """
    # --- Extract Body Keypoints ---
    body_kps = []
    body_confs = []
    original_kps = {} # To store original wrist/nose positions for denormalization

    for skeleton, conf in zip(original_skeletons, original_confs):
        skeleton = skeleton[0] # Shape (133, 2)
        conf = conf[0] # Shape (133,)
        
        # Body keypoints: nose + hips/shoulders/elbows/wrists
        body_kp2d = skeleton[[0] + [i for i in range(3, 11)], :]
        body_confidence = conf[[0] + [i for i in range(3, 11)]]
        body_kps.append(body_kp2d)
        body_confs.append(body_confidence)

        # Store original keypoints needed for part-specific offsets
        original_kps.setdefault('left_wrist', []).append(skeleton[91, :])
        original_kps.setdefault('right_wrist', []).append(skeleton[112, :])
        original_kps.setdefault('nose', []).append(skeleton[53, :])

    body_kps = np.stack(body_kps, axis=0)
    body_confs = np.stack(body_confs, axis=0)
    
    for key in original_kps:
        original_kps[key] = np.stack(original_kps[key], axis=0)

    motion = np.concatenate([body_kps, body_confs[..., None]], axis=-1)
    
    # --- Calculate Scale and Offset from Body ---
    valid_coords = motion[motion[..., 2] > thr][:, :2]

    if len(valid_coords) < 4:
        return 0, [0, 0], original_kps

    xmin = min(valid_coords[:, 0])
    xmax = max(valid_coords[:, 0])
    ymin = min(valid_coords[:, 1])
    ymax = max(valid_coords[:, 1])

    ratio = 1
    scale = max(xmax - xmin, ymax - ymin) * ratio

    if scale == 0:
        return 0, [0, 0], original_kps

    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    offset = [xs, ys]

    return scale, offset, original_kps


def denormalize_data(normalized_data, scale, offset, original_kps):
    """
    Denormalizes pose data using the provided scale and offset.

    Args:
        normalized_data (dict): Dict of normalized keypoints {'body':..., 'left':...}
        scale (float): The scaling factor from get_normalization_params.
        offset (list): The [xs, ys] offset from get_normalization_params.
        original_kps (dict): Dict of original keypoints for wrist/nose positions.

    Returns:
        dict: A dictionary with denormalized keypoints for each part.
    """
    if scale == 0:
        print("Warning: Scale is 0, cannot denormalize. Returning zeroed data.")
        return {part: np.zeros_like(data) for part, data in normalized_data.items()}

    denormalized_output = {}

    for part, data in normalized_data.items():
        # The data has shape (T, N, 3) where last dim is (x, y, conf)
        coords = data[..., :2] # Extract just (x, y)
        
        # Reverse the normalization: norm_coords = ((original_coords - offset) / scale - 0.5) * 2
        # original_coords = ((norm_coords / 2) + 0.5) * scale + offset
        denorm_coords = ((coords / 2.0) + 0.5) * scale + offset
        
        # For parts that were centered, add their original center back
        if part == 'left':
            denorm_coords += original_kps['left_wrist'][:, np.newaxis, :]
        elif part == 'right':
            denorm_coords += original_kps['right_wrist'][:, np.newaxis, :]
        elif part == 'face_all':
            denorm_coords += original_kps['nose'][:, np.newaxis, :]

        # Re-attach confidence scores
        denormalized_output[part] = np.concatenate([denorm_coords, data[..., 2:3]], axis=-1)
        
    return denormalized_output


def denormalize_from_file(normalized_pkl_path, original_pkl_path):
    """
    High-level function to denormalize a file using its original counterpart.
    """
    # Load normalized data
    with open(normalized_pkl_path, 'rb') as f:
        normalized_data = pickle.load(f)
        
    # Load original data to get params
    with open(original_pkl_path, 'rb') as f:
        original_data = pickle.load(f)
    
    keypoints = np.array(original_data['keypoints'])
    scores = np.array(original_data['scores'])
    T = keypoints.shape[0]
    skeletons = [keypoints[i] for i in range(T)]
    confs = [scores[i] for i in range(T)]

    # 1. Get normalization parameters from original file
    scale, offset, original_kps = get_normalization_params(skeletons, confs)

    # 2. Denormalize the data
    denormalized_data = denormalize_data(normalized_data, scale, offset, original_kps)
    
    return denormalized_data


if __name__ == "__main__":
    print("--- Denormalization Script Example ---")
    
    # Example paths
    # This is the file that has been normalized by norm_cosign.py
    normalized_path = 'data/norm/0a0px34qrq.pkl'
    # This is the original file before normalization
    original_path = 'data/Asl-pose/0a0px34qrq.pkl'
    # Path to save the recovered (denormalized) file
    output_path = 'data/norm/recovered_0a0px34qrq.pkl'

    print(f"Normalized file: {normalized_path}")
    print(f"Original file for reference: {original_path}")

    if not os.path.exists(normalized_path) or not os.path.exists(original_path):
        print("\nError: Example files not found. Please adjust the paths.")
        print("This script expects the following files to exist:")
        print(f"- {normalized_path}")
        print(f"- {original_path}")
        sys.exit(1)

    # Perform the denormalization
    recovered_data = denormalize_from_file(normalized_path, original_path)

    # Save the recovered data
    with open(output_path, 'wb') as f:
        pickle.dump(recovered_data, f)

    print(f"\nSuccessfully denormalized and saved recovered data to: {output_path}")

    # Verification (optional)
    print("\nVerification: Comparing a sample point.")
    original_body_kp = np.array(pickle.load(open(original_path, 'rb'))['keypoints'])[0, 0, 0, :]
    recovered_body_kp = recovered_data['body'][0, 0, :2]
    print(f"Original body keypoint 0 (frame 0): {original_body_kp}")
    print(f"Recovered body keypoint 0 (frame 0): {recovered_body_kp}")
    print("Note: A small difference is expected due to floating point precision.")
