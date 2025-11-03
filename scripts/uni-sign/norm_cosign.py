#!/usr/bin/env python3
"""
CoSign-style normalization script extracted from Uni-Sign project.
This script normalizes pose keypoints using group-wise centralization and scale normalization.

Usage:
    python normalize_cosign.py input.pkl output.pkl
"""

import pickle
import numpy as np
import copy
import sys


def load_part_kp(skeletons, confs, force_ok=False):
    """
    Load and normalize keypoints by part (body, left hand, right hand, face).

    Args:
        skeletons: List of (1, 133, 2) keypoint arrays
        confs: List of (1, 133) confidence arrays
        force_ok: Whether to force processing even with low confidence

    Returns:
        Dictionary with normalized keypoints for each part
    """
    thr = 0.3
    kps_with_scores = {}
    scale = None

    for part in ['body', 'left', 'right', 'face_all']:
        kps = []
        confidences = []

        for skeleton, conf in zip(skeletons, confs):
            # skeleton shape: (1, 133, 2), conf shape: (1, 133)
            skeleton = skeleton[0]  # Now (133, 2)
            conf = conf[0]  # Now (133,)

            if part == 'body':
                # Body keypoints: nose + hips/shoulders/elbows/wrists
                hand_kp2d = skeleton[[0] + [i for i in range(3, 11)], :]
                confidence = conf[[0] + [i for i in range(3, 11)]]
            elif part == 'left':
                # Left hand keypoints (indices 91-112) - centered on wrist
                hand_kp2d = skeleton[91:112, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]  # Subtract wrist position
                confidence = conf[91:112]
            elif part == 'right':
                # Right hand keypoints (indices 112-133) - centered on wrist
                hand_kp2d = skeleton[112:133, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]  # Subtract wrist position
                confidence = conf[112:133]
            elif part == 'face_all':
                # Face keypoints - centered on nose
                hand_kp2d = skeleton[[i for i in list(range(23,23+17))[::2]] + [i for i in range(83, 83 + 8)] + [53], :]
                hand_kp2d = hand_kp2d - hand_kp2d[-1, :]  # Subtract nose position
                confidence = conf[[i for i in list(range(23,23+17))[::2]] + [i for i in range(83, 83 + 8)] + [53]]
            else:
                raise NotImplementedError

            kps.append(hand_kp2d)
            confidences.append(confidence)

        kps = np.stack(kps, axis=0)
        confidences = np.stack(confidences, axis=0)

        if part == 'body':
            # For body, compute the scale from bounding box
            if force_ok:
                result, scale, _ = crop_scale(np.concatenate([kps, confidences[...,None]], axis=-1), thr)
            else:
                result, scale, _ = crop_scale(np.concatenate([kps, confidences[...,None]], axis=-1), thr)
        else:
            # For other parts, use the body scale
            assert scale is not None
            result = np.concatenate([kps, confidences[...,None]], axis=-1)
            if scale == 0:
                result = np.zeros(result.shape)
            else:
                result[...,:2] = (result[..., :2]) / scale
                result = np.clip(result, -1, 1)
                # Mask low confidence keypoints
                result[result[...,2] <= thr] = 0

        kps_with_scores[part] = result

    return kps_with_scores


def crop_scale(motion, thr):
    """
    Normalize keypoints to [-1, 1] based on bounding box.

    Args:
        motion: Array of shape [(M), T, N, 3] with (x, y, confidence)
        thr: Confidence threshold for valid keypoints

    Returns:
        result: Normalized motion
        scale: Scaling factor used
        offset: Translation offset [xs, ys]
    """
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2] > thr][:,:2]

    if len(valid_coords) < 4:
        return np.zeros(motion.shape), 0, None

    xmin = min(valid_coords[:,0])
    xmax = max(valid_coords[:,0])
    ymin = min(valid_coords[:,1])
    ymax = max(valid_coords[:,1])

    ratio = 1
    scale = max(xmax-xmin, ymax-ymin) * ratio

    if scale == 0:
        return np.zeros(motion.shape), 0, None

    # Center the bounding box
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2

    # Normalize to [0, 1] then to [-1, 1]
    result[...,:2] = (motion[..., :2] - [xs,ys]) / scale
    result[...,:2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)

    # Mask low confidence keypoints
    result[result[...,2] <= thr] = 0

    return result, scale, [xs,ys]


def normalize_pkl(input_path, output_path):
    """
    Normalize a pickle file containing keypoints and scores.

    Args:
        input_path: Path to input .pkl file
        output_path: Path to output normalized .pkl file
    """
    print(f"Loading {input_path}...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    if 'keypoints' not in data or 'scores' not in data:
        raise ValueError("Input pickle must contain 'keypoints' and 'scores' keys")

    keypoints = np.array(data['keypoints'])
    scores = np.array(data['scores'])

    print(f"Input shape: keypoints={keypoints.shape}, scores={scores.shape}")

    # Convert to list format expected by load_part_kp
    # Input shape: (T, 1, 133, 2) -> need (1, 133, 2) per frame
    T = keypoints.shape[0]
    skeletons = [keypoints[i] for i in range(T)]  # Each is (1, 133, 2)
    confs = [scores[i] for i in range(T)]  # Each is (1, 133)

    print("Normalizing keypoints...")
    kps_with_scores = load_part_kp(skeletons, confs, force_ok=True)

    # Save normalized data
    print(f"Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(kps_with_scores, f)

    print("Done!")
    print("\nNormalized output contains:")
    for part, data in kps_with_scores.items():
        print(f"  {part}: shape={data.shape}, range=[{data.min():.3f}, {data.max():.3f}]")


if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Usage: python normalize_cosign.py input.pkl output.pkl")
    #     sys.exit(1)

    input_path = 'data/Asl-pose/0a0px34qrq.pkl'
    output_path = 'data/norm/0a0px34qrq.pkl'

    normalize_pkl(input_path, output_path)
