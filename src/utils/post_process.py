import numpy as np
from typing import Tuple


def post_process_align_naive(ref_kpts: np.ndarray, result_animation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a simple alignment by matching bounding boxes between the reference
    keypoints and each frame of the animation.

    Args:
        ref_kpts (np.ndarray): Reference keypoints, shape (68, 2).
        result_animation (np.ndarray): Sequence of keypoints to align, shape (N, 68, 2).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Aligned animation (N, 68, 2) and the reference keypoints.
    """
    if ref_kpts is None or result_animation is None:
        raise ValueError("Reference keypoints and result animation must be provided.")

    if ref_kpts.shape[-2:] != (68, 2):
        raise ValueError(f"Expected ref_kpts shape (?, 68, 2), got {ref_kpts.shape}")

    if result_animation.shape[-2:] != (68, 2):
        raise ValueError(f"Expected result_animation shape (?, 68, 2), got {result_animation.shape}")

    ref_min = ref_kpts.min(axis=0)
    ref_max = ref_kpts.max(axis=0)
    ref_center = (ref_min + ref_max) / 2.0
    ref_size = ref_max - ref_min
    ref_size[ref_size < 1e-6] = 1e-6

    aligned_animation = np.zeros_like(result_animation)

    for idx, frame in enumerate(result_animation):
        frame_min = frame.min(axis=0)
        frame_max = frame.max(axis=0)
        frame_center = (frame_min + frame_max) / 2.0
        frame_size = frame_max - frame_min
        frame_size[frame_size < 1e-6] = 1e-6

        centered_frame = frame - frame_center
        scale_factors = ref_size / frame_size
        scaled_frame = centered_frame * scale_factors
        aligned_frame = scaled_frame + ref_center

        aligned_animation[idx] = aligned_frame

    return aligned_animation, ref_kpts
