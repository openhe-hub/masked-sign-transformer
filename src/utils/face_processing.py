import pickle
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from config_loader import config
from utils.post_process import post_process_align_naive

NECK_ID = 1
R_SHOULDER_ID = 2
R_ELBOW_ID = 3
R_WRIST_ID = 4
L_SHOULDER_ID = 5
L_ELBOW_ID = 6
L_WRIST_ID = 7
FACE_BODY_INDICES = [0, 14, 15, 16, 17]
RIGHT_HAND_SLICE = slice(86, 107)
LEFT_HAND_SLICE = slice(107, 128)
EPS = 1e-6


def _load_reference_pose_xy() -> Optional[np.ndarray]:
    """
    Load the full reference pose (xy coordinates) from the configuration.
    """
    ref_pose_path = config["data"].get("ref_img_pose")
    if not ref_pose_path:
        return None

    ref_path = Path(ref_pose_path)
    if not ref_path.exists():
        print(f"Warning: reference pose file not found at {ref_path}")
        return None

    with open(ref_path, "rb") as f:
        data = pickle.load(f)
    
    # import ipdb; ipdb.set_trace()

    if isinstance(data, dict) and "keypoints" in data:
        keypoints = np.asarray(data["keypoints"], dtype=np.float32)
    else:
        keypoints = np.asarray(data, dtype=np.float32)

    if keypoints.ndim == 3:
        keypoints = keypoints[0]

    if keypoints.shape[-1] >= 2:
        return keypoints[:, :2]

    print("Warning: reference pose does not contain xy coordinates.")
    return None


@lru_cache(maxsize=1)
def load_reference_face_keypoints() -> Optional[np.ndarray]:
    full_pose = _load_reference_pose_xy()
    if full_pose is None or full_pose.shape[0] < 86:
        print("Warning: reference keypoints do not contain full face data.")
        return None
    return full_pose[18:86]


def align_face_inplace(sequence: np.ndarray) -> None:
    """
    Apply naive alignment to the face keypoints in-place.

    Args:
        sequence (np.ndarray): Array of shape (T, 128, 2).
    """
    ref_face = load_reference_face_keypoints()
    # import ipdb; ipdb.set_trace()
    if ref_face is None:
        return

    face_seq = sequence[:, 18:86, :]
    aligned_face, _ = post_process_align_naive(ref_face, face_seq)
    sequence[:, 18:86, :] = aligned_face


def _align_subset_to_reference(subset: np.ndarray, ref_subset: np.ndarray) -> None:
    ref_min = ref_subset.min(axis=0)
    ref_max = ref_subset.max(axis=0)
    ref_center = (ref_min + ref_max) / 2.0
    ref_size = ref_max - ref_min
    ref_size[ref_size < EPS] = 1.0

    frame_min = subset.min(axis=0)
    frame_max = subset.max(axis=0)
    frame_center = (frame_min + frame_max) / 2.0
    frame_size = frame_max - frame_min
    frame_size[frame_size < EPS] = 1.0

    scale = ref_size / frame_size
    subset[:] = (subset - frame_center) * scale + ref_center


def graft_upper_body_inplace(sequence: np.ndarray) -> None:
    """
    Adjust arms and hands to match the reference limb lengths while preserving orientation.

    Args:
        sequence (np.ndarray): Array of shape (T, 128, 2).
    """
    ref_pose = _load_reference_pose_xy()
    if ref_pose is None or ref_pose.shape[0] < 128:
        return

    ref_body = ref_pose[:18]

    for frame in sequence:
        body = frame[:18]
        left_hand = frame[LEFT_HAND_SLICE]
        right_hand = frame[RIGHT_HAND_SLICE]

        dist_frame_left = np.linalg.norm(body[L_SHOULDER_ID] - body[NECK_ID])
        dist_ref_left = np.linalg.norm(ref_body[L_SHOULDER_ID] - ref_body[NECK_ID])
        scale_left = dist_ref_left / (dist_frame_left + EPS) if dist_frame_left > EPS else 1.0

        dist_frame_right = np.linalg.norm(body[R_SHOULDER_ID] - body[NECK_ID])
        dist_ref_right = np.linalg.norm(ref_body[R_SHOULDER_ID] - ref_body[NECK_ID])
        scale_right = dist_ref_right / (dist_frame_right + EPS) if dist_frame_right > EPS else 1.0

        # Left arm
        vec_l_sh_el = body[L_ELBOW_ID] - body[L_SHOULDER_ID]
        vec_l_el_wr = body[L_WRIST_ID] - body[L_ELBOW_ID]
        new_l_elbow = body[L_SHOULDER_ID] + vec_l_sh_el * scale_left
        new_l_wrist = new_l_elbow + vec_l_el_wr * scale_left
        orig_l_wrist = body[L_WRIST_ID].copy()
        body[L_ELBOW_ID] = new_l_elbow
        body[L_WRIST_ID] = new_l_wrist
        left_hand[:] = new_l_wrist + (left_hand - orig_l_wrist) * scale_left

        # Right arm
        vec_r_sh_el = body[R_ELBOW_ID] - body[R_SHOULDER_ID]
        vec_r_el_wr = body[R_WRIST_ID] - body[R_ELBOW_ID]
        new_r_elbow = body[R_SHOULDER_ID] + vec_r_sh_el * scale_right
        new_r_wrist = new_r_elbow + vec_r_el_wr * scale_right
        orig_r_wrist = body[R_WRIST_ID].copy()
        body[R_ELBOW_ID] = new_r_elbow
        body[R_WRIST_ID] = new_r_wrist
        right_hand[:] = new_r_wrist + (right_hand - orig_r_wrist) * scale_right

        _align_subset_to_reference(body[FACE_BODY_INDICES], ref_body[FACE_BODY_INDICES])
