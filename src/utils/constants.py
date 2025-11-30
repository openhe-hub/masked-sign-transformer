"""
This file contains constants used across the project.
"""

# Total number of keypoints in the pose data
TOTAL_KEYPOINTS = 133 # Example value, adjust if necessary

# Indices for different body parts based on a common pose estimation model
# (e.g., DWPOSE, OpenPose). These should be verified against the actual data format.

# Based on ssl_models_crossvideo.py's comments and common layouts:
# body: [0] (nose) + [3-10] (upper body) = 9 keypoints
# left/right hand: 21 keypoints each
# face: simplified face landmarks = 18 keypoints

PART_KP_INDICES = {
    'body': list(range(0, 18)),
    # 'face_all': list(range(70, 88)), # Simplified face (18 kps)
    'left': list(range(91, 112)),   # Left hand (21 kps)
    'right': list(range(112, 133)), # Right hand (21 kps)
}

# Optional extended part indices used by bridge models/datasets.
FACE_KP_INDICES = list(range(18, 86))


def get_part_kp_indices(include_face: bool = False):
    """
    Returns a copy of the PART_KP_INDICES dict, optionally extending it with face indices.
    """
    part_indices = dict(PART_KP_INDICES)
    if include_face and 'face' not in part_indices:
        part_indices['face'] = FACE_KP_INDICES
    return part_indices

# Verify that the indices are correct and do not overlap if they shouldn't.
# This is a basic sanity check.
all_indices = []
for part, indices in PART_KP_INDICES.items():
    all_indices.extend(indices)
if len(all_indices) != len(set(all_indices)):
    print("Warning: Overlapping keypoint indices between parts.")
