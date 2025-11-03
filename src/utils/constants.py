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
    'body': [0, 3, 4, 5, 6, 7, 8, 9, 10],
    'face_all': list(range(70, 88)), # Simplified face (18 kps)
    'left': list(range(91, 112)),   # Left hand (21 kps)
    'right': list(range(112, 133)), # Right hand (21 kps)
}

# Verify that the indices are correct and do not overlap if they shouldn't.
# This is a basic sanity check.
all_indices = []
for part, indices in PART_KP_INDICES.items():
    all_indices.extend(indices)
if len(all_indices) != len(set(all_indices)):
    print("Warning: Overlapping keypoint indices between parts.")

