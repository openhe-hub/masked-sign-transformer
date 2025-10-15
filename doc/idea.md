## Stage 1 (Single Direction): Based on MAE + Transformer
### Application
If we want to track / predict a T+1 frame keypoints providing `N_window` historical frames
### Dataset
* Size: (N_video, N_frame, N_kps, 3), 3 for `x`, `y`, `conf`
### Spatial-Temporal Mask Design
Plan to test 3 modes:
1. By Randon: randomly ban `x%` kps
2. By Part: ban some certain parts, e.g. hands
3. By Confidence: ban confidence lower than `conf_thres`

## Stage 2 (Double Direction): Based on Bert
### Application
If we want to add transition to Segment A & Segment B, especially in chatsign dict, providing `N_window_prev` & `N_window_next` historical frames