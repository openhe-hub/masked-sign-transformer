import math
import numpy as np
import matplotlib
import cv2

eps = 0.01
stickwidth = 4
BODY_LIMB_SEQ = [
    [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
    [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
    [1, 16], [16, 18], [3, 17], [6, 18],
]
HAND_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15],
    [15, 16], [0, 17], [17, 18], [18, 19], [19, 20],
]
BODY_CONF_THRESHOLD = 0.3


def alpha_blend_color(color, alpha):
    """Blend color intensity according to confidence."""
    return [int(c * float(np.clip(alpha, 0.0, 1.0))) for c in color]


def draw_bodypose(canvas, candidate, subset, score):
    H, W, _ = canvas.shape
    candidate = np.asarray(candidate, dtype=np.float32)
    subset = np.asarray(subset)
    score = np.asarray(score, dtype=np.float32)

    if subset.ndim == 1:
        subset = subset[np.newaxis, ...]
        score = score[np.newaxis, ...]

    colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
        [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
        [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
        [255, 0, 255], [255, 0, 170], [255, 0, 85],
    ]

    max_limbs = min(17, len(BODY_LIMB_SEQ))
    for limb_idx in range(max_limbs):
        limb = BODY_LIMB_SEQ[limb_idx]
        subset_cols = np.array(limb) - 1
        if np.any(subset_cols >= subset.shape[1]):
            continue
        for person_idx in range(subset.shape[0]):
            index = subset[person_idx, subset_cols].astype(int)
            conf_pair = score[person_idx, subset_cols]
            if np.any(index < 0) or np.any(index >= candidate.shape[0]):
                continue
            if conf_pair[0] < BODY_CONF_THRESHOLD or conf_pair[1] < BODY_CONF_THRESHOLD:
                continue
            pts = candidate[index]
            Y = pts[:, 0] * float(W)
            X = pts[:, 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            if length < eps:
                continue
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)),
                (int(length / 2), stickwidth),
                int(angle),
                0,
                360,
                1,
            )
            cv2.fillConvexPoly(canvas, polygon, alpha_blend_color(colors[limb_idx], conf_pair[0] * conf_pair[1]))

    canvas = (canvas * 0.6).astype(np.uint8)

    joint_count = min(18, subset.shape[1])
    for person_idx in range(subset.shape[0]):
        for joint in range(joint_count):
            idx = int(subset[person_idx, joint])
            if idx == -1 or idx >= candidate.shape[0]:
                continue
            x, y = candidate[idx][:2]
            conf = score[person_idx, joint]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (x, y), 4, alpha_blend_color(colors[joint], conf), thickness=-1)

    return canvas


def draw_handpose(canvas, all_hand_peaks, all_hand_scores):
    H, W, _ = canvas.shape
    if not isinstance(all_hand_peaks, (list, tuple)):
        return canvas

    for peaks, scores in zip(all_hand_peaks, all_hand_scores):
        peaks = np.asarray(peaks, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        if peaks.shape != (21, 2) or scores.shape[0] != 21:
            continue

        for edge_idx, (s_idx, e_idx) in enumerate(HAND_EDGES):
            x1, y1 = peaks[s_idx]
            x2, y2 = peaks[e_idx]
            if min(x1, y1, x2, y2) <= eps:
                continue
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            edge_score = int(scores[s_idx] * scores[e_idx] * 255)
            color = matplotlib.colors.hsv_to_rgb([edge_idx / float(len(HAND_EDGES)), 1.0, 1.0]) * edge_score
            cv2.line(canvas, (x1, y1), (x2, y2), color.astype(np.uint8).tolist(), thickness=2)

        for kp_idx, (x, y) in enumerate(peaks):
            if min(x, y) <= eps:
                continue
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (x, y), 4, (0, 0, int(scores[kp_idx] * 255)), thickness=-1)

    return canvas


def draw_facepose(canvas, all_lmks, all_scores):
    H, W, _ = canvas.shape
    all_lmks = np.asarray(all_lmks, dtype=np.float32)
    all_scores = np.asarray(all_scores, dtype=np.float32)

    if all_lmks.ndim != 3 or all_scores.ndim != 2:
        return canvas

    for lmks, scores in zip(all_lmks, all_scores):
        for (x, y), conf in zip(lmks, scores):
            if min(x, y) <= eps:
                continue
            x = int(x * W)
            y = int(y * H)
            c = int(conf * 255)
            cv2.circle(canvas, (x, y), 3, (c, c, c), thickness=-1)

    return canvas


def _ensure_length(xy, conf, target):
    xy = np.asarray(xy, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32)
    if xy.shape[0] < target:
        pad = target - xy.shape[0]
        xy = np.pad(xy, ((0, pad), (0, 0)), mode="constant")
        conf = np.pad(conf, (0, pad), constant_values=0.0)
    elif xy.shape[0] > target:
        xy = xy[:target]
        conf = conf[:target]
    return xy, np.clip(conf, 0.0, 1.0)


def _build_body_scores(subset, joint_conf):
    subset = np.asarray(subset, dtype=np.int32)
    if subset.ndim == 1:
        subset = subset[np.newaxis, ...]
    scores = np.zeros_like(subset, dtype=np.float32)
    for person_idx in range(subset.shape[0]):
        for joint in range(min(18, subset.shape[1])):
            idx = subset[person_idx, joint]
            if idx == -1:
                continue
            if 0 <= idx < joint_conf.shape[0]:
                scores[person_idx, joint] = joint_conf[idx]
    return subset, scores


def draw_pose(pose, subset, H, W, ref_w=2160):
    pose = np.asarray(pose, dtype=np.float32)
    if pose.ndim != 2 or pose.shape[1] < 2:
        raise ValueError("Expected pose shaped (N, >=2).")

    xy = pose[:, :2]
    conf = pose[:, 2] if pose.shape[1] > 2 else np.ones(xy.shape[0], dtype=np.float32)
    xy, conf = _ensure_length(xy, conf, 128)

    body_xy = xy[:18]
    body_conf = conf[:18]

    subset_arr, subset_scores = _build_body_scores(subset, body_conf)

    faces_xy = xy[18:86].reshape(1, 68, 2)
    faces_conf = conf[18:86].reshape(1, 68)

    right_hand_xy = xy[86:107]
    right_hand_conf = conf[86:107]
    left_hand_xy = xy[107:128]
    left_hand_conf = conf[107:128]

    hands_xy = [right_hand_xy, left_hand_xy]
    hands_conf = [right_hand_conf, left_hand_conf]

    sz = min(H, W)
    sr = (ref_w / sz) if sz != ref_w else 1
    canvas = np.zeros(shape=(int(H * sr), int(W * sr), 3), dtype=np.uint8)

    canvas = draw_bodypose(canvas, body_xy, subset_arr, score=subset_scores)
    canvas = draw_handpose(canvas, hands_xy, hands_conf)
    canvas = draw_facepose(canvas, faces_xy, faces_conf)

    return cv2.cvtColor(cv2.resize(canvas, (W, H)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
