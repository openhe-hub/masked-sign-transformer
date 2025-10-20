import math
import numpy as np
import cv2

eps = 0.01
stickwidth = 4

def draw_bodypose(canvas, candidate, subset, color, mask=None):
    """
    Draws body keypoints and skeletal connections using subset data.
    """
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)
    
    red = (0, 0, 255)
    blue = (255, 0, 0)
    bone_color = (0, 255, 0)

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    # Draw connections
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            Y , X = candidate[index.astype(int), 0], candidate[index.astype(int), 1]
            if (X[0] == -1 and Y[0] == -1) or (X[1] == -1 and Y[1] == -1): 
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, bone_color)

    # Draw keypoints
    # Iterate through the 18 body part indices in the subset
    for i in range(len(candidate)):
        x, y = candidate[i][0:2]
        if i >= 18: continue
        if x < eps or y < eps:
            continue
        x, y = int(x * W), int(y * H) 
        point_color = color
        if mask is not None:
            # The mask corresponds to the 18 body parts
            point_color = red if mask[i] else blue
        cv2.circle(canvas, (x, y), 4, point_color, thickness=-1)
    
    return canvas

def draw_handpose(canvas, all_hand_peaks, color, mask=None):
    H, W, C = canvas.shape
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    
    red = (0, 0, 255)
    blue = (255, 0, 0)

    for hand_idx, peaks in enumerate(all_hand_peaks):
        hand_mask = mask[hand_idx] if mask is not None else None
        for ie, e in enumerate(edges):
            p1_idx, p2_idx = e
            x1, y1 = peaks[p1_idx]; x2, y2 = peaks[p2_idx]
            x1 = int(x1 * W); y1 = int(y1 * H); x2 = int(x2 * W); y2 = int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                line_color = color
                if hand_mask is not None:
                    line_color = red if hand_mask[p1_idx] or hand_mask[p2_idx] else blue
                cv2.line(canvas, (x1, y1), (x2, y2), line_color, thickness=2)
        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x = int(x * W); y = int(y * H)
            if x > eps and y > eps:
                point_color = color
                if hand_mask is not None:
                    point_color = red if hand_mask[i] else blue
                cv2.circle(canvas, (x, y), 4, point_color, thickness=-1)
    return canvas

def draw_facepose(canvas, all_lmks, color, mask=None):
    H, W, C = canvas.shape
    red = (0, 0, 255)
    blue = (255, 0, 0)
    for lmks in all_lmks:
        face_mask = mask
        for i, lmk in enumerate(lmks):
            x, y = lmk
            x = int(x * W); y = int(y * H)
            if x > eps and y > eps:
                point_color = color
                if face_mask is not None:
                    point_color = red if face_mask[i] else blue
                cv2.circle(canvas, (x, y), 3, point_color, thickness=-1)
    return canvas

def draw_pose(pose, subset, H, W, ref_w=2160, color=(255, 0, 0), mask=None):
    # `pose` is the full keypoints array, `subset` contains the indices for one person
    bodies = pose[:18]
    faces = pose[18:86].reshape(1, 68, 2)
    hands = pose[86:].reshape(2, 21, 2)

    body_mask = None
    face_mask = None
    hand_mask = None
    if mask is not None:
        body_mask = mask[:18]
        face_mask = mask[18:86]
        hand_mask = mask[86:].reshape(2, 21)

    sz = min(H, W)
    sr = (ref_w / sz) if sz != ref_w else 1
    canvas = np.ones(shape=(int(H*sr), int(W*sr), 3), dtype=np.uint8) * 255

    # Pass the full pose array as candidate, and the first person's subset
    # Assuming subset has shape (num_people, 20), we take the first person.
    canvas = draw_bodypose(canvas, pose, subset, color=color, mask=body_mask)

    canvas = draw_handpose(canvas, hands, color=color, mask=hand_mask)
    canvas = draw_facepose(canvas, faces, color=color, mask=face_mask)
    
    return cv2.cvtColor(cv2.resize(canvas, (W, H)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
