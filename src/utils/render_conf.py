import math
import numpy as np
import matplotlib
import cv2

eps = 0.01
stickwidth = 4

def alpha_blend_color(color, alpha):
    """blend color according to point conf
    """
    return [int(c * alpha) for c in color]

def draw_bodypose(canvas, candidate, subset):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            # conf = score[n][np.array(limbSeq[i]) - 1]
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, alpha_blend_color(colors[i], 1.0))

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            conf = 1.0 # score[n][i]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, alpha_blend_color(colors[i], conf), thickness=-1)

    return canvas

def draw_handpose(canvas, all_hand_peaks):
    H, W, C = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for peaks in all_hand_peaks:

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            score = 255 # int(scores[e[0]] * scores[e[1]] * 255)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(canvas, (x1, y1), (x2, y2), 
                         matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * score, thickness=2)

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            score = 255 # int(scores[i] * 255)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 4, (0, 0, score), thickness=-1)
    return canvas

def draw_facepose(canvas, all_lmks):
    H, W, C = canvas.shape
    for lmks in all_lmks:
        for lmk in lmks:
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            conf = 255 # int(score * 255)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 3, (conf, conf, conf), thickness=-1)
    return canvas

def draw_pose(pose, subset, H, W, ref_w=2160):
    # `pose` is the full keypoints array, `subset` contains the indices for one person
    bodies = pose[:18]
    faces = pose[18:86].reshape(1, 68, 2)
    hands = pose[86:].reshape(2, 21, 2)

    sz = min(H, W)
    sr = (ref_w / sz) if sz != ref_w else 1
    canvas = np.ones(shape=(int(H*sr), int(W*sr), 3), dtype=np.uint8)

    # Pass the full pose array as candidate, and the first person's subset
    # Assuming subset has shape (num_people, 20), we take the first person.
    canvas = draw_bodypose(canvas, pose, subset)

    canvas = draw_handpose(canvas, hands)
    canvas = draw_facepose(canvas, faces)
    
    return cv2.cvtColor(cv2.resize(canvas, (W, H)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
