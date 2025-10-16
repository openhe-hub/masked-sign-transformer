import math
import numpy as np
import cv2

eps = 0.01

def alpha_blend_color(color, alpha):
    """
    根据点的置信度混合颜色。
    """
    # 确保alpha在0到1之间
    alpha = max(0.0, min(1.0, alpha))
    return [int(c * alpha) for c in color]

def draw_bodypose(canvas, candidate, subset, score, color):
    """
    使用指定的单一颜色绘制身体姿态。
    """
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    # 移除了固定的多色列表
    # colors = [...]

    for i in range(len(limbSeq)): # 循环肢体序列
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            conf = score[n][np.array(limbSeq[i]) - 1]
            if conf[0] < 0.3 or conf[1] < 0.3:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            # 使用传入的单一颜色进行绘制
            cv2.fillConvexPoly(canvas, polygon, alpha_blend_color(color, conf[0] * conf[1]))

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18): # 循环18个关键点
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            conf = score[n][i]
            x = int(x * W)
            y = int(y * H)
            # 使用传入的单一颜色进行绘制
            cv2.circle(canvas, (int(x), int(y)), 4, alpha_blend_color(color, conf), thickness=-1)

    return canvas

def draw_handpose(canvas, all_hand_peaks, all_hand_scores, color):
    """
    使用指定的单一颜色绘制手部姿态。
    """
    H, W, C = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for peaks, scores in zip(all_hand_peaks, all_hand_scores):

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            
            # 移除了基于matplotlib的hsv颜色生成
            # 使用传入的单一颜色进行绘制
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                # 使用两个连接点的分数的乘积作为透明度
                alpha = scores[e[0]] * scores[e[1]]
                cv2.line(canvas, (x1, y1), (x2, y2), alpha_blend_color(color, alpha), thickness=2)

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            
            if x > eps and y > eps:
                # 移除了 (0, 0, score) 的颜色
                # 使用传入的单一颜色进行绘制
                alpha = scores[i]
                cv2.circle(canvas, (x, y), 4, alpha_blend_color(color, alpha), thickness=-1)
    return canvas

def draw_facepose(canvas, all_lmks, all_scores, color):
    """
    使用指定的单一颜色绘制面部关键点。
    """
    H, W, C = canvas.shape
    for lmks, scores in zip(all_lmks, all_scores):
        for lmk, score in zip(lmks, scores):
            x, y = lmk
            x = int(x * W)
            y = int(y * H)

            if x > eps and y > eps:
                # 移除了 (conf, conf, conf) 的灰度颜色
                # 使用传入的单一颜色进行绘制
                alpha = score
                cv2.circle(canvas, (x, y), 3, alpha_blend_color(color, alpha), thickness=-1)
    return canvas

def draw_pose(pose, H, W, ref_w=2160, color=(255, 255, 255)):
    """
    可视化dwpose的输出。

    Args:
        pose (List): 来自DWposeDetector的输出。
        H (int): 高度。
        W (int): 宽度。
        ref_w (int, optional): 参考宽度。默认为2160。
        color (tuple, optional): 用于绘制的BGR颜色。默认为白色 (255, 255, 255)。

    Returns:
        np.ndarray: RGB模式的图像像素值。
    """
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']

    sz = min(H, W)
    sr = (ref_w / sz) if sz != ref_w else 1

    ########################################## 创建空白画布 ##################################################
    canvas = np.zeros(shape=(int(H*sr), int(W*sr), 3), dtype=np.uint8)

    ########################################### 绘制身体姿态 #####################################################
    canvas = draw_bodypose(canvas, candidate, subset, score=bodies['score'], color=color)

    ########################################### 绘制手部姿态 #####################################################
    canvas = draw_handpose(canvas, hands, pose['hands_score'], color=color)

    ########################################### 绘制面部姿态 #####################################################
    canvas = draw_facepose(canvas, faces, pose['faces_score'], color=color)

    return cv2.cvtColor(cv2.resize(canvas, (W, H)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)