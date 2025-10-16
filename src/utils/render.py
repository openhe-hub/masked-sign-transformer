import math
import numpy as np
import cv2

eps = 0.01

# 因为不再使用置信度进行颜色混合，这个函数现在只是一个占位符，可以删除，但为了清晰保留了原函数名对应的逻辑部分
# 旧的 alpha_blend_color 函数已经被移除

def draw_bodypose(canvas, candidate, color):
    """
    使用指定的单一颜色绘制身体关键点（不画连接线）。
    完全不依赖 subset 数据。
    """
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    
    # 只需遍历 candidate 中的所有点。

    # 遍历 candidate 中的所有关节点
    # 注意：这里的 i 只是 candidate 中的索引，不再对应 limbSeq 或关键点类型
    for i in range(len(candidate)):
        x, y = candidate[i][0:2]
        # 注意：这里的 score[0][i] 假设 score 是一个包含至少一个人的列表，
        # 且 candidate[i] 对应 score[0][i]。
        # 如果你的 score 只是一个平坦的列表，你需要调整为 score[i]
        
        # 鉴于原始代码中 bodies['score'] 是 (N_people, 18) 的结构，
        # 且你只有一个人，我们假定 score[0] 是这个人的18个点的置信度。
        # 如果你确定 candidate 的长度不大于 18，并且你想对齐置信度：
        
        # --------------------------------------------------------------------------------
        # *** 关键假设：只有一个人，且 candidate 和 score[0] 是对应的 ***
        # --------------------------------------------------------------------------------
        
        # 如果你确定 body 的关节点只有18个，并且 candidate 包含了这18个点：
        if i >= 18: continue # 限制只处理前18个关节点（OpenPose/DWPose标准）

        # 检查点是否有效 (原代码中是检查 index != -1，这里我们检查坐标有效性)
        if x < eps or y < eps:
            continue
            
        x = int(x * W)
        y = int(y * H)
        
        # 假设我们只画出点，不依赖置信度进行颜色变化
        cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)

    # ----------------------------------------------------------------------
    # 提示：由于你移除了subset，我们无法像以前那样准确地知道 candidate 列表中
    # 的哪些点是有效的。上面的代码简单地假设 candidate 列表中的点都是你想要画的
    # 身体关键点（前18个）。
    # ----------------------------------------------------------------------
    
    return canvas

def draw_handpose(canvas, all_hand_peaks, color):
    H, W, C = canvas.shape
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    for peaks in all_hand_peaks:
        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]; x2, y2 = peaks[e[1]]
            x1 = int(x1 * W); y1 = int(y1 * H); x2 = int(x2 * W); y2 = int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(canvas, (x1, y1), (x2, y2), color, thickness=2)
        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x = int(x * W); y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 4, color, thickness=-1)
    return canvas

def draw_facepose(canvas, all_lmks, color):
    H, W, C = canvas.shape
    for lmks in all_lmks:
        for lmk in lmks:
            x, y = lmk
            x = int(x * W); y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 3, color, thickness=-1)
    return canvas

def draw_pose(pose, H, W, ref_w=2160, color=(255, 0, 0)):
    bodies = pose[:18]
    faces = pose[18:86].reshape(1, 68, 2)
    hands = pose[86:].reshape(2, 21, 2)

    sz = min(H, W)
    sr = (ref_w / sz) if sz != ref_w else 1
    canvas = np.ones(shape=(int(H*sr), int(W*sr), 3), dtype=np.uint8) * 255

    # draw_bodypose 现在只画点，不画线，且不依赖 subset
    canvas = draw_bodypose(canvas, bodies, color=color)

    canvas = draw_handpose(canvas, hands, color=color)
    canvas = draw_facepose(canvas, faces, color=color)
    
    return cv2.cvtColor(cv2.resize(canvas, (W, H)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)