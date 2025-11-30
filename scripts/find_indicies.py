# file: find_first_20_indices.py
import pickle
import matplotlib.pyplot as plt
import numpy as np

# ==================== 配置 ====================
# 置信度阈值，低于这个分数的点将被忽略
CONFIDENCE_THRESHOLD = 0.3 
# 选择你想看哪一帧
FRAME_TO_VIEW = 89
# ✨ 我们只关心前 N 个关键点
NUM_POINTS_TO_CONSIDER = 133
# ============================================

# --- 1. 加载数据 ---
file_path = 'data/Asl-pose/0a0px34qrq.pkl' 
try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
except FileNotFoundError:
    print(f"错误：找不到文件 '{file_path}'。")
    exit()

# 提取指定帧的完整数据
keypoints_full = np.squeeze(data['keypoints'])[FRAME_TO_VIEW]
scores_full = np.squeeze(data['scores'])[FRAME_TO_VIEW]

# --- 2. 筛选我们关心的点 ---
# ✨ 首先，只获取前20个点的坐标和分数
keypoints_subset = keypoints_full[:NUM_POINTS_TO_CONSIDER]
scores_subset = scores_full[:NUM_POINTS_TO_CONSIDER]

# ✨ 然后，在这些点中，找到分数高于阈值的点的索引 (这些索引是相对于0-19的)
high_conf_indices = np.where(scores_subset > CONFIDENCE_THRESHOLD)[0]

# 根据这些索引，提取出最终要绘制的有效关键点坐标
points_to_plot = keypoints_subset[high_conf_indices]

if len(points_to_plot) == 0:
    print(f"警告：在第 {FRAME_TO_VIEW} 帧的前 {NUM_POINTS_TO_CONSIDER} 个点中，")
    print(f"没有找到置信度高于 {CONFIDENCE_THRESHOLD} 的点。")
    exit()

print(f"在第 {FRAME_TO_VIEW} 帧的前 {NUM_POINTS_TO_CONSIDER} 个点中，找到 {len(points_to_plot)} 个高置信度的点进行显示。")

# --- 3. 准备绘图 ---
fig, ax = plt.subplots(figsize=(12, 12))

# ✨ 计算绘图范围 (只使用最终要绘制的点)
x_min, x_max = np.min(points_to_plot[:, 0]), np.max(points_to_plot[:, 0])
y_min, y_max = np.min(points_to_plot[:, 1]), np.max(points_to_plot[:, 1])
padding = (x_max - x_min) * 0.2 # 增加边距，让标签更清晰
ax.set_xlim(x_min - padding, x_max + padding)
ax.set_ylim(y_min - padding, y_max + padding)

# --- 4. 绘制点和它们的索引 ---
# 直接绘制所有筛选后的点
ax.scatter(points_to_plot[:, 0], points_to_plot[:, 1], c='blue', alpha=0.8, s=50)

# ✨ 在每个点旁边标注它们 *原始的* 索引号
# 我们遍历 high_conf_indices，它包含了我们需要的原始索引 (0-19)
for i in high_conf_indices:
    x, y = keypoints_subset[i]
    ax.text(x + 0.008, y, str(i), fontsize=10, color='red', weight='bold')

# --- 5. 设置图形属性 ---
ax.invert_yaxis()
ax.set_title(f'Keypoint Indices 0-{NUM_POINTS_TO_CONSIDER-1} (Frame {FRAME_TO_VIEW}, Conf > {CONFIDENCE_THRESHOLD})')
ax.set_xlabel('X-coordinate')
ax.set_ylabel('Y-coordinate')
ax.grid(True)
ax.set_aspect('equal', adjustable='box')

plt.show()