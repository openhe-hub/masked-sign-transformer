import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# ==================== ✨ 1. 在这里指定您要可视化的节点索引 ====================
#
# 请将下面列表中的示例索引替换为您真正关心的 50 个节点的索引。
# 例如，手部和上半身的索引。
#
# 这是一个包含 50 个随机数字的占位符，请务必修改它！
INDICES_TO_PLOT = [
    # 上半身 (示例8个)
    0, 5, 6, 7, 8 , 9,  10,
    # # 左手 (21个)
    91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,107, 108, 109, 110, 111, 
    # # 右手 (21个)
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132
]
# ==============================================================================


# --- 2. 加载并准备数据 ---

file_path = 'data/Asl-pose/0a0px34qrq.pkl' 
try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
except FileNotFoundError:
    print(f"错误：找不到文件 '{file_path}'。请检查路径是否正确。")
    exit()

# 提取完整的关键点和置信度分数
keypoints_data_full = np.squeeze(np.array(data['keypoints']))
scores_data_full = np.squeeze(np.array(data['scores']))

# ✨ 从完整数据中只选择我们指定的索引
keypoints_data = keypoints_data_full[:, INDICES_TO_PLOT, :]
scores_data = scores_data_full[:, INDICES_TO_PLOT]

# 校验数据格式
if not (keypoints_data.ndim == 3 and keypoints_data.shape[2] == 2):
    print("数据格式不匹配或不正确。")
    exit()

n_frames = keypoints_data.shape[0]
n_keypoints_to_plot = keypoints_data.shape[1]
print(f"文件加载成功，共找到 {n_frames} 帧数据。")
print(f"每帧将可视化指定的 {n_keypoints_to_plot} / {keypoints_data_full.shape[1]} 个关键点。")


# --- 3. 设置绘图和动画 ---

fig, ax = plt.subplots(figsize=(8, 8))

# ✨ 预计算坐标范围 (只基于我们关注的点，这样可以自动缩放到合适大小)
all_x = keypoints_data[:, :, 0]
all_y = keypoints_data[:, :, 1]
x_min, x_max = np.min(all_x), np.max(all_x)
y_min, y_max = np.min(all_y), np.max(all_y)
padding = (x_max - x_min) * 0.1
ax.set_xlim(x_min - padding, x_max + padding)
ax.set_ylim(y_min - padding, y_max + padding)

ax.invert_yaxis()
ax.set_title('Specific Pose Keypoints Animation (Alpha indicates score)')
ax.set_xlabel('X-coordinate')
ax.set_ylabel('Y-coordinate')
ax.grid(True)
ax.set_aspect('equal', adjustable='box')

scatter = ax.scatter([], [])
frame_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, va='top', fontsize=12)


# --- 4. 定义动画更新函数 (此部分无需修改) ---

# 定义一个固定的基准颜色 (RGB格式, 范围0-1)
BASE_COLOR_RGB = np.array([0.12, 0.46, 0.70]) 

def update(frame_index):
    """每一帧都会调用此函数来更新图形"""
    # ✨ 注意：这里的 keypoints_data 和 scores_data 已经是被筛选过的子集了
    keypoints = keypoints_data[frame_index]
    scores = scores_data[frame_index]
    
    # 创建一个 RGBA 颜色数组
    num_points = keypoints.shape[0] # 这里会自动变成 50
    colors_rgba = np.zeros((num_points, 4))
    colors_rgba[:, :3] = BASE_COLOR_RGB
    
    clipped_scores = np.clip(scores, 0, 1)
    colors_rgba[:, 3] = clipped_scores

    # 更新散点图的位置
    scatter.set_offsets(keypoints)
    
    # 更新每个点的颜色和透明度
    scatter.set_facecolors(colors_rgba)
    
    # 更新帧数文本
    frame_text.set_text(f'Frame: {frame_index + 1}/{n_frames}')
    
    return scatter, frame_text


# --- 5. 创建并保存动画 ---

ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)

output_filename = 'output/specific_pose_animation.gif'
print(f"正在生成动画，请稍候... 将保存为 '{output_filename}'")
try:
    # 确保输出目录存在
    import os
    os.makedirs('output', exist_ok=True)
    ani.save(output_filename, writer='pillow', fps=20)
    print(f"动画已成功保存为 '{output_filename}'！")
except Exception as e:
    print(f"保存失败: {e}")
    print("请确保您已经安装了 'Pillow' 库 (pip install Pillow)")