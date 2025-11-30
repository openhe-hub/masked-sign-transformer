import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# --- 1. 加载并准备数据 ---

# 请确保将路径替换为您的文件的实际路径
file_path = 'data/Asl-pose/0a0px34qrq.pkl' 
try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
except FileNotFoundError:
    print(f"错误：找不到文件 '{file_path}'。请检查路径是否正确。")
    exit()

# 提取关键点和置信度分数
keypoints_data_raw = np.array(data['keypoints'])
scores_data_raw = np.array(data['scores'])

# 使用 np.squeeze 移除所有大小为 1 的维度
keypoints_data = np.squeeze(keypoints_data_raw)
scores_data = np.squeeze(scores_data_raw)

# 校验数据格式
if not (keypoints_data.ndim == 3 and keypoints_data.shape[2] == 2 and
        scores_data.ndim == 2 and
        keypoints_data.shape[0] == scores_data.shape[0] and
        keypoints_data.shape[1] == scores_data.shape[1]):
    print("数据格式不匹配或不正确。")
    print("期望格式: keypoints (n_frames, n_points, 2), scores (n_frames, n_points)")
    exit()

n_frames = keypoints_data.shape[0]
n_keypoints = keypoints_data.shape[1]
print(f"文件加载成功，共找到 {n_frames} 帧数据，每帧 {n_keypoints} 个关键点。")


# --- 2. 设置绘图和动画 ---

fig, ax = plt.subplots(figsize=(8, 8))

# 预计算坐标范围
all_x = keypoints_data[:, :, 0]
all_y = keypoints_data[:, :, 1]
x_min, x_max = np.min(all_x), np.max(all_x)
y_min, y_max = np.min(all_y), np.max(all_y)
padding = (x_max - x_min) * 0.1
ax.set_xlim(x_min - padding, x_max + padding)
ax.set_ylim(y_min - padding, y_max + padding)

ax.invert_yaxis()
ax.set_title('Pose Keypoints Animation (Alpha indicates score)')
ax.set_xlabel('X-coordinate')
ax.set_ylabel('Y-coordinate')
ax.grid(True)
ax.set_aspect('equal', adjustable='box')

scatter = ax.scatter([], [])
frame_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, va='top', fontsize=12)


# --- 3. 定义动画更新函数 ---

# 定义一个固定的基准颜色 (RGB格式, 范围0-1)
BASE_COLOR_RGB = np.array([0.12, 0.46, 0.70]) 

def update(frame_index):
    """每一帧都会调用此函数来更新图形"""
    keypoints = keypoints_data[frame_index]
    scores = scores_data[frame_index]
    
    # 创建一个 RGBA 颜色数组
    num_points = keypoints.shape[0]
    colors_rgba = np.zeros((num_points, 4))
    colors_rgba[:, :3] = BASE_COLOR_RGB
    
    # ✨ FIX: 使用 np.clip 确保 score 值在 [0, 1] 范围内
    # This prevents the "ValueError: RGBA values should be within 0-1 range"
    clipped_scores = np.clip(scores, 0, 1)
    colors_rgba[:, 3] = clipped_scores

    # 更新散点图的位置
    scatter.set_offsets(keypoints)
    
    # 更新每个点的颜色和透明度
    scatter.set_facecolors(colors_rgba)
    
    # 更新帧数文本
    frame_text.set_text(f'Frame: {frame_index + 1}/{n_frames}')
    
    return scatter, frame_text


# --- 4. 创建并保存动画 ---

ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)

output_filename = 'output/pose_animation_with_alpha.gif'
print(f"正在生成动画，请稍候... 将保存为 '{output_filename}'")
try:
    ani.save(output_filename, writer='pillow', fps=20)
    print(f"动画已成功保存为 '{output_filename}'！")
except Exception as e:
    print(f"保存失败: {e}")
    print("请确保您已经安装了 'Pillow' 库 (pip install Pillow)")

# plt.show()