import os
import pickle
import random

# --- 参数设置 ---

# 创建用于存储虚拟数据的文件夹名称
output_folder = 'data/dummy_data'

# 要生成的 .pkl 文件数量
num_files = 5

# 每个 .pkl 文件中的帧数
frames_per_file = 50

# 每一帧中的关键点数量
keypoints_per_frame = 134

# 关键点 x 坐标的范围
x_range = (0, 1920)

# 关键点 y 坐标的范围
y_range = (0, 1080)

# --- 脚本主体 ---

def generate_dummy_data():
    """
    生成并保存虚拟数据集。
    """
    # 如果文件夹不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"文件夹 '{output_folder}' 已创建。")

    # 循环创建指定数量的 .pkl 文件
    for i in range(num_files):
        all_frames = []
        
        # 循环创建指定数量的帧
        for _ in range(frames_per_file):
            current_frame_keypoints = []
            
            # 循环创建指定数量的关键点
            for _ in range(keypoints_per_frame):
                # 生成随机的 x, y 和 confidence 值
                x = random.randint(x_range[0], x_range[1])
                y = random.randint(y_range[0], y_range[1])
                confidence = random.uniform(0.0, 1.0)
                
                # 将关键点数据添加为 [x, y, confidence] 格式的列表
                current_frame_keypoints.append([x, y, confidence])
            
            # 将当前帧的所有关键点添加到帧列表中
            all_frames.append(current_frame_keypoints)
        
        # 定义 .pkl 文件的路径和名称
        file_path = os.path.join(output_folder, f'data_{i+1}.pkl')
        
        # 将所有帧的数据写入 .pkl 文件
        with open(file_path, 'wb') as f:
            pickle.dump(all_frames, f)
            
        print(f"已生成文件: {file_path}")

if __name__ == '__main__':
    generate_dummy_data()
    print("\n虚拟数据集生成完毕！")