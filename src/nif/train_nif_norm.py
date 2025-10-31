import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os
import glob
import random
from scipy.spatial import procrustes

# --- 1. 数据加载 & 预处理模块 (核心修改) ---

def load_all_sequences(data_dir):
    # ... (此函数无变化) ...
    video_files = sorted(glob.glob(os.path.join(data_dir, '*_kps.pkl')))
    if not video_files: raise FileNotFoundError(f"在目录 {data_dir} 中没有找到任何 '_kps.pkl' 文件。")
    print(f"找到了 {len(video_files)} 个视频文件。")
    all_sequences = [[frame['keypoints'][18:86, :2] for frame in pickle.load(open(video_path, 'rb'))] for video_path in video_files]
    return [np.array(seq) for seq in all_sequences]

def load_ref_kpts(ref_path):
    # ... (此函数无变化) ...
    with open(ref_path, 'rb') as fp:
        ref_data = pickle.load(fp)
    return ref_data['keypoints'][18:86, :2]

# --- NEW: Scale normalization function ---
def normalize_sequences_by_scale(ref_kpts, all_sequences):
    """
    Normalizes the scale of all sequences based on the reference keypoints.
    The scale is determined by the distance between the outer eye corners.
    """
    print("Normalizing scale of all training data...")
    # dlib 68 points: left eye corner is 36, right eye corner is 45
    LEFT_EYE_CORNER_IDX = 36
    RIGHT_EYE_CORNER_IDX = 45
    
    # 1. Calculate the reference scale
    ref_left_eye = ref_kpts[LEFT_EYE_CORNER_IDX]
    ref_right_eye = ref_kpts[RIGHT_EYE_CORNER_IDX]
    scale_ref = np.linalg.norm(ref_left_eye - ref_right_eye)
    
    normalized_sequences = []
    for seq in tqdm(all_sequences, desc="Normalizing sequences"):
        new_seq = np.zeros_like(seq)
        for i in range(seq.shape[0]):
            frame = seq[i]
            
            # 2. Calculate the scale of the current frame
            frame_left_eye = frame[LEFT_EYE_CORNER_IDX]
            frame_right_eye = frame[RIGHT_EYE_CORNER_IDX]
            scale_frame = np.linalg.norm(frame_left_eye - frame_right_eye)
            
            if scale_frame < 1e-6: # Avoid division by zero
                scale_frame = scale_ref
            
            # 3. Calculate the normalization factor
            norm_factor = scale_ref / scale_frame
            
            # 4. Center the frame, scale it, then move it back
            frame_center = frame.mean(axis=0)
            centered_frame = frame - frame_center
            scaled_frame = centered_frame * norm_factor
            new_frame = scaled_frame + frame_center
            
            new_seq[i] = new_frame
        normalized_sequences.append(new_seq)
        
    print("Normalization finished.")
    return normalized_sequences


# --- 2. NIF 模型定义 (无变化) ---
class PositionalEncoding(nn.Module):
    # ... (代码与之前相同) ...
    def __init__(self, d_input, n_freqs):
        super().__init__()
        self.d_output = d_input * (1 + 2 * n_freqs)
        self.embed_fns = [lambda x: x]
        freq_bands = 2.**torch.linspace(0., n_freqs - 1, n_freqs)
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))
    def forward(self, x):
        return torch.cat([fn(x) for fn in self.embed_fns], dim=-1)

class UniversalMotionNIF(nn.Module):
    # ... (代码与之前相同) ...
    def __init__(self, pos_embed_dim, motion_dim, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(pos_embed_dim + motion_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        torch.nn.init.constant_(self.model[-1].weight, 0)
        torch.nn.init.constant_(self.model[-1].bias, 0)
    def forward(self, pos_embed, motion_code):
        motion_code_exp = motion_code.expand(pos_embed.shape[0], -1)
        x = torch.cat([pos_embed, motion_code_exp], dim=-1)
        return self.model(x)

# --- 3. 主控制器 (无变化) ---
class UniversalAnimator:
    # ... (所有代码与之前完全相同) ...
    def __init__(self, ref_kpts, all_sequences, motion_dim=32, n_freqs=10, lr=5e-4, checkpoint_dir='checkpoints'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.ref_mean = ref_kpts.mean(axis=0)
        ref_centered = ref_kpts - self.ref_mean
        self.ref = torch.from_numpy(ref_centered).float().to(self.device)
        self.pos_encoder = PositionalEncoding(d_input=2, n_freqs=n_freqs)
        self.ref_encoded = self.pos_encoder(self.ref)
        self.all_sequences = all_sequences
        self.total_frames = sum(len(seq) for seq in self.all_sequences)
        print(f"数据集中总帧数为: {self.total_frames}")
        self.frame_map = []
        for seq_idx, seq in enumerate(self.all_sequences):
            for frame_idx in range(len(seq)):
                self.frame_map.append((seq_idx, frame_idx))
        self.motion_codes = nn.Embedding(self.total_frames, motion_dim).to(self.device)
        torch.nn.init.normal_(self.motion_codes.weight, std=0.01)
        self.deformation_field = UniversalMotionNIF(
            self.pos_encoder.d_output, motion_dim
        ).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.deformation_field.parameters()},
            {'params': self.motion_codes.parameters()},
        ], lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.5)
        self.start_epoch = 0
    def _save_checkpoint(self, epoch, is_best=False):
        state = {'epoch': epoch + 1,'deformation_field_state_dict': self.deformation_field.state_dict(),'motion_codes_state_dict': self.motion_codes.state_dict(),'optimizer_state_dict': self.optimizer.state_dict(),'scheduler_state_dict': self.scheduler.state_dict(),}
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(state, latest_path)
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(state, best_path)
    def _load_checkpoint(self, path=None):
        if path is None:
            path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        if not os.path.exists(path):
            print("No checkpoint found. Starting training from scratch.")
            return
        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        self.deformation_field.load_state_dict(checkpoint['deformation_field_state_dict'])
        self.motion_codes.load_state_dict(checkpoint['motion_codes_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {self.start_epoch}")
    def forward_pass(self, motion_code):
        displacement = self.deformation_field(self.ref_encoded, motion_code)
        return self.ref + displacement
    def train(self, epochs=8000, batch_size=16, save_every=100):
        self._load_checkpoint() 
        best_loss = float('inf')
        pbar = tqdm(range(self.start_epoch, epochs), initial=self.start_epoch, total=epochs)
        for epoch in pbar:
            self.optimizer.zero_grad()
            global_frame_indices = torch.randint(0, self.total_frames, (batch_size,), device=self.device)
            target_kpts_list = [self.all_sequences[seq_idx][frame_idx] for seq_idx, frame_idx in [self.frame_map[i] for i in global_frame_indices.cpu().numpy()]]
            target_kpts_batch = torch.from_numpy(np.array(target_kpts_list)).float().to(self.device)
            target_kpts_centered = target_kpts_batch - torch.from_numpy(self.ref_mean).float().to(self.device)
            z_motion = self.motion_codes(global_frame_indices)
            reconstructed_kpts = torch.stack([self.forward_pass(z_motion[i]) for i in range(batch_size)])
            loss = nn.MSELoss()(reconstructed_kpts, target_kpts_centered)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            pbar.set_description(f"Loss: {loss.item():.6f}")
            if (epoch + 1) % save_every == 0:
                is_best = loss.item() < best_loss
                if is_best: best_loss = loss.item()
                self._save_checkpoint(epoch, is_best=is_best)
        self._save_checkpoint(epochs - 1)
        print("Training finished.")
    def animate(self, driving_video_idx=0):
        print(f"Generating animation driven by video {driving_video_idx}...")
        with torch.no_grad():
            start_idx = sum(len(self.all_sequences[i]) for i in range(driving_video_idx))
            end_idx = start_idx + len(self.all_sequences[driving_video_idx])
            driving_global_indices = torch.arange(start_idx, end_idx, device=self.device)
            driving_motion_codes = self.motion_codes(driving_global_indices)
            animation = torch.stack([self.forward_pass(code) for code in driving_motion_codes])
        return (animation.cpu().numpy() + self.ref_mean)

# --- 4. Post-processing (无变化) ---
def post_process_align(ref_kpts, result_animation):
    print("Performing post-processing alignment (with position correction)...")
    aligned_animation = np.zeros_like(result_animation)
    ref_center = ref_kpts.mean(axis=0)
    for i in range(result_animation.shape[0]):
        frame_to_align = result_animation[i]
        mtx1, mtx2, disparity = procrustes(ref_kpts, frame_to_align)
        frame_aligned_corrected = mtx2 + ref_center
        aligned_animation[i] = frame_aligned_corrected
    print("Alignment finished.")
    return aligned_animation

# --- 5. 可视化 (无变化) ---
def visualize_animation_2d(ref, result_seq, driving_seq=None):
    # ... (代码与之前相同) ...
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(len(result_seq)):
        ax.cla()
        ax.scatter(ref[:, 0], ref[:, 1], c='gray', marker='.', s=50, label='Reference Face')
        if driving_seq is not None and i < len(driving_seq):
             ax.scatter(driving_seq[i][:, 0], driving_seq[i][:, 1], c='blue', marker='o', s=70, label='Example Driving Video')
        ax.scatter(result_seq[i][:, 0], result_seq[i][:, 1], c='red', marker='x', s=100, label='Animated Result')
        ax.legend()
        ax.set_aspect('equal', 'box'); ax.invert_yaxis()
        plt.title(f"Frame {i+1}/{len(result_seq)}"); plt.grid(True)
        plt.pause(0.05)
    plt.show()

# --- 6. 主流程 (已修改) ---
if __name__ == '__main__':
    DATA_DIR = 'data/chatsign_200_wo_retarget/'
    REF_PATH = 'assets/pose/ref.pkl'
    CHECKPOINT_DIR = 'checkpoints/nif_checkpoints_normalized' # Use a new checkpoint dir for the new data
    
    # 1. Load data
    raw_sequences = load_all_sequences(DATA_DIR)
    ref_kpts = load_ref_kpts(REF_PATH)
    
    # 2. --- NEW: Apply scale normalization ---
    normalized_sequences = normalize_sequences_by_scale(ref_kpts, raw_sequences)
    
    # 3. Instantiate animator WITH NORMALIZED DATA
    animator = UniversalAnimator(
        ref_kpts, 
        normalized_sequences, # Pass the normalized data here
        motion_dim=32, 
        lr=1e-3, 
        checkpoint_dir=CHECKPOINT_DIR
    )
    
    # 4. Train (if needed)
    animator.train(epochs=8000, batch_size=32, save_every=500)
    
    # 5. Load model and generate animation
    best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    if os.path.exists(best_model_path):
        animator._load_checkpoint(best_model_path)
    else:
        animator._load_checkpoint() 

    DRIVING_VIDEO_ID = 0
    result_animation = animator.animate(driving_video_idx=DRIVING_VIDEO_ID)
    
    # 6. Apply post-processing
    # This step is now more of a "fine-tuning" alignment for rotation/translation
    aligned_animation = post_process_align(ref_kpts, result_animation)
    
    # 7. Visualize
    # Use the ORIGINAL (raw) driving video for visual comparison
    driving_example_seq = raw_sequences[DRIVING_VIDEO_ID] 
    visualize_animation_2d(ref_kpts, aligned_animation, driving_seq=driving_example_seq)