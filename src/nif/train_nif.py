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

from utils.pickle_compat import load_pickle

# --- 1. Data Loading (No changes) ---
def load_all_sequences(data_dir):
    video_files = sorted(glob.glob(os.path.join(data_dir, '*_kps.pkl')))
    if not video_files:
        raise FileNotFoundError(f"No '_kps.pkl' files found in {data_dir}.")
    print(f"Found {len(video_files)} video files.")
    all_sequences = []
    for video_path in video_files:
        with open(video_path, 'rb') as fp:
            frames = load_pickle(fp)
        all_sequences.append([frame['keypoints'][18:86, :2] for frame in frames])
    return [np.array(seq) for seq in all_sequences]

def load_ref_kpts(ref_path):
    with open(ref_path, 'rb') as fp:
        ref_data = load_pickle(fp)
    return ref_data['keypoints'][18:86, :2]

# --- 2. NIF Model (No changes) ---
class PositionalEncoding(nn.Module):
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

# --- 3. Main Controller (With Checkpointing) ---

class UniversalAnimator:
    def __init__(self, ref_kpts, all_sequences, motion_dim=32, n_freqs=10, lr=5e-4, checkpoint_dir='checkpoints'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # --- NEW: Checkpoint directory ---
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.ref_mean = ref_kpts.mean(axis=0)
        ref_centered = ref_kpts - self.ref_mean
        self.ref = torch.from_numpy(ref_centered).float().to(self.device)
        self.pos_encoder = PositionalEncoding(d_input=2, n_freqs=n_freqs)
        self.ref_encoded = self.pos_encoder(self.ref)
        
        self.all_sequences = all_sequences
        self.total_frames = sum(len(seq) for seq in self.all_sequences)
        print(f"Total frames in dataset: {self.total_frames}")
        
        self.frame_map = []
        for seq_idx, seq in enumerate(self.all_sequences):
            for frame_idx in range(len(seq)):
                self.frame_map.append((seq_idx, frame_idx))

        self.motion_codes = nn.Embedding(self.total_frames, motion_dim).to(self.device)
        self.deformation_field = UniversalMotionNIF(
            self.pos_encoder.d_output, motion_dim
        ).to(self.device)

        # We initialize parameters here, but they might be overwritten by a checkpoint
        torch.nn.init.normal_(self.motion_codes.weight, std=0.01)

        self.optimizer = torch.optim.Adam([
            {'params': self.deformation_field.parameters()},
            {'params': self.motion_codes.parameters()},
        ], lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.5)

        self.start_epoch = 0 # This will be updated if we load a checkpoint

    # --- NEW: Checkpoint saving method ---
    def _save_checkpoint(self, epoch, is_best=False):
        """Saves the model state."""
        state = {
            'epoch': epoch + 1,
            'deformation_field_state_dict': self.deformation_field.state_dict(),
            'motion_codes_state_dict': self.motion_codes.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        
        # Save the latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(state, latest_path)
        print(f"Epoch {epoch+1}: Saved latest checkpoint to {latest_path}")
        
        # Optionally, save a separate file for the best model so far
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(state, best_path)
            print(f"Epoch {epoch+1}: Saved BEST model to {best_path}")

    # --- NEW: Checkpoint loading method ---
    def _load_checkpoint(self, path=None):
        """Loads the model state from a checkpoint file."""
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
        # --- MODIFIED: Load checkpoint before starting ---
        self._load_checkpoint() 

        print("Starting training on all videos to learn a universal motion space...")
        best_loss = float('inf')
        
        # Use a tqdm progress bar that starts from the correct epoch
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

            # --- MODIFIED: Save checkpoint periodically ---
            if (epoch + 1) % save_every == 0:
                is_best = loss.item() < best_loss
                if is_best:
                    best_loss = loss.item()
                self._save_checkpoint(epoch, is_best=is_best)

        # Save one last time at the end of training
        self._save_checkpoint(epochs - 1)
        print("Training finished.")

    def animate(self, driving_video_idx=0):
        # --- MODIFIED: Ensure model is loaded for inference ---
        # If not training, you might want to load the best model
        # self._load_checkpoint(os.path.join(self.checkpoint_dir, 'best_model.pth'))
        
        print(f"Generating animation driven by video {driving_video_idx}...")
        with torch.no_grad():
            start_idx = sum(len(self.all_sequences[i]) for i in range(driving_video_idx))
            end_idx = start_idx + len(self.all_sequences[driving_video_idx])
            
            driving_global_indices = torch.arange(start_idx, end_idx, device=self.device)
            driving_motion_codes = self.motion_codes(driving_global_indices)
            
            animation = torch.stack([self.forward_pass(code) for code in driving_motion_codes])
            
        return (animation.cpu().numpy() + self.ref_mean)

def post_process_align(ref_kpts, result_animation):
    """
    Aligns each frame of the result animation to the reference keypoints.
    
    Args:
        ref_kpts (np.ndarray): The reference points (shape: [68, 2]).
        result_animation (np.ndarray): The generated animation (shape: [N, 68, 2]).
        
    Returns:
        np.ndarray: The aligned animation (shape: [N, 68, 2]).
    """
    print("Performing post-processing alignment...")
    print(ref_kpts.shape)
    print(result_animation.shape)
    aligned_animation = np.zeros_like(result_animation)
    
    for i in range(result_animation.shape[0]):
        frame_to_align = result_animation[i]
        
        # procrustes(matrix_a, matrix_b) finds the optimal transform to map b -> a.
        # It returns three values: mtx1 (aligned a), mtx2 (aligned b), disparity.
        # We only need the aligned version of our generated frame.
        ref_aligned, frame_aligned, disparity = procrustes(ref_kpts, frame_to_align)
        
        aligned_animation[i] = frame_aligned
        
    print("Alignment finished.")
    return aligned_animation, ref_aligned

def post_process_align_naive(ref_kpts, result_animation):
    """
    一个更朴素的对齐方法，通过匹配中心点和包围盒尺寸来进行归一化。
    
    Args:
        ref_kpts (np.ndarray): The reference points (shape: [68, 2]).
        result_animation (np.ndarray): The generated animation (shape: [N, 68, 2]).
        
    Returns:
        np.ndarray: The aligned animation (shape: [N, 68, 2]).
        np.ndarray: The original reference keypoints for visualization.
    """
    print("Performing naive alignment using bounding box normalization...")
    
    # 1. 计算参考系的几何属性 (我们的"标尺")
    ref_min = ref_kpts.min(axis=0)
    ref_max = ref_kpts.max(axis=0)
    ref_center = (ref_min + ref_max) / 2.0
    ref_size = ref_max - ref_min  # [width, height]
    
    # 防止除以零
    ref_size[ref_size < 1e-6] = 1e-6

    aligned_animation = np.zeros_like(result_animation)
    
    for i in range(result_animation.shape[0]):
        frame = result_animation[i]
        
        # 2. 计算当前帧的几何属性
        frame_min = frame.min(axis=0)
        frame_max = frame.max(axis=0)
        frame_center = (frame_min + frame_max) / 2.0
        frame_size = frame_max - frame_min # [width, height]
        
        # 防止除以零
        frame_size[frame_size < 1e-6] = 1e-6

        # 3. 执行归一化
        # 首先，将当前帧中心移到原点
        centered_frame = frame - frame_center
        
        # 其次，根据参考系的尺寸进行缩放
        # 计算缩放比例 (x和y轴分开算)
        scale_factors = ref_size / frame_size # [scale_x, scale_y]
        # 应用缩放
        scaled_frame = centered_frame * scale_factors
        
        # 最后，将缩放后的帧移动到参考系的中心
        aligned_frame = scaled_frame + ref_center
        
        aligned_animation[i] = aligned_frame
        
    print("Naive alignment finished.")
    # 我们返回原始的ref_kpts，因为这个方法不会改变它
    # procrustes会返回一个中心化的ref, 所以之前有ref_aligned
    return aligned_animation, ref_kpts

# --- 4. Visualization (No changes) ---
def visualize_animation_2d(ref, result_seq, driving_seq=None):
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

# --- 5. Main Execution ---
if __name__ == '__main__':
    DATA_DIR = 'data/chatsign_200_wo_retarget/'
    REF_PATH = 'assets/pose/ref.pkl'
    CHECKPOINT_DIR = 'checkpoints/nif_checkpoints' # Define checkpoint directory
    
    all_sequences = load_all_sequences(DATA_DIR)
    ref_kpts = load_ref_kpts(REF_PATH)
    
    animator = UniversalAnimator(
        ref_kpts, 
        all_sequences, 
        motion_dim=32, 
        lr=1e-3,
        checkpoint_dir=CHECKPOINT_DIR
    )
    
    # The train function will now automatically try to resume
    # animator.train(epochs=8000, batch_size=32, save_every=500) # Save every 500 epochs
    
    # --- Animation Generation ---
    DRIVING_VIDEO_ID = 10 
    
    animator._load_checkpoint(os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
    
    result_animation = animator.animate(driving_video_idx=DRIVING_VIDEO_ID)

    aligned_animation, ref_aligned = post_process_align_naive(ref_kpts, result_animation)
    
    # --- Visualization ---
    driving_example_seq = all_sequences[DRIVING_VIDEO_ID]
    aligned_seq, ref_aligned = post_process_align_naive(ref_kpts, driving_example_seq)
    visualize_animation_2d(ref_aligned, aligned_seq, driving_seq=driving_example_seq)
    
