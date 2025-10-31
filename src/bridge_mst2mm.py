import cv2
import torch
import pickle
from torch.utils.data import DataLoader
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import pil_to_tensor, resize, center_crop
import numpy as np
import argparse
import math

from config_loader import config
from models.model import PoseTransformer as PoseTransformerV1
from models.model_v2 import PoseTransformerV2
from models.model_v3 import PoseTransformerV3
from datasets.dataset import PoseDataset as PoseDatasetV1
from datasets.dataset_v3 import PoseDatasetV3
from datasets.dataset_continuos import PoseDatasetContinuos
from utils.render_conf import draw_pose
from utils.constants import ASPECT_RATIO

def preprocess_ref(image_path, resolution=576):
    image_pixels = pil_loader(image_path)
    image_pixels = pil_to_tensor(image_pixels) # (c, h, w)
    h, w = image_pixels.shape[-2:]
    ############################ compute target h/w according to original aspect ratio ###############################
    if h>w:
        w_target, h_target = resolution, int(resolution / ASPECT_RATIO // 64) * 64
    else:
        w_target, h_target = int(resolution / ASPECT_RATIO // 64) * 64, resolution
    h_w_ratio = float(h) / float(w)
    if h_w_ratio < h_target / w_target:
        h_resize, w_resize = h_target, math.ceil(h_target / h_w_ratio)
    else:
        h_resize, w_resize = math.ceil(w_target * h_w_ratio), w_target
    image_pixels = resize(image_pixels, [h_resize, w_resize], antialias=None)
    image_pixels = center_crop(image_pixels, [h_target, w_target])
    image_pixels = image_pixels.permute((1, 2, 0)).numpy()
    return image_pixels

def render_animation(seq, subset, ref_image):
    frames = []
    height, width, _ = ref_image.shape
    for i, seq in enumerate(seq):
        # The draw_pose function returns a (C, H, W) numpy array, so we need to transpose it
        pose_img = draw_pose(seq, subset, height, width) # seq[t] is (n_kps, features)
        # Add title to the frame
        frames.append(pose_img)
    return np.stack(frames)

def inference(checkpoint_path, output_path="reconstructed_sequence.mp4", index_range=[0,100]):
    """
    Loads a model checkpoint, performs inference on a sample from the dataset,
    and saves the reconstructed animation.
    """
    # Set a fixed seed for reproducibility
    np.random.seed(0)
    
    # Get config
    device = config['training']['device']
    n_kps = config['data']['n_kps']
    features_per_kp = 2 # config['data']['features_per_kp']
    seq_len = config['data']['sequence_length']
    model_version = config['model'].get('version', 'v1')

    # Load model
    if model_version == 'v5':
        print("Instantiating Model Version: v5 (Asymmetric Encoder-Decoder)")
        model = PoseTransformerV3().to(device)
        dataset = PoseDatasetV3()
    elif model_version == 'v4':
        print("Instantiating Model Version: v4 (Per-Keypoint Tokenization)")
        model = PoseTransformerV2().to(device)
        dataset = PoseDatasetV1()
    else:
        print("Instantiating Model Version: v1-3 (Frame-level Tokenization)")
        model = PoseTransformerV1().to(device)
        dataset = PoseDatasetContinuos()

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Model loaded from {checkpoint_path}")

    all_pose_pixels = []
    all_video_info = []

    for data_index in range(index_range[0], index_range[1]):
        if data_index >= len(dataset): break
        masked_sequence, mask, original_sequence, subset, video_info = dataset[data_index]

        # Add batch dimension and move to device
        masked_sequence = masked_sequence.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)

        with torch.no_grad():
            # Get model prediction for the full sequence
            predictions = model(masked_sequence, mask)

        # The model outputs the reconstructed sequence.
        # For visualization, we want to fill in the masked parts of the input with the predictions.
        
        # Reshape mask to match sequence dimensions for easier processing
        expanded_mask = mask.unsqueeze(-1).expand(-1, -1, -1, features_per_kp)
        expanded_mask = expanded_mask.reshape(
            predictions.shape[0], predictions.shape[1], n_kps * features_per_kp
        )

        # Create reconstructed sequence
        reconstructed_sequence = masked_sequence.clone()
        reconstructed_sequence[expanded_mask] = predictions[expanded_mask]

        # Move to CPU and convert to numpy for visualization
        reconstructed_sequence_np = reconstructed_sequence.squeeze(0).cpu().numpy()
        original_sequence_np = original_sequence.numpy()
        masked_sequence_np = masked_sequence.squeeze(0).cpu().numpy()

        # Reshape the data from (seq_len, n_kps * features) to (seq_len, n_kps, features)
        reconstructed_sequence_np = reconstructed_sequence_np.reshape(seq_len, n_kps, features_per_kp)
        original_sequence_np = original_sequence_np.reshape(seq_len, n_kps, features_per_kp)
        masked_sequence_np = masked_sequence_np.reshape(seq_len, n_kps, features_per_kp)

        mask_np = mask.squeeze(0).cpu().numpy()

        # Set masked keypoints in the input to a position where they are not visible
        masked_sequence_np[mask_np] = -1

        # bridge part

        # 1. get ref img pixels
        ref_img_path = config['data']['ref_img_path']
        image_pixels = preprocess_ref(ref_img_path)
        # 2. get ref img pose
        image_pose = None
        with open(config['data']['ref_img_pose'], 'rb') as fp:
            image_pose = pickle.load(fp)
        # 3. get ref video pose
        video_poses = render_animation(reconstructed_sequence_np, subset[0], image_pixels)
        pose_pixels = np.concatenate([np.expand_dims(image_pose, 0), video_poses])
        # 4. trans
        image_pixels = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))
        pose_pixels, image_pixels = torch.from_numpy(pose_pixels.copy()) / 127.5 - 1, torch.from_numpy(image_pixels) / 127.5 - 1
        all_pose_pixels.append(pose_pixels)
        all_video_info.append(video_info)

    # 5. save
    print(torch.cat(all_pose_pixels, dim=0).shape)
    print(image_pixels.shape)
    export_dict = {
        'pose_pixels': torch.cat(all_pose_pixels, dim=0),
        'image_pixels': image_pixels,
        'video_info': all_video_info,
    }
    print(all_video_info)
    with open(f'output/bridge/seg_all3.pkl', 'wb') as fp:
        pickle.dump(export_dict, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for PoseTransformer")
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to the model checkpoint (e.g., checkpoints/v2/pose_transformer_mae_epoch_100.pth)')
    parser.add_argument('--output', type=str, default='output/reconstruction.mp4', 
                        help='Path to save the output video')
    parser.add_argument('--index_begin', type=int, default=0, 
                        help='Index of the data sample to use for inference')
    parser.add_argument('--index_end', type=int, default=10, 
                        help='Index of the data sample to use for inference')
    args = parser.parse_args()

    inference(args.checkpoint, args.output, [args.index_begin, args.index_end])
