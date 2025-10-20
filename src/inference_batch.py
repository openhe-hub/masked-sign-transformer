import cv2
import numpy as np
import torch
import os

from config_loader import config
from models.model import PoseTransformer as PoseTransformerV1
from models.model_v2 import PoseTransformerV2
from dataset import PoseDataset
from utils.render import draw_pose

def render_animation(sequences, subset, titles, output_path, masks=None, H=1080, W=1080, fps=2):
    """
    Renders a side-by-side animation of multiple pose sequences.
    
    Args:
        sequences (list of np.ndarray): List of pose sequences to render. 
                                       Each sequence has shape (seq_len, n_kps, features).
        titles (list of str): List of titles for each sequence.
        output_path (str): Path to save the output video.
        masks (list of np.ndarray or None): Optional list of masks. Each mask has shape (seq_len, n_kps).
        H (int): Height of the canvas for each individual render.
        W (int): Width of the canvas for each individual render.
        fps (int): Frames per second for the output video.
    """
    num_sequences = len(sequences)
    seq_len = sequences[0].shape[0]
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # The final video width will be the sum of individual widths
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (W * num_sequences, H))

    if not video_writer.isOpened():
        print("Error: Could not open video writer.")
        return

    for t in range(seq_len):
        frames = []
        for i, seq in enumerate(sequences):
            frame_mask = None
            if masks and masks[i] is not None:
                frame_mask = masks[i][t]
            # if i == 1 and t == 0:
            #     import ipdb; ipdb.set_trace()
            # The draw_pose function returns a (C, H, W) numpy array, so we need to transpose it
            pose_img = draw_pose(seq[t], subset[t], H, W, mask=frame_mask) # seq[t] is (n_kps, features)
            pose_img = pose_img.transpose(1, 2, 0) # Convert to (H, W, C) for OpenCV
            pose_img = cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR) # Convert RGB to BGR for OpenCV

            # Add title to the frame
            cv2.putText(pose_img, titles[i], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.5, (0, 0, 0), 2, cv2.LINE_AA)
            frames.append(pose_img)
        
        # Concatenate frames horizontally
        combined_frame = np.concatenate(frames, axis=1)
        
        video_writer.write(combined_frame)
        
    video_writer.release()

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
    if model_version == 'v2':
        print("Instantiating Model Version: v2 (Per-Keypoint Tokenization)")
        model = PoseTransformerV2().to(device)
    else:
        print("Instantiating Model Version: v1 (Frame-level Tokenization)")
        model = PoseTransformerV1().to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Model loaded from {checkpoint_path}")

    # Get a sample from the dataset
    dataset = PoseDataset()

    for data_index in range(index_range[0], index_range[1]):
        masked_sequence, mask, original_sequence, subset = dataset[data_index]

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

        print("Rendering animation...")
        render_animation(
            sequences=[original_sequence_np, masked_sequence_np, reconstructed_sequence_np],
            subset=subset,
            output_path=f"output/video/result_{data_index}.mp4",
            titles=['Original', 'Masked Input', 'Reconstructed'],
            masks=[None, None, mask_np]
        )
        
        print(f"Reconstructed sequence saved to output/video/result_{data_index}.mp4")

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
