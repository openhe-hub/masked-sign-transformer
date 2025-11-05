import argparse
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from config_loader import config
from datasets.dataset_v3 import PoseDatasetV3
from models.model_v4 import PoseTransformerV4
from utils.face_processing import align_face_inplace
from utils.render_conf import draw_pose

from inference_part import (  # reuse shared utilities
    PART_TO_GLOBAL_INDICES,
    assemble_full_sequence,
    create_subset_placeholder,
    denormalize_keypoints,
    upsample_predictions,
)


def render_reconstruction(
    sequence: np.ndarray,
    subset: np.ndarray,
    output_path: str,
    height: int = 1080,
    width: int = 1080,
    fps: int = 30,
) -> None:
    """
    Render the reconstructed sequence using the mm-style visualization (confidence-based colors).
    """
    seq_len = sequence.shape[0]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        raise RuntimeError(f"Failed to open video writer at {output_path}")

    for t in range(seq_len):
        pose_img = draw_pose(sequence[t], subset[t], height, width)
        pose_img = pose_img.transpose(1, 2, 0)  # (H, W, C)
        pose_img = cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR)
        video_writer.write(pose_img)

    video_writer.release()


def to_numpy_subset(subset_list: Optional[List[np.ndarray]], seq_len: int) -> np.ndarray:
    """
    Convert stored subset list to a numpy array compatible with draw_pose.
    Falls back to a synthetic placeholder when subset is missing.
    """
    if subset_list:
        stacked = np.stack([np.asarray(item) for item in subset_list], axis=0)
        return stacked
    return create_subset_placeholder(seq_len)


def inference(
    checkpoint_path: str,
    output_dir: str,
    index_begin: int,
    index_end: int,
    step: int,
    height: int,
    width: int,
    fps: int,
    seed: int,
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = config["training"]["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PoseTransformerV4().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    dataset = PoseDatasetV3()
    os.makedirs(output_dir, exist_ok=True)

    end_index = min(index_end, len(dataset))
    if index_begin >= end_index:
        raise ValueError(
            f"Invalid index range: begin={index_begin}, end={index_end}, dataset size={len(dataset)}."
        )

    for data_index in range(index_begin, end_index):
        sample_index = data_index * max(step, 1)
        if sample_index >= len(dataset):
            break

        masked_sequence, input_mask, original_sequence, meta_info = dataset[sample_index]
        norm_params = meta_info.get("norm_params")
        subset_list = meta_info.get("subset")

        parts = list(masked_sequence.keys())
        if not set(parts).issubset(PART_TO_GLOBAL_INDICES.keys()):
            unsupported = set(parts) - set(PART_TO_GLOBAL_INDICES.keys())
            raise KeyError(f"Encountered unsupported parts: {unsupported}")

        masked_on_device = {
            part: tensor.unsqueeze(0).to(device) for part, tensor in masked_sequence.items()
        }
        mask_on_device = {
            part: input_mask[part].unsqueeze(0).to(device) for part in masked_sequence
        }

        with torch.no_grad():
            predictions = model(masked_on_device)

        reconstructed_parts_np: Dict[str, np.ndarray] = {}

        for part in parts:
            mask_cpu = input_mask[part].numpy()
            pred_part = predictions[part]
            target_len = masked_on_device[part].shape[1]
            pred_part = upsample_predictions(pred_part, target_len)

            reconstructed_part = masked_on_device[part].clone()
            mask_broadcast = mask_on_device[part].unsqueeze(-1).expand_as(reconstructed_part)
            reconstructed_part[mask_broadcast] = pred_part[mask_broadcast]
            reconstructed_parts_np[part] = reconstructed_part.squeeze(0).cpu().numpy()

        reconstructed_full = assemble_full_sequence(reconstructed_parts_np)
        face_keypoints = meta_info.get("face_keypoints")
        if face_keypoints is not None:
            face_np = np.asarray(face_keypoints, dtype=np.float32)[..., :2]
            reconstructed_full[:, 18:86, :] = face_np

        if norm_params is not None:
            reconstructed_full = denormalize_keypoints(reconstructed_full, norm_params)
        align_face_inplace(reconstructed_full)

        subset = to_numpy_subset(subset_list, reconstructed_full.shape[0])

        output_path = os.path.join(output_dir, f"mm_result_{sample_index}.mp4")
        print(f"[{data_index}] Rendering to {output_path}")
        render_reconstruction(
            sequence=reconstructed_full,
            subset=subset,
            output_path=output_path,
            height=height,
            width=width,
            fps=fps,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference script (mm-style render) for PoseTransformerV4 with PoseDatasetV3."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--output-dir", type=str, default="output/video/mm", help="Directory to store rendered videos.")
    parser.add_argument("--index-begin", type=int, default=0, help="Starting sample index (inclusive).")
    parser.add_argument("--index-end", type=int, default=10, help="Ending sample index (exclusive).")
    parser.add_argument("--step", type=int, default=10, help="Sampling stride over dataset indices.")
    parser.add_argument("--height", type=int, default=1080, help="Output video height.")
    parser.add_argument("--width", type=int, default=1080, help="Output video width.")
    parser.add_argument("--fps", type=int, default=30, help="Output video frames per second.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inference(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        index_begin=args.index_begin,
        index_end=args.index_end,
        step=args.step,
        height=args.height,
        width=args.width,
        fps=args.fps,
        seed=args.seed,
    )
