import argparse
import os
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config_loader import config
from datasets.dataset_v3 import PoseDatasetV3
from models.model_v4 import PoseTransformerV4
from utils.face_processing import align_face_inplace
from utils.render import draw_pose


# Mapping from part name to the indices in the flattened 128-keypoint layout expected by draw_pose.
PART_TO_GLOBAL_INDICES: Dict[str, np.ndarray] = {
    "body": np.arange(0, 18, dtype=np.int64),
    "right": np.arange(86, 107, dtype=np.int64),
    "left": np.arange(107, 128, dtype=np.int64),
}

# Total keypoints and features per keypoint in the visualization tensor.
TOTAL_KEYPOINTS = int(max(idx.max() for idx in PART_TO_GLOBAL_INDICES.values()) + 1)
FEATURES_PER_KEYPOINT = 2

BODY_SLICE = slice(0, 18)
FACE_SLICE = slice(18, 86)
LEFT_HAND_SLICE = slice(86, 107)
RIGHT_HAND_SLICE = slice(107, 128)


def denormalize_keypoints(
    normalized: np.ndarray,
    norm_params: Sequence[dict],
) -> np.ndarray:
    """
    Restore per-frame keypoints from the normalized coordinate space back to the original pixel space.

    Args:
        normalized: Array shaped (T, 128, 2) containing normalized xy coordinates.
        norm_params: Sequence of length T where each item stores the normalization parameters
                     produced during preprocessing (body_scale, body_offset, *_center).

    Returns:
        Array shaped (T, 128, 2) with denormalized coordinates. Frames with invalid parameters
        fall back to the normalized values.
    """
    if normalized.ndim != 3 or normalized.shape[1:] != (TOTAL_KEYPOINTS, FEATURES_PER_KEYPOINT):
        raise ValueError(
            f"'normalized' must be of shape (T, {TOTAL_KEYPOINTS}, {FEATURES_PER_KEYPOINT}), "
            f"but received {normalized.shape}."
        )

    if len(norm_params) != normalized.shape[0]:
        raise ValueError(
            f"Length mismatch between data ({normalized.shape[0]}) and norm_params ({len(norm_params)})."
        )

    normalized = normalized.astype(np.float32, copy=False)
    denorm = normalized.copy()

    for frame_idx, params in enumerate(norm_params):
        if not isinstance(params, dict):
            continue

        body_scale = float(params.get("body_scale", 0.0))
        body_offset = np.asarray(params.get("body_offset", [0.0, 0.0]), dtype=np.float32)

        if body_scale <= 0 or body_offset.shape != (2,):
            # Unable to denormalize without valid body parameters; keep normalized values.
            continue

        # Denormalize body keypoints.
        norm_body = normalized[frame_idx, BODY_SLICE, :]
        denorm[frame_idx, BODY_SLICE, :] = ((norm_body / 2.0) + 0.5) * body_scale + body_offset

        part_mappings = {
            "face_center": FACE_SLICE,
            "left_hand_center": LEFT_HAND_SLICE,
            "right_hand_center": RIGHT_HAND_SLICE,
        }

        for center_key, indices in part_mappings.items():
            center = params.get(center_key)
            if center is None:
                continue
            center = np.asarray(center, dtype=np.float32)
            if center.shape != (2,):
                continue

            norm_part = normalized[frame_idx, indices, :]
            denorm[frame_idx, indices, :] = (norm_part * body_scale) + center

    return denorm


def upsample_predictions(prediction: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Upsample model predictions along the temporal axis so they match the original sequence length.
    The model applies temporal pooling, so predictions may be shorter than the input sequence.
    """
    batch_size, pred_len, num_kps, channels = prediction.shape

    if pred_len == target_len:
        return prediction

    if pred_len == 1:
        return prediction.repeat(1, target_len, 1, 1)

    flattened = prediction.permute(0, 2, 3, 1).reshape(batch_size, num_kps * channels, pred_len)
    upsampled = F.interpolate(flattened, size=target_len, mode="linear", align_corners=False)
    upsampled = upsampled.view(batch_size, num_kps, channels, target_len).permute(0, 3, 1, 2)
    return upsampled


def assemble_full_sequence(part_sequences: Dict[str, np.ndarray], fill_value: float = -1.0) -> np.ndarray:
    """
    Merge per-part sequences back into the 128-keypoint layout used by the original renderer.
    Missing keypoints (e.g., face landmarks) are filled with `fill_value`.
    """
    if not part_sequences:
        raise ValueError("part_sequences is empty; nothing to assemble.")

    seq_len = next(iter(part_sequences.values())).shape[0]
    full_sequence = np.full(
        (seq_len, TOTAL_KEYPOINTS, FEATURES_PER_KEYPOINT),
        fill_value,
        dtype=np.float32,
    )

    for part_name, global_indices in PART_TO_GLOBAL_INDICES.items():
        if part_name not in part_sequences:
            continue

        part_seq = part_sequences[part_name]
        if part_seq.shape[1] != len(global_indices):
            raise ValueError(
                f"Part '{part_name}' expected {len(global_indices)} keypoints "
                f"but received {part_seq.shape[1]}."
            )
        full_sequence[:, global_indices, :] = part_seq

    return full_sequence


def assemble_full_mask(part_masks: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Combine per-part boolean masks into a single mask over the 128-keypoint layout.
    """
    if not part_masks:
        raise ValueError("part_masks is empty; nothing to assemble.")

    seq_len = next(iter(part_masks.values())).shape[0]
    full_mask = np.zeros((seq_len, TOTAL_KEYPOINTS), dtype=bool)

    for part_name, global_indices in PART_TO_GLOBAL_INDICES.items():
        if part_name not in part_masks:
            continue

        part_mask = part_masks[part_name]
        if part_mask.shape[1] != len(global_indices):
            raise ValueError(
                f"Part '{part_name}' expected {len(global_indices)} keypoints "
                f"but received mask with shape {part_mask.shape}."
            )
        full_mask[:, global_indices] = part_mask

    return full_mask


def create_subset_placeholder(seq_len: int) -> np.ndarray:
    """
    Build a placeholder subset array compatible with draw_pose's expectations.
    The original subset arrays come from OpenPose and index the 18 body keypoints.
    """
    subset = -np.ones((seq_len, 1, 20), dtype=np.int32)
    subset[:, 0, :18] = np.arange(18, dtype=np.int32)
    return subset


def render_animation(
    sequences: List[np.ndarray],
    subset: np.ndarray,
    titles: List[str],
    output_path: str,
    masks: Optional[List[Optional[np.ndarray]]] = None,
    height: int = 1080,
    width: int = 1080,
    fps: int = 2,
) -> None:
    """
    Render side-by-side pose sequences using the existing draw_pose helper.
    """
    if not sequences:
        raise ValueError("No sequences provided for rendering.")

    num_sequences = len(sequences)
    seq_len = sequences[0].shape[0]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width * num_sequences, height))

    if not video_writer.isOpened():
        raise RuntimeError(f"Failed to open video writer at {output_path}")

    for t in range(seq_len):
        frames = []
        for idx, sequence in enumerate(sequences):
            frame_mask = None
            if idx != 0:
                frame_mask = masks[idx][t]
            pose_img = draw_pose(sequence[t], subset[t], height, width, mask=frame_mask)
            pose_img = pose_img.transpose(1, 2, 0)  # (H, W, C)
            pose_img = cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR)
            cv2.putText(
                pose_img,
                titles[idx],
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            frames.append(pose_img)

        combined = np.concatenate(frames, axis=1)
        video_writer.write(combined)

    video_writer.release()


def inference(
    checkpoint_path: str,
    output_dir: str,
    index_begin: int,
    index_end: int,
    height: int,
    width: int,
    fps: int,
    seed: int,
) -> None:
    """
    Run inference with PoseTransformerV4 on PoseDatasetV3 samples and render reconstruction videos.
    """
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
        masked_sequence, input_mask, original_sequence, meta_info = dataset[data_index * args.step]
        filename, norm_params, subset = meta_info['filename'], meta_info['norm_params'], meta_info['subset']
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
        original_parts_np: Dict[str, np.ndarray] = {}
        masked_parts_np: Dict[str, np.ndarray] = {}
        mask_parts_np: Dict[str, np.ndarray] = {}

        for part in parts:
            mask_cpu = input_mask[part].numpy()
            mask_parts_np[part] = mask_cpu

            original_np = original_sequence[part].numpy()
            original_parts_np[part] = original_np

            masked_np = masked_sequence[part].numpy()
            masked_parts_np[part] = np.where(mask_cpu[..., None], -1.0, masked_np)

            pred_part = predictions[part]
            target_len = masked_on_device[part].shape[1]
            pred_part = upsample_predictions(pred_part, target_len)

            reconstructed_part = masked_on_device[part].clone()
            mask_broadcast = mask_on_device[part].unsqueeze(-1).expand_as(reconstructed_part)
            reconstructed_part[mask_broadcast] = pred_part[mask_broadcast]
            reconstructed_parts_np[part] = reconstructed_part.squeeze(0).cpu().numpy()

        original_full = assemble_full_sequence(original_parts_np)
        reconstructed_full = assemble_full_sequence(reconstructed_parts_np)
        masked_full = assemble_full_sequence(masked_parts_np)
        mask_full = assemble_full_mask(mask_parts_np)
        face_keypoints = meta_info.get("face_keypoints")
        if face_keypoints is not None:
            face_np = np.asarray(face_keypoints, dtype=np.float32)[..., :2]
            original_full[:, 18:86, :] = face_np
            masked_full[:, 18:86, :] = face_np
            reconstructed_full[:, 18:86, :] = face_np

        norm_params = np.array(norm_params)
        original_full = denormalize_keypoints(original_full, norm_params)
        masked_full = denormalize_keypoints(masked_full, norm_params)
        reconstructed_full = denormalize_keypoints(reconstructed_full, norm_params)
        align_face_inplace(original_full)
        align_face_inplace(masked_full)
        align_face_inplace(reconstructed_full)

        output_path = os.path.join(output_dir, f"result_{data_index}.mp4")
        print(f"[{data_index}] Rendering to {output_path}")

        render_animation(
            sequences=[original_full, reconstructed_full],  # (N_frame, N_kps, 2)
            subset=subset,
            titles=["Original", "Reconstructed"],
            output_path=output_path,
            masks=[None, mask_full],
            height=height,
            width=width,
            fps=fps,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference script for PoseTransformerV4 with PoseDatasetV3.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--output-dir", type=str, default="output/video_v4_2", help="Directory to store rendered videos.")
    parser.add_argument("--index-begin", type=int, default=0, help="Starting sample index (inclusive).")
    parser.add_argument("--index-end", type=int, default=10, help="Ending sample index (exclusive).")
    parser.add_argument("--step", type=int, default=10, help="Ending sample index (exclusive).")
    parser.add_argument("--height", type=int, default=1080, help="Output video height.")
    parser.add_argument("--width", type=int, default=1080, help="Output video width.")
    parser.add_argument("--fps", type=int, default=2, help="Output video frames per second.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible masking.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inference(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        index_begin=args.index_begin,
        index_end=args.index_end,
        height=args.height,
        width=args.width,
        fps=args.fps,
        seed=args.seed,
    )
