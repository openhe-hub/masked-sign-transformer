import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from config_loader import config
from datasets.dataset_v6_bridge import PoseBridgeDatasetV6, PART_SLICE_FALLBACK
from models.model_v6_bridge import PoseBridgeTransformerV6
from utils.render_conf_confaware import draw_pose
from inference_part import create_subset_placeholder, denormalize_keypoints


def assemble_full_sequence(part_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Assemble per-part tensors into a full (seq_len, 128, 3) sequence.
    """
    seq_len = next(iter(part_dict.values())).shape[0]
    full = np.zeros((seq_len, 128, 3), dtype=np.float32)
    for part, tensor in part_dict.items():
        if part not in PART_SLICE_FALLBACK:
            continue
        part_slice = PART_SLICE_FALLBACK[part]
        part_len = part_slice.stop - part_slice.start
        if tensor.shape[1] != part_len:
            raise ValueError(f"Unexpected length for part {part}: {tensor.shape[1]} vs {part_len}")
        full[:, part_slice, :] = tensor
    return full


def subset_to_array(subset_list, seq_len: int) -> np.ndarray:
    placeholder = create_subset_placeholder(seq_len)
    if not subset_list or len(subset_list) != seq_len:
        return placeholder
    frames = []
    for idx, subset in enumerate(subset_list):
        if subset is None:
            frames.append(placeholder[idx])
            continue
        arr = np.asarray(subset)
        if arr.ndim == 1:
            arr = arr[None, :]
        frame_subset = placeholder[idx].copy()
        cols = min(frame_subset.shape[-1], arr.shape[-1])
        frame_subset[:, :cols] = arr[:, :cols]
        frames.append(frame_subset)
    return np.stack(frames, axis=0)


def render_sequences(
    sequences: List[np.ndarray],
    titles: List[str],
    output_path: Path,
    fps: int,
    canvas_size: int = 1080,
    subset: np.ndarray = None,
    highlight_gap: tuple = None,
    highlight_indices: List[int] = None,
) -> None:
    """
    Render multiple sequences side by side.
    """
    assert len(sequences) == len(titles)
    seq_len = sequences[0].shape[0]
    if subset is None:
        subset = create_subset_placeholder(seq_len)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (canvas_size * len(sequences), canvas_size))
    if not video_writer.isOpened():
        raise RuntimeError(f"Failed to open video writer at {output_path}")

    highlight_indices = highlight_indices or []
    gap_range = highlight_gap or (0, 0)
    gap_start, gap_end = gap_range

    for t in range(seq_len):
        frames = []
        for idx, seq in enumerate(sequences):
            gap_mask_frame = None
            if idx in highlight_indices and gap_start <= t < gap_end:
                gap_mask_frame = np.ones(seq.shape[1], dtype=bool)
            pose_img = draw_pose(
                seq[t],
                subset[t],
                canvas_size,
                canvas_size,
                highlight_mask=gap_mask_frame,
            )
            frame = pose_img.transpose(1, 2, 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(frame, titles[idx], (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3, cv2.LINE_AA)
            frames.append(frame)
        combined = np.concatenate(frames, axis=1)
        video_writer.write(combined)

    video_writer.release()


def inference(
    checkpoint_path: str,
    output_dir: str,
    limit: int,
    fps: int,
    num_samples: int,
    start_index: int,
) -> None:
    device = config["training"]["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = PoseBridgeDatasetV6(return_meta=True)
    model = PoseBridgeTransformerV6().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    end_index = min(len(dataset), start_index + num_samples)

    gap_len = config['masking']['gap_length']
    pre_len = config['masking']['pre_context_length']
    gap_range = (pre_len, pre_len + gap_len)

    for sample_idx in range(start_index, end_index):
        sample = dataset[sample_idx]
        if len(sample) == 6:
            pre_dict, post_dict, pre_mask, post_mask, gap_target_dict, meta_info = sample
        else:
            pre_dict, post_dict, pre_mask, post_mask, gap_target_dict = sample
            meta_info = None

        pre_batch = {k: v.unsqueeze(0).to(device) for k, v in pre_dict.items()}
        post_batch = {k: v.unsqueeze(0).to(device) for k, v in post_dict.items()}
        pre_mask_batch = {k: v.unsqueeze(0).to(device) for k, v in pre_mask.items()}
        post_mask_batch = {k: v.unsqueeze(0).to(device) for k, v in post_mask.items()}

        with torch.no_grad():
            pred_gap = model(pre_batch, post_batch, pre_mask_batch, post_mask_batch)

        pred_parts = {k: v.squeeze(0).cpu().numpy() for k, v in pred_gap.items()}
        pre_np = {k: v.cpu().numpy() for k, v in pre_dict.items()}
        post_np = {k: v.cpu().numpy() for k, v in post_dict.items()}
        gap_np = {k: v.cpu().numpy() for k, v in gap_target_dict.items()}

        full_target_parts = {k: np.concatenate([pre_np[k], gap_np[k], post_np[k]], axis=0) for k in pre_np}
        full_pred_parts = {k: np.concatenate([pre_np[k], pred_parts[k], post_np[k]], axis=0) for k in pre_np}

        target_seq = assemble_full_sequence(full_target_parts)
        pred_seq = assemble_full_sequence(full_pred_parts)

        norm_params = meta_info.get('norm_params') if meta_info else None
        if norm_params is not None and len(norm_params) == target_seq.shape[0]:
            target_xy = denormalize_keypoints(target_seq[..., :2], norm_params)
            pred_xy = denormalize_keypoints(pred_seq[..., :2], norm_params)
            target_seq[..., :2] = target_xy
            pred_seq[..., :2] = pred_xy

        subset_seq = meta_info.get('subset') if meta_info else None
        subset_array = subset_to_array(subset_seq, target_seq.shape[0])

        sample_name = f"sample_{sample_idx:05d}"
        output_path = Path(output_dir) / f"{sample_name}.mp4"
        render_sequences(
            sequences=[target_seq, pred_seq],
            titles=["Ground Truth", "Predicted"],
            output_path=output_path,
            fps=fps,
            subset=subset_array,
            highlight_gap=gap_range,
            highlight_indices=[1],
        )
        print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bridge inference and visualization script.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to bridge model checkpoint.")
    parser.add_argument("--output_dir", type=str, default="output/bridge_inference", help="Directory to save videos.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize.")
    parser.add_argument("--start_index", type=int, default=0, help="Dataset index to start visualization from.")
    parser.add_argument("--fps", type=int, default=5, help="FPS for output videos.")

    args = parser.parse_args()

    inference(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        limit=config["data"].get("limit", 0),
        fps=args.fps,
        num_samples=args.num_samples,
        start_index=args.start_index,
    )
