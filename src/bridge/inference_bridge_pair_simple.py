import argparse
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from config_loader import config
from datasets.dataset_v6_bridge import PART_SLICE_FALLBACK
from inference_part import create_subset_placeholder, denormalize_keypoints
from models.model_v6_bridge import PoseBridgeTransformerV6
from utils.constants import get_part_kp_indices
from utils.pickle_compat import load_pickle
from utils.render_conf_confaware import draw_pose

DEFAULT_CANVAS_SIZE = 1080


@dataclass
class MotionSample:
    name: str
    path: str
    parts: Dict[str, np.ndarray]
    subset: Optional[List]
    norm_params: Optional[List]
    num_frames: int
    raw_data: object


def resolve_device(device_arg: Optional[str]) -> str:
    device = device_arg or config["training"]["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_motion_path(motion_arg: str) -> str:
    if os.path.isfile(motion_arg):
        return os.path.abspath(motion_arg)
    data_root = config["data"]["data_dir"]
    candidate = os.path.join(data_root, motion_arg)
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)
    raise FileNotFoundError(f"Cannot locate motion data at '{motion_arg}' or '{candidate}'.")


def load_bridge_model(checkpoint_path: str, device: str) -> PoseBridgeTransformerV6:
    model = PoseBridgeTransformerV6().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_motion_sample(file_arg: str, parts: Sequence[str]) -> MotionSample:
    file_path = resolve_motion_path(file_arg)
    with open(file_path, "rb") as fp:
        data_obj = load_pickle(fp)

    subset_list: Optional[List] = None
    norm_params: Optional[List] = None

    if isinstance(data_obj, dict):
        source: Dict[str, np.ndarray] = {}
        for part in parts:
            key = part
            if part == "face" and key not in data_obj and "face_all" in data_obj:
                key = "face_all"
            if key not in data_obj:
                raise ValueError(f"Missing part '{part}' in {file_path}.")
            source[part] = np.asarray(data_obj[key], dtype=np.float32)
        num_frames = source[parts[0]].shape[0]
    elif isinstance(data_obj, list):
        keypoints = np.stack([frame["keypoints"] for frame in data_obj], axis=0).astype(np.float32)
        subset_list = [frame.get("subset") for frame in data_obj]
        norm_params = [frame.get("norm_params") for frame in data_obj]
        source = {}
        for part in parts:
            if part not in PART_SLICE_FALLBACK:
                raise ValueError(f"No slice information for part '{part}'.")
            sl = PART_SLICE_FALLBACK[part]
            source[part] = keypoints[:, sl, :]
        num_frames = keypoints.shape[0]
    else:
        raise ValueError(f"Unsupported data format for {file_path}.")

    return MotionSample(
        name=Path(file_path).stem,
        path=file_path,
        parts=source,
        subset=subset_list,
        norm_params=norm_params,
        num_frames=num_frames,
        raw_data=data_obj,
    )


def slice_segment(
    part_dict: Dict[str, np.ndarray],
    subset_list: Optional[List],
    norm_list: Optional[List],
    start_idx: int,
    length: int,
) -> Tuple[Dict[str, np.ndarray], Optional[List], Optional[List]]:
    total_frames = next(iter(part_dict.values())).shape[0]
    if start_idx < 0 or start_idx + length > total_frames:
        raise ValueError(f"Segment start {start_idx} + length {length} exceeds total frames {total_frames}.")
    segment = {k: v[start_idx : start_idx + length].astype(np.float32, copy=True) for k, v in part_dict.items()}
    subset_chunk = None if subset_list is None else subset_list[start_idx : start_idx + length]
    norm_chunk = None if norm_list is None else norm_list[start_idx : start_idx + length]
    return segment, subset_chunk, norm_chunk


def assemble_full_sequence(part_dict: Dict[str, np.ndarray]) -> np.ndarray:
    seq_len = next(iter(part_dict.values())).shape[0]
    full = np.zeros((seq_len, 128, 3), dtype=np.float32)
    for part, tensor in part_dict.items():
        if part not in PART_SLICE_FALLBACK:
            continue
        part_slice = PART_SLICE_FALLBACK[part]
        full[:, part_slice, :] = tensor
    return full


def subset_to_array(subset_list: Optional[List], seq_len: int) -> np.ndarray:
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


def build_full_sequence_from_raw(
    raw_data,
    part_dict: Dict[str, np.ndarray],
    subset_list: Optional[List],
    norm_list: Optional[List],
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(raw_data, list):
        keypoints = np.stack([frame["keypoints"] for frame in raw_data], axis=0).astype(np.float32)
        subset_seq = subset_to_array([frame.get("subset") for frame in raw_data], len(raw_data))
        norm_params = [frame.get("norm_params") for frame in raw_data]
        if norm_params and len(norm_params) == keypoints.shape[0]:
            xy = denormalize_keypoints(keypoints[..., :2], norm_params)
            keypoints[..., :2] = xy
        return keypoints, subset_seq

    seq = assemble_full_sequence(part_dict)
    subset_seq = subset_to_array(subset_list, seq.shape[0])
    if norm_list and len(norm_list) == seq.shape[0]:
        xy = denormalize_keypoints(seq[..., :2], norm_list)
        seq[..., :2] = xy
    return seq, subset_seq


def find_last_valid_norm(norm_list: Optional[List]) -> Optional[dict]:
    if not norm_list:
        return None
    for item in reversed(norm_list):
        if isinstance(item, dict):
            return item
    return None


def find_first_valid_norm(norm_list: Optional[List]) -> Optional[dict]:
    if not norm_list:
        return None
    for item in norm_list:
        if isinstance(item, dict):
            return item
    return None


def interpolate_value(start_value, end_value, alpha: float):
    if start_value is None and end_value is None:
        return None
    if start_value is None:
        start_value = end_value
    if end_value is None:
        end_value = start_value
    start_arr = np.asarray(start_value, dtype=np.float32)
    end_arr = np.asarray(end_value, dtype=np.float32)
    blended = (1.0 - alpha) * start_arr + alpha * end_arr
    if blended.ndim == 0:
        return float(blended)
    return blended


def build_gap_norm_params(
    pre_norm: Optional[List],
    post_norm: Optional[List],
    gap_len: int,
) -> List[Optional[dict]]:
    if gap_len <= 0:
        return []

    start_params = find_last_valid_norm(pre_norm)
    end_params = find_first_valid_norm(post_norm)

    if start_params is None and end_params is None:
        return [None] * gap_len

    if start_params is None:
        start_params = end_params
    if end_params is None:
        end_params = start_params

    keys = set(start_params.keys()) | set(end_params.keys())
    gap_params: List[Optional[dict]] = []

    for idx in range(gap_len):
        alpha = (idx + 1) / (gap_len + 1)
        blended = {}
        for key in keys:
            blended_value = interpolate_value(start_params.get(key), end_params.get(key), alpha)
            if blended_value is not None:
                blended[key] = blended_value
        gap_params.append(blended)
    return gap_params


def ensure_list(chunk: Optional[List], length: int) -> List:
    if chunk is None:
        return [None] * length
    if len(chunk) != length:
        raise ValueError(f"Chunk length mismatch: expected {length}, got {len(chunk)}.")
    return chunk


def sequence_to_pose_pixels(
    sequence: np.ndarray,
    subset: np.ndarray,
    canvas_size: int,
    ref_pixels: Optional[np.ndarray],
) -> torch.Tensor:
    frames = []
    target_h = ref_pixels.shape[1] if ref_pixels is not None else canvas_size
    target_w = ref_pixels.shape[2] if ref_pixels is not None else canvas_size
    for t in range(sequence.shape[0]):
        pose_img = draw_pose(sequence[t], subset[t], canvas_size, canvas_size)
        if pose_img.shape[1] != target_h or pose_img.shape[2] != target_w:
            pose_img_hw = pose_img.transpose(1, 2, 0)
            pose_img_hw = cv2.resize(pose_img_hw, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            pose_img = pose_img_hw.transpose(2, 0, 1)
        frames.append(pose_img)
    video_poses = np.stack(frames, axis=0)
    pose_tensor = torch.from_numpy(video_poses.copy()).float() / 127.5 - 1
    return pose_tensor


def load_reference_pixels():
    ref_path = config["data"]["ref_img_pose_im"]
    with open(ref_path, "rb") as fp:
        ref_pixels = pickle.load(fp)
    return np.asarray(ref_pixels)


def render_single_sequence(
    sequence: np.ndarray,
    subset: np.ndarray,
    output_path: Path,
    fps: int,
    canvas_size: int,
    pre_frames: int,
    bridge_frames: int,
) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (canvas_size, canvas_size))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer at {output_path}")

    label_pos = (int(canvas_size * 0.05), int(canvas_size * 0.9))
    bridge_start = pre_frames
    bridge_end = pre_frames + bridge_frames

    for t in range(sequence.shape[0]):
        pose_img = draw_pose(sequence[t], subset[t], canvas_size, canvas_size)
        frame = pose_img.transpose(1, 2, 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if t < bridge_start:
            text, color = "PRE", (255, 255, 255)
        elif t < bridge_end:
            text, color = "BRIDGE", (0, 0, 255)
        else:
            text, color = "POST", (0, 200, 0)

        cv2.putText(
            frame,
            text,
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (0, 0, 0),
            6,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            color,
            3,
            cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()


def bridge_between_motions(
    checkpoint_path: str,
    motion_a_arg: str,
    motion_b_arg: str,
    output_dir: str,
    fps: int,
    canvas_size: int,
    save_pkl: bool,
    device_arg: Optional[str],
) -> Path:
    device = resolve_device(device_arg)
    model = load_bridge_model(checkpoint_path, device)

    parts = list(get_part_kp_indices(include_face=True).keys())
    motion_a = load_motion_sample(motion_a_arg, parts)
    motion_b = load_motion_sample(motion_b_arg, parts)

    pre_len = config["masking"]["pre_context_length"]
    post_len = config["masking"]["post_context_length"]
    gap_len = config["masking"]["gap_length"]

    if motion_a.num_frames < pre_len:
        raise ValueError(f"{motion_a.name} has only {motion_a.num_frames} frames, needs >= {pre_len}.")
    if motion_b.num_frames < post_len:
        raise ValueError(f"{motion_b.name} has only {motion_b.num_frames} frames, needs >= {post_len}.")

    pre_start = motion_a.num_frames - pre_len
    post_start = 0

    pre_segment, pre_subset_chunk, pre_norm_chunk = slice_segment(
        motion_a.parts, motion_a.subset, motion_a.norm_params, pre_start, pre_len
    )
    post_segment, post_subset_chunk, post_norm_chunk = slice_segment(
        motion_b.parts, motion_b.subset, motion_b.norm_params, post_start, post_len
    )

    pre_batch = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in pre_segment.items()}
    post_batch = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in post_segment.items()}
    pre_mask = {k: torch.zeros(1, pre_len, dtype=torch.bool, device=device) for k in pre_segment}
    post_mask = {k: torch.zeros(1, post_len, dtype=torch.bool, device=device) for k in post_segment}

    with torch.no_grad():
        pred_gap = model(pre_batch, post_batch, pre_mask, post_mask)

    pred_parts = {k: v.squeeze(0).cpu().numpy() for k, v in pred_gap.items()}
    full_pred_parts = {
        k: np.concatenate([pre_segment[k], pred_parts[k], post_segment[k]], axis=0) for k in pre_segment
    }
    pred_seq = assemble_full_sequence(full_pred_parts)

    norm_pre = ensure_list(pre_norm_chunk, pre_len)
    norm_post = ensure_list(post_norm_chunk, post_len)
    gap_norm = build_gap_norm_params(norm_pre, norm_post, gap_len)
    norm_combined = norm_pre + gap_norm + norm_post
    if norm_combined and len(norm_combined) == pred_seq.shape[0]:
        xy = denormalize_keypoints(pred_seq[..., :2], norm_combined)
        pred_seq[..., :2] = xy

    full_a_seq, subset_a_full = build_full_sequence_from_raw(
        motion_a.raw_data, motion_a.parts, motion_a.subset, motion_a.norm_params
    )
    full_b_seq, subset_b_full = build_full_sequence_from_raw(
        motion_b.raw_data, motion_b.parts, motion_b.subset, motion_b.norm_params
    )

    gap_only = pred_seq[pre_len : pre_len + gap_len]
    subset_gap = create_subset_placeholder(gap_len)

    combined_seq = np.concatenate([full_a_seq, gap_only, full_b_seq], axis=0)
    combined_subset = np.concatenate([subset_a_full, subset_gap, subset_b_full], axis=0)

    os.makedirs(output_dir, exist_ok=True)
    output_name = f"{motion_a.name}__{motion_b.name}.mp4"
    output_path = Path(output_dir) / output_name
    render_single_sequence(
        sequence=combined_seq,
        subset=combined_subset,
        output_path=output_path,
        fps=fps,
        canvas_size=canvas_size,
        pre_frames=full_a_seq.shape[0],
        bridge_frames=gap_len,
    )
    print(f"Saved bridge visualization to {output_path}")

    if save_pkl:
        ref_pixels = load_reference_pixels()
        pose_pixels = sequence_to_pose_pixels(combined_seq, combined_subset, canvas_size, ref_pixels)
        ref_tensor = torch.from_numpy(np.expand_dims(ref_pixels, axis=0)).float() / 127.5 - 1
        pose_pixels = torch.cat([ref_tensor, pose_pixels], dim=0)
        pkl_path = Path(output_dir) / f"{motion_a.name}__{motion_b.name}.pkl"
        with open(pkl_path, "wb") as fp:
            pickle.dump(
                {
                    "pose_pixels": pose_pixels,
                    "motion_a_data": motion_a.raw_data,
                    "motion_b_data": motion_b.raw_data,
                    "pre_start": pre_start,
                    "post_start": post_start,
                    "gap_length": gap_len,
                },
                fp,
            )
        print(f"Saved combined pose pickle to {pkl_path}")

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a bridge motion between two sequences with a single-column visualization."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the bridge model checkpoint.")
    parser.add_argument("--motion_a", type=str, required=True, help="Path or name of the first motion PKL.")
    parser.add_argument("--motion_b", type=str, required=True, help="Path or name of the second motion PKL.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/bridge_pair_simple",
        help="Directory to store rendered videos (and optional PKLs).",
    )
    parser.add_argument("--fps", type=int, default=5, help="FPS for the rendered mp4.")
    parser.add_argument("--canvas_size", type=int, default=DEFAULT_CANVAS_SIZE, help="Canvas size for rendering.")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu, cuda, or auto).")
    parser.add_argument("--save_pkl", action="store_true", help="Export the combined pose_pixels pickle.")
    return parser.parse_args()


def main():
    args = parse_args()
    bridge_between_motions(
        checkpoint_path=args.checkpoint,
        motion_a_arg=args.motion_a,
        motion_b_arg=args.motion_b,
        output_dir=args.output_dir,
        fps=args.fps,
        canvas_size=args.canvas_size,
        save_pkl=args.save_pkl,
        device_arg=args.device,
    )


if __name__ == "__main__":
    main()
