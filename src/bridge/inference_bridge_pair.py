import argparse
import os
import sys
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from config_loader import config
from datasets.dataset_v6_bridge import PART_SLICE_FALLBACK
from models.model_v6_bridge import PoseBridgeTransformerV6
from utils.constants import get_part_kp_indices
from utils.pickle_compat import load_pickle
from utils.render_conf_confaware import draw_pose
from inference_part import create_subset_placeholder, denormalize_keypoints

DEFAULT_CANVAS_SIZE = 1080


def assemble_full_sequence(part_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Assemble per-part tensors into a full (seq_len, 128, 3) sequence.
    """
    if not part_dict:
        raise ValueError("part_dict is empty.")
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


def resolve_device(device_arg: Optional[str] = None) -> str:
    """
    Resolve the computation device from config or explicit argument.
    """
    device = device_arg or config["training"]["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_bridge_model(checkpoint_path: str, device: str) -> PoseBridgeTransformerV6:
    model = PoseBridgeTransformerV6().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_reference_pixels() -> np.ndarray:
    ref_path = config['data']['ref_img_pose_im']
    with open(ref_path, 'rb') as fp:
        ref_pixels = pickle.load(fp)
    return np.asarray(ref_pixels)


def sequence_to_pose_pixels(
    sequence: np.ndarray,
    subset: np.ndarray,
    canvas_size: int,
    ref_pixels: np.ndarray,
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
    pose_stack = video_poses
    if ref_pixels is not None:
        pose_stack = np.concatenate([np.expand_dims(ref_pixels, axis=0), pose_stack], axis=0)
    pose_tensor = torch.from_numpy(pose_stack.copy()).float() / 127.5 - 1
    return pose_tensor


def build_full_sequence_from_raw(
    raw_data,
    part_dict: Dict[str, np.ndarray],
    subset_list: Optional[List],
    norm_list: Optional[List],
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(raw_data, list):
        keypoints = np.stack([frame['keypoints'] for frame in raw_data], axis=0)
        norm_params = [frame.get('norm_params') for frame in raw_data]
        subset_seq = subset_to_array([frame.get('subset') for frame in raw_data], len(raw_data))
        if norm_params and len(norm_params) == keypoints.shape[0]:
            keypoints_xy = denormalize_keypoints(keypoints[..., :2], norm_params)
            keypoints[..., :2] = keypoints_xy
        return keypoints, subset_seq

    seq = assemble_full_sequence(part_dict)
    if norm_list and len(norm_list) == seq.shape[0]:
        seq_xy = denormalize_keypoints(seq[..., :2], norm_list)
        seq[..., :2] = seq_xy
    subset_seq = subset_to_array(subset_list, seq.shape[0])
    return seq, subset_seq


def render_sequences(
    sequences: List[np.ndarray],
    titles: List[str],
    output_path: Path,
    fps: int,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    subset: Union[np.ndarray, Sequence[Optional[np.ndarray]]] = None,
    highlight_gaps: Optional[Union[Tuple[int, int], Sequence[Optional[Tuple[int, int]]]]] = None,
    highlight_indices: Optional[List[int]] = None,
    highlight_text: str = "BRIDGE",
    context_zones: Optional[List[Tuple[int, int, int, str, Tuple[int, int, int]]]] = None,
) -> None:
    """
    Render multiple sequences side by side.
    """
    assert len(sequences) == len(titles)
    seq_len = sequences[0].shape[0]
    if subset is None:
        subset_list = [create_subset_placeholder(seq_len) for _ in sequences]
    elif isinstance(subset, (list, tuple)):
        subset_list = []
        for idx in range(len(sequences)):
            sub = subset[idx] if idx < len(subset) else None
            if sub is None or sub.shape[0] != seq_len:
                subset_list.append(create_subset_placeholder(seq_len))
            else:
                subset_list.append(sub)
    else:
        subset_arr = subset
        if subset_arr.shape[0] != seq_len:
            subset_arr = create_subset_placeholder(seq_len)
        subset_list = [subset_arr for _ in sequences]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (canvas_size * len(sequences), canvas_size))
    if not video_writer.isOpened():
        raise RuntimeError(f"Failed to open video writer at {output_path}")

    if highlight_gaps is None:
        gap_ranges = [None] * len(sequences)
    elif isinstance(highlight_gaps, tuple):
        gap_ranges = [highlight_gaps] * len(sequences)
    else:
        gap_ranges = list(highlight_gaps)
        if len(gap_ranges) < len(sequences):
            gap_ranges.extend([None] * (len(sequences) - len(gap_ranges)))
        gap_ranges = gap_ranges[: len(sequences)]

    highlight_indices = set(highlight_indices) if highlight_indices else None

    context_zones = context_zones or []
    overlay_pos = (int(canvas_size * 0.25), canvas_size - 50)

    for t in range(seq_len):
        frames = []
        for idx, seq in enumerate(sequences):
            pose_img = draw_pose(
                seq[t],
                subset_list[idx][t],
                canvas_size,
                canvas_size,
            )
            frame = pose_img.transpose(1, 2, 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(frame, titles[idx], (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3, cv2.LINE_AA)
            for target_idx, zone_start, zone_end, zone_label, zone_color in context_zones:
                if zone_label and idx == target_idx and zone_start <= t < zone_end:
                    cv2.putText(
                        frame,
                        zone_label,
                        overlay_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.8,
                        zone_color,
                        5,
                        cv2.LINE_AA,
                    )
            gap_range = gap_ranges[idx] if gap_ranges else None
            should_highlight = (
                highlight_text
                and gap_range is not None
                and (highlight_indices is None or idx in highlight_indices)
                and gap_range[0] <= t < gap_range[1]
            )
            if should_highlight:
                cv2.putText(
                    frame,
                    highlight_text,
                    overlay_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.2,
                    (0, 0, 255),
                    6,
                    cv2.LINE_AA,
                )
            frames.append(frame)
        combined = np.concatenate(frames, axis=1)
        video_writer.write(combined)

    video_writer.release()


def resolve_motion_path(motion_arg: str) -> str:
    """
    Resolve a motion path argument to an absolute file path.
    """
    if os.path.isfile(motion_arg):
        return os.path.abspath(motion_arg)
    data_root = os.path.abspath(os.path.join(ROOT_DIR, config['data']['data_dir']))
    candidate = os.path.join(data_root, motion_arg)
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f"Cannot find motion data at '{motion_arg}' or '{candidate}'.")


def load_motion_parts(
    file_path: str,
    parts: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], Optional[List], Optional[List], int, object]:
    with open(file_path, 'rb') as f:
        data_obj = load_pickle(f)

    subset_list = None
    norm_params = None

    if isinstance(data_obj, dict):
        source = {}
        for part in parts:
            key = part
            if part == 'face' and key not in data_obj and 'face_all' in data_obj:
                key = 'face_all'
            if key not in data_obj:
                raise ValueError(f"Missing part '{part}' in {file_path}.")
            source[part] = np.asarray(data_obj[key], dtype=np.float32)
        num_frames = source[parts[0]].shape[0]
    elif isinstance(data_obj, list):
        keypoints = np.stack([frame['keypoints'] for frame in data_obj], axis=0).astype(np.float32)
        subset_list = [frame.get('subset') for frame in data_obj]
        norm_params = [frame.get('norm_params') for frame in data_obj]
        source = {}
        for part in parts:
            if part not in PART_SLICE_FALLBACK:
                raise ValueError(f"No slice information for part '{part}'.")
            sl = PART_SLICE_FALLBACK[part]
            source[part] = keypoints[:, sl, :]
        num_frames = keypoints.shape[0]
    else:
        raise ValueError(f"Unsupported data format for {file_path}.")

    return source, subset_list, norm_params, num_frames, data_obj


def select_segment(
    part_dict: Dict[str, np.ndarray],
    subset_list: Optional[List],
    norm_list: Optional[List],
    start_idx: int,
    length: int,
    label: str,
) -> Tuple[Dict[str, np.ndarray], Optional[List], Optional[List], int]:
    total_frames = next(iter(part_dict.values())).shape[0]
    if total_frames < length:
        raise ValueError(f"{label} only has {total_frames} frames but needs {length}.")

    if start_idx < 0:
        start_idx = total_frames - length

    if start_idx + length > total_frames:
        raise ValueError(f"{label} start {start_idx} out of range for length {length} (total {total_frames}).")

    segment = {
        part: data[start_idx : start_idx + length].astype(np.float32, copy=True)
        for part, data in part_dict.items()
    }
    subset_chunk = None
    if subset_list is not None:
        subset_chunk = subset_list[start_idx : start_idx + length]
    norm_chunk = None
    if norm_list is not None:
        norm_chunk = norm_list[start_idx : start_idx + length]
    return segment, subset_chunk, norm_chunk, start_idx


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


def bridge_pair_inference(
    checkpoint_path: str,
    motion_a: str,
    motion_b: str,
    start_a: int,
    start_b: int,
    output_dir: str,
    fps: int,
    save_pkl: bool = False,
    full_only: bool = False,
) -> Path:
    ref_pixels = load_reference_pixels() if save_pkl else None
    return bridge_pair_inference_with_model(
        checkpoint_path=checkpoint_path,
        motion_a=motion_a,
        motion_b=motion_b,
        start_a=start_a,
        start_b=start_b,
        output_dir=output_dir,
        fps=fps,
        save_pkl=save_pkl,
        model=None,
        device=None,
        ref_pixels=ref_pixels,
        full_only=full_only,
    )


def bridge_pair_inference_with_model(
    checkpoint_path: Optional[str],
    motion_a: str,
    motion_b: str,
    start_a: int,
    start_b: int,
    output_dir: str,
    fps: int,
    save_pkl: bool,
    model: Optional[PoseBridgeTransformerV6],
    device: Optional[str],
    ref_pixels: Optional[np.ndarray],
    full_only: bool,
) -> Path:
    device = resolve_device(device)

    own_model = False
    if model is None:
        if not checkpoint_path:
            raise ValueError("checkpoint_path must be provided when model is None.")
        model = load_bridge_model(checkpoint_path, device)
        own_model = True

    pre_len = config['masking']['pre_context_length']
    post_len = config['masking']['post_context_length']
    gap_len = config['masking']['gap_length']

    os.makedirs(output_dir, exist_ok=True)

    parts = list(get_part_kp_indices(include_face=True).keys())

    motion_a_path = resolve_motion_path(motion_a)
    motion_b_path = resolve_motion_path(motion_b)
    motion_a_name = Path(motion_a_path).stem
    motion_b_name = Path(motion_b_path).stem

    a_parts, a_subset, a_norm, a_total, a_raw = load_motion_parts(motion_a_path, parts)
    b_parts, b_subset, b_norm, b_total, b_raw = load_motion_parts(motion_b_path, parts)

    pre_segment, pre_subset_chunk, pre_norm_chunk, actual_start_a = select_segment(
        a_parts, a_subset, a_norm, start_a, pre_len, f"{motion_a_name} (pre)"
    )
    post_segment, post_subset_chunk, post_norm_chunk, actual_start_b = select_segment(
        b_parts, b_subset, b_norm, start_b, post_len, f"{motion_b_name} (post)"
    )

    print(
        f"Using frames {actual_start_a}:{actual_start_a + pre_len} from {motion_a_name} "
        f"and {actual_start_b}:{actual_start_b + post_len} from {motion_b_name}."
    )

    pre_batch = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in pre_segment.items()}
    post_batch = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in post_segment.items()}
    pre_mask_batch = {k: torch.zeros(1, pre_len, dtype=torch.bool, device=device) for k in pre_segment}
    post_mask_batch = {k: torch.zeros(1, post_len, dtype=torch.bool, device=device) for k in post_segment}

    with torch.no_grad():
        pred_gap = model(pre_batch, post_batch, pre_mask_batch, post_mask_batch)

    pred_parts = {k: v.squeeze(0).cpu().numpy() for k, v in pred_gap.items()}
    pre_np = {k: v.copy() for k, v in pre_segment.items()}
    post_np = {k: v.copy() for k, v in post_segment.items()}

    gap_placeholder = {
        k: np.full((gap_len, v.shape[1], v.shape[2]), -1.0, dtype=np.float32) for k, v in pre_np.items()
    }

    full_pred_parts = {
        k: np.concatenate([pre_np[k], pred_parts[k], post_np[k]], axis=0) for k in pre_np
    }
    context_parts = {
        k: np.concatenate([pre_np[k], gap_placeholder[k], post_np[k]], axis=0) for k in pre_np
    }

    pred_seq = assemble_full_sequence(full_pred_parts)
    context_seq = assemble_full_sequence(context_parts)

    norm_pre = ensure_list(pre_norm_chunk, pre_len)
    norm_post = ensure_list(post_norm_chunk, post_len)
    gap_norm = build_gap_norm_params(norm_pre, norm_post, gap_len)
    norm_combined = norm_pre + gap_norm + norm_post

    subset_pre = subset_to_array(pre_subset_chunk, pre_len)
    subset_post = subset_to_array(post_subset_chunk, post_len)
    subset_gap = create_subset_placeholder(gap_len)
    subset_combined = np.concatenate([subset_pre, subset_gap, subset_post], axis=0)

    full_a_seq, subset_a_full = build_full_sequence_from_raw(a_raw, a_parts, a_subset, a_norm)
    full_b_seq, subset_b_full = build_full_sequence_from_raw(b_raw, b_parts, b_subset, b_norm)
    gap_only_seq = pred_seq[pre_len:pre_len + gap_len]
    subset_gap_full = create_subset_placeholder(gap_len)
    combined_seq = np.concatenate([full_a_seq, gap_only_seq, full_b_seq], axis=0)
    combined_subset = np.concatenate([subset_a_full, subset_gap_full, subset_b_full], axis=0)

    context_xy = context_seq[..., :2]
    pred_xy = pred_seq[..., :2]

    gap_range = (pre_len, pre_len + gap_len)

    if norm_combined and len(norm_combined) == pred_seq.shape[0]:
        context_xy = denormalize_keypoints(context_xy, norm_combined)
        pred_xy = denormalize_keypoints(pred_xy, norm_combined)
        context_seq[..., :2] = context_xy
        pred_seq[..., :2] = pred_xy
        context_seq[gap_range[0]:gap_range[1], :, :2] = -1.0
    output_name = f"{motion_a_name}__{motion_b_name}.mp4"
    output_path = Path(output_dir) / output_name

    combined_gap_range = (full_a_seq.shape[0], full_a_seq.shape[0] + gap_len)
    if full_only:
        context_zones = [
            (0, 0, full_a_seq.shape[0], "PRE", (255, 0, 0)),
            (0, combined_gap_range[1], combined_seq.shape[0], "POST", (0, 200, 0)),
        ]
        render_sequences(
            sequences=[combined_seq],
            titles=["Predicted (A + Bridge + B)"],
            output_path=output_path,
            fps=fps,
            subset=[combined_subset],
            highlight_gaps=[combined_gap_range],
            highlight_indices=[0],
            context_zones=context_zones,
        )
    else:
        context_zones = [
            (1, 0, pre_len, "PRE", (255, 0, 0)),
            (1, pre_len + gap_len, pred_seq.shape[0], "POST", (0, 200, 0)),
            (2, 0, full_a_seq.shape[0], "PRE", (255, 0, 0)),
            (2, combined_gap_range[1], combined_seq.shape[0], "POST", (0, 200, 0)),
        ]
        render_sequences(
            sequences=[context_seq, pred_seq, combined_seq],
            titles=["Context Only", "Bridge Result", "Full A+Bridge+B"],
            output_path=output_path,
            fps=fps,
            subset=[subset_combined, subset_combined, combined_subset],
            highlight_gaps=[None, gap_range, combined_gap_range],
            highlight_indices=[1, 2],
            context_zones=context_zones,
        )
    print(f"Saved bridge visualization to {output_path}")

    if save_pkl:
        if ref_pixels is None:
            ref_pixels = load_reference_pixels()
        pose_pixels = sequence_to_pose_pixels(combined_seq, combined_subset, DEFAULT_CANVAS_SIZE, ref_pixels)
        pkl_path = Path(output_dir) / f"{motion_a_name}__{motion_b_name}.pkl"
        with open(pkl_path, 'wb') as fp:
            pickle.dump(
                {
                    'pose_pixels': pose_pixels,
                    'motion_a_name': motion_a_name,
                    'motion_b_name': motion_b_name,
                    'motion_a_data': a_raw,
                    'motion_b_data': b_raw,
                    'pre_start': actual_start_a,
                    'post_start': actual_start_b,
                    'pre_length': pre_len,
                    'gap_length': gap_len,
                    'post_length': post_len,
                    'gap_range': gap_range,
                },
                fp,
            )
        print(f"Saved pickle outputs to {pkl_path}")

    if own_model:
        del model
    return output_path


def parse_pairs_file(pairs_file: str, default_fps: int) -> List[dict]:
    pairs: List[dict] = []
    with open(pairs_file, 'r', encoding='utf-8') as fp:
        for line_no, raw_line in enumerate(fp, 1):
            line = raw_line.split('#', 1)[0].strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(',')]
            if len(parts) < 2:
                print(f"Skipping line {line_no} in {pairs_file}: need at least motion_a,motion_b.")
                continue
            def safe_int(value, default):
                if value in ("", None):
                    return default
                try:
                    return int(value)
                except ValueError:
                    return default
            motion_a = parts[0]
            motion_b = parts[1]
            start_a = safe_int(parts[2], -1) if len(parts) > 2 else -1
            start_b = safe_int(parts[3], 0) if len(parts) > 3 else 0
            fps = safe_int(parts[4], default_fps) if len(parts) > 4 else default_fps
            save_pkl = False
            if len(parts) > 5 and parts[5]:
                save_pkl = parts[5].lower() in ("1", "true", "yes")
            pairs.append(
                {
                    "motion_a": motion_a,
                    "motion_b": motion_b,
                    "start_a": start_a,
                    "start_b": start_b,
                    "fps": fps,
                    "save_pkl": save_pkl,
                }
            )
    return pairs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bridge inference between two independent motions.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to bridge model checkpoint.")
    parser.add_argument("--motion_a", type=str, help="First motion file (path or basename).")
    parser.add_argument("--motion_b", type=str, help="Second motion file (path or basename).")
    parser.add_argument(
        "--start_a",
        type=int,
        default=-1,
        help="Start index for Motion A context (default: last pre_context_length frames).",
    )
    parser.add_argument(
        "--start_b",
        type=int,
        default=0,
        help="Start index for Motion B context (default: first post_context_length frames).",
    )
    parser.add_argument("--output_dir", type=str, default="output/bridge_pair", help="Directory for outputs.")
    parser.add_argument("--fps", type=int, default=5, help="FPS for the rendered video.")
    parser.add_argument("--save_pkl", action="store_true", help="Save pickle outputs for inspection.")
    parser.add_argument(
        "--pairs_file",
        type=str,
        help="Optional CSV file listing multiple pairs: motion_a,motion_b,start_a,start_b,fps,save_pkl",
    )
    parser.add_argument("--full_only", action="store_true", help="Render only the full A+Bridge+B column.")

    args = parser.parse_args()

    if args.pairs_file:
        pair_entries = parse_pairs_file(args.pairs_file, args.fps)
        if not pair_entries:
            raise ValueError(f"No valid pairs found in {args.pairs_file}.")
        device = resolve_device(None)
        model = load_bridge_model(args.checkpoint, device)
        need_ref = args.save_pkl or any(entry["save_pkl"] for entry in pair_entries)
        ref_pixels = load_reference_pixels() if need_ref else None
        for entry in pair_entries:
            pair_save_pkl = args.save_pkl or entry["save_pkl"]
            bridge_pair_inference_with_model(
                checkpoint_path=args.checkpoint,
                motion_a=entry["motion_a"],
                motion_b=entry["motion_b"],
                start_a=entry["start_a"],
                start_b=entry["start_b"],
                output_dir=args.output_dir,
                fps=entry["fps"],
                save_pkl=pair_save_pkl,
                model=model,
                device=device,
                ref_pixels=ref_pixels,
                full_only=args.full_only,
            )
    else:
        if not args.motion_a or not args.motion_b:
            parser.error("Either specify --pairs_file or provide both --motion_a and --motion_b.")
        bridge_pair_inference(
            checkpoint_path=args.checkpoint,
            motion_a=args.motion_a,
            motion_b=args.motion_b,
            start_a=args.start_a,
            start_b=args.start_b,
            output_dir=args.output_dir,
            fps=args.fps,
            save_pkl=args.save_pkl,
            full_only=args.full_only,
        )
