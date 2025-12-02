import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import pickle
import toml
from tqdm.auto import tqdm

from config_loader import config
from datasets.dataset_v3_continuous import PoseDatasetV3Continuous
from models.model_v4 import PoseTransformerV4
from utils.face_processing import align_face_inplace, graft_upper_body_inplace
from utils.render_conf_confaware import draw_pose
from utils.interpolate_xy import interpolate_xy

from inference_part import (
    PART_TO_GLOBAL_INDICES,
    assemble_full_sequence,
    create_subset_placeholder,
    denormalize_keypoints,
    upsample_predictions,
)


def sequence_to_frames_mm(
    sequence: np.ndarray,
    subset: np.ndarray,
    height: int,
    width: int,
):
    frames = []
    for t in range(sequence.shape[0]):
        pose_img = draw_pose(sequence[t], subset[t], height, width)
        # frame = cv2.cvtColor(pose_img.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        frames.append(pose_img)
    return np.stack(frames)


def export_video_from_frames(
    frames: np.ndarray,
    output_path: Path,
    fps: int,
    *,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> None:
    if frames.size == 0:
        raise ValueError("Cannot export video: received an empty frame buffer.")

    num_frames, _, frame_h, frame_w = frames.shape
    target_h = height or frame_h
    target_w = width or frame_w

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (target_w, target_h))
    if not video_writer.isOpened():
        raise RuntimeError(f"Failed to open video writer at {output_path}")

    resize_needed = target_h != frame_h or target_w != frame_w
    for idx in range(num_frames):
        frame_rgb = frames[idx].transpose(1, 2, 0)  # (H, W, 3)
        if resize_needed:
            frame_rgb = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

    video_writer.release()


def to_numpy_subset(subset_list: Optional[Iterable[np.ndarray]], seq_len: int) -> np.ndarray:
    if subset_list:
        stacked = np.stack([np.asarray(item) for item in subset_list], axis=0)
        return stacked
    return create_subset_placeholder(seq_len)


def compute_head_bbox_size(keypoints: np.ndarray) -> Tuple[float, float]:
    """
    Compute average head bounding box size across all frames.

    Args:
        keypoints: (num_frames, num_keypoints, 2) array

    Returns:
        (avg_width, avg_height) of head bbox
    """
    face_kpts = keypoints[:, 18:86, :]  # Face keypoints across all frames

    widths = []
    heights = []

    for frame_idx in range(face_kpts.shape[0]):
        kpts = face_kpts[frame_idx]

        # Filter valid points
        valid_mask = ~(np.isnan(kpts).any(axis=1) | (kpts == 0).all(axis=1))
        if not valid_mask.any():
            continue

        valid_kpts = kpts[valid_mask]
        x_min, y_min = valid_kpts.min(axis=0)
        x_max, y_max = valid_kpts.max(axis=0)

        widths.append(x_max - x_min)
        heights.append(y_max - y_min)

    if not widths:
        return 0.0, 0.0

    return float(np.mean(widths)), float(np.mean(heights))


def normalize_head_size(
    keypoints: np.ndarray,
    target_width: float,
    target_height: float,
) -> np.ndarray:
    """
    Normalize head size to target dimensions for all frames.

    Args:
        keypoints: (num_frames, num_keypoints, 2) array
        target_width: target head width
        target_height: target head height

    Returns:
        Normalized keypoints with same shape
    """
    result = keypoints.copy()

    for frame_idx in range(keypoints.shape[0]):
        face_kpts = keypoints[frame_idx, 18:86, :]

        # Filter valid points
        valid_mask = ~(np.isnan(face_kpts).any(axis=1) | (face_kpts == 0).all(axis=1))
        if not valid_mask.any():
            continue

        valid_kpts = face_kpts[valid_mask]

        # Compute current bbox
        x_min, y_min = valid_kpts.min(axis=0)
        x_max, y_max = valid_kpts.max(axis=0)
        current_width = x_max - x_min
        current_height = y_max - y_min

        if current_width == 0 or current_height == 0:
            continue

        # Compute center
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        # Compute scale factors
        scale_x = target_width / current_width
        scale_y = target_height / current_height

        # Apply scaling to face keypoints (18:86)
        face_kpts_frame = result[frame_idx, 18:86, :]
        face_kpts_frame[:, 0] = center_x + (face_kpts_frame[:, 0] - center_x) * scale_x
        face_kpts_frame[:, 1] = center_y + (face_kpts_frame[:, 1] - center_y) * scale_y
        result[frame_idx, 18:86, :] = face_kpts_frame

    return result


def generate_interpolation_frames(
    start_entry: Dict[str, np.ndarray],
    end_entry: Dict[str, np.ndarray],
    interp_frames: int,
    ref_height: int,
    ref_width: int,
) -> Optional[np.ndarray]:
    """Generate interpolation frames between two consecutive entries."""
    if interp_frames <= 0:
        return None

    per_pair_steps = interp_frames + 2
    start_pose_xy = start_entry["keypoints_xy"][-1]
    end_pose_xy = end_entry["keypoints_xy"][0]

    interpolated = interpolate_xy(
        start_pose_xy,
        end_pose_xy,
        num_steps=per_pair_steps,
        drop_endpoints=True,
    )

    if interpolated.size == 0:
        return None

    # Interpolate confidence values
    start_conf = start_entry["keypoints_with_conf"][-1, :, 2]
    end_conf = end_entry["keypoints_with_conf"][0, :, 2]
    alphas = np.linspace(0.0, 1.0, per_pair_steps, dtype=np.float32)[1:-1]
    conf_interp = (
        (1.0 - alphas[:, None]) * start_conf[None, :]
        + alphas[:, None] * end_conf[None, :]
    )

    interpolated_with_conf = np.concatenate(
        [interpolated, conf_interp[..., np.newaxis]],
        axis=-1,
    )

    # Repeat subset template
    subset_template = start_entry["subset"][-1]
    repeated_subset = np.repeat(
        subset_template[np.newaxis, ...],
        interpolated_with_conf.shape[0],
        axis=0,
    )

    return sequence_to_frames_mm(
        interpolated_with_conf,
        repeated_subset,
        ref_height,
        ref_width,
    )


def load_motion_groups(config_path: str) -> Dict[str, List[str]]:
    """
    Load all motion groups from TOML configuration.

    Returns dict mapping group_name -> list of filenames (basename only).
    """
    with open(config_path, 'r') as f:
        motion_config = toml.load(f)

    if 'groups' not in motion_config:
        raise ValueError(f"No 'groups' section found in {config_path}")

    all_groups = {}
    data_dir = Path(config["data"]["data_dir"])

    for group_name, group in motion_config['groups'].items():
        file_paths = group.get('files', [])
        if not file_paths:
            print(f"Warning: Group '{group_name}' has no files defined (skipping)")
            continue

        # Extract basenames from paths and verify files exist
        file_list = []
        for file_path in file_paths:
            file_path = Path(file_path)
            basename = file_path.name

            # Check if file exists in data_dir
            full_path = data_dir / basename
            if not full_path.exists():
                # Try the path as-is if it's not in data_dir
                if Path(file_path).exists():
                    full_path = Path(file_path)
                    basename = full_path.name
                else:
                    print(f"Warning: File not found: {basename} (skipping)")
                    continue

            file_list.append(basename)

        if file_list:
            all_groups[group_name] = file_list

    return all_groups


def load_motion_group(config_path: str, group_name: str) -> List[str]:
    """
    Load motion file list from TOML configuration.

    Returns list of filenames (basename only) that will be looked up in data_dir.
    """
    with open(config_path, 'r') as f:
        motion_config = toml.load(f)

    if 'groups' not in motion_config:
        raise ValueError(f"No 'groups' section found in {config_path}")

    if group_name not in motion_config['groups']:
        available = ', '.join(motion_config['groups'].keys())
        raise ValueError(f"Group '{group_name}' not found. Available groups: {available}")

    group = motion_config['groups'][group_name]
    file_paths = group.get('files', [])

    if not file_paths:
        raise ValueError(f"Group '{group_name}' has no files defined")

    # Extract basenames from paths and verify files exist
    file_list = []
    data_dir = Path(config["data"]["data_dir"])

    for file_path in file_paths:
        file_path = Path(file_path)
        basename = file_path.name

        # Check if file exists in data_dir
        full_path = data_dir / basename
        if not full_path.exists():
            # Try the path as-is if it's not in data_dir
            if Path(file_path).exists():
                full_path = Path(file_path)
                basename = full_path.name
            else:
                print(f"Warning: File not found: {basename} (skipping)")
                continue

        file_list.append(basename)

    if not file_list:
        raise ValueError(f"No valid files found for group '{group_name}'")

    print(f"Loaded motion group '{group_name}': {group.get('description', 'No description')}")
    print(f"Files to process: {file_list}")

    return file_list


def inference(
    checkpoint_path: str,
    output_dir: str,
    bridge_output_dir: str,
    window_stride: Optional[int],
    height: int,
    width: int,
    fps: int,
    seed: int,
    interp_frames: int,
    motion_config: Optional[str] = None,
    motion_group: Optional[str] = None,
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

    # Load file list from motion group if specified
    file_list = None
    if motion_config and motion_group:
        file_list = load_motion_group(motion_config, motion_group)
        # Create subdirectory for motion group
        output_dir = str(Path(output_dir) / motion_group)
        bridge_output_dir = str(Path(bridge_output_dir) / motion_group)
        print(f"Using motion group output directories:")
        print(f"  Videos: {output_dir}")
        print(f"  Bridge: {bridge_output_dir}")

    dataset = PoseDatasetV3Continuous(window_stride=window_stride, file_list=file_list)
    os.makedirs(output_dir, exist_ok=True)

    accumulated_sequences: Dict[str, List[np.ndarray]] = {}
    accumulated_keypoints: Dict[str, List[np.ndarray]] = {}
    accumulated_keypoints_with_conf: Dict[str, List[np.ndarray]] = {}
    accumulated_subsets: Dict[str, List[np.ndarray]] = {}
    filename_order: List[str] = []

    with open(config['data']['ref_img_pose_im'], 'rb') as fp:
        ref_pixels = pickle.load(fp)

    for sample_index in tqdm(range(len(dataset)), desc="Samples", unit="clip"):
        masked_sequence, input_mask, _, meta_info = dataset[sample_index]
        filename = meta_info.get("filename", f"sample_{sample_index}")
        norm_params = meta_info.get("norm_params")
        subset_list = meta_info.get("subset")
        confidence_sequence = meta_info.get("confidence")

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

        confidence_full = None
        if confidence_sequence is not None:
            confidence_full = np.asarray(confidence_sequence, dtype=np.float32)
            if confidence_full.ndim == 3 and confidence_full.shape[-1] == 1:
                confidence_full = confidence_full[..., 0]
            if confidence_full.shape[0] != reconstructed_full.shape[0] or confidence_full.shape[1] != reconstructed_full.shape[1]:
                confidence_full = None
        if confidence_full is None:
            confidence_full = np.ones(
                (reconstructed_full.shape[0], reconstructed_full.shape[1]),
                dtype=np.float32,
            )

        reconstructed_with_conf = np.concatenate(
            [reconstructed_full, confidence_full[..., np.newaxis]],
            axis=-1,
        )

        subset = to_numpy_subset(subset_list, reconstructed_full.shape[0])

        frames = sequence_to_frames_mm(
            reconstructed_with_conf,
            subset,
            ref_pixels.shape[1],
            ref_pixels.shape[2],
        )

        if filename not in accumulated_sequences:
            accumulated_sequences[filename] = []
            accumulated_keypoints[filename] = []
            accumulated_keypoints_with_conf[filename] = []
            accumulated_subsets[filename] = []
            filename_order.append(filename)

        accumulated_sequences[filename].append(frames)
        accumulated_keypoints[filename].append(reconstructed_full)
        accumulated_keypoints_with_conf[filename].append(reconstructed_with_conf)
        accumulated_subsets[filename].append(subset)

    os.makedirs(bridge_output_dir, exist_ok=True)
    file_entries: List[Dict[str, np.ndarray]] = []
    for filename in tqdm(filename_order, desc="Aggregating clips", unit="file"):
        frame_chunks = accumulated_sequences[filename]
        keypoint_chunks = accumulated_keypoints[filename]
        keypoints_conf_chunks = accumulated_keypoints_with_conf[filename]
        subset_chunks = accumulated_subsets[filename]

        video_poses = np.concatenate(frame_chunks, axis=0)
        keypoints_xy_full = np.concatenate(keypoint_chunks, axis=0)
        keypoints_with_conf_full = np.concatenate(keypoints_conf_chunks, axis=0)
        subset_full = np.concatenate(subset_chunks, axis=0)

        file_entries.append(
            {
                "name": filename,
                "video_poses": video_poses,
                "keypoints_xy": keypoints_xy_full,
                "keypoints_with_conf": keypoints_with_conf_full,
                "subset": subset_full,
            }
        )

    # Compute global target head size (median across all videos)
    print("Computing global target head size...")
    all_widths = []
    all_heights = []
    for entry in file_entries:
        w, h = compute_head_bbox_size(entry["keypoints_xy"])
        if w > 0 and h > 0:
            all_widths.append(w)
            all_heights.append(h)

    if all_widths:
        target_head_width = float(np.median(all_widths))
        target_head_height = float(np.median(all_heights))
        print(f"Target head size: width={target_head_width:.4f}, height={target_head_height:.4f}")

        # Normalize head size for all entries
        print("Normalizing head sizes...")
        for entry in tqdm(file_entries, desc="Normalizing heads", unit="file"):
            # Normalize keypoints_xy
            entry["keypoints_xy"] = normalize_head_size(
                entry["keypoints_xy"],
                target_head_width,
                target_head_height,
            )

            # Update keypoints_with_conf (preserve confidence values)
            entry["keypoints_with_conf"][:, :, :2] = entry["keypoints_xy"]

            # Re-render frames with normalized keypoints
            entry["video_poses"] = sequence_to_frames_mm(
                entry["keypoints_with_conf"],
                entry["subset"],
                ref_pixels.shape[1],
                ref_pixels.shape[2],
            )
    else:
        print("Warning: Could not compute global head size, skipping normalization")

    total_pose_pixels: List[torch.Tensor] = []

    for file_idx, entry in enumerate(tqdm(file_entries, desc="Exporting outputs", unit="file")):
        filename = str(entry["name"])
        video_poses = entry["video_poses"]
        pose_pixels = np.concatenate([np.expand_dims(ref_pixels, axis=0), video_poses], axis=0)
        pose_pixels = torch.from_numpy(pose_pixels.copy()) / 127.5 - 1

        # Generate interpolation frames if needed
        transition_frames_np = None
        if file_idx < len(file_entries) - 1:
            next_entry = file_entries[file_idx + 1]
            transition_frames_np = generate_interpolation_frames(
                entry, next_entry, interp_frames, ref_pixels.shape[1], ref_pixels.shape[2]
            )
            if transition_frames_np is not None:
                pose_pixels_interp = torch.from_numpy(transition_frames_np.copy()) / 127.5 - 1
                pose_pixels = torch.cat([pose_pixels, pose_pixels_interp], dim=0)

        # Save bridge pickle file
        export_dict = {
            'pose_pixels': pose_pixels,
            'keypoints_xy': entry["keypoints_xy"],
        }
        output_path = Path(bridge_output_dir) / filename
        with open(output_path, 'wb') as fp:
            pickle.dump(export_dict, fp)
        print(f"[{filename}] saved {pose_pixels.shape[0]} frames to {output_path}, shape = {export_dict['pose_pixels'].shape}")

        # Accumulate for total pickle
        if file_idx == 0:
            total_pose_pixels.append(pose_pixels)
        else:
            total_pose_pixels.append(pose_pixels[1:])

        # Export main video
        video_filename = Path(filename).with_suffix(".mp4").name
        video_output_path = Path(output_dir) / video_filename
        export_video_from_frames(
            video_poses,
            video_output_path,
            fps=fps,
            height=height,
            width=width,
        )
        print(f"[{filename}] exported video to {video_output_path}")

        # Export interpolation video if exists
        if transition_frames_np is not None:
            interp_video_path = Path(output_dir) / f"{Path(video_filename).stem}_interp.mp4"
            export_video_from_frames(
                transition_frames_np,
                interp_video_path,
                fps=fps,
                height=height,
                width=width,
            )
            print(f"[{filename}] exported interpolation video to {interp_video_path}")

    if total_pose_pixels:
        # Save total pickle with group name
        if motion_group:
            total_filename = f"total_{motion_group}.pkl"
        else:
            total_filename = "total_20.pkl"
        total_path = Path(bridge_output_dir) / total_filename
        concatenated = torch.cat(total_pose_pixels, dim=0)
        with open(total_path, 'wb') as fp:
            pickle.dump({'pose_pixels': concatenated}, fp)
        print(f"[{total_filename}] saved concatenated pose_pixels with shape {concatenated.shape} to {total_path}")

        # Generate total.mp4 with all frames concatenated
        total_video_frames = []
        for file_idx, entry in enumerate(file_entries):
            total_video_frames.append(entry["video_poses"])

            # Add interpolation frames between consecutive videos
            if file_idx < len(file_entries) - 1:
                transition_frames = generate_interpolation_frames(
                    entry, file_entries[file_idx + 1], interp_frames,
                    ref_pixels.shape[1], ref_pixels.shape[2]
                )
                if transition_frames is not None:
                    total_video_frames.append(transition_frames)

        if total_video_frames:
            total_frames_concat = np.concatenate(total_video_frames, axis=0)
            total_video_path = Path(output_dir) / "total.mp4"
            export_video_from_frames(
                total_frames_concat,
                total_video_path,
                fps=fps,
                height=height,
                width=width,
            )
            print(f"[total.mp4] exported concatenated video with {total_frames_concat.shape[0]} frames to {total_video_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge window reconstructions into one mm-style video per original clip."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/video/test_interpolate2",
        help="Directory to store merged videos.",
    )
    parser.add_argument(
        "--bridge-output-dir",
        type=str,
        default="output/bridge/test_interpolate2",
        help="Directory to store bridge pickle files.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=None,
        help="Stride (in frames) between windows; defaults to sequence length for non-overlap.",
    )
    parser.add_argument("--height", type=int, default=1080, help="Output video height.")
    parser.add_argument("--width", type=int, default=1080, help="Output video width.")
    parser.add_argument("--fps", type=int, default=5, help="Output video frames per second.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument(
        "--interp-frames",
        type=int,
        default=10,
        help="Number of in-between frames to synthesize per pair using XY interpolation.",
    )
    parser.add_argument(
        "--motion-config",
        type=str,
        default=None,
        help="Path to TOML file containing motion group configurations.",
    )
    parser.add_argument(
        "--motion-group",
        type=str,
        default=None,
        help="Name of motion group to process from the config file. Use 'all' to process all groups.",
    )
    args = parser.parse_args()
    if args.interp_frames < 0:
        parser.error("--interp-frames must be non-negative.")
    if (args.motion_config is None) != (args.motion_group is None):
        parser.error("--motion-config and --motion-group must be used together")
    return args


if __name__ == "__main__":
    args = parse_args()

    # Handle 'all' groups option
    if args.motion_config and args.motion_group == "all":
        all_groups = load_motion_groups(args.motion_config)
        print(f"Processing all groups: {list(all_groups.keys())}")

        for group_name in all_groups.keys():
            print(f"\n{'='*60}")
            print(f"Processing group: {group_name}")
            print(f"{'='*60}\n")

            inference(
                checkpoint_path=args.checkpoint,
                output_dir=args.output_dir,
                bridge_output_dir=args.bridge_output_dir,
                window_stride=args.window_stride,
                height=args.height,
                width=args.width,
                fps=args.fps,
                seed=args.seed,
                interp_frames=args.interp_frames,
                motion_config=args.motion_config,
                motion_group=group_name,
            )

        # Collect all total files into a single directory
        print(f"\n{'='*60}")
        print(f"Collecting all total files...")
        print(f"{'='*60}\n")

        import shutil
        all_totals_dir = Path(args.bridge_output_dir) / "all_totals"
        os.makedirs(all_totals_dir, exist_ok=True)

        for group_name in all_groups.keys():
            source_file = Path(args.bridge_output_dir) / group_name / f"total_{group_name}.pkl"
            if source_file.exists():
                dest_file = all_totals_dir / f"total_{group_name}.pkl"
                shutil.copy2(source_file, dest_file)
                print(f"Copied {source_file.name} to {all_totals_dir}")

        print(f"\nAll total files collected in: {all_totals_dir}")
        print(f"\n{'='*60}")
        print(f"All groups processed successfully!")
        print(f"{'='*60}")
    else:
        inference(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            bridge_output_dir=args.bridge_output_dir,
            window_stride=args.window_stride,
            height=args.height,
            width=args.width,
            fps=args.fps,
            seed=args.seed,
            interp_frames=args.interp_frames,
            motion_config=args.motion_config,
            motion_group=args.motion_group,
        )
