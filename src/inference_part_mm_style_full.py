import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
import torch
import pickle
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


def inference(
    checkpoint_path: str,
    output_dir: str,
    window_stride: Optional[int],
    height: int,
    width: int,
    fps: int,
    seed: int,
    interp_frames: int,
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

    dataset = PoseDatasetV3Continuous(window_stride=window_stride)
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

    os.makedirs("output/bridge/phoenix", exist_ok=True)
    file_entries: List[Dict[str, np.ndarray]] = []
    for filename in tqdm(filename_order, desc="Aggregating clips", unit="file"):
        frame_chunks = accumulated_sequences[filename]
        keypoint_chunks = accumulated_keypoints[filename]
        keypoints_conf_chunks = accumulated_keypoints_with_conf[filename]
        subset_chunks = accumulated_subsets[filename]

        video_poses = np.concatenate(frame_chunks, axis=0)
        # keypoints_xy_full = np.concatenate(keypoint_chunks, axis=0)
        # keypoints_with_conf_full = np.concatenate(keypoints_conf_chunks, axis=0)
        subset_full = np.concatenate(subset_chunks, axis=0)

        file_entries.append(
            {
                "name": filename,
                "video_poses": video_poses,
                # "keypoints_xy": keypoints_xy_full,
                # "keypoints_with_conf": keypoints_with_conf_full,
                "subset": subset_full,
            }
        )

    # total_pose_pixels: List[torch.Tensor] = []

    for file_idx, entry in enumerate(tqdm(file_entries, desc="Exporting outputs", unit="file")):
        filename = entry["name"]
        video_poses = entry["video_poses"]
        pose_pixels = np.concatenate([np.expand_dims(ref_pixels, axis=0), video_poses], axis=0)
        pose_pixels = torch.from_numpy(pose_pixels.copy()) / 127.5 - 1
        export_dict = {'pose_pixels': pose_pixels}

        transition_frames_np: Optional[np.ndarray] = None
        # if interp_frames > 0 and file_idx < len(file_entries) - 1:
        #     next_entry = file_entries[file_idx + 1]
        #     per_pair_steps = interp_frames + 2

        #     start_pose_xy = entry["keypoints_xy"][-1]
        #     end_pose_xy = next_entry["keypoints_xy"][0]
        #     interpolated = interpolate_xy(
        #         start_pose_xy,
        #         end_pose_xy,
        #         num_steps=per_pair_steps,
        #         drop_endpoints=True,
        #     )

        #     if interpolated.size > 0:
        #         start_conf = entry["keypoints_with_conf"][-1, :, 2]
        #         end_conf = next_entry["keypoints_with_conf"][0, :, 2]
        #         alphas = np.linspace(0.0, 1.0, per_pair_steps, dtype=np.float32)[1:-1]
        #         conf_interp = (
        #             (1.0 - alphas[:, None]) * start_conf[None, :]
        #             + alphas[:, None] * end_conf[None, :]
        #         )
        #         interpolated_with_conf = np.concatenate(
        #             [interpolated, conf_interp[..., np.newaxis]],
        #             axis=-1,
        #         )

        #         subset_template = entry["subset"][-1]
        #         repeated_subset = np.repeat(
        #             subset_template[np.newaxis, ...],
        #             interpolated_with_conf.shape[0],
        #             axis=0,
        #         )
        #         transition_frames_np = sequence_to_frames_mm(
        #             interpolated_with_conf,
        #             repeated_subset,
        #             ref_pixels.shape[1],
        #             ref_pixels.shape[2],
        #         )
        #         pose_pixels_interp = torch.from_numpy(transition_frames_np.copy()) / 127.5 - 1
        #         pose_pixels = torch.cat([pose_pixels, pose_pixels_interp], dim=0)
        #         export_dict['pose_pixels'] = pose_pixels

        output_path = Path("output/bridge/phoenix") / filename
        with open(output_path, 'wb') as fp:
            pickle.dump(export_dict, fp)
        print(f"[{filename}] saved {pose_pixels.shape[0]} frames to {output_path}, shape = {export_dict['pose_pixels'].shape}")

        # if file_idx == 0:
        #     total_pose_pixels.append(pose_pixels)
        # else:
        #     total_pose_pixels.append(pose_pixels[1:])

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

        # if transition_frames_np is not None:
        #     interp_video_path = Path(output_dir) / f"{Path(video_filename).stem}_interp.mp4"
        #     export_video_from_frames(
        #         transition_frames_np,
        #         interp_video_path,
        #         fps=fps,
        #         height=height,
        #         width=width,
        #     )
        #     print(f"[{filename}] exported interpolation video to {interp_video_path}")

    # if total_pose_pixels:
    #     total_path = Path("output/bridge/phoenix") / "total.pkl"
    #     concatenated = torch.cat(total_pose_pixels, dim=0)
    #     with open(total_path, 'wb') as fp:
    #         pickle.dump({'pose_pixels': concatenated}, fp)
    #     print(f"[total.pkl] saved concatenated pose_pixels with shape {concatenated.shape} to {total_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge window reconstructions into one mm-style video per original clip."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/video/phoenix",
        help="Directory to store merged videos.",
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
        default=0,
        help="Number of in-between frames to synthesize per pair using XY interpolation.",
    )
    args = parser.parse_args()
    if args.interp_frames < 0:
        parser.error("--interp-frames must be non-negative.")
    return args


if __name__ == "__main__":
    args = parse_args()
    inference(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        window_stride=args.window_stride,
        height=args.height,
        width=args.width,
        fps=args.fps,
        seed=args.seed,
        interp_frames=args.interp_frames,
    )
