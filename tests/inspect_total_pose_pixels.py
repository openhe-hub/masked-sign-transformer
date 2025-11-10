#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_pose_pixels(path: Path) -> torch.Tensor:
    with path.open("rb") as handle:
        payload: Any = pickle.load(handle)

    if not isinstance(payload, dict) or "pose_pixels" not in payload:
        raise ValueError(f"{path} does not contain a 'pose_pixels' entry.")

    pose_pixels = payload["pose_pixels"]
    if isinstance(pose_pixels, np.ndarray):
        pose_pixels = torch.from_numpy(pose_pixels)
    if not isinstance(pose_pixels, torch.Tensor):
        raise TypeError(f"'pose_pixels' must be a torch.Tensor or np.ndarray, got {type(pose_pixels)}")

    return pose_pixels


def summarize_tensor(tensor: torch.Tensor) -> None:
    shape = tuple(tensor.shape)
    dtype = tensor.dtype
    device = tensor.device
    num_frames = shape[0] if tensor.ndim >= 1 else 0

    min_val = float(tensor.min().item()) if tensor.numel() else float("nan")
    max_val = float(tensor.max().item()) if tensor.numel() else float("nan")
    mean_val = float(tensor.mean().item()) if tensor.numel() else float("nan")
    std_val = float(tensor.std(unbiased=False).item()) if tensor.numel() else float("nan")

    print("=== total.pkl summary ===")
    print(f"Shape      : {shape}")
    print(f"Dtype/Device: {dtype} / {device}")
    print(f"Frames     : {num_frames}")
    print(f"Value range: [{min_val:.4f}, {max_val:.4f}]")
    print(f"Mean/Std   : {mean_val:.4f} / {std_val:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect concatenated pose_pixels inside total.pkl.")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("output/bridge/chatsign_50k_bridge/total.pkl"),
        help="Path to total.pkl (default: output/bridge/chatsign_50k_bridge/total.pkl)",
    )
    args = parser.parse_args()

    path = args.path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"total.pkl not found at {path}")

    pose_pixels = load_pose_pixels(path)
    summarize_tensor(pose_pixels)


if __name__ == "__main__":
    main()
