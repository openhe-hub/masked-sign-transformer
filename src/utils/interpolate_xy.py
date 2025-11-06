#!/usr/bin/env python3
"""CLI utility to interpolate between two 2D pose frames with 128 joints."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np

def _normalize_pose(array: Any) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        if arr.size != 256:
            raise ValueError(
                f"Expected flattened pose with 256 values, got {arr.size} values."
            )
        arr = arr.reshape(128, 2)
    elif arr.ndim == 2:
        if arr.shape == (2, 128):
            arr = arr.T
        elif arr.shape != (128, 2):
            raise ValueError(
                f"Expected pose with shape (128, 2) or (2, 128), got {arr.shape}."
            )
    else:
        raise ValueError(f"Expected array with 1 or 2 dimensions, got {arr.ndim}.")
    return arr

def load_pose(file_path: Path) -> np.ndarray:
    file_path = file_path.expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Pose file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        data = np.load(file_path)
    elif suffix in {".txt", ".csv"}:
        delimiter = "," if suffix == ".csv" else None
        data = np.loadtxt(file_path, delimiter=delimiter)
    elif suffix == ".json":
        import json

        with file_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        raise ValueError(
            "Unsupported file type. Use .npy, .txt, .csv, or .json pose files."
        )

    return _normalize_pose(data)

def interpolate_xy(
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    num_steps: int,
    drop_endpoints: bool = False,
) -> np.ndarray:
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2 to include start and end poses.")

    start = _normalize_pose(start_xy)
    end = _normalize_pose(end_xy)

    alphas = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)
    poses = (1.0 - alphas[:, None, None]) * start + alphas[:, None, None] * end

    if drop_endpoints:
        if num_steps <= 2:
            raise ValueError("Cannot drop endpoints when num_steps <= 2.")
        poses = poses[1:-1]

    return poses

def save_interpolations(
    interpolated: np.ndarray,
    output_path: Path,
    precision: int = 6,
) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()
    if suffix == ".npy":
        np.save(output_path, interpolated)
    elif suffix in {".txt", ".csv"}:
        delimiter = "," if suffix == ".csv" else " "
        flat = interpolated.reshape(interpolated.shape[0], -1)
        fmt = f"%.{precision}f"
        np.savetxt(output_path, flat, fmt=fmt, delimiter=delimiter)
    else:
        raise ValueError("Output file must use .npy, .txt, or .csv extension.")

    return output_path

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate between two 2D pose frames (128 joints) using linear blending."
        )
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Path to the starting pose (.npy/.txt/.csv/.json).",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="Path to the ending pose (.npy/.txt/.csv/.json).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=15,
        help="Number of interpolation steps (>= 2, inclusive of start/end).",
    )
    parser.add_argument(
        "--drop-endpoints",
        action="store_true",
        help="Drop the original start and end poses from the output sequence.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional output file (.npy/.txt/.csv) for the interpolated sequence.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Decimal precision when writing text output (ignored for .npy).",
    )
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.num_steps < 2:
        parser.error("--num-steps must be at least 2.")
    if args.drop_endpoints and args.num_steps <= 2:
        parser.error("--drop-endpoints requires --num-steps >= 3.")

    start_pose = load_pose(Path(args.start))
    end_pose = load_pose(Path(args.end))

    interpolated = interpolate_xy(
        start_pose,
        end_pose,
        num_steps=args.num_steps,
        drop_endpoints=args.drop_endpoints,
    )

    print("Interpolation complete.")
    print(f"Start pose shape: {start_pose.shape}")
    print(f"End pose shape: {end_pose.shape}")
    print(
        f"Generated {interpolated.shape[0]} frames with per-frame shape {interpolated.shape[1:]}"
    )

    if args.output:
        output_path = save_interpolations(
            interpolated,
            output_path=Path(args.output),
            precision=args.precision,
        )
        print(f"Saved interpolated sequence to {output_path}")
    else:
        sample = interpolated[[0, -1]]
        print("Preview (first and last frames, flattened):")
        print(sample.reshape(sample.shape[0], -1))

if __name__ == "__main__":
    main()
