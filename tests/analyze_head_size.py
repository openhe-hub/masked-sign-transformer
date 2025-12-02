import argparse
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def compute_bbox_size(keypoints: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute bounding box size for face keypoints (indices 18-86).

    Args:
        keypoints: (num_keypoints, 2) array of (x, y) coordinates

    Returns:
        (width, height, area) of the bounding box
    """
    face_kpts = keypoints[18:86, :]  # Face keypoints

    # Remove invalid points (e.g., zeros or NaNs)
    valid_mask = ~(np.isnan(face_kpts).any(axis=1) | (face_kpts == 0).all(axis=1))
    if not valid_mask.any():
        return 0.0, 0.0, 0.0

    valid_kpts = face_kpts[valid_mask]

    x_min, y_min = valid_kpts.min(axis=0)
    x_max, y_max = valid_kpts.max(axis=0)

    width = x_max - x_min
    height = y_max - y_min
    area = width * height

    return width, height, area


def analyze_head_sizes(pickle_files: List[Path], output_dir: Path):
    """Analyze head size variations across multiple pickle files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_widths = []
    all_heights = []
    all_areas = []
    file_boundaries = [0]  # Track where each file starts

    for pkl_file in pickle_files:
        print(f"Processing {pkl_file.name}...")
        with open(pkl_file, 'rb') as fp:
            data = pickle.load(fp)

        # Expecting keypoints_xy: (num_frames, num_keypoints, 2)
        if 'keypoints_xy' in data:
            keypoints_seq = data['keypoints_xy']
        else:
            print(f"Warning: {pkl_file.name} doesn't contain 'keypoints_xy', skipping.")
            continue

        widths = []
        heights = []
        areas = []

        for frame_idx in range(keypoints_seq.shape[0]):
            kpts = keypoints_seq[frame_idx]
            w, h, a = compute_bbox_size(kpts)
            widths.append(w)
            heights.append(h)
            areas.append(a)

        all_widths.extend(widths)
        all_heights.extend(heights)
        all_areas.extend(areas)
        file_boundaries.append(len(all_widths))

    all_widths = np.array(all_widths)
    all_heights = np.array(all_heights)
    all_areas = np.array(all_areas)

    # Print statistics
    print("\n=== Head Size Statistics ===")
    print(f"Width  - Mean: {all_widths.mean():.2f}, Std: {all_widths.std():.2f}, "
          f"Min: {all_widths.min():.2f}, Max: {all_widths.max():.2f}")
    print(f"Height - Mean: {all_heights.mean():.2f}, Std: {all_heights.std():.2f}, "
          f"Min: {all_heights.min():.2f}, Max: {all_heights.max():.2f}")
    print(f"Area   - Mean: {all_areas.mean():.2f}, Std: {all_areas.std():.2f}, "
          f"Min: {all_areas.min():.2f}, Max: {all_areas.max():.2f}")

    # Calculate coefficient of variation (CV)
    width_cv = (all_widths.std() / all_widths.mean()) * 100
    height_cv = (all_heights.std() / all_heights.mean()) * 100
    area_cv = (all_areas.std() / all_areas.mean()) * 100
    print(f"\nCoefficient of Variation (%):")
    print(f"Width: {width_cv:.2f}%, Height: {height_cv:.2f}%, Area: {area_cv:.2f}%")

    # Plot time series
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Width over time
    axes[0].plot(all_widths, linewidth=0.8, alpha=0.7)
    axes[0].axhline(all_widths.mean(), color='r', linestyle='--', label='Mean')
    axes[0].set_ylabel('Width (pixels)')
    axes[0].set_title('Face Bounding Box Width Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Height over time
    axes[1].plot(all_heights, linewidth=0.8, alpha=0.7)
    axes[1].axhline(all_heights.mean(), color='r', linestyle='--', label='Mean')
    axes[1].set_ylabel('Height (pixels)')
    axes[1].set_title('Face Bounding Box Height Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Area over time
    axes[2].plot(all_areas, linewidth=0.8, alpha=0.7)
    axes[2].axhline(all_areas.mean(), color='r', linestyle='--', label='Mean')
    axes[2].set_ylabel('Area (pixels²)')
    axes[2].set_xlabel('Frame Index')
    axes[2].set_title('Face Bounding Box Area Over Time')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Add vertical lines for file boundaries
    for ax in axes:
        for boundary in file_boundaries[1:-1]:
            ax.axvline(boundary, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plot_path = output_dir / 'head_size_analysis.png'
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")

    # Plot histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(all_widths, bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Width (pixels)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Width Distribution')
    axes[0].axvline(all_widths.mean(), color='r', linestyle='--', label='Mean')
    axes[0].legend()

    axes[1].hist(all_heights, bins=50, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Height (pixels)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Height Distribution')
    axes[1].axvline(all_heights.mean(), color='r', linestyle='--', label='Mean')
    axes[1].legend()

    axes[2].hist(all_areas, bins=50, alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Area (pixels²)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Area Distribution')
    axes[2].axvline(all_areas.mean(), color='r', linestyle='--', label='Mean')
    axes[2].legend()

    plt.tight_layout()
    hist_path = output_dir / 'head_size_distribution.png'
    plt.savefig(hist_path, dpi=150)
    print(f"Histogram saved to {hist_path}")

    # Save statistics to file
    stats_path = output_dir / 'head_size_stats.txt'
    with open(stats_path, 'w') as f:
        f.write("=== Head Size Statistics ===\n\n")
        f.write(f"Total frames analyzed: {len(all_widths)}\n")
        f.write(f"Number of files: {len(pickle_files)}\n\n")
        f.write(f"Width  - Mean: {all_widths.mean():.2f}, Std: {all_widths.std():.2f}, "
                f"Min: {all_widths.min():.2f}, Max: {all_widths.max():.2f}\n")
        f.write(f"Height - Mean: {all_heights.mean():.2f}, Std: {all_heights.std():.2f}, "
                f"Min: {all_heights.min():.2f}, Max: {all_heights.max():.2f}\n")
        f.write(f"Area   - Mean: {all_areas.mean():.2f}, Std: {all_areas.std():.2f}, "
                f"Min: {all_areas.min():.2f}, Max: {all_areas.max():.2f}\n\n")
        f.write(f"Coefficient of Variation:\n")
        f.write(f"Width: {width_cv:.2f}%\n")
        f.write(f"Height: {height_cv:.2f}%\n")
        f.write(f"Area: {area_cv:.2f}%\n")
    print(f"Statistics saved to {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze head size variations in pose keypoints")
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to bridge output directory or specific pickle file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/analysis/head_size',
        help='Output directory for analysis results'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    # Collect pickle files
    if input_path.is_file():
        pickle_files = [input_path]
    elif input_path.is_dir():
        pickle_files = sorted(input_path.glob('*.pkl'))
        pickle_files = [f for f in pickle_files if f.name != 'total_20.pkl']  # Skip total file
    else:
        raise ValueError(f"Invalid input path: {input_path}")

    if not pickle_files:
        raise ValueError(f"No pickle files found in {input_path}")

    print(f"Found {len(pickle_files)} pickle files")
    analyze_head_sizes(pickle_files, output_dir)


if __name__ == "__main__":
    main()
