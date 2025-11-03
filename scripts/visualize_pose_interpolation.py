#!/usr/bin/env python3
"""
Visualize interpolation between two random frames in both pose space and latent space.

Compares:
1. Direct pose interpolation (linear interpolation in 50x3 pose space)
2. Latent interpolation (interpolate in 64D latent space, then decode)

Usage:
    python visualize_pose_interpolation.py \
        --checkpoint checkpoints/framewise_ae/phoenix_64d/best_epoch861.ckpt \
        --data_path /path/to/phoenix \
        --num_steps 15 \
        --output_dir visualizations/interpolation
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import random

from phoenix.framewise_ae_phoenix import FramewiseAEPhoenixModule
from pose_data import PoseRepresentationDataset, PoseVisualizer
from pose_data.datasets import PhoenixRawDataset, PhoenixRelativeOffsetRepresentation


def linear_interpolate(start, end, num_steps):
    """Linear interpolation between start and end."""
    alphas = np.linspace(0, 1, num_steps)
    interpolated = []
    for alpha in alphas:
        interpolated.append((1 - alpha) * start + alpha * end)
    return np.array(interpolated)


def slerp(start, end, num_steps):
    """Spherical linear interpolation (for latent space)."""
    # Normalize vectors
    start_norm = start / (np.linalg.norm(start) + 1e-8)
    end_norm = end / (np.linalg.norm(end) + 1e-8)

    # Calculate angle between vectors
    dot = np.clip(np.dot(start_norm, end_norm), -1.0, 1.0)
    theta = np.arccos(dot)

    # If angle is very small, use linear interpolation
    if theta < 0.01:
        return linear_interpolate(start, end, num_steps)

    # SLERP formula
    alphas = np.linspace(0, 1, num_steps)
    interpolated = []
    for alpha in alphas:
        interp = (np.sin((1 - alpha) * theta) / np.sin(theta)) * start + \
                 (np.sin(alpha * theta) / np.sin(theta)) * end
        interpolated.append(interp)

    return np.array(interpolated)


def denormalize_pose_sequence(normalized_poses, topology):
    """Convert relative representation to absolute coordinates."""
    T = len(normalized_poses)
    absolute_poses = np.zeros_like(normalized_poses)

    for t in range(T):
        relative_frame = normalized_poses[t]
        absolute_frame = np.zeros_like(relative_frame)

        # Convert parent-relative to absolute
        for joint_idx in range(topology.num_joints):
            parent_idx = topology.parent_map.get(joint_idx)
            if parent_idx is None:
                absolute_frame[joint_idx] = relative_frame[joint_idx]
            else:
                absolute_frame[joint_idx] = relative_frame[joint_idx] + absolute_frame[parent_idx]

        # Undo neck y-offset
        absolute_frame[:, 1] -= PhoenixRelativeOffsetRepresentation.NECK_Y_OFFSET

        absolute_poses[t] = absolute_frame

    return absolute_poses


def main():
    parser = argparse.ArgumentParser(
        description='Visualize pose vs latent space interpolation'
    )

    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to AE checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to Phoenix data')

    # Interpolation settings
    parser.add_argument('--num_steps', type=int, default=15,
                        help='Number of interpolation steps')
    parser.add_argument('--use_slerp', action='store_true',
                        help='Use SLERP instead of linear for latent interpolation')

    # Sampling
    parser.add_argument('--split', type=str, default='dev',
                        choices=['train', 'dev', 'test'])
    parser.add_argument('--sample_indices', type=int, nargs=2, default=None,
                        help='Specific sample indices (two) to use')
    parser.add_argument('--frame_indices', type=int, nargs=2, default=None,
                        help='Specific frame indices (two) within samples')

    # Visualization
    parser.add_argument('--output_dir', type=str, default='./visualizations/interpolation',
                        help='Output directory')
    parser.add_argument('--fps', type=int, default=10,
                        help='FPS for output GIF')

    # System
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("POSE vs LATENT SPACE INTERPOLATION")
    print("="*60)

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = FramewiseAEPhoenixModule.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.to(args.device)
    print(f"Model loaded! Latent dim: {model.model.total_latent_dim}")

    # Load dataset
    print(f"\nLoading {args.split} data...")
    representation = PhoenixRelativeOffsetRepresentation()
    raw_dataset = PhoenixRawDataset(
        data_path=args.data_path,
        split=args.split,
        max_frames=300,
        filter_oov=False,
        cleanup_glosses=True
    )
    dataset = PoseRepresentationDataset(raw_dataset, representation)
    print(f"Dataset size: {len(dataset)}")

    # Sample two random samples and frames
    if args.sample_indices is not None:
        sample_idx1, sample_idx2 = args.sample_indices
    else:
        sample_idx1, sample_idx2 = random.sample(range(len(dataset)), 2)

    print(f"\nSelected samples: {sample_idx1}, {sample_idx2}")

    # Get samples
    sample1 = dataset[sample_idx1]
    sample2 = dataset[sample_idx2]

    # Select random frames
    if args.frame_indices is not None:
        frame_idx1, frame_idx2 = args.frame_indices
    else:
        frame_idx1 = random.randint(0, len(sample1['pose']) - 1)
        frame_idx2 = random.randint(0, len(sample2['pose']) - 1)

    print(f"Selected frames: {frame_idx1} (from sample {sample_idx1}), {frame_idx2} (from sample {sample_idx2})")

    # Extract frames
    pose1 = sample1['pose'][frame_idx1]  # [50, 3]
    pose2 = sample2['pose'][frame_idx2]  # [50, 3]

    print(f"\nSample 1: {sample1['sample_id']} | Gloss: {sample1['gloss'][:50]}")
    print(f"Sample 2: {sample2['sample_id']} | Gloss: {sample2['gloss'][:50]}")

    # Method 1: Direct pose space interpolation
    print(f"\n[1/2] Interpolating in POSE SPACE (50x3 = 150D)...")
    pose_interp = linear_interpolate(pose1, pose2, args.num_steps)  # [num_steps, 50, 3]
    print(f"  Generated {len(pose_interp)} interpolated poses")

    # Method 2: Latent space interpolation
    print(f"\n[2/2] Interpolating in LATENT SPACE (64D)...")

    # Encode both poses
    with torch.no_grad():
        pose1_tensor = torch.from_numpy(pose1).unsqueeze(0).unsqueeze(0).to(args.device)  # [1, 1, 50, 3]
        pose2_tensor = torch.from_numpy(pose2).unsqueeze(0).unsqueeze(0).to(args.device)  # [1, 1, 50, 3]

        z1_dict = model.model.encode(pose1_tensor, torch.tensor([1]))
        z2_dict = model.model.encode(pose2_tensor, torch.tensor([1]))

        z1 = z1_dict['full'][0, 0].cpu().numpy()  # [64]
        z2 = z2_dict['full'][0, 0].cpu().numpy()  # [64]

    print(f"  Encoded to latent: z1.shape={z1.shape}, z2.shape={z2.shape}")
    print(f"  z1 stats: mean={z1.mean():.4f}, std={z1.std():.4f}")
    print(f"  z2 stats: mean={z2.mean():.4f}, std={z2.std():.4f}")

    # Interpolate in latent space
    if args.use_slerp:
        print("  Using SLERP interpolation...")
        z_interp = slerp(z1, z2, args.num_steps)  # [num_steps, 64]
    else:
        print("  Using linear interpolation...")
        z_interp = linear_interpolate(z1, z2, args.num_steps)  # [num_steps, 64]

    # Decode interpolated latents
    with torch.no_grad():
        z_interp_tensor = torch.from_numpy(z_interp).unsqueeze(0).to(args.device)  # [1, num_steps, 64]

        # Split into body/hand components
        z_split = model.model.split_full_latent(z_interp_tensor)

        # Decode
        latent_interp_tensor = model.model.decode(z_split, torch.tensor([args.num_steps]))
        latent_interp = latent_interp_tensor[0].cpu().numpy()  # [num_steps, 50, 3]

    print(f"  Decoded {len(latent_interp)} poses from latents")

    # Denormalize for visualization
    topology = PhoenixRawDataset.TOPOLOGY
    pose_interp_abs = denormalize_pose_sequence(pose_interp, topology)
    latent_interp_abs = denormalize_pose_sequence(latent_interp, topology)

    # Create visualizer
    skeleton_connections = [
        (0, 1), (1, 2), (2, 3), (3, 29), (1, 5), (5, 6), (6, 8),
        (8, 9), (8, 13), (8, 17), (8, 21), (8, 25),
        (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16),
        (17, 18), (18, 19), (19, 20),
        (21, 22), (22, 23), (23, 24),
        (25, 26), (26, 27), (27, 28),
        (29, 30), (29, 34), (29, 38), (29, 42), (29, 46),
        (30, 31), (31, 32), (32, 33),
        (34, 35), (35, 36), (36, 37),
        (38, 39), (39, 40), (40, 41),
        (42, 43), (43, 44), (44, 45),
        (46, 47), (47, 48), (48, 49),
    ]

    visualizer = PoseVisualizer(
        topology=topology,
        skeleton_connections=skeleton_connections,
        image_size=(640, 640),
        point_radius=4,
        line_thickness=2
    )

    # Create comparison GIF
    output_file = output_dir / f"interp_s{sample_idx1}f{frame_idx1}_to_s{sample_idx2}f{frame_idx2}_{args.num_steps}steps.gif"

    print(f"\nCreating comparison GIF: {output_file.name}")
    visualizer.create_comparison_gif(
        poses_dict={
            'Pose Space': pose_interp_abs,
            'Latent Space': latent_interp_abs
        },
        output_path=str(output_file),
        title=f"Interpolation: Pose vs Latent ({args.num_steps} steps)",
        fps=args.fps,
        max_frames=None,
        show_progress=True,
        layout='horizontal'
    )

    # Save metadata
    metadata_file = output_file.with_suffix('.txt')
    with open(metadata_file, 'w') as f:
        f.write(f"POSE vs LATENT SPACE INTERPOLATION\n")
        f.write(f"="*60 + "\n\n")
        f.write(f"Sample 1:\n")
        f.write(f"  Index: {sample_idx1}\n")
        f.write(f"  ID: {sample1['sample_id']}\n")
        f.write(f"  Frame: {frame_idx1}\n")
        f.write(f"  Gloss: {sample1['gloss']}\n\n")
        f.write(f"Sample 2:\n")
        f.write(f"  Index: {sample_idx2}\n")
        f.write(f"  ID: {sample2['sample_id']}\n")
        f.write(f"  Frame: {frame_idx2}\n")
        f.write(f"  Gloss: {sample2['gloss']}\n\n")
        f.write(f"Interpolation:\n")
        f.write(f"  Steps: {args.num_steps}\n")
        f.write(f"  Method: {'SLERP' if args.use_slerp else 'Linear'}\n")
        f.write(f"  Latent dim: {model.model.total_latent_dim}\n")
        f.write(f"  Pose dim: 150 (50 joints × 3)\n\n")
        f.write(f"Latent stats:\n")
        f.write(f"  z1: mean={z1.mean():.4f}, std={z1.std():.4f}\n")
        f.write(f"  z2: mean={z2.mean():.4f}, std={z2.std():.4f}\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")

    print(f"\n{'='*60}")
    print(f"SUCCESS!")
    print(f"{'='*60}")
    print(f"GIF: {output_file}")
    print(f"Metadata: {metadata_file}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
