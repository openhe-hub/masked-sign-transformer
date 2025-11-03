#!/usr/bin/env python3
"""
Batch normalization script using CoSign-style normalization.
This script processes all .pkl files in an input directory and saves the 
normalized files to an output directory.

Usage:
    python batch_norm_cosign.py <input_directory> <output_directory>
"""

import os
import sys
import argparse
from norm_cosign import normalize_pkl

def batch_normalize(input_dir, output_dir):
    """
    Normalizes all .pkl files from input_dir and saves them to output_dir.

    Args:
        input_dir (str): The directory containing the .pkl files to process.
        output_dir (str): The directory where normalized .pkl files will be saved.
    """
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found at '{input_dir}'")
        return

    if not os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' does not exist. Creating it.")
        os.makedirs(output_dir)

    pkl_files = [f for f in os.listdir(input_dir) if f.endswith('.pkl')]

    if not pkl_files:
        print(f"No .pkl files found in '{input_dir}'.")
        return

    print(f"Found {len(pkl_files)} .pkl files to process.")

    for i, file_name in enumerate(pkl_files):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        
        print(f"\n--- Processing file {i+1}/{len(pkl_files)}: {file_name} ---")
        
        try:
            normalize_pkl(input_path, output_path)
        except Exception as e:
            print(f"Could not process file {file_name}. Error: {e}")
            # Optionally, continue to the next file
            continue

    print("\nBatch processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch normalize pose keypoints from .pkl files."
    )
    parser.add_argument(
        "input_dir", 
        type=str, 
        help="Directory containing the input .pkl files."
    )
    parser.add_argument(
        "output_dir", 
        type=str, 
        help="Directory to save the normalized .pkl files."
    )

    args = parser.parse_args()

    batch_normalize(args.input_dir, args.output_dir)
