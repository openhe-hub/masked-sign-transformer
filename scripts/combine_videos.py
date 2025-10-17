import cv2
import os
import argparse
import natsort

def combine_videos(input_folder, output_file):
    """
    Combines all MP4 files in a given folder into a single MP4 file.

    Args:
        input_folder (str): The path to the folder containing the MP4 files.
        output_file (str): The path to save the combined MP4 file.
    """
    # Get a list of all mp4 files in the input folder
    video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]
    if not video_files:
        print("No MP4 files found in the specified folder.")
        return

    # Sort the files naturally to handle numbers correctly (e.g., result_1.mp4, result_10.mp4)
    video_files = natsort.natsorted(video_files)
    
    video_paths = [os.path.join(input_folder, f) for f in video_files]

    # Read the first video to get properties (width, height, fps)
    first_video = cv2.VideoCapture(video_paths[0])
    if not first_video.isOpened():
        print(f"Error opening video file: {video_paths[0]}")
        return

    width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = first_video.get(cv2.CAP_PROP_FPS)
    first_video.release()

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print("Error: Could not open video writer.")
        return

    print(f"Combining {len(video_paths)} videos into {output_file}...")

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open {video_path}. Skipping.")
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_writer.write(frame)
        
        cap.release()
        print(f"Processed {video_path}")

    video_writer.release()
    print(f"Successfully combined videos into {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple MP4 videos from a folder into a single video.")
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Path to the folder containing the MP4 files.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to the output combined MP4 file.')
    
    args = parser.parse_args()

    combine_videos(args.input_folder, args.output_file)
