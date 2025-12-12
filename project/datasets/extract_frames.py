"""
Extract frames from videos for dataset preparation
Supports multiple video formats and customizable extraction parameters
"""

import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from typing import List, Tuple

def extract_frames_from_video(video_path: Path, output_dir: Path,
                            fps: float = 1.0, start_time: float = 0.0,
                            end_time: float = None, prefix: str = None) -> List[Path]:
    """
    Extract frames from a single video

    Args:
        video_path: Path to input video
        output_dir: Output directory for frames
        fps: Frames per second to extract
        start_time: Start time in seconds
        end_time: End time in seconds (None for full video)
        prefix: Prefix for frame filenames

    Returns:
        List of extracted frame paths
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return []

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    # Set extraction parameters
    frame_interval = int(video_fps / fps) if fps > 0 else 1
    start_frame = int(start_time * video_fps)
    end_frame = int(end_time * video_fps) if end_time else total_frames

    # Set video position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    extracted_frames = []
    frame_count = start_frame

    prefix = prefix or video_path.stem

    while frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame
        frame_filename = f"{prefix}_{frame_count:04d}.jpg"
        frame_path = output_dir / frame_filename
        cv2.imwrite(str(frame_path), frame)
        extracted_frames.append(frame_path)

        # Skip frames
        frame_count += frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

    cap.release()
    return extracted_frames

def process_video_batch(video_paths: List[Path], output_base_dir: Path,
                       fps: float = 1.0, max_workers: int = 4) -> List[Tuple[Path, List[Path]]]:
    """
    Process multiple videos in parallel

    Args:
        video_paths: List of video paths
        output_base_dir: Base output directory
        fps: Frames per second to extract
        max_workers: Maximum number of worker threads

    Returns:
        List of (video_path, frame_paths) tuples
    """
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for video_path in video_paths:
            # Create output directory for this video
            video_output_dir = output_base_dir / video_path.stem
            video_output_dir.mkdir(parents=True, exist_ok=True)

            future = executor.submit(extract_frames_from_video, video_path, video_output_dir, fps)
            futures.append((video_path, future))

        for video_path, future in tqdm(futures, desc="Processing videos"):
            try:
                frame_paths = future.result()
                results.append((video_path, frame_paths))
            except Exception as e:
                print(f"Error processing {video_path}: {e}")

    return results

def extract_frames_recursive(input_dir: Path, output_dir: Path,
                           video_extensions: List[str] = None,
                           fps: float = 1.0, max_workers: int = 4):
    """
    Recursively extract frames from all videos in a directory

    Args:
        input_dir: Input directory containing videos
        output_dir: Output directory for frames
        video_extensions: List of video extensions to process
        fps: Frames per second to extract
        max_workers: Maximum number of worker threads
    """
    if video_extensions is None:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']

    # Find all video files
    video_paths = []
    for ext in video_extensions:
        video_paths.extend(list(input_dir.rglob(f'*{ext}')))

    print(f"Found {len(video_paths)} video files")

    if not video_paths:
        print("No video files found!")
        return

    # Process videos
    results = process_video_batch(video_paths, output_dir, fps, max_workers)

    # Print summary
    total_frames = sum(len(frame_paths) for _, frame_paths in results)
    print("\nExtraction completed!")
    print(f"Processed {len(results)} videos")
    print(f"Extracted {total_frames} frames")

def main():
    parser = argparse.ArgumentParser(description='Extract frames from videos')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing videos')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for extracted frames')
    parser.add_argument('--fps', type=float, default=1.0,
                       help='Frames per second to extract')
    parser.add_argument('--extensions', nargs='+',
                       default=['.mp4', '.avi', '.mov', '.mkv'],
                       help='Video file extensions to process')
    parser.add_argument('--max_workers', type=int, default=4,
                       help='Maximum number of worker threads')
    parser.add_argument('--recursive', action='store_true',
                       help='Process videos recursively')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.recursive:
        extract_frames_recursive(input_dir, output_dir, args.extensions,
                               args.fps, args.max_workers)
    else:
        # Process videos in input directory only
        video_paths = []
        for ext in args.extensions:
            video_paths.extend(list(input_dir.glob(f'*{ext}')))

        if video_paths:
            results = process_video_batch(video_paths, output_dir, args.fps, args.max_workers)
            total_frames = sum(len(frame_paths) for _, frame_paths in results)
            print(f"Extracted {total_frames} frames from {len(results)} videos")
        else:
            print("No video files found in input directory")

if __name__ == '__main__':
    main()
