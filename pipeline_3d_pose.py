#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-End 3D Pose Estimation Pipeline

This script provides a complete pipeline from videos + 2D keypoints to 3D visualization.
Features:
- Processes videos and 2D keypoints to generate 3D keypoints
- Creates beautiful 3D visualization videos
- Debug mode: Visualizes original vs reprojected 2D keypoints
- Saves camera pose per video for reuse
- Clean, modular design for easy integration

Usage:
    python pipeline_3d_pose.py --video_path /path/to/videos/swing1
    python pipeline_3d_pose.py --video_path /path/to/videos/swing1 --debug
    python pipeline_3d_pose.py --video_path /path/to/videos/swing1 --skip_processing
"""

import sys
import os
import argparse
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.dataset.twoview_custom import TwoViewCustom
from src.pipeline_utils import create_default_config
from src.pipeline_logger import PipelineLogger
from src.pipeline_processing import process_all_frames
from src.pipeline_visualization import create_3d_visualization, create_debug_visualization


def main():
    parser = argparse.ArgumentParser(
        description='End-to-End 3D Pose Estimation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python pipeline_3d_pose.py --video_path /path/to/videos/swing1
  
  # With debug mode
  python pipeline_3d_pose.py --video_path /path/to/videos/swing1 --debug
  
  # Skip processing, only visualize
  python pipeline_3d_pose.py --video_path /path/to/videos/swing1 --skip_processing
  
  # Process multiple videos
  for video in swing1 swing2 swing3; do
    python pipeline_3d_pose.py --video_path /path/to/videos/$video
  done
        """
    )
    
    parser.add_argument('--video_path', type=str, required=True,
                       help='Direct path to video folder containing videos and keypoints JSON (e.g., /path/to/videos/swing1).')
    parser.add_argument('--world_cam_id', type=int, default=1,
                       help='World camera ID (0=front, 1=side, default: 1)')
    parser.add_argument('--skip_processing', action='store_true',
                       help='Skip 3D processing, only create visualization')
    parser.add_argument('--skip_visualization', action='store_true',
                       help='Skip visualization, only process 3D keypoints')
    parser.add_argument('--debug', action='store_true',
                       help='Create debug visualization (2D keypoints comparison)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Output video FPS (default: 30)')
    parser.add_argument('--no_save_pose', action='store_true',
                       help='Do not save/load camera pose (estimate per video)')
    parser.add_argument('--expected_height_cm', type=float, default=None,
                       help='Expected person height in cm (optional, for height calculation only). NOT used for camera pose optimization.')
    parser.add_argument('--height_range_cm', type=float, nargs=2, default=None,
                       metavar=('MIN', 'MAX'),
                       help='Height range in cm (e.g., 160 180). Optional, for height calculation only.')
    
    args = parser.parse_args()
    
    # Validate video_path
    if not os.path.exists(args.video_path):
        print(f"✗ Error: Video path does not exist: {args.video_path}")
        return
    
    video_path = os.path.abspath(args.video_path)
    video_name = os.path.basename(video_path)
    data_dir = os.path.dirname(video_path)
    
    # Verify structure: video_path should contain videos and keypoints JSON files
    video_files = [f for f in os.listdir(video_path) if f.endswith(('.mp4', '.avi'))]
    json_files = [f for f in os.listdir(video_path) if f.endswith('.json')]
    
    if len(video_files) < 2:
        print(f"✗ Error: Expected at least 2 video files in {video_path}, found {len(video_files)}")
        return
    if len(json_files) < 2:
        print(f"✗ Error: Expected at least 2 JSON keypoint files in {video_path}, found {len(json_files)}")
        return
    
    print(f"✓ Using video path: {video_path}")
    
    # Calculate expected height from range if provided (for height calculation only)
    if args.height_range_cm:
        expected_height_m = np.mean(args.height_range_cm) / 100.0
        print(f"Using height range {args.height_range_cm[0]}-{args.height_range_cm[1]} cm (average: {expected_height_m*100:.1f} cm) for height calculation")
    elif args.expected_height_cm:
        expected_height_m = args.expected_height_cm / 100.0
    else:
        expected_height_m = None  # Use original scale from triangulation
    
    # Initialize logger
    log_dir = os.path.join(data_dir, 'processed', video_name, 'logs')
    logger = PipelineLogger(log_dir, video_name, log_level=20)  # INFO level (video_name used for log filename)
    
    logger.info("="*70)
    logger.info("3D Pose Estimation Pipeline")
    logger.info("="*70)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Video name: {video_name}")
    logger.info(f"World camera: {'side' if args.world_cam_id == 1 else 'front'}")
    if expected_height_m:
        logger.info(f"Expected height: {expected_height_m*100:.1f} cm (for height calculation only)")
    else:
        logger.info("Using original author's scale (no height normalization)")
    logger.info(f"Log file: {logger.get_log_file()}")
    logger.info("="*70)
    
    # Initialize dataset
    config = create_default_config()
    dataset = TwoViewCustom(None, None, config, video_path=video_path)
    
    # Process frames
    if not args.skip_processing:
        results_dir = process_all_frames(
            dataset, config, 
            wrld_cam_id=args.world_cam_id,
            use_saved_pose=not args.no_save_pose,
            use_single_pose=True,
            expected_height_m=expected_height_m,
            logger=logger
        )
    else:
        results_dir = os.path.join(data_dir, 'processed', video_name, 'results')
        if not os.path.exists(results_dir):
            logger.error("Results directory not found! Run without --skip_processing first.")
            return
    
    # Create visualizations
    if not args.skip_visualization:
        logger.info("Creating 3D visualization video")
        create_3d_visualization(video_path, fps=args.fps, logger=logger)
    
    # Always create debug visualization (shows 2D keypoint comparison)
    logger.info("Creating DEBUG visualization (2D keypoints: Original vs Reprojected)")
    debug_file = create_debug_visualization(video_path, fps=args.fps, logger=logger)
    if debug_file:
        logger.info(f"Debug visualization saved: {debug_file}")
    else:
        logger.warning("Debug visualization could not be created (video files may be missing)")
    
    logger.info("="*70)
    logger.info("Pipeline Complete!")
    logger.info("="*70)
    logger.info(f"Results: {results_dir}")
    logger.info(f"3D visualization: {os.path.join(results_dir, '3d_keypoints_animation.mp4')}")
    logger.info(f"Debug visualization: {os.path.join(results_dir, 'debug_2d_keypoints_comparison.mp4')}")
    logger.info(f"Log file: {logger.get_log_file()}")
    logger.info("="*70)
    
    print(f"\n✓ Pipeline complete! Check log file: {logger.get_log_file()}")


if __name__ == '__main__':
    main()
