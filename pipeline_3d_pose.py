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
    
Note: By default, automatically loads config/pipeline_config.yml if it exists.
      Use --config_file to specify a different config file.
"""

import sys
import os
import argparse
import numpy as np
import yaml

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
                       help='Do not save/load camera pose (estimate per video) [DEPRECATED: use --reestimate_pose]')
    parser.add_argument('--reestimate_pose', action='store_true',
                       help='Force re-estimation of camera pose (ignore saved pose if exists)')
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to YAML config file (e.g., config/custom_config.yml). If not provided, automatically loads config/pipeline_config.yml if it exists, otherwise uses defaults.')
    
    args = parser.parse_args()
    
    # Validate video_path
    if not os.path.exists(args.video_path):
        print(f"✗ Error: Video path does not exist: {args.video_path}")
        return
    
    video_path = os.path.abspath(args.video_path)
    video_name = os.path.basename(video_path)
    data_dir = os.path.dirname(video_path)
    
    # Verify structure: video_path should contain keypoints JSON files
    # Video files are optional (only needed for debug visualization)
    json_files = [f for f in os.listdir(video_path) if f.endswith('.json')]
    video_files = [f for f in os.listdir(video_path) if f.endswith(('.mp4', '.avi'))]
    
    if len(json_files) < 2:
        print(f"✗ Error: Expected at least 2 JSON keypoint files in {video_path}, found {len(json_files)}")
        return
    
    # Video files are optional - warn if not found but don't fail
    if len(video_files) < 2:
        print(f"⚠ Warning: Found {len(video_files)} video files. Debug visualization will be skipped.")
        print(f"  (Videos are optional - camera pose estimation only requires JSON keypoint files)")
    else:
        print(f"✓ Found {len(video_files)} video files and {len(json_files)} JSON files")
    
    print(f"✓ Using video path: {video_path}")
    
    # Initialize logger
    log_dir = os.path.join(data_dir, 'processed', video_name, 'logs')
    logger = PipelineLogger(log_dir, video_name, log_level=20)  # INFO level (video_name used for log filename)
    
    logger.info("="*70)
    logger.info("3D Pose Estimation Pipeline")
    logger.info("="*70)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Video name: {video_name}")
    logger.info(f"World camera: {'side' if args.world_cam_id == 1 else 'front'}")
    logger.info(f"Log file: {logger.get_log_file()}")
    logger.info("="*70)
    
    # Load configuration
    # Default behavior: automatically load config/pipeline_config.yml if it exists
    # User can override with --config_file flag
    default_config_path = os.path.join(os.path.dirname(__file__), 'config', 'pipeline_config.yml')
    
    # Determine which config file to use
    if args.config_file:
        # User explicitly provided a config file (override default)
        config_file_path = args.config_file
    elif os.path.exists(default_config_path):
        # Use default config file if it exists
        config_file_path = default_config_path
    else:
        # No config file available, use defaults
        config_file_path = None
    
    # Start with default config
    config = create_default_config()
    
    # Load and merge config from file if available
    if config_file_path:
        if os.path.exists(config_file_path):
            with open(config_file_path, 'r') as f:
                file_config = yaml.safe_load(f)
                # Merge file config into default config
                for key, value in file_config.items():
                    if isinstance(value, dict) and key in config:
                        config[key].update(value)
                    else:
                        config[key] = value
            if logger:
                if config_file_path == default_config_path:
                    logger.info(f"Loaded configuration from default file: {config_file_path}")
                else:
                    logger.info(f"Loaded configuration from: {config_file_path}")
        else:
            if logger:
                logger.warning(f"Config file not found: {config_file_path}, using defaults")
    else:
        if logger:
            logger.info("Using default configuration (config/pipeline_config.yml not found)")
    
    # Override reestimate_pose from command line if provided
    if args.reestimate_pose:
        if 'camera_pose' not in config:
            config['camera_pose'] = {}
        config['camera_pose']['reestimate_pose'] = True
        if logger:
            logger.info("Command-line flag --reestimate_pose: forcing camera pose re-estimation")
    
    # Initialize dataset
    dataset = TwoViewCustom(None, None, config, video_path=video_path)
    
    # Process frames
    if not args.skip_processing:
        results_dir = process_all_frames(
            dataset, config, 
            wrld_cam_id=args.world_cam_id,
            use_saved_pose=not args.no_save_pose,
            use_single_pose=True,
            logger=logger
        )
    else:
        results_dir = os.path.join(data_dir, 'processed', video_name, 'results')
        if not os.path.exists(results_dir):
            logger.error("Results directory not found! Run without --skip_processing first.")
            return
    
    # Create visualizations
    create_3d_animation = config.get('visualization', {}).get('create_3d_animation', True)
    if not args.skip_visualization and create_3d_animation:
        logger.info("Creating 3D visualization video")
        create_3d_visualization(video_path, fps=args.fps, config=config, logger=logger)
    elif not create_3d_animation:
        logger.info("Skipping 3D animation creation (disabled in config)")
    
    # Create debug visualization only if video files are available
    video_files = [f for f in os.listdir(video_path) if f.endswith(('.mp4', '.avi'))]
    if len(video_files) >= 2:
        logger.info("Creating DEBUG visualization (3D views + 2D keypoints: Original vs Reprojected)")
        # Get number of threads for debug visualization (default: 8)
        num_threads = config.get('visualization', {}).get('debug_num_threads', 8)
        debug_file = create_debug_visualization(video_path, fps=args.fps, config=config, 
                                              logger=logger, num_threads=num_threads)
        if debug_file:
            logger.info(f"Debug visualization saved: {debug_file}")
        else:
            logger.warning("Debug visualization could not be created")
    else:
        logger.info("Skipping debug visualization (video files not found - only JSON keypoint files provided)")
    
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
