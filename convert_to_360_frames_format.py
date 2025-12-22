#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert your 360-frame keypoint data to the expected format.
This script helps convert various formats to the format needed for processing.
"""

import json
import os
import argparse
import numpy as np


def convert_single_file_to_frames(input_file, output_dir, cam_name):
    """
    Convert a single JSON file with 360 frames to frame-wise JSON files.
    
    Supports formats:
    1. {'keypoints': [[[x,y],...], [[x,y],...], ...]}  - 360 frames
    2. {'frames': [{'keypoints': [...]}, ...]}  - 360 frames
    """
    print(f"\nConverting {input_file}...")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    frames_data = []
    
    # Check format
    if 'keypoints' in data:
        kpts = data['keypoints']
        if isinstance(kpts, list) and len(kpts) > 0:
            if isinstance(kpts[0], list) and len(kpts[0]) > 0:
                if isinstance(kpts[0][0], list):
                    # Format: [[[x,y],...], [[x,y],...], ...] - List of frames
                    print(f"  Detected: List of {len(kpts)} frames")
                    frames_data = kpts
                else:
                    # Single frame
                    print(f"  Detected: Single frame (need 360 frames)")
                    return False
    
    elif 'frames' in data:
        frames = data['frames']
        print(f"  Detected: Frames array with {len(frames)} frames")
        frames_data = [frame.get('keypoints', []) for frame in frames]
    
    if len(frames_data) == 0:
        print(f"  ✗ Could not extract frames from {input_file}")
        return False
    
    print(f"  ✓ Extracted {len(frames_data)} frames")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each frame as separate JSON file
    for frame_id, frame_keypoints in enumerate(frames_data):
        output_file = os.path.join(output_dir, f'{frame_id:08d}.json')
        frame_json = {
            'joint_type': 'Uplift',
            'bodies': [{
                'id': 0,
                'joints': frame_keypoints
            }]
        }
        
        with open(output_file, 'w') as f:
            json.dump(frame_json, f, indent=2)
    
    print(f"  ✓ Saved {len(frames_data)} frame files to {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert 360-frame keypoint JSON to frame-wise format'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/mnt/server-2TB/workspace/tnhnam/UncaliPose',
        help='Root data directory'
    )
    parser.add_argument(
        '--video_name',
        type=str,
        default='swing1',
        help='Video name'
    )
    parser.add_argument(
        '--front_file',
        type=str,
        default=None,
        help='Path to front camera JSON file (if different from default)'
    )
    parser.add_argument(
        '--side_file',
        type=str,
        default=None,
        help='Path to side camera JSON file (if different from default)'
    )
    
    args = parser.parse_args()
    
    # Default file paths
    if args.front_file is None:
        args.front_file = os.path.join(args.data_dir, 'keypoints', args.video_name, 'front.json')
    if args.side_file is None:
        args.side_file = os.path.join(args.data_dir, 'keypoints', args.video_name, 'side.json')
    
    print("=" * 70)
    print("Converting 360-frame keypoint files to frame-wise format")
    print("=" * 70)
    
    # Output directories
    front_output = os.path.join(args.data_dir, 'keypoints', args.video_name, 'front')
    side_output = os.path.join(args.data_dir, 'keypoints', args.video_name, 'side')
    
    success = True
    
    # Convert front camera
    if os.path.exists(args.front_file):
        if not convert_single_file_to_frames(args.front_file, front_output, 'front'):
            success = False
    else:
        print(f"\n✗ Front file not found: {args.front_file}")
        success = False
    
    # Convert side camera
    if os.path.exists(args.side_file):
        if not convert_single_file_to_frames(args.side_file, side_output, 'side'):
            success = False
    else:
        print(f"\n✗ Side file not found: {args.side_file}")
        success = False
    
    if success:
        print("\n" + "=" * 70)
        print("✓ Conversion complete!")
        print("=" * 70)
        print("\nNow you can process all 360 frames:")
        print(f"  python process_360_frames.py --data_dir {args.data_dir} --video_name {args.video_name}")
    else:
        print("\n" + "=" * 70)
        print("✗ Conversion failed. Check your JSON format.")
        print("=" * 70)


if __name__ == '__main__':
    main()

