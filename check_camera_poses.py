#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Check where camera poses are saved and how they vary across frames.
"""

import json
import numpy as np
import sys
import os

def check_camera_poses(data_dir, video_name, frame_ids=None):
    """Check camera poses across different frames."""
    if frame_ids is None:
        frame_ids = [0, 50, 100, 200, 300]
    
    results_dir = os.path.join(data_dir, 'processed', video_name, 'results')
    
    print("="*70)
    print("Camera Pose Analysis")
    print("="*70)
    print()
    
    poses_by_frame = {}
    
    for frame_id in frame_ids:
        result_file = os.path.join(results_dir, f'frame_{frame_id:08d}_3d_keypoints.json')
        
        if not os.path.exists(result_file):
            print(f"Frame {frame_id}: File not found")
            continue
        
        data = json.load(open(result_file))
        
        if 'camera_poses' in data:
            poses = [np.array(M2) for M2 in data['camera_poses']]
            world_cam = data.get('world_camera', 'N/A')
            
            poses_by_frame[frame_id] = poses
            
            print(f"Frame {frame_id}:")
            print(f"  World camera: {world_cam}")
            print(f"  Number of camera poses: {len(poses)}")
            
            if len(poses) > 0:
                M2 = poses[0]  # First camera pose (relative to world camera)
                R = M2[:, :3]  # Rotation matrix
                t = M2[:, 3]   # Translation vector
                
                print(f"  First camera pose (M2):")
                print(f"    Rotation (first row): {R[0, :]}")
                print(f"    Translation: {t}")
                print(f"    Translation norm: {np.linalg.norm(t):.4f}")
        else:
            print(f"Frame {frame_id}: No camera_poses found")
        
        print()
    
    # Compare poses across frames
    if len(poses_by_frame) > 1:
        print("="*70)
        print("Pose Variation Across Frames")
        print("="*70)
        
        frame_list = sorted(poses_by_frame.keys())
        base_frame = frame_list[0]
        base_pose = poses_by_frame[base_frame][0] if len(poses_by_frame[base_frame]) > 0 else None
        
        if base_pose is not None:
            base_R = base_pose[:, :3]
            
            for frame_id in frame_list[1:]:
                if len(poses_by_frame[frame_id]) > 0:
                    current_pose = poses_by_frame[frame_id][0]
                    current_R = current_pose[:, :3]
                    
                    # Compute rotation difference
                    R_diff = current_R @ base_R.T
                    # Angle between rotations
                    trace = np.trace(R_diff)
                    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1)) * 180 / np.pi
                    
                    # Translation difference
                    t_diff = np.linalg.norm(current_pose[:, 3] - base_pose[:, 3])
                    
                    print(f"Frame {frame_id} vs Frame {base_frame}:")
                    print(f"  Rotation angle difference: {angle:.4f} degrees")
                    print(f"  Translation difference: {t_diff:.4f} meters")
                    print()
    
    print("="*70)
    print("Summary:")
    print("="*70)
    print(f"Camera poses are saved in: {results_dir}/frame_XXXXXX_3d_keypoints.json")
    print(f"  Key: 'camera_poses' (list of 3x4 matrices)")
    print(f"  Each frame has its own estimated camera pose")
    print(f"  Camera poses are estimated per frame using 2D keypoint correspondences")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--video_name', type=str, default='swing1')
    parser.add_argument('--frame_ids', type=int, nargs='+', default=None)
    args = parser.parse_args()
    
    check_camera_poses(args.data_dir, args.video_name, args.frame_ids)

