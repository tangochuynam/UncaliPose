#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Process 360 frames of 3D keypoints and create visualization video.
This script handles JSON files with 360 frames.
"""

import sys
import os
import argparse
import json
import numpy as np
import yaml

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.dataset.twoview_custom import TwoViewCustom
from src.multiview_3d import solveMultiView3DHumanPoses
from src.bundle_adjustment import bundleAdjustmentWrapper
from src import twoview_3d as twov3d
from src import basic_3d_operations as b3dop


def create_default_config():
    """Create default configuration."""
    return {
        'boxprocessing': {
            'box_joints_margin': [1.0, 1.1],
            'box_size_thold': [20, 20],
            'box_ios_thold': 0.3,
            'joints_inside_img_ratio': 0.6,
            'box_inside_img_ratio': 0.6,
            'resize': [128, 256],
            'replace_old': False,
            'verbose': False
        },
        'reid': {
            'num_prev_frames': 0,
            'trking_method': 'person_id',
            'trk_feat_method': 'max_with_sign_voting'
        },
        'correspondence': {
            'noise_sz': 0,
            'verbose': False
        },
        'multiview': {
            'wrld_cam_id': 1,  # Use side camera (index 1) as world/reference camera
            'verbose': False,
            'M2_angle_thold': 15
        },
        'bundle': {
            'max_iter': 100,
            'fix_cam_pose': False,
            'verbose': False
        }
    }


def calculate_person_height(keypoints_3d, joint_names, validate=True):
    """
    Calculate person height from 3D keypoints.
    
    Args:
        keypoints_3d: Array of 3D keypoints (N, 3)
        joint_names: List of joint names
        validate: If True, validate keypoints for reasonable values
    
    Returns:
        Height in meters, or None if invalid
    """
    if len(keypoints_3d) != len(joint_names):
        return None
    
    # Validate keypoints: check for extreme coordinate values (likely triangulation errors)
    if validate:
        # Check if any coordinate is unreasonably large (>10 meters suggests error)
        max_coord = np.abs(keypoints_3d[~np.isnan(keypoints_3d).any(axis=1)]).max() if not np.isnan(keypoints_3d).all() else 0
        if max_coord > 10.0:  # More than 10 meters is unreasonable for human pose
            return None  # Invalid keypoints, likely triangulation error
    
    head_idx = joint_names.index('head') if 'head' in joint_names else None
    center_hip_idx = joint_names.index('center_hip') if 'center_hip' in joint_names else None
    right_ankle_idx = joint_names.index('right_ankle') if 'right_ankle' in joint_names else None
    left_ankle_idx = joint_names.index('left_ankle') if 'left_ankle' in joint_names else None
    
    if head_idx is None or center_hip_idx is None:
        return None
    
    head_pos = keypoints_3d[head_idx]
    hip_pos = keypoints_3d[center_hip_idx]
    
    if np.isnan(head_pos).any() or np.isnan(hip_pos).any():
        return None
    
    head_to_hip = np.linalg.norm(head_pos - hip_pos)
    
    # Validate head-to-hip distance (should be reasonable, e.g., 0.2-0.5 meters)
    if validate and (head_to_hip < 0.1 or head_to_hip > 0.8):
        return None
    
    ankle_positions = []
    if right_ankle_idx is not None and not np.isnan(keypoints_3d[right_ankle_idx]).any():
        ankle_positions.append(keypoints_3d[right_ankle_idx])
    if left_ankle_idx is not None and not np.isnan(keypoints_3d[left_ankle_idx]).any():
        ankle_positions.append(keypoints_3d[left_ankle_idx])
    
    if len(ankle_positions) == 0:
        return None
    
    ankle_pos = np.mean(ankle_positions, axis=0)
    hip_to_ankle = np.linalg.norm(hip_pos - ankle_pos)
    
    # Validate hip-to-ankle distance (should be reasonable, e.g., 0.3-1.0 meters)
    if validate and (hip_to_ankle < 0.2 or hip_to_ankle > 1.2):
        return None
    
    height = head_to_hip + hip_to_ankle
    
    # Final validation: total height should be reasonable (0.5-2.5 meters)
    if validate and (height < 0.3 or height > 2.5):
        return None
    
    return height


def calculate_reprojection_rmse(keypoints_3d, keypoints_2d_dict, cam_params_dict, wrld_cam_id, M2_fixed=None):
    """
    Calculate reprojection RMSE by projecting 3D keypoints back to 2D.
    
    Args:
        keypoints_3d: 3D keypoints (N, 3) in world coordinate system
        keypoints_2d_dict: Dictionary of 2D keypoints per camera {'front': (N, 2), 'side': (N, 2)}
        cam_params_dict: Camera parameters dictionary
        wrld_cam_id: World camera ID
        M2_fixed: Fixed camera pose (relative to world camera) if using single pose
    
    Returns:
        rmse_dict: Dictionary of RMSE per camera {'front': rmse, 'side': rmse}
        total_rmse: Overall RMSE across all cameras
    """
    rmse_dict = {}
    all_errors = []
    
    wrld_cam_name = list(cam_params_dict.keys())[wrld_cam_id]
    
    # World camera (reference) - identity pose
    M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    
    for cam_name, kpts_2d in keypoints_2d_dict.items():
        if cam_name == wrld_cam_name:
            # World camera - use identity pose
            M = M1
        else:
            # Other camera - use fixed pose if provided
            if M2_fixed is not None and len(M2_fixed) > 0:
                M = M2_fixed[0]  # First (and only) relative pose
            else:
                # Fallback: use identity (shouldn't happen)
                M = M1
        
        cam_params = cam_params_dict[cam_name]
        K = cam_params['K']
        D = cam_params['distCoef']
        R = M[:, :3]
        t = M[:, 3:4].flatten()  # Convert to 1D array
        
        # Project 3D points to 2D
        valid_mask = ~np.isnan(keypoints_3d).any(axis=1) & ~np.isnan(kpts_2d).any(axis=1)
        if np.sum(valid_mask) == 0:
            rmse_dict[cam_name] = None
            continue
        
        kpts_3d_valid = keypoints_3d[valid_mask]
        kpts_2d_valid = kpts_2d[valid_mask]
        
        # Project 3D to 2D
        kpts_2d_proj = b3dop.projectPoints(kpts_3d_valid.T, K, D, R, t).T
        
        # Calculate RMSE
        errors = np.linalg.norm(kpts_2d_proj - kpts_2d_valid, axis=1)
        rmse = np.sqrt(np.mean(errors**2))
        rmse_dict[cam_name] = rmse
        all_errors.extend(errors.tolist())
    
    total_rmse = np.sqrt(np.mean(np.array(all_errors)**2)) if len(all_errors) > 0 else None
    
    return rmse_dict, total_rmse


def triangulate_with_fixed_pose(keypoints_2d_dict, cam_params_dict, wrld_cam_id, M2_fixed):
    """
    Triangulate 3D points directly using fixed camera pose.
    
    Args:
        keypoints_2d_dict: Dictionary of 2D keypoints per camera
        cam_params_dict: Camera parameters dictionary
        wrld_cam_id: World camera ID
        M2_fixed: Fixed camera pose (relative to world camera)
    
    Returns:
        keypoints_3d: 3D keypoints (N, 3)
    """
    wrld_cam_name = list(cam_params_dict.keys())[wrld_cam_id]
    other_cam_name = [name for name in cam_params_dict.keys() if name != wrld_cam_name][0]
    
    # Get camera parameters
    wrld_cam_params = cam_params_dict[wrld_cam_name]
    other_cam_params = cam_params_dict[other_cam_name]
    
    K1, D1 = wrld_cam_params['K'], wrld_cam_params['distCoef']
    K2, D2 = other_cam_params['K'], other_cam_params['distCoef']
    
    # World camera pose (identity)
    M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    # Other camera pose (relative to world)
    M2 = M2_fixed[0] if len(M2_fixed) > 0 else M1
    
    # Get 2D keypoints
    pts1 = keypoints_2d_dict[wrld_cam_name]
    pts2 = keypoints_2d_dict[other_cam_name]
    
    # Triangulate
    keypoints_3d = twov3d.triangulatePoints(K1, D1, M1, K2, D2, M2, pts1, pts2)
    
    return keypoints_3d


def find_best_camera_pose(dataset, config, wrld_cam_id, candidate_frame_ids=None, max_candidates=5):
    """
    Try multiple frames to find the best camera pose based on reprojection RMSE.
    
    Args:
        dataset: TwoViewCustom dataset
        config: Configuration dictionary
        wrld_cam_id: World camera ID
        candidate_frame_ids: List of frame IDs to try (None = auto-select)
        max_candidates: Maximum number of frames to try
    
    Returns:
        best_pose: Best camera pose (M2s list)
        best_rmse: Best RMSE value
        best_frame_id: Frame ID that produced best pose
    """
    pose_files = os.listdir(os.path.join(dataset.pose2d_file_dir, dataset.cam_names[0]))
    n_frames = len([f for f in pose_files if f.endswith('.json')])
    
    # Select candidate frames
    if candidate_frame_ids is None:
        # Try first, middle, last, and a few random frames
        candidate_frame_ids = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]
        candidate_frame_ids = [f for f in candidate_frame_ids if f < n_frames][:max_candidates]
    
    print(f"Trying {len(candidate_frame_ids)} frames to find best camera pose: {candidate_frame_ids}")
    
    best_pose = None
    best_rmse = float('inf')
    best_frame_id = None
    
    for frame_id in candidate_frame_ids:
        try:
            # Get 2D joints
            joints_dict = dataset.getSingleFrameMultiView2DJoints(frame_id)
            
            # Create point correspondences
            pts_corresp_dict = {}
            n_persons = 0
            
            for cam_name in dataset.cam_names:
                pts_corresp_dict[cam_name] = {'keypoints': [], 'box_files': []}
                
                if cam_name in joints_dict and len(joints_dict[cam_name]) > 0:
                    person_key = list(joints_dict[cam_name].keys())[0]
                    joints_2d = joints_dict[cam_name][person_key]
                    pts_corresp_dict[cam_name]['keypoints'] = joints_2d
                    n_persons = 1
                else:
                    pts_corresp_dict[cam_name]['keypoints'] = np.full((16, 2), np.nan)
            
            # Ensure both cameras have same shape
            if len(pts_corresp_dict['front']['keypoints']) == 0:
                pts_corresp_dict['front']['keypoints'] = np.full((16, 2), np.nan)
            if len(pts_corresp_dict['side']['keypoints']) == 0:
                pts_corresp_dict['side']['keypoints'] = np.full((16, 2), np.nan)
            
            # Solve 3D poses to get camera pose
            Pts_init, BA_input_init, wrld_cam_name_init = solveMultiView3DHumanPoses(
                pts_corresp_dict,
                dataset.cam_params_dict,
                n_persons,
                Pts_prev=None,
                wrld_cam_id=wrld_cam_id,
                verbose=False
            )
            
            # Bundle adjustment to refine pose
            Pts_BA_init, M2s_BA_init = bundleAdjustmentWrapper(
                BA_input_init,
                fix_cam_pose=False,
                wrld_cam_id=wrld_cam_id,
                max_iter=config['bundle']['max_iter'],
                verbose=False
            )
            
            # Validate initial result
            valid_mask = ~np.isnan(Pts_BA_init).any(axis=1)
            if valid_mask.any():
                max_coord = np.abs(Pts_BA_init[valid_mask]).max()
                if max_coord > 10.0:
                    # Invalid triangulation, skip this frame
                    continue
            
            # Calculate reprojection RMSE
            keypoints_2d_dict = {
                cam_name: pts_corresp_dict[cam_name]['keypoints']
                for cam_name in dataset.cam_names
            }
            
            rmse_dict, total_rmse = calculate_reprojection_rmse(
                Pts_BA_init, keypoints_2d_dict, dataset.cam_params_dict,
                wrld_cam_id, M2s_BA_init
            )
            
            if total_rmse is not None and total_rmse < best_rmse:
                best_rmse = total_rmse
                best_pose = M2s_BA_init
                best_frame_id = frame_id
                print(f"  Frame {frame_id}: RMSE = {total_rmse:.2f} pixels ✓")
            else:
                if total_rmse is not None:
                    print(f"  Frame {frame_id}: RMSE = {total_rmse:.2f} pixels")
                else:
                    print(f"  Frame {frame_id}: Failed to calculate RMSE")
                    
        except Exception as e:
            print(f"  Frame {frame_id}: Error - {str(e)[:50]}")
            continue
    
    if best_pose is not None:
        print(f"\n✓ Best camera pose from frame {best_frame_id} with RMSE = {best_rmse:.2f} pixels")
    else:
        print(f"\n⚠ Warning: Could not find valid camera pose from candidate frames")
    
    return best_pose, best_rmse, best_frame_id


def process_360_frames_direct(dataset, config, wrld_cam_id=1, use_single_pose=False, pose_frame_id=None):  # Default: side camera (index 1) as world
    """
    Process all 360 frames directly from keypoints.
    
    Args:
        dataset: TwoViewCustom dataset instance
        config: Configuration dictionary
        wrld_cam_id: World camera ID (0 for front, 1 for side)
        use_single_pose: If True, use a single camera pose for all frames
        pose_frame_id: Frame index to use for pose estimation (None = first frame, -1 = last frame)
    """
    # Determine number of frames
    pose_files = os.listdir(os.path.join(dataset.pose2d_file_dir, dataset.cam_names[0]))
    n_frames = len([f for f in pose_files if f.endswith('.json')])
    
    # Determine which frame to use for pose estimation
    if use_single_pose:
        if pose_frame_id is None:
            pose_frame_id = 0  # First frame
        elif pose_frame_id == -1:
            pose_frame_id = n_frames - 1  # Last frame
        elif pose_frame_id < 0 or pose_frame_id >= n_frames:
            print(f"⚠ Warning: Invalid pose_frame_id {pose_frame_id}, using first frame (0)")
            pose_frame_id = 0
        
        print(f"\n{'='*70}")
        print(f"Using SINGLE camera pose for all {n_frames} frames")
        print(f"Pose estimated from frame {pose_frame_id}")
        print(f"{'='*70}\n")
        
        # Find best camera pose by trying multiple frames
        print(f"Finding best camera pose by trying multiple frames...")
        
        # Determine candidate frames
        if pose_frame_id is not None and pose_frame_id != -1:
            # User specified a frame, try that one and a few nearby
            candidate_frames = [pose_frame_id]
            for offset in [1, -1, 2, -2]:
                candidate = pose_frame_id + offset
                if 0 <= candidate < n_frames:
                    candidate_frames.append(candidate)
        else:
            # Try multiple frames: first, middle, last, and a few others
            candidate_frames = None
        
        estimated_pose, best_rmse, best_frame_id = find_best_camera_pose(
            dataset, config, wrld_cam_id, candidate_frame_ids=candidate_frames
        )
        
        if estimated_pose is None:
            print("✗ Could not find valid camera pose, falling back to per-frame estimation")
            use_single_pose = False
        else:
            print(f"✓ Using camera pose from frame {best_frame_id} (RMSE: {best_rmse:.2f} pixels)")
    else:
        print(f"\n{'='*70}")
        print(f"Processing {n_frames} frames (per-frame pose estimation)")
        print(f"{'='*70}\n")
        estimated_pose = None
    
    joint_names = [
        "right_ankle", "right_knee", "right_hip", "left_hip",
        "left_knee", "left_ankle", "center_hip", "center_shoulder",
        "neck", "head", "right_wrist", "right_elbow",
        "right_shoulder", "left_shoulder", "left_elbow", "left_wrist"
    ]
    
    all_results = []
    heights = []
    
    for frame_id in range(n_frames):
        if (frame_id + 1) % 50 == 0 or frame_id == 0:
            print(f"Processing frame {frame_id+1}/{n_frames}...", end=' ')
        elif (frame_id + 1) % 10 == 0:
            print(".", end='', flush=True)
        
        try:
            # Get 2D joints directly
            joints_dict = dataset.getSingleFrameMultiView2DJoints(frame_id)
            
            # Create point correspondences directly from joints
            pts_corresp_dict = {}
            n_persons = 0
            
            for cam_name in dataset.cam_names:
                pts_corresp_dict[cam_name] = {'keypoints': [], 'box_files': []}
                
                if cam_name in joints_dict and len(joints_dict[cam_name]) > 0:
                    # Get first person's joints
                    person_key = list(joints_dict[cam_name].keys())[0]
                    joints_2d = joints_dict[cam_name][person_key]
                    pts_corresp_dict[cam_name]['keypoints'] = joints_2d
                    n_persons = 1
                else:
                    # No keypoints for this camera
                    pts_corresp_dict[cam_name]['keypoints'] = np.full((16, 2), np.nan)
            
            # Ensure both cameras have same shape
            if len(pts_corresp_dict['front']['keypoints']) == 0:
                pts_corresp_dict['front']['keypoints'] = np.full((16, 2), np.nan)
            if len(pts_corresp_dict['side']['keypoints']) == 0:
                pts_corresp_dict['side']['keypoints'] = np.full((16, 2), np.nan)
            
            # If using single pose, use DIRECT triangulation with fixed pose
            if use_single_pose and estimated_pose is not None:
                # Direct triangulation using fixed camera pose (pure math, no estimation)
                keypoints_2d_dict = {
                    cam_name: np.array(pts_corresp_dict[cam_name]['keypoints'])
                    for cam_name in dataset.cam_names
                }
                
                Pts_BA = triangulate_with_fixed_pose(
                    keypoints_2d_dict,
                    dataset.cam_params_dict,
                    wrld_cam_id,
                    estimated_pose
                )
                
                M2s_BA = estimated_pose
                wrld_cam_name = list(dataset.cam_names)[wrld_cam_id]
                
                # Optional: Bundle adjustment with fixed pose (only optimizes 3D points)
                # This can help refine the 3D points while keeping camera pose fixed
                try:
                    # Prepare BA input
                    BA_input = {
                        'p1': keypoints_2d_dict[wrld_cam_name],
                        'p2s': [keypoints_2d_dict[cam] for cam in dataset.cam_names if cam != wrld_cam_name],
                        'P': Pts_BA,
                        'K1': dataset.cam_params_dict[wrld_cam_name]['K'],
                        'D1': dataset.cam_params_dict[wrld_cam_name]['distCoef'],
                        'M1': np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]),
                        'K2s': [dataset.cam_params_dict[cam]['K'] for cam in dataset.cam_names if cam != wrld_cam_name],
                        'D2s': [dataset.cam_params_dict[cam]['distCoef'] for cam in dataset.cam_names if cam != wrld_cam_name],
                        'M2s': estimated_pose
                    }
                    
                    # Bundle adjustment with fixed camera pose (only optimizes 3D points)
                    Pts_BA, _ = bundleAdjustmentWrapper(
                        BA_input,
                        fix_cam_pose=True,  # Fix camera pose
                        wrld_cam_id=wrld_cam_id,
                        max_iter=config['bundle']['max_iter'],
                        verbose=False
                    )
                except:
                    # If bundle adjustment fails, use direct triangulation result
                    pass
            else:
                # Per-frame pose estimation (original method)
                Pts, BA_input, wrld_cam_name = solveMultiView3DHumanPoses(
                    pts_corresp_dict,
                    dataset.cam_params_dict,
                    n_persons,
                    Pts_prev=None,
                    wrld_cam_id=wrld_cam_id,
                    verbose=False
                )
                
                # Validate initial triangulation
                valid_mask_initial = ~np.isnan(Pts).any(axis=1)
                if valid_mask_initial.any():
                    valid_pts_initial = Pts[valid_mask_initial]
                    max_coord_initial = np.abs(valid_pts_initial).max()
                    
                    if max_coord_initial > 10.0:
                        has_invalid_keypoints = True
                        max_coord = max_coord_initial
                        Pts_BA = Pts
                        M2s_BA = BA_input['M2s'] if 'M2s' in BA_input else []
                    else:
                        has_invalid_keypoints = False
                        fix_cam_pose = False
                        
                        Pts_BA, M2s_BA = bundleAdjustmentWrapper(
                            BA_input,
                            fix_cam_pose=fix_cam_pose,
                            wrld_cam_id=wrld_cam_id,
                            max_iter=config['bundle']['max_iter'],
                            verbose=False
                        )
                else:
                    has_invalid_keypoints = True
                    max_coord = 0
                    Pts_BA = Pts
                    M2s_BA = BA_input['M2s'] if 'M2s' in BA_input else []
            
            # Validate 3D keypoints
            valid_mask_pts = ~np.isnan(Pts_BA).any(axis=1)
            if valid_mask_pts.any():
                valid_pts = Pts_BA[valid_mask_pts]
                max_coord = np.abs(valid_pts).max()
                min_coord = np.abs(valid_pts).min()
            else:
                max_coord = 0
                min_coord = 0
            
            # Check for invalid keypoints
            has_invalid_keypoints = max_coord > 10.0
            if not has_invalid_keypoints and valid_mask_pts.any():
                extreme_count = np.sum(np.abs(valid_pts) > 5.0)
                if extreme_count > len(valid_pts) * 0.3:
                    has_invalid_keypoints = True
            
            # Calculate reprojection RMSE to validate 3D keypoints
            keypoints_2d_dict = {
                cam_name: np.array(pts_corresp_dict[cam_name]['keypoints'])
                for cam_name in dataset.cam_names
            }
            
            rmse_dict, total_rmse = calculate_reprojection_rmse(
                Pts_BA, keypoints_2d_dict, dataset.cam_params_dict,
                wrld_cam_id, M2s_BA if use_single_pose and estimated_pose is not None else None
            )
            
            # Log RMSE for monitoring
            if (frame_id + 1) % 50 == 0 or frame_id == 0:
                if total_rmse is not None:
                    print(f"RMSE: {total_rmse:.2f}px", end=' ')
                else:
                    print("RMSE: N/A", end=' ')
            
            if has_invalid_keypoints:
                # Log warning for frames with invalid keypoints
                if (frame_id + 1) % 50 == 0 or frame_id == 0 or frame_id in [152, 163, 164, 165]:
                    print(f"⚠ Frame {frame_id}: Invalid 3D keypoints (max_coord={max_coord:.1f}m)")
            
            # Calculate height
            n_joints = len(Pts_BA) // n_persons if n_persons > 0 else 16
            frame_heights = []
            
            for person_id in range(n_persons):
                start_idx = person_id * n_joints
                end_idx = (person_id + 1) * n_joints
                person_keypoints = Pts_BA[start_idx:end_idx]
                
                height = calculate_person_height(person_keypoints, joint_names, validate=True)
                if height is not None:
                    frame_heights.append(height)
                    heights.append(height)
                elif has_invalid_keypoints:
                    # Invalid height due to bad keypoints
                    if (frame_id + 1) % 50 == 0 or frame_id == 0:
                        print(f"⚠ (Height calculation failed due to invalid keypoints)")
            
            # Prepare camera poses for both cameras
            # M2s_BA contains poses of non-reference cameras relative to reference
            # For 2 cameras: if side is reference, M2s_BA[0] is front relative to side
            camera_poses_dict = {}
            cam_names_list = list(dataset.cam_names)
            
            # Reference camera (world camera) - identity pose
            identity_pose = np.array([[1.0, 0.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0, 0.0],
                                      [0.0, 0.0, 1.0, 0.0]])
            camera_poses_dict[wrld_cam_name] = {
                'pose_matrix': identity_pose.tolist(),
                'rotation_matrix': identity_pose[:, :3].tolist(),
                'translation_vector': identity_pose[:, 3].tolist(),
                'is_reference': True
            }
            
            # Other cameras - relative poses
            other_cam_idx = 0
            for cam_name in cam_names_list:
                if cam_name != wrld_cam_name:
                    if other_cam_idx < len(M2s_BA):
                        M2 = M2s_BA[other_cam_idx]
                        R = M2[:, :3]
                        t = M2[:, 3]
                        camera_poses_dict[cam_name] = {
                            'pose_matrix': M2.tolist(),
                            'rotation_matrix': R.tolist(),
                            'translation_vector': t.tolist(),
                            'is_reference': False,
                            'relative_to': wrld_cam_name
                        }
                        other_cam_idx += 1
            
            # Save results
            result_dir = os.path.join(dataset.data_dir, 'processed', dataset.video_name, 'results')
            os.makedirs(result_dir, exist_ok=True)
            
            result_file = os.path.join(result_dir, f'frame_{frame_id:08d}_3d_keypoints.json')
            result_data = {
                'frame_id': int(frame_id),
                'n_persons': int(n_persons),
                'n_joints': int(n_joints),
                'keypoints_3d': Pts_BA.tolist(),
                'keypoints_by_person': {},
                'heights': {},
                'camera_poses': [M2.tolist() for M2 in M2s_BA],  # Keep for backward compatibility
                'camera_poses_dict': camera_poses_dict,  # New: explicit poses for both cameras
                'world_camera': str(wrld_cam_name),
                'keypoints_valid': not has_invalid_keypoints,  # Flag for overall validity
                'max_coordinate': float(max_coord) if valid_mask_pts.any() else None
            }
            
            # Organize by person
            for person_id in range(n_persons):
                start_idx = person_id * n_joints
                end_idx = (person_id + 1) * n_joints
                person_keypoints = Pts_BA[start_idx:end_idx]
                
                height = calculate_person_height(person_keypoints, joint_names, validate=True)
                
                # If keypoints are invalid, don't calculate height
                if has_invalid_keypoints:
                    height = None
                
                result_data['keypoints_by_person'][f'person_{person_id}'] = {
                    'joints': person_keypoints.tolist(),
                    'joint_names': joint_names,
                    'height_m': float(height) if height is not None else None,
                    'height_cm': float(height * 100) if height is not None else None,
                    'keypoints_valid': not has_invalid_keypoints,  # Flag indicating if keypoints are valid
                    'reprojection_rmse': {
                        'total_rmse_pixels': float(total_rmse) if total_rmse is not None else None,
                        'per_camera_rmse': {
                            cam: float(rmse) if rmse is not None else None
                            for cam, rmse in rmse_dict.items()
                        }
                    }
                }
                result_data['heights'][f'person_{person_id}'] = {
                    'height_m': float(height) if height is not None else None,
                    'height_cm': float(height * 100) if height is not None else None
                }
            
            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            all_results.append({
                'frame_id': frame_id,
                'success': True,
                'n_persons': n_persons,
                'heights': frame_heights
            })
            
            if (frame_id + 1) % 50 == 0 or frame_id == 0:
                if frame_heights:
                    print(f"✓ (height: {frame_heights[0]*100:.1f}cm)")
                else:
                    print("✓")
            
        except Exception as e:
            if (frame_id + 1) % 50 == 0 or frame_id == 0:
                print(f"✗ Error: {str(e)[:50]}")
            all_results.append({
                'frame_id': frame_id,
                'success': False,
                'error': str(e)
            })
    
    print()  # New line after progress dots
    
    # Save summary
    summary_file = os.path.join(
        dataset.data_dir, 'processed', dataset.video_name,
        'results', 'summary.json'
    )
    
    summary_data = {
        'total_frames': len(all_results),
        'successful_frames': sum(1 for r in all_results if r['success']),
        'height_statistics': {
            'mean_height_m': float(np.mean(heights)) if heights else None,
            'mean_height_cm': float(np.mean(heights) * 100) if heights else None,
            'std_height_m': float(np.std(heights)) if heights else None,
            'min_height_m': float(np.min(heights)) if heights else None,
            'max_height_m': float(np.max(heights)) if heights else None,
        },
        'results': all_results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Processing complete!")
    print(f"Successful frames: {summary_data['successful_frames']}/{summary_data['total_frames']}")
    if heights:
        print(f"Average height: {np.mean(heights)*100:.1f} cm")
        print(f"Height range: {np.min(heights)*100:.1f} - {np.max(heights)*100:.1f} cm")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*70}\n")
    
    return summary_file


def main():
    parser = argparse.ArgumentParser(
        description='Process 360 frames of 3D keypoints'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Root directory containing keypoints'
    )
    parser.add_argument(
        '--video_name',
        type=str,
        default='swing1',
        help='Name of the video folder'
    )
    parser.add_argument(
        '--world_cam_id',
        type=int,
        default=1,  # Default: side camera (index 1) as world/reference
        help='World camera ID: 0 for front, 1 for side (default: 1 for side)'
    )
    parser.add_argument(
        '--use_single_pose',
        action='store_true',
        help='Use a single camera pose for all frames (estimate from one frame, reuse for all)'
    )
    parser.add_argument(
        '--pose_frame_id',
        type=int,
        default=None,
        help='Frame index to use for pose estimation when --use_single_pose is set. None=first frame (0), -1=last frame, or specify frame index'
    )
    
    args = parser.parse_args()
    
    config = create_default_config()
    
    print(f"Initializing dataset from: {args.data_dir}")
    dataset = TwoViewCustom(
        data_dir=args.data_dir,
        video_name=args.video_name,
        config=config
    )
    
    # Prepare data
    print("\nPreparing data structure from keypoints...")
    n_frames = dataset.prepareDataFromKeypoints()
    print(f"✓ Prepared {n_frames} frames\n")
    
    if n_frames != 360:
        print(f"⚠ Warning: Expected 360 frames, but found {n_frames} frames")
        print("If your JSON files contain 360 frames, make sure the format is:")
        print("  Option 1: {'keypoints': [[[x,y],...], [[x,y],...], ...]}  (360 frames)")
        print("  Option 2: {'frames': [{'keypoints': [...]}, ...]}  (360 frames)")
        print("  Option 3: 360 separate JSON files in front/ and side/ directories")
        print()
    
    # Process all frames
    summary_file = process_360_frames_direct(
        dataset,
        config,
        wrld_cam_id=args.world_cam_id,
        use_single_pose=args.use_single_pose,
        pose_frame_id=args.pose_frame_id
    )
    
    print(f"\n✓ All {n_frames} frames processed! Summary: {summary_file}")
    print("\nNext step: Create visualization video:")
    print(f"  python visualize_3d_keypoints_video.py --data_dir {args.data_dir} --video_name {args.video_name} --fps 30")


if __name__ == '__main__':
    main()

