#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for 3D pose estimation pipeline.
"""

import numpy as np
from src import basic_3d_operations as b3dop


# ============================================================================
# Joint and Skeleton Definitions
# ============================================================================

JOINT_NAMES = [
    "right_ankle", "right_knee", "right_hip", "left_hip",
    "left_knee", "left_ankle", "center_hip", "center_shoulder",
    "neck", "head", "right_wrist", "right_elbow",
    "right_shoulder", "left_shoulder", "left_elbow", "left_wrist"
]

SKELETON_EDGES = [
    (0, 1), (1, 2), (5, 4), (4, 3),  # Legs
    (2, 6), (3, 6),  # Hips to center
    (6, 7), (7, 8), (8, 9),  # Spine
    (12, 11), (11, 10), (7, 12),  # Right arm
    (13, 14), (14, 15), (7, 13)  # Left arm
]

COLORS_2D = {
    'original': (0, 255, 0),      # Green
    'reprojected': (255, 0, 0),  # Red
    'error': (0, 0, 255)         # Blue
}


# ============================================================================
# Height Calculation
# ============================================================================

def calculate_ankle_height_difference(keypoints_3d, joint_names=None):
    """
    Calculate the height difference between left and right ankles.
    
    Args:
        keypoints_3d: [Nx3] array of 3D keypoints
        joint_names: List of joint names (optional, uses JOINT_NAMES if None)
    
    Returns:
        height_diff: Absolute difference in Y coordinate (vertical) between ankles in meters
        Returns None if ankles are not available
    """
    if joint_names is None:
        joint_names = JOINT_NAMES
    
    if len(keypoints_3d) != len(joint_names):
        return None
    
    # Joint indices: right_ankle=0, left_ankle=5 (for 16-joint Uplift format)
    right_ankle_idx = 0
    left_ankle_idx = 5
    
    # Check if both ankles are valid
    if (np.isnan(keypoints_3d[right_ankle_idx]).any() or 
        np.isnan(keypoints_3d[left_ankle_idx]).any()):
        return None
    
    # Y coordinate is the vertical axis (index 1)
    right_ankle_y = keypoints_3d[right_ankle_idx, 1]
    left_ankle_y = keypoints_3d[left_ankle_idx, 1]
    
    # Return absolute difference
    height_diff = abs(right_ankle_y - left_ankle_y)
    return height_diff


def calculate_person_height(keypoints_3d, joint_names, validate=False):
    """Calculate person height from 3D keypoints."""
    if len(keypoints_3d) != len(joint_names):
        return None
    
    if validate:
        valid_kpts = keypoints_3d[~np.isnan(keypoints_3d).any(axis=1)]
        if len(valid_kpts) > 0:
            max_coord = np.abs(valid_kpts).max()
            if max_coord > 10.0:
                return None
    
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
    
    ankle_positions = []
    if right_ankle_idx is not None and not np.isnan(keypoints_3d[right_ankle_idx]).any():
        ankle_positions.append(keypoints_3d[right_ankle_idx])
    if left_ankle_idx is not None and not np.isnan(keypoints_3d[left_ankle_idx]).any():
        ankle_positions.append(keypoints_3d[left_ankle_idx])
    
    if len(ankle_positions) == 0:
        return None
    
    ankle_pos = np.mean(ankle_positions, axis=0)
    hip_to_ankle = np.linalg.norm(hip_pos - ankle_pos)
    height = head_to_hip + hip_to_ankle
    
    if validate:
        # Updated ranges to accommodate normalized coordinates (1.7m average height)
        if not (0.2 <= head_to_hip <= 1.0 and 0.3 <= hip_to_ankle <= 1.2 and 0.5 <= height <= 2.5):
            return None
    
    return height


# ============================================================================
# Reprojection RMSE
# ============================================================================

def calculate_reprojection_rmse(keypoints_3d, keypoints_2d_dict, cam_params_dict, wrld_cam_id, M2_fixed=None):
    """
    Calculate reprojection RMSE.
    
    Args:
        keypoints_3d: 3D keypoints (N, 3) in world coordinate system
        keypoints_2d_dict: Dictionary of 2D keypoints per camera
        cam_params_dict: Camera parameters dictionary
        wrld_cam_id: World camera ID
        M2_fixed: Fixed camera pose (relative to world camera) if using single pose
    
    Returns:
        rmse_dict: Dictionary of RMSE per camera
        total_rmse: Overall RMSE across all cameras
    """
    rmse_dict = {}
    all_errors = []
    
    wrld_cam_name = list(cam_params_dict.keys())[wrld_cam_id]
    M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    
    for cam_name, kpts_2d in keypoints_2d_dict.items():
        if cam_name == wrld_cam_name:
            # World camera - use identity pose
            # Points are already in world coordinate system, so R=I, t=0
            R = np.eye(3)
            t = np.zeros(3)
        else:
            # Other camera - need to transform from world to camera coordinate system
            # M2 = [R | t] represents the pose of the front camera relative to side camera
            # In standard convention: M2 transforms points FROM world (side) TO camera (front)
            # projectPoints does: R @ X + t, so we can use M2 directly
            if M2_fixed is not None and len(M2_fixed) > 0:
                M2 = M2_fixed[0]
                R = M2[:, :3]
                t = M2[:, 3]
            else:
                R = np.eye(3)
                t = np.zeros(3)
        
        cam_params = cam_params_dict[cam_name]
        K = cam_params['K']
        D = cam_params['distCoef']
        
        valid_mask = ~np.isnan(keypoints_3d).any(axis=1) & ~np.isnan(kpts_2d).any(axis=1)
        if np.sum(valid_mask) == 0:
            rmse_dict[cam_name] = None
            continue
        
        kpts_3d_valid = keypoints_3d[valid_mask]
        kpts_2d_valid = kpts_2d[valid_mask]
        
        # projectPoints does: R @ X + t, then projects
        # So we pass R and t directly (it will transform from world to camera)
        kpts_2d_proj = b3dop.projectPoints(kpts_3d_valid.T, K, D, R, t).T
        
        errors = np.linalg.norm(kpts_2d_proj - kpts_2d_valid, axis=1)
        rmse = np.sqrt(np.mean(errors**2))
        rmse_dict[cam_name] = rmse
        all_errors.extend(errors.tolist())
    
    total_rmse = np.sqrt(np.mean(np.array(all_errors)**2)) if len(all_errors) > 0 else None
    return rmse_dict, total_rmse


# ============================================================================
# Scale Normalization
# ============================================================================

def normalize_scale(keypoints_3d, expected_height_m=1.7, joint_names=None):
    """
    Normalize 3D keypoints scale based on expected human height.
    
    Args:
        keypoints_3d: 3D keypoints (N, 3)
        expected_height_m: Expected human height in meters (default: 1.7m)
        joint_names: Joint names list (uses default if None)
    
    Returns:
        Normalized keypoints_3d with correct scale
    """
    if joint_names is None:
        joint_names = JOINT_NAMES
    
    valid_mask = ~np.isnan(keypoints_3d).any(axis=1)
    if not valid_mask.any():
        return keypoints_3d
    
    # Calculate current height
    current_height = calculate_person_height(keypoints_3d, joint_names, validate=False)
    
    if current_height is None or current_height <= 0:
        return keypoints_3d
    
    # Calculate scale factor
    scale_factor = expected_height_m / current_height
    
    # Normalize all keypoints
    normalized_keypoints = keypoints_3d.copy()
    normalized_keypoints[valid_mask] = keypoints_3d[valid_mask] * scale_factor
    
    return normalized_keypoints


def scale_camera_pose_by_height(M2, keypoints_3d, expected_height_m=1.7, joint_names=None):
    """
    Scale camera pose translation to match expected height.
    
    This resolves scale ambiguity by using height constraint.
    
    Args:
        M2: Camera pose matrix [3x4] (R | t)
        keypoints_3d: Triangulated 3D keypoints [N, 3]
        expected_height_m: Expected person height in meters
        joint_names: Joint names list (uses default if None)
    
    Returns:
        M2_scaled: Scaled camera pose matrix
        scale_factor: Applied scale factor
    """
    if joint_names is None:
        joint_names = JOINT_NAMES
    
    # Calculate current height from triangulated points
    current_height = calculate_person_height(keypoints_3d, joint_names, validate=False)
    
    if current_height is None or current_height <= 0:
        return M2, 1.0
    
    # Calculate scale factor
    scale_factor = expected_height_m / current_height
    
    # Scale translation vector (rotation stays the same)
    M2_scaled = M2.copy()
    M2_scaled[:, 3] = M2[:, 3] * scale_factor
    
    return M2_scaled, scale_factor


# ============================================================================
# Configuration
# ============================================================================

def create_default_config():
    """Create default configuration."""
    return {
        'boxprocessing': {
            'box_joints_margin': [1.0, 1.1],
            'box_size_thold': 0.05,
            'resize': None
        },
        'reid': {
            'model': 'osnet',
            'batch_size': 32
        },
        'correspondence': {
            'method': 'hungarian',
            'threshold': 0.5
        },
        'clustering': {
            'method': 'kmeans',
            'n_clusters': 2
        },
        'multiview': {
            'M2_angle_thold': 15
        },
        'bundle': {
            'max_iter': 50,
            'alpha': 1.0
        },
        'visualization': {
            'auto_calculate_view_angle': True,  # If False, use fixed values below
            'fixed_view_azim': 0.0,  # Fixed azimuth angle in degrees (used when auto_calculate_view_angle=False)
            'fixed_view_elev': 0.0,  # Fixed elevation angle in degrees
            'view_angle_sample_frames': 5  # Number of frames to sample for auto calculation
        },
        'camera_pose': {
            'reestimate_pose': False,  # If True, always re-estimate (ignore saved pose)
            'max_candidates': 20,  # Maximum number of frames to sample for camera pose estimation
            'max_rmse_threshold': 4.0,  # Maximum allowed RMSE in pixels
            'target_rmse': 2.0,  # Target RMSE in pixels
            'enforce_ankle_constraint': True,  # Enable ankle height balance constraint
            'max_ankle_diff_cm': 5.0,  # Maximum allowed ankle height difference in cm
            'ankle_weight': 2.0  # Weight for ankle difference in combined score (1cm = ankle_weight pixels penalty)
        }
    }


# ============================================================================
# JSON Combination
# ============================================================================

def combine_frame_json_files(results_dir, output_file=None):
    """
    Combine all frame JSON files into one big JSON file.
    
    Args:
        results_dir: Directory containing frame_*_3d_keypoints.json files
        output_file: Output file path (default: results_dir/all_frames_3d_keypoints.json)
    
    Returns:
        Path to output file
    """
    import os
    import json
    import glob
    
    if output_file is None:
        output_file = os.path.join(results_dir, 'all_frames_3d_keypoints.json')
    
    frame_files = sorted(glob.glob(os.path.join(results_dir, 'frame_*_3d_keypoints.json')))
    
    if len(frame_files) == 0:
        raise ValueError(f"No frame JSON files found in {results_dir}")
    
    all_frames_data = {
        'n_frames': len(frame_files),
        'frames': []
    }
    
    for frame_file in frame_files:
        with open(frame_file, 'r') as f:
            frame_data = json.load(f)
        all_frames_data['frames'].append(frame_data)
    
    with open(output_file, 'w') as f:
        json.dump(all_frames_data, f, indent=2)
    
    return output_file
