#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Camera pose management for 3D pose estimation pipeline.
"""

import os
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation
from src.multiview_3d import solveMultiView3DHumanPoses
from src.bundle_adjustment import bundleAdjustmentWrapper
from src import twoview_3d as twov3d
from src.pipeline_utils import (calculate_reprojection_rmse, 
    calculate_ankle_height_difference,
    JOINT_NAMES
)


class CameraPoseManager:
    """Manages camera pose estimation and storage."""
    
    def __init__(self, video_path, logger=None):
        """
        Initialize camera pose manager.
        
        Args:
            video_path: Path to video folder containing videos and keypoints
            logger: Optional logger instance
        """
        self.video_path = os.path.abspath(video_path)
        self.video_name = os.path.basename(self.video_path)
        self.data_dir = os.path.dirname(self.video_path)
        self.logger = logger
        self.pose_dir = os.path.join(self.data_dir, 'processed', self.video_name)
        self.pose_file = os.path.join(self.pose_dir, 'camera_pose.json')
    
    def load(self):
        """Load saved camera pose. Supports multiple formats (old, new with calibration, new with estimated_pose)."""
        if not os.path.exists(self.pose_file):
            return None, None, None, None
        
        try:
            with open(self.pose_file, 'r') as f:
                data = json.load(f)
                
                # Try new format with estimated_pose (default format - only pose, no intrinsics)
                if 'estimated_pose' in data:
                    est_pose = data['estimated_pose']
                    if 'rotation' in est_pose and 'translation' in est_pose:
                        rx, ry, rz = est_pose['rotation']
                        tx, ty, tz = est_pose['translation']
                        
                        # Convert axis-angle to rotation matrix
                        rotation = Rotation.from_rotvec([rx, ry, rz])
                        R = rotation.as_matrix()
                        
                        # Construct 3x4 pose matrix [R|t]
                        pose = np.hstack([R, np.array([[tx], [ty], [tz]])])
                        
                        return [pose], data.get('frame_id', None), data.get('rmse', None), data.get('estimation_time', None)
                
                # Try new format with calibration (includes intrinsics)
                if 'calibration' in data:
                    # Extract pose from calibration data (camera 1's relative pose)
                    calibration = data['calibration']
                    if len(calibration) >= 2:
                        # Get camera 1's pose (index 1)
                        cam1_data = calibration[1]
                        if len(cam1_data) >= 16:
                            rx, ry, rz = cam1_data[10], cam1_data[11], cam1_data[12]
                            tx, ty, tz = cam1_data[13], cam1_data[14], cam1_data[15]
                            
                            # Convert axis-angle to rotation matrix
                            rotation = Rotation.from_rotvec([rx, ry, rz])
                            R = rotation.as_matrix()
                            
                            # Construct 3x4 pose matrix [R|t]
                            pose = np.hstack([R, np.array([[tx], [ty], [tz]])])
                            
                            return [pose], data.get('frame_id', None), data.get('rmse', None), data.get('estimation_time', None)
                
                # Fallback to old format (has 'pose' key - 3x4 matrix)
                if 'pose' in data:
                    pose = np.array(data['pose'])
                    # Ensure it's in the right format (list of 3x4 matrices)
                    if len(pose.shape) == 2:
                        pose = [pose]  # Wrap in list if single matrix
                    return pose, data.get('frame_id', None), data.get('rmse', None), data.get('estimation_time', None)
                
                # If none of the formats found
                if self.logger:
                    self.logger.warning("Camera pose file does not contain 'pose', 'estimated_pose', or 'calibration' key")
                return None, None, None, None
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            if self.logger:
                self.logger.warning(f"Error loading camera pose file: {e}")
            return None, None, None, None
    
    def save(self, pose, frame_id, rmse, estimation_time=None, dataset=None, wrld_cam_id=1, include_intrinsics=False):
        """
        Save camera pose.
        
        By default, only saves the estimated relative pose (R, t).
        If include_intrinsics=True, also saves intrinsic and distortion parameters (default values, not estimated).
        
        Args:
            pose: Camera pose matrix [3x4] or list of pose matrices
            frame_id: Frame ID used for pose estimation
            rmse: Reprojection RMSE
            estimation_time: Time taken for estimation
            dataset: Dataset instance (optional, needed if include_intrinsics=True)
            wrld_cam_id: World camera ID
            include_intrinsics: If True, include intrinsic and distortion parameters (default: False)
        
        Format when include_intrinsics=True:
            [index, fx, fy, cx, cy, d1, d2, d3, d4, d5, rx, ry, rz, tx, ty, tz]
            Where index 0 is for side view and 1 for front view.
            The pose (rx, ry, rz, tx, ty, tz) is the relative transformation from camera index to camera index+1.
            Note: Intrinsics and distortion are DEFAULT values, NOT estimated by UncaliPose.
        """
        os.makedirs(self.pose_dir, exist_ok=True)
        
        # Convert pose to numpy array (handle both numpy array and list)
        if isinstance(pose, list):
            if len(pose) > 0:
                pose_matrix = np.array(pose[0]) if isinstance(pose[0], np.ndarray) else np.array(pose[0])
            else:
                pose_matrix = np.array(pose)
        else:
            pose_matrix = np.array(pose)
        
        # Ensure pose is 3x4 matrix [R|t]
        if pose_matrix.shape != (3, 4):
            raise ValueError(f"Expected pose matrix shape (3, 4), got {pose_matrix.shape}")
        
        # Extract rotation and translation
        R = pose_matrix[:, :3]  # 3x3 rotation matrix
        t = pose_matrix[:, 3:4].flatten()  # 3x1 translation vector
        
        # Convert rotation matrix to axis-angle representation (rx, ry, rz)
        rotation = Rotation.from_matrix(R)
        rx, ry, rz = rotation.as_rotvec()
        
        # Prepare output data
        output_data = {
            'frame_id': frame_id,
            'rmse': float(rmse) if rmse is not None else None,
            'estimation_time': float(estimation_time) if estimation_time is not None else None,
            # Keep old format for backward compatibility
            'pose': pose_matrix.tolist(),
            # Estimated pose in axis-angle format (this is what UncaliPose actually estimates)
            'estimated_pose': {
                'rotation': [float(rx), float(ry), float(rz)],  # axis-angle representation
                'translation': [float(t[0]), float(t[1]), float(t[2])]  # translation vector
            }
        }
        
        # Optionally include intrinsics and distortion (default values, not estimated)
        if include_intrinsics:
            calibration_data = []
            
            # Camera index mapping: 0 = side view, 1 = front view
            if dataset is not None and hasattr(dataset, 'cam_params_dict'):
                # Camera 0: side view (world camera)
                if 'side' in dataset.cam_params_dict:
                    cam_params_side = dataset.cam_params_dict['side']
                    K_side = cam_params_side['K']
                    D_side = cam_params_side['distCoef']
                    
                    # Extract parameters
                    fx_side = float(K_side[0, 0])
                    fy_side = float(K_side[1, 1])
                    cx_side = float(K_side[0, 2])
                    cy_side = float(K_side[1, 2])
                    d1_side, d2_side, d3_side, d4_side, d5_side = [float(D_side[i]) if i < len(D_side) else 0.0 for i in range(5)]
                    
                    # For camera 0 (side), relative pose is identity (no transformation from itself)
                    calibration_data.append([
                        0,  # index (side view)
                        fx_side, fy_side, cx_side, cy_side,  # intrinsic (default, not estimated)
                        d1_side, d2_side, d3_side, d4_side, d5_side,  # distortion (default, not estimated)
                        0.0, 0.0, 0.0,  # rx, ry, rz (identity rotation)
                        0.0, 0.0, 0.0   # tx, ty, tz (zero translation)
                    ])
                
                # Camera 1: front view
                if 'front' in dataset.cam_params_dict:
                    cam_params_front = dataset.cam_params_dict['front']
                    K_front = cam_params_front['K']
                    D_front = cam_params_front['distCoef']
                    
                    # Extract parameters
                    fx_front = float(K_front[0, 0])
                    fy_front = float(K_front[1, 1])
                    cx_front = float(K_front[0, 2])
                    cy_front = float(K_front[1, 2])
                    d1_front, d2_front, d3_front, d4_front, d5_front = [float(D_front[i]) if i < len(D_front) else 0.0 for i in range(5)]
                    
                    # For camera 1 (front), relative pose is from camera 0 (side) to camera 1 (front)
                    calibration_data.append([
                        1,  # index (front view)
                        fx_front, fy_front, cx_front, cy_front,  # intrinsic (default, not estimated)
                        d1_front, d2_front, d3_front, d4_front, d5_front,  # distortion (default, not estimated)
                        float(rx), float(ry), float(rz),  # rotation from side (0) to front (1) - ESTIMATED
                        float(t[0]), float(t[1]), float(t[2])  # translation from side (0) to front (1) - ESTIMATED
                    ])
            else:
                # Fallback: if dataset not provided, use default values
                if self.logger:
                    self.logger.warning("Dataset not provided, using default camera parameters for intrinsics")
                
                # Default intrinsic parameters (NOT estimated, just defaults)
                default_fx, default_fy = 1000.0, 1000.0
                default_cx, default_cy = 960.0, 540.0  # Assuming 1920x1080 / 2
                
                # Camera 0: side
                calibration_data.append([
                    0, default_fx, default_fy, default_cx, default_cy,
                    0.0, 0.0, 0.0, 0.0, 0.0,  # distortion (default zeros)
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # pose (identity)
                ])
                
                # Camera 1: front
                calibration_data.append([
                    1, default_fx, default_fy, default_cx, default_cy,
                    0.0, 0.0, 0.0, 0.0, 0.0,  # distortion (default zeros)
                    float(rx), float(ry), float(rz),  # rotation - ESTIMATED
                    float(t[0]), float(t[1]), float(t[2])  # translation - ESTIMATED
                ])
            
            output_data['calibration'] = calibration_data
            output_data['note'] = 'Intrinsics and distortion are default values, NOT estimated. Only pose (rx, ry, rz, tx, ty, tz) is estimated.'
        
        # Save to file
        with open(self.pose_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        if self.logger:
            self.logger.info(f"Camera pose saved to: {self.pose_file}")
            if include_intrinsics:
                self.logger.info(f"Format: [index, fx, fy, cx, cy, d1-d5, rx, ry, rz, tx, ty, tz]")
                self.logger.info(f"Note: Only pose (rx, ry, rz, tx, ty, tz) is estimated. Intrinsics/distortion are defaults.")
            else:
                self.logger.info(f"Saved estimated pose only (rotation + translation)")
    
    def find_best_pose(self, dataset, config, wrld_cam_id, candidate_frame_ids=None, 
                       max_candidates=40, max_rmse_threshold=4.0,
                       target_rmse=2.0, sample_frames=True, 
                       enforce_ankle_constraint=True, max_ankle_diff_cm=5.0, ankle_weight=2.0):
        """
        Find best camera pose by trying multiple frames.
        
        Args:
            dataset: Dataset instance
            config: Configuration dictionary
            wrld_cam_id: World camera ID
            candidate_frame_ids: List of frame IDs to try (None = auto-select)
            max_candidates: Maximum number of candidates
            max_rmse_threshold: Maximum allowed RMSE in pixels
            target_rmse: Target RMSE in pixels (default: 2.0)
            sample_frames: If True, sample frames throughout video for better pose estimation
        
        Returns:
            best_pose: Best camera pose
            best_rmse: Best RMSE value
            best_frame_id: Frame ID used for best pose
            elapsed_time: Time taken in seconds
        """
        start_time = time.time()
        
        # Get number of frames from dataset (loaded from keypoints_2d JSON files)
        # The dataset should have n_frames attribute after prepareDataFromKeypoints()
        if hasattr(dataset, 'n_frames') and dataset.n_frames is not None:
            n_frames = dataset.n_frames
        else:
            # Fallback: try to determine from keypoint files
            # Check if pose2d_file_dir exists (backward compatibility)
            pose2d_dir = os.path.join(dataset.pose2d_file_dir, dataset.cam_names[0])
            if os.path.exists(pose2d_dir):
                pose_files = os.listdir(pose2d_dir)
                n_frames = len([f for f in pose_files if f.endswith('.json')])
            else:
                # Load from source JSON files directly
                # Try to get frame count from keypoints_2d in JSON files
                import glob
                json_files = glob.glob(os.path.join(dataset.video_dir, '*.json'))
                if json_files:
                    import json
                    with open(json_files[0], 'r') as f:
                        kpts_data = json.load(f)
                        if 'keypoints_2d' in kpts_data:
                            n_frames = len(kpts_data['keypoints_2d'])
                        else:
                            # Default fallback
                            n_frames = 360  # Common default
                else:
                    n_frames = 360  # Default fallback
        
        if self.logger:
            self.logger.info(f"Total frames detected: {n_frames}")
        
        if candidate_frame_ids is None:
            if sample_frames and n_frames > max_candidates:
                # Sample frames throughout the video for better pose estimation
                step = max(1, n_frames // (max_candidates + 1))
                candidate_frame_ids = [i * step for i in range(1, max_candidates + 1)]
                candidate_frame_ids.append(n_frames - 1)  # Always include last frame
                candidate_frame_ids = sorted(list(set([f for f in candidate_frame_ids if f < n_frames])))[:max_candidates]
            else:
                candidate_frame_ids = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]
                candidate_frame_ids = [f for f in candidate_frame_ids if f < n_frames][:max_candidates]
        
        if self.logger:
            self.logger.info(f"Trying {len(candidate_frame_ids)} frames: {candidate_frame_ids}")
            self.logger.info(f"RMSE threshold: < {max_rmse_threshold} pixels")
            if target_rmse < max_rmse_threshold:
                self.logger.info(f"Target RMSE: < {target_rmse} pixels")
            if enforce_ankle_constraint:
                self.logger.info(f"Ankle height constraint: Optimizing for balanced ankles (general approach)")
                self.logger.info(f"  - Bundle adjustment: weight=500.0 enforces ankle balance during optimization")
                self.logger.info(f"  - Selection: Combined objective = RMSE + {ankle_weight:.1f} * ankle_diff_cm")
                self.logger.info(f"  - Strategy: Select frame with LOWEST combined score (no hard thresholds)")
                self.logger.info(f"  - This naturally balances RMSE and ankle difference")
        
        best_pose = None
        best_rmse = float('inf')
        best_frame_id = None
        best_ankle_diff = float('inf')  # Track best ankle height difference
        best_combined_score = float('inf')  # Track best combined objective score (RMSE + weight * ankle_diff)
        
        for frame_id in candidate_frame_ids:
            try:
                joints_dict = dataset.getSingleFrameMultiView2DJoints(frame_id)
                
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
                
                if len(pts_corresp_dict['front']['keypoints']) == 0:
                    pts_corresp_dict['front']['keypoints'] = np.full((16, 2), np.nan)
                if len(pts_corresp_dict['side']['keypoints']) == 0:
                    pts_corresp_dict['side']['keypoints'] = np.full((16, 2), np.nan)
                
                Pts_init, BA_input_init, wrld_cam_name_init = solveMultiView3DHumanPoses(
                    pts_corresp_dict, dataset.cam_params_dict, n_persons,
                    Pts_prev=None, wrld_cam_id=wrld_cam_id, verbose=False
                )
                
                # First optimization: Estimate camera pose with ankle constraint
                # The bundle adjustment already includes ankle constraint (weight=500.0)
                Pts_BA_init, M2s_BA_init = bundleAdjustmentWrapper(
                    BA_input_init, fix_cam_pose=False, wrld_cam_id=wrld_cam_id,
                    max_iter=config['bundle']['max_iter'], verbose=False
                )
                
                valid_mask = ~np.isnan(Pts_BA_init).any(axis=1)
                if valid_mask.any():
                    max_coord = np.abs(Pts_BA_init[valid_mask]).max()
                    if max_coord > 10.0:
                        continue
                
                keypoints_2d_dict = {
                    cam_name: pts_corresp_dict[cam_name]['keypoints']
                    for cam_name in dataset.cam_names
                }
                
                rmse_dict, total_rmse = calculate_reprojection_rmse(
                    Pts_BA_init, keypoints_2d_dict, dataset.cam_params_dict,
                    wrld_cam_id, M2s_BA_init
                )
                
                # Calculate ankle height difference (this is already optimized in bundle adjustment)
                ankle_diff = None
                if enforce_ankle_constraint:
                    ankle_diff = calculate_ankle_height_difference(Pts_BA_init, JOINT_NAMES)
                    ankle_diff_cm = ankle_diff * 100.0 if ankle_diff is not None else None
                else:
                    ankle_diff_cm = None
                
                # GENERAL APPROACH: Use combined objective score to optimize for both RMSE and ankle balance
                # The bundle adjustment already optimizes for ankle balance (weight=500.0)
                # Here we use a combined score to select the best pose - no hard thresholds
                
                # Combined objective: score = RMSE + weight * ankle_diff_cm
                # Lower score is better - naturally balances both objectives
                combined_score = None
                if enforce_ankle_constraint and ankle_diff is not None and total_rmse is not None:
                    # Weight ankle difference relative to RMSE
                    # The weight determines how much we prioritize ankle balance vs RMSE
                    # Higher weight = more emphasis on ankle balance
                    # ankle_weight is passed as parameter (default: 2.0)
                    combined_score = total_rmse + ankle_weight * ankle_diff_cm
                elif total_rmse is not None:
                    # No ankle constraint - just use RMSE
                    combined_score = total_rmse
                
                # Simple selection: Choose frame with lowest combined score
                # No hard thresholds - just optimize the combined objective
                is_better = False
                if combined_score is not None:
                    # Basic validation: skip frames with extreme errors
                    if total_rmse is not None and total_rmse > 20.0:  # Skip obviously bad frames
                        is_better = False
                    elif best_pose is None:
                        # No best pose yet, this is better
                        is_better = True
                    elif combined_score < best_combined_score:
                        # This frame has better combined score (lower is better)
                        is_better = True
                    elif abs(combined_score - best_combined_score) < 0.01:
                        # Very similar combined score, prefer better ankle balance
                        if ankle_diff is not None and ankle_diff < best_ankle_diff:
                            is_better = True
                elif total_rmse is not None and not enforce_ankle_constraint:
                    # Fallback: No ankle constraint - use RMSE only
                    if best_pose is None or total_rmse < best_rmse:
                        is_better = True
                
                # For logging purposes
                is_target = total_rmse is not None and total_rmse <= target_rmse if 'target_rmse' in locals() else False
                ankle_ok = False
                if enforce_ankle_constraint and ankle_diff is not None:
                    ankle_ok = ankle_diff_cm <= max_ankle_diff_cm
                
                if is_better:
                    best_rmse = total_rmse
                    best_pose = M2s_BA_init
                    best_frame_id = frame_id
                    if ankle_diff is not None:
                        best_ankle_diff = ankle_diff
                    if combined_score is not None:
                        best_combined_score = combined_score
                    if self.logger:
                        target_status = "✓✓ (target achieved)" if is_target else "✓ (within threshold)"
                        ankle_status = ""
                        if ankle_diff_cm is not None:
                            if ankle_ok:
                                ankle_status = f", ankle diff: {ankle_diff_cm:.2f}cm ✓"
                            else:
                                ankle_status = f", ankle diff: {ankle_diff_cm:.2f}cm ✗"
                        score_info = f", combined score: {combined_score:.2f}" if combined_score is not None else ""
                        self.logger.info(
                            f"Frame {frame_id}: RMSE = {total_rmse:.2f} pixels {target_status}{ankle_status}{score_info}"
                        )
                else:
                    if total_rmse is not None:
                        status = "✗ (exceeds threshold)" if total_rmse > max_rmse_threshold else ""
                        ankle_status = ""
                        if enforce_ankle_constraint and ankle_diff_cm is not None:
                            if ankle_diff_cm > max_ankle_diff_cm:
                                ankle_status = f", ankle diff: {ankle_diff_cm:.2f}cm ✗"
                            else:
                                ankle_status = f", ankle diff: {ankle_diff_cm:.2f}cm"
                        score_info = f", combined score: {combined_score:.2f}" if combined_score is not None else ""
                        if self.logger:
                            self.logger.info(
                                f"Frame {frame_id}: RMSE = {total_rmse:.2f} pixels {status}{ankle_status}{score_info}"
                            )
                        
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Frame {frame_id}: Error - {str(e)[:50]}")
                continue
        
        elapsed_time = time.time() - start_time
        
        if best_pose is not None:
            if best_rmse <= max_rmse_threshold or (enforce_ankle_constraint and best_ankle_diff != float('inf') and best_ankle_diff*100 <= max_ankle_diff_cm):
                if self.logger:
                    ankle_info = ""
                    if enforce_ankle_constraint and best_ankle_diff != float('inf'):
                        ankle_status = "✓" if best_ankle_diff*100 <= max_ankle_diff_cm else "✗"
                        ankle_info = f", ankle diff: {best_ankle_diff*100:.2f}cm {ankle_status}"
                    rmse_status = "within threshold" if best_rmse <= max_rmse_threshold else "acceptable for ankle priority"
                    self.logger.info(
                        f"Best pose from frame {best_frame_id} "
                        f"(RMSE: {best_rmse:.2f} pixels, {rmse_status}{ankle_info})"
                    )
                    self.logger.info(f"Camera pose estimation time: {elapsed_time:.2f} seconds")
            else:
                if self.logger:
                    self.logger.warning(
                        f"Best pose from frame {best_frame_id} "
                        f"(RMSE: {best_rmse:.2f} pixels, exceeds threshold {max_rmse_threshold})"
                    )
                    self.logger.info(f"Camera pose estimation time: {elapsed_time:.2f} seconds")
        else:
            if self.logger:
                self.logger.warning(
                    f"Could not find valid camera pose "
                    f"(all candidates exceeded RMSE threshold {max_rmse_threshold})"
                )
                self.logger.info(f"Camera pose estimation time: {elapsed_time:.2f} seconds")
        
        return best_pose, best_rmse, best_frame_id, elapsed_time


def triangulate_with_fixed_pose(keypoints_2d_dict, cam_params_dict, wrld_cam_id, M2_fixed, 
                                normalize_scale_flag=False):
    """Triangulate 3D points using fixed camera pose."""
    from src.pipeline_utils import normalize_scale
    
    wrld_cam_name = list(cam_params_dict.keys())[wrld_cam_id]
    other_cam_name = [name for name in cam_params_dict.keys() if name != wrld_cam_name][0]
    
    wrld_cam_params = cam_params_dict[wrld_cam_name]
    other_cam_params = cam_params_dict[other_cam_name]
    
    K1, D1 = wrld_cam_params['K'], wrld_cam_params['distCoef']
    K2, D2 = other_cam_params['K'], other_cam_params['distCoef']
    
    M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    M2 = M2_fixed[0] if len(M2_fixed) > 0 else M1
    
    pts1 = keypoints_2d_dict[wrld_cam_name]
    pts2 = keypoints_2d_dict[other_cam_name]
    
    keypoints_3d = twov3d.triangulatePoints(K1, D1, M1, K2, D2, M2, pts1, pts2)
    
    # Normalize scale if requested (using default height)
    if normalize_scale_flag:
        keypoints_3d = normalize_scale(keypoints_3d, expected_height_m=1.7)
    
    return keypoints_3d

