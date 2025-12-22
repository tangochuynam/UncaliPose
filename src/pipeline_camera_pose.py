#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Camera pose management for 3D pose estimation pipeline.
"""

import os
import json
import time
import numpy as np
from src.multiview_3d import solveMultiView3DHumanPoses
from src.bundle_adjustment import bundleAdjustmentWrapper
from src import twoview_3d as twov3d
from src.pipeline_utils import (
    calculate_person_height, calculate_reprojection_rmse,
    scale_camera_pose_by_height, normalize_scale,
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
        """Load saved camera pose."""
        if not os.path.exists(self.pose_file):
            return None, None, None, None
        
        try:
            with open(self.pose_file, 'r') as f:
                data = json.load(f)
                pose = np.array(data['pose'])
                # Ensure it's in the right format (list of 3x4 matrices)
                if len(pose.shape) == 2:
                    pose = [pose]  # Wrap in list if single matrix
                return pose, data.get('frame_id', None), data.get('rmse', None), data.get('estimation_time', None)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            if self.logger:
                self.logger.warning(f"Error loading camera pose file: {e}")
            return None, None, None, None
    
    def save(self, pose, frame_id, rmse, estimation_time=None):
        """Save camera pose."""
        os.makedirs(self.pose_dir, exist_ok=True)
        
        # Convert pose to list (handle both numpy array and list)
        if isinstance(pose, list):
            pose_list = []
            for p in pose:
                if isinstance(p, np.ndarray):
                    pose_list.append(p.tolist())
                else:
                    pose_list.append(p)
            if len(pose_list) > 0:
                pose_list = pose_list[0]
        else:
            pose_list = pose.tolist()
        
        with open(self.pose_file, 'w') as f:
            json.dump({
                'pose': pose_list,
                'frame_id': frame_id,
                'rmse': float(rmse) if rmse is not None else None,
                'estimation_time': float(estimation_time) if estimation_time is not None else None
            }, f, indent=2)
        
        if self.logger:
            self.logger.info(f"Camera pose saved to: {self.pose_file}")
    
    def find_best_pose(self, dataset, config, wrld_cam_id, candidate_frame_ids=None, 
                       max_candidates=5, expected_height_m=None, max_rmse_threshold=4.0,
                       target_rmse=2.0, sample_frames=True):
        """
        Find best camera pose by trying multiple frames.
        
        Args:
            dataset: Dataset instance
            config: Configuration dictionary
            wrld_cam_id: World camera ID
            candidate_frame_ids: List of frame IDs to try (None = auto-select)
            max_candidates: Maximum number of candidates
            expected_height_m: Expected person height in meters (not used, kept for compatibility)
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
        
        pose_files = os.listdir(os.path.join(dataset.pose2d_file_dir, dataset.cam_names[0]))
        n_frames = len([f for f in pose_files if f.endswith('.json')])
        
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
        
        best_pose = None
        best_rmse = float('inf')
        best_frame_id = None
        
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
                
                Pts_BA_init, M2s_BA_init = bundleAdjustmentWrapper(
                    BA_input_init, fix_cam_pose=False, wrld_cam_id=wrld_cam_id,
                    max_iter=config['bundle']['max_iter'], verbose=False
                )
                
                valid_mask = ~np.isnan(Pts_BA_init).any(axis=1)
                if valid_mask.any():
                    max_coord = np.abs(Pts_BA_init[valid_mask]).max()
                    if max_coord > 10.0:
                        continue
                
                # NOTE: Removed height-based scale optimization to match original author's implementation
                # The original code does not use height constraints during camera pose estimation
                # Scale normalization happens later during processing if needed
                
                keypoints_2d_dict = {
                    cam_name: pts_corresp_dict[cam_name]['keypoints']
                    for cam_name in dataset.cam_names
                }
                
                rmse_dict, total_rmse = calculate_reprojection_rmse(
                    Pts_BA_init, keypoints_2d_dict, dataset.cam_params_dict,
                    wrld_cam_id, M2s_BA_init
                )
                
                # Check if RMSE is acceptable and better than current best
                # Prioritize frames with RMSE < target_rmse
                is_target = total_rmse is not None and total_rmse <= target_rmse
                is_acceptable = total_rmse is not None and total_rmse <= max_rmse_threshold
                is_better = total_rmse is not None and total_rmse < best_rmse
                
                # Prefer target RMSE frames, but accept any within threshold
                if is_acceptable and (is_better or (is_target and best_rmse > target_rmse)):
                    best_rmse = total_rmse
                    best_pose = M2s_BA_init
                    best_frame_id = frame_id
                    if self.logger:
                        target_status = "✓✓ (target achieved)" if is_target else "✓ (within threshold)"
                        self.logger.info(
                            f"Frame {frame_id}: RMSE = {total_rmse:.2f} pixels {target_status}"
                        )
                else:
                    if total_rmse is not None:
                        status = "✗ (exceeds threshold)" if total_rmse > max_rmse_threshold else ""
                        if self.logger:
                            self.logger.info(
                                f"Frame {frame_id}: RMSE = {total_rmse:.2f} pixels {status}"
                            )
                        
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Frame {frame_id}: Error - {str(e)[:50]}")
                continue
        
        elapsed_time = time.time() - start_time
        
        if best_pose is not None:
            if best_rmse <= max_rmse_threshold:
                if self.logger:
                    self.logger.info(
                        f"Best pose from frame {best_frame_id} "
                        f"(RMSE: {best_rmse:.2f} pixels, within threshold)"
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
                                normalize_scale_flag=False, expected_height_m=1.7):
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
    
    # Normalize scale if requested
    if normalize_scale_flag:
        keypoints_3d = normalize_scale(keypoints_3d, expected_height_m=expected_height_m)
    
    return keypoints_3d

