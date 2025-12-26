#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Frame processing functions for 3D pose estimation pipeline.
"""

import os
import json
import glob
import numpy as np
from src.multiview_3d import solveMultiView3DHumanPoses
from src.bundle_adjustment import bundleAdjustmentWrapper
from src.pipeline_utils import (
    JOINT_NAMES, calculate_person_height, calculate_reprojection_rmse,
    normalize_scale
)
from src.pipeline_camera_pose import triangulate_with_fixed_pose


def process_single_frame(frame_id, dataset, config, wrld_cam_id, estimated_pose=None,
                        use_single_pose=True, logger=None):
    """
    Process a single frame to generate 3D keypoints.
    
    Args:
        frame_id: Frame ID to process
        dataset: Dataset instance
        config: Configuration dictionary
        wrld_cam_id: World camera ID
        estimated_pose: Pre-estimated camera pose (if using single pose)
        use_single_pose: Whether to use single pose for all frames
        logger: Optional logger instance
    
    Returns:
        result_data: Dictionary containing frame results
    """
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
        
        # Triangulate
        if use_single_pose and estimated_pose is not None:
            keypoints_2d_dict = {
                cam_name: np.array(pts_corresp_dict[cam_name]['keypoints'])
                for cam_name in dataset.cam_names
            }
            
            Pts_BA = triangulate_with_fixed_pose(
                keypoints_2d_dict, dataset.cam_params_dict, wrld_cam_id, estimated_pose,
                normalize_scale_flag=False  # Use original scale from triangulation
            )
            
            M2s_BA = estimated_pose
            wrld_cam_name = list(dataset.cam_names)[wrld_cam_id]
            
            # Optional bundle adjustment
            try:
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
                
                Pts_BA, _ = bundleAdjustmentWrapper(
                    BA_input, fix_cam_pose=True, wrld_cam_id=wrld_cam_id,
                    max_iter=config['bundle']['max_iter'], verbose=False
                )
                # NOTE: Removed scale normalization after bundle adjustment
                # This matches the original author's implementation and prevents RMSE issues
                # The camera pose and 3D points must stay at the same scale
            except Exception as e:
                if logger:
                    logger.debug(f"Bundle adjustment failed for frame {frame_id}: {e}")
        else:
            Pts, BA_input, wrld_cam_name = solveMultiView3DHumanPoses(
                pts_corresp_dict, dataset.cam_params_dict, n_persons,
                Pts_prev=None, wrld_cam_id=wrld_cam_id, verbose=False
            )
            
            valid_mask = ~np.isnan(Pts).any(axis=1)
            if valid_mask.any():
                max_coord = np.abs(Pts[valid_mask]).max()
                if max_coord > 10.0:
                    Pts_BA = Pts
                    M2s_BA = BA_input['M2s'] if 'M2s' in BA_input else []
                else:
                    Pts_BA, M2s_BA = bundleAdjustmentWrapper(
                        BA_input, fix_cam_pose=False, wrld_cam_id=wrld_cam_id,
                        max_iter=config['bundle']['max_iter'], verbose=False
                    )
            else:
                Pts_BA = Pts
                M2s_BA = BA_input['M2s'] if 'M2s' in BA_input else []
        
        # Validate
        valid_mask_pts = ~np.isnan(Pts_BA).any(axis=1)
        if valid_mask_pts.any():
            valid_pts = Pts_BA[valid_mask_pts]
            max_coord = np.abs(valid_pts).max()
        else:
            max_coord = 0
        
        has_invalid_keypoints = max_coord > 10.0
        
        # Calculate RMSE
        keypoints_2d_dict = {
            cam_name: np.array(pts_corresp_dict[cam_name]['keypoints'])
            for cam_name in dataset.cam_names
        }
        
        rmse_dict, total_rmse = calculate_reprojection_rmse(
            Pts_BA, keypoints_2d_dict, dataset.cam_params_dict,
            wrld_cam_id, M2s_BA if use_single_pose and estimated_pose is not None else None
        )
        
        # Log RMSE if high
        if logger:
            logger.log_rmse(frame_id, rmse_dict, total_rmse)
            if total_rmse and total_rmse > 10.0:
                logger.log_high_rmse(frame_id, rmse_dict, threshold=10.0)
        
        # Calculate height
        n_joints = len(Pts_BA) // n_persons if n_persons > 0 else 16
        frame_heights = []
        
        for person_id in range(n_persons):
            start_idx = person_id * n_joints
            end_idx = (person_id + 1) * n_joints
            person_keypoints = Pts_BA[start_idx:end_idx]
            
            height = calculate_person_height(person_keypoints, JOINT_NAMES, validate=True)
            if height is not None and not has_invalid_keypoints:
                frame_heights.append(height)
        
        # Prepare result data
        camera_poses_dict = {}
        cam_names_list = list(dataset.cam_names)
        
        for i, cam_name in enumerate(cam_names_list):
            if i == wrld_cam_id:
                camera_poses_dict[cam_name] = {
                    'pose_matrix': np.eye(4)[:3, :].tolist(),
                    'rotation_matrix': np.eye(3).tolist(),
                    'translation_vector': [0.0, 0.0, 0.0],
                    'is_reference': True
                }
            else:
                if len(M2s_BA) > 0:
                    M2 = M2s_BA[0]
                    camera_poses_dict[cam_name] = {
                        'pose_matrix': M2.tolist(),
                        'rotation_matrix': M2[:, :3].tolist(),
                        'translation_vector': M2[:, 3].tolist(),
                        'is_reference': False,
                        'relative_to': cam_names_list[wrld_cam_id]
                    }
        
        result_data = {
            'frame_id': int(frame_id),
            'n_persons': int(n_persons),
            'n_joints': int(n_joints),
            'keypoints_3d': Pts_BA.tolist(),
            'keypoints_by_person': {},
            'heights': {},
            'camera_poses': [M2.tolist() for M2 in M2s_BA],
            'camera_poses_dict': camera_poses_dict,
            'world_camera': str(wrld_cam_name),
            'keypoints_valid': not has_invalid_keypoints,
            'max_coordinate': float(max_coord) if valid_mask_pts.any() else None,
            'reprojection_rmse': {
                'total_rmse_pixels': float(total_rmse) if total_rmse is not None else None,
                'per_camera_rmse': {
                    cam: float(rmse) if rmse is not None else None
                    for cam, rmse in rmse_dict.items()
                }
            },
            'frame_heights': [float(h) for h in frame_heights]
        }
        
        for person_id in range(n_persons):
            start_idx = person_id * n_joints
            end_idx = (person_id + 1) * n_joints
            person_keypoints = Pts_BA[start_idx:end_idx]
            
            height = calculate_person_height(person_keypoints, JOINT_NAMES, validate=True)
            if has_invalid_keypoints:
                height = None
            
            result_data['keypoints_by_person'][f'person_{person_id}'] = {
                'joints': person_keypoints.tolist(),
                'joint_names': JOINT_NAMES,
                'height_m': float(height) if height is not None else None,
                'height_cm': float(height * 100) if height is not None else None,
                'keypoints_valid': not has_invalid_keypoints,
                'reprojection_rmse': result_data['reprojection_rmse']
            }
            result_data['heights'][f'person_{person_id}'] = {
                'height_m': float(height) if height is not None else None,
                'height_cm': float(height * 100) if height is not None else None
            }
        
        return result_data
        
    except Exception as e:
        if logger:
            logger.error(f"Error processing frame {frame_id}: {e}")
        return {
            'frame_id': int(frame_id),
            'success': False,
            'error': str(e)
        }


def process_all_frames(dataset, config, wrld_cam_id=1, use_saved_pose=True, 
                      use_single_pose=True, logger=None):
    """
    Process all frames and generate 3D keypoints.
    
    Args:
        dataset: Dataset instance
        config: Configuration dictionary
        wrld_cam_id: World camera ID
        use_saved_pose: Whether to use saved camera pose
        use_single_pose: Whether to use single pose for all frames
        logger: Optional logger instance
    
    Returns:
        results_dir: Directory containing results
    """
    # Prepare data first
    if logger:
        logger.info("Preparing data from keypoints...")
    dataset.prepareDataFromKeypoints()
    
    # Determine number of frames
    pose_dir = os.path.join(dataset.pose2d_file_dir, dataset.cam_names[0])
    if os.path.exists(pose_dir):
        pose_files = glob.glob(os.path.join(pose_dir, '*.json'))
        n_frames = len(pose_files)
    else:
        # Fallback: check keypoints directly
        keypoint_files = glob.glob(os.path.join(dataset.keypoints_dir, dataset.cam_names[0], '*.json'))
        if not keypoint_files:
            # Try video directory (where keypoints JSON files are stored)
            keypoint_files = glob.glob(os.path.join(dataset.video_dir, '*_swing_infor.json'))
            if not keypoint_files:
                keypoint_files = glob.glob(os.path.join(dataset.video_dir, '*.json'))
        
        if keypoint_files:
            with open(keypoint_files[0], 'r') as f:
                kpts_data = json.load(f)
                if 'keypoints_2d' in kpts_data and isinstance(kpts_data['keypoints_2d'], list):
                    n_frames = len(kpts_data['keypoints_2d'])
                else:
                    n_frames = 1
        else:
            # Check existing results
            results_dir = os.path.join(dataset.data_dir, 'processed', dataset.video_name, 'results')
            if os.path.exists(results_dir):
                existing_frames = glob.glob(os.path.join(results_dir, 'frame_*_3d_keypoints.json'))
                n_frames = len(existing_frames) if existing_frames else 360
            else:
                n_frames = 360  # Default assumption
    
    if logger:
        logger.info(f"Processing {n_frames} frames")
    
    # Load or estimate camera pose
    from src.pipeline_camera_pose import CameraPoseManager
    
    # Get video_path from dataset (use video_dir which is the actual video path)
    video_path = dataset.video_dir
    pose_manager = CameraPoseManager(video_path, logger=logger)
    estimated_pose = None
    pose_frame_id = None
    pose_rmse = None
    
    # Check if we should re-estimate (from config or parameter)
    pose_config = config.get('camera_pose', {}) if config else {}
    reestimate_pose = pose_config.get('reestimate_pose', False)
    
    if logger:
        logger.info(f"Camera pose config: reestimate_pose={reestimate_pose}, use_saved_pose={use_saved_pose}")
        logger.info(f"Camera pose file path: {pose_manager.pose_file}")
        logger.info(f"Camera pose file exists: {os.path.exists(pose_manager.pose_file)}")
    
    # If reestimate_pose is True, skip loading saved pose
    if reestimate_pose:
        if logger:
            logger.info("Re-estimation requested: ignoring saved camera pose (if exists)")
        # Don't load saved pose, will estimate new one below
    elif use_saved_pose:
        # Try to load saved pose
        estimated_pose, pose_frame_id, pose_rmse, pose_time = pose_manager.load()
        if estimated_pose is not None:
            if logger:
                logger.info(f"Loaded saved camera pose (frame {pose_frame_id}, RMSE: {pose_rmse:.2f}px)")
                if pose_time:
                    logger.info(f"Original estimation time: {pose_time:.2f} seconds")
            use_single_pose = True
        else:
            if logger:
                logger.warning(f"Saved camera pose not found or could not be loaded from: {pose_manager.pose_file}")
                logger.warning("Will proceed to estimate new camera pose")
    
    # Only estimate new pose if we don't have one and reestimate_pose is False
    # If reestimate_pose is True, we always estimate (even if saved pose exists)
    if estimated_pose is None and use_single_pose:
        if logger:
            logger.info("Finding best camera pose...")
        
        # Get camera pose config (with defaults) - already loaded above
        max_candidates = pose_config.get('max_candidates', 40)
        max_rmse_threshold = pose_config.get('max_rmse_threshold', 4.0)
        target_rmse = pose_config.get('target_rmse', 2.0)
        enforce_ankle_constraint = pose_config.get('enforce_ankle_constraint', True)
        max_ankle_diff_cm = pose_config.get('max_ankle_diff_cm', 5.0)
        ankle_weight = pose_config.get('ankle_weight', 2.0)
        
        if logger:
            logger.info(f"RMSE threshold: < {max_rmse_threshold} pixels")
            logger.info(f"Max candidates: {max_candidates} frames")
        
        estimated_pose, pose_rmse, pose_frame_id, pose_time = pose_manager.find_best_pose(
            dataset, config, wrld_cam_id,
            max_rmse_threshold=max_rmse_threshold,
            max_candidates=max_candidates,
            target_rmse=target_rmse,
            enforce_ankle_constraint=enforce_ankle_constraint,
            max_ankle_diff_cm=max_ankle_diff_cm,
            ankle_weight=ankle_weight
        )
        if logger and pose_time:
            logger.info(f"Camera pose estimation completed in {pose_time:.2f} seconds")
        
        if estimated_pose is not None:
            # Get save_intrinsics flag from config (default: False - only save estimated pose)
            save_intrinsics = pose_config.get('save_intrinsics', False)
            pose_manager.save(estimated_pose, pose_frame_id, pose_rmse, pose_time, 
                            dataset=dataset, wrld_cam_id=wrld_cam_id, include_intrinsics=save_intrinsics)
        else:
            if logger:
                logger.warning("Falling back to per-frame pose estimation")
            use_single_pose = False
    
    # Process frames
    results_dir = os.path.join(dataset.data_dir, 'processed', dataset.video_name, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = []
    heights = []
    all_frame_data = []  # Collect all frame data for combined JSON
    
    for frame_id in range(n_frames):
        if (frame_id + 1) % 50 == 0 or frame_id == 0:
            print(f"Processing frame {frame_id+1}/{n_frames}...", end=' ')
            if logger:
                logger.info(f"Processing frame {frame_id+1}/{n_frames}...")
        elif (frame_id + 1) % 10 == 0:
            print(".", end='', flush=True)
        
        result_data = process_single_frame(
            frame_id, dataset, config, wrld_cam_id, estimated_pose,
            use_single_pose, logger
        )
        
        if 'error' in result_data:
            all_results.append(result_data)
            continue
        
        # Don't save individual frame files - only collect for combined JSON
        all_frame_data.append(result_data)
        
        # Collect statistics
        frame_heights = result_data.get('frame_heights', [])
        heights.extend(frame_heights)
        
        if (frame_id + 1) % 50 == 0 or frame_id == 0:
            if frame_heights:
                height_cm = frame_heights[0] * 100
                total_rmse = result_data.get('reprojection_rmse', {}).get('total_rmse_pixels')
                if total_rmse is not None:
                    print(f"RMSE: {total_rmse:.2f}px ✓ (height: {height_cm:.1f}cm)")
                else:
                    print(f"✓ (height: {height_cm:.1f}cm)")
            else:
                print()
        
        all_results.append({
            'frame_id': frame_id,
            'success': True,
            'n_persons': result_data.get('n_persons', 0),
            'heights': frame_heights
        })
    
    # Save summary
    if logger:
        logger.info(f"Processing complete!")
        logger.info(f"Successful frames: {sum(1 for r in all_results if r.get('success', False))}/{len(all_results)}")
        if heights:
            logger.info(f"Average height: {np.mean(heights)*100:.1f} cm")
            logger.info(f"Height range: {np.min(heights)*100:.1f} - {np.max(heights)*100:.1f} cm")
    
    summary_file = os.path.join(results_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'total_frames': n_frames,
            'successful_frames': sum(1 for r in all_results if r.get('success', False)),
            'average_height_cm': float(np.mean(heights)*100) if heights else None,
            'height_range_cm': [float(np.min(heights)*100), float(np.max(heights)*100)] if heights else None,
            'camera_pose_frame_id': pose_frame_id,
            'camera_pose_rmse': float(pose_rmse) if pose_rmse is not None else None
        }, f, indent=2)
    
    if logger:
        logger.info(f"Summary saved to: {summary_file}")
    
    # Save combined JSON file (instead of individual files)
    try:
        combined_file = os.path.join(results_dir, 'all_frames_3d_keypoints.json')
        combined_data = {
            'n_frames': len(all_frame_data),
            'frames': all_frame_data
        }
        with open(combined_file, 'w') as f:
            json.dump(combined_data, f, indent=2)
        if logger:
            logger.info(f"Combined all frames into: {combined_file}")
    except Exception as e:
        if logger:
            logger.warning(f"Could not save combined JSON file: {e}")
    
    return results_dir

