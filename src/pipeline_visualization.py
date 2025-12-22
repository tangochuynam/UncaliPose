#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization functions for 3D pose estimation pipeline.
"""

import os
import json
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.pipeline_utils import JOINT_NAMES, SKELETON_EDGES, COLORS_2D
from src import basic_3d_operations as b3dop


def draw_skeleton_3d(ax, keypoints_3d, joint_names, frame_id=None, height=None):
    """
    Draw beautiful 3D skeleton in front view.
    
    Coordinate system:
    - X: horizontal (left-right)
    - Y: vertical (up-down, upward positive)
    - Z: depth (forward-backward)
    """
    ax.clear()
    
    valid_mask = ~np.isnan(keypoints_3d).any(axis=1)
    if not valid_mask.any():
        ax.text2D(0.5, 0.5, "No valid keypoints", transform=ax.transAxes, 
                 color='red', fontsize=20, ha='center')
        ax.set_title(f'Frame {frame_id} | No Data', fontsize=14, fontweight='bold', pad=20)
        return
    
    # Transform coordinates for correct front view display
    # Original keypoints are in world coordinate system (side camera as reference)
    # Our coordinate system: X=horizontal (left-right), Y=vertical (downward positive), Z=depth (forward-backward)
    # Desired display: X=horizontal, Y=vertical (upward positive), Z=depth
    # 
    # IMPORTANT: Matplotlib 3D axes are:
    #   - X: left-right (horizontal)
    #   - Y: forward-backward (depth)
    #   - Z: up-down (vertical)
    #
    # So we need to map:
    #   - Our X → matplotlib X (horizontal)
    #   - Our Y (vertical, downward) → matplotlib Z (vertical, upward) → need to negate
    #   - Our Z (depth) → matplotlib Y (depth)
    
    transformed_keypoints = keypoints_3d.copy()
    # Map: [X, Y, Z] → [X, Z, -Y]
    # X stays the same (horizontal)
    # Y (vertical, downward) → Z (vertical, upward) by negating
    # Z (depth) → Y (depth)
    transformed_keypoints[:, 0] = keypoints_3d[:, 0]  # X stays
    transformed_keypoints[:, 1] = keypoints_3d[:, 2]   # Z → Y (depth)
    transformed_keypoints[:, 2] = -keypoints_3d[:, 1]  # -Y → Z (vertical, upward)
    
    valid_transformed = transformed_keypoints[valid_mask]
    
    # Draw skeleton edges
    for edge in SKELETON_EDGES:
        if valid_mask[edge[0]] and valid_mask[edge[1]]:
            pts = transformed_keypoints[[edge[0], edge[1]]]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'b-', linewidth=2, alpha=0.7)
    
    # Draw joints
    ax.scatter(valid_transformed[:, 0], valid_transformed[:, 1], valid_transformed[:, 2],
              c='red', s=50, alpha=0.8, edgecolors='black', linewidths=1)
    
    # Set labels and limits
    # Note: After transformation, matplotlib axes are:
    #   X: horizontal (left-right) - our X
    #   Y: depth (forward-backward) - our Z
    #   Z: vertical (up-down, upward positive) - our -Y
    # But we label them according to the user's desired coordinate system:
    #   X: horizontal, Y: vertical, Z: depth
    ax.set_xlabel('X (m)', fontsize=11, fontweight='bold')  # Horizontal
    ax.set_ylabel('Z (m)', fontsize=11, fontweight='bold')  # Depth (matplotlib Y = our Z)
    ax.set_zlabel('Y (m)', fontsize=11, fontweight='bold')  # Vertical (matplotlib Z = our -Y)
    
    # Set equal aspect ratio
    if valid_transformed.size > 0:
        max_range = np.array([valid_transformed[:, 0].max() - valid_transformed[:, 0].min(),
                             valid_transformed[:, 1].max() - valid_transformed[:, 1].min(),
                             valid_transformed[:, 2].max() - valid_transformed[:, 2].min()]).max() / 2.0
        mid_x = (valid_transformed[:, 0].max() + valid_transformed[:, 0].min()) * 0.5
        mid_y = (valid_transformed[:, 1].max() + valid_transformed[:, 1].min()) * 0.5
        mid_z = (valid_transformed[:, 2].max() + valid_transformed[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Title (no height display)
    title = f'Frame {frame_id}' if frame_id is not None else '3D Pose'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Front view: elev=0 (horizontal), azim=0 (facing front)
    ax.view_init(elev=0, azim=0)


def draw_2d_keypoints_on_frame(frame, keypoints_2d, color=(0, 255, 0), thickness=2, radius=5):
    """Draw 2D keypoints on a video frame."""
    frame_copy = frame.copy()
    
    for i, (x, y) in enumerate(keypoints_2d):
        if not np.isnan(x) and not np.isnan(y):
            x, y = int(x), int(y)
            cv2.circle(frame_copy, (x, y), radius, color, -1)
            cv2.circle(frame_copy, (x, y), radius + 2, (255, 255, 255), thickness)
    
    # Draw skeleton
    for edge in SKELETON_EDGES:
        pt1_idx, pt2_idx = edge
        if (pt1_idx < len(keypoints_2d) and pt2_idx < len(keypoints_2d) and
            not np.isnan(keypoints_2d[pt1_idx]).any() and not np.isnan(keypoints_2d[pt2_idx]).any()):
            pt1 = (int(keypoints_2d[pt1_idx][0]), int(keypoints_2d[pt1_idx][1]))
            pt2 = (int(keypoints_2d[pt2_idx][0]), int(keypoints_2d[pt2_idx][1]))
            cv2.line(frame_copy, pt1, pt2, color, thickness)
    
    return frame_copy


def create_3d_visualization(video_path, output_file=None, fps=30, logger=None):
    """Create 3D keypoint visualization video."""
    video_path = os.path.abspath(video_path)
    video_name = os.path.basename(video_path)
    data_dir = os.path.dirname(video_path)
    results_dir = os.path.join(data_dir, 'processed', video_name, 'results')
    
    # Try to load from combined JSON first, then fall back to individual files
    combined_file = os.path.join(results_dir, 'all_frames_3d_keypoints.json')
    if os.path.exists(combined_file):
        with open(combined_file, 'r') as f:
            combined_data = json.load(f)
        frame_data_list = combined_data.get('frames', [])
        if logger:
            logger.info(f"Loading {len(frame_data_list)} frames from combined JSON")
    else:
        # Fall back to individual files
        frame_files = sorted(glob.glob(os.path.join(results_dir, 'frame_*_3d_keypoints.json')))
        if len(frame_files) == 0:
            if logger:
                logger.error("No 3D keypoint files found!")
            return None
        frame_data_list = []
        for frame_file in frame_files:
            with open(frame_file, 'r') as f:
                frame_data_list.append(json.load(f))
    
    if len(frame_data_list) == 0:
        if logger:
            logger.error("No frame data found!")
        return None
    
    if logger:
        logger.info(f"Creating 3D visualization video")
        logger.info(f"Loading {len(frame_data_list)} frames...")
    
    if output_file is None:
        output_file = os.path.join(results_dir, '3d_keypoints_animation.mp4')
    
    # Setup figure
    dpi = 100
    width, height = 1920, 1080
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi, facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    
    # Video writer - try multiple codecs for compatibility
    codecs = ['mp4v', 'XVID', 'MJPG', 'X264']
    out = None
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_file, fourcc, fps, (int(width), int(height)))
        if out.isOpened():
            if logger:
                logger.info(f"Using codec: {codec}")
            break
    
    if out is None or not out.isOpened():
        if logger:
            logger.error("Could not initialize video writer with any codec")
        plt.close(fig)
        return None
    
    last_valid_keypoints = None
    frames_written = 0
    
    for frame_idx, data in enumerate(frame_data_list):
        if (frame_idx + 1) % 50 == 0 and logger:
            logger.info(f"Processing frame {frame_idx+1}/{len(frame_data_list)}...")
        
        frame_id = data['frame_id']
        person_data = data['keypoints_by_person'].get('person_0', {})
        keypoints_3d = np.array(person_data.get('joints', []))
        keypoints_valid = person_data.get('keypoints_valid', True)
        
        if keypoints_valid and len(keypoints_3d) > 0:
            draw_skeleton_3d(ax, keypoints_3d, JOINT_NAMES, frame_id=frame_id)
            last_valid_keypoints = keypoints_3d
        elif last_valid_keypoints is not None:
            draw_skeleton_3d(ax, last_valid_keypoints, JOINT_NAMES, frame_id=frame_id)
        else:
            ax.clear()
            ax.text2D(0.5, 0.5, "Invalid Frame", transform=ax.transAxes, 
                     color='red', fontsize=20, ha='center')
            ax.set_title(f'Frame {frame_id} | INVALID', fontsize=14, fontweight='bold', pad=20, color='red')
        
        # Render to numpy array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Convert RGB to BGR for OpenCV
        buf = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        
        # Resize if needed
        if buf.shape[:2] != (int(height), int(width)):
            buf = cv2.resize(buf, (int(width), int(height)))
        
        # Write frame
        out.write(buf)
        frames_written += 1
    
    out.release()
    plt.close(fig)
    
    if logger:
        logger.info(f"Wrote {frames_written} frames to video")
        logger.info(f"3D visualization video saved: {output_file}")
    
    return output_file


def create_debug_visualization(video_path, output_file=None, fps=30, logger=None):
    """Create debug visualization showing original vs reprojected 2D keypoints."""
    video_path = os.path.abspath(video_path)
    video_name = os.path.basename(video_path)
    data_dir = os.path.dirname(video_path)
    results_dir = os.path.join(data_dir, 'processed', video_name, 'results')
    
    # Try to load from combined JSON first, then fall back to individual files
    combined_file = os.path.join(results_dir, 'all_frames_3d_keypoints.json')
    if os.path.exists(combined_file):
        with open(combined_file, 'r') as f:
            combined_data = json.load(f)
        frame_data_list = combined_data.get('frames', [])
        if logger:
            logger.info(f"Loading {len(frame_data_list)} frames from combined JSON")
    else:
        # Fall back to individual files
        frame_files = sorted(glob.glob(os.path.join(results_dir, 'frame_*_3d_keypoints.json')))
        if len(frame_files) == 0:
            if logger:
                logger.error("No 3D keypoint files found!")
            return None
        frame_data_list = []
        for frame_file in frame_files:
            with open(frame_file, 'r') as f:
                frame_data_list.append(json.load(f))
    
    if len(frame_data_list) == 0:
        if logger:
            logger.error("No frame data found!")
        return None
    
    if logger:
        logger.info("Creating DEBUG visualization (2D keypoints comparison)")
    
    # Video directory is the video_path itself
    video_dir = video_path
    
    video_files = {}
    video_caps = {}
    
    for cam_name in ['front', 'side']:
        # Try to find video file - check multiple patterns
        video_file = None
        
        # Pattern 1: Direct name match
        direct_paths = [
            os.path.join(video_dir, f'{cam_name}.mp4'),
            os.path.join(video_dir, f'{cam_name}.avi'),
        ]
        for path in direct_paths:
            if os.path.exists(path):
                video_file = path
                break
        
        # Pattern 2: Search for files containing camera identifier
        if not video_file:
            if cam_name == 'front':
                patterns = [
                    os.path.join(video_dir, '*44CA7CF5*.mp4'),
                    os.path.join(video_dir, '*44CA7CF5*.avi'),
                ]
            else:  # side
                patterns = [
                    os.path.join(video_dir, '*FBD2D8A3*.mp4'),
                    os.path.join(video_dir, '*FBD2D8A3*.avi'),
                ]
            
            for pattern in patterns:
                matches = glob.glob(pattern)
                if matches:
                    video_file = matches[0]
                    break
        
        # Pattern 3: Generic search
        if not video_file:
            patterns = [
                os.path.join(video_dir, f'*{cam_name}*.mp4'),
                os.path.join(video_dir, f'*{cam_name}*.avi'),
            ]
            for pattern in patterns:
                matches = glob.glob(pattern)
                if matches:
                    video_file = matches[0]
                    break
        
        if video_file and os.path.exists(video_file):
            video_files[cam_name] = video_file
            video_caps[cam_name] = cv2.VideoCapture(video_file)
            if logger:
                logger.info(f"Found {cam_name} video: {os.path.basename(video_file)}")
        else:
            if logger:
                logger.warning(f"{cam_name} video not found, skipping debug visualization")
            return None
    
    # Load dataset for camera parameters
    from src.pipeline_utils import create_default_config
    from src.dataset.twoview_custom import TwoViewCustom
    
    config = create_default_config()
    dataset = TwoViewCustom(None, None, config, video_path=video_path)
    dataset.prepareDataFromKeypoints()
    
    # Load saved camera pose if available (for single pose mode)
    saved_camera_pose = None
    camera_pose_file = os.path.join(data_dir, 'processed', video_name, 'camera_pose.json')
    if os.path.exists(camera_pose_file):
        try:
            with open(camera_pose_file, 'r') as f:
                pose_data = json.load(f)
                saved_camera_pose = [np.array(pose_data['pose'])]
                if logger:
                    logger.info(f"Loaded saved camera pose for reprojection")
        except Exception as e:
            if logger:
                logger.warning(f"Could not load saved camera pose: {e}")
    
    if output_file is None:
        output_file = os.path.join(results_dir, 'debug_2d_keypoints_comparison.mp4')
    
    # Get video properties
    ret, frame = video_caps['front'].read()
    if not ret:
        if logger:
            logger.error("Could not read video frames")
        return None
    
    height, width = frame.shape[:2]
    video_caps['front'].set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width * 2, height * 2))
    
    if logger:
        logger.info(f"Processing {len(frame_data_list)} frames...")
    
    for frame_idx, data in enumerate(frame_data_list):
        if (frame_idx + 1) % 50 == 0 and logger:
            logger.info(f"Frame {frame_idx+1}/{len(frame_data_list)}...")
        
        frame_id = data['frame_id']
        frames_vis = {}
        
        # Read video frames and create visualizations for each camera
        for cam_name in ['front', 'side']:
            video_caps[cam_name].set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = video_caps[cam_name].read()
            if not ret:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Load original 2D keypoints
            pose2d_file = os.path.join(data_dir, 'processed', video_name, 'pose2d_label', cam_name, f'{frame_id:08d}.json')
            original_kpts = None
            if os.path.exists(pose2d_file):
                with open(pose2d_file, 'r') as f:
                    pose_data = json.load(f)
                    if 'bodies' in pose_data and len(pose_data['bodies']) > 0:
                        original_kpts = np.array(pose_data['bodies'][0]['joints'])
            
            # Get reprojected 2D keypoints
            person_data = data['keypoints_by_person'].get('person_0', {})
            keypoints_3d = np.array(person_data.get('joints', []))
            
            reprojected_kpts = None
            if len(keypoints_3d) > 0:
                cam_params = dataset.cam_params_dict[cam_name]
                K = cam_params['K']
                D = cam_params['distCoef']
                
                # Get camera pose - use saved pose if available (consistent with RMSE calculation)
                wrld_cam_id = 1 if data['world_camera'] == 'side' else 0
                wrld_cam_name = data['world_camera']
                
                if cam_name == wrld_cam_name:
                    # World camera - use identity pose
                    R = np.eye(3)
                    t = np.zeros(3)
                else:
                    # Other camera - prefer saved pose (for single pose mode), otherwise use frame-specific pose
                    if saved_camera_pose is not None and len(saved_camera_pose) > 0:
                        M = saved_camera_pose[0]
                        R = M[:, :3]
                        t = M[:, 3]
                    elif 'camera_poses_dict' in data and cam_name in data['camera_poses_dict']:
                        M = np.array(data['camera_poses_dict'][cam_name]['pose_matrix'])
                        R = M[:, :3]
                        t = M[:, 3]
                    else:
                        R = np.eye(3)
                        t = np.zeros(3)
                
                # Project 3D to 2D
                valid_mask = ~np.isnan(keypoints_3d).any(axis=1)
                if valid_mask.any():
                    kpts_3d_valid = keypoints_3d[valid_mask]
                    kpts_2d_proj = b3dop.projectPoints(kpts_3d_valid.T, K, D, R, t).T
                    
                    reprojected_kpts = np.full((len(keypoints_3d), 2), np.nan)
                    reprojected_kpts[valid_mask] = kpts_2d_proj
            
            # Create two visualizations: one with original, one with reprojected
            frame_original = frame.copy()
            frame_reprojected = frame.copy()
            
            # Draw original keypoints only (green)
            if original_kpts is not None:
                frame_original = draw_2d_keypoints_on_frame(frame_original, original_kpts, 
                                                      color=COLORS_2D['original'], thickness=2, radius=4)
            
            # Draw reprojected keypoints and error lines (red + blue lines)
            if reprojected_kpts is not None:
                frame_reprojected = draw_2d_keypoints_on_frame(frame_reprojected, reprojected_kpts,
                                                       color=COLORS_2D['reprojected'], thickness=2, radius=3)
                
                # Draw error lines connecting original to reprojected
                if original_kpts is not None:
                    for i in range(min(len(original_kpts), len(reprojected_kpts))):
                        if (not np.isnan(original_kpts[i]).any() and 
                            not np.isnan(reprojected_kpts[i]).any()):
                            pt1 = (int(original_kpts[i][0]), int(original_kpts[i][1]))
                            pt2 = (int(reprojected_kpts[i][0]), int(reprojected_kpts[i][1]))
                            cv2.line(frame_reprojected, pt1, pt2, COLORS_2D['error'], 1)
            
            # Add text to both frames
            rmse_data = person_data.get('reprojection_rmse', {})
            rmse = rmse_data.get('per_camera_rmse', {}).get(cam_name, None)
            rmse_text = f"RMSE: {rmse:.2f}px" if rmse is not None else "RMSE: N/A"
            
            # Original frame text
            cv2.putText(frame_original, f"{cam_name.upper()} - ORIGINAL | Frame {frame_id}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame_original, "Green: Input 2D Keypoints", (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Reprojected frame text
            cv2.putText(frame_reprojected, f"{cam_name.upper()} - REPROJECTED | Frame {frame_id}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame_reprojected, rmse_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame_reprojected, "Red: 3D->2D Reprojected | Blue: Error Lines", (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            frames_vis[cam_name] = {
                'original': frame_original,
                'reprojected': frame_reprojected
            }
        
        # Create 2x2 grid: front original | front reprojected, side original | side reprojected
        top_row = np.hstack([frames_vis['front']['original'], frames_vis['front']['reprojected']])
        bottom_row = np.hstack([frames_vis['side']['original'], frames_vis['side']['reprojected']])
        combined = np.vstack([top_row, bottom_row])
        out.write(combined)
    
    # Release resources
    for cam_name in video_caps:
        video_caps[cam_name].release()
    out.release()
    
    if logger:
        logger.info(f"Debug visualization saved: {output_file}")
    return output_file

