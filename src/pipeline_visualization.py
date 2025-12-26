#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization functions for 3D pose estimation pipeline.
"""

import os
import json
import glob
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.pipeline_utils import JOINT_NAMES, SKELETON_EDGES, COLORS_2D
from src import basic_3d_operations as b3dop


def draw_skeleton_3d(ax, keypoints_3d, joint_names, frame_id=None, height=None, view_azim=0.0, view_elev=0.0, 
                    radius_padding_multiplier=1.2, radius_minimum=0.3):
    """
    Draw beautiful 3D skeleton in front view.
    
    Coordinate system:
    - X: horizontal (left-right)
    - Y: vertical (up-down, upward positive)
    - Z: depth (forward-backward)
    
    Args:
        view_azim: Azimuth angle for view (default: 0.0 for front view)
        view_elev: Elevation angle for view (default: 0.0 for horizontal view)
        radius_padding_multiplier: Multiplier for padding around skeleton (default: 1.2)
                                   Lower values = skeleton appears larger
        radius_minimum: Minimum RADIUS for axis limits (default: 0.3)
                        Lower values = skeleton appears larger
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
    
    # Get hip center (index 6 = center_hip) for axis limits
    hip_idx = 6  # center_hip index
    if hip_idx < len(keypoints_3d) and not np.isnan(keypoints_3d[hip_idx]).any():
        hip_original = keypoints_3d[hip_idx]
        # Transform hip coordinates: [X, Y, Z] → [X, Z, -Y]
        xroot = hip_original[0]  # X stays the same
        yroot = hip_original[2]  # Z → Y (depth)
        zroot = -hip_original[1]  # -Y → Z (vertical, upward)
    else:
        # Fallback: use center of valid keypoints
        if valid_transformed.size > 0:
            xroot = np.nanmean(valid_transformed[:, 0])
            yroot = np.nanmean(valid_transformed[:, 1])
            zroot = np.nanmean(valid_transformed[:, 2])
        else:
            xroot, yroot, zroot = 0.0, 0.0, 0.0
    
    # Calculate dynamic RADIUS based on actual keypoint range
    # Use a minimum of 0.7 but scale up if keypoints span larger area
    if valid_transformed.size > 0:
        # Calculate the maximum distance from hip center to any keypoint
        distances_from_hip = np.sqrt(
            (valid_transformed[:, 0] - xroot)**2 +
            (valid_transformed[:, 1] - yroot)**2 +
            (valid_transformed[:, 2] - zroot)**2
        )
        max_distance = np.nanmax(distances_from_hip) if len(distances_from_hip) > 0 else 0.0
        
        # Use configurable padding multiplier and minimum radius
        # Lower values = skeleton appears larger (tighter fit)
        # Higher values = skeleton appears smaller (more padding)
        RADIUS = max(radius_minimum, max_distance * radius_padding_multiplier)
    else:
        RADIUS = 0.7  # Default fallback
    
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
    ax.set_xlabel('X', fontsize=11, fontweight='bold')  # Horizontal
    ax.set_ylabel('Z', fontsize=11, fontweight='bold')  # Depth (matplotlib Y = our Z)
    ax.set_zlabel('Y', fontsize=11, fontweight='bold')  # Vertical (matplotlib Z = our -Y)
    
    # Set axis limits centered on hip with RADIUS = 0.7
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    
    # Set equal aspect ratio for better visualization
    # Use equal aspect for all axes to make skeleton appear larger and more proportional
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for all axes
    
    # Title (no height display)
    title = f'Frame {frame_id}' if frame_id is not None else '3D Pose'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Use provided view angles (fixed for all frames to ensure consistency)
    ax.view_init(elev=view_elev, azim=view_azim)


def render_3d_skeleton_to_image(keypoints_3d, joint_names, frame_id=None, 
                                view_azim=0.0, view_elev=0.0, width=640, height=480, dpi=100,
                                radius_padding_multiplier=1.2, radius_minimum=0.3):
    """
    Render 3D skeleton to a numpy image array.
    
    Args:
        keypoints_3d: [Nx3] array of 3D keypoints
        joint_names: List of joint names
        frame_id: Frame ID for title
        view_azim: Azimuth angle for view
        view_elev: Elevation angle for view
        width: Output image width
        height: Output image height
        dpi: DPI for rendering
        radius_padding_multiplier: Multiplier for padding around skeleton (default: 1.2)
        radius_minimum: Minimum RADIUS for axis limits (default: 0.3)
    
    Returns:
        image: [HxWx3] numpy array (BGR format for OpenCV)
    """
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi, facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    
    draw_skeleton_3d(ax, keypoints_3d, joint_names, frame_id=frame_id, 
                     view_azim=view_azim, view_elev=view_elev,
                     radius_padding_multiplier=radius_padding_multiplier,
                     radius_minimum=radius_minimum)
    
    # Render to numpy array
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Convert RGB to BGR for OpenCV
    image = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    
    plt.close(fig)
    return image


def draw_2d_keypoints_on_frame(frame, keypoints_2d, color=(0, 255, 0), thickness=2, radius=5, 
                               draw_skeleton=True, skeleton_color=None):
    """
    Draw 2D keypoints on a video frame.
    
    Args:
        frame: Input frame (BGR format)
        keypoints_2d: [Nx2] array of 2D keypoints
        color: Keypoint color (BGR format)
        thickness: Line thickness
        radius: Keypoint circle radius
        draw_skeleton: Whether to draw skeleton connections
        skeleton_color: Color for skeleton lines (if None, uses keypoint color)
    """
    frame_copy = frame.copy()
    
    if skeleton_color is None:
        skeleton_color = color
    
    for i, (x, y) in enumerate(keypoints_2d):
        if not np.isnan(x) and not np.isnan(y):
            x, y = int(x), int(y)
            cv2.circle(frame_copy, (x, y), radius, color, -1)
            cv2.circle(frame_copy, (x, y), radius + 2, (255, 255, 255), thickness)
    
    # Draw skeleton
    if draw_skeleton:
        for edge in SKELETON_EDGES:
            pt1_idx, pt2_idx = edge
            if (pt1_idx < len(keypoints_2d) and pt2_idx < len(keypoints_2d) and
                not np.isnan(keypoints_2d[pt1_idx]).any() and not np.isnan(keypoints_2d[pt2_idx]).any()):
                pt1 = (int(keypoints_2d[pt1_idx][0]), int(keypoints_2d[pt1_idx][1]))
                pt2 = (int(keypoints_2d[pt2_idx][0]), int(keypoints_2d[pt2_idx][1]))
                cv2.line(frame_copy, pt1, pt2, skeleton_color, thickness)
    
    return frame_copy


def load_original_2d_keypoints(video_path, cam_name, frame_id, dataset):
    """
    Load original 2D keypoints from JSON files in video directory.
    
    Args:
        video_path: Path to video directory
        cam_name: Camera name ('front' or 'side')
        frame_id: Frame ID
        dataset: Dataset instance (for keypoint loading)
    
    Returns:
        keypoints_2d: [Nx2] array of 2D keypoints, or None if not found
    """
    # Try to load from dataset's keypoint files
    try:
        joints_dict = dataset.getSingleFrameMultiView2DJoints(frame_id)
        if cam_name in joints_dict and len(joints_dict[cam_name]) > 0:
            person_key = list(joints_dict[cam_name].keys())[0]
            keypoints_2d = joints_dict[cam_name][person_key]
            # Ensure it's a numpy array with shape (N, 2)
            if isinstance(keypoints_2d, list):
                keypoints_2d = np.array(keypoints_2d)
            if len(keypoints_2d.shape) == 1:
                # Reshape if needed
                keypoints_2d = keypoints_2d.reshape(-1, 2)
            return keypoints_2d
    except Exception as e:
        # Silently fail - keypoints may not be available
        pass
    
    return None


def calculate_front_view_angle(keypoints_3d, valid_mask):
    """
    Calculate the optimal front view angle (azim) based on person's orientation.
    Returns azim angle in degrees, or None if cannot be calculated.
    
    Note: The person faces perpendicular to the shoulder line. If shoulders are horizontal
    (left-right), the person faces forward/backward. To show the front view, we need to
    rotate 90 degrees from the side view.
    """
    # Apply coordinate transformation (same as in draw_skeleton_3d)
    transformed = keypoints_3d.copy()
    transformed[:, 0] = keypoints_3d[:, 0]  # X stays
    transformed[:, 1] = keypoints_3d[:, 2]   # Z → Y (depth)
    transformed[:, 2] = -keypoints_3d[:, 1]  # -Y → Z (vertical, upward)
    
    # Use shoulders to determine orientation
    if valid_mask[12] and valid_mask[13]:  # Both shoulders visible
        right_shoulder = transformed[12]
        left_shoulder = transformed[13]
        # Shoulder vector in XY plane (horizontal plane in matplotlib)
        shoulder_vec = right_shoulder[:2] - left_shoulder[:2]  # [X, Y] components
        # Calculate angle of shoulder line
        angle = np.degrees(np.arctan2(shoulder_vec[1], shoulder_vec[0]))
        # Person faces perpendicular to shoulders (90 degrees from shoulder direction)
        # If current calculation gives side view, we need to subtract 90° to get front view
        # The shoulder line direction gives us the side view angle, so front view is -90°
        azim = -angle - 90
        # Normalize to [-180, 180]
        if azim > 180:
            azim -= 360
        elif azim < -180:
            azim += 360
        return azim
    else:
        # Fallback: use head-to-hip vector
        if valid_mask[9] and valid_mask[6]:  # Head and hip visible
            head = transformed[9]
            hip = transformed[6]
            body_vec = (head - hip)[:2]  # Only X and Y components
            if np.linalg.norm(body_vec) > 0.01:
                body_angle = np.degrees(np.arctan2(body_vec[1], body_vec[0]))
                # Subtract 90° to get front view from side view
                azim = -body_angle - 90
                if azim > 180:
                    azim -= 360
                elif azim < -180:
                    azim += 360
                return azim
    
    # Cannot determine orientation
    return None


def create_3d_visualization(video_path, output_file=None, fps=30, config=None, logger=None):
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
    
    # Get visualization config (with defaults)
    viz_config = config.get('visualization', {}) if config else {}
    auto_calculate = viz_config.get('auto_calculate_view_angle', True)
    fixed_azim = viz_config.get('fixed_view_azim', 0.0)
    fixed_elev = viz_config.get('fixed_view_elev', 0.0)
    sample_frames_count = viz_config.get('view_angle_sample_frames', 5)
    radius_padding_multiplier = viz_config.get('radius_padding_multiplier', 1.2)
    radius_minimum = viz_config.get('radius_minimum', 0.3)
    
    # Calculate or use fixed front view angle
    if auto_calculate:
        # Calculate optimal front view angle from first valid frame
        # This ensures consistent view angle across all frames
        front_view_azim = 0.0
        front_view_elev = 0.0
        
        # Try to find optimal view angle from first few frames
        sample_frames = min(sample_frames_count, len(frame_data_list))
        azim_angles = []
        
        for frame_idx in range(sample_frames):
            data = frame_data_list[frame_idx]
            person_data = data['keypoints_by_person'].get('person_0', {})
            keypoints_3d = np.array(person_data.get('joints', []))
            keypoints_valid = person_data.get('keypoints_valid', True)
            
            if keypoints_valid and len(keypoints_3d) > 0:
                valid_mask = ~np.isnan(keypoints_3d).any(axis=1)
                azim = calculate_front_view_angle(keypoints_3d, valid_mask)
                if azim is not None:
                    azim_angles.append(azim)
        
        if len(azim_angles) > 0:
            # Use average of first few frames to determine consistent view angle
            front_view_azim = np.mean(azim_angles)
            if logger:
                logger.info(f"Calculated front view angle: azim={front_view_azim:.1f}° (from {len(azim_angles)} sample frames)")
        else:
            if logger:
                logger.warning("Could not calculate front view angle, using default azim=0°")
    else:
        # Use fixed values from config
        front_view_azim = fixed_azim
        front_view_elev = fixed_elev
        if logger:
            logger.info(f"Using FIXED view angle from config: elev={front_view_elev}°, azim={front_view_azim:.1f}°")
    
    last_valid_keypoints = None
    frames_written = 0
    
    # Progress bar setup
    total_frames = len(frame_data_list)
    print(f"\nCreating 3D visualization: {total_frames} frames")
    if logger:
        logger.info(f"Using fixed view angle: elev={front_view_elev}°, azim={front_view_azim:.1f}°")
    
    for frame_idx, data in enumerate(frame_data_list):
        # Progress bar
        progress = (frame_idx + 1) / total_frames
        bar_length = 50
        filled = int(bar_length * progress)
        bar = '=' * filled + '-' * (bar_length - filled)
        print(f'\r[{bar}] {frame_idx+1}/{total_frames} ({progress*100:.1f}%)', end='', flush=True)
        
        if (frame_idx + 1) % 50 == 0 and logger:
            logger.info(f"Processing frame {frame_idx+1}/{len(frame_data_list)}...")
        
        frame_id = data['frame_id']
        person_data = data['keypoints_by_person'].get('person_0', {})
        keypoints_3d = np.array(person_data.get('joints', []))
        keypoints_valid = person_data.get('keypoints_valid', True)
        
        if keypoints_valid and len(keypoints_3d) > 0:
            draw_skeleton_3d(ax, keypoints_3d, JOINT_NAMES, frame_id=frame_id, 
                          view_azim=front_view_azim, view_elev=front_view_elev,
                          radius_padding_multiplier=radius_padding_multiplier,
                          radius_minimum=radius_minimum)
            last_valid_keypoints = keypoints_3d
        elif last_valid_keypoints is not None:
            draw_skeleton_3d(ax, last_valid_keypoints, JOINT_NAMES, frame_id=frame_id,
                          view_azim=front_view_azim, view_elev=front_view_elev,
                          radius_padding_multiplier=radius_padding_multiplier,
                          radius_minimum=radius_minimum)
        else:
            ax.clear()
            ax.text2D(0.5, 0.5, "Invalid Frame", transform=ax.transAxes, 
                     color='red', fontsize=20, ha='center')
            ax.set_title(f'Frame {frame_id} | INVALID', fontsize=14, fontweight='bold', pad=20, color='red')
        
        # Render to numpy array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Resize to target dimensions
        buf = cv2.resize(buf, (int(width), int(height)))
        
        # Convert RGB to BGR for OpenCV
        buf = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        
        out.write(buf)
        frames_written += 1
    
    plt.close(fig)
    out.release()
    
    print()  # New line after progress bar
    
    if logger:
        logger.info(f"Wrote {frames_written} frames to video")
        logger.info(f"3D visualization video saved: {output_file}")
    
    return output_file


# ============================================================================
# Debug Visualization Functions (Refactored for Multi-threading)
# ============================================================================

def _load_debug_data(video_path, results_dir, logger=None):
    """Load frame data and initialize video captures."""
    # Load frame data
    combined_file = os.path.join(results_dir, 'all_frames_3d_keypoints.json')
    if os.path.exists(combined_file):
        with open(combined_file, 'r') as f:
            combined_data = json.load(f)
        frame_data_list = combined_data.get('frames', [])
        if logger:
            logger.info(f"Loading {len(frame_data_list)} frames from combined JSON")
    else:
        frame_files = sorted(glob.glob(os.path.join(results_dir, 'frame_*_3d_keypoints.json')))
        if len(frame_files) == 0:
            if logger:
                logger.error("No 3D keypoint files found!")
            return None, None
        frame_data_list = []
        for frame_file in frame_files:
            with open(frame_file, 'r') as f:
                frame_data_list.append(json.load(f))
    
    if len(frame_data_list) == 0:
        if logger:
            logger.error("No frame data found!")
        return None, None
    
    # Find and open video files
    video_dir = video_path
    video_caps = {}
    
    for cam_name in ['front', 'side']:
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
        
        # Pattern 2: Search for files containing camera name in filename
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
            video_caps[cam_name] = cv2.VideoCapture(video_file)
            if logger:
                logger.info(f"Found {cam_name} video: {os.path.basename(video_file)}")
        else:
            if logger:
                logger.warning(f"{cam_name} video not found, skipping debug visualization")
            return None, None
    
    return frame_data_list, video_caps


def _process_single_debug_frame(frame_data, frame_idx, video_path, video_files, dataset, 
                                saved_camera_pose, front_view_azim, side_view_azim,
                                video_width, video_height, view_3d_width, view_3d_height,
                                radius_padding_multiplier=1.2, radius_minimum=0.3):
    """
    Process a single frame to create all debug visualization panels.
    
    Note: Each thread creates its own VideoCapture instances for thread safety.
    
    Returns:
        combined_frame: Combined 3x2 grid frame, or None if error
        frame_id: Frame ID for ordering
    """
    try:
        # Create thread-local video captures (OpenCV VideoCapture is not thread-safe)
        video_caps = {}
        for cam_name in ['front', 'side']:
            if cam_name in video_files:
                video_caps[cam_name] = cv2.VideoCapture(video_files[cam_name])
        
        frame_id = frame_data['frame_id']
        person_data = frame_data['keypoints_by_person'].get('person_0', {})
        keypoints_3d = np.array(person_data.get('joints', []))
        
        panels = {}
        
        # 1. 3D Front View
        if len(keypoints_3d) > 0:
            panel_3d_front = render_3d_skeleton_to_image(
                keypoints_3d, JOINT_NAMES, frame_id=frame_id,
                view_azim=front_view_azim, view_elev=0.0,
                width=view_3d_width, height=view_3d_height,
                radius_padding_multiplier=radius_padding_multiplier,
                radius_minimum=radius_minimum
            )
        else:
            panel_3d_front = np.zeros((view_3d_height, view_3d_width, 3), dtype=np.uint8)
            cv2.putText(panel_3d_front, "No 3D Data", (10, view_3d_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(panel_3d_front, "3D Front View (azim=0)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        panels['3d_front'] = panel_3d_front
        
        # 2. 3D Side View
        if len(keypoints_3d) > 0:
            panel_3d_side = render_3d_skeleton_to_image(
                keypoints_3d, JOINT_NAMES, frame_id=frame_id,
                view_azim=side_view_azim, view_elev=0.0,
                width=view_3d_width, height=view_3d_height,
                radius_padding_multiplier=radius_padding_multiplier,
                radius_minimum=radius_minimum
            )
        else:
            panel_3d_side = np.zeros((view_3d_height, view_3d_width, 3), dtype=np.uint8)
            cv2.putText(panel_3d_side, "No 3D Data", (10, view_3d_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(panel_3d_side, f"3D Side View (azim={side_view_azim:.0f})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        panels['3d_side'] = panel_3d_side
        
        # Process 2D views for front and side cameras
        for cam_name in ['front', 'side']:
            if cam_name not in video_caps:
                frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
            else:
                video_cap = video_caps[cam_name]
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = video_cap.read()
                if not ret:
                    frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
            
            # Load original 2D keypoints
            original_kpts = load_original_2d_keypoints(video_path, cam_name, frame_id, dataset)
            
            # Get reprojected 2D keypoints
            reprojected_kpts = None
            if len(keypoints_3d) > 0:
                cam_params = dataset.cam_params_dict[cam_name]
                K = cam_params['K']
                D = cam_params['distCoef']
                
                # Get camera pose
                wrld_cam_id = 1 if frame_data.get('world_camera', 'side') == 'side' else 0
                wrld_cam_name = frame_data.get('world_camera', 'side')
                
                if cam_name == wrld_cam_name:
                    R = np.eye(3)
                    t = np.zeros(3)
                else:
                    if saved_camera_pose is not None and len(saved_camera_pose) > 0:
                        M = saved_camera_pose[0]
                        R = M[:, :3]
                        t = M[:, 3]
                    elif 'camera_poses_dict' in frame_data and cam_name in frame_data['camera_poses_dict']:
                        M = np.array(frame_data['camera_poses_dict'][cam_name]['pose_matrix'])
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
            
            # Create original and reprojected frames
            frame_original = frame.copy()
            frame_reprojected = frame.copy()
            
            # Draw original keypoints (green)
            if original_kpts is not None:
                frame_original = draw_2d_keypoints_on_frame(
                    frame_original, original_kpts, 
                    color=COLORS_2D['original'], thickness=2, radius=4
                )
            
            # Draw reprojected keypoints (red) - make them clearly visible
            # First, draw original keypoints as small dots for reference
            if original_kpts is not None:
                for i, (x, y) in enumerate(original_kpts):
                    if not np.isnan(x) and not np.isnan(y):
                        x, y = int(x), int(y)
                        cv2.circle(frame_reprojected, (x, y), 2, COLORS_2D['original'], -1)
            
            # Now draw reprojected keypoints (red) - larger and more visible
            if reprojected_kpts is not None:
                frame_reprojected = draw_2d_keypoints_on_frame(
                    frame_reprojected, reprojected_kpts,
                    color=COLORS_2D['reprojected'], thickness=3, radius=6,
                    skeleton_color=COLORS_2D['reprojected']
                )
                
                # Draw error lines (blue) connecting original to reprojected
                if original_kpts is not None:
                    for i in range(min(len(original_kpts), len(reprojected_kpts))):
                        if (not np.isnan(original_kpts[i]).any() and 
                            not np.isnan(reprojected_kpts[i]).any()):
                            pt1 = (int(original_kpts[i][0]), int(original_kpts[i][1]))
                            pt2 = (int(reprojected_kpts[i][0]), int(reprojected_kpts[i][1]))
                            cv2.line(frame_reprojected, pt1, pt2, COLORS_2D['error'], 2)
            
            # Add text labels
            rmse_data = person_data.get('reprojection_rmse', {})
            rmse = rmse_data.get('per_camera_rmse', {}).get(cam_name, None)
            rmse_text = f"RMSE: {rmse:.2f}px" if rmse is not None else "RMSE: N/A"
            
            cv2.putText(frame_original, f"{cam_name.upper()} - ORIGINAL | Frame {frame_id}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_original, "Green: Input 2D Keypoints", 
                       (10, video_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(frame_reprojected, f"{cam_name.upper()} - REPROJECTED | Frame {frame_id}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_reprojected, rmse_text, 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_reprojected, "Red: 3D->2D Reprojected Keypoints", 
                       (10, video_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame_reprojected, "Blue: Error Lines | Green dots: Original", 
                       (10, video_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            panels[f'{cam_name}_original'] = frame_original
            panels[f'{cam_name}_reprojected'] = frame_reprojected
        
        # Create 3x2 grid layout
        top_row = np.hstack([
            panels['3d_front'],
            panels['front_original'],
            panels['front_reprojected']
        ])
        
        bottom_row = np.hstack([
            panels['3d_side'],
            panels['side_original'],
            panels['side_reprojected']
        ])
        
        combined = np.vstack([top_row, bottom_row])
        
        # Release thread-local video captures
        for cam_name in video_caps:
            if video_caps[cam_name] is not None:
                video_caps[cam_name].release()
        
        return combined, frame_id
        
    except Exception as e:
        print(f"Error processing frame {frame_data.get('frame_id', frame_idx)}: {e}")
        # Release video captures even on error
        try:
            if 'video_caps' in locals():
                for cam_name in video_caps:
                    if video_caps[cam_name] is not None:
                        video_caps[cam_name].release()
        except:
            pass
        return None, frame_data.get('frame_id', frame_idx)


def create_debug_visualization(video_path, output_file=None, fps=30, config=None, logger=None, num_threads=8):
    """
    Create combined debug visualization with 3D views and 2D keypoint comparison.
    
    Layout: 3 columns x 2 rows
    Top row: 3D front view (azim=0), 2D front original, 2D front reprojected
    Bottom row: 3D side view (azim=90), 2D side original, 2D side reprojected
    
    Args:
        video_path: Path to video directory
        output_file: Output video file path (optional)
        fps: Video FPS
        config: Configuration dictionary
        logger: Logger instance
        num_threads: Number of threads for parallel processing (default: 8)
    """
    video_path = os.path.abspath(video_path)
    video_name = os.path.basename(video_path)
    data_dir = os.path.dirname(video_path)
    results_dir = os.path.join(data_dir, 'processed', video_name, 'results')
    
    if logger:
        logger.info("Creating DEBUG visualization (3D + 2D keypoints comparison)")
    
    # Load data
    frame_data_list, video_caps = _load_debug_data(video_path, results_dir, logger)
    if frame_data_list is None or video_caps is None:
        return None
    
    # Extract video file paths (for thread-local VideoCapture creation)
    video_files = {}
    for cam_name in video_caps:
        # Get the video file path from the VideoCapture (we need to store it)
        # Since we can't get the path from VideoCapture, we'll need to find it again
        video_dir = video_path
        video_file = None
        
        direct_paths = [
            os.path.join(video_dir, f'{cam_name}.mp4'),
            os.path.join(video_dir, f'{cam_name}.avi'),
        ]
        for path in direct_paths:
            if os.path.exists(path):
                video_file = path
                break
        
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
        
        if video_file:
            video_files[cam_name] = video_file
    
    # Get video properties FIRST (before releasing video_caps)
    ret, frame = video_caps['front'].read()
    if not ret:
        if logger:
            logger.error("Could not read video frames")
        # Release video captures before returning
        for cam_name in video_caps:
            video_caps[cam_name].release()
        return None
    
    video_height, video_width = frame.shape[:2]
    video_caps['front'].set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Release the original video captures (threads will create their own)
    for cam_name in video_caps:
        video_caps[cam_name].release()
    
    # Load dataset for camera parameters and keypoint loading
    from src.pipeline_utils import create_default_config
    from src.dataset.twoview_custom import TwoViewCustom
    
    config_dict = config if config else create_default_config()
    dataset = TwoViewCustom(None, None, config_dict, video_path=video_path)
    dataset.prepareDataFromKeypoints()
    
    # Load saved camera pose
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
    
    view_3d_width = video_width
    view_3d_height = video_height
    
    # Output filename
    if output_file is None:
        output_file = os.path.join(results_dir, f'{video_name}_debug_2d_keypoints_comparison.mp4')
    
    # Video writer
    output_width = video_width * 3
    output_height = video_height * 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (output_width, output_height))
    
    if not out.isOpened():
        if logger:
            logger.error("Could not initialize video writer")
        return None
    
    total_frames = len(frame_data_list)
    if logger:
        logger.info(f"Processing {total_frames} frames with {num_threads} threads...")
    
    # Get view angles
    viz_config = config_dict.get('visualization', {}) if config_dict else {}
    auto_calculate = viz_config.get('auto_calculate_view_angle', True)
    fixed_azim = viz_config.get('fixed_view_azim', 0.0)
    radius_padding_multiplier = viz_config.get('radius_padding_multiplier', 1.2)
    radius_minimum = viz_config.get('radius_minimum', 0.3)
    
    front_view_azim = 0.0
    if auto_calculate and len(frame_data_list) > 0:
        first_data = frame_data_list[0]
        person_data = first_data['keypoints_by_person'].get('person_0', {})
        keypoints_3d = np.array(person_data.get('joints', []))
        if len(keypoints_3d) > 0:
            valid_mask = ~np.isnan(keypoints_3d).any(axis=1)
            calculated_azim = calculate_front_view_angle(keypoints_3d, valid_mask)
            if calculated_azim is not None:
                front_view_azim = calculated_azim
    else:
        front_view_azim = fixed_azim
    
    side_view_azim = front_view_azim - 90
    if side_view_azim > 180:
        side_view_azim -= 360
    elif side_view_azim < -180:
        side_view_azim += 360
    
    # Process frames in parallel
    print(f"\nCreating debug visualization: {total_frames} frames (using {num_threads} threads)")
    
    # Verify multi-threading is actually being used
    start_time = time.time()
    if logger:
        logger.info(f"Starting multi-threaded processing with {num_threads} workers")
    
    # Create a list to store results with frame indices
    results = {}
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        futures = {}
        for frame_idx, frame_data in enumerate(frame_data_list):
            future = executor.submit(
                _process_single_debug_frame,
                frame_data, frame_idx, video_path, video_files, dataset,
                saved_camera_pose, front_view_azim, side_view_azim,
                video_width, video_height, view_3d_width, view_3d_height,
                radius_padding_multiplier, radius_minimum
            )
            futures[future] = frame_idx
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(futures):
            frame_idx = futures[future]
            try:
                combined_frame, frame_id = future.result()
                if combined_frame is not None:
                    results[frame_id] = combined_frame
                completed += 1
                
                # Progress bar
                progress = completed / total_frames
                bar_length = 50
                filled = int(bar_length * progress)
                bar = '=' * filled + '-' * (bar_length - filled)
                print(f'\r[{bar}] {completed}/{total_frames} ({progress*100:.1f}%)', end='', flush=True)
                
                if completed % 50 == 0 and logger:
                    logger.info(f"Processed {completed}/{total_frames} frames...")
            except Exception as e:
                if logger:
                    logger.error(f"Error processing frame {frame_idx}: {e}")
    
    print()  # New line after progress bar
    
    # Log timing to verify multi-threading performance
    elapsed_time = time.time() - start_time
    if logger:
        logger.info(f"Multi-threaded processing completed in {elapsed_time:.2f} seconds")
        logger.info(f"Average time per frame: {elapsed_time/total_frames:.3f} seconds")
        logger.info(f"Processing rate: {total_frames/elapsed_time:.1f} frames/second")
    
    # Write frames in order
    if logger:
        logger.info(f"Writing {len(results)} frames to video...")
    
    for frame_id in sorted(results.keys()):
        out.write(results[frame_id])
    
    # Release resources
    out.release()
    
    if logger:
        logger.info(f"Debug visualization saved: {output_file}")
    return output_file
