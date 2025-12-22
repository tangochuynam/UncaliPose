#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create beautiful 3D keypoint visualization as MP4 video.
"""

import sys
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import glob
import cv2

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


# Uplift Order skeleton connections
SKELETON_EDGES = [
    (0, 1), (1, 2), (5, 4), (4, 3),  # Legs
    (2, 6), (3, 6),  # Hips to center
    (6, 7), (7, 8), (8, 9),  # Spine
    (12, 11), (11, 10), (7, 12),  # Right arm
    (13, 14), (14, 15), (7, 13)  # Left arm
]

JOINT_NAMES = [
    "right_ankle", "right_knee", "right_hip", "left_hip",
    "left_knee", "left_ankle", "center_hip", "center_shoulder",
    "neck", "head", "right_wrist", "right_elbow",
    "right_shoulder", "left_shoulder", "left_elbow", "left_wrist"
]


def load_all_frames(data_dir, video_name):
    """Load all 3D keypoint frames."""
    results_dir = os.path.join(data_dir, 'processed', video_name, 'results')
    frame_files = sorted(glob.glob(os.path.join(results_dir, 'frame_*_3d_keypoints.json')))
    
    all_frames = []
    for frame_file in frame_files:
        with open(frame_file, 'r') as f:
            data = json.load(f)
            all_frames.append(data)
    
    return all_frames


def draw_beautiful_skeleton_3d(ax, keypoints_3d, joint_names, frame_id=None, height=None):
    """Draw beautiful 3D skeleton with enhanced styling."""
    ax.clear()
    
    # Filter out NaN keypoints
    valid_mask = ~np.isnan(keypoints_3d).any(axis=1)
    valid_keypoints = keypoints_3d[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_keypoints) == 0:
        return
    
    # Transform coordinates to make person stand vertically
    # Original: X=left-right, Y=vertical (downward positive, head at Y=-0.15, ankles at Y=0.23), Z=depth
    # Transform: (X, Y, Z) -> (X, -Z, -Y) to make Z vertical upward
    # This makes: X=left-right, Y=depth (into screen, inverted), Z=vertical (upward)
    transformed_keypoints = keypoints_3d.copy()
    transformed_keypoints[:, 1] = -keypoints_3d[:, 2]  # Y becomes -Z (depth, inverted)
    transformed_keypoints[:, 2] = -keypoints_3d[:, 1]  # Z becomes -Y (vertical, inverted so head is up)
    # Now: X=left-right, Y=depth (into screen), Z=vertical (upward, head higher than feet)
    
    # Update valid points
    valid_transformed = transformed_keypoints[valid_mask]
    
    # Joint sizes and colors
    joint_sizes = [50, 60, 70, 70, 60, 50, 80, 90, 100, 120, 50, 60, 70, 70, 60, 50]
    joint_colors = ['blue', 'blue', 'blue', 'blue', 'blue', 'blue', 
                    'green', 'green', 'green', 'yellow', 
                    'red', 'red', 'red', 'red', 'red', 'red']
    
    # Draw skeleton edges with transformed coordinates
    for edge in SKELETON_EDGES:
        i, j = edge
        if i < len(transformed_keypoints) and j < len(transformed_keypoints):
            if valid_mask[i] and valid_mask[j]:
                x_line = [transformed_keypoints[i, 0], transformed_keypoints[j, 0]]
                y_line = [transformed_keypoints[i, 1], transformed_keypoints[j, 1]]
                z_line = [transformed_keypoints[i, 2], transformed_keypoints[j, 2]]
                
                # Color coding: legs=blue, torso=green, arms=red
                if i in [0, 1, 2, 3, 4, 5, 6]:  # Legs and hips
                    color = 'blue'
                    linewidth = 2.5
                elif i in [7, 8, 9]:  # Torso/head
                    color = 'green'
                    linewidth = 3.0
                else:  # Arms
                    color = 'red'
                    linewidth = 2.5
                
                ax.plot(x_line, y_line, z_line, color=color, linewidth=linewidth, alpha=0.8)
    
    # Draw joints with transformed coordinates
    for idx, (x, y, z) in enumerate(valid_transformed):
        orig_idx = valid_indices[idx]
        if orig_idx < len(joint_sizes):
            ax.scatter([x], [y], [z], 
                      c=joint_colors[orig_idx], 
                      s=joint_sizes[orig_idx], 
                      alpha=0.9,
                      edgecolors='black',
                      linewidths=1.5)
    
    # Set labels and title with better styling
    ax.set_xlabel('X (m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Depth (m)', fontsize=11, fontweight='bold')
    ax.set_zlabel('Height (m)', fontsize=11, fontweight='bold')
    
    title = f'Frame {frame_id}' if frame_id is not None else '3D Human Pose'
    if height is not None:
        title += f' | Height: {height*100:.1f} cm'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Set equal aspect ratio with padding
    if len(valid_transformed) > 0:
        center = np.mean(valid_transformed, axis=0)
        max_range = np.max(np.abs(valid_transformed - center)) * 1.3
        
        ax.set_xlim([center[0] - max_range, center[0] + max_range])
        ax.set_ylim([center[1] - max_range, center[1] + max_range])
        ax.set_zlim([center[2] - max_range, center[2] + max_range])
    
    # Set viewing angle to show person standing vertically
    # elev: elevation angle (0=horizontal, 90=top view)
    # azim: azimuth angle (rotation around Z axis)
    # For a person standing, we want: elev around 10-15, azim around 30-45
    ax.view_init(elev=15, azim=40)
    
    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False


def create_mp4_video_opencv(data_dir, video_name, output_file=None, fps=30, width=None, height=None):
    """Create MP4 video using OpenCV (more reliable than matplotlib animation)."""
    print("Loading 3D keypoint data...")
    all_frames = load_all_frames(data_dir, video_name)
    
    if len(all_frames) == 0:
        print("✗ No frames found!")
        return None
    
    print(f"✓ Loaded {len(all_frames)} frames")
    
    # Get structure from first frame
    first_frame = all_frames[0]
    n_joints = first_frame['n_joints']
    n_persons = first_frame['n_persons']
    person_id = 0
    
    # Determine video dimensions from first frame or use defaults
    if width is None or height is None:
        # Try to get from camera parameters if available
        try:
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
            from src.dataset.twoview_custom import TwoViewCustom
            dataset = TwoViewCustom(data_dir, video_name, config=None)
            if dataset.cam_params_dict:
                # Use resolution from first camera
                first_cam = list(dataset.cam_params_dict.keys())[0]
                width, height = dataset.cam_params_dict[first_cam]['resolution']
                print(f"Using resolution from video: {width}x{height}")
        except:
            pass
        
        # Fallback to defaults
        if width is None:
            width = 1920
        if height is None:
            height = 1080
    
    # Setup figure - use actual pixel dimensions
    dpi = 100
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi, facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    
    print(f"Creating {len(all_frames)} frames for video at {width}x{height}...")
    
    # Create video writer
    if output_file is None:
        output_dir = os.path.join(data_dir, 'processed', video_name, 'results')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, '3d_keypoints_animation.mp4')
    
    # Use matplotlib to render frames, then OpenCV to create video
    frames_list = []
    
    for frame_idx, frame_data in enumerate(all_frames):
        frame_id = frame_data['frame_id']
        
        # Check if keypoints are valid
        keypoints_valid = frame_data.get('keypoints_valid', True)
        if f'person_{person_id}' in frame_data['keypoints_by_person']:
            person_data = frame_data['keypoints_by_person'][f'person_{person_id}']
            keypoints_valid = person_data.get('keypoints_valid', keypoints_valid)
        
        # Skip invalid frames or use previous valid frame
        if not keypoints_valid:
            # Use previous valid frame's keypoints (or skip if first frame is invalid)
            if len(frames_list) > 0:
                # Reuse last valid frame
                frames_list.append(frames_list[-1])
                if (frame_idx + 1) % 50 == 0:
                    print(f"  Skipped invalid frame {frame_id} (using previous frame)")
                continue
            else:
                # First frame is invalid, skip it
                if (frame_idx + 1) % 50 == 0:
                    print(f"  Skipped invalid frame {frame_id}")
                continue
        
        # Get keypoints for first person
        if f'person_{person_id}' in frame_data['keypoints_by_person']:
            person_data = frame_data['keypoints_by_person'][f'person_{person_id}']
            keypoints_3d = np.array(person_data['joints'])
            height = person_data.get('height_m', None)
        else:
            keypoints_3d = np.array(frame_data['keypoints_3d'])
            if n_persons > 1:
                start_idx = person_id * n_joints
                end_idx = (person_id + 1) * n_joints
                keypoints_3d = keypoints_3d[start_idx:end_idx]
            height = frame_data.get('heights', {}).get(f'person_{person_id}', {}).get('height_m', None)
        
        draw_beautiful_skeleton_3d(ax, keypoints_3d, JOINT_NAMES, frame_id=frame_id, height=height)
        
        # Render frame to numpy array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        canvas_width, canvas_height = fig.canvas.get_width_height()
        buf = buf.reshape((canvas_height, canvas_width, 3))
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        frames_list.append(frame_bgr)
        
        if (frame_idx + 1) % 10 == 0:
            print(f"  Processed {frame_idx + 1}/{len(all_frames)} frames...")
    
    plt.close(fig)
    
    if len(frames_list) == 0:
        print("✗ No frames to write!")
        return None
    
    # Get actual frame dimensions
    actual_height, actual_width = frames_list[0].shape[:2]
    print(f"Frame dimensions: {actual_width}x{actual_height}")
    
    # Create video using OpenCV
    print(f"Saving MP4 video to: {output_file}")
    # Try different codecs
    codecs_to_try = [
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
        ('avc1', cv2.VideoWriter_fourcc(*'avc1')),
    ]
    
    out = None
    for codec_name, fourcc in codecs_to_try:
        try:
            out = cv2.VideoWriter(output_file, fourcc, float(fps), (int(actual_width), int(actual_height)))
            if out.isOpened():
                print(f"Using codec: {codec_name}")
                break
            else:
                out.release()
                out = None
        except:
            continue
    
    if out is None or not out.isOpened():
        print("✗ Failed to create video writer. Trying matplotlib method...")
        return create_mp4_video_matplotlib(data_dir, video_name, output_file, fps)
    
    # Write frames
    for i, frame in enumerate(frames_list):
        # Resize if needed
        if frame.shape[:2] != (actual_height, actual_width):
            frame = cv2.resize(frame, (int(actual_width), int(actual_height)))
        out.write(frame)
    
    out.release()
    
    # Verify file was created
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"✓ Beautiful 3D animation MP4 created: {output_file}")
        print(f"  Video: {len(all_frames)} frames, {fps} fps, {actual_width}x{actual_height}")
        print(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
        return output_file
    else:
        print("✗ Video file is empty or not created. Trying matplotlib method...")
        return create_mp4_video_matplotlib(data_dir, video_name, output_file, fps)


def create_mp4_video_matplotlib(data_dir, video_name, output_file=None, fps=30):
    """Create MP4 video using matplotlib animation (alternative method)."""
    print("Loading 3D keypoint data...")
    all_frames = load_all_frames(data_dir, video_name)
    
    if len(all_frames) == 0:
        print("✗ No frames found!")
        return None
    
    print(f"✓ Loaded {len(all_frames)} frames")
    
    # Get structure from first frame
    first_frame = all_frames[0]
    n_joints = first_frame['n_joints']
    n_persons = first_frame['n_persons']
    person_id = 0
    
    # Setup figure
    fig = plt.figure(figsize=(16, 12), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    
    # Animation function
    def animate(frame_idx):
        frame_data = all_frames[frame_idx]
        frame_id = frame_data['frame_id']
        
        # Get keypoints for first person
        if f'person_{person_id}' in frame_data['keypoints_by_person']:
            person_data = frame_data['keypoints_by_person'][f'person_{person_id}']
            keypoints_3d = np.array(person_data['joints'])
            height = person_data.get('height_m', None)
        else:
            keypoints_3d = np.array(frame_data['keypoints_3d'])
            if n_persons > 1:
                start_idx = person_id * n_joints
                end_idx = (person_id + 1) * n_joints
                keypoints_3d = keypoints_3d[start_idx:end_idx]
            height = frame_data.get('heights', {}).get(f'person_{person_id}', {}).get('height_m', None)
        
        draw_beautiful_skeleton_3d(ax, keypoints_3d, JOINT_NAMES, frame_id=frame_id, height=height)
        return ax
    
    # Create animation
    print("Creating animation...")
    anim = animation.FuncAnimation(
        fig, animate, frames=len(all_frames),
        interval=1000/fps, blit=False, repeat=True
    )
    
    # Save video
    if output_file is None:
        output_dir = os.path.join(data_dir, 'processed', video_name, 'results')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, '3d_keypoints_animation.mp4')
    
    print(f"Saving MP4 video to: {output_file}")
    try:
        # Try using ffmpeg writer
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='UncaliPose'), bitrate=1800)
        anim.save(output_file, writer=writer)
        print(f"✓ Beautiful 3D animation MP4 created: {output_file}")
        plt.close(fig)
        return output_file
    except Exception as e:
        print(f"✗ Error with matplotlib animation: {e}")
        print("Trying OpenCV method instead...")
        plt.close(fig)
        return create_mp4_video_opencv(data_dir, video_name, output_file, fps)


def main():
    parser = argparse.ArgumentParser(
        description='Create beautiful 3D keypoint MP4 video'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Root directory containing processed results'
    )
    parser.add_argument(
        '--video_name',
        type=str,
        default='swing1',
        help='Name of the video folder'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Output MP4 file path (optional)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second for video (default: 30)'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=1920,
        help='Video width in pixels (default: 1920)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=1080,
        help='Video height in pixels (default: 1080)'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='opencv',
        choices=['opencv', 'matplotlib'],
        help='Video creation method (default: opencv)'
    )
    
    args = parser.parse_args()
    
    if args.method == 'opencv':
        output_file = create_mp4_video_opencv(
            args.data_dir,
            args.video_name,
            output_file=args.output_file,
            fps=args.fps,
            width=args.width,
            height=args.height
        )
    else:
        output_file = create_mp4_video_matplotlib(
            args.data_dir,
            args.video_name,
            output_file=args.output_file,
            fps=args.fps
        )
    
    if output_file:
        print(f"\n{'='*70}")
        print(f"✓ SUCCESS! Beautiful 3D animation MP4 created:")
        print(f"  {output_file}")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

