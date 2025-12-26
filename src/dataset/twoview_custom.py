#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   Custom dataset class for 2-view (front and side camera) setup.
   Author: Adapted for UncaliPose
"""
import sys
sys.path.append('./..')
from .. import basic_3d_operations as b3dops
from .. import box_processing as bp
from collections import defaultdict
import pickle as pkl
import numpy as np
import glob
import json
import cv2
import os


class TwoViewCustom(object):
    """
    Custom dataset class for 2-view setup (front and side camera).
    
    Expected data structure:
    
    **Option 1: Direct video path mode (recommended)**
    A single folder containing:
    - **REQUIRED**: At least 2 keypoints JSON files (one for front, one for side)
      - One file with 'front' in filename (e.g., `front_keypoints.json`, `*front*.json`)
      - One file with 'side' in filename (e.g., `side_keypoints.json`, `*side*.json`)
    - **OPTIONAL**: 2 video files (if full dataset)
      - One file with 'front' in filename (e.g., `front_video.mp4`, `*front*.mp4`)
      - One file with 'side' in filename (e.g., `side_video.mp4`, `*side*.mp4`)
    
    Example:
    ```
    videos/swing1/
    ├── front_keypoints.json  (REQUIRED)
    ├── side_keypoints.json   (REQUIRED)
    ├── front_video.mp4       (OPTIONAL)
    └── side_video.mp4        (OPTIONAL)
    ```
    
    **Option 2: Traditional separated structure**
    ```
    data_dir/
    ├── videos/
    │   └── swing1/
    │       ├── front.mp4 (or front/)  (OPTIONAL)
    │       └── side.mp4 (or side/)   (OPTIONAL)
    └── keypoints/
        └── swing1/
            ├── front.json (or front/*.json)  (REQUIRED)
            └── side.json (or side/*.json)    (REQUIRED)
    ```
    
    **IMPORTANT File Naming Requirements:**
    - All files (JSON and video) MUST contain 'front' or 'side' in their filename (case-insensitive)
    - The pipeline will raise an error if it cannot identify which file is front/side
    - Examples of valid names: `front_keypoints.json`, `side_video.mp4`, `my_front_camera.mp4`
    - Examples of invalid names: `camera1.json`, `video1.mp4` (no front/side identifier)
    """
    
    def __init__(self, data_dir, video_name='swing1', config=None, video_path=None):
        """
        Initialize the two-view dataset.
        
        Args:
            data_dir: Root directory containing videos and keypoints (if video_path is None)
            video_name: Name of the video folder (default: 'swing1', ignored if video_path is provided)
            config: Configuration dictionary
            video_path: Direct path to video folder (overrides data_dir/video_name)
                       This folder should contain at least 2 keypoints JSON files (front and side)
                       and optionally 2 video files (front and side) if available.
        
        Raises:
            ValueError: If keypoints JSON files cannot be identified as front/side (missing 'front' or 'side' in filename)
            FileNotFoundError: If required keypoints files are not found
        """
        if video_path:
            # Direct path mode: video_path contains videos and keypoints JSON
            video_path = os.path.abspath(video_path)
            video_name = os.path.basename(video_path)
            data_dir = os.path.dirname(video_path)
            # In this mode, videos and keypoints are in the same directory
            self.video_dir = video_path
            self.keypoints_dir = video_path  # Same directory
            self.video_path = video_path  # Store for reference
        else:
            # Traditional mode: data_dir/videos/video_name and data_dir/keypoints/video_name
            if data_dir[-1] == '/': 
                data_dir = data_dir[:-1]
            self.video_dir = os.path.join(data_dir, 'videos', video_name)
            self.keypoints_dir = os.path.join(data_dir, 'keypoints', video_name)
            self.video_path = self.video_dir  # Use video_dir as video_path
        
        self.data_dir = data_dir
        self.video_name = video_name
        self.data_name = video_name
        
        # Camera names
        self.cam_names = ['front', 'side']
        self.num_cam = 2
        
        # Map camera names to video file names (if known)
        # This allows matching specific video files to cameras
        self.video_file_map = {
            'front': None,  # Will be auto-detected
            'side': None    # Will be auto-detected
        }
        self._detectVideoFiles()
        
        # Setup directories (for compatibility, but don't create them)
        self.video_frame_dir = os.path.join(data_dir, 'processed', video_name, 'video_frame')
        self.pose2d_file_dir = os.path.join(data_dir, 'processed', video_name, 'pose2d_label')
        self.boxcrop_dir = os.path.join(data_dir, 'processed', video_name, 'box_crop')
        self.calibration_file = os.path.join(data_dir, 'processed', video_name, 'calibration.json')
        
        # Joint configuration (Uplift Order - 16 joints)
        # Order: right_ankle(0), right_knee(1), right_hip(2), left_hip(3), 
        #        left_knee(4), left_ankle(5), center_hip(6), center_shoulder(7),
        #        neck(8), head(9), right_wrist(10), right_elbow(11), 
        #        right_shoulder(12), left_shoulder(13), left_elbow(14), left_wrist(15)
        self.joints_ids = np.arange(16)  # Uplift format has 16 joints
        self.body_edges = np.array([
            # Lower body - right leg
            [0, 1],   # right_ankle -> right_knee
            [1, 2],   # right_knee -> right_hip
            # Lower body - left leg
            [5, 4],   # left_ankle -> left_knee
            [4, 3],   # left_knee -> left_hip
            # Lower body - center connections
            [2, 6],   # right_hip -> center_hip
            [3, 6],   # left_hip -> center_hip
            # Upper body - center spine
            [6, 7],   # center_hip -> center_shoulder
            [7, 8],   # center_shoulder -> neck
            [8, 9],   # neck -> head
            # Upper body - right arm
            [12, 11], # right_shoulder -> right_elbow
            [11, 10], # right_elbow -> right_wrist
            [7, 12],  # center_shoulder -> right_shoulder
            # Upper body - left arm
            [13, 14], # left_shoulder -> left_elbow
            [14, 15], # left_elbow -> left_wrist
            [7, 13]   # center_shoulder -> left_shoulder
        ])
        
        self.config = config
        
        # Initialize camera parameters (will be estimated if not provided)
        if os.path.exists(self.calibration_file):
            self.cam_params_dict = self._loadCalibrationParameters()
        else:
            # Create default camera parameters (will be estimated during processing)
            # This will use actual video resolution if available
            self.cam_params_dict = self._initDefaultCameraParams()
            
            # Update resolution from actual video files if available
            self._updateResolutionFromVideos()
    
    def _getVideoResolution(self, cam_name):
        """
        Get actual resolution from video file or extracted frames.
        Returns (width, height) or None if not available.
        """
        # Try to get from extracted video frames first
        frame_dir = os.path.join(self.video_frame_dir, cam_name)
        if os.path.exists(frame_dir):
            frame_files = sorted(glob.glob(os.path.join(frame_dir, '*.png'))) + \
                         sorted(glob.glob(os.path.join(frame_dir, '*.jpg')))
            if len(frame_files) > 0:
                try:
                    img = cv2.imread(frame_files[0])
                    if img is not None:
                        height, width = img.shape[:2]
                        return (width, height)
                except:
                    pass
        
        # Try exact match first
        video_path = os.path.join(self.video_dir, f'{cam_name}.mp4')
        if os.path.exists(video_path) and os.path.isfile(video_path):
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    if width > 0 and height > 0:
                        return (width, height)
            except:
                pass
        
        # Try directory of images
        video_path = os.path.join(self.video_dir, cam_name)
        if os.path.isdir(video_path):
            img_files = sorted(glob.glob(os.path.join(video_path, '*.png'))) + \
                       sorted(glob.glob(os.path.join(video_path, '*.jpg')))
            if len(img_files) > 0:
                try:
                    img = cv2.imread(img_files[0])
                    if img is not None:
                        height, width = img.shape[:2]
                        return (width, height)
                except:
                    pass
        
        # Try to find video file using the mapping
        if cam_name in self.video_file_map and self.video_file_map[cam_name] is not None:
            video_path = self.video_file_map[cam_name]
            if os.path.exists(video_path):
                try:
                    cap = cv2.VideoCapture(video_path)
                    if cap.isOpened():
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cap.release()
                        if width > 0 and height > 0:
                            return (width, height)
                except:
                    pass
        
        # Fallback: try to find any video file in the directory
        if os.path.exists(self.video_dir):
            video_files = glob.glob(os.path.join(self.video_dir, '*.mp4'))
            if len(video_files) > 0:
                # Use mapping or fallback to index
                cam_index = self.cam_names.index(cam_name) if cam_name in self.cam_names else 0
                if cam_index < len(video_files):
                    video_path = video_files[cam_index]
                    try:
                        cap = cv2.VideoCapture(video_path)
                        if cap.isOpened():
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            cap.release()
                            if width > 0 and height > 0:
                                return (width, height)
                    except:
                        pass
        
        return None
    
    def _detectVideoFiles(self):
        """
        Detect and map video files to camera names.
        Matches files by 'front' or 'side' string in filename (case-insensitive).
        """
        if not os.path.exists(self.video_dir):
            return
        
        video_files = sorted(glob.glob(os.path.join(self.video_dir, '*.mp4'))) + \
                     sorted(glob.glob(os.path.join(self.video_dir, '*.avi')))
        
        # Match video files by 'front' or 'side' string in filename
        for video_file in video_files:
            filename_lower = os.path.basename(video_file).lower()
            if 'front' in filename_lower and self.video_file_map.get('front') is None:
                self.video_file_map['front'] = video_file
            elif 'side' in filename_lower and self.video_file_map.get('side') is None:
                self.video_file_map['side'] = video_file
        
        # If not all cameras mapped, use order (first = front, second = side)
        unmapped_cams = [cam for cam in self.cam_names if self.video_file_map[cam] is None]
        unmapped_files = [f for f in video_files if f not in self.video_file_map.values()]
        
        for i, cam_name in enumerate(unmapped_cams):
            if i < len(unmapped_files):
                self.video_file_map[cam_name] = unmapped_files[i]
    
    def _updateResolutionFromVideos(self):
        """
        Update camera resolution from actual video files.
        This is called after initialization to get real video dimensions.
        """
        for cam_name in self.cam_names:
            resolution = self._getVideoResolution(cam_name)
            if resolution is not None:
                width, height = resolution
                # Update camera parameters with actual resolution
                if cam_name in self.cam_params_dict:
                    # Update K matrix with new resolution
                    focal = self.cam_params_dict[cam_name]['K'][0, 0]
                    self.cam_params_dict[cam_name]['K'] = np.array([
                        [focal, 0, width / 2],
                        [0, focal, height / 2],
                        [0, 0, 1]
                    ])
                    self.cam_params_dict[cam_name]['resolution'] = resolution
                    print(f"Updated {cam_name} resolution to {resolution} from video")
    
    def _initDefaultCameraParams(self):
        """
        Initialize default camera parameters.
        These will be refined during pose estimation.
        First tries to get resolution from JSON files (keypoints_2d), then from videos.
        """
        cameras = {}
        default_focal = 1000.0  # Approximate focal length in pixels
        
        for i, cam_name in enumerate(self.cam_names):
            # Priority order for resolution:
            # 1. Explicit resolution field in JSON (img_resolution, resolution, width/height) - most reliable
            # 2. Video file resolution (reliable if video exists)
            # 3. Infer from keypoint coordinates (less reliable, keypoints may not cover full image)
            # 4. Default resolution
            
            resolution = None
            resolution_source = None
            
            # First, try to get explicit resolution from JSON
            json_res = self._inferResolutionFromKeypoints(cam_name)
            if json_res:
                resolution = json_res
                # Check if it came from explicit field or was inferred
                # (The function checks for img_resolution/resolution/width/height first)
                resolution_source = "JSON (img_resolution field)"
            
            # Prefer video resolution over inferred keypoint resolution (but not over explicit JSON resolution)
            if resolution is None:
                video_res = self._getVideoResolution(cam_name)
                if video_res is not None:
                    resolution = video_res
                    resolution_source = "video file"
            
            if resolution is None:
                # Final fallback
                resolution = (1920, 1080)
                resolution_source = "default"
                print(f"Warning: Using default resolution {resolution} for {cam_name}")
            else:
                print(f"Using resolution {resolution} for {cam_name} from {resolution_source}")
            
            width, height = resolution
            
            K = np.array([
                [default_focal, 0, width / 2],
                [0, default_focal, height / 2],
                [0, 0, 1]
            ])
            
            # Default extrinsic (identity for first camera, will be estimated)
            R = np.eye(3)
            t = np.zeros((3, 1))
            M = np.concatenate([R, t], axis=1)
            
            cameras[cam_name] = {
                'K': K,
                'distCoef': np.zeros(5),
                'R': R,
                't': t,
                'M': M,
                'resolution': resolution,
                'name': cam_name
            }
        return cameras
    
    def _inferResolutionFromKeypoints(self, cam_name):
        """
        Try to infer image resolution from keypoint coordinates in JSON files.
        First tries to read from video directory JSON files (with keypoints_2d key),
        then falls back to keypoints directory.
        Returns (width, height) or None.
        """
        try:
            # First, try to find JSON files in videos directory (with keypoints_2d key)
            keypoint_file = None
            
            # Try video directory first (for keypoints_2d format)
            # Match by camera name in filename (case-insensitive)
            video_json_files = glob.glob(os.path.join(self.video_dir, '*.json'))
            matching_files = [f for f in video_json_files if cam_name in os.path.basename(f).lower()]
            if matching_files:
                keypoint_file = matching_files[0]
            
            # If not found, try alternative matching
            if not keypoint_file and self.keypoints_dir == self.video_dir:
                all_json_files = glob.glob(os.path.join(self.keypoints_dir, '*.json'))
                matching_files = [f for f in all_json_files if cam_name in os.path.basename(f).lower()]
                if matching_files:
                    keypoint_file = matching_files[0]
            
            # Fallback to keypoints directory
            if not keypoint_file:
                keypoint_file = os.path.join(self.keypoints_dir, f'{cam_name}.json')
                if not os.path.exists(keypoint_file):
                    keypoint_dir = os.path.join(self.keypoints_dir, cam_name)
                    if os.path.isdir(keypoint_dir):
                        keypoint_files = sorted(glob.glob(os.path.join(keypoint_dir, '*.json')))
                        if len(keypoint_files) > 0:
                            keypoint_file = keypoint_files[0]
            
            if keypoint_file and os.path.exists(keypoint_file):
                with open(keypoint_file, 'r') as f:
                    data = json.load(f)
                
                # First, check if JSON has explicit resolution fields (priority order)
                # Check img_resolution first (common in keypoint JSON files)
                if 'img_resolution' in data:
                    res = data['img_resolution']
                    if isinstance(res, (list, tuple)) and len(res) >= 2:
                        return (int(res[0]), int(res[1]))
                
                # Check resolution field
                if 'resolution' in data:
                    res = data['resolution']
                    if isinstance(res, (list, tuple)) and len(res) >= 2:
                        return (int(res[0]), int(res[1]))
                    elif isinstance(res, dict):
                        if 'width' in res and 'height' in res:
                            return (int(res['width']), int(res['height']))
                
                # Check width/height fields
                if 'width' in data and 'height' in data:
                    return (int(data['width']), int(data['height']))
                
                # Extract keypoints - try keypoints_2d format first
                kpts = None
                if 'keypoints_2d' in data:
                    kpts_data = data['keypoints_2d']
                    if isinstance(kpts_data, list) and len(kpts_data) > 0:
                        # Get first frame
                        frame_kpts = kpts_data[0]
                        if isinstance(frame_kpts, list) and len(frame_kpts) > 0:
                            if isinstance(frame_kpts[0], list):
                                # Already in [[x, y], ...] format
                                kpts = np.array([[pt[0], pt[1]] if len(pt) >= 2 else [np.nan, np.nan] for pt in frame_kpts])
                            else:
                                # Flat list, try to reshape
                                kpts = np.array(frame_kpts).reshape(-1, 2)
                
                # Fallback to 'keypoints' format
                if kpts is None and 'keypoints' in data:
                    kpts_data = data['keypoints']
                    if isinstance(kpts_data, list) and len(kpts_data) > 0:
                        if isinstance(kpts_data[0], list):
                            if isinstance(kpts_data[0][0], list):
                                # List of frames - use first frame
                                kpts = np.array(kpts_data[0])
                            else:
                                # Single frame
                                kpts = np.array(kpts_data)
                
                if kpts is not None and len(kpts) > 0:
                    # Get max coordinates and add padding
                    valid_mask = ~np.isnan(kpts).any(axis=1)
                    if np.any(valid_mask):
                        valid_kpts = kpts[valid_mask]
                        max_x = np.max(valid_kpts[:, 0]) if valid_kpts.shape[1] >= 1 else 0
                        max_y = np.max(valid_kpts[:, 1]) if valid_kpts.shape[1] >= 2 else 0
                        
                        # Add 20% padding and round to common resolutions
                        width = int((max_x * 1.2) // 100 * 100)  # Round to nearest 100
                        height = int((max_y * 1.2) // 100 * 100)
                        
                        # Ensure reasonable minimum
                        width = max(width, 640)
                        height = max(height, 480)
                        
                        # Round to common video resolutions
                        common_widths = [640, 1280, 1920, 2560]
                        common_heights = [480, 720, 1080, 1440]
                        
                        width = min(common_widths, key=lambda x: abs(x - width))
                        height = min(common_heights, key=lambda x: abs(x - height))
                        
                        return (width, height)
        except Exception as e:
            pass
        
        return None
    
    def _loadCalibrationParameters(self):
        """Load camera calibration from JSON file."""
        cameras_raw = json.load(open(self.calibration_file, 'r'))
        cameras = {}
        for cam_name, cam_params in cameras_raw.items():
            cameras[cam_name] = {
                'K': np.array(cam_params['K']),
                'distCoef': np.array(cam_params['distCoef']),
                'R': np.array(cam_params['R']),
                't': np.array(cam_params['t']).reshape((3, 1)),
                'M': np.array(cam_params['M']),
                'resolution': tuple(cam_params['resolution']),
                'name': cam_name
            }
        return cameras
    
    def loadKeypoints(self, keypoint_file):
        """
        Load keypoints from JSON file.
        Supports multiple formats:
        1. Uplift format: {'keypoints': [x1, y1, x2, y2, ...]} or [[x1, y1], [x2, y2], ...]
        2. COCO format: {'keypoints': [x1, y1, v1, x2, y2, v2, ...]}
        3. Custom format: {'keypoints': [[x1, y1], [x2, y2], ...]}
        4. OpenPose format: {'people': [{'pose_keypoints_2d': [...]}]}
        
        Expected: 16 keypoints in Uplift Order:
        0: right_ankle, 1: right_knee, 2: right_hip, 3: left_hip,
        4: left_knee, 5: left_ankle, 6: center_hip, 7: center_shoulder,
        8: neck, 9: head, 10: right_wrist, 11: right_elbow,
        12: right_shoulder, 13: left_shoulder, 14: left_elbow, 15: left_wrist
        """
        with open(keypoint_file, 'r') as f:
            data = json.load(f)
        
        # Try different formats
        if 'keypoints' in data:
            kpts_data = data['keypoints']
            
            # Format 1: List of [x, y] pairs: [[x1, y1], [x2, y2], ...]
            if isinstance(kpts_data, list) and len(kpts_data) > 0:
                if isinstance(kpts_data[0], list):
                    kpts = np.array(kpts_data)
                    # Ensure we have exactly 16 keypoints
                    if len(kpts) == 16:
                        return kpts
                    elif len(kpts) > 16:
                        return kpts[:16]
                    else:
                        # Pad with NaN if less than 16
                        padded = np.full((16, 2), np.nan)
                        padded[:len(kpts)] = kpts
                        return padded
            
            # Format 2: Flat list [x1, y1, x2, y2, ...] or [x1, y1, v1, x2, y2, v2, ...]
            kpts_flat = np.array(kpts_data)
            
            # Check if it's [x, y, v, ...] format (COCO style)
            if len(kpts_flat) % 3 == 0:
                kpts = kpts_flat.reshape(-1, 3)[:, :2]  # Take only x, y
            # Otherwise assume [x, y, x, y, ...] format
            elif len(kpts_flat) % 2 == 0:
                kpts = kpts_flat.reshape(-1, 2)
            else:
                raise ValueError(f"Invalid keypoint format: length {len(kpts_flat)} is not divisible by 2 or 3")
            
            # Ensure we have exactly 16 keypoints
            if len(kpts) == 16:
                return kpts
            elif len(kpts) > 16:
                return kpts[:16]
            else:
                # Pad with NaN if less than 16
                padded = np.full((16, 2), np.nan)
                padded[:len(kpts)] = kpts
                return padded
        
        # OpenPose format
        if 'people' in data and len(data['people']) > 0:
            kpts_flat = np.array(data['people'][0]['pose_keypoints_2d'])
            kpts = kpts_flat.reshape(-1, 3)[:, :2]
            # Ensure 16 keypoints
            if len(kpts) >= 16:
                return kpts[:16]
            else:
                padded = np.full((16, 2), np.nan)
                padded[:len(kpts)] = kpts
                return padded
        
        # Direct list format
        if isinstance(data, list):
            kpts = np.array(data)
            if len(kpts) == 16:
                return kpts
            elif len(kpts) > 16:
                return kpts[:16]
            else:
                padded = np.full((16, 2), np.nan)
                padded[:len(kpts)] = kpts
                return padded
        
        # Format with 'bodies' key (from prepareDataFromKeypoints)
        if 'bodies' in data and len(data['bodies']) > 0:
            # Get first body's joints
            body = data['bodies'][0]
            if 'joints' in body:
                kpts = np.array(body['joints'])
                if len(kpts) == 16:
                    return kpts
                elif len(kpts) > 16:
                    return kpts[:16]
                else:
                    padded = np.full((16, 2), np.nan)
                    padded[:len(kpts)] = kpts
                    return padded
        
        raise ValueError(f"Unknown keypoint format in {keypoint_file}")
    
    def extractFramesFromVideo(self, video_path, cam_name, max_frames=None):
        """
        Extract frames from video file.
        
        Args:
            video_path: Path to video file or directory of images
            cam_name: Camera name
            max_frames: Maximum number of frames to extract (None for all)
        """
        cam_frame_dir = os.path.join(self.video_frame_dir, cam_name)
        os.makedirs(cam_frame_dir, exist_ok=True)
        
        # Check if already extracted
        existing_frames = sorted(glob.glob(os.path.join(cam_frame_dir, '*.png')))
        if len(existing_frames) > 0:
            print(f"Frames already extracted for {cam_name}, skipping...")
            return len(existing_frames)
        
        # Check if video_path is a directory of images
        if os.path.isdir(video_path):
            image_files = sorted(glob.glob(os.path.join(video_path, '*')))
            image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for i, img_path in enumerate(image_files):
                if max_frames and i >= max_frames:
                    break
                frame_id = i
                frame_file = os.path.join(cam_frame_dir, f'{frame_id:08d}.png')
                img = cv2.imread(img_path)
                cv2.imwrite(frame_file, img)
            return len(image_files)
        
        # Extract from video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_count >= max_frames:
                break
            
            frame_file = os.path.join(cam_frame_dir, f'{frame_count:08d}.png')
            cv2.imwrite(frame_file, frame)
            frame_count += 1
        
        cap.release()
        return frame_count
    
    # ============================================================================
    # File Discovery and Validation Methods
    # ============================================================================
    
    def _findKeypointFiles(self):
        """
        Find keypoint JSON files for all cameras.
        
        Returns:
            tuple: (keypoint_files dict, keypoints_from_videos dict)
        """
        keypoint_files = {}
        keypoints_from_videos = {}
        
        for cam_name in self.cam_names:
            found = False
            
            # Look for JSON files containing camera name in filename (case-insensitive)
            video_json_files = glob.glob(os.path.join(self.video_dir, '*.json'))
            matching_files = [f for f in video_json_files if cam_name in os.path.basename(f).lower()]
            
            if matching_files:
                # Use the first matching file
                keypoint_files[cam_name] = [matching_files[0]]
                keypoints_from_videos[cam_name] = True
                print(f"Found keypoints in video directory for {cam_name}: {os.path.basename(matching_files[0])}")
                found = True
            
            # If not found, try alternative matching (for video_path mode)
            if not found and self.keypoints_dir == self.video_dir:
                # In video_path mode, keypoints_dir == video_dir, so look for any JSON files
                all_json_files = glob.glob(os.path.join(self.keypoints_dir, '*.json'))
                # Match by camera name in filename (case-insensitive)
                matching_files = [f for f in all_json_files if cam_name in os.path.basename(f).lower()]
                if matching_files:
                    keypoint_files[cam_name] = [matching_files[0]]
                    keypoints_from_videos[cam_name] = True
                    print(f"Found keypoints for {cam_name} by pattern matching: {os.path.basename(matching_files[0])}")
                    found = True
            
            if found:
                continue
            
            # Fallback to keypoints directory structure
            cam_keypoint_dir = os.path.join(self.keypoints_dir, cam_name)
            
            # Check if it's a directory or single file
            if os.path.isdir(cam_keypoint_dir):
                # Directory of JSON files (one per frame)
                keypoint_files[cam_name] = sorted(glob.glob(os.path.join(cam_keypoint_dir, '*.json')))
                keypoints_from_videos[cam_name] = False
            elif os.path.isfile(cam_keypoint_dir + '.json'):
                # Single JSON file
                keypoint_files[cam_name] = [cam_keypoint_dir + '.json']
                keypoints_from_videos[cam_name] = False
            else:
                # Try with .json extension
                json_file = os.path.join(self.keypoints_dir, f'{cam_name}.json')
                if os.path.exists(json_file):
                    keypoint_files[cam_name] = [json_file]
                    keypoints_from_videos[cam_name] = False
                else:
                    # Check if there are JSON files but they don't contain 'front' or 'side' in filename
                    all_json_files = glob.glob(os.path.join(self.video_dir, '*.json')) + \
                                   glob.glob(os.path.join(self.keypoints_dir, '*.json'))
                    all_json_files = list(set(all_json_files))  # Remove duplicates
                    
                    if all_json_files:
                        # Check if any JSON files exist but don't have 'front' or 'side' in name
                        json_basenames = [os.path.basename(f) for f in all_json_files]
                        has_front = any('front' in name.lower() for name in json_basenames)
                        has_side = any('side' in name.lower() for name in json_basenames)
                        
                        if not has_front or not has_side:
                            error_msg = f"\n{'='*70}\n"
                            error_msg += f"ERROR: Cannot identify front/side camera files\n"
                            error_msg += f"{'='*70}\n"
                            error_msg += f"Found {len(all_json_files)} JSON file(s) but cannot determine which is front/side:\n"
                            for f in all_json_files:
                                error_msg += f"  - {os.path.basename(f)}\n"
                            error_msg += f"\nSOLUTION: Rename your JSON files to include 'front' or 'side' in the filename.\n"
                            error_msg += f"Examples:\n"
                            error_msg += f"  - front_keypoints.json (or *front*.json)\n"
                            error_msg += f"  - side_keypoints.json (or *side*.json)\n"
                            error_msg += f"\nSearched in:\n"
                            error_msg += f"  - {self.video_dir}\n"
                            error_msg += f"  - {self.keypoints_dir}\n"
                            error_msg += f"{'='*70}\n"
                            raise ValueError(error_msg)
                    
                    # No JSON files found at all
                    raise FileNotFoundError(
                        f"Keypoint file not found for {cam_name}.\n"
                        f"Searched in: {self.video_dir} and {self.keypoints_dir}\n"
                        f"Expected: JSON files containing 'front' or 'side' in filename"
                    )
        
        return keypoint_files, keypoints_from_videos
    
    def _raiseFileIdentificationError(self, all_json_files):
        """Raise error when JSON files cannot be identified as front/side."""
        json_basenames = [os.path.basename(f) for f in all_json_files]
        has_front = any('front' in name.lower() for name in json_basenames)
        has_side = any('side' in name.lower() for name in json_basenames)
        
        if not has_front or not has_side:
            error_msg = f"\n{'='*70}\n"
            error_msg += f"ERROR: Cannot identify front/side camera files\n"
            error_msg += f"{'='*70}\n"
            error_msg += f"Found {len(all_json_files)} JSON file(s) but cannot determine which is front/side:\n"
            for f in all_json_files:
                error_msg += f"  - {os.path.basename(f)}\n"
            error_msg += f"\nSOLUTION: Rename your JSON files to include 'front' or 'side' in the filename.\n"
            error_msg += f"Examples:\n"
            error_msg += f"  - front_keypoints.json (or *front*.json)\n"
            error_msg += f"  - side_keypoints.json (or *side*.json)\n"
            error_msg += f"\nSearched in:\n"
            error_msg += f"  - {self.video_dir}\n"
            error_msg += f"  - {self.keypoints_dir}\n"
            error_msg += f"{'='*70}\n"
            raise ValueError(error_msg)
    
    def _validateJsonFile(self, json_file, cam_name):
        """
        Validate that JSON file contains required keys.
        
        Args:
            json_file: Path to JSON file
            cam_name: Camera name (for error messages)
        
        Raises:
            ValueError: If required keys are missing or invalid
        """
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Validate 'keypoints_2d' key exists
            if 'keypoints_2d' not in data:
                error_msg = f"\n{'='*70}\n"
                error_msg += f"ERROR: Missing required key 'keypoints_2d' in JSON file\n"
                error_msg += f"{'='*70}\n"
                error_msg += f"File: {os.path.basename(json_file)}\n"
                error_msg += f"Camera: {cam_name}\n"
                error_msg += f"\nJSON files MUST contain the 'keypoints_2d' key with 2D keypoint data.\n"
                error_msg += f"Found keys in file: {list(data.keys())[:10]}\n"
                error_msg += f"{'='*70}\n"
                raise ValueError(error_msg)
            
            # Validate 'img_resolution' key exists, or check if we can get from video
            has_img_resolution = 'img_resolution' in data
            has_resolution = 'resolution' in data
            has_width_height = 'width' in data and 'height' in data
            
            if not (has_img_resolution or has_resolution or has_width_height):
                # Try to get resolution from video file
                video_res = self._getVideoResolution(cam_name)
                if video_res is None:
                    error_msg = f"\n{'='*70}\n"
                    error_msg += f"ERROR: Missing required key 'img_resolution' in JSON file\n"
                    error_msg += f"{'='*70}\n"
                    error_msg += f"File: {os.path.basename(json_file)}\n"
                    error_msg += f"Camera: {cam_name}\n"
                    error_msg += f"\nJSON files MUST contain the 'img_resolution' key with [width, height].\n"
                    error_msg += f"Alternatively, provide a video file to extract resolution from.\n"
                    error_msg += f"\nFound keys in file: {list(data.keys())[:10]}\n"
                    error_msg += f"\nSOLUTION:\n"
                    error_msg += f"  1. Add 'img_resolution': [width, height] to your JSON file, OR\n"
                    error_msg += f"  2. Provide a video file (with '{cam_name}' in filename) to extract resolution\n"
                    error_msg += f"\nExample JSON structure:\n"
                    error_msg += f"  {{\n"
                    error_msg += f"    \"keypoints_2d\": [...],\n"
                    error_msg += f"    \"img_resolution\": [960, 1080]\n"
                    error_msg += f"  }}\n"
                    error_msg += f"{'='*70}\n"
                    raise ValueError(error_msg)
                else:
                    print(f"Warning: JSON file for {cam_name} missing 'img_resolution', using resolution from video: {video_res}")
            else:
                # Validate img_resolution format
                if has_img_resolution:
                    res = data['img_resolution']
                    if not (isinstance(res, (list, tuple)) and len(res) >= 2):
                        error_msg = f"\n{'='*70}\n"
                        error_msg += f"ERROR: Invalid 'img_resolution' format in JSON file\n"
                        error_msg += f"{'='*70}\n"
                        error_msg += f"File: {os.path.basename(json_file)}\n"
                        error_msg += f"Camera: {cam_name}\n"
                        error_msg += f"\n'img_resolution' must be a list/tuple with [width, height].\n"
                        error_msg += f"Found: {res}\n"
                        error_msg += f"\nExample: \"img_resolution\": [960, 1080]\n"
                        error_msg += f"{'='*70}\n"
                        raise ValueError(error_msg)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file for {cam_name}: {json_file}\nError: {e}")
        except Exception as e:
            if isinstance(e, (ValueError, FileNotFoundError)):
                raise
            raise ValueError(f"Error reading JSON file for {cam_name}: {json_file}\nError: {e}")
    
    def _validateAllJsonFiles(self, keypoint_files):
        """Validate all JSON files have required keys."""
        for cam_name in self.cam_names:
            if cam_name not in keypoint_files or len(keypoint_files[cam_name]) == 0:
                continue
            
            json_file = keypoint_files[cam_name][0]
            self._validateJsonFile(json_file, cam_name)
    
    # ============================================================================
    # Frame Counting Methods
    # ============================================================================
    
    def _countFrames(self, keypoint_files):
        """
        Determine the number of frames from keypoint files.
        
        Args:
            keypoint_files: Dict mapping camera names to lists of keypoint file paths
        
        Returns:
            int: Number of frames
        """
        # If single file, check if it contains multiple frames
        if all(len(files) == 1 for files in keypoint_files.values()):
            # Check if single file contains multiple frames
            sample_file = keypoint_files[self.cam_names[0]][0]
            kpts_data = json.load(open(sample_file, 'r'))
            
            # Check for keypoints_2d key (from video JSON files)
            if 'keypoints_2d' in kpts_data:
                kpts_list = kpts_data['keypoints_2d']
                if isinstance(kpts_list, list) and len(kpts_list) > 0:
                    n_frames = len(kpts_list)
                    print(f"Found {n_frames} frames in 'keypoints_2d' array")
                else:
                    n_frames = 1
                    print(f"Found 1 frame")
            elif 'keypoints' in kpts_data:
                kpts = kpts_data['keypoints']
                # Check if it's a list of frames: [[[x,y],...], [[x,y],...], ...]
                if isinstance(kpts, list) and len(kpts) > 0:
                    if isinstance(kpts[0], list) and len(kpts[0]) > 0:
                        if isinstance(kpts[0][0], list):
                            # This is a list of frames!
                            n_frames = len(kpts)
                            print(f"Found {n_frames} frames in single JSON file")
                        else:
                            # Single frame
                            n_frames = 1
                            print(f"Found 1 frame (single frame format)")
                    else:
                        n_frames = 1
                        print(f"Found 1 frame")
                else:
                    n_frames = 1
                    print(f"Found 1 frame")
            elif 'frames' in kpts_data:
                n_frames = len(kpts_data['frames'])
                print(f"Found {n_frames} frames in 'frames' array")
            else:
                n_frames = 1
                print(f"Found 1 frame")
        else:
            # Multiple files - one per frame
            n_frames = min(len(files) for files in keypoint_files.values())
            print(f"Found {n_frames} frames (multiple files)")
        
        # Process each frame
        for frame_id in range(n_frames):
            bodies = []
            
            for cam_name in self.cam_names:
                # Load keypoints for this camera and frame
                if len(keypoint_files[cam_name]) == 1:
                    # Single file - might contain all frames or single frame
                    kpts_data = json.load(open(keypoint_files[cam_name][0], 'r'))
                    
                    # Check for keypoints_2d key (from video JSON files)
                    # Format: [[x, y, visibility], [x, y, visibility], ...] for each frame
                    if 'keypoints_2d' in kpts_data:
                        kpts_list = kpts_data['keypoints_2d']
                        if isinstance(kpts_list, list) and len(kpts_list) > 0:
                            if frame_id < len(kpts_list):
                                # Extract x, y coordinates (ignore visibility for now)
                                frame_kpts = kpts_list[frame_id]
                                if isinstance(frame_kpts, list) and len(frame_kpts) > 0:
                                    # Convert [[x, y, v], ...] to [[x, y], ...]
                                    kpts = np.array([[pt[0], pt[1]] for pt in frame_kpts if len(pt) >= 2])
                                else:
                                    kpts = np.array([])
                            else:
                                # Use last frame if frame_id exceeds
                                last_frame = kpts_list[-1]
                                kpts = np.array([[pt[0], pt[1]] for pt in last_frame if len(pt) >= 2])
                        else:
                            kpts = np.array([])
                    # Check if it's a list of frames
                    elif 'keypoints' in kpts_data:
                        kpts_list = kpts_data['keypoints']
                        if isinstance(kpts_list, list) and len(kpts_list) > 0:
                            # Check if first element is a list of coordinates (list of frames)
                            if isinstance(kpts_list[0], list) and len(kpts_list[0]) > 0:
                                if isinstance(kpts_list[0][0], list):
                                    # List of frames: [[[x,y],...], [[x,y],...], ...]
                                    # Each element is a frame with 16 keypoints
                                    if frame_id < len(kpts_list):
                                        kpts = np.array(kpts_list[frame_id])
                                    else:
                                        # Use last frame if frame_id exceeds
                                        kpts = np.array(kpts_list[-1])
                                else:
                                    # Single frame: [[x,y], [x,y], ...] - 16 keypoints
                                    kpts = np.array(kpts_list)
                            else:
                                # Single frame or flat list
                                kpts = self.loadKeypoints(keypoint_files[cam_name][0])
                        else:
                            kpts = self.loadKeypoints(keypoint_files[cam_name][0])
                    elif 'frames' in kpts_data:
                        # Format: {"frames": [{"keypoints": [...]}, ...]}
                        if frame_id < len(kpts_data['frames']):
                            frame_data = kpts_data['frames'][frame_id]
                            if 'keypoints' in frame_data:
                                kpts = np.array(frame_data['keypoints'])
                            else:
                                kpts = self.loadKeypoints(keypoint_files[cam_name][0])
                        else:
                            kpts = np.array(kpts_data['frames'][-1].get('keypoints', []))
                    else:
                        # Try loading as single frame
                        kpts = self.loadKeypoints(keypoint_files[cam_name][0])
                else:
                    # Multiple files - one per frame
                    kpts = self.loadKeypoints(keypoint_files[cam_name][frame_id])
                
                # Ensure correct number of joints (16 for Uplift format)
                if len(kpts) > 16:
                    kpts = kpts[:16]
                elif len(kpts) < 16:
                    # Pad with NaN if needed
                    padded = np.full((16, 2), np.nan)
                    padded[:len(kpts)] = kpts
                    kpts = padded
                
                # NOTE: We no longer save pose2d_label files to avoid unnecessary file I/O
                # Keypoints are loaded directly from source JSON files when needed
        
        # Store n_frames as instance variable for easy access
        self.n_frames = n_frames
        
        print(f"Prepared {n_frames} frames of keypoint data")
        return n_frames
    
    def _loadKeypointsFromJson(self, json_file, frame_id=0):
        """
        Load keypoints from a JSON file for a specific frame.
        
        Args:
            json_file: Path to JSON file
            frame_id: Frame index (for multi-frame files)
        
        Returns:
            np.ndarray: Keypoints array of shape (16, 2)
        """
        kpts_data = json.load(open(json_file, 'r'))
        
        # Check for keypoints_2d key (from video JSON files)
        if 'keypoints_2d' in kpts_data:
            kpts_list = kpts_data['keypoints_2d']
            if isinstance(kpts_list, list) and len(kpts_list) > 0:
                if frame_id < len(kpts_list):
                    frame_kpts = kpts_list[frame_id]
                    if isinstance(frame_kpts, list) and len(frame_kpts) > 0:
                        kpts = np.array([[pt[0], pt[1]] for pt in frame_kpts if len(pt) >= 2])
                    else:
                        kpts = np.array([])
                else:
                    last_frame = kpts_list[-1]
                    kpts = np.array([[pt[0], pt[1]] for pt in last_frame if len(pt) >= 2])
            else:
                kpts = np.array([])
        elif 'keypoints' in kpts_data:
            kpts_list = kpts_data['keypoints']
            if isinstance(kpts_list, list) and len(kpts_list) > 0:
                if isinstance(kpts_list[0], list) and len(kpts_list[0]) > 0:
                    if isinstance(kpts_list[0][0], list):
                        # List of frames
                        if frame_id < len(kpts_list):
                            kpts = np.array(kpts_list[frame_id])
                        else:
                            kpts = np.array(kpts_list[-1])
                    else:
                        # Single frame
                        kpts = np.array(kpts_list)
                else:
                    kpts = self.loadKeypoints(json_file)
            else:
                kpts = self.loadKeypoints(json_file)
        elif 'frames' in kpts_data:
            if frame_id < len(kpts_data['frames']):
                frame_data = kpts_data['frames'][frame_id]
                if 'keypoints' in frame_data:
                    kpts = np.array(frame_data['keypoints'])
                else:
                    kpts = self.loadKeypoints(json_file)
            else:
                kpts = np.array(kpts_data['frames'][-1].get('keypoints', []))
        else:
            kpts = self.loadKeypoints(json_file)
        
        # Ensure correct number of joints (16 for Uplift format)
        if len(kpts) > 16:
            kpts = kpts[:16]
        elif len(kpts) < 16:
            padded = np.full((16, 2), np.nan)
            padded[:len(kpts)] = kpts
            kpts = padded
        
        return kpts
    
    # ============================================================================
    # Main Data Preparation Method
    # ============================================================================
    
    def prepareDataFromKeypoints(self):
        """
        Prepare data structure from keypoint files.
        Converts keypoint files to the expected format.
        First tries to read from videos directory JSON files (keypoints_2d key),
        then falls back to keypoints directory.
        
        Validates that JSON files contain:
        - 'keypoints_2d': Required key with 2D keypoint data
        - 'img_resolution': Required key with [width, height], or video file must be provided
        
        Raises:
            ValueError: If required keys are missing in JSON files
        """
        print("Preparing data from keypoints...")
        
        # Find keypoint files
        keypoint_files, keypoints_from_videos = self._findKeypointFiles()
        
        # Validate JSON files
        self._validateAllJsonFiles(keypoint_files)
        
        # Count frames
        n_frames = self._countFrames(keypoint_files)
        
        # Process each frame (validation only - we don't save files anymore)
        for frame_id in range(n_frames):
            for cam_name in self.cam_names:
                # Load keypoints for validation (we don't save them anymore)
                if len(keypoint_files[cam_name]) == 1:
                    self._loadKeypointsFromJson(keypoint_files[cam_name][0], frame_id)
                else:
                    self._loadKeypointsFromJson(keypoint_files[cam_name][frame_id], 0)
        
        # Store n_frames as instance variable for easy access
        self.n_frames = n_frames
        
        print(f"Prepared {n_frames} frames of keypoint data")
        return n_frames
    
    # ============================================================================
    # Frame Data Access Methods
    # ============================================================================
    
    def getSingleFrameMultiView2DJoints(self, frame_id):
        """Extract 2D joints for a single frame from all views."""
        joints_dict = {}
        
        for cam_name in self.cam_names:
            # First try pose2d_file_dir (for backward compatibility with old data)
            pose_file = os.path.join(self.pose2d_file_dir, cam_name, f'{frame_id:08d}.json')
            
            if os.path.exists(pose_file):
                frame_joints = json.load(open(pose_file, 'r'))
                frame_joints_dict = {}
                
                for body in frame_joints['bodies']:
                    person_id = body['id']
                    joints_2d = np.array(body['joints'])
                    if len(joints_2d) > len(self.joints_ids):
                        joints_2d = joints_2d[:len(self.joints_ids)]
                    frame_joints_dict[(frame_id, person_id)] = joints_2d
                
                joints_dict[cam_name] = frame_joints_dict
            else:
                # Load directly from source JSON files (keypoints_2d format)
                # Find the keypoint file for this camera
                keypoint_file = None
                
                # Try to find JSON file in video directory
                if self.keypoints_dir == self.video_dir:
                    # Look for JSON files matching camera name in filename (case-insensitive)
                    json_files = glob.glob(os.path.join(self.video_dir, '*.json'))
                    matching = [f for f in json_files if cam_name in os.path.basename(f).lower()]
                    if matching:
                        keypoint_file = matching[0]
                    else:
                        # If no match found, check if JSON files exist but don't have front/side in name
                        if json_files:
                            json_basenames = [os.path.basename(f) for f in json_files]
                            has_front = any('front' in name.lower() for name in json_basenames)
                            has_side = any('side' in name.lower() for name in json_basenames)
                            
                            if not has_front or not has_side:
                                error_msg = f"\n{'='*70}\n"
                                error_msg += f"ERROR: Cannot identify {cam_name} camera file\n"
                                error_msg += f"{'='*70}\n"
                                error_msg += f"Found {len(json_files)} JSON file(s) but cannot determine which is {cam_name}:\n"
                                for f in json_files:
                                    error_msg += f"  - {os.path.basename(f)}\n"
                                error_msg += f"\nSOLUTION: Rename your JSON files to include 'front' or 'side' in the filename.\n"
                                error_msg += f"Examples:\n"
                                error_msg += f"  - front_keypoints.json (or *front*.json)\n"
                                error_msg += f"  - side_keypoints.json (or *side*.json)\n"
                                error_msg += f"{'='*70}\n"
                                raise ValueError(error_msg)
                
                if keypoint_file and os.path.exists(keypoint_file):
                    try:
                        with open(keypoint_file, 'r') as f:
                            kpts_data = json.load(f)
                        
                        # Check for keypoints_2d format
                        if 'keypoints_2d' in kpts_data:
                            kpts_list = kpts_data['keypoints_2d']
                            if isinstance(kpts_list, list) and len(kpts_list) > frame_id:
                                frame_kpts = kpts_list[frame_id]
                                # Convert to [[x, y], ...] format
                                if isinstance(frame_kpts, list) and len(frame_kpts) > 0:
                                    if isinstance(frame_kpts[0], list):
                                        # Already in [[x, y], ...] format
                                        keypoints_2d = np.array([[pt[0], pt[1]] if len(pt) >= 2 else [np.nan, np.nan] for pt in frame_kpts])
                                    else:
                                        # Flat list, try to reshape
                                        keypoints_2d = np.array(frame_kpts).reshape(-1, 2)
                                    
                                    # Ensure 16 keypoints
                                    if len(keypoints_2d) > 16:
                                        keypoints_2d = keypoints_2d[:16]
                                    elif len(keypoints_2d) < 16:
                                        padded = np.full((16, 2), np.nan)
                                        padded[:len(keypoints_2d)] = keypoints_2d
                                        keypoints_2d = padded
                                    
                                    frame_joints_dict = {(frame_id, 0): keypoints_2d}
                                    joints_dict[cam_name] = frame_joints_dict
                                    continue
                    except Exception:
                        pass
                
                # Fallback: empty dict
                joints_dict[cam_name] = {}
        
        return joints_dict
    
    def fetchVideoFrameFile(self, cam_name, frame_id):
        """Get path to video frame file."""
        return os.path.join(self.video_frame_dir, cam_name, f'{frame_id:08d}.png')
    
    def getSingleFrameMultiViewBoxes(self, frame_id, **kwargs):
        """
        Crop bounding boxes from multiple views for a video frame.
        Uses config if available, otherwise uses default parameters.
        """
        if self.config is not None and 'boxprocessing' in self.config:
            box_config = self.config['boxprocessing']
            box_joints_margin = box_config.get('box_joints_margin', [1.0, 1.1])
            box_size_thold = box_config.get('box_size_thold', [20, 20])
            box_ios_thold = box_config.get('box_ios_thold', 0.3)
            joints_inside_img_ratio = box_config.get('joints_inside_img_ratio', 0.6)
            box_inside_img_ratio = box_config.get('box_inside_img_ratio', 0.6)
            resize = box_config.get('resize', [128, 256])
            replace_old = box_config.get('replace_old', False)
            verbose = box_config.get('verbose', True)
        else:
            box_joints_margin = [1.0, 1.1]
            box_size_thold = [20, 20]
            box_ios_thold = 0.3
            joints_inside_img_ratio = 0.6
            box_inside_img_ratio = 0.6
            resize = [128, 256]
            replace_old = False
            verbose = True
        
        save_crop_dir = os.path.join(self.boxcrop_dir, f'frame{frame_id:08d}')
        box_joint_map_file = os.path.join(save_crop_dir, 'box_joints_map.pkl')
        
        if os.path.exists(box_joint_map_file) and not replace_old:
            return save_crop_dir
        
        if not os.path.exists(save_crop_dir):
            os.makedirs(save_crop_dir)
        
        joints_dict = self.getSingleFrameMultiView2DJoints(frame_id)
        box_joints_map = {}
        
        if verbose:
            print(f'Getting bounding box crops for frame {frame_id}')
        
        for cam_name, camera_joints_dict in joints_dict.items():
            image_file = self.fetchVideoFrameFile(cam_name, frame_id)
            if not os.path.exists(image_file):
                continue
            
            im = cv2.imread(image_file)
            im_size = (im.shape[1], im.shape[0])  # width, height
            
            boxes0, prefixes, joints_list = [], [], []
            for ids, joints in camera_joints_dict.items():
                person_id = ids[-1]
                box = bp.cutBoxAroundJoints(im_size, joints, margin_ratio=box_joints_margin)
                boxes0.append(box)
                prefixes.append(f'{cam_name}_{person_id}')
                _, _, joints_vis = bp.countNumJointsInsideImage(im_size, joints)
                joints_list.append(joints_vis)
            
            # Filter boxes
            boxes, idxes1 = bp.removeBlockedBoxes(boxes0, box_ios_thold=box_ios_thold)
            boxes, idxes2 = bp.removeOutsideViewJoints(
                boxes, joints_inside_img_ratio=joints_inside_img_ratio)
            boxes, idxes3 = bp.removeSmallBoxes(boxes, box_size_thold)
            boxes, idxes4 = bp.removeOutsideViewBoxes(boxes, box_inside_img_ratio)
            
            # Crop and save boxes
            for i, (box, prefix) in enumerate(zip(boxes, prefixes)):
                x, y, w, h = box
                box_img = im[y:y+h, x:x+w]
                box_img_resized = cv2.resize(box_img, tuple(resize))
                
                box_filename = f'{prefix}_{x}_{y}_{w}_{h}.jpg'
                box_filepath = os.path.join(save_crop_dir, box_filename)
                cv2.imwrite(box_filepath, box_img_resized)
                
                # Store joint mapping
                joints_vis = joints_list[i]
                box_joints_map[box_filename] = np.concatenate([
                    joints_vis, np.ones((len(joints_vis), 1))
                ], axis=1).T
        
        # Save box-joint mapping
        pkl.dump(box_joints_map, open(box_joint_map_file, 'wb'))
        return save_crop_dir
    
    def getFrameReIDFeat(self, frame_id, reid_model=None, reid_log_file=None):
        """Get ReID features for bounding boxes (placeholder - can be implemented later)."""
        frame_box_dir = os.path.join(self.boxcrop_dir, f'frame{frame_id:08d}')
        reid_feat_file = os.path.join(frame_box_dir, 'box_reid_feat.pkl')
        
        # For now, return a dummy feature dict
        # In production, you would extract actual ReID features here
        if not os.path.exists(reid_feat_file):
            # Create dummy features (this is a placeholder)
            box_files = glob.glob(os.path.join(frame_box_dir, '*.jpg'))
            reid_feat = {box_file: np.random.rand(512) for box_file in box_files}
            pkl.dump(reid_feat, open(reid_feat_file, 'wb'))
        
        return pkl.load(open(reid_feat_file, 'rb'))
    
    def genPtsCorrepFromBoxClus(self, boxfile_clusters, verbose=True):
        """Generate 2D-2D point correspondences from box clusters."""
        if verbose:
            print("\nGetting 2D-2D correspondences:\n====================")
        
        # Get frame directory from first box file
        values = [v for v in boxfile_clusters.values() if v and len(v) > 0]
        if not values:
            raise ValueError("No valid box clusters found")
        
        frame_box_dir = os.path.join(self.boxcrop_dir, values[0][0].split('/')[-2])
        box_joints_map = pkl.load(open(os.path.join(frame_box_dir, 'box_joints_map.pkl'), 'rb'))
        
        persons = sorted(boxfile_clusters.keys())
        cam_names = self.cam_names
        
        corresp_dict = {}
        for cam_name in cam_names:
            corresp_dict[cam_name] = {'keypoints': [], 'box_files': []}
        
        for person in persons:
            if verbose:
                print(f'Getting point correspondences for person {person}')
            
            # Initialize with NaN
            for cam_name in cam_names:
                corresp_dict[cam_name]['keypoints'].append(
                    np.ones((len(self.joints_ids), 2)) * np.nan)
                corresp_dict[cam_name]['box_files'].append(None)
            
            # Fill in visible keypoints
            for boxfile in boxfile_clusters[person]:
                box_filename = os.path.basename(boxfile)
                # Extract camera name from filename (format: cam_name_person_id_x_y_w_h.jpg)
                cam_name = box_filename.split('_')[0]
                
                if box_filename in box_joints_map:
                    joints_2d = box_joints_map[box_filename][:2].T
                    if len(joints_2d) == len(self.joints_ids):
                        corresp_dict[cam_name]['keypoints'][-1] = joints_2d
                        corresp_dict[cam_name]['box_files'][-1] = boxfile
        
        # Concatenate all persons
        for cam_name in cam_names:
            if len(corresp_dict[cam_name]['keypoints']) > 0:
                corresp_dict[cam_name]['keypoints'] = np.concatenate(
                    corresp_dict[cam_name]['keypoints'], axis=0)
            else:
                corresp_dict[cam_name]['keypoints'] = np.array([]).reshape(0, 2)
        
        # Save point correspondences
        save_file = os.path.join(frame_box_dir, 'box_gen_pt_corr.pkl')
        pkl.dump(corresp_dict, open(save_file, 'wb'))
        n_persons = len(persons)
        
        if verbose:
            print(f'\nSaved to:\n"{save_file}"')
            print("====================\n")
        
        return corresp_dict, n_persons

