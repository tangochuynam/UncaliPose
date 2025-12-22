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
    data_dir/
    ├── videos/
    │   └── swing1/
    │       ├── front.mp4 (or front/)
    │       └── side.mp4 (or side/)
    └── keypoints/
        └── swing1/
            ├── front.json (or front/*.json)
            └── side.json (or side/*.json)
    """
    
    def __init__(self, data_dir, video_name='swing1', config=None, video_path=None):
        """
        Initialize the two-view dataset.
        
        Args:
            data_dir: Root directory containing videos and keypoints (if video_path is None)
            video_name: Name of the video folder (default: 'swing1', ignored if video_path is provided)
            config: Configuration dictionary
            video_path: Direct path to video folder (overrides data_dir/video_name)
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
        
        # Setup directories
        self.video_frame_dir = os.path.join(data_dir, 'processed', video_name, 'video_frame')
        self.pose2d_file_dir = os.path.join(data_dir, 'processed', video_name, 'pose2d_label')
        self.boxcrop_dir = os.path.join(data_dir, 'processed', video_name, 'box_crop')
        self.calibration_file = os.path.join(data_dir, 'processed', video_name, 'calibration.json')
        
        # Create directories if they don't exist
        os.makedirs(self.video_frame_dir, exist_ok=True)
        os.makedirs(self.pose2d_file_dir, exist_ok=True)
        os.makedirs(self.boxcrop_dir, exist_ok=True)
        
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
        Handles specific file naming patterns.
        """
        if not os.path.exists(self.video_dir):
            return
        
        video_files = sorted(glob.glob(os.path.join(self.video_dir, '*.mp4')))
        
        # Known mappings (user-specified)
        known_mappings = {
            '44CA7CF5-E031-44AA-97DC-8047B513EAB6': 'front',
            'FBD2D8A3-A7F7-4351-911A-56794228B7ED': 'side'
        }
        
        # Try to match video files to cameras
        for video_file in video_files:
            filename = os.path.basename(video_file)
            # Check for known patterns
            for pattern, cam_name in known_mappings.items():
                if pattern in filename:
                    self.video_file_map[cam_name] = video_file
                    break
        
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
        Uses actual video resolution if available.
        """
        cameras = {}
        default_focal = 1000.0  # Approximate focal length in pixels
        
        for i, cam_name in enumerate(self.cam_names):
            # Get actual resolution from video
            resolution = self._getVideoResolution(cam_name)
            
            if resolution is None:
                # Fallback to default if video not available
                # Try to infer from keypoints if available
                resolution = self._inferResolutionFromKeypoints(cam_name)
                if resolution is None:
                    resolution = (1920, 1080)  # Final fallback
                    print(f"Warning: Using default resolution {resolution} for {cam_name}")
                else:
                    print(f"Using inferred resolution {resolution} for {cam_name} from keypoints")
            else:
                print(f"Using actual video resolution {resolution} for {cam_name}")
            
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
        Try to infer image resolution from keypoint coordinates.
        Returns (width, height) or None.
        """
        try:
            # Check keypoint files
            keypoint_file = os.path.join(self.keypoints_dir, f'{cam_name}.json')
            if not os.path.exists(keypoint_file):
                keypoint_dir = os.path.join(self.keypoints_dir, cam_name)
                if os.path.isdir(keypoint_dir):
                    keypoint_files = sorted(glob.glob(os.path.join(keypoint_dir, '*.json')))
                    if len(keypoint_files) > 0:
                        keypoint_file = keypoint_files[0]
            
            if os.path.exists(keypoint_file):
                with open(keypoint_file, 'r') as f:
                    data = json.load(f)
                
                # Extract keypoints
                kpts = None
                if 'keypoints' in data:
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
                    max_x = np.nanmax(kpts[:, 0]) if kpts.shape[1] >= 1 else 0
                    max_y = np.nanmax(kpts[:, 1]) if kpts.shape[1] >= 2 else 0
                    
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
    
    def prepareDataFromKeypoints(self):
        """
        Prepare data structure from keypoint files.
        Converts keypoint files to the expected format.
        First tries to read from videos directory JSON files (keypoints_2d key),
        then falls back to keypoints directory.
        """
        print("Preparing data from keypoints...")
        
        # First, try to find keypoint files in videos directory (with keypoints_2d key)
        keypoint_files = {}
        keypoints_from_videos = {}
        
        # Map camera names to video file patterns
        video_patterns = {
            'front': '44CA7CF5-E031-44AA-97DC-8047B513EAB6',
            'side': 'FBD2D8A3-A7F7-4351-911A-56794228B7ED'
        }
        
        # Try to find JSON files in videos directory
        for cam_name in self.cam_names:
            found = False
            if cam_name in video_patterns:
                pattern = video_patterns[cam_name]
                # Look for JSON files matching the pattern
                video_json_files = glob.glob(os.path.join(self.video_dir, f'*{pattern}*.json'))
                if video_json_files:
                    # Use the first matching file
                    keypoint_files[cam_name] = [video_json_files[0]]
                    keypoints_from_videos[cam_name] = True
                    print(f"Found keypoints in video directory for {cam_name}: {os.path.basename(video_json_files[0])}")
                    found = True
            
            # If not found by pattern, try alternative matching (for video_path mode)
            if not found and self.keypoints_dir == self.video_dir:
                # In video_path mode, keypoints_dir == video_dir, so look for any JSON files
                all_json_files = glob.glob(os.path.join(self.keypoints_dir, '*.json'))
                # Try to match by filename pattern
                if cam_name == 'front':
                    matching_files = [f for f in all_json_files if '44CA7CF5' in os.path.basename(f)]
                    if matching_files:
                        keypoint_files[cam_name] = [matching_files[0]]
                        keypoints_from_videos[cam_name] = True
                        print(f"Found keypoints for {cam_name} by alternative pattern matching: {os.path.basename(matching_files[0])}")
                        found = True
                elif cam_name == 'side':
                    matching_files = [f for f in all_json_files if 'FBD2D8A3' in os.path.basename(f)]
                    if matching_files:
                        keypoint_files[cam_name] = [matching_files[0]]
                        keypoints_from_videos[cam_name] = True
                        print(f"Found keypoints for {cam_name} by alternative pattern matching: {os.path.basename(matching_files[0])}")
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
                    raise FileNotFoundError(f"Keypoint file not found for {cam_name}. Searched in: {self.video_dir} and {self.keypoints_dir}")
        
        # Determine number of frames
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
                
                # Save 2D pose label
                cam_pose_dir = os.path.join(self.pose2d_file_dir, cam_name)
                os.makedirs(cam_pose_dir, exist_ok=True)
                
                pose_file = os.path.join(cam_pose_dir, f'{frame_id:08d}.json')
                pose_data = {
                    'joint_type': 'Uplift',
                    'bodies': [{'id': 0, 'joints': kpts.tolist()}]
                }
                json.dump(pose_data, open(pose_file, 'w'))
        
        print(f"Prepared {n_frames} frames of keypoint data")
        return n_frames
    
    def getSingleFrameMultiView2DJoints(self, frame_id):
        """Extract 2D joints for a single frame from all views."""
        joints_dict = {}
        
        for cam_name in self.cam_names:
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

