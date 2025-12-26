# 3D Pose Estimation Pipeline - User Guide

## Overview

This pipeline processes synchronized multi-view videos with 2D keypoints to generate 3D human pose estimates. It supports two input modes: traditional directory structure and direct video path.

## Prerequisites

1. **Python Environment**: Ensure you have a Python environment with required dependencies installed
   - Required packages: numpy, opencv-python, scipy, matplotlib, pyyaml
   - See `requirements.txt` or setup instructions for details

2. **Input Data Requirements**:
   - 2 synchronized video files (front and side camera views)
   - 2 JSON files containing 2D keypoints (one per camera view)
   - Videos and keypoints must be synchronized (same number of frames)
   - **File naming**: Video and JSON files **MUST** contain 'front' or 'side' in their filename (case-insensitive)
     - Examples: `front_video.mp4`, `side_keypoints.json`, `my_front_camera.mp4`
     - The pipeline will raise an error if it cannot identify which file is front/side

## Input Data Format

### Option 1: Direct Video Path (Recommended)

**Directory Structure:**
```
/path/to/videos/swing1/
├── front_video.mp4 (must contain 'front' in filename)
├── side_video.mp4 (must contain 'side' in filename)
├── front_keypoints.json (must contain 'front' in filename)
└── side_keypoints.json (must contain 'side' in filename)
```

**Important**: File names must contain 'front' or 'side' (case-insensitive) to be automatically identified.

**Keypoint JSON Format:**
```json
{
  "keypoints_2d": [
    [[x1, y1], [x2, y2], ..., [x16, y16]],  // Frame 0
    [[x1, y1], [x2, y2], ..., [x16, y16]],  // Frame 1
    ...
  ]
}
```

### Option 2: Traditional Directory Structure

**Directory Structure:**
```
/path/to/data/
├── videos/
│   └── swing1/
│       ├── front.mp4
│       └── side.mp4
└── keypoints/
    └── swing1/
        ├── front.json
        └── side.json
```

## Running the Pipeline

### Basic Usage

#### Method 1: Direct Video Path (Easiest)

```bash
python pipeline_3d_pose.py --video_path /path/to/videos/swing1
```

**Note**: The pipeline automatically loads `config/pipeline_config.yml` if it exists.
You don't need to specify `--config_file` unless you want to use a different config file.

#### Method 2: Traditional Structure

```bash
python pipeline_3d_pose.py \
    --data_dir /path/to/data \
    --video_name swing1
```

### Command-Line Arguments

#### Required Arguments (choose one):

- `--video_path PATH`: Direct path to folder containing videos and keypoints JSON files
  - Example: `--video_path /path/to/videos/swing1`
  - **OR**
- `--data_dir PATH`: Root directory containing `videos/` and `keypoints/` folders
  - Example: `--data_dir /path/to/data`
- `--video_name NAME`: Name of the video folder (required if using `--data_dir`)
  - Example: `--video_name swing1`

#### Optional Arguments:

- `--world_cam_id ID`: World camera ID (0=front, 1=side, default: 1)
  ```bash
  --world_cam_id 1  # Use side camera as world reference
  ```

- `--skip_processing`: Skip 3D processing, only create visualization
  ```bash
  --skip_processing  # Useful if you already have processed results
  ```

- `--skip_visualization`: Skip visualization, only process 3D keypoints
  ```bash
  --skip_visualization  # Process only, don't create videos
  ```

- `--fps FPS`: Output video FPS (default: 30)
  ```bash
  --fps 30
  ```

- `--no_save_pose`: Don't save/load camera pose (estimate per video)
  ```bash
  --no_save_pose  # Force re-estimation of camera pose
  ```

- `--config_file PATH`: Path to YAML config file (optional)
  ```bash
  --config_file config/custom_config.yml  # Override default config
  ```
  **Note**: By default, the pipeline automatically loads `config/pipeline_config.yml` if it exists.
  If the file doesn't exist, it uses default configuration. Use `--config_file` to specify a different config file.

## Examples

### Example 1: Basic Run with Direct Path

```bash
python pipeline_3d_pose.py --video_path /path/to/videos/swing1
```

### Example 2: Process Multiple Videos

```bash
# Process swing1
python pipeline_3d_pose.py --video_path /path/to/videos/swing1

# Process swing2
python pipeline_3d_pose.py --video_path /path/to/videos/swing2

# Process swing3
python pipeline_3d_pose.py --video_path /path/to/videos/swing3
```

### Example 3: Re-create Visualization Only

If you've already processed the 3D keypoints and just want to update the visualization:

```bash
python pipeline_3d_pose.py \
    --video_path /path/to/videos/swing1 \
    --skip_processing
```

### Example 4: Process Only (No Visualization)

If you only need the 3D keypoints data:

```bash
python pipeline_3d_pose.py \
    --video_path /path/to/videos/swing1 \
    --skip_visualization
```

### Example 5: Force Re-estimation of Camera Pose

If you want to re-estimate the camera pose (e.g., after changing parameters):

```bash
python pipeline_3d_pose.py \
    --video_path /path/to/videos/swing1 \
    --no_save_pose
```

### Example 6: Using Custom Configuration File

By default, the pipeline automatically loads `config/pipeline_config.yml` if it exists.
To use a different config file:

```bash
python pipeline_3d_pose.py \
    --video_path /path/to/videos/swing1 \
    --config_file config/custom_config.yml
```

**Configuration Behavior**:
- If `config/pipeline_config.yml` exists → automatically loads it (no flag needed)
- If `config/pipeline_config.yml` doesn't exist → uses default configuration
- Use `--config_file` to override and specify a different config file

## Output Files

After running the pipeline, you'll find the following outputs:

### Directory Structure:
```
/path/to/data/processed/swing1/
├── results/
│   ├── frame_00000000_3d_keypoints.json  # Individual frame files
│   ├── frame_00000001_3d_keypoints.json
│   ├── ...
│   ├── all_frames_3d_keypoints.json      # Combined JSON (all frames)
│   ├── summary.json                      # Processing summary
│   ├── 3d_keypoints_animation.mp4        # 3D visualization video
│   └── debug_2d_keypoints_comparison.mp4 # Debug visualization (2D comparison)
├── camera_pose.json                      # Estimated camera pose
└── logs/
    └── pipeline_swing1_TIMESTAMP.log    # Detailed log file
```

### Output Files Description:

1. **Individual Frame JSON Files** (`frame_*_3d_keypoints.json`):
   - Contains 3D keypoints, height, RMSE, and metadata for each frame
   - Format: One file per frame

2. **Combined JSON File** (`all_frames_3d_keypoints.json`):
   - Contains all frames in a single JSON file
   - Structure:
     ```json
     {
       "n_frames": 360,
       "frames": [
         { "frame_id": 0, "keypoints_by_person": {...}, ... },
         { "frame_id": 1, "keypoints_by_person": {...}, ... },
         ...
       ]
     }
     ```

3. **Summary JSON** (`summary.json`):
   - Processing statistics
   - Camera pose information
   - Height statistics (if available)

4. **3D Visualization Video** (`3d_keypoints_animation.mp4`):
   - Front view of 3D skeleton animation
   - Coordinate system: X=horizontal, Y=vertical, Z=depth
   - No height display in title

5. **Debug Visualization Video** (`debug_2d_keypoints_comparison.mp4`):
   - 2x2 grid showing:
     - Top row: Front camera (Original 2D | Reprojected 2D)
     - Bottom row: Side camera (Original 2D | Reprojected 2D)
   - Color coding: Green (original), Red (reprojected), Blue (error lines)
   - RMSE displayed per frame

6. **Camera Pose JSON** (`camera_pose.json`):
   - Estimated camera pose (rotation and translation)
   - Frame ID used for estimation
   - RMSE value
   - Estimation time

7. **Log File** (`pipeline_swing1_TIMESTAMP.log`):
   - Detailed processing log
   - RMSE values for each frame
   - Warnings and errors
   - Camera pose estimation timing

## Understanding the Output

### 3D Keypoints Format

Each frame JSON contains:
```json
{
  "frame_id": 0,
  "keypoints_by_person": {
    "person_0": {
      "joints": [[x, y, z], [x, y, z], ...],  // 16 joints in meters
      "height_m": 1.085,
      "keypoints_valid": true,
      "reprojection_rmse": {
        "per_camera_rmse": {
          "front": 1.46,
          "side": 1.88
        },
        "total_rmse_pixels": 1.68
      }
    }
  }
}
```

### Joint Order (16 joints - Uplift Order):
0. right_ankle
1. right_knee
2. right_hip
3. left_hip
4. left_knee
5. left_ankle
6. center_hip
7. center_shoulder
8. neck
9. head
10. right_wrist
11. right_elbow
12. right_shoulder
13. left_shoulder
14. left_elbow
15. left_wrist

### Coordinate System

- **X-axis**: Horizontal (left-right)
- **Y-axis**: Vertical (up-down, upward positive)
- **Z-axis**: Depth (forward-backward)

## Troubleshooting

### Issue: "Keypoint file not found"

**Solution**: 
- Check that JSON files are in the correct location
- For direct path mode, ensure JSON files are in the same directory as videos
- Verify JSON files contain `keypoints_2d` key with array of frames

### Issue: "Video file not found" or "Cannot identify front/side camera files"

**Solution**:
- **CRITICAL**: Video and JSON file names **MUST** contain 'front' or 'side' (case-insensitive)
- For front camera: Filename must contain 'front' (e.g., `front_video.mp4`, `my_front_camera.mp4`, `front_keypoints.json`)
- For side camera: Filename must contain 'side' (e.g., `side_video.mp4`, `my_side_camera.mp4`, `side_keypoints.json`)
- If you get an error "Cannot identify front/side camera files", rename your files to include 'front' or 'side'
- Examples of **INVALID** names: `camera1.mp4`, `video1.json`, `keypoints.json` (no front/side identifier)
- Examples of **VALID** names: `front_camera.mp4`, `side_video.mp4`, `my_front_keypoints.json`

### Issue: High RMSE (> 4 pixels)

**Solution**:
- Check that 2D keypoints are accurate
- Verify videos are synchronized
- Try re-estimating camera pose with `--no_save_pose`
- Check log file for specific frame issues

### Issue: "Results directory not found"

**Solution**:
- Run without `--skip_processing` first to generate results
- Or check that processed results exist in `processed/video_name/results/`

## Performance Tips

1. **Camera Pose Reuse**: The pipeline saves camera pose after first estimation. Subsequent runs with the same video will reuse it (unless `--no_save_pose` is used).

2. **Skip Processing**: If you only need to update visualization, use `--skip_processing` to save time.

3. **Skip Visualization**: If you only need 3D keypoints data, use `--skip_visualization` to save time.

4. **Batch Processing**: Process multiple videos in a loop:
   ```bash
   for video in swing1 swing2 swing3; do
       python pipeline_3d_pose.py --video_path /path/to/videos/$video
   done
   ```

## Quick Reference

### Most Common Commands:

```bash
# Full pipeline run
python pipeline_3d_pose.py --video_path /path/to/videos/swing1

# Re-create visualization only
python pipeline_3d_pose.py --video_path /path/to/videos/swing1 --skip_processing

# Process only (no visualization)
python pipeline_3d_pose.py --video_path /path/to/videos/swing1 --skip_visualization

# Force re-estimate camera pose
python pipeline_3d_pose.py --video_path /path/to/videos/swing1 --no_save_pose
```

## Support

For issues or questions:
1. Check the log file in `processed/video_name/logs/`
2. Review the summary JSON in `processed/video_name/results/summary.json`
3. Check RMSE values in the debug visualization video

