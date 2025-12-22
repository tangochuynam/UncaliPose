#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Check the format of your 360-frame keypoint JSON files.
This will help determine the correct format to process.
"""

import json
import os
import sys

def check_json_format(file_path):
    """Check the format of a JSON keypoint file."""
    if not os.path.exists(file_path):
        print(f"✗ File not found: {file_path}")
        return None
    
    print(f"\nChecking: {file_path}")
    print("-" * 70)
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        
        # Check for 'keypoints' key
        if 'keypoints' in data:
            kpts = data['keypoints']
            print(f"\n'keypoints' found:")
            print(f"  Type: {type(kpts)}")
            print(f"  Length: {len(kpts)}")
            
            if len(kpts) > 0:
                print(f"  First element type: {type(kpts[0])}")
                
                if isinstance(kpts[0], list):
                    print(f"  First element length: {len(kpts[0])}")
                    
                    if len(kpts[0]) > 0:
                        print(f"  First element[0] type: {type(kpts[0][0])}")
                        
                        if isinstance(kpts[0][0], list):
                            # This is: [[[x,y],...], [[x,y],...], ...] - List of frames!
                            print(f"  First element[0] length: {len(kpts[0][0])}")
                            print(f"\n  ✓ FORMAT: List of {len(kpts)} frames")
                            print(f"     Each frame has {len(kpts[0])} keypoints")
                            print(f"     Each keypoint is [x, y]")
                            return 'frames_list', len(kpts)
                        elif isinstance(kpts[0][0], (int, float)):
                            # This is: [[x,y], [x,y], ...] - Single frame
                            print(f"\n  ✓ FORMAT: Single frame with {len(kpts)} keypoints")
                            return 'single_frame', 1
                
                elif isinstance(kpts[0], (int, float)):
                    # Flat list: [x1, y1, x2, y2, ...]
                    print(f"\n  ✓ FORMAT: Flat list (single frame)")
                    return 'flat_list', 1
        
        # Check for 'frames' key
        if 'frames' in data:
            frames = data['frames']
            print(f"\n'frames' found:")
            print(f"  Type: {type(frames)}")
            print(f"  Length: {len(frames)}")
            print(f"\n  ✓ FORMAT: Frames array with {len(frames)} frames")
            return 'frames_array', len(frames)
    
    elif isinstance(data, list):
        print(f"Length: {len(data)}")
        if len(data) > 0:
            print(f"First element type: {type(data[0])}")
            print(f"\n  ✓ FORMAT: Direct list (possibly {len(data)} frames)")
            return 'direct_list', len(data)
    
    return None, 0


def main():
    data_dir = '/mnt/server-2TB/workspace/tnhnam/UncaliPose'
    
    # Check different possible locations
    files_to_check = [
        ('keypoints/swing1/front.json', 'Front camera'),
        ('keypoints/swing1/side.json', 'Side camera'),
        ('keypoints/swing1/front/front.json', 'Front camera (subdir)'),
        ('keypoints/swing1/side/side.json', 'Side camera (subdir)'),
    ]
    
    print("=" * 70)
    print("Checking 360-frame keypoint JSON format")
    print("=" * 70)
    
    results = {}
    for rel_path, description in files_to_check:
        full_path = os.path.join(data_dir, rel_path)
        if os.path.exists(full_path):
            print(f"\n{description}:")
            format_type, n_frames = check_json_format(full_path)
            results[description] = (format_type, n_frames, full_path)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if results:
        for desc, (fmt, n, path) in results.items():
            print(f"{desc}:")
            print(f"  Format: {fmt}")
            print(f"  Frames detected: {n}")
            print(f"  Path: {path}")
            print()
        
        # Check if we found 360 frames
        total_frames = max([n for _, n, _ in results.values()] + [0])
        if total_frames == 360:
            print("✓ Found 360 frames! Ready to process.")
        elif total_frames > 1:
            print(f"⚠ Found {total_frames} frames (expected 360)")
            print("  Make sure your JSON files contain all 360 frames.")
        else:
            print("⚠ Only found single frame format.")
            print("  For 360 frames, your JSON should be:")
            print("    {'keypoints': [[[x,y],...], [[x,y],...], ..., [[x,y],...]]}")
            print("    (360 frames, each with 16 keypoints)")
    else:
        print("✗ No keypoint JSON files found!")
        print("\nExpected locations:")
        for rel_path, _ in files_to_check:
            print(f"  {os.path.join(data_dir, rel_path)}")


if __name__ == '__main__':
    main()

