#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Logging utility for 3D pose estimation pipeline.
"""

import os
import logging
from datetime import datetime
from pathlib import Path


class PipelineLogger:
    """Logger for 3D pose estimation pipeline."""
    
    def __init__(self, log_dir, video_name, log_level=logging.INFO):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save log files
            video_name: Name of the video being processed
            log_level: Logging level (default: INFO)
        """
        self.log_dir = Path(log_dir)
        self.video_name = video_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"pipeline_{video_name}_{timestamp}.log"
        
        # Setup logger
        self.logger = logging.getLogger(f"Pipeline_{video_name}")
        self.logger.setLevel(log_level)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # File handler
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler (with simpler format)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter('%(levelname)-8s | %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        self.log_file = log_file
        self.logger.info(f"Logger initialized. Log file: {log_file}")
    
    def info(self, message):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message):
        """Log debug message."""
        self.logger.debug(message)
    
    def log_rmse(self, frame_id, rmse_dict, total_rmse):
        """Log RMSE for a frame."""
        front_rmse = rmse_dict.get('front', 'N/A')
        side_rmse = rmse_dict.get('side', 'N/A')
        self.logger.info(
            f"Frame {frame_id:4d} | RMSE - Front: {front_rmse:6.2f}px, "
            f"Side: {side_rmse:6.2f}px, Total: {total_rmse:6.2f}px"
        )
    
    def log_high_rmse(self, frame_id, rmse_dict, threshold=10.0):
        """Log warning for high RMSE."""
        for cam_name, rmse in rmse_dict.items():
            if rmse is not None and rmse > threshold:
                self.logger.warning(
                    f"Frame {frame_id:4d} | High RMSE on {cam_name}: {rmse:.2f}px "
                    f"(threshold: {threshold:.1f}px)"
                )
    
    def log_camera_pose(self, frame_id, pose_info):
        """Log camera pose information."""
        self.logger.info(f"Frame {frame_id:4d} | Camera pose: {pose_info}")
    
    def get_log_file(self):
        """Get path to log file."""
        return str(self.log_file)
    
    def read_log(self, lines=None):
        """
        Read log file.
        
        Args:
            lines: Number of lines to read (None = all)
        
        Returns:
            List of log lines
        """
        if not self.log_file.exists():
            return []
        
        with open(self.log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        
        if lines is None:
            return all_lines
        else:
            return all_lines[-lines:]


def read_log_file(log_file_path, lines=None, filter_level=None):
    """
    Read a log file.
    
    Args:
        log_file_path: Path to log file
        lines: Number of lines to read (None = all)
        filter_level: Filter by log level (e.g., 'WARNING', 'ERROR')
    
    Returns:
        List of log lines
    """
    log_file = Path(log_file_path)
    if not log_file.exists():
        return []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    if filter_level:
        all_lines = [line for line in all_lines if filter_level in line]
    
    if lines is None:
        return all_lines
    else:
        return all_lines[-lines:]


def find_latest_log(log_dir, video_name=None):
    """
    Find the latest log file.
    
    Args:
        log_dir: Directory containing log files
        video_name: Optional video name filter
    
    Returns:
        Path to latest log file or None
    """
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return None
    
    pattern = f"pipeline_{video_name}_*.log" if video_name else "pipeline_*.log"
    log_files = sorted(log_dir.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    
    return log_files[0] if log_files else None

