# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Video processing utilities for VLN evaluation.
"""

import os
import zipfile
from typing import List, Any


def compress_videos(output_dir: str, all_episodes: List[Any], chunk_size: int = 400) -> None:
    """
    Compress video files into zip files and delete source files.
    
    Groups videos into chunks (e.g., 400 videos per zip) to avoid creating
    huge zip files. Naming convention: 0-400.zip, 401-800.zip, etc.
    
    Args:
        output_dir: Output directory containing 'videos' subdirectory
        all_episodes: List of episodes (objects with episode_id attribute) or 
                     list of result dicts with 'episode_id' key
        chunk_size: Number of videos per zip file (default: 400)
    """
    video_dir = os.path.join(output_dir, "videos")
    if not os.path.exists(video_dir):
        return
    
    print("Compressing videos...")
    
    def get_episode_id(episode):
        """Get episode_id from either an object or dict."""
        if isinstance(episode, dict):
            return str(episode.get('episode_id', ''))
        return str(getattr(episode, 'episode_id', ''))
    
    for i in range(0, len(all_episodes), chunk_size):
        start_idx = i
        end_idx = min(i + chunk_size, len(all_episodes))
        
        # Naming logic: 0-400, 401-800, etc.
        if start_idx == 0:
            zip_name = f"0-{end_idx}.zip"
        else:
            zip_name = f"{start_idx+1}-{end_idx}.zip"
        
        zip_path = os.path.join(video_dir, zip_name)
        
        added_count = 0
        # Get all mp4 files in the directory once to speed up lookup
        all_video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for j in range(start_idx, end_idx):
                ep_id = get_episode_id(all_episodes[j])
                # Match files like "{ep_id}.mp4" or "{ep_id}_{success}.mp4"
                matched_files = [f for f in all_video_files 
                               if f == f"{ep_id}.mp4" or f.startswith(f"{ep_id}_")]
                
                for video_file in matched_files:
                    video_path = os.path.join(video_dir, video_file)
                    if os.path.exists(video_path):
                        zf.write(video_path, video_file)
                        os.remove(video_path)
                        added_count += 1
        
        if added_count > 0:
            print(f"Created {zip_name} with {added_count} videos.")
        else:
            if os.path.exists(zip_path):
                os.remove(zip_path)
    
    print("Video compression completed.")
