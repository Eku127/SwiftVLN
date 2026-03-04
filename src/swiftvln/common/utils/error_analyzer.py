# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Error analysis utilities for VLN evaluation.

Implements post-hoc trajectory analysis to classify navigation failures
into different error types: STUCK, LOOPING, DEVIATION, STOP_FAILURE, EARLY_STOP.
"""

import numpy as np
from typing import List, Dict, Any, Optional


class TrajectoryRecorder:
    """
    Records agent positions at each navigation step for post-hoc analysis.
    """
    
    def __init__(self):
        self.positions: List[np.ndarray] = []
        self.path_lengths: List[float] = []  # Cumulative path length at each step
        
    def add_step(self, position: np.ndarray) -> None:
        """
        Add a position to the trajectory.
        
        Args:
            position: Agent position as numpy array [x, y, z]
        """
        self.positions.append(position.copy())
        
        # Calculate cumulative path length
        if len(self.positions) == 1:
            self.path_lengths.append(0.0)
        else:
            prev_pos = self.positions[-2]
            curr_pos = self.positions[-1]
            dist = np.linalg.norm(curr_pos - prev_pos)
            self.path_lengths.append(self.path_lengths[-1] + dist)
    
    def get_trajectory(self) -> List[np.ndarray]:
        """Get the recorded trajectory."""
        return self.positions
    
    def get_path_lengths(self) -> List[float]:
        """Get cumulative path lengths."""
        return self.path_lengths


class ErrorAnalyzer:
    """
    Analyzes navigation trajectories to identify error patterns.
    
    Error types:
    - STUCK: Agent gets stuck (minimal movement over consecutive steps)
    - LOOPING: Agent returns to previous locations (circular movement)
    - DEVIATION: Agent deviates from GT path without recovery
    - STOP_FAILURE: Agent reaches goal but fails to stop
    - EARLY_STOP: Agent stops too early without reaching goal
    """
    
    # Thresholds (hardcoded as per requirements)
    STUCK_WINDOW = 5  # Number of consecutive steps to check
    STUCK_THRESHOLD = 0.1  # Minimum movement (meters) to avoid STUCK label
    
    LOOPING_DISTANCE_THRESHOLD = 1.0  # Distance to same location (meters)
    LOOPING_PATH_THRESHOLD = 3.0  # Minimum path length between visits (meters)
    
    DEVIATION_THRESHOLD = 3.0  # Distance from GT path to be considered deviated (meters)
    RECOVERY_THRESHOLD = 2.0  # Distance to GT path to be considered recovered (meters)
    GOAL_THRESHOLD = 3.0  # Distance to goal to be considered successful (meters)
    
    def __init__(self):
        pass
    
    def analyze(
        self,
        trajectory: List[np.ndarray],
        path_lengths: List[float],
        gt_path: Optional[List[np.ndarray]],
        goal_position: np.ndarray,
        final_distance: float,
        oracle_success: bool
    ) -> Dict[str, Any]:
        """
        Analyze trajectory and generate error tags.
        
        Args:
            trajectory: List of agent positions
            path_lengths: Cumulative path lengths at each step
            gt_path: Ground truth reference path (optional, for DEVIATION analysis)
            goal_position: Goal position
            final_distance: Final distance to goal
            oracle_success: Whether agent ever reached goal during navigation
            
        Returns:
            Dictionary containing:
                - error_tags: List of error tag strings
                - had_deviation: Whether deviation occurred
                - deviation_recovered: Whether agent recovered from deviation
        """
        error_tags = []
        had_deviation = False
        deviation_recovered = False
        
        if len(trajectory) == 0:
            return {
                "error_tags": [],
                "had_deviation": False,
                "deviation_recovered": False,
            }
        
        # Check STUCK
        if self._check_stuck(trajectory):
            error_tags.append("STUCK")
        
        # Check LOOPING
        if self._check_looping(trajectory, path_lengths):
            error_tags.append("LOOPING")
        
        # Check DEVIATION (requires GT path)
        if gt_path is not None and len(gt_path) > 0:
            deviation_info = self._check_deviation(trajectory, gt_path, goal_position, final_distance)
            if deviation_info['occurred']:
                had_deviation = True
                deviation_recovered = deviation_info['recovered']
                
                if not deviation_info['recovered']:
                    error_tags.append("DEVIATION")
        
        # For FAILURE episodes only, check specific failure modes
        if final_distance > self.GOAL_THRESHOLD:
            # STOP_FAILURE: reached goal but didn't stop (equivalent to oracle_success)
            if oracle_success:
                error_tags.append("STOP_FAILURE")
            # EARLY_STOP: stopped early without reaching goal and without fatal deviation
            elif "DEVIATION" not in error_tags:
                error_tags.append("EARLY_STOP")
        
        return {
            "error_tags": error_tags,
            "had_deviation": had_deviation,
            "deviation_recovered": deviation_recovered,
        }
    
    def _check_stuck(self, trajectory: List[np.ndarray]) -> bool:
        """
        Check if agent got stuck (minimal movement over consecutive steps).
        
        Args:
            trajectory: List of positions
            
        Returns:
            True if STUCK pattern detected
        """
        if len(trajectory) < self.STUCK_WINDOW:
            return False
        
        # Slide window through trajectory
        for i in range(len(trajectory) - self.STUCK_WINDOW + 1):
            window_positions = trajectory[i:i + self.STUCK_WINDOW]
            
            # Calculate total displacement in window
            total_displacement = 0.0
            for j in range(len(window_positions) - 1):
                dist = np.linalg.norm(window_positions[j + 1] - window_positions[j])
                total_displacement += dist
            
            if total_displacement < self.STUCK_THRESHOLD:
                return True
        
        return False
    
    def _check_looping(self, trajectory: List[np.ndarray], path_lengths: List[float]) -> bool:
        """
        Check if agent exhibits looping behavior.
        
        Args:
            trajectory: List of positions
            path_lengths: Cumulative path lengths
            
        Returns:
            True if LOOPING pattern detected
        """
        if len(trajectory) < 2:
            return False
        
        # Check each position against all previous positions
        for i in range(1, len(trajectory)):
            curr_pos = trajectory[i]
            curr_path_len = path_lengths[i]
            
            for j in range(i):
                prev_pos = trajectory[j]
                prev_path_len = path_lengths[j]
                
                # Check distance condition: close to previous location
                spatial_distance = np.linalg.norm(curr_pos - prev_pos)
                
                # Check path condition: traveled significant distance between visits
                path_distance = curr_path_len - prev_path_len
                
                if (spatial_distance < self.LOOPING_DISTANCE_THRESHOLD and 
                    path_distance > self.LOOPING_PATH_THRESHOLD):
                    return True
        
        return False
    
    def _check_deviation(
        self,
        trajectory: List[np.ndarray],
        gt_path: List[np.ndarray],
        goal_position: np.ndarray,
        final_distance: float
    ) -> Dict[str, bool]:
        """
        Check if agent deviated from GT path and whether it recovered.
        
        Recovery is defined as:
        - Returning to within RECOVERY_THRESHOLD of GT path, OR
        - Reaching within GOAL_THRESHOLD of goal
        
        Args:
            trajectory: List of agent positions
            gt_path: Ground truth reference path
            goal_position: Goal position
            final_distance: Final distance to goal
            
        Returns:
            Dictionary with 'occurred' (deviation happened) and 'recovered' (agent recovered)
        """
        if len(gt_path) == 0:
            return {'occurred': False, 'recovered': False}
        
        deviation_occurred = False
        recovery_occurred = False
        
        for i, pos in enumerate(trajectory):
            # Calculate minimum distance to GT path
            min_dist_to_path = min(np.linalg.norm(pos - gt_pos) for gt_pos in gt_path)
            
            # Check if deviated
            if min_dist_to_path > self.DEVIATION_THRESHOLD:
                deviation_occurred = True
            
            # After deviation, check for recovery
            if deviation_occurred:
                # Recovery condition 1: Return to GT path
                if min_dist_to_path <= self.RECOVERY_THRESHOLD:
                    recovery_occurred = True
                
                # Recovery condition 2: Reach goal (check distance to goal)
                dist_to_goal = np.linalg.norm(pos - goal_position)
                if dist_to_goal <= self.GOAL_THRESHOLD:
                    recovery_occurred = True
        
        # Also check final success as recovery indicator
        if deviation_occurred and final_distance <= self.GOAL_THRESHOLD:
            recovery_occurred = True
        
        return {
            'occurred': deviation_occurred,
            'recovered': recovery_occurred if deviation_occurred else False
        }
