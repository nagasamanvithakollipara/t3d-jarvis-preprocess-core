#!/usr/bin/env python3
"""
Object Tracker Module

Integrates SORT (Simple Online and Realtime Tracking) algorithm with SAM2
for robust object tracking across video frames.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
import os
from datetime import datetime


class TrackState(Enum):
    """Track state enumeration"""
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


@dataclass
class TrackedObject:
    """Represents a tracked object across frames"""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    mask: np.ndarray  # Binary mask
    confidence: float
    state: TrackState
    frame_history: List[Dict]  # History of detections
    first_frame: int
    last_frame: int
    total_frames: int
    class_name: str = "object"
    
    def update(self, bbox: np.ndarray, mask: np.ndarray, confidence: float, frame_idx: int):
        """Update object with new detection"""
        self.bbox = bbox
        self.mask = mask
        self.confidence = confidence
        self.last_frame = frame_idx
        self.total_frames += 1
        
        # Add to history
        self.frame_history.append({
            'frame_idx': frame_idx,
            'bbox': bbox.copy(),
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 100 frames in history to prevent memory issues
        if len(self.frame_history) > 100:
            self.frame_history = self.frame_history[-100:]


class SORTTracker:
    """
    SORT (Simple Online and Realtime Tracking) implementation
    Adapted for integration with SAM2 masks
    """
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[TrackedObject] = []
        self.next_id = 0
        self.frame_count = 0
        
    def update(self, detections: List[Dict], frame_idx: int) -> List[TrackedObject]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of dicts with 'bbox', 'mask', 'confidence', 'class_name'
            frame_idx: Current frame index
            
        Returns:
            List of active tracked objects
        """
        self.frame_count += 1
        
        # Extract bboxes from detections
        bboxes = np.array([det['bbox'] for det in detections])
        masks = [det['mask'] for det in detections]
        confidences = [det['confidence'] for det in detections]
        class_names = [det.get('class_name', 'object') for det in detections]
        
        if len(bboxes) == 0:
            # No detections, update existing tracks
            self._update_existing_tracks(frame_idx)
            return self._get_active_tracks()
        
        # Associate detections with existing tracks
        if len(self.trackers) > 0:
            # Calculate IoU between existing tracks and new detections
            iou_matrix = self._calculate_iou_matrix(bboxes)
            matched_indices = self._hungarian_assignment(iou_matrix)
            
            # Update matched tracks
            matched_detections = set()
            matched_tracks = set()
            
            for track_idx, det_idx in matched_indices:
                if iou_matrix[track_idx, det_idx] >= self.iou_threshold:
                    # Update existing track
                    self.trackers[track_idx].update(
                        bboxes[det_idx], 
                        masks[det_idx], 
                        confidences[det_idx], 
                        frame_idx
                    )
                    self.trackers[track_idx].state = TrackState.TRACKED
                    matched_detections.add(det_idx)
                    matched_tracks.add(track_idx)
            
            # Create new tracks for unmatched detections
            for det_idx in range(len(detections)):
                if det_idx not in matched_detections:
                    self._create_new_track(
                        bboxes[det_idx], 
                        masks[det_idx], 
                        confidences[det_idx], 
                        class_names[det_idx], 
                        frame_idx
                    )
            
            # Mark unmatched tracks as lost
            for track_idx in range(len(self.trackers)):
                if track_idx not in matched_tracks:
                    self.trackers[track_idx].state = TrackState.LOST
        else:
            # First frame, create tracks for all detections
            for i in range(len(detections)):
                self._create_new_track(
                    bboxes[i], 
                    masks[i], 
                    confidences[i], 
                    class_names[i], 
                    frame_idx
                )
        
        # Update existing tracks and remove old ones
        self._update_existing_tracks(frame_idx)
        
        return self._get_active_tracks()
    
    def _create_new_track(self, bbox: np.ndarray, mask: np.ndarray, 
                          confidence: float, class_name: str, frame_idx: int):
        """Create a new track"""
        track = TrackedObject(
            track_id=self.next_id,
            bbox=bbox,
            mask=mask,
            confidence=confidence,
            state=TrackState.NEW,
            frame_history=[],
            first_frame=frame_idx,
            last_frame=frame_idx,
            total_frames=1,
            class_name=class_name
        )
        track.update(bbox, mask, confidence, frame_idx)
        self.trackers.append(track)
        self.next_id += 1
    
    def _update_existing_tracks(self, frame_idx: int):
        """Update existing tracks and remove old ones"""
        active_tracks = []
        for track in self.trackers:
            if track.state == TrackState.LOST:
                # Check if track should be removed
                if frame_idx - track.last_frame > self.max_age:
                    track.state = TrackState.REMOVED
                else:
                    active_tracks.append(track)
            elif track.state == TrackState.TRACKED:
                active_tracks.append(track)
        
        self.trackers = active_tracks
    
    def _get_active_tracks(self) -> List[TrackedObject]:
        """Get list of active tracks"""
        return [track for track in self.trackers if track.state != TrackState.REMOVED]
    
    def _calculate_iou_matrix(self, bboxes: np.ndarray) -> np.ndarray:
        """Calculate IoU matrix between existing tracks and new detections"""
        if len(self.trackers) == 0:
            return np.array([])
        
        iou_matrix = np.zeros((len(self.trackers), len(bboxes)))
        
        for i, track in enumerate(self.trackers):
            for j, bbox in enumerate(bboxes):
                iou_matrix[i, j] = self._calculate_iou(track.bbox, bbox)
        
        return iou_matrix
    
    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate IoU between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _hungarian_assignment(self, cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Simple Hungarian algorithm for assignment problem"""
        if cost_matrix.size == 0:
            return []
        
        # For simplicity, use greedy assignment
        # In production, consider using scipy.optimize.linear_sum_assignment
        assignments = []
        used_rows = set()
        used_cols = set()
        
        # Sort by cost (descending for IoU)
        indices = np.unravel_index(np.argsort(-cost_matrix, axis=None), cost_matrix.shape)
        
        for row, col in zip(indices[0], indices[1]):
            if row not in used_rows and col not in used_cols:
                assignments.append((row, col))
                used_rows.add(row)
                used_cols.add(col)
        
        return assignments


class TrackingVisualizer:
    """Visualization utilities for tracked objects"""
    
    def __init__(self):
        self.colors = self._generate_colors()
    
    def _generate_colors(self, num_colors: int = 100) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for tracking visualization"""
        np.random.seed(42)
        colors = []
        for i in range(num_colors):
            color = (0, 0, 0)  # Black color

            colors.append(color)
        return colors
    
    def draw_tracks(self, frame: np.ndarray, tracks: List[TrackedObject], 
                   frame_idx: int) -> np.ndarray:
        """Draw tracking information on frame"""
        result = frame.copy()
        
        for track in tracks:
            if track.state == TrackState.REMOVED:
                continue
                
            # Get color for this track
            color = self.colors[track.track_id % len(self.colors)]
            
            # Draw bounding box
            x1, y1, x2, y2 = track.bbox.astype(int)
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID and info
            label = f"ID: {track.track_id}"
            if track.class_name != "object":
                label += f" ({track.class_name})"
            
            # Calculate text position
            text_x = x1
            text_y = y1 - 10 if y1 > 20 else y1 + 20
            
            # Draw text background
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(result, (text_x, text_y - text_height - 5), 
                         (text_x + text_width, text_y + 5), color, -1)
            
            # Draw text
            cv2.putText(result, label, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw confidence score
            conf_label = f"Conf: {track.confidence:.2f}"
            conf_y = text_y + text_height + 20
            
            # Draw confidence background
            (conf_width, conf_height), _ = cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (text_x, conf_y - conf_height - 3), 
                         (text_x + conf_width, conf_y + 3), color, -1)
            
            # Draw confidence text
            cv2.putText(result, conf_label, (text_x, conf_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw track trail (last 10 positions)
            if len(track.frame_history) > 1:
                trail_points = []
                for hist in track.frame_history[-10:]:
                    center_x = int((hist['bbox'][0] + hist['bbox'][2]) / 2)
                    center_y = int((hist['bbox'][1] + hist['bbox'][3]) / 2)
                    trail_points.append((center_x, center_y))
                
                if len(trail_points) > 1:
                    for i in range(1, len(trail_points)):
                        alpha = i / len(trail_points)
                        trail_color = tuple(int(c * alpha) for c in color)
                        cv2.line(result, trail_points[i-1], trail_points[i], trail_color, 2)
        
        return result
    
    def create_tracking_summary(self, tracks: List[TrackedObject], 
                              total_frames: int) -> Dict:
        """Create summary statistics for tracking"""
        if not tracks:
            return {}
        
        active_tracks = [t for t in tracks if t.state != TrackState.REMOVED]
        
        summary = {
            'total_tracks': len(tracks),
            'active_tracks': len(active_tracks),
            'total_frames': total_frames,
            'track_statistics': []
        }
        
        for track in tracks:
            track_stats = {
                'track_id': track.track_id,
                'class_name': track.class_name,
                'first_frame': track.first_frame,
                'last_frame': track.last_frame,
                'total_frames': track.total_frames,
                'average_confidence': np.mean([h['confidence'] for h in track.frame_history]),
                'state': track.state.name,
                'bbox_history': [h['bbox'].tolist() for h in track.frame_history]
            }
            summary['track_statistics'].append(track_stats)
        
        return summary


class TrackingManager:
    """Main tracking manager that integrates with SAM2 pipeline"""
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.tracker = SORTTracker(max_age, min_hits, iou_threshold)
        self.visualizer = TrackingVisualizer()
        self.tracking_history: List[List[TrackedObject]] = []
        self.frame_count = 0
        
    def update_tracking(self, frame_idx: int, sam2_results: List[Dict]) -> List[TrackedObject]:
        """
        Update tracking with SAM2 results
        
        Args:
            frame_idx: Current frame index
            sam2_results: List of SAM2 detection results with masks and bboxes
            
        Returns:
            List of tracked objects
        """
        # Convert SAM2 results to tracking format
        detections = []
        for result in sam2_results:
            # Extract bbox from mask
            mask = result['mask']
            bbox = self._mask_to_bbox(mask)
            confidence = result.get('confidence', 0.5)
            class_name = result.get('class_name', 'object')
            
            detections.append({
                'bbox': bbox,
                'mask': mask,
                'confidence': confidence,
                'class_name': class_name
            })
        
        # Update tracker
        active_tracks = self.tracker.update(detections, frame_idx)
        
        # Store tracking history
        self.tracking_history.append(active_tracks.copy())
        self.frame_count += 1
        
        return active_tracks
    
    def _mask_to_bbox(self, mask: np.ndarray) -> np.ndarray:
        """Convert binary mask to bounding box"""
        if not np.any(mask):
            return np.array([0, 0, 0, 0])
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        
        return np.array([x1, y1, x2, y2])
    
    def get_tracking_summary(self) -> Dict:
        """Get comprehensive tracking summary"""
        return self.visualizer.create_tracking_summary(
            self.tracker.trackers, 
            self.frame_count
        )
    
    def save_tracking_data(self, output_path: str):
        """Save tracking data to JSON file"""
        summary = self.get_tracking_summary()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Tracking data saved to: {output_path}")
    
    def visualize_frame(self, frame: np.ndarray, tracks: List[TrackedObject], 
                       frame_idx: int) -> np.ndarray:
        """Visualize tracking on a single frame"""
        return self.visualizer.draw_tracks(frame, tracks, frame_idx)



