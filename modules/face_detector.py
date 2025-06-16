import json
import logging
from .gaze_tracking import GazeTracker

class FaceDetector:
    """
    Class for gaze direction detection
    """
    def __init__(self, mirror_image=True):
        self.mirror_image = mirror_image
        self.gaze_tracker = GazeTracker(mirror_image=mirror_image)

    @property
    def pupils_located(self):
        return self.gaze_tracker.pupils_located

    def horizontal_ratio(self):
        return self.gaze_tracker.horizontal_ratio()

    def vertical_ratio(self):
        return self.gaze_tracker.vertical_ratio()

    def is_right(self):
        return self.gaze_tracker.is_right()

    def is_left(self):
        return self.gaze_tracker.is_left()

    def is_center(self):
        return self.gaze_tracker.is_center()

    def is_down(self):
        return self.gaze_tracker.is_down()

    def detect_direction(self, frame):
        """Detect gaze direction using tracker"""
        return self.gaze_tracker.detect_gaze_direction(frame)
    
    def set_mirror_mode(self, mirror_mode):
        """Update mirror mode for gaze tracker"""
        self.mirror_image = mirror_mode
        if hasattr(self.gaze_tracker, 'mirror_image'):
            self.gaze_tracker.mirror_image = mirror_mode