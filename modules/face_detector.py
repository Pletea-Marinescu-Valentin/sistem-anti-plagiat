import json
import logging
from .gaze_tracking import GazeTracker
from .object_detector import ObjectDetector

class FaceDetector:
    """
    Class for gaze direction detection
    """
    def __init__(self, mirror_image=True):
        self.mirror_image = mirror_image
        self.gaze_tracker = GazeTracker(mirror_image=mirror_image)
        
        # Initialize object detector
        try:
            with open("config.json", 'r') as f:
                config = json.load(f)
            self.object_detector = ObjectDetector(config)
        except Exception as e:
            logging.error(f"Failed to initialize object detector: {e}")
            self.object_detector = None

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
        """Detect gaze direction using tracker - EXACT CA ÎNAINTE"""
        return self.gaze_tracker.detect_gaze_direction(frame)
    
    def detect_with_objects(self, frame):
        """Detect both gaze direction and prohibited objects"""
        
        direction, annotated_frame, h_ratio, v_ratio = self.detect_direction(frame)
        
        detected_objects = []
        if self.object_detector:
            try:
                detected_objects, annotated_frame = self.object_detector.detect_objects(annotated_frame)
            except Exception as e:
                logging.error(f"Object detection error: {e}")
        
        return {
            'direction': direction,
            'h_ratio': h_ratio,
            'v_ratio': v_ratio,
            'objects': detected_objects,
            'annotated_frame': annotated_frame
        }
    
    def set_mirror_mode(self, mirror_mode):
        """Update mirror mode for both gaze tracker and object detector"""
        self.mirror_image = mirror_mode
        if hasattr(self.gaze_tracker, 'mirror_image'):
            self.gaze_tracker.mirror_image = mirror_mode
        if self.object_detector:
            self.object_detector.set_mirror_state(mirror_mode)