import json
import logging
from .gaze_tracking.gaze_tracker import GazeTracker

class FaceDetector:
    """
    Enhanced FaceDetector cu suport pentru MediaPipe și dlib
    """
    def __init__(self, mirror_image=True, use_mediapipe=True):
        self.mirror_image = mirror_image
        self.use_mediapipe = use_mediapipe
        
        # Inițializează GazeTracker cu MediaPipe sau dlib
        try:
            self.gaze_tracker = GazeTracker(
                mirror_image=mirror_image, 
                use_mediapipe=use_mediapipe
            )
            
            if use_mediapipe:
                logging.info("FaceDetector inițializat cu MediaPipe")
            else:
                logging.info("FaceDetector inițializat cu dlib")
                
        except Exception as e:
            logging.error(f"Eroare la inițializarea FaceDetector: {e}")
            # Fallback la dlib dacă MediaPipe nu funcționează
            if use_mediapipe:
                logging.info("Revin la dlib ca fallback...")
                self.use_mediapipe = False
                self.gaze_tracker = GazeTracker(
                    mirror_image=mirror_image, 
                    use_mediapipe=False
                )

    @property
    def pupils_located(self):
        """Check if pupils are located"""
        return self.gaze_tracker.pupils_located

    def horizontal_ratio(self):
        """Get horizontal gaze ratio"""
        return self.gaze_tracker.horizontal_ratio()

    def vertical_ratio(self):
        """Get vertical gaze ratio"""
        return self.gaze_tracker.vertical_ratio()

    def is_right(self):
        """Check if looking right"""
        return self.gaze_tracker.is_right()

    def is_left(self):
        """Check if looking left"""
        return self.gaze_tracker.is_left()

    def is_center(self):
        """Check if looking center"""
        return self.gaze_tracker.is_center()

    def is_down(self):
        """Enhanced down detection"""
        return self.gaze_tracker.is_down()

    def detect_direction(self, frame):
        """Detect gaze direction using enhanced tracker"""
        return self.gaze_tracker.detect_gaze_direction(frame)
    
    def set_mirror_mode(self, mirror_mode):
        """Update mirror mode for gaze tracker"""
        self.mirror_image = mirror_mode
        if hasattr(self.gaze_tracker, 'mirror_image'):
            self.gaze_tracker.mirror_image = mirror_mode
            
    def set_image_mode(self, enabled=True):
        """Set image mode for static image processing"""
        if hasattr(self.gaze_tracker, 'set_image_mode'):
            self.gaze_tracker.set_image_mode(enabled)
            
    def reset_state(self):
        """Reset tracker state (useful for static images)"""
        if hasattr(self.gaze_tracker, 'reset_all_state'):
            self.gaze_tracker.reset_all_state()
            
    def switch_to_mediapipe(self):
        """Switch to MediaPipe detection"""
        try:
            self.use_mediapipe = True
            self.gaze_tracker = GazeTracker(
                mirror_image=self.mirror_image, 
                use_mediapipe=True
            )
            logging.info("Switched to MediaPipe successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to switch to MediaPipe: {e}")
            return False
            
    def switch_to_dlib(self):
        """Switch to dlib detection"""
        try:
            self.use_mediapipe = False
            self.gaze_tracker = GazeTracker(
                mirror_image=self.mirror_image, 
                use_mediapipe=False
            )
            logging.info("Switched to dlib successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to switch to dlib: {e}")
            return False
            
    def get_detection_info(self):
        """Get information about current detection method"""
        return {
            'method': 'MediaPipe' if self.use_mediapipe else 'dlib',
            'mirror_image': self.mirror_image,
            'pupils_located': self.pupils_located,
            'h_ratio': self.horizontal_ratio() if self.pupils_located else None,
            'v_ratio': self.vertical_ratio() if self.pupils_located else None
        }