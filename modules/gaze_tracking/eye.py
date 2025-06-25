import math
import numpy as np
import cv2
from .pupil import Pupil
from .pupil_tracker import KalmanPupilTracker


class Eye(object):
    """
    Eye class using MediaPipe landmarks for pupil detection
    """

    # MediaPipe eye indices 468 points
    LEFT_EYE_MP_INDICES = [
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
    ]
    
    RIGHT_EYE_MP_INDICES = [
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
    ]
    
    # Iris points MediaPipe
    LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
    RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

    def __init__(self, original_frame, landmarks_px, side):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None
        self.landmark_points = None
        self.blinking = None
        
        # Add Kalman tracker for pupil
        self.pupil_tracker = KalmanPupilTracker()
        self.raw_pupil_position = None
        
        self._analyze(original_frame, landmarks_px, side)

    @staticmethod
    def _middle_point(p1, p2):
        """Calculate midpoint between two points"""
        x = int((p1[0] + p2[0]) / 2)
        y = int((p1[1] + p2[1]) / 2)
        return (x, y)

    def _isolate_eye(self, frame, landmarks_px, eye_indices):
        """Isolate eye using MediaPipe landmarks"""
        try:
            # Get eye points
            eye_points = []
            for idx in eye_indices:
                if idx < len(landmarks_px):
                    eye_points.append(landmarks_px[idx])
                else:
                    eye_points.append((0, 0))
            
            if len(eye_points) < 6:
                return False
            
            # Convert to numpy array for processing
            region = np.array(eye_points, dtype=np.int32)
            self.landmark_points = region

            # Create mask for eye isolation
            height, width = frame.shape[:2]
            mask = np.zeros((height, width), np.uint8)
            
            # Use convex hull for better eye shape
            hull = cv2.convexHull(region)
            cv2.fillPoly(mask, [hull], 255)
            
            # Apply mask
            eye = cv2.bitwise_and(frame, frame, mask=mask)

            # Calculate bounding box with margin
            x_coords = [p[0] for p in eye_points]
            y_coords = [p[1] for p in eye_points]
            
            min_x = max(0, min(x_coords) - 10)
            max_x = min(width, max(x_coords) + 10)
            min_y = max(0, min(y_coords) - 10)
            max_y = min(height, max(y_coords) + 10)

            # Check if region is valid
            if min_x >= max_x or min_y >= max_y:
                return False

            # Save cropped frame and origin
            self.frame = eye[min_y:max_y, min_x:max_x]
            self.origin = (min_x, min_y)

            # Calculate eye center
            if self.frame is not None and self.frame.size > 0:
                height, width = self.frame.shape[:2]
                self.center = (width / 2, height / 2)
            else:
                self.center = (1, 1)
            
            return True
            
        except Exception as e:
            return False

    def _blinking_ratio(self, landmarks_px, eye_indices):
        """Calculate blinking ratio for MediaPipe"""
        try:
            if len(eye_indices) < 6:
                return None
                
            # Use extreme points of the eye
            eye_points = [landmarks_px[i] for i in eye_indices if i < len(landmarks_px)]
            
            if len(eye_points) < 6:
                return None
            
            # Find extreme points
            leftmost = min(eye_points, key=lambda p: p[0])
            rightmost = max(eye_points, key=lambda p: p[0])
            topmost = min(eye_points, key=lambda p: p[1])
            bottommost = max(eye_points, key=lambda p: p[1])
            
            # Calculate dimensions
            eye_width = math.hypot(rightmost[0] - leftmost[0], rightmost[1] - leftmost[1])
            eye_height = math.hypot(topmost[0] - bottommost[0], topmost[1] - bottommost[1])

            if eye_height == 0:
                return None
                
            ratio = eye_width / eye_height
            return ratio
            
        except Exception as e:
            return None

    def _analyze(self, original_frame, landmarks_px, side):
        """Detect and isolate eye, then initialize Pupil object"""
        
        # Select eye indices based on side
        if side == 0:  # left eye
            eye_indices = self.LEFT_EYE_MP_INDICES
        elif side == 1:  # right eye
            eye_indices = self.RIGHT_EYE_MP_INDICES
        else:
            return

        # Calculate blinking ratio
        self.blinking = self._blinking_ratio(landmarks_px, eye_indices)

        # Isolate eye
        if not self._isolate_eye(original_frame, landmarks_px, eye_indices):
            return

        # Create pupil if eye was isolated successfully
        if self.frame is not None and self.frame.size > 0:
            self.pupil = Pupil(self.frame)

            # Store raw detection
            self.raw_pupil_position = (self.pupil.x, self.pupil.y)
            
            # Get confidence from pupil detection
            confidence = getattr(self.pupil, 'confidence', 0.5)
            
            # Check for outliers
            if not self.pupil_tracker.is_outlier(self.pupil.x, self.pupil.y):
                # Update tracker with new measurement
                filtered_x, filtered_y = self.pupil_tracker.update(
                    self.pupil.x, self.pupil.y, confidence
                )
                
                # Update pupil position with filtered values
                self.pupil.x = filtered_x
                self.pupil.y = filtered_y
            else:
                # Outlier detected - use prediction instead
                prediction = self.pupil_tracker.predict()
                if prediction:
                    self.pupil.x, self.pupil.y = prediction

    def is_valid_detection(self):
        """Enhanced validation of pupil detection quality"""
        if not self.pupil:
            return False
        
        if self.frame is None:
            return False
            
        eye_height, eye_width = self.frame.shape[:2]
        
        if eye_width == 0 or eye_height == 0:
            return False
        
        # Check if pupil is within realistic eye boundaries
        margin = 5
        if (self.pupil.x < margin or self.pupil.x > eye_width - margin or 
            self.pupil.y < margin or self.pupil.y > eye_height - margin):
            return False
        
        # Check distance from eye center
        center_x, center_y = eye_width // 2, eye_height // 2
        distance_from_center = np.sqrt((self.pupil.x - center_x)**2 + (self.pupil.y - center_y)**2)
        max_distance = min(eye_width, eye_height) * 0.4
        
        if distance_from_center > max_distance:
            return False
        
        # Validate pupil intensity
        if hasattr(self.pupil, 'confidence') and self.pupil.confidence < 0.3:
            return False
        
        # Check local intensity around pupil position
        try:
            pupil_region = self.frame[max(0, self.pupil.y-3):min(eye_height, self.pupil.y+3),
                                     max(0, self.pupil.x-3):min(eye_width, self.pupil.x+3)]
            
            if pupil_region.size > 0:
                pupil_intensity = np.mean(pupil_region)
                frame_intensity = np.mean(self.frame)
                
                # Pupil should be significantly darker than average
                if pupil_intensity > frame_intensity * 0.7:
                    return False
        except:
            return False
        
        return True

    def detection_confidence(self):
        """Calculate confidence score for this eye's detection"""
        if not self.pupil or not hasattr(self.pupil, 'confidence'):
            return 0.0
        
        confidence = self.pupil.confidence
        
        # Boost confidence if pupil is well-centered
        if self.frame is not None:
            center_x, center_y = self.frame.shape[1] // 2, self.frame.shape[0] // 2
            distance = np.sqrt((self.pupil.x - center_x)**2 + (self.pupil.y - center_y)**2)
            max_distance = min(self.frame.shape[1], self.frame.shape[0]) * 0.5
            
            if max_distance > 0:
                center_bonus = 1.0 - (distance / max_distance)
                confidence += center_bonus * 0.2
        
        return min(confidence, 1.0)

    def get_pupil_velocity(self):
        """Get current pupil velocity"""
        return self.pupil_tracker.get_velocity()
        
    def get_movement_stability(self):
        """Calculate how stable the pupil movement is"""
        vx, vy = self.get_pupil_velocity()
        velocity_magnitude = np.sqrt(vx**2 + vy**2)
        
        # Stability inversely related to velocity
        stability = 1.0 / (1.0 + velocity_magnitude / 10.0)
        return stability