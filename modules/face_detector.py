import json
import logging
import cv2
import time
import numpy as np
import mediapipe as mp
from .gaze_tracking.gaze_tracker import GazeTracker

class FaceDetector:
    """
    Face detector using MediaPipe for gaze tracking with head pose compensation
    """
    def __init__(self, mirror_image=True):
        self.mirror_image = mirror_image
        self._init_mediapipe()
        
        # Initialize GazeTracker pentru head pose compensation
        self.gaze_tracker = GazeTracker(mirror_image=mirror_image, use_mediapipe=True)
        
        # Performance tracking
        self.last_detection_time = 0
        
    def _init_mediapipe(self):
        """Initialize MediaPipe face mesh"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.config = self._load_config()
        self.current_h_ratio = 0.5
        self.current_v_ratio = 0.5
        self.pupils_detected = False
        self.last_direction = "center"
        self.last_annotated_frame = None
    
    def _load_config(self):
        """Load configuration from JSON file"""
        try:
            with open("config.json", 'r') as f:
                return json.load(f)
        except:
            return {
                'detection': {
                    'gaze': {
                        'left_limit': 0.65,
                        'right_limit': 0.35,
                        'down_limit': 0.6
                    }
                }
            }

    @property
    def pupils_located(self):
        """Check if pupils are detected"""
        return self.pupils_detected

    def horizontal_ratio(self):
        """Get horizontal gaze ratio"""
        return self.current_h_ratio

    def vertical_ratio(self):
        """Get vertical gaze ratio"""
        return self.current_v_ratio

    def is_right(self):
        """Check if looking right"""
        return self.current_h_ratio < self.config['detection']['gaze']['right_limit']

    def is_left(self):
        """Check if looking left"""
        return self.current_h_ratio > self.config['detection']['gaze']['left_limit']

    def is_center(self):
        """Check if looking center"""
        h_ok = self.config['detection']['gaze']['right_limit'] <= self.current_h_ratio <= self.config['detection']['gaze']['left_limit']
        v_ok = self.current_v_ratio <= self.config['detection']['gaze']['down_limit']
        return h_ok and v_ok

    def is_down(self):
        """Check if looking down"""
        return self.current_v_ratio > self.config['detection']['gaze']['down_limit'] or self.current_v_ratio > 0.75

    def detect_direction(self, frame):
        """Detect gaze direction from frame with head pose compensation"""
        start_time = time.time()
        
        if frame is None:
            return "center", frame, 0.5, 0.5
        
        try:
            self.gaze_tracker.refresh(frame)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                self.pupils_detected = False
                self.last_detection_time = (time.time() - start_time) * 1000  # ms
                return "no_face", frame, self.current_h_ratio, self.current_v_ratio
            
            face_landmarks = results.multi_face_landmarks[0]
            height, width = frame.shape[:2]
            
            h_ratio, v_ratio = self._calculate_ratios(face_landmarks, width, height)
            
            self.current_h_ratio = h_ratio
            self.current_v_ratio = v_ratio
            self.pupils_detected = True
            
            # Record processing time
            self.last_detection_time = (time.time() - start_time) * 1000  # ms
            
            direction = self._determine_direction(h_ratio, v_ratio)
            self.last_direction = direction
            
            annotated_frame = self._create_annotated_frame(frame, h_ratio, v_ratio, direction)
            self.last_annotated_frame = annotated_frame
            
            return direction, annotated_frame, h_ratio, v_ratio
            
        except Exception as e:
            self.pupils_detected = False
            return "center", frame, self.current_h_ratio, self.current_v_ratio
    
    def _calculate_ratios(self, face_landmarks, width, height):
        """Calculate H and V ratios from face landmarks"""
        try:
            nose_tip = face_landmarks.landmark[1]
            forehead = face_landmarks.landmark[10] 
            chin = face_landmarks.landmark[152]
            left_face = face_landmarks.landmark[234]
            right_face = face_landmarks.landmark[454]
            
            # Calculate H ratio
            nose_x = nose_tip.x * width
            left_x = left_face.x * width
            right_x = right_face.x * width
            
            if right_x > left_x and (right_x - left_x) > 0:
                h_ratio = (nose_x - left_x) / (right_x - left_x)
            else:
                h_ratio = 0.5
            
            # Calculate V ratio
            nose_y = nose_tip.y * height
            forehead_y = forehead.y * height
            chin_y = chin.y * height
            
            if chin_y > forehead_y and (chin_y - forehead_y) > 0:
                v_ratio = (nose_y - forehead_y) / (chin_y - forehead_y)
            else:
                v_ratio = 0.5
            
            h_ratio = max(0.0, min(1.0, h_ratio))
            v_ratio = max(0.0, min(1.0, v_ratio))
            
            return h_ratio, v_ratio
            
        except Exception as e:
            return 0.5, 0.5
    
    def _determine_direction(self, h_ratio, v_ratio):
        """Determine gaze direction from ratios"""
        if v_ratio > self.config['detection']['gaze']['down_limit'] or v_ratio > 0.75:
            return "down"
        elif h_ratio > self.config['detection']['gaze']['left_limit']:
            return "left" if not self.mirror_image else "right"
        elif h_ratio < self.config['detection']['gaze']['right_limit']:
            return "right" if not self.mirror_image else "left"
        else:
            return "center"
    
    def _create_annotated_frame(self, frame, h_ratio, v_ratio, direction):
        """Create annotated frame with eye tracking"""
        annotated = frame.copy()
        
        if direction == "center":
            color = (0, 255, 0)
        elif direction == "no_face":
            color = (128, 128, 128)
        else:
            color = (0, 0, 255)
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                height, width = frame.shape[:2]
                
                # Left eye landmarks
                left_eye_landmarks = [33, 160, 158, 133, 153, 144]
                left_eye_points = []
                for idx in left_eye_landmarks:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    left_eye_points.append((x, y))
                
                if left_eye_points:
                    left_eye_x = int(np.mean([p[0] for p in left_eye_points]))
                    left_eye_y = int(np.mean([p[1] for p in left_eye_points]))
                    self.left_eye_coords = (left_eye_x, left_eye_y)
                else:
                    self.left_eye_coords = None
                
                # Right eye landmarks
                right_eye_landmarks = [362, 385, 387, 263, 373, 380]
                right_eye_points = []
                for idx in right_eye_landmarks:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    right_eye_points.append((x, y))
                
                if right_eye_points:
                    right_eye_x = int(np.mean([p[0] for p in right_eye_points]))
                    right_eye_y = int(np.mean([p[1] for p in right_eye_points]))
                    self.right_eye_coords = (right_eye_x, right_eye_y)
                else:
                    self.right_eye_coords = None
                
                # Draw eye circles - single smaller circle per eye
                if self.left_eye_coords:
                    cv2.circle(annotated, self.left_eye_coords, 4, color, -1)
                
                if self.right_eye_coords:
                    cv2.circle(annotated, self.right_eye_coords, 4, color, -1)
            
        except Exception as e:
            height, width = frame.shape[:2]
            self.left_eye_coords = (int(width * 0.35), int(height * 0.4))
            self.right_eye_coords = (int(width * 0.65), int(height * 0.4))
            
            cv2.circle(annotated, self.left_eye_coords, 4, color, -1)
            cv2.circle(annotated, self.right_eye_coords, 4, color, -1)
        
        return annotated
    
    def set_mirror_mode(self, mirror_mode):
        """Update mirror mode"""
        self.mirror_image = mirror_mode
        if hasattr(self, 'gaze_tracker'):
            self.gaze_tracker.mirror_image = mirror_mode
    
    def pupil_left_coords(self):
        """Get left pupil coordinates"""
        return getattr(self, 'left_eye_coords', None)
    
    def pupil_right_coords(self):
        """Get right pupil coordinates"""
        return getattr(self, 'right_eye_coords', None)