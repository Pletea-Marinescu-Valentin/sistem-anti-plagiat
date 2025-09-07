from __future__ import division
import cv2
import json
import logging
import numpy as np
import gc
import mediapipe as mp
from .eye import Eye

class GazeTracker(object):
    """Gaze tracker with MediaPipe support"""

    def __init__(self, mirror_image=True, config_path="config.json", use_mediapipe=True):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.mirror_image = mirror_image
        self.use_mediapipe = use_mediapipe

        # Initialize detection systems
        if self.use_mediapipe:
            self._init_mediapipe()

        self.frames_without_detection = 0
        self.max_frames_without_detection = 10
        self.last_valid_direction = "center"

        # Processing optimization
        self.process_every_n_frames = 2
        self.frame_count = 0

        # Filtering values
        self.prev_h_ratio = 0.5
        self.prev_v_ratio = 0.5
        self.h_ratio = 0.5
        self.v_ratio = 0.5
        self.smoothing_factor = 0.3

        # Load configuration
        self.config = self.load_config(config_path)
        # Base thresholds (will be dynamically adjusted)
        self.base_left_limit = self.config["detection"]["gaze"]["left_limit"]
        self.base_right_limit = self.config["detection"]["gaze"]["right_limit"]
        self.base_down_limit = self.config["detection"]["gaze"]["down_limit"]
        
        # Current adjusted thresholds
        self.left_limit = self.base_left_limit
        self.right_limit = self.base_right_limit
        self.down_limit = self.base_down_limit
        
        # Head pose compensation parameters
        self.head_pose_compensation_enabled = True
        self.horizontal_rotation_threshold = 15.0  # degrees
        self.vertical_tilt_threshold = 10.0  # degrees
        self.horizontal_threshold_adjustment = 0.1
        
        # Current head pose angles
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        
        # Head pose calculation throttling
        self.head_pose_frame_count = 0
        self.head_pose_update_interval = 25  # Calculate head pose every 25 frames

        # Image mode for static image processing
        self._image_mode = False

    def _init_mediapipe(self):
        """Initialize MediaPipe Face Mesh"""
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("MediaPipe Face Mesh initialized successfully")
        except Exception as e:
            print(f"MediaPipe initialization error: {e}")
            self.use_mediapipe = False

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    def reset_all_state(self):
        """Complete reset for static images"""
        print("Complete gaze tracker reset...")
        
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.frames_without_detection = 0
        self.last_valid_direction = "center"
        self.frame_count = 0
        self.prev_h_ratio = 0.5
        self.prev_v_ratio = 0.5
        self.h_ratio = 0.5
        self.v_ratio = 0.5
        
        if hasattr(self, 'last_annotated_frame'):
            delattr(self, 'last_annotated_frame')
        
        gc.collect()
        print("Reset complete")

    def set_image_mode(self, enabled=True):
        """Activate/Deactivate mode for static images"""
        self._image_mode = enabled
        if enabled:
            print("Image mode enabled")
            self.smoothing_factor = 0.0
            self.process_every_n_frames = 1
            
            # Update MediaPipe for static images
            if self.use_mediapipe:
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
        else:
            print("Video mode enabled")
            self.smoothing_factor = 0.3
            self.process_every_n_frames = 3
            
            # Update MediaPipe for video
            if self.use_mediapipe:
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )

    def _analyze_mediapipe(self):
        """Detect face and eyes using MediaPipe"""
        if self.frame is None:
            return False

        try:
            # Convert frame to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                self.frames_without_detection += 1
                return False

            # Get face landmarks
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert landmarks to pixel coordinates
            height, width = self.frame.shape[:2]
            landmarks_px = []
            
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                landmarks_px.append((x, y))

            # Calculate head orientation with MediaPipe
            new_h_ratio, new_v_ratio = self.calculate_head_orientation_mediapipe(landmarks_px)

            # Apply smoothing for video only
            if self._image_mode or self.smoothing_factor == 0.0:
                self.h_ratio = new_h_ratio
                self.v_ratio = new_v_ratio
            else:
                self.h_ratio = self.prev_h_ratio * (1 - self.smoothing_factor) + new_h_ratio * self.smoothing_factor
                self.v_ratio = self.prev_v_ratio * (1 - self.smoothing_factor) + new_v_ratio * self.smoothing_factor

            # Update previous values
            self.prev_h_ratio = self.h_ratio
            self.prev_v_ratio = self.v_ratio

            # Convert frame to grayscale for eye processing
            gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            # Initialize eyes with MediaPipe landmarks
            self.eye_left = Eye(gray_frame, landmarks_px, 0)
            self.eye_right = Eye(gray_frame, landmarks_px, 1)

            self.frames_without_detection = 0
            return True
            
        except Exception as e:
            logging.error(f"MediaPipe analysis error: {e}")
            self.frames_without_detection += 1
            return False

    def _analyze(self):
        """Analyze frame using configured method"""
        if self.use_mediapipe:
            return self._analyze_mediapipe()

    def calculate_head_orientation_mediapipe(self, landmarks_px):
        """Calculate head orientation using MediaPipe landmarks"""
        try:
            # Key landmarks for head orientation MediaPipe indices
            nose_tip = landmarks_px[1] if len(landmarks_px) > 1 else (0, 0)
            nose_bridge = landmarks_px[6] if len(landmarks_px) > 6 else nose_tip
            
            # Face edges for H calculation
            left_face = landmarks_px[234] if len(landmarks_px) > 234 else (0, 0)
            right_face = landmarks_px[454] if len(landmarks_px) > 454 else (0, 0)
            
            # Points for V calculation
            forehead = landmarks_px[10] if len(landmarks_px) > 10 else (0, 0)
            chin = landmarks_px[152] if len(landmarks_px) > 152 else (0, 0)  # Updated to use landmark 152

            # Calculate head pose angles for compensation (throttled to reduce lag)
            self.head_pose_frame_count += 1
            if self.head_pose_frame_count >= self.head_pose_update_interval:
                self.head_pose_frame_count = 0
                yaw_angle, pitch_angle = self.calculate_head_pose_angles(landmarks_px)
                self.current_yaw = yaw_angle
                self.current_pitch = pitch_angle
                
                # Apply head pose compensation if enabled
                if self.head_pose_compensation_enabled:
                    self.adjust_thresholds_for_head_pose(yaw_angle, pitch_angle)

            # Calculate H ratio horizontal
            if left_face != (0, 0) and right_face != (0, 0):
                total_width = abs(right_face[0] - left_face[0])
                if total_width > 0:
                    nose_offset = nose_tip[0] - left_face[0]
                    h_ratio = nose_offset / total_width
                else:
                    h_ratio = 0.5
            else:
                h_ratio = 0.5

            # Calculate V ratio vertical
            if forehead != (0, 0) and chin != (0, 0):
                total_height = abs(chin[1] - forehead[1])
                if total_height > 0:
                    nose_offset_v = nose_tip[1] - forehead[1]
                    v_ratio = nose_offset_v / total_height
                else:
                    v_ratio = 0.5
            else:
                v_ratio = 0.5

            # Clamp values
            h_ratio = max(0.0, min(1.0, h_ratio))
            v_ratio = max(0.0, min(1.0, v_ratio))

            return h_ratio, v_ratio
            
        except Exception as e:
            logging.error(f"MediaPipe head orientation calculation error: {e}")
            return 0.5, 0.5

    def calculate_head_pose_angles(self, landmarks_px):
        try:
            # Key landmarks for angle calculation
            nose_tip = landmarks_px[1] if len(landmarks_px) > 1 else None
            left_face = landmarks_px[234] if len(landmarks_px) > 234 else None
            right_face = landmarks_px[454] if len(landmarks_px) > 454 else None
            forehead = landmarks_px[10] if len(landmarks_px) > 10 else None
            chin = landmarks_px[152] if len(landmarks_px) > 152 else None
            
            if not all([nose_tip, left_face, right_face, forehead, chin]):
                return 0.0, 0.0
            
            # Calculate horizontal rotation (yaw) angle
            face_width = abs(right_face[0] - left_face[0])
            if face_width > 0:
                # Center of face horizontally
                face_center_x = (left_face[0] + right_face[0]) / 2
                nose_offset_x = nose_tip[0] - face_center_x
                
                yaw_ratio = nose_offset_x / (face_width / 2)
                yaw_angle = yaw_ratio * 30.0
            else:
                yaw_angle = 0.0
            
            # Calculate vertical tilt (pitch) angle
            face_height = abs(chin[1] - forehead[1])
            if face_height > 0:
                # Center of face vertically
                face_center_y = (forehead[1] + chin[1]) / 2
                nose_offset_y = nose_tip[1] - face_center_y
                
                pitch_ratio = nose_offset_y / (face_height / 2)
                pitch_angle = pitch_ratio * 25.0  # Scale to approximate degrees
            else:
                pitch_angle = 0.0
            
            # Clamp angles to reasonable ranges
            yaw_angle = max(-45.0, min(45.0, yaw_angle))
            pitch_angle = max(-30.0, min(30.0, pitch_angle))
            
            return yaw_angle, pitch_angle
            
        except Exception as e:
            logging.error(f"Head pose angle calculation error: {e}")
            return 0.0, 0.0

    def adjust_thresholds_for_head_pose(self, yaw_angle, pitch_angle):
        try:
            # Reset to base thresholds
            self.left_limit = self.base_left_limit
            self.right_limit = self.base_right_limit
            self.down_limit = self.base_down_limit
            
            # Horizontal adjustment for yaw rotation
            if abs(yaw_angle) > self.horizontal_rotation_threshold:
                adjustment = self.horizontal_threshold_adjustment
                
                if yaw_angle > 0:  # Head turned right
                    # Make right detection more sensitive, left less sensitive
                    self.right_limit += adjustment
                    self.left_limit -= adjustment
                else:  # Head turned left
                    # Make left detection more sensitive, right less sensitive
                    self.left_limit += adjustment
                    self.right_limit -= adjustment
            
            # Vertical adjustment for pitch tilt
            if abs(pitch_angle) > self.vertical_tilt_threshold:
                # Proportional adjustment based on tilt angle
                tilt_factor = abs(pitch_angle) / 30.0  # Normalize to 0-1 range
                vertical_adjustment = 0.1 * tilt_factor
                
                if pitch_angle > 0:  # Head tilted down
                    # Make down detection less sensitive
                    self.down_limit += vertical_adjustment
                else:  # Head tilted up
                    # Make down detection more sensitive
                    self.down_limit -= vertical_adjustment
            
            # Ensure thresholds remain within valid ranges
            self.left_limit = max(0.1, min(0.9, self.left_limit))
            self.right_limit = max(0.1, min(0.9, self.right_limit))
            self.down_limit = max(0.1, min(0.9, self.down_limit))
            
        except Exception as e:
            logging.error(f"Threshold adjustment error: {e}")

    def refresh(self, frame):
        """Update frame and analyze"""
        self.frame = frame
        return self._analyze()

    @property
    def pupils_located(self):
        """Check if at least one pupil is detected"""
        if self.eye_left is None and self.eye_right is None:
            return False
        
        left_valid = (self.eye_left is not None and 
                        self.eye_left.pupil is not None and 
                        self.eye_left.is_valid_detection())
        
        right_valid = (self.eye_right is not None and 
                        self.eye_right.pupil is not None and 
                        self.eye_right.is_valid_detection())
        
        return left_valid or right_valid

    @property
    def both_pupils_located(self):
        """Check if both pupils are detected"""
        if self.eye_left is None or self.eye_right is None:
            return False
        
        left_valid = (self.eye_left.pupil is not None and 
                        self.eye_left.is_valid_detection())
        
        right_valid = (self.eye_right.pupil is not None and 
                        self.eye_right.is_valid_detection())
        
        return left_valid and right_valid

    def horizontal_ratio(self):
        """Returns horizontal gaze ratio with MediaPipe support"""
        if not self.pupils_located:
            return self.prev_h_ratio if hasattr(self, 'prev_h_ratio') else 0.5
        
        if not self.both_pupils_located:
            return self._horizontal_ratio_single_eye()
        
        try:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2
        except (AttributeError, ZeroDivisionError):
            return 0.5

    def vertical_ratio(self):
        """Returns vertical gaze ratio with enhanced down detection"""
        if not self.pupils_located:
            return self.prev_v_ratio if hasattr(self, 'prev_v_ratio') else 0.5
        
        if not self.both_pupils_located:
            return self._vertical_ratio_single_eye()
        
        try:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2
        except (AttributeError, ZeroDivisionError):
            return 0.5

    def _horizontal_ratio_single_eye(self):
        """Calculate horizontal ratio when only one eye is visible"""
        if self.eye_left is None or self.eye_right is None:
            return None
            
        if self.eye_left.pupil and (self.eye_right is None or not self.eye_right.pupil):
            try:
                ratio = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
                return max(0.1, min(0.9, ratio * 1.2))
            except (AttributeError, ZeroDivisionError):
                return 0.5
        elif self.eye_right.pupil and (self.eye_left is None or not self.eye_left.pupil):
            try:
                ratio = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
                return max(0.1, min(0.9, ratio * 0.8))
            except (AttributeError, ZeroDivisionError):
                return 0.5
        return None

    def _vertical_ratio_single_eye(self):
        """Calculate vertical ratio when only one eye is visible"""
        if self.eye_left is None or self.eye_right is None:
            return None
            
        if self.eye_left.pupil and (self.eye_right is None or not self.eye_right.pupil):
            try:
                return self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            except (AttributeError, ZeroDivisionError):
                return 0.5
        elif self.eye_right.pupil and (self.eye_left is None or not self.eye_left.pupil):
            try:
                return self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            except (AttributeError, ZeroDivisionError):
                return 0.5
        return None

    def is_right(self):
        """Check right gaze with enhanced detection"""
        if hasattr(self, 'h_ratio'):
            return self.h_ratio < self.right_limit
        return False

    def is_left(self):
        """Check left gaze with enhanced detection"""
        if hasattr(self, 'h_ratio'):
            return self.h_ratio > self.left_limit
        return False

    def is_down(self):
        """Enhanced down detection using both pupil and head pose"""
        if not hasattr(self, 'v_ratio'):
            return False
        
        # Standard pupil-based detection
        pupil_down = self.v_ratio > self.down_limit
        
        # Extreme detection for very tilted head
        extreme_down = self.v_ratio > 0.75
        
        # If using MediaPipe, better tilted head detection
        if self.use_mediapipe and hasattr(self, 'frame') and self.frame is not None:
            head_tilt_down = self._detect_extreme_head_tilt()
            return pupil_down or extreme_down or head_tilt_down
        
        return pupil_down or extreme_down

    def _detect_extreme_head_tilt(self):
        """Detect extreme head tilt down using MediaPipe"""
        try:
            if not self.use_mediapipe or self.frame is None:
                return False
                
            # Use landmarks to detect extreme tilt
            rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return False
                
            face_landmarks = results.multi_face_landmarks[0]
            height, width = self.frame.shape[:2]
            
            # Calculate relative position of key points
            # Nose point 1, forehead 10, chin 152
            nose = face_landmarks.landmark[1]
            forehead = face_landmarks.landmark[10]
            chin = face_landmarks.landmark[152]
            
            # Convert to pixel coordinates
            nose_y = nose.y * height
            forehead_y = forehead.y * height
            chin_y = chin.y * height
            
            # Calculate ratios for tilted head detection
            if chin_y > forehead_y:  # Basic check
                face_height = chin_y - forehead_y
                if face_height > 0:
                    nose_position = (nose_y - forehead_y) / face_height
                    
                    # If nose is very low in face >0.8, head is probably tilted down
                    if nose_position > 0.8:
                        return True
                        
            return False
            
        except Exception as e:
            logging.error(f"Extreme tilt detection error: {e}")
            return False

    def is_center(self):
        """Check center gaze with enhanced detection"""
        if not hasattr(self, 'h_ratio') or not hasattr(self, 'v_ratio'):
            return False
            
        h_ok = self.right_limit <= self.h_ratio <= self.left_limit
        v_ok = self.v_ratio <= self.down_limit
        
        return h_ok and v_ok

    def determine_direction(self):
        """Determine gaze direction with enhanced down detection"""
        if not self.pupils_located:
            if self.frames_without_detection > self.max_frames_without_detection:
                return "no_face"
            else:
                return self.last_valid_direction

        # Determine direction with enhanced logic
        if self.is_down():  # Check DOWN first priority for tilted head
            direction = "down"
        elif self.is_right():
            direction = "right" if not self.mirror_image else "left"
        elif self.is_left():
            direction = "left" if not self.mirror_image else "right"
        else:
            direction = "center"

        self.last_valid_direction = direction
        return direction

    def annotated_frame(self):
        """Create annotated frame with enhanced visualization"""
        if self.frame is None:
            return None

        frame = self.frame.copy()

        if self.pupils_located:
            # Color based on direction
            is_center_view = self.is_center()

            if is_center_view:
                color = (0, 255, 0)  # Green for center
            else:
                color = (0, 0, 255)  # Red for other directions

            # Draw pupil circles
            try:
                coords_left = self.pupil_left_coords()
                coords_right = self.pupil_right_coords()

                if coords_left:
                    x_left, y_left = coords_left
                    cv2.circle(frame, (x_left, y_left), 8, color, -1)

                if coords_right:
                    x_right, y_right = coords_right
                    cv2.circle(frame, (x_right, y_right), 8, color, -1)
            except:
                pass

        return frame, self.h_ratio if hasattr(self, 'h_ratio') else 0.5, self.v_ratio if hasattr(self, 'v_ratio') else 0.5

    def pupil_left_coords(self):
        """Get left pupil coordinates"""
        try:
            if self.eye_left and self.eye_left.pupil and self.eye_left.origin:
                x = self.eye_left.origin[0] + self.eye_left.pupil.x
                y = self.eye_left.origin[1] + self.eye_left.pupil.y
                return (int(x), int(y))
        except:
            pass
        return None

    def pupil_right_coords(self):
        """Get right pupil coordinates"""
        try:
            if self.eye_right and self.eye_right.pupil and self.eye_right.origin:
                x = self.eye_right.origin[0] + self.eye_right.pupil.x
                y = self.eye_right.origin[1] + self.eye_right.pupil.y
                return (int(x), int(y))
        except:
            pass
        return None

    def detect_gaze_direction(self, frame):
        """Process frame and detect gaze direction with enhanced capabilities"""
        
        # For static images, completely reset state
        if self._image_mode:
            self.reset_all_state()
        
        self.frame_count += 1

        # For images, process every frame; for video, skip frames
        process_frame = True
        if not self._image_mode:
            process_frame = (self.frame_count % self.process_every_n_frames == 0)

        if process_frame:
            # Update frame and analyze
            self.frame = frame
            self._analyze()

            # Direction determination with enhanced logic
            direction = self.determine_direction()

            # Save annotated frame
            annotated_result = self.annotated_frame()
            if annotated_result is not None:
                self.last_annotated_frame, self.h_ratio, self.v_ratio = annotated_result
            else:
                self.last_annotated_frame = frame
        else:
            # For unprocessed frames, use saved direction
            direction = self.last_valid_direction

        # Return last valid annotated frame
        if hasattr(self, 'last_annotated_frame') and self.last_annotated_frame is not None:
            return direction, self.last_annotated_frame, self.h_ratio, self.v_ratio
        else:
            return direction, frame, self.h_ratio, self.v_ratio

    def get_head_pose_info(self):
        """
        Get current head pose information for monitoring and debugging
        
        Returns:
            dict: Head pose information including angles and adjusted thresholds
        """
        return {
            'yaw_angle': round(self.current_yaw, 2),
            'pitch_angle': round(self.current_pitch, 2),
            'compensation_enabled': self.head_pose_compensation_enabled,
            'base_thresholds': {
                'left_limit': self.base_left_limit,
                'right_limit': self.base_right_limit,
                'down_limit': self.base_down_limit
            },
            'adjusted_thresholds': {
                'left_limit': round(self.left_limit, 3),
                'right_limit': round(self.right_limit, 3),
                'down_limit': round(self.down_limit, 3)
            },
            'threshold_adjustments': {
                'horizontal_threshold': self.horizontal_rotation_threshold,
                'vertical_threshold': self.vertical_tilt_threshold,
                'horizontal_adjustment': self.horizontal_threshold_adjustment
            }
        }

    def set_head_pose_compensation(self, enabled):
        """
        Enable or disable head pose compensation
        
        Args:
            enabled (bool): Whether to enable head pose compensation
        """
        self.head_pose_compensation_enabled = enabled
        if not enabled:
            # Reset to base thresholds
            self.left_limit = self.base_left_limit
            self.right_limit = self.base_right_limit
            self.down_limit = self.base_down_limit