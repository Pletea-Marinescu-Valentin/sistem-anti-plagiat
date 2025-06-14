from __future__ import division
import cv2
import dlib
import json
import numpy as np
from .eye import Eye

class GazeTracker(object):
    """this class tracks the user's gaze direction and provides information about eye and pupil positions"""

    def __init__(self, mirror_image=True, config_path="config.json"):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.mirror_image = mirror_image

        # face detector
        self._face_detector = dlib.get_frontal_face_detector()

        # facial landmarks predictor
        model_path = "shape_predictor_68_face_landmarks.dat"
        self._predictor = dlib.shape_predictor(model_path)

        self.frames_without_detection = 0
        self.max_frames_without_detection = 10
        self.last_valid_direction = "center"

        # processing optimization
        self.process_every_n_frames = 2
        self.frame_count = 0

        # values for filtering
        self.prev_h_ratio = 0.5
        self.prev_v_ratio = 0.5
        self.h_ratio = 0.5
        self.v_ratio = 0.5

        # filtering factor
        self.smoothing_factor = 0.3

        # load configuration
        self.config = self.load_config(config_path)
        self.left_limit = self.config["detection"]["gaze"]["left_limit"]
        self.right_limit = self.config["detection"]["gaze"]["right_limit"]
        self.down_limit = self.config["detection"]["gaze"]["down_limit"]

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    def _analyze(self):
        """detects face and eyes"""
        if self.frame is None:
            return False

        frame_copy = self.frame.copy()

        # resize for faster processing
        scale_factor = 0.5
        small_frame = cv2.resize(frame_copy, None, fx=scale_factor, fy=scale_factor)

        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # face detection
        faces = self._face_detector(gray, 1)

        if len(faces) == 0:
            self.frames_without_detection += 1
            return False

        try:
            # convert coordinates
            face_orig = dlib.rectangle(
                int(faces[0].left() / scale_factor),
                int(faces[0].top() / scale_factor),
                int(faces[0].right() / scale_factor),
                int(faces[0].bottom() / scale_factor)
            )

            # convert original frame
            gray_orig = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)

            landmarks = self._predictor(gray_orig, face_orig)

            # calculate head orientation
            new_h_ratio, new_v_ratio = self.calculate_head_orientation(landmarks)

            # filter values
            self.h_ratio = self.prev_h_ratio * (1 - self.smoothing_factor) + new_h_ratio * self.smoothing_factor
            self.v_ratio = self.prev_v_ratio * (1 - self.smoothing_factor) + new_v_ratio * self.smoothing_factor

            # update values
            self.prev_h_ratio = self.h_ratio
            self.prev_v_ratio = self.v_ratio

            # initialize eyes
            self.eye_left = Eye(gray_orig, landmarks, 0)
            self.eye_right = Eye(gray_orig, landmarks, 1)

            self.frames_without_detection = 0
            return True
        except Exception as e:
            self.frames_without_detection += 1
            print(f"face analysis error: {e}")
            return False

    def refresh(self, frame):
        """update frame and analyze"""
        self.frame = frame
        return self._analyze()
    
    def calculate_gaze_vector(self):
        """Calculate gaze direction vector with confidence weighting"""
        if not self.pupils_located:
            return None, None
        
        # Get relative pupil positions (normalized to eye center)
        left_gaze_x = (self.eye_left.pupil.x - self.eye_left.center[0]) / self.eye_left.center[0]
        left_gaze_y = (self.eye_left.pupil.y - self.eye_left.center[1]) / self.eye_left.center[1]
        
        right_gaze_x = (self.eye_right.pupil.x - self.eye_right.center[0]) / self.eye_right.center[0]
        right_gaze_y = (self.eye_right.pupil.y - self.eye_right.center[1]) / self.eye_right.center[1]
        
        # Weight by detection confidence
        left_confidence = self.eye_left.detection_confidence() if hasattr(self.eye_left, 'detection_confidence') else 0.5
        right_confidence = self.eye_right.detection_confidence() if hasattr(self.eye_right, 'detection_confidence') else 0.5
        
        total_confidence = left_confidence + right_confidence
        
        if total_confidence > 0:
            # Confidence-weighted average
            gaze_x = (left_gaze_x * left_confidence + right_gaze_x * right_confidence) / total_confidence
            gaze_y = (left_gaze_y * left_confidence + right_gaze_y * right_confidence) / total_confidence
        else:
            # Simple average if no confidence data
            gaze_x = (left_gaze_x + right_gaze_x) / 2
            gaze_y = (left_gaze_y + right_gaze_y) / 2
        
        return gaze_x, gaze_y

    def horizontal_ratio(self):
        """Enhanced horizontal gaze ratio with temporal smoothing"""
        if not self.pupils_located:
            return self.prev_h_ratio if hasattr(self, 'prev_h_ratio') else 0.5
        
        # Get current ratio
        current_ratio = self._calculate_base_horizontal_ratio()
        
        # Check movement stability
        left_stable = True
        right_stable = True
        
        if self.eye_left and hasattr(self.eye_left, 'get_movement_stability'):
            left_stable = self.eye_left.get_movement_stability() > 0.7
            
        if self.eye_right and hasattr(self.eye_right, 'get_movement_stability'):
            right_stable = self.eye_right.get_movement_stability() > 0.7
        
        # Adjust smoothing based on stability
        if left_stable and right_stable:
            # High stability - less smoothing needed
            smoothing = 0.2
        else:
            # Low stability - more smoothing
            smoothing = 0.5
        
        # Apply temporal smoothing
        if hasattr(self, 'prev_h_ratio'):
            filtered_ratio = self.prev_h_ratio * smoothing + current_ratio * (1 - smoothing)
        else:
            filtered_ratio = current_ratio
        
        self.prev_h_ratio = filtered_ratio
        return np.clip(filtered_ratio, 0.0, 1.0)

    def detect_head_profile(self):
        """Detect if head is in profile position"""
        # Check if eyes are initialized
        if self.eye_left is None or self.eye_right is None:
            return True
        
        # Check if origins exist
        if not hasattr(self.eye_left, 'origin') or not hasattr(self.eye_right, 'origin'):
            return True
            
        if not self.eye_left.origin or not self.eye_right.origin:
            return True
        
        eye_distance = abs(self.eye_right.origin[0] - self.eye_left.origin[0])
        expected_distance = self.frame.shape[1] * 0.15
        
        return eye_distance < expected_distance * 0.5

    def calculate_detection_confidence(self):
        """Calculate overall detection confidence with movement stability"""
        confidence = 0.0
        
        # Check if eyes are initialized
        if self.eye_left is None or self.eye_right is None:
            return 0.0
        
        # Base confidence from pupil detection
        if self.both_pupils_located:
            left_conf = self.eye_left.detection_confidence() if hasattr(self.eye_left, 'detection_confidence') else 0.5
            right_conf = self.eye_right.detection_confidence() if hasattr(self.eye_right, 'detection_confidence') else 0.5
            confidence += (left_conf + right_conf) / 2 * 0.6  # 60% weight
        elif self.pupils_located:
            single_conf = 0.5
            if self.eye_left.pupil:
                single_conf = self.eye_left.detection_confidence() if hasattr(self.eye_left, 'detection_confidence') else 0.5
            elif self.eye_right.pupil:
                single_conf = self.eye_right.detection_confidence() if hasattr(self.eye_right, 'detection_confidence') else 0.5
            confidence += single_conf * 0.4  # Lower confidence for single eye
        
        # Head pose factor - penalize extreme angles
        try:
            if not self.detect_head_profile():
                confidence += 0.2  # Bonus for frontal face
            else:
                confidence -= 0.1  # Penalty for profile view
        except (AttributeError, TypeError):
            pass
        
        # Frame quality factor
        if self.frame is not None:
            # Check illumination quality
            frame_mean = np.mean(self.frame)
            if 50 < frame_mean < 200:  # Good illumination range
                confidence += 0.1
            
            # Check for motion blur (via Laplacian variance)
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY) if len(self.frame.shape) == 3 else self.frame
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var > 100:  # Sharp image
                confidence += 0.1
        
        # Add movement stability factor
        if self.eye_left and hasattr(self.eye_left, 'get_movement_stability'):
            left_stability = self.eye_left.get_movement_stability()
            confidence += left_stability * 0.1
            
        if self.eye_right and hasattr(self.eye_right, 'get_movement_stability'):
            right_stability = self.eye_right.get_movement_stability()
            confidence += right_stability * 0.1
        
        return min(confidence, 1.0)

    def _horizontal_ratio_single_eye(self):
        """Calculate horizontal ratio when only one eye is visible"""
        # Check if eyes are initialized
        if self.eye_left is None or self.eye_right is None:
            return None
            
        if self.eye_left.pupil and (self.eye_right is None or not self.eye_right.pupil):
            ratio = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            # Adjustment for left eye alone
            return max(0.1, min(0.9, ratio * 1.2))
        elif self.eye_right.pupil and (self.eye_left is None or not self.eye_left.pupil):
            ratio = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            # Adjustment for right eye alone
            return max(0.1, min(0.9, ratio * 0.8))
        return None

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the vertical direction of the gaze.
        The extreme top is 0.0, the center is 0.5 and the extreme bottom is 1.0"""
        if not self.pupils_located:
            return None
        
        # Try single eye if both pupils not available
        if not self.both_pupils_located:
            return self._vertical_ratio_single_eye()
        
        pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
        pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
        return (pupil_left + pupil_right) / 2

    def _vertical_ratio_single_eye(self):
        """Calculate vertical ratio when only one eye is visible"""
        # Check if eyes are initialized
        if self.eye_left is None or self.eye_right is None:
            return None
            
        if self.eye_left.pupil and (self.eye_right is None or not self.eye_right.pupil):
            return self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
        elif self.eye_right.pupil and (self.eye_left is None or not self.eye_left.pupil):
            return self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
        return None

    @property
    def pupils_located(self):
        """Check if at least one pupil is detected with good confidence"""
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
        """Check if both pupils are detected with good confidence"""
        if self.eye_left is None or self.eye_right is None:
            return False
        
        left_valid = (self.eye_left.pupil is not None and 
                      self.eye_left.is_valid_detection())
        
        right_valid = (self.eye_right.pupil is not None and 
                       self.eye_right.is_valid_detection())
        
        return left_valid and right_valid

    def is_right(self):
        """check right gaze"""
        if hasattr(self, 'h_ratio'):
            if self.h_ratio < self.right_limit:
                return True
        return False

    def is_left(self):
        """check left gaze"""
        if hasattr(self, 'h_ratio'):
            if self.h_ratio > self.left_limit:
                return True
        return False

    def is_down(self):
        """check downward gaze"""
        if hasattr(self, 'v_ratio'):
            if self.v_ratio > self.down_limit:
                return True
        return False

    def is_center(self):
        """check center gaze"""
        if hasattr(self, 'h_ratio') and hasattr(self, 'v_ratio'):
            h_ok = self.right_limit <= self.h_ratio <= self.left_limit
            v_ok = self.v_ratio <= self.down_limit
            if h_ok and v_ok:
                return True
        return False

    def annotated_frame(self):
        """creeaza frame adnotat"""
        if self.frame is None:
            return None

        frame = self.frame.copy()

        if self.pupils_located:
            # culoare directie
            is_center_view = self.is_center()

            if is_center_view:
                color = (0, 255, 0)  # verde centru
            else:
                color = (0, 0, 255)  # rosu alta directie

            # cercuri pupile
            coords_left = self.pupil_left_coords()
            coords_right = self.pupil_right_coords()

            x_left, y_left = coords_left
            x_right, y_right = coords_right

            cv2.circle(frame, (x_left, y_left), 8, color, -1)
            cv2.circle(frame, (x_right, y_right), 8, color, -1)

        return frame, self.h_ratio if hasattr(self, 'h_ratio') else 0.5, self.v_ratio if hasattr(self, 'v_ratio') else 0.5

    def pupil_left_coords(self):
        """coordonate pupila stanga"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)
        return None

    def pupil_right_coords(self):
        """coordonate pupila dreapta"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)
        return None

    def determine_direction(self):
        """determine gaze direction"""
        if not self.pupils_located:
            if self.frames_without_detection > self.max_frames_without_detection:
                return "no_face"
            else:
                return self.last_valid_direction

        # determine direction
        if self.is_right():
            if not self.mirror_image:
                direction = "right"
            else:
                direction = "left"
        elif self.is_left():
            if not self.mirror_image:
                direction = "left"
            else:
                direction = "right"
        elif self.is_down():
            direction = "down"
        else:
            direction = "center"

        # last valid direction
        self.last_valid_direction = direction

        return direction

    def detect_gaze_direction(self, frame):
        """proceseaza frame si detecteaza directia"""
        self.frame_count += 1

        # procesare la n frame-uri
        if self.frame_count % self.process_every_n_frames == 0:
            # update frame si analiza
            self.frame = frame
            self._analyze()

            # determinare directie
            direction = self.determine_direction()

            # salvam frame-ul cu anotari pentru a-l reutiliza
            annotated_result = self.annotated_frame()
            if annotated_result is not None:
                self.last_annotated_frame, self.h_ratio, self.v_ratio = annotated_result
            else:
                self.last_annotated_frame = frame
        else:
            # pentru frame-uri neprocesate, folosim directia salvata
            direction = self.last_valid_direction

        # IMPORTANT: Returnam mereu ultimul frame anotat valid pentru a evita palpairea
        if hasattr(self, 'last_annotated_frame') and self.last_annotated_frame is not None:
            return direction, self.last_annotated_frame, self.h_ratio, self.v_ratio
        else:
            return direction, frame, self.h_ratio, self.v_ratio

    def calculate_head_orientation(self, landmarks):
        """calculeaza orientarea capului"""
        if landmarks is None:
            return 0.5, 0.5

        # pentru orientare orizontala
        nose_point = (landmarks.part(30).x, landmarks.part(30).y)
        left_face = (landmarks.part(0).x, landmarks.part(0).y)
        right_face = (landmarks.part(16).x, landmarks.part(16).y)

        # distante
        left_dist = abs(nose_point[0] - left_face[0])
        right_dist = abs(nose_point[0] - right_face[0])

        # calcul raport orizontal
        total_dist = left_dist + right_dist

        if total_dist == 0:
            h_ratio = 0.5
        else:
            h_ratio = left_dist / total_dist

        # pentru orientare verticala
        eye_left = (landmarks.part(36).x, landmarks.part(36).y)
        eye_right = (landmarks.part(45).x, landmarks.part(45).y)
        mouth = (landmarks.part(57).x, landmarks.part(57).y)

        # punct mediu ochi
        eyes_mid_x = (eye_left[0] + eye_right[0]) // 2
        eyes_mid_y = (eye_left[1] + eye_right[1]) // 2
        eyes_mid = (eyes_mid_x, eyes_mid_y)

        # calcul raport vertical
        face_height = abs(eyes_mid[1] - mouth[1])

        if face_height == 0:
            v_ratio = 0.5
        else:
            nose_to_eyes = abs(nose_point[1] - eyes_mid[1])
            v_ratio = nose_to_eyes / face_height

        return h_ratio, v_ratio

    def calculate_head_orientation_advanced(self, landmarks):
        """Advanced head orientation calculation with 3D pose estimation"""
        if landmarks is None:
            return 0.5, 0.5, 0.0
        
        # Key facial points for 3D pose estimation
        nose_tip = np.array([landmarks.part(30).x, landmarks.part(30).y])
        nose_bridge = np.array([landmarks.part(27).x, landmarks.part(27).y])
        left_eye = np.array([landmarks.part(36).x, landmarks.part(36).y])
        right_eye = np.array([landmarks.part(45).x, landmarks.part(45).y])
        left_mouth = np.array([landmarks.part(48).x, landmarks.part(48).y])
        right_mouth = np.array([landmarks.part(54).x, landmarks.part(54).y])
        chin = np.array([landmarks.part(8).x, landmarks.part(8).y])
        
        # Calculate roll angle (head tilt)
        eye_vector = right_eye - left_eye
        roll_angle = np.arctan2(eye_vector[1], eye_vector[0])
        
        # Calculate yaw (left-right head rotation)
        face_center = (left_eye + right_eye) / 2
        nose_to_center = nose_tip - face_center
        face_width = np.linalg.norm(right_eye - left_eye)
        
        if face_width > 0:
            yaw_ratio = np.dot(nose_to_center, np.array([1, 0])) / face_width
            yaw_ratio = np.clip(yaw_ratio * 2 + 0.5, 0.0, 1.0)  # Normalize to 0-1
        else:
            yaw_ratio = 0.5
        
        # Calculate pitch (up-down head rotation)
        mouth_center = (left_mouth + right_mouth) / 2
        face_height = np.linalg.norm(chin - face_center)
        
        if face_height > 0:
            nose_offset = np.dot(nose_tip - face_center, np.array([0, 1])) / face_height
            pitch_ratio = np.clip(nose_offset + 0.5, 0.0, 1.0)
        else:
            pitch_ratio = 0.5
        
        return yaw_ratio, pitch_ratio, roll_angle

    def calibrate_user(self, calibration_frames, instruction="look_center"):
        """Calibrate system for specific user's gaze patterns"""
        if not hasattr(self, 'user_baseline'):
            self.user_baseline = {
                'center_h': [],
                'center_v': [],
                'left_h': [],
                'right_h': [],
                'up_v': [],
                'down_v': []
            }
        
        calibration_data = []
        
        for frame in calibration_frames:
            self.frame = frame
            if self._analyze() and self.pupils_located:
                h_ratio = self.horizontal_ratio()
                v_ratio = self.vertical_ratio()
                
                if h_ratio is not None and v_ratio is not None:
                    calibration_data.append((h_ratio, v_ratio))
        
        if len(calibration_data) < 5:  # Need minimum samples
            return False
        
        # Store calibration data based on instruction
        h_values = [data[0] for data in calibration_data]
        v_values = [data[1] for data in calibration_data]
        
        if instruction == "look_center":
            self.user_baseline['center_h'].extend(h_values)
            self.user_baseline['center_v'].extend(v_values)
        elif instruction == "look_left":
            self.user_baseline['left_h'].extend(h_values)
        elif instruction == "look_right":
            self.user_baseline['right_h'].extend(h_values)
        elif instruction == "look_up":
            self.user_baseline['up_v'].extend(v_values)
        elif instruction == "look_down":
            self.user_baseline['down_v'].extend(v_values)
        
        # Update personalized thresholds
        self._update_personalized_thresholds()
        return True

    def _update_personalized_thresholds(self):
        """Update detection thresholds based on user calibration"""
        if self.user_baseline['center_h']:
            center_h_mean = np.mean(self.user_baseline['center_h'])
            center_h_std = np.std(self.user_baseline['center_h'])
            
            # Adjust thresholds based on user's natural gaze pattern
            self.left_limit = center_h_mean + 2.5 * center_h_std
            self.right_limit = center_h_mean - 2.5 * center_h_std
            
            # Ensure reasonable bounds
            self.left_limit = min(0.8, max(0.6, self.left_limit))
            self.right_limit = max(0.2, min(0.4, self.right_limit))
        
        if self.user_baseline['center_v']:
            center_v_mean = np.mean(self.user_baseline['center_v'])
            center_v_std = np.std(self.user_baseline['center_v'])
            
            self.down_limit = center_v_mean + 2.5 * center_v_std
            self.down_limit = min(0.8, max(0.6, self.down_limit))