import math
import numpy as np
import cv2
from .pupil import Pupil
from .pupil_tracker import KalmanPupilTracker


class Eye(object):
    """
    Clasa Eye îmbunătățită care poate funcționa cu landmarks MediaPipe sau dlib
    """

    # Indicii pentru ochii din MediaPipe (468 puncte)
    LEFT_EYE_MP_INDICES = [
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
    ]
    
    RIGHT_EYE_MP_INDICES = [
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
    ]
    
    # Punctele pentru iris (MediaPipe)
    LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
    RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

    # Indicii dlib originali (pentru compatibilitate)
    LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

    def __init__(self, original_frame, landmarks, side, landmark_type="dlib"):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None
        self.landmark_points = None
        self.blinking = None
        self.landmark_type = landmark_type  # "dlib" sau "mediapipe"
        
        # Add Kalman tracker for pupil
        self.pupil_tracker = KalmanPupilTracker()
        self.raw_pupil_position = None
        
        self._analyze(original_frame, landmarks, side)

    @staticmethod
    def _middle_point(p1, p2):
        """Calculează punctul de mijloc între două puncte"""
        if hasattr(p1, 'x'):  # dlib point
            x = int((p1.x + p2.x) / 2)
            y = int((p1.y + p2.y) / 2)
        else:  # tuplu (x, y)
            x = int((p1[0] + p2[0]) / 2)
            y = int((p1[1] + p2[1]) / 2)
        return (x, y)

    def _isolate_mediapipe(self, frame, landmarks_px, eye_indices):
        """Izolează ochiul folosind landmarks MediaPipe"""
        try:
            # Obține punctele ochiului
            eye_points = []
            for idx in eye_indices:
                if idx < len(landmarks_px):
                    eye_points.append(landmarks_px[idx])
                else:
                    # Fallback dacă indexul nu există
                    eye_points.append((0, 0))
            
            if len(eye_points) < 6:
                return False
            
            # Convertește la numpy array pentru procesare
            region = np.array(eye_points, dtype=np.int32)
            self.landmark_points = region

            # Creează mască pentru izolarea ochiului
            height, width = frame.shape[:2]
            mask = np.zeros((height, width), np.uint8)
            
            # Folosește convex hull pentru o formă mai bună a ochiului
            hull = cv2.convexHull(region)
            cv2.fillPoly(mask, [hull], 255)
            
            # Aplică masca
            eye = cv2.bitwise_and(frame, frame, mask=mask)

            # Calculează bounding box cu marjă
            x_coords = [p[0] for p in eye_points]
            y_coords = [p[1] for p in eye_points]
            
            min_x = max(0, min(x_coords) - 10)
            max_x = min(width, max(x_coords) + 10)
            min_y = max(0, min(y_coords) - 10)
            max_y = min(height, max(y_coords) + 10)

            # Verifică dacă regiunea este validă
            if min_x >= max_x or min_y >= max_y:
                return False

            # Salvează frame-ul croppat și originea
            self.frame = eye[min_y:max_y, min_x:max_x]
            self.origin = (min_x, min_y)

            # Calculează centrul ochiului
            if self.frame is not None and self.frame.size > 0:
                height, width = self.frame.shape[:2]
                self.center = (width / 2, height / 2)
            else:
                self.center = (1, 1)
            
            return True
            
        except Exception as e:
            print(f"Eroare la izolarea ochiului MediaPipe: {e}")
            return False

    def _isolate_dlib(self, frame, landmarks, points):
        """Izolarea originală cu dlib (păstrată pentru compatibilitate)"""
        try:
            region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
            region = region.astype(np.int32)
            self.landmark_points = region

            height, width = frame.shape[:2]
            black_frame = np.zeros((height, width), np.uint8)
            mask = np.full((height, width), 255, np.uint8)
            cv2.fillPoly(mask, [region], (0, 0, 0))
            eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

            margin = 7
            min_x = max(0, np.min(region[:, 0]) - margin)
            max_x = min(width, np.max(region[:, 0]) + margin)
            min_y = max(0, np.min(region[:, 1]) - margin)
            max_y = min(height, np.max(region[:, 1]) + margin)

            if min_x >= max_x or min_y >= max_y:
                return False

            self.frame = eye[min_y:max_y, min_x:max_x]
            self.origin = (min_x, min_y)

            if self.frame is not None and self.frame.size > 0:
                height, width = self.frame.shape[:2]
                self.center = (width / 2, height / 2)
            else:
                self.center = (1, 1)
                
            return True
            
        except Exception as e:
            print(f"Eroare la izolarea ochiului dlib: {e}")
            return False

    def _blinking_ratio_mediapipe(self, landmarks_px, eye_indices):
        """Calculează ratio-ul de clipire pentru MediaPipe"""
        try:
            if len(eye_indices) < 6:
                return None
                
            # Folosește punctele extreme ale ochiului
            eye_points = [landmarks_px[i] for i in eye_indices if i < len(landmarks_px)]
            
            if len(eye_points) < 6:
                return None
            
            # Găsește punctele extreme
            leftmost = min(eye_points, key=lambda p: p[0])
            rightmost = max(eye_points, key=lambda p: p[0])
            topmost = min(eye_points, key=lambda p: p[1])
            bottommost = max(eye_points, key=lambda p: p[1])
            
            # Calculează dimensiunile
            eye_width = math.hypot(rightmost[0] - leftmost[0], rightmost[1] - leftmost[1])
            eye_height = math.hypot(topmost[0] - bottommost[0], topmost[1] - bottommost[1])

            if eye_height == 0:
                return None
                
            ratio = eye_width / eye_height
            return ratio
            
        except Exception as e:
            print(f"Eroare la calculul ratio-ului de clipire MediaPipe: {e}")
            return None

    def _blinking_ratio_dlib(self, landmarks, points):
        """Calculează ratio-ul de clipire pentru dlib (original)"""
        try:
            left = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
            right = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)
            top = self._middle_point(landmarks.part(points[1]), landmarks.part(points[2]))
            bottom = self._middle_point(landmarks.part(points[5]), landmarks.part(points[4]))

            eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
            eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

            try:
                ratio = eye_width / eye_height
            except ZeroDivisionError:
                ratio = None

            return ratio
        except Exception as e:
            print(f"Eroare la calculul ratio-ului de clipire dlib: {e}")
            return None

    def _analyze(self, original_frame, landmarks, side):
        """Detectează și izolează ochiul, apoi inițializează obiectul Pupil"""
        
        if self.landmark_type == "mediapipe":
            # Folosește landmarks MediaPipe
            if side == 0:  # ochi stâng
                eye_indices = self.LEFT_EYE_MP_INDICES
            elif side == 1:  # ochi drept
                eye_indices = self.RIGHT_EYE_MP_INDICES
            else:
                return

            # Calculează blinking ratio
            self.blinking = self._blinking_ratio_mediapipe(landmarks, eye_indices)

            # Izolează ochiul
            if not self._isolate_mediapipe(original_frame, landmarks, eye_indices):
                return
                
        else:
            # Folosește landmarks dlib (original)
            if side == 0:
                points = self.LEFT_EYE_POINTS
            elif side == 1:
                points = self.RIGHT_EYE_POINTS
            else:
                return

            self.blinking = self._blinking_ratio_dlib(landmarks, points)
            
            if not self._isolate_dlib(original_frame, landmarks, points):
                return

        # Creează pupila dacă ochiul a fost izolat cu succes
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