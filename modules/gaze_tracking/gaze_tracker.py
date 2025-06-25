from __future__ import division
import cv2
import dlib
import json
import numpy as np
import gc
import mediapipe as mp
from .eye import Eye

class GazeTracker(object):
    """Enhanced gaze tracker cu suport pentru MediaPipe și dlib"""

    def __init__(self, mirror_image=True, config_path="config.json", use_mediapipe=True):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.mirror_image = mirror_image
        self.use_mediapipe = use_mediapipe

        # Initialize detection systems
        if self.use_mediapipe:
            self._init_mediapipe()
        else:
            self._init_dlib()

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
        self.left_limit = self.config["detection"]["gaze"]["left_limit"]
        self.right_limit = self.config["detection"]["gaze"]["right_limit"]
        self.down_limit = self.config["detection"]["gaze"]["down_limit"]

        # Image mode for static image processing
        self._image_mode = False

    def _init_mediapipe(self):
        """Inițializează MediaPipe Face Mesh"""
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("MediaPipe Face Mesh inițializat cu succes")
        except Exception as e:
            print(f"Eroare la inițializarea MediaPipe: {e}")
            print("Revin la dlib...")
            self.use_mediapipe = False
            self._init_dlib()

    def _init_dlib(self):
        """Inițializează dlib (fallback)"""
        self._face_detector = dlib.get_frontal_face_detector()
        model_path = "shape_predictor_68_face_landmarks.dat"
        self._predictor = dlib.shape_predictor(model_path)
        print("dlib inițializat cu succes")

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    def reset_all_state(self):
        """Reset complet pentru imagini statice"""
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
            
            # Actualizează MediaPipe pentru imagini statice
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
            self.process_every_n_frames = 2
            
            # Actualizează MediaPipe pentru video
            if self.use_mediapipe:
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )

    def _analyze_mediapipe(self):
        """Detectează față și ochi folosind MediaPipe"""
        if self.frame is None:
            return False

        try:
            # Convertește frame-ul la RGB pentru MediaPipe
            rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            
            # Procesează cu MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                self.frames_without_detection += 1
                return False

            # Obține landmarks-urile feței
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convertește landmarks la coordonate pixel
            height, width = self.frame.shape[:2]
            landmarks_px = []
            
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                landmarks_px.append((x, y))

            # Calculează orientarea capului cu MediaPipe
            new_h_ratio, new_v_ratio = self.calculate_head_orientation_mediapipe(landmarks_px)

            # Aplică smoothing doar pentru video
            if self._image_mode or self.smoothing_factor == 0.0:
                self.h_ratio = new_h_ratio
                self.v_ratio = new_v_ratio
            else:
                self.h_ratio = self.prev_h_ratio * (1 - self.smoothing_factor) + new_h_ratio * self.smoothing_factor
                self.v_ratio = self.prev_v_ratio * (1 - self.smoothing_factor) + new_v_ratio * self.smoothing_factor

            # Actualizează valorile precedente
            self.prev_h_ratio = self.h_ratio
            self.prev_v_ratio = self.v_ratio

            # Convertește frame-ul la grayscale pentru procesarea ochilor
            gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            # Inițializează ochii cu landmarks MediaPipe
            self.eye_left = Eye(gray_frame, landmarks_px, 0, landmark_type="mediapipe")
            self.eye_right = Eye(gray_frame, landmarks_px, 1, landmark_type="mediapipe")

            self.frames_without_detection = 0
            return True
            
        except Exception as e:
            print(f"Eroare în analiza MediaPipe: {e}")
            self.frames_without_detection += 1
            return False

    def _analyze_dlib(self):
        """Detectează față și ochi folosind dlib (metoda originală)"""
        if self.frame is None:
            return False

        frame_copy = self.frame.copy()

        # Redimensionează pentru procesare mai rapidă
        scale_factor = 0.5
        small_frame = cv2.resize(frame_copy, None, fx=scale_factor, fy=scale_factor)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Detectează față
        faces = self._face_detector(gray, 1)

        if len(faces) == 0:
            self.frames_without_detection += 1
            return False

        try:
            # Convertește coordonatele
            face_orig = dlib.rectangle(
                int(faces[0].left() / scale_factor),
                int(faces[0].top() / scale_factor),
                int(faces[0].right() / scale_factor),
                int(faces[0].bottom() / scale_factor)
            )

            gray_orig = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
            landmarks = self._predictor(gray_orig, face_orig)

            # Calculează orientarea capului
            new_h_ratio, new_v_ratio = self.calculate_head_orientation_dlib(landmarks)

            # Aplică smoothing
            if self._image_mode or self.smoothing_factor == 0.0:
                self.h_ratio = new_h_ratio
                self.v_ratio = new_v_ratio
            else:
                self.h_ratio = self.prev_h_ratio * (1 - self.smoothing_factor) + new_h_ratio * self.smoothing_factor
                self.v_ratio = self.prev_v_ratio * (1 - self.smoothing_factor) + new_v_ratio * self.smoothing_factor

            self.prev_h_ratio = self.h_ratio
            self.prev_v_ratio = self.v_ratio

            # Inițializează ochii cu landmarks dlib
            self.eye_left = Eye(gray_orig, landmarks, 0, landmark_type="dlib")
            self.eye_right = Eye(gray_orig, landmarks, 1, landmark_type="dlib")

            self.frames_without_detection = 0
            return True
            
        except Exception as e:
            self.frames_without_detection += 1
            print(f"Eroare în analiza dlib: {e}")
            return False

    def _analyze(self):
        """Analizează frame-ul folosind metoda configurată"""
        if self.use_mediapipe:
            return self._analyze_mediapipe()
        else:
            return self._analyze_dlib()

    def calculate_head_orientation_mediapipe(self, landmarks_px):
        """Calculează orientarea capului folosind landmarks MediaPipe"""
        try:
            # Puncte cheie pentru orientarea capului (MediaPipe indices)
            # Nas: 1, 2, 5, 6
            # Față stânga: 234, 93, 132, 172
            # Față dreapta: 454, 323, 361, 397
            # Sus: 10, 151
            # Jos: 175, 18
            
            nose_tip = landmarks_px[1] if len(landmarks_px) > 1 else (0, 0)
            nose_bridge = landmarks_px[6] if len(landmarks_px) > 6 else nose_tip
            
            # Margini față pentru calcul H
            left_face = landmarks_px[234] if len(landmarks_px) > 234 else (0, 0)
            right_face = landmarks_px[454] if len(landmarks_px) > 454 else (0, 0)
            
            # Puncte pentru calcul V
            forehead = landmarks_px[10] if len(landmarks_px) > 10 else (0, 0)
            chin = landmarks_px[175] if len(landmarks_px) > 175 else (0, 0)

            # Calculează H ratio (orizontal)
            if left_face != (0, 0) and right_face != (0, 0):
                total_width = abs(right_face[0] - left_face[0])
                if total_width > 0:
                    nose_offset = nose_tip[0] - left_face[0]
                    h_ratio = nose_offset / total_width
                else:
                    h_ratio = 0.5
            else:
                h_ratio = 0.5

            # Calculează V ratio (vertical)
            if forehead != (0, 0) and chin != (0, 0):
                total_height = abs(chin[1] - forehead[1])
                if total_height > 0:
                    nose_offset_v = nose_tip[1] - forehead[1]
                    v_ratio = nose_offset_v / total_height
                else:
                    v_ratio = 0.5
            else:
                v_ratio = 0.5

            # Limitează valorile
            h_ratio = max(0.0, min(1.0, h_ratio))
            v_ratio = max(0.0, min(1.0, v_ratio))

            return h_ratio, v_ratio
            
        except Exception as e:
            print(f"Eroare în calculul orientării capului MediaPipe: {e}")
            return 0.5, 0.5

    def calculate_head_orientation_dlib(self, landmarks):
        """Calculează orientarea capului folosind landmarks dlib (metoda originală)"""
        if landmarks is None:
            return 0.5, 0.5

        # Pentru orientare orizontală
        nose_point = (landmarks.part(30).x, landmarks.part(30).y)
        left_face = (landmarks.part(0).x, landmarks.part(0).y)
        right_face = (landmarks.part(16).x, landmarks.part(16).y)

        # Calculează distanțele
        left_dist = abs(nose_point[0] - left_face[0])
        right_dist = abs(nose_point[0] - right_face[0])

        total_dist = left_dist + right_dist
        if total_dist == 0:
            h_ratio = 0.5
        else:
            h_ratio = left_dist / total_dist

        # Pentru orientare verticală
        eye_left = (landmarks.part(36).x, landmarks.part(36).y)
        eye_right = (landmarks.part(45).x, landmarks.part(45).y)
        mouth = (landmarks.part(57).x, landmarks.part(57).y)

        eyes_mid_x = (eye_left[0] + eye_right[0]) // 2
        eyes_mid_y = (eye_left[1] + eye_right[1]) // 2
        eyes_mid = (eyes_mid_x, eyes_mid_y)

        face_height = abs(eyes_mid[1] - mouth[1])

        if face_height == 0:
            v_ratio = 0.5
        else:
            nose_to_eyes = abs(nose_point[1] - eyes_mid[1])
            v_ratio = nose_to_eyes / face_height

        return h_ratio, v_ratio

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
        """Returns horizontal gaze ratio with MediaPipe or dlib support"""
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
        
        # Detectie standard bazata pe pupile
        pupil_down = self.v_ratio > self.down_limit
        
        # Detectie extrema pentru capul foarte aplecat
        extreme_down = self.v_ratio > 0.75
        
        # Daca folosim MediaPipe, putem detecta mai bine capul aplecat
        if self.use_mediapipe and hasattr(self, 'frame') and self.frame is not None:
            head_tilt_down = self._detect_extreme_head_tilt()
            return pupil_down or extreme_down or head_tilt_down
        
        return pupil_down or extreme_down

    def _detect_extreme_head_tilt(self):
        """Detectează înclinarea extremă a capului în jos folosind MediaPipe"""
        try:
            if not self.use_mediapipe or self.frame is None:
                return False
                
            # Folosește landmarks pentru a detecta înclinarea extremă
            rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return False
                
            face_landmarks = results.multi_face_landmarks[0]
            height, width = self.frame.shape[:2]
            
            # Calculează poziția relativă a unor puncte cheie
            # Punct nas (1), fruntea (10), bărbia (152)
            nose = face_landmarks.landmark[1]
            forehead = face_landmarks.landmark[10]
            chin = face_landmarks.landmark[152]
            
            # Convertește la coordonate pixel
            nose_y = nose.y * height
            forehead_y = forehead.y * height
            chin_y = chin.y * height
            
            # Calculează ratios pentru detectia capului aplecat
            if chin_y > forehead_y:  # Verificare de bază
                face_height = chin_y - forehead_y
                if face_height > 0:
                    nose_position = (nose_y - forehead_y) / face_height
                    
                    # Dacă nasul este foarte jos în față (>0.8), capul e probabil aplecat în jos
                    if nose_position > 0.8:
                        return True
                        
            return False
            
        except Exception as e:
            print(f"Eroare în detectia înclinării extreme: {e}")
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
        if self.is_down():  # Verifică mai întâi DOWN (prioritate pentru capul aplecat)
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
        
        # Pentru imagini statice, resetează starea complet
        if self._image_mode:
            self.reset_all_state()
        
        self.frame_count += 1

        # Pentru imagini, procesează la fiecare frame; pentru video, skip frames
        process_frame = True
        if not self._image_mode:
            process_frame = (self.frame_count % self.process_every_n_frames == 0)

        if process_frame:
            # Update frame și analiză
            self.frame = frame
            self._analyze()

            # Determinare direcție cu logică îmbunătățită
            direction = self.determine_direction()

            # Salvează frame-ul cu adnotări
            annotated_result = self.annotated_frame()
            if annotated_result is not None:
                self.last_annotated_frame, self.h_ratio, self.v_ratio = annotated_result
            else:
                self.last_annotated_frame = frame
        else:
            # Pentru frame-uri neprocesate, folosește direcția salvată
            direction = self.last_valid_direction

        # Returnează ultimul frame anotat valid
        if hasattr(self, 'last_annotated_frame') and self.last_annotated_frame is not None:
            return direction, self.last_annotated_frame, self.h_ratio, self.v_ratio
        else:
            return direction, frame, self.h_ratio, self.v_ratio