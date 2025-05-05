from __future__ import division
import cv2
import dlib
import json
from .eye import Eye

class GazeTracker(object):
    """aceasta clasa urmareste directia privirii utilizatorului si ofera informatii despre pozitia ochilor si a pupilelor"""

    def __init__(self, mirror_image=True, config_path="config.json"):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.mirror_image = mirror_image

        # detector fete
        self._face_detector = dlib.get_frontal_face_detector()

        # predictor repere faciale
        model_path = "shape_predictor_68_face_landmarks.dat"
        self._predictor = dlib.shape_predictor(model_path)

        self.frames_without_detection = 0
        self.max_frames_without_detection = 10
        self.last_valid_direction = "center"

        # optimizare procesare
        self.process_every_n_frames = 2
        self.frame_count = 0

        # valori pentru filtrare
        self.prev_h_ratio = 0.5
        self.prev_v_ratio = 0.5
        self.h_ratio = 0.5
        self.v_ratio = 0.5

        # factor filtrare
        self.smoothing_factor = 0.3

        # incarcare configuratie
        self.config = self.load_config(config_path)
        self.left_limit = self.config["detection"]["gaze"]["left_limit"]
        self.right_limit = self.config["detection"]["gaze"]["right_limit"]
        self.down_limit = self.config["detection"]["gaze"]["down_limit"]

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    @property
    def pupils_located(self):
        """verifica daca pupilele sunt localizate"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyze(self):
        """detecteaza fata si ochii"""
        if self.frame is None:
            return False

        frame_copy = self.frame.copy()

        # resize pentru procesare mai rapida
        scale_factor = 0.5
        small_frame = cv2.resize(frame_copy, None, fx=scale_factor, fy=scale_factor)

        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # detectare fete
        faces = self._face_detector(gray, 1)

        if len(faces) == 0:
            self.frames_without_detection += 1
            return False

        try:
            # convertire coordonate
            face_orig = dlib.rectangle(
                int(faces[0].left() / scale_factor),
                int(faces[0].top() / scale_factor),
                int(faces[0].right() / scale_factor),
                int(faces[0].bottom() / scale_factor)
            )

            # conversie frame original
            gray_orig = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)

            landmarks = self._predictor(gray_orig, face_orig)

            # calcul orientare cap
            new_h_ratio, new_v_ratio = self.calculate_head_orientation(landmarks)

            # filtrare valori
            self.h_ratio = self.prev_h_ratio * (1 - self.smoothing_factor) + new_h_ratio * self.smoothing_factor
            self.v_ratio = self.prev_v_ratio * (1 - self.smoothing_factor) + new_v_ratio * self.smoothing_factor

            # update valori
            self.prev_h_ratio = self.h_ratio
            self.prev_v_ratio = self.v_ratio

            # initializare ochi
            self.eye_left = Eye(gray_orig, landmarks, 0)
            self.eye_right = Eye(gray_orig, landmarks, 1)

            self.frames_without_detection = 0
            return True
        except Exception as e:
            self.frames_without_detection += 1
            print(f"eroare analiza fata: {e}")
            return False

    def refresh(self, frame):
        """update frame si analiza"""
        self.frame = frame
        return self._analyze()
    
    def horizontal_ratio(self):
        """calculeaza raport orizontal"""
        if not self.pupils_located:
            return 0.5

        # calcul pozitie pupile
        pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
        pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
        pupil_ratio = (pupil_left + pupil_right) / 2

        # ajustare factor pozitie cap
        if self.eye_left.origin and self.eye_right.origin:
            left_center_x = self.eye_left.origin[0] + self.eye_left.center[0]
            right_center_x = self.eye_right.origin[0] + self.eye_right.center[0]

            # distanta centre ochi
            eye_distance = right_center_x - left_center_x

            # width frame
            if self.frame is not None:
                frame_width = self.frame.shape[1]
            else:
                frame_width = 640

            # normalizare distanta
            eye_position_factor = eye_distance / (frame_width * 0.3)

            # ajustare ratio
            adjusted_ratio = pupil_ratio

            # cap intors stanga
            if eye_position_factor < 0.8:
                adjusted_ratio = max(0.6, pupil_ratio)
            # cap intors dreapta
            elif eye_position_factor > 1.2:
                adjusted_ratio = min(0.4, pupil_ratio)

            return adjusted_ratio

        return pupil_ratio

    def vertical_ratio(self):
        """calculeaza raport vertical"""
        if not self.pupils_located:
            return 0.5

        # calcul pozitie pupile
        pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
        pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
        pupil_ratio = (pupil_left + pupil_right) / 2

        # ajustare factor pozitie cap
        if self.eye_left.origin and self.eye_right.origin:
            eye_center_y = (self.eye_left.origin[1] + self.eye_left.center[1] +
                            self.eye_right.origin[1] + self.eye_right.center[1]) / 2
            mouth_y = self.eye_left.origin[1] + self.eye_left.center[1] + 50  # aproximare pentru gura

            # distanta ochi-gura
            face_height = abs(eye_center_y - mouth_y)

            # normalizare distanta
            if face_height > 0:
                eye_position_factor = abs(self.eye_left.origin[1] - eye_center_y) / face_height
            else:
                eye_position_factor = 0.5

            # ajustare ratio
            adjusted_ratio = pupil_ratio

            # cap inclinat in jos
            if eye_position_factor > 0.6:
                adjusted_ratio = min(0.4, pupil_ratio)
            # cap inclinat in sus
            elif eye_position_factor < 0.4:
                adjusted_ratio = max(0.6, pupil_ratio)

            return adjusted_ratio

        return pupil_ratio

    def is_right(self):
        """verifica privire dreapta"""
        if hasattr(self, 'h_ratio'):
            if self.h_ratio < self.right_limit:
                return True
        return False

    def is_left(self):
        """verifica privire stanga"""
        if hasattr(self, 'h_ratio'):
            if self.h_ratio > self.left_limit:
                return True
        return False

    def is_down(self):
        """verifica privire jos"""
        if hasattr(self, 'v_ratio'):
            if self.v_ratio > self.down_limit:
                return True
        return False

    def is_center(self):
        """verifica privire centru"""
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
        """determina directia privirii"""
        if not self.pupils_located:
            if self.frames_without_detection > self.max_frames_without_detection:
                return "no_face"
            else:
                return self.last_valid_direction

        # determinare directie
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

        # ultima directie valida
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