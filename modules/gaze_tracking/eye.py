import math
import numpy as np
import cv2
from .pupil import Pupil


class Eye(object):
    """clasa care izoleaza zona ochiului din imagine si initiaza detectarea pupilei"""

    # punctele de referinta pentru ochii stang si drept (conform modelului facial cu 68 de puncte)
    LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

    def __init__(self, original_frame, landmarks, side):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None
        self.landmark_points = None
        self.blinking = None

        self._analyze(original_frame, landmarks, side)

    @staticmethod
    def _middle_point(p1, p2):
        # coordonatele mijlocului celor doua puncte
        x = int((p1.x + p2.x) / 2)
        y = int((p1.y + p2.y) / 2)
        return (x, y)

    def _isolate(self, frame, landmarks, points):
        """izoleaza ochiul din imaginea fetei folosind o masca"""
        # converteste punctele de referinta in coordonate pentru crearea mastii
        region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
        region = region.astype(np.int32)
        self.landmark_points = region

        # creeaza o masca pentru a izola doar ochiul
        height, width = frame.shape[:2]
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

        # decupeaza zona ochiului cu o margine
        margin = 7
        min_x = max(0, np.min(region[:, 0]) - margin)
        max_x = min(width, np.max(region[:, 0]) + margin)
        min_y = max(0, np.min(region[:, 1]) - margin)
        max_y = min(height, np.max(region[:, 1]) + margin)

        # verifica daca zona decupata este valida
        if min_x >= max_x or min_y >= max_y:
            return

        # salveaza cadrul decupat si coordonatele de origine
        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        # calculeaza centrul ochiului
        if self.frame is not None and self.frame.size > 0:
            height, width = self.frame.shape[:2]
        else:
            height, width = 1, 1
        self.center = (width / 2, height / 2)

    def _blinking_ratio(self, landmarks, points):
        """calculeaza raportul care indica daca ochiul este inchis (clipire)"""
        # obtine coordonatele punctelor importante pentru calculul clipirii
        left = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
        right = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)
        top = self._middle_point(landmarks.part(points[1]), landmarks.part(points[2]))
        bottom = self._middle_point(landmarks.part(points[5]), landmarks.part(points[4]))

        # calculeaza latimea si inaltimea ochiului
        eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
        eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

        # calculeaza raportul latime/inaltime
        try:
            ratio = eye_width / eye_height
        except ZeroDivisionError:
            ratio = None

        return ratio

    def _analyze(self, original_frame, landmarks, side):
        """detecteaza si izoleaza ochiul, apoi initializeaza obiectul Pupil"""
        # determina punctele de referinta in functie de ochiul selectat
        if side == 0:
            points = self.LEFT_EYE_POINTS
        elif side == 1:
            points = self.RIGHT_EYE_POINTS
        else:
            return

        # calculeaza raportul de clipire
        self.blinking = self._blinking_ratio(landmarks, points)

        # izoleaza ochiul
        self._isolate(original_frame, landmarks, points)

        # verifica daca ochiul a fost izolat corect si creeaza pupila
        if self.frame is not None and self.frame.size > 0:
            self.pupil = Pupil(self.frame)