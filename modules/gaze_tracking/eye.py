import math
import numpy as np
import cv2
from .pupil import Pupil


class Eye(object):
    """class that isolates the eye region from the image and initiates pupil detection"""

    # reference points for left and right eyes (according to 68-point facial model)
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
        # coordinates of the midpoint between two points
        x = int((p1.x + p2.x) / 2)
        y = int((p1.y + p2.y) / 2)
        return (x, y)

    def _isolate(self, frame, landmarks, points):
        """isolates the eye from the face image using a mask"""
        # convert reference points to coordinates for mask creation
        region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
        region = region.astype(np.int32)
        self.landmark_points = region

        # create a mask to isolate only the eye
        height, width = frame.shape[:2]
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

        # crop eye region with margin
        margin = 7
        min_x = max(0, np.min(region[:, 0]) - margin)
        max_x = min(width, np.max(region[:, 0]) + margin)
        min_y = max(0, np.min(region[:, 1]) - margin)
        max_y = min(height, np.max(region[:, 1]) + margin)

        # check if cropped region is valid
        if min_x >= max_x or min_y >= max_y:
            return

        # save cropped frame and origin coordinates
        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        # calculate eye center
        if self.frame is not None and self.frame.size > 0:
            height, width = self.frame.shape[:2]
        else:
            height, width = 1, 1
        self.center = (width / 2, height / 2)

    def _blinking_ratio(self, landmarks, points):
        """calculates ratio indicating if eye is closed (blinking)"""
        # get coordinates of important points for blink calculation
        left = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
        right = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)
        top = self._middle_point(landmarks.part(points[1]), landmarks.part(points[2]))
        bottom = self._middle_point(landmarks.part(points[5]), landmarks.part(points[4]))

        # calculate eye width and height
        eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
        eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

        # calculate width/height ratio
        try:
            ratio = eye_width / eye_height
        except ZeroDivisionError:
            ratio = None

        return ratio

    def _analyze(self, original_frame, landmarks, side):
        """detects and isolates eye, then initializes Pupil object"""
        # determine reference points based on selected eye
        if side == 0:
            points = self.LEFT_EYE_POINTS
        elif side == 1:
            points = self.RIGHT_EYE_POINTS
        else:
            return

        # calculate blinking ratio
        self.blinking = self._blinking_ratio(landmarks, points)

        # isolate eye
        self._isolate(original_frame, landmarks, points)

        # check if eye was isolated correctly and create pupil
        if self.frame is not None and self.frame.size > 0:
            self.pupil = Pupil(self.frame)