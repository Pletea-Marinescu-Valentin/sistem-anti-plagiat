import cv2

class Pupil(object):
    """detects the iris of the eye and estimates pupil position"""
    def __init__(self, eye_frame):
        self.x = None
        self.y = None

        # iris detection, if we have a valid frame
        if eye_frame is not None and eye_frame.size > 0:
            self.detect_iris(eye_frame)

    def detect_iris(self, eye_frame):
        """detects pupil using the darkest point method in the image"""
        # apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(eye_frame, (7, 7), 0)

        # consider the darkest point in the image
        _, _, min_loc, _ = cv2.minMaxLoc(blurred)

        # set pupil coordinates
        self.x = min_loc[0]
        self.y = min_loc[1]