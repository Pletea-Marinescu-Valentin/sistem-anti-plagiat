import cv2

class Pupil(object):
   """detecteaza irisul ochiului si estimeaza pozitia pupilei"""
   def __init__(self, eye_frame):
       self.x = None
       self.y = None

       # detectare iris, daca avem un cadru valid
       if eye_frame is not None and eye_frame.size > 0:
           self.detect_iris(eye_frame)

   def detect_iris(self, eye_frame):
       """detecteaza pupila folosind metoda celui mai intunecat punct din imagine"""
       # aplicam blur Gaussian pentru a reduce zgomotul
       blurred = cv2.GaussianBlur(eye_frame, (7, 7), 0)

       # consideram cel mai intunecat punct din imagine
       _, _, min_loc, _ = cv2.minMaxLoc(blurred)

       # setam coordonatele pupilei
       self.x = min_loc[0]
       self.y = min_loc[1]