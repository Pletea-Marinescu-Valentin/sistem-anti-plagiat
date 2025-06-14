import cv2
import numpy as np

class Pupil(object):
    """Enhanced pupil detection using contour analysis and circularity checks"""
    
    def __init__(self, eye_frame):
        self.x = None
        self.y = None
        self.confidence = 0.0  # Detection confidence score
        self.radius = 0  # Estimated pupil radius

        if eye_frame is not None and eye_frame.size > 0:
            self.detect_iris(eye_frame)

    def detect_iris(self, eye_frame):
        """Enhanced pupil detection using multiple validation methods"""
        if eye_frame is None or eye_frame.size == 0:
            return
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(eye_frame, (5, 5), 0)
        
        # Adaptive thresholding to isolate dark regions
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours and select the most circular one
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_contour = None
        best_circularity = 0
        best_center = None
        best_radius = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area - pupil should be reasonable size
            min_area = eye_frame.shape[0] * eye_frame.shape[1] * 0.01  # 1% of eye area
            max_area = eye_frame.shape[0] * eye_frame.shape[1] * 0.3   # 30% of eye area
            
            if area < min_area or area > max_area:
                continue
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Check if this contour is more circular and reasonably sized
            if circularity > best_circularity and circularity > 0.4:
                # Get contour center and radius
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                # Validate position - should be within eye bounds with margin
                margin = 5
                if (margin < x < eye_frame.shape[1] - margin and 
                    margin < y < eye_frame.shape[0] - margin):
                    
                    best_circularity = circularity
                    best_contour = contour
                    best_center = (int(x), int(y))
                    best_radius = int(radius)
        
        if best_contour is not None:
            self.x, self.y = best_center
            self.radius = best_radius
            self.confidence = best_circularity
        else:
            # Fallback to darkest point method
            self._fallback_detection(blurred)
    
    def _fallback_detection(self, blurred_frame):
        """Fallback detection using darkest point method"""
        _, _, min_loc, _ = cv2.minMaxLoc(blurred_frame)
        self.x, self.y = min_loc
        self.confidence = 0.3  # Lower confidence for fallback method