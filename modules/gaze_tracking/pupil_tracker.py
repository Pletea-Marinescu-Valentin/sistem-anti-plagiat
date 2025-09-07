import numpy as np
import cv2

class KalmanPupilTracker:
    """Kalman Filter for smooth pupil position tracking"""
    
    def __init__(self):
        # Initialize Kalman Filter for 2D position tracking
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 states (x, y, vx, vy), 2 measurements (x, y)
        
        # State transition matrix (constant velocity model)
        # [x, y, vx, vy] -> [x+vx*dt, y+vy*dt, vx, vy]
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix - we measure x and y position
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance - how much we trust our motion model
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        
        # Measurement noise covariance - how much we trust our measurements
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        
        # Error covariance matrix
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
        
        self.initialized = False
        self.last_valid_position = None
        self.confidence_threshold = 0.3
        self.max_velocity = 50  # Maximum expected pupil velocity (pixels/frame)
        
    def initialize(self, x, y):
        """Initialize tracker with first valid pupil position"""
        self.kalman.statePre = np.array([x, y, 0, 0], dtype=np.float32)
        self.kalman.statePost = np.array([x, y, 0, 0], dtype=np.float32)
        self.last_valid_position = (x, y)
        self.initialized = True
        
    def predict(self):
        """Predict next pupil position"""
        if not self.initialized:
            return None
            
        prediction = self.kalman.predict()
        predicted_x, predicted_y = prediction[0], prediction[1]
        predicted_vx, predicted_vy = prediction[2], prediction[3]
        
        # Limit velocity to reasonable bounds
        velocity_magnitude = np.sqrt(predicted_vx**2 + predicted_vy**2)
        if velocity_magnitude > self.max_velocity:
            scale = self.max_velocity / velocity_magnitude
            predicted_vx *= scale
            predicted_vy *= scale
            
        return (float(predicted_x), float(predicted_y))
        
    def update(self, measured_x, measured_y, confidence=1.0):
        """Update tracker with new measurement"""
        if not self.initialized:
            self.initialize(measured_x, measured_y)
            return (measured_x, measured_y)
            
        # Get prediction first
        prediction = self.predict()
        
        if confidence < self.confidence_threshold:
            # Low confidence measurement
            if prediction:
                return prediction
            else:
                return self.last_valid_position
        
        # High confidence measurement - update Kalman filter
        measurement = np.array([[measured_x], [measured_y]], dtype=np.float32)
        
        # Adjust measurement noise based on confidence
        measurement_noise = 0.1 / confidence
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        
        # Update with measurement
        self.kalman.correct(measurement)
        
        # Get corrected state
        state = self.kalman.statePost
        corrected_x, corrected_y = state[0], state[1]
        
        self.last_valid_position = (float(corrected_x), float(corrected_y))
        return self.last_valid_position
        
    def get_velocity(self):
        """Get current estimated velocity"""
        if not self.initialized:
            return (0, 0)
            
        state = self.kalman.statePost
        return (float(state[2]), float(state[3]))
        
    def is_outlier(self, x, y, threshold=30):
        """Check if measurement is an outlier"""
        if not self.initialized:
            return False
            
        prediction = self.predict()
        if prediction is None:
            return False
            
        pred_x, pred_y = prediction
        distance = np.sqrt((x - pred_x)**2 + (y - pred_y)**2)
        
        return distance > threshold