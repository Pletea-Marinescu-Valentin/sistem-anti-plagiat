import cv2
import torch
from ultralytics import YOLO
import numpy as np
import logging
import os

class ObjectDetector:
    """Custom trained detector for smartphone detection"""
    
    def __init__(self, config):
        self.config = config
        self.mirror_mode = config.get("camera", {}).get("mirror_image", False)
        
        # Load custom YOLO model
        try:
            self.model = YOLO('models/best.pt')
            print("Using custom trained model: models/best.pt")
            logging.info("Custom YOLO model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load custom model: {e}")
            self.model = None
        
        # Simplificat
        self.confidence_threshold = 0.7
        self.last_detections = []
    
    def set_mirror_state(self, mirror_state):
        """Update mirror state from GUI"""
        self.mirror_mode = mirror_state
        
    def detect_objects(self, frame):
        """SUPER SIMPLU - fără complicații"""
        if self.model is None or frame is None:
            return []
        
        try:
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            detected_objects = []
            
            # Process results
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for i in range(len(boxes)):
                    try:
                        # Get box data
                        box = boxes[i]
                        
                        # Get coordinates
                        xyxy = box.xyxy.cpu().numpy()[0]
                        if len(xyxy) != 4:
                            continue
                            
                        x1, y1, x2, y2 = map(int, xyxy)
                        
                        # Get confidence
                        conf = float(box.conf.cpu().numpy()[0])
                        
                        # Get class
                        cls = int(box.cls.cpu().numpy()[0])
                        
                        if cls == 0:
                            detected_objects.append(('phone', conf, x1, y1, x2, y2))                            
                            frame = self._draw_detection(frame, 'phone', conf, x1, y1, x2, y2)
                            
                    except Exception as box_error:
                        print(f"Box error: {box_error}")
                        continue
            
            self.last_detections = detected_objects
            return detected_objects
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def _draw_detection(self, frame, obj_type, confidence, x1, y1, x2, y2):
        """Draw detection box"""
        color = (0, 0, 255)
        label = f"PHONE {confidence:.2f}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
