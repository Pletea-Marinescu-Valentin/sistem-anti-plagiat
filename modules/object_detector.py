import cv2
import torch
from ultralytics import YOLO
import numpy as np
import logging
import os

class ObjectDetector:
    """Custom trained detector for smartphone and smartwatch detection"""
    
    def __init__(self, config):
        self.config = config
        self.mirror_mode = config.get("camera", {}).get("mirror_image", False)
        
        # Load both models
        self.phone_model = None
        self.smartwatch_model = None
        
        # Load phone model
        try:
            if os.path.exists('models/phone_model.pt'):
                self.phone_model = YOLO('models/phone_model.pt')
                print("Phone model loaded: models/phone_model.pt")
                logging.info("Phone YOLO model loaded successfully")
            else:
                print("Phone model not found: models/phone_model.pt")
        except Exception as e:
            logging.error(f"Failed to load phone model: {e}")
            print(f"Phone model error: {e}")
        
        # Load smartwatch model
        try:
            if os.path.exists('models/smartwatch_model.pt'):
                self.smartwatch_model = YOLO('models/smartwatch_model.pt')
                print("Smartwatch model loaded: models/smartwatch_model.pt")
                logging.info("Smartwatch YOLO model loaded successfully")
            else:
                print("Smartwatch model not found: models/smartwatch_model.pt")
        except Exception as e:
            logging.error(f"Failed to load smartwatch model: {e}")
            print(f"Smartwatch model error: {e}")
        
        # Check if at least one model loaded
        if self.phone_model is None and self.smartwatch_model is None:
            print("No models loaded! Please check models/ directory")
            logging.error("No detection models available")
        
        # Detection settings
        self.confidence_threshold = 0.7
        self.last_detections = []
    
    def set_mirror_state(self, mirror_state):
        """Update mirror state from GUI"""
        self.mirror_mode = mirror_state
        
    def detect_objects(self, frame):
        """Detect both phones and smartwatches"""
        if frame is None:
            return []
        
        if self.phone_model is None and self.smartwatch_model is None:
            return []
        
        detected_objects = []
        
        try:
            # Detect phones
            if self.phone_model is not None:
                phone_detections = self._detect_with_model(
                    frame, self.phone_model, 'phone'
                )
                detected_objects.extend(phone_detections)
            
            # Detect smartwatches
            if self.smartwatch_model is not None:
                smartwatch_detections = self._detect_with_model(
                    frame, self.smartwatch_model, 'smartwatch'
                )
                detected_objects.extend(smartwatch_detections)
            
            # Draw all detections
            for obj_type, conf, x1, y1, x2, y2 in detected_objects:
                frame = self._draw_detection(frame, obj_type, conf, x1, y1, x2, y2)
            
            self.last_detections = detected_objects
            return detected_objects
            
        except Exception as e:
            print(f"Detection error: {e}")
            logging.error(f"Detection error: {e}")
            return []
    
    def _detect_with_model(self, frame, model, obj_type):
        """Run detection with specific model"""
        detections = []
        
        try:
            results = model(frame, conf=self.confidence_threshold, verbose=False)
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for i in range(len(boxes)):
                    try:
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
                        
                        # Phone model: class 0 = Mobile-phone
                        # Smartwatch model: class 0 = smartwatch
                        if cls == 0:
                            detections.append((obj_type, conf, x1, y1, x2, y2))
                            
                    except Exception as box_error:
                        print(f"Box processing error: {box_error}")
                        continue
        
        except Exception as e:
            print(f"Model {obj_type} detection error: {e}")
        
        return detections
    
    def _draw_detection(self, frame, obj_type, confidence, x1, y1, x2, y2):
        """Draw detection box with different colors for different objects"""
        
        # Different colors for different objects
        if obj_type == 'phone':
            color = (0, 0, 255)  # Red for phones
            label = f"PHONE {confidence:.2f}"
        elif obj_type == 'smartwatch':
            color = (0, 255, 0)  # Green for smartwatches
            label = f"SMARTWATCH {confidence:.2f}"
        else:
            color = (255, 0, 0)  # Blue for unknown
            label = f"{obj_type.upper()} {confidence:.2f}"
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1-30), (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def get_detection_summary(self):
        """Get summary of current detections"""
        if not self.last_detections:
            return "No objects detected"
        
        phone_count = sum(1 for obj_type, _, _, _, _, _ in self.last_detections if obj_type == 'phone')
        smartwatch_count = sum(1 for obj_type, _, _, _, _, _ in self.last_detections if obj_type == 'smartwatch')
        
        summary = []
        if phone_count > 0:
            summary.append(f"{phone_count} phone(s)")
        if smartwatch_count > 0:
            summary.append(f"{smartwatch_count} smartwatch(es)")
        
        return ", ".join(summary) if summary else "No objects detected"
