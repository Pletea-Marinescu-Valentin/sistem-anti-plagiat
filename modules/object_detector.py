import cv2
import torch
from ultralytics import YOLO
import numpy as np
import logging

class ObjectDetector:
    """Simple and efficient detector for phone and smartwatch only"""
    
    def __init__(self, config):
        self.config = config
        # Store mirror mode from config - INIȚIAL
        self.mirror_mode = config.get("camera", {}).get("mirror_image", False)
        
        # Load single YOLO model
        try:
            self.model = YOLO('yolov8n.pt')  # Lightweight model
            logging.info("YOLO model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {e}")
            self.model = None
        
        # Detection configuration
        self.confidence_threshold = 0.5
        self.phone_class_id = 67  # COCO class ID for cell phone
        
        # Performance optimization
        self.frame_count = 0
        self.process_every_n_frames = 5  # Process every 5th frame
        self.last_detections = []
        
        # Stability tracking
        self.detection_history = []
        self.max_history = 5
    
    def set_mirror_state(self, mirror_state):
        """Update mirror state from GUI"""
        self.mirror_mode = mirror_state
        
    def detect_objects(self, frame):
        """Main detection method - only phones and smartwatches"""
        if self.model is None or frame is None:
            return [], frame
        
        # Frame skipping for performance
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            return self.last_detections, frame
        
        # Reset counter periodically
        if self.frame_count > 1000:
            self.frame_count = 0
        
        try:
            # Resize for faster processing
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = 640
                new_height = int(height * scale)
                resized_frame = cv2.resize(frame, (new_width, new_height))
            else:
                resized_frame = frame
                scale = 1.0
            
            # Run YOLO detection
            results = self.model(resized_frame, conf=0.3, verbose=False)
            
            # Process results
            detected_objects = []
            annotated_frame = frame.copy()
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get detection info
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[class_id]
                        
                        # Scale back to original frame size
                        if scale != 1.0:
                            x1, y1, x2, y2 = [coord / scale for coord in [x1, y1, x2, y2]]
                        
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        
                        # Check if it's a target object
                        obj_type = self._classify_object(class_name, class_id, confidence, x1, y1, x2, y2)
                        
                        if obj_type:
                            detected_objects.append((obj_type, confidence))
                            annotated_frame = self._draw_detection(
                                annotated_frame, obj_type, confidence, x1, y1, x2, y2
                            )
            
            # Apply temporal filtering
            stable_objects = self._apply_temporal_filtering(detected_objects)
            self.last_detections = stable_objects
            
            return stable_objects, annotated_frame
            
        except Exception as e:
            logging.error(f"Object detection error: {e}")
            return self.last_detections, frame
    
    def _classify_object(self, class_name, class_id, confidence, x1, y1, x2, y2):
        """Classify detected object as phone, smartwatch, or ignore"""
        
        # Calculate dimensions
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = height / width if width > 0 else 0
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return None
        
        # Phone detection
        if (class_id == self.phone_class_id or 
            'phone' in class_name.lower() or 
            'mobile' in class_name.lower() or 
            'cell' in class_name.lower()):
            
            # Phone validation: should be rectangular and reasonably sized
            if (width >= 30 and height >= 50 and  # Minimum size
                area >= 1500 and area <= 50000 and  # Area bounds
                aspect_ratio >= 1.2 and aspect_ratio <= 3.0):  # Phone-like aspect ratio
                return 'phone'
        
        # Smartwatch detection (detect as watch/clock and filter by size)
        if ('watch' in class_name.lower() or 
            'clock' in class_name.lower()):
            
            # Smartwatch validation: should be more square and smaller
            if (width >= 20 and width <= 100 and  # Size constraints
                height >= 20 and height <= 100 and
                area >= 400 and area <= 10000 and  # Area bounds
                aspect_ratio >= 0.6 and aspect_ratio <= 1.7):  # More square-ish
                return 'smartwatch'
        
        # Additional heuristic: small rectangular objects might be smartwatches
        if (width >= 25 and width <= 80 and 
            height >= 25 and height <= 80 and
            area >= 600 and area <= 6400 and
            0.7 <= aspect_ratio <= 1.4 and
            confidence > 0.4):
            return 'smartwatch'
        
        return None
    
    def _apply_temporal_filtering(self, current_detections):
        """Apply temporal filtering to reduce false positives"""
        # Add to history
        self.detection_history.append(current_detections)
        
        # Keep only recent history
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
        
        # Count object types in recent frames
        object_counts = {}
        for frame_detections in self.detection_history:
            for obj_type, confidence in frame_detections:
                if obj_type not in object_counts:
                    object_counts[obj_type] = 0
                object_counts[obj_type] += 1
        
        # Only keep objects detected in multiple frames
        stable_detections = []
        for obj_type, confidence in current_detections:
            if object_counts.get(obj_type, 0) >= 2:  # Seen in at least 2 frames
                stable_detections.append((obj_type, confidence))
        
        return stable_detections
    
    def _draw_detection(self, frame, obj_type, confidence, x1, y1, x2, y2):
        """Draw bounding box and label with correct mirror logic"""
        # Colors: Red for phone, Blue for smartwatch
        if obj_type == 'phone':
            color = (0, 0, 255)  # Red
            label = f"PHONE: {confidence:.2f}"
        else:  # smartwatch
            color = (255, 0, 0)  # Blue
            label = f"SMARTWATCH: {confidence:.2f}"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Text configuration
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Label position
        label_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
        
        if self.mirror_mode:
            text_canvas = np.zeros((text_height + 15, text_width + 15, 3), dtype=np.uint8)
            
            cv2.putText(text_canvas, label, (5, text_height + 5), 
                        font, font_scale, (255, 255, 255), thickness)
            
            text_canvas_flipped = cv2.flip(text_canvas, 1)
            
            text_x = max(0, min(x1, frame.shape[1] - text_canvas_flipped.shape[1]))
            text_y_start = max(0, min(label_y - text_height - 10, frame.shape[0] - text_canvas_flipped.shape[0]))
            
            cv2.rectangle(frame, 
                        (text_x, text_y_start),
                        (text_x + text_canvas_flipped.shape[1], text_y_start + text_canvas_flipped.shape[0]),
                        color, -1)
            
            mask = text_canvas_flipped > 0
            frame[text_y_start:text_y_start + text_canvas_flipped.shape[0], 
                text_x:text_x + text_canvas_flipped.shape[1]][mask] = text_canvas_flipped[mask]
            
        else:
            cv2.rectangle(frame, 
                        (x1, label_y - text_height - 5),
                        (x1 + text_width + 10, label_y + baseline + 5),
                        color, -1)
            
            cv2.putText(frame, label, (x1 + 5, label_y), font, font_scale, (255, 255, 255), thickness)
        
        return frame
