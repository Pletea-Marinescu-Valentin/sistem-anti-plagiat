import cv2
import torch
import threading
import time
from collections import deque
from ultralytics import YOLO
import numpy as np
import logging
import os

class ObjectDetector:
    """Custom trained detector for smartphone and smartwatch detection"""
    
    def __init__(self, config):
        self.config = config
        self.mirror_mode = config.get("camera", {}).get("mirror_image", False)
        
        # Threading for asynchronous processing
        self.detection_thread = None
        self.processing_queue = deque(maxlen=5)
        self.results_queue = deque(maxlen=3)
        self.is_processing = False
        self.last_detection_time = 0
        
        # Performance optimizations
        self.frame_skip_counter = 0
        self.detection_interval = 20  # Process only 1 in 20 frames
        self.resize_factor = 0.6  # Resize to 60% for detection
        self.result_cache_time = 0.3  # Cache results for 0.3s
        
        # Load both models
        self.phone_model = None
        self.smartwatch_model = None
        
        # Load phone model with optimizations
        try:
            if os.path.exists('models/phone_model.pt'):
                self.phone_model = YOLO('models/phone_model.pt')
                self._optimize_model(self.phone_model)
                print("Phone model loaded: models/phone_model.pt")
                logging.info("Phone YOLO model loaded successfully")
            else:
                print("Phone model not found: models/phone_model.pt")
        except Exception as e:
            logging.error(f"Failed to load phone model: {e}")
            print(f"Phone model error: {e}")
        
        # Load smartwatch model with optimizations
        try:
            if os.path.exists('models/smartwatch_model.pt'):
                self.smartwatch_model = YOLO('models/smartwatch_model.pt')
                self._optimize_model(self.smartwatch_model)
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
        self.confidence_threshold = 0.65  # Slightly higher for better performance
        self.last_detections = []
        
        # Start background processing if models available
        if self.phone_model or self.smartwatch_model:
            self._start_background_processing()
    
    def _optimize_model(self, model):
        """Apply optimizations to YOLO model"""
        try:
            if torch.cuda.is_available():
                model.to('cuda')
            else:
                torch.set_num_threads(2)  # Limit CPU threads
        except Exception as e:
            logging.error(f"Model optimization error: {e}")
    
    def _start_background_processing(self):
        """Start background thread for detection processing"""
        self.is_processing = True
        self.detection_thread = threading.Thread(
            target=self._background_detection_loop, 
            daemon=True
        )
        self.detection_thread.start()
    
    def _background_detection_loop(self):
        """Background loop for processing detection frames"""
        while self.is_processing:
            if self.processing_queue:
                try:
                    frame_data = self.processing_queue.popleft()
                    frame, timestamp = frame_data
                    
                    # Process frame in background
                    detections, processing_time = self._process_frame_background(frame)
                    
                    # Store results
                    self.results_queue.append({
                        'detections': detections,
                        'timestamp': timestamp,
                        'processing_time': processing_time
                    })
                    
                except Exception as e:
                    logging.error(f"Background detection error: {e}")
            else:
                time.sleep(0.01)  # Short pause if no frames
    
    def _process_frame_background(self, frame):
        """Process frame in background thread"""
        start_time = time.time()
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
            
            # Measure processing time in milliseconds
            processing_time = (time.time() - start_time) * 1000
            
            # Limit detections for performance
            return detected_objects[:5], processing_time
            
        except Exception as e:
            logging.error(f"Background frame processing error: {e}")
            return [], 0
    
    def set_mirror_state(self, mirror_state):
        """Update mirror state from GUI"""
        self.mirror_mode = mirror_state
        
    def detect_objects(self, frame):
        """Detect both phones and smartwatches with optimizations"""
        if frame is None:
            return []
        
        if self.phone_model is None and self.smartwatch_model is None:
            return []
        
        current_time = time.time()
        
        # Frame skipping optimization
        self.frame_skip_counter += 1
        if self.frame_skip_counter < self.detection_interval:
            # Draw cached detections on current frame
            for obj_type, conf, x1, y1, x2, y2 in self.last_detections:
                frame = self._draw_detection(frame, obj_type, conf, x1, y1, x2, y2)
            return self.last_detections
        
        self.frame_skip_counter = 0
        
        # Check result cache
        if (current_time - self.last_detection_time) < self.result_cache_time:
            # Draw cached detections
            for obj_type, conf, x1, y1, x2, y2 in self.last_detections:
                frame = self._draw_detection(frame, obj_type, conf, x1, y1, x2, y2)
            return self.last_detections
        
        # Add frame to processing queue
        if len(self.processing_queue) < 3:  # Prevent queue overflow
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, None, fx=self.resize_factor, fy=self.resize_factor)
            self.processing_queue.append((small_frame, current_time))
        
        # Get latest results from background processing
        if self.results_queue:
            latest_result = self.results_queue[-1]
            detected_objects = latest_result['detections']
            processing_time = latest_result.get('processing_time', 0)
            
            # Scale coordinates back to original frame size
            scale_factor = 1.0 / self.resize_factor
            scaled_detections = []
            for obj_type, conf, x1, y1, x2, y2 in detected_objects:
                scaled_x1 = int(x1 * scale_factor)
                scaled_y1 = int(y1 * scale_factor)
                scaled_x2 = int(x2 * scale_factor)
                scaled_y2 = int(y2 * scale_factor)
                scaled_detections.append((obj_type, conf, scaled_x1, scaled_y1, scaled_x2, scaled_y2))
            
            self.last_detections = scaled_detections
            self.last_detection_time = processing_time  # Now stores actual processing time in ms
        
        # Draw all detections on current frame
        for obj_type, conf, x1, y1, x2, y2 in self.last_detections:
            frame = self._draw_detection(frame, obj_type, conf, x1, y1, x2, y2)
        
        return self.last_detections
    
    def _detect_with_model(self, frame, model, obj_type):
        """Run detection with specific model - optimized version"""
        detections = []
        
        try:
            # Optimized YOLO inference settings
            results = model(
                frame, 
                conf=self.confidence_threshold, 
                verbose=False,
                stream=False,
                half=True if torch.cuda.is_available() else False,  # FP16 on GPU
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                # Limit processing to first 5 detections for performance
                max_boxes = min(len(boxes), 5)
                
                for i in range(max_boxes):
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
                        
                        # Validate detection size to filter noise
                        width, height = x2 - x1, y2 - y1
                        if cls == 0 and width > 15 and height > 15:  # Minimum size filter
                            detections.append((obj_type, conf, x1, y1, x2, y2))
                            
                    except Exception as box_error:
                        continue  # Skip problematic boxes
        
        except Exception as e:
            logging.error(f"Model {obj_type} detection error: {e}")
        
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
        
        # Draw rectangle with optimized thickness
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background - optimized
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(frame, (x1, y1-label_height-10), (x1 + label_width, y1), color, -1)
        
        # Draw label text - optimized
        cv2.putText(frame, label, (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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
    
    def cleanup(self):
        """Cleanup resources when shutting down"""
        self.is_processing = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.0)
