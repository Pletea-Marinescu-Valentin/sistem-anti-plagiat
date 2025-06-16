import os
import cv2
import json
import time
import hashlib
from datetime import datetime
from modules.object_detector import ObjectDetector

class DeviceDetectionAnalyzer:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.input_dir = "device_test_images"
        self.output_dir = "device_analysis_results"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize object detector
        self.object_detector = ObjectDetector(self.config)

    def load_config(self):
        """Load configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {"camera": {"mirror_image": False}}

    def get_image_files(self):
        """Get all test images"""
        if not os.path.exists(self.input_dir):
            print(f"Input directory not found: {self.input_dir}")
            return []
        
        image_files = []
        for file in sorted(os.listdir(self.input_dir)):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                full_path = os.path.join(self.input_dir, file)
                if os.path.isfile(full_path):
                    image_files.append({
                        'filename': file,
                        'full_path': full_path
                    })
        
        return image_files

    def extract_expected_type(self, filename):
        """Extract expected device type from filename"""
        filename_lower = filename.lower()
        
        if filename_lower.startswith('phone_'):
            return 'phone'
        elif filename_lower.startswith('smartwatch_'):
            return 'smartwatch'
        
        return None

    def analyze_single_image(self, image_info):
        """Analyze single image with object_detector.py"""
        filename = image_info['filename']
        print(f"Analyzing: {filename}")
        
        # Load image
        frame = cv2.imread(image_info['full_path'])
        if frame is None:
            print(f"Could not load image")
            return None
        
        # Get expected device type
        expected_type = self.extract_expected_type(filename)
        if not expected_type:
            print(f"Could not determine expected type from filename")
            return None
        
        print(f"Expected: {expected_type}")
        
        # Run object detection - FIX PENTRU UNPACKING
        detected_objects = self.object_detector.detect_objects(frame)
        
        # Creează frame annotat manual pentru analyzer
        annotated_frame = frame.copy()
        
        # Desenează manual box-urile pentru analyzer
        for obj in detected_objects:
            if len(obj) >= 6:  # ('phone', conf, x1, y1, x2, y2)
                obj_type, confidence, x1, y1, x2, y2 = obj
                
                # Desenează box
                color = (0, 0, 255) if obj_type == 'phone' else (255, 0, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Desenează label
                label = f"{obj_type.upper()} {confidence:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        print(f"Raw detection results: {detected_objects}")  # Debug info
        
        # Process detection results
        detected_phones = [obj for obj in detected_objects if obj[0] == 'phone']
        detected_watches = [obj for obj in detected_objects if obj[0] == 'smartwatch']
        
        # Determine if detection is correct
        if expected_type == 'phone':
            is_correct = len(detected_phones) > 0
            detected_type = 'phone' if len(detected_phones) > 0 else 'none'
            confidence = max([obj[1] for obj in detected_phones]) if detected_phones else 0.0
        else:  # smartwatch
            is_correct = len(detected_watches) > 0
            detected_type = 'smartwatch' if len(detected_watches) > 0 else 'none'
            confidence = max([obj[1] for obj in detected_watches]) if detected_watches else 0.0
        
        # Output results
        status = "CORRECT" if is_correct else "WRONG"
        print(f"All objects detected: {len(detected_objects)}")
        print(f"Phones detected: {len(detected_phones)}")
        print(f"Smartwatches detected: {len(detected_watches)}")
        print(f"Result: {status} (confidence: {confidence:.3f})")
        
        # Create detailed annotated image
        final_annotated = self.create_detailed_annotation(
            annotated_frame, filename, expected_type, detected_type, 
            detected_objects, is_correct, confidence
        )
        
        return {
            'filename': filename,
            'expected_type': expected_type,
            'detected_type': detected_type,
            'detected_objects': detected_objects,
            'is_correct': is_correct,
            'confidence': confidence,
            'annotated_frame': final_annotated
        }

    def create_detailed_annotation(self, annotated_frame, filename, expected_type, 
                                    detected_type, detected_objects, is_correct, confidence):
        """Create detailed annotated image"""
        # Start with annotated frame from ObjectDetector (already has bounding boxes)
        frame = annotated_frame.copy()
        
        # Colors for results
        result_color = (0, 255, 0) if is_correct else (0, 0, 255)
        phone_color = (0, 0, 255)      # Red
        watch_color = (255, 0, 0)      # Blue
        
        # Main result
        expected_display = expected_type.upper()
        detected_display = detected_type.upper()
        
        cv2.putText(frame, f"EXPECTED: {expected_display}", 
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                    phone_color if expected_type == 'phone' else watch_color, 2)
        
        cv2.putText(frame, f"DETECTED: {detected_display}", 
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, result_color, 2)
        
        # Status
        status = "CORRECT" if is_correct else "WRONG"
        cv2.putText(frame, status, 
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, result_color, 3)
        
        # Confidence
        cv2.putText(frame, f"Confidence: {confidence:.3f}", 
                    (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Detection summary
        phones_detected = len([obj for obj in detected_objects if obj[0] == 'phone'])
        watches_detected = len([obj for obj in detected_objects if obj[0] == 'smartwatch'])
        total_objects = len(detected_objects)
        
        cv2.putText(frame, f"Total: {total_objects} | Phones: {phones_detected} | Watches: {watches_detected}", 
                    (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # File info
        cv2.putText(frame, f"File: {filename}", 
                    (10, frame.shape[0] - 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Hash for verification
        frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        image_hash = hashlib.md5(frame_bytes).hexdigest()[:8]
        cv2.putText(frame, f"Hash: {image_hash}", 
                    (10, frame.shape[0] - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Analyzed: {timestamp}", 
                    (10, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        
        return frame

    def analyze_all_images(self):
        """Analyze all device test images"""
        print("Device Detection Analysis using object_detector.py")
        
        image_files = self.get_image_files()
        if not image_files:
            print("No images found!")
            return
        
        results = []
        phone_stats = {'total': 0, 'correct': 0}
        watch_stats = {'total': 0, 'correct': 0}
        
        start_time = time.time()
        
        for i, image_info in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing...")
            
            result = self.analyze_single_image(image_info)
            if result:
                results.append(result)
                
                # Update statistics
                if result['expected_type'] == 'phone':
                    phone_stats['total'] += 1
                    if result['is_correct']:
                        phone_stats['correct'] += 1
                elif result['expected_type'] == 'smartwatch':
                    watch_stats['total'] += 1
                    if result['is_correct']:
                        watch_stats['correct'] += 1
                
                # Save annotated image
                output_filename = f"analyzed_{result['filename']}"
                output_path = os.path.join(self.output_dir, output_filename)
                cv2.imwrite(output_path, result['annotated_frame'])
                print(f"Saved: {output_filename}")
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        total_correct = phone_stats['correct'] + watch_stats['correct']
        total_images = phone_stats['total'] + watch_stats['total']
        overall_accuracy = (total_correct / total_images * 100) if total_images > 0 else 0
        
        phone_accuracy = (phone_stats['correct'] / phone_stats['total'] * 100) if phone_stats['total'] > 0 else 0
        watch_accuracy = (watch_stats['correct'] / watch_stats['total'] * 100) if watch_stats['total'] > 0 else 0
        
        # Display results
        print(f"\n" + "="*60)
        print(f"DEVICE DETECTION ANALYSIS COMPLETE!")
        print(f"="*60)
        print(f"Total images: {total_images}")
        print(f"Correct detections: {total_correct}")
        print(f"Overall accuracy: {overall_accuracy:.1f}%")
        print(f"Analysis time: {total_time:.1f} seconds")
        
        print(f"\nPhone Detection:")
        print(f"  Images tested: {phone_stats['total']}")
        print(f"  Correct detections: {phone_stats['correct']}")
        print(f"  Accuracy: {phone_accuracy:.1f}%")
        
        print(f"\nSmartwatch Detection:")
        print(f"  Images tested: {watch_stats['total']}")
        print(f"  Correct detections: {watch_stats['correct']}")
        print(f"  Accuracy: {watch_accuracy:.1f}%")
        
        print(f"\nResults saved to: {os.path.abspath(self.output_dir)}")
        
        # Show detailed results
        print(f"\nDetailed Results:")
        for result in results:
            status_icon = "✓" if result['is_correct'] else "✗"
            print(f"{result['filename']} -> Expected: {result['expected_type']}, "
                    f"Detected: {result['detected_type']} (conf: {result['confidence']:.3f})")

def main():
    print("Device Detection Analyzer using object_detector.py")
    
    try:
        analyzer = DeviceDetectionAnalyzer()
        analyzer.analyze_all_images()
        
        print(f"\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()