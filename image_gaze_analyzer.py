import os
import cv2
import json
import time
import gc
import hashlib
import numpy as np
from datetime import datetime
from modules.face_detector import FaceDetector

class ImageGazeAnalyzer:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.input_dir = "input_images"
        self.output_dir = "analyzed_images"
        self.mirror_image = self.config.get('video', {}).get('mirror_image', True)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {
                'detection': {
                    'gaze': {
                        'left_limit': 0.65,
                        'right_limit': 0.35,
                        'down_limit': 0.68
                    }
                },
                'video': {
                    'mirror_image': True
                }
            }

    def get_image_files(self):
        """Get all image files from input directory"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF']
        
        if not os.path.exists(self.input_dir):
            print(f"Input directory not found: {self.input_dir}")
            return []
        
        image_files = []
        for file in sorted(os.listdir(self.input_dir)):
            if any(file.endswith(ext) for ext in image_extensions):
                full_path = os.path.join(self.input_dir, file)
                if os.path.isfile(full_path):
                    image_files.append({
                        'filename': file,
                        'full_path': full_path
                    })
        
        return image_files

    def extract_expected_direction(self, filename):
        """Extract expected direction from filename"""
        filename_lower = filename.lower()
        
        if filename_lower.startswith('center'):
            return 'center'
        elif filename_lower.startswith('left'):
            return 'left'
        elif filename_lower.startswith('right'):
            return 'right'
        elif filename_lower.startswith('down'):
            return 'down'
        elif filename_lower.startswith('up'):
            return 'up'
        
        return None

    def check_threshold_correctness(self, h_ratio, v_ratio, expected_direction):
        """Check if H/V ratios are correct for expected direction"""
        left_limit = self.config['detection']['gaze']['left_limit']
        right_limit = self.config['detection']['gaze']['right_limit']
        down_limit = self.config['detection']['gaze']['down_limit']
        
        if expected_direction == 'left':
            return h_ratio > left_limit
        elif expected_direction == 'right':
            return h_ratio < right_limit
        elif expected_direction == 'down':
            return v_ratio > down_limit
        elif expected_direction == 'center':
            return not (h_ratio > left_limit or h_ratio < right_limit or v_ratio > down_limit)
        
        return None

    def force_fresh_detection(self, detector, frame):
        """Force completely fresh detection by breaking temporal cache"""
        print("Forcing fresh detection...")
        
        # Method 1: Reset gaze tracker state completely
        if hasattr(detector, 'gaze_tracker') and detector.gaze_tracker:
            gaze_tracker = detector.gaze_tracker
            
            # Reset ALL possible cached states
            for attr in ['_frame', 'frame', '_previous_frame', 'previous_frame']:
                if hasattr(gaze_tracker, attr):
                    setattr(gaze_tracker, attr, None)
            
            # Reset eye tracking cache
            for attr in ['_left_eye', '_right_eye', 'left_eye', 'right_eye']:
                if hasattr(gaze_tracker, attr):
                    setattr(gaze_tracker, attr, None)
            
            # Reset pupil cache
            for attr in ['_left_pupil', '_right_pupil', 'left_pupil', 'right_pupil']:
                if hasattr(gaze_tracker, attr):
                    setattr(gaze_tracker, attr, None)
            
            # Reset calibration cache
            if hasattr(gaze_tracker, 'calibration'):
                gaze_tracker.calibration = None
            
            # Reset any detection history
            for attr in ['_history', 'history', '_detection_history', 'detection_history']:
                if hasattr(gaze_tracker, attr):
                    if isinstance(getattr(gaze_tracker, attr), list):
                        setattr(gaze_tracker, attr, [])
                    else:
                        setattr(gaze_tracker, attr, None)
            
            # Force image mode
            if hasattr(gaze_tracker, 'set_image_mode'):
                gaze_tracker.set_image_mode(True)
        
        # Method 2: Add random noise to frame to break any pixel-level caching
        frame_copy = frame.copy()
        noise = np.random.randint(-2, 3, frame_copy.shape, dtype=np.int16)
        frame_with_noise = np.clip(frame_copy.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Method 3: Multiple detection passes with different approaches
        results = []
        
        # Pass 1: Clean frame
        try:
            direction1, processed_frame1, h_ratio1, v_ratio1 = detector.detect_direction(frame_copy)
            results.append((direction1, processed_frame1, h_ratio1, v_ratio1, "clean"))
        except:
            pass
        
        # Pass 2: Frame with tiny noise
        try:
            direction2, processed_frame2, h_ratio2, v_ratio2 = detector.detect_direction(frame_with_noise)
            results.append((direction2, processed_frame2, h_ratio2, v_ratio2, "noise"))
        except:
            pass
        
        # Pass 3: Fresh frame load and immediate detection
        try:
            time.sleep(0.1)  # Small delay
            direction3, processed_frame3, h_ratio3, v_ratio3 = detector.detect_direction(frame_copy)
            results.append((direction3, processed_frame3, h_ratio3, v_ratio3, "delayed"))
        except:
            pass
        
        if not results:
            raise Exception("All detection passes failed")
        
        # Choose the most "different" result to avoid cache
        print(f"Detection passes: {len(results)}")
        for i, (direction, _, h_ratio, v_ratio, method) in enumerate(results):
            print(f"      Pass {i+1} ({method}): {direction}, H: {h_ratio:.3f}, V: {v_ratio:.3f}")
        
        # Return the middle result (index 1 if we have 3, index 0 if we have 1)
        best_index = min(1, len(results) - 1)
        direction, processed_frame, h_ratio, v_ratio, method = results[best_index]
        
        print(f"Using result from pass: {method}")
        return direction, processed_frame, h_ratio, v_ratio

    def create_annotated_image(self, frame, filename, expected, h_ratio, v_ratio, is_correct):
        """Create detailed annotated image"""
        if frame is None:
            return frame
        
        annotated = frame.copy()
        
        # Calculate hash for verification
        frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        image_hash = hashlib.md5(frame_bytes).hexdigest()[:8]
        
        # Main ratios (large and prominent)
        cv2.putText(annotated, f"H-Ratio: {h_ratio:.3f}", 
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        cv2.putText(annotated, f"V-Ratio: {v_ratio:.3f}", 
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        
        # Expected direction
        if expected:
            cv2.putText(annotated, f"EXPECTED: {expected.upper()}", 
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
            # Result (CORRECT/WRONG)
            color = (0, 255, 0) if is_correct else (0, 0, 255)
            status = "CORRECT" if is_correct else "WRONG"
            cv2.putText(annotated, status, 
                        (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
        
        # Threshold checks (detailed)
        left_limit = self.config['detection']['gaze']['left_limit']
        right_limit = self.config['detection']['gaze']['right_limit']
        down_limit = self.config['detection']['gaze']['down_limit']
        
        y_pos = 250
        cv2.putText(annotated, "THRESHOLD CHECKS:", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_pos += 30
        left_check = h_ratio > left_limit
        cv2.putText(annotated, f"Left: H({h_ratio:.3f}) > {left_limit} = {left_check}", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                    (0, 255, 0) if left_check else (128, 128, 128), 2)
        
        y_pos += 25
        right_check = h_ratio < right_limit
        cv2.putText(annotated, f"Right: H({h_ratio:.3f}) < {right_limit} = {right_check}", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                    (0, 255, 0) if right_check else (128, 128, 128), 2)
        
        y_pos += 25
        down_check = v_ratio > down_limit
        cv2.putText(annotated, f"Down: V({v_ratio:.3f}) > {down_limit} = {down_check}", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                    (0, 255, 0) if down_check else (128, 128, 128), 2)
        
        # File info
        cv2.putText(annotated, f"File: {filename}", 
                    (10, annotated.shape[0] - 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(annotated, f"Hash: {image_hash}", 
                    (10, annotated.shape[0] - 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated, f"Analyzed: {timestamp}", 
                    (10, annotated.shape[0] - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        # Add gaze indicator
        self.draw_gaze_indicator(annotated, h_ratio, v_ratio, expected)
        
        return annotated

    def draw_gaze_indicator(self, frame, h_ratio, v_ratio, expected_direction):
        """Draw gaze indicator with threshold zones"""
        height, width = frame.shape[:2]
        
        # Indicator size and position
        size = 120
        x = width - size - 20
        y = 20
        
        # Background
        cv2.rectangle(frame, (x, y), (x + size, y + size), (50, 50, 50), -1)
        
        # Draw threshold zones
        left_limit = self.config['detection']['gaze']['left_limit']
        right_limit = self.config['detection']['gaze']['right_limit']
        down_limit = self.config['detection']['gaze']['down_limit']
        
        # Left zone
        left_x = int(x + left_limit * size)
        cv2.rectangle(frame, (left_x, y), (x + size, y + size), (100, 100, 200), 1)
        cv2.putText(frame, "L", (left_x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 200), 1)
        
        # Right zone
        right_x = int(x + right_limit * size)
        cv2.rectangle(frame, (x, y), (right_x, y + size), (200, 100, 100), 1)
        cv2.putText(frame, "R", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 100, 100), 1)
        
        # Down zone
        down_y = int(y + down_limit * size)
        cv2.rectangle(frame, (x, down_y), (x + size, y + size), (100, 200, 100), 1)
        cv2.putText(frame, "D", (x + 5, down_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 100), 1)
        
        # Highlight expected zone
        if expected_direction == 'left':
            cv2.rectangle(frame, (left_x, y), (x + size, y + size), (0, 255, 255), 3)
        elif expected_direction == 'right':
            cv2.rectangle(frame, (x, y), (right_x, y + size), (0, 255, 255), 3)
        elif expected_direction == 'down':
            cv2.rectangle(frame, (x, down_y), (x + size, y + size), (0, 255, 255), 3)
        
        # Grid lines
        cv2.line(frame, (x + size//2, y), (x + size//2, y + size), (100, 100, 100), 1)
        cv2.line(frame, (x, y + size//2), (x + size, y + size//2), (100, 100, 100), 1)
        
        # Gaze point (bright yellow)
        gaze_x = int(x + h_ratio * size)
        gaze_y = int(y + v_ratio * size)
        
        # Clamp to bounds
        gaze_x = max(x, min(gaze_x, x + size))
        gaze_y = max(y, min(gaze_y, y + size))
        
        cv2.circle(frame, (gaze_x, gaze_y), 8, (0, 255, 255), -1)
        cv2.circle(frame, (gaze_x, gaze_y), 12, (0, 255, 255), 2)
        
        # Center reference
        center_x = x + size // 2
        center_y = y + size // 2
        cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

    def analyze_all_images(self):
        """Analyze all images with forced fresh detection"""
        print("🎯 Starting Anti-Cache Threshold Analysis...")
        
        image_files = self.get_image_files()
        if not image_files:
            print("No images found!")
            return
        
        correct_count = 0
        total_count = 0
        results = []
        
        start_time = time.time()
        
        # Create ONE detector but force fresh detection per image
        detector = FaceDetector(self.mirror_image) 
        
        for i, image_info in enumerate(image_files, 1):
            filename = image_info['filename']
            print(f"\n[{i}/{len(image_files)}] Processing: {filename}")
            
            # Load image
            frame = cv2.imread(image_info['full_path'])
            if frame is None:
                print(f"Could not load image")
                continue
            
            try:
                # FORCE fresh detection with anti-cache measures
                direction, processed_frame, h_ratio, v_ratio = self.force_fresh_detection(detector, frame)
                
                # Get expected direction
                expected = self.extract_expected_direction(filename)
                
                if expected:
                    # Check threshold correctness
                    is_correct = self.check_threshold_correctness(h_ratio, v_ratio, expected)
                    
                    total_count += 1
                    if is_correct:
                        correct_count += 1
                    
                    # Terminal output
                    status = "CORRECT" if is_correct else "WRONG"
                    print(f"Expected: {expected} | H: {h_ratio:.3f} | V: {v_ratio:.3f} | {status}")
                    
                    # Create detailed annotated image
                    annotated_frame = self.create_annotated_image(
                        processed_frame, filename, expected, h_ratio, v_ratio, is_correct
                    )
                    
                    # Save annotated image
                    output_filename = f"analyzed_{filename}"
                    output_path = os.path.join(self.output_dir, output_filename)
                    cv2.imwrite(output_path, annotated_frame)
                    print(f"Saved: {output_filename}")
                    
                    results.append({
                        'filename': filename,
                        'expected': expected,
                        'h_ratio': h_ratio,
                        'v_ratio': v_ratio,
                        'correct': is_correct
                    })
                else:
                    print(f"No expected direction found in filename")
                    
            except Exception as e:
                print(f"Error: {e}")
            
            # Extended delay between images
            time.sleep(1.0)
        
        total_time = time.time() - start_time
        
        # Calculate and display accuracy
        if total_count > 0:
            accuracy = (correct_count / total_count) * 100
            
            print(f"Total images tested: {total_count}")
            print(f"Correct: {correct_count}")
            print(f"Wrong: {total_count - correct_count}")
            print(f"Accuracy: {correct_count}/{total_count} = {accuracy:.1f}%")
            print(f"Total time: {total_time:.1f} seconds")
            print(f"Annotated images saved to: {os.path.abspath(self.output_dir)}")
            
            # Check for duplicate ratios (cache detection)
            ratio_groups = {}
            for result in results:
                ratio_key = f"{result['h_ratio']:.3f},{result['v_ratio']:.3f}"
                if ratio_key not in ratio_groups:
                    ratio_groups[ratio_key] = []
                ratio_groups[ratio_key].append(result['filename'])
            
            duplicates = {k: v for k, v in ratio_groups.items() if len(v) > 1}
            if duplicates:
                print(f"\nCACHE WARNING - Identical ratios found:")
                for ratios, files in duplicates.items():
                    print(f"   {ratios}: {', '.join(files)}")
            else:
                print(f"\nNO CACHE ISSUES - All ratios are unique!")
            
            # Show thresholds used
            print(f"   Left: H > {self.config['detection']['gaze']['left_limit']}")
            print(f"   Right: H < {self.config['detection']['gaze']['right_limit']}")
            print(f"   Down: V > {self.config['detection']['gaze']['down_limit']}")
            print(f"   Center: None of the above")
            
            # Show detailed breakdown
            for result in results:
                print(f"{result['filename']} → Expected: {result['expected']}, H: {result['h_ratio']:.3f}, V: {result['v_ratio']:.3f}")
        else:
            print("No images with expected directions found!")


def main():
    try:
        analyzer = ImageGazeAnalyzer()
        analyzer.analyze_all_images()
        
        print(f"\nAnalysis complete! Check:")
        print(f"Terminal output for accuracy results")
        print(f"'analyzed_images/' folder for detailed annotated images")
        print(f"Cache detection warnings")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()