import os
import cv2
import json
import time
import gc
import hashlib
import logging
import numpy as np
from datetime import datetime
from modules.face_detector import FaceDetector

class ImageGazeAnalyzer:
    def __init__(self, config_path="config.json", use_mediapipe=True):
        self.config_path = config_path
        self.config = self.load_config()
        self.input_dir = "input_images"
        self.output_dir = "analyzed_images_mediapipe" if use_mediapipe else "analyzed_images_dlib"
        self.mirror_image = self.config.get('video', {}).get('mirror_image', True)
        self.use_mediapipe = use_mediapipe
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Setup logging to file"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        method = "mediapipe" if self.use_mediapipe else "dlib"
        log_file = os.path.join(log_dir, f"gaze_analysis_{method}_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file
        
        # Log startup
        self.logger.info("=" * 60)
        self.logger.info(f"GAZE ANALYSIS SESSION STARTED - {method.upper()}")
        self.logger.info("=" * 60)
        self.logger.info(f"Detection method: {method}")
        self.logger.info(f"Log file: {os.path.abspath(log_file)}")
        self.logger.info(f"Input directory: {os.path.abspath(self.input_dir)}")
        self.logger.info(f"Output directory: {os.path.abspath(self.output_dir)}")
        self.logger.info(f"Mirror image: {self.mirror_image}")

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
                        'down_limit': 0.55
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
            self.logger.error(f"Input directory not found: {self.input_dir}")
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
        
        self.logger.info(f"Found {len(image_files)} image files")
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
            # Enhanced down detection - consider both standard and extreme thresholds
            return v_ratio > down_limit or v_ratio > 0.75
        elif expected_direction == 'center':
            return not (h_ratio > left_limit or h_ratio < right_limit or v_ratio > down_limit)
        
        return None

    def force_fresh_detection(self, detector, frame):
        """Force completely fresh detection by resetting detector state"""
        self.logger.info("Forcing fresh detection...")
        
        # Reset detector state completely
        if hasattr(detector, 'reset_state'):
            detector.reset_state()
        
        # Set image mode for static processing
        if hasattr(detector, 'set_image_mode'):
            detector.set_image_mode(True)
        
        # Multiple detection passes with different approaches
        results = []
        
        # Pass 1: Clean frame
        try:
            detector.reset_state()  # Reset before each pass
            direction1, processed_frame1, h_ratio1, v_ratio1 = detector.detect_direction(frame.copy())
            results.append((direction1, processed_frame1, h_ratio1, v_ratio1, "clean"))
            self.logger.info(f"Pass 1 (clean): {direction1}, H: {h_ratio1:.3f}, V: {v_ratio1:.3f}")
        except Exception as e:
            self.logger.warning(f"Pass 1 failed: {e}")
        
        # Pass 2: Small delay and fresh detection
        try:
            time.sleep(0.05)
            detector.reset_state()
            direction2, processed_frame2, h_ratio2, v_ratio2 = detector.detect_direction(frame.copy())
            results.append((direction2, processed_frame2, h_ratio2, v_ratio2, "delayed"))
            self.logger.info(f"Pass 2 (delayed): {direction2}, H: {h_ratio2:.3f}, V: {v_ratio2:.3f}")
        except Exception as e:
            self.logger.warning(f"Pass 2 failed: {e}")
        
        # Pass 3: Different frame processing
        try:
            detector.reset_state()
            frame_copy = cv2.GaussianBlur(frame, (3, 3), 0)  # Very light blur
            direction3, processed_frame3, h_ratio3, v_ratio3 = detector.detect_direction(frame_copy)
            results.append((direction3, processed_frame3, h_ratio3, v_ratio3, "blurred"))
            self.logger.info(f"Pass 3 (blurred): {direction3}, H: {h_ratio3:.3f}, V: {v_ratio3:.3f}")
        except Exception as e:
            self.logger.warning(f"Pass 3 failed: {e}")
        
        if not results:
            raise Exception("All detection passes failed")
        
        # Choose the most consistent result or the middle one
        self.logger.info(f"Total detection passes: {len(results)}")
        
        # Use the first successful result for consistency
        direction, processed_frame, h_ratio, v_ratio, method = results[0]
        self.logger.info(f"Using result from: {method}")
        
        return direction, processed_frame, h_ratio, v_ratio

    def create_annotated_image(self, frame, filename, expected, h_ratio, v_ratio, is_correct):
        """Create detailed annotated image with MediaPipe/dlib info"""
        if frame is None:
            return frame
        
        annotated = frame.copy()
        
        # Calculate hash for verification
        frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        image_hash = hashlib.md5(frame_bytes).hexdigest()[:8]
        
        # Detection method indicator
        method = "MediaPipe" if self.use_mediapipe else "dlib"
        cv2.putText(annotated, f"Method: {method}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Main ratios (large and prominent)
        cv2.putText(annotated, f"H-Ratio: {h_ratio:.3f}", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        cv2.putText(annotated, f"V-Ratio: {v_ratio:.3f}", 
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        
        # Expected direction
        if expected:
            cv2.putText(annotated, f"EXPECTED: {expected.upper()}", 
                        (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
            # Result (CORRECT/WRONG)
            color = (0, 255, 0) if is_correct else (0, 0, 255)
            status = "CORRECT" if is_correct else "WRONG"
            cv2.putText(annotated, status, 
                        (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
        
        # Enhanced threshold checks
        left_limit = self.config['detection']['gaze']['left_limit']
        right_limit = self.config['detection']['gaze']['right_limit']
        down_limit = self.config['detection']['gaze']['down_limit']
        
        y_pos = 270
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
        extreme_down_check = v_ratio > 0.75
        cv2.putText(annotated, f"Down: V({v_ratio:.3f}) > {down_limit} = {down_check}", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                    (0, 255, 0) if down_check else (128, 128, 128), 2)
        
        y_pos += 25
        cv2.putText(annotated, f"Extreme Down: V({v_ratio:.3f}) > 0.75 = {extreme_down_check}", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                    (0, 255, 0) if extreme_down_check else (128, 128, 128), 2)
        
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
        
        # Add enhanced gaze indicator
        self.draw_enhanced_gaze_indicator(annotated, h_ratio, v_ratio, expected)
        
        return annotated

    def draw_enhanced_gaze_indicator(self, frame, h_ratio, v_ratio, expected_direction):
        """Draw enhanced gaze indicator with MediaPipe zones"""
        height, width = frame.shape[:2]
        
        # Indicator size and position
        size = 140
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
        
        # Down zone (standard)
        down_y = int(y + down_limit * size)
        cv2.rectangle(frame, (x, down_y), (x + size, y + size), (100, 200, 100), 1)
        cv2.putText(frame, "D", (x + 5, down_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 100), 1)
        
        # Extreme down zone (enhanced for MediaPipe)
        extreme_down_y = int(y + 0.75 * size)
        cv2.rectangle(frame, (x, extreme_down_y), (x + size, y + size), (50, 150, 50), 2)
        cv2.putText(frame, "ED", (x + size - 25, extreme_down_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 150, 50), 1)
        
        # Highlight expected zone
        if expected_direction == 'left':
            cv2.rectangle(frame, (left_x, y), (x + size, y + size), (0, 255, 255), 3)
        elif expected_direction == 'right':
            cv2.rectangle(frame, (x, y), (right_x, y + size), (0, 255, 255), 3)
        elif expected_direction == 'down':
            cv2.rectangle(frame, (x, down_y), (x + size, y + size), (0, 255, 255), 3)
            cv2.rectangle(frame, (x, extreme_down_y), (x + size, y + size), (0, 255, 255), 2)
        
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
        
        # Method indicator
        method_text = "MP" if self.use_mediapipe else "DL"
        cv2.putText(frame, method_text, (x + size - 25, y + size - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def analyze_all_images(self):
        """Analyze all images with enhanced MediaPipe detection"""
        method_name = "MediaPipe" if self.use_mediapipe else "dlib"
        self.logger.info(f"Starting Enhanced Analysis with {method_name}...")
        
        image_files = self.get_image_files()
        if not image_files:
            self.logger.error("No images found!")
            return
        
        correct_count = 0
        total_count = 0
        results = []
        down_images_results = []  # Special tracking for down images
        
        start_time = time.time()
        
        # Create detector with specified method
        detector = FaceDetector(self.mirror_image, use_mediapipe=self.use_mediapipe)
        self.logger.info(f"Created FaceDetector with {method_name}, mirror_image={self.mirror_image}")
        
        for i, image_info in enumerate(image_files, 1):
            filename = image_info['filename']
            self.logger.info(f"\n[{i}/{len(image_files)}] Processing: {filename}")
            
            # Load image
            frame = cv2.imread(image_info['full_path'])
            if frame is None:
                self.logger.error(f"Could not load image: {filename}")
                continue
            
            try:
                # Force fresh detection
                direction, processed_frame, h_ratio, v_ratio = self.force_fresh_detection(detector, frame)
                
                # Get expected direction
                expected = self.extract_expected_direction(filename)
                
                if expected:
                    # Check threshold correctness
                    is_correct = self.check_threshold_correctness(h_ratio, v_ratio, expected)
                    
                    total_count += 1
                    if is_correct:
                        correct_count += 1
                    
                    # Special tracking for down images
                    if expected == 'down':
                        down_images_results.append({
                            'filename': filename,
                            'h_ratio': h_ratio,
                            'v_ratio': v_ratio,
                            'correct': is_correct,
                            'standard_down': v_ratio > self.config['detection']['gaze']['down_limit'],
                            'extreme_down': v_ratio > 0.75
                        })
                    
                    # Log results
                    status = "CORRECT" if is_correct else "WRONG"
                    self.logger.info(f"Expected: {expected} | H: {h_ratio:.3f} | V: {v_ratio:.3f} | {status}")
                    
                    # Create detailed annotated image
                    annotated_frame = self.create_annotated_image(
                        processed_frame, filename, expected, h_ratio, v_ratio, is_correct
                    )
                    
                    # Save annotated image
                    method_prefix = "mp" if self.use_mediapipe else "dl"
                    output_filename = f"{method_prefix}_{filename}"
                    output_path = os.path.join(self.output_dir, output_filename)
                    cv2.imwrite(output_path, annotated_frame)
                    self.logger.info(f"Saved: {output_filename}")
                    
                    results.append({
                        'filename': filename,
                        'expected': expected,
                        'h_ratio': h_ratio,
                        'v_ratio': v_ratio,
                        'correct': is_correct
                    })
                else:
                    self.logger.warning(f"No expected direction found in filename: {filename}")
                    
            except Exception as e:
                self.logger.error(f"Error processing {filename}: {e}")
            
            # Cleanup and delay
            gc.collect()
            time.sleep(0.1)
        
        total_time = time.time() - start_time
        
        # Calculate and display enhanced results
        if total_count > 0:
            accuracy = (correct_count / total_count) * 100
            
            self.logger.info("\n" + "=" * 60)
            self.logger.info(f"ANALYSIS COMPLETE - {method_name.upper()} RESULTS")
            self.logger.info("=" * 60)
            self.logger.info(f"Detection method: {method_name}")
            self.logger.info(f"Total images tested: {total_count}")
            self.logger.info(f"Correct: {correct_count}")
            self.logger.info(f"Wrong: {total_count - correct_count}")
            self.logger.info(f"Overall Accuracy: {correct_count}/{total_count} = {accuracy:.1f}%")
            self.logger.info(f"Total time: {total_time:.1f} seconds")
            
            # Enhanced down detection analysis
            if down_images_results:
                down_total = len(down_images_results)
                down_correct = sum(1 for r in down_images_results if r['correct'])
                down_accuracy = (down_correct / down_total) * 100
                
                standard_down_detected = sum(1 for r in down_images_results if r['standard_down'])
                extreme_down_detected = sum(1 for r in down_images_results if r['extreme_down'])
                
                self.logger.info("\nDOWN DETECTION ANALYSIS:")
                self.logger.info(f"Down images total: {down_total}")
                self.logger.info(f"Down images correct: {down_correct}/{down_total} = {down_accuracy:.1f}%")
                self.logger.info(f"Standard down threshold (>{self.config['detection']['gaze']['down_limit']}) detected: {standard_down_detected}")
                self.logger.info(f"Extreme down threshold (>0.75) detected: {extreme_down_detected}")
                
                if self.use_mediapipe:
                    self.logger.info("MediaPipe should handle tilted heads better!")
            
            # Accuracy by direction
            direction_stats = {}
            for result in results:
                direction = result['expected']
                if direction not in direction_stats:
                    direction_stats[direction] = {'total': 0, 'correct': 0}
                direction_stats[direction]['total'] += 1
                if result['correct']:
                    direction_stats[direction]['correct'] += 1
            
            self.logger.info("\nACCURACY BY DIRECTION:")
            for direction, stats in direction_stats.items():
                accuracy = (stats['correct'] / stats['total']) * 100
                self.logger.info(f"  {direction.upper()}: {stats['correct']}/{stats['total']} = {accuracy:.1f}%")
            
            # Check for duplicate ratios
            ratio_groups = {}
            for result in results:
                ratio_key = f"{result['h_ratio']:.3f},{result['v_ratio']:.3f}"
                if ratio_key not in ratio_groups:
                    ratio_groups[ratio_key] = []
                ratio_groups[ratio_key].append(result['filename'])
            
            duplicates = {k: v for k, v in ratio_groups.items() if len(v) > 1}
            if duplicates:
                self.logger.warning("CACHE WARNING - Identical ratios found:")
                for ratios, files in duplicates.items():
                    self.logger.warning(f"   {ratios}: {', '.join(files)}")
            else:
                self.logger.info("NO CACHE ISSUES - All ratios are unique!")
            
            # Configuration info
            self.logger.info("\nTHRESHOLD CONFIGURATION:")
            self.logger.info(f"   Left: H > {self.config['detection']['gaze']['left_limit']}")
            self.logger.info(f"   Right: H < {self.config['detection']['gaze']['right_limit']}")
            self.logger.info(f"   Down: V > {self.config['detection']['gaze']['down_limit']} OR V > 0.75 (extreme)")
            self.logger.info(f"   Center: None of the above")
            
            self.logger.info(f"\nAnnotated images saved to: {os.path.abspath(self.output_dir)}")
            
            # Final comparison note
            if self.use_mediapipe:
                self.logger.info("\nMEDIAPIPE ADVANTAGES:")
                self.logger.info("- Better detection at extreme head angles")
                self.logger.info("- More robust eye tracking when head is tilted")
                self.logger.info("- Enhanced down detection for tilted heads")
            
        else:
            self.logger.error("No images with expected directions found!")

        self.logger.info("=" * 60)
        self.logger.info("SESSION ENDED")
        self.logger.info("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze gaze with MediaPipe or dlib')
    parser.add_argument('--method', choices=['mediapipe', 'dlib'], default='mediapipe',
                        help='Detection method to use (default: mediapipe)')
    
    args = parser.parse_args()
    
    use_mediapipe = (args.method == 'mediapipe')
    
    try:
        analyzer = ImageGazeAnalyzer(use_mediapipe=use_mediapipe)
        analyzer.analyze_all_images()
        
        method_name = "MediaPipe" if use_mediapipe else "dlib"
        print(f"\n{method_name} analysis complete!")
        print(f"Check log file: {analyzer.log_file}")
        print(f"Annotated images saved to: {analyzer.output_dir}/")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()