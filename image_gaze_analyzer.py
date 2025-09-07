import os
import sys
import cv2
import json
import time
import hashlib
import numpy as np
from datetime import datetime

# Eliminate ALL warnings before importing mediapipe
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress stdout/stderr temporarily for mediapipe import
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

# Import mediapipe quietly
with SuppressOutput():
    import mediapipe as mp

class CleanMediaPipeAnalyzer:
    def __init__(self, config_path="config.json"):
        print("Starting MediaPipe Gaze Analysis...")
        
        self.config_path = config_path
        self.config = self.load_config()
        self.input_dir = "input_images"
        self.output_dir = "analyzed_images_mediapipe"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize MediaPipe quietly
        print("Initializing MediaPipe...")
        with SuppressOutput():
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
        print("MediaPipe initialized successfully")

    def load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                print(f"Config loaded: left={config['detection']['gaze']['left_limit']}, right={config['detection']['gaze']['right_limit']}, down={config['detection']['gaze']['down_limit']}")
                return config
        except Exception as e:
            print(f"Using default config due to error: {e}")
            return {
                'detection': {
                    'gaze': {
                        'left_limit': 0.65,
                        'right_limit': 0.35,
                        'down_limit': 0.6
                    }
                }
            }

    def get_image_files(self):
        """Get all image files from input directory"""
        if not os.path.exists(self.input_dir):
            print(f"Input directory not found: {self.input_dir}")
            return []
        
        image_files = []
        for file in sorted(os.listdir(self.input_dir)):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                full_path = os.path.join(self.input_dir, file)
                if os.path.isfile(full_path):
                    image_files.append({
                        'filename': file,
                        'full_path': full_path
                    })
        
        print(f"Found {len(image_files)} images")
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
        
        return None

    def calculate_ratios(self, face_landmarks, width, height):
        """Calculate H and V ratios using MediaPipe landmarks"""
        try:
            # Key landmarks
            nose_tip = face_landmarks.landmark[1]
            forehead = face_landmarks.landmark[10] 
            chin = face_landmarks.landmark[152]
            left_face = face_landmarks.landmark[234]
            right_face = face_landmarks.landmark[454]
            
            # H ratio calculation
            nose_x = nose_tip.x * width
            left_x = left_face.x * width
            right_x = right_face.x * width
            
            if right_x > left_x and (right_x - left_x) > 0:
                h_ratio = (nose_x - left_x) / (right_x - left_x)
            else:
                h_ratio = 0.5
            
            # V ratio calculation
            nose_y = nose_tip.y * height
            forehead_y = forehead.y * height
            chin_y = chin.y * height
            
            if chin_y > forehead_y and (chin_y - forehead_y) > 0:
                v_ratio = (nose_y - forehead_y) / (chin_y - forehead_y)
            else:
                v_ratio = 0.5
            
            # Clamp values to 0-1
            h_ratio = max(0.0, min(1.0, h_ratio))
            v_ratio = max(0.0, min(1.0, v_ratio))
            
            return h_ratio, v_ratio
            
        except Exception as e:
            return 0.5, 0.5

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
            # Enhanced down detection - standard OR extreme threshold
            return v_ratio > down_limit or v_ratio > 0.75
        elif expected_direction == 'center':
            return not (h_ratio > left_limit or h_ratio < right_limit or (v_ratio > down_limit or v_ratio > 0.75))
        
        return None

    def create_annotated_image(self, frame, filename, expected, h_ratio, v_ratio, is_correct):
        """Create detailed annotated image"""
        if frame is None:
            return frame
        
        annotated = frame.copy()
        
        # Calculate hash for verification
        frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        image_hash = hashlib.md5(frame_bytes).hexdigest()[:8]
        
        # MediaPipe indicator
        cv2.putText(annotated, "MediaPipe Detection", 
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
        
        # Threshold checks
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
        cv2.putText(annotated, f"Down: V({v_ratio:.3f}) > {down_limit} = {down_check}", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                    (0, 255, 0) if down_check else (128, 128, 128), 2)
        
        y_pos += 25
        extreme_down_check = v_ratio > 0.75
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
        
        return annotated

    def analyze_all_images(self):
        """Analyze all images with MediaPipe"""
        print("Starting analysis...")
        
        image_files = self.get_image_files()
        if not image_files:
            print("No images found!")
            return
        
        results = []
        failed_to_load = 0
        
        start_time = time.time()
        
        print("=" * 60)
        
        for i, image_info in enumerate(image_files, 1):
            filename = image_info['filename']
            
            # Progress indicator
            if i % 50 == 0 or i == len(image_files):
                print(f"Progress: {i}/{len(image_files)} ({i/len(image_files)*100:.1f}%)")
            
            # Get expected direction first
            expected = self.extract_expected_direction(filename)
            if not expected:
                continue  # Skip files without expected direction
            
            # Load image
            frame = cv2.imread(image_info['full_path'])
            if frame is None:
                print(f"Could not load: {filename}")
                failed_to_load += 1
                # Count as failed detection (incorrect)
                results.append({
                    'filename': filename,
                    'expected': expected,
                    'h_ratio': 0.5,
                    'v_ratio': 0.5,
                    'correct': False,
                    'detection_failed': True
                })
                continue
            
            try:
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe (suppress any remaining output)
                with SuppressOutput():
                    results_mp = self.face_mesh.process(rgb_frame)
                
                if not results_mp.multi_face_landmarks:
                    # No face detected - count as incorrect
                    results.append({
                        'filename': filename,
                        'expected': expected,
                        'h_ratio': 0.5,
                        'v_ratio': 0.5,
                        'correct': False,
                        'detection_failed': True
                    })
                    continue
                
                # Calculate ratios
                face_landmarks = results_mp.multi_face_landmarks[0]
                height, width = frame.shape[:2]
                h_ratio, v_ratio = self.calculate_ratios(face_landmarks, width, height)
                
                # Check threshold correctness
                is_correct = self.check_threshold_correctness(h_ratio, v_ratio, expected)
                
                # Create annotated image
                annotated_frame = self.create_annotated_image(
                    frame, filename, expected, h_ratio, v_ratio, is_correct
                )
                
                # Save annotated image
                output_filename = f"mp_{filename}"
                output_path = os.path.join(self.output_dir, output_filename)
                cv2.imwrite(output_path, annotated_frame)
                
                results.append({
                    'filename': filename,
                    'expected': expected,
                    'h_ratio': h_ratio,
                    'v_ratio': v_ratio,
                    'correct': is_correct,
                    'detection_failed': False
                })
                    
            except Exception as e:
                # Error in processing - count as incorrect
                results.append({
                    'filename': filename,
                    'expected': expected,
                    'h_ratio': 0.5,
                    'v_ratio': 0.5,
                    'correct': False,
                    'detection_failed': True
                })
                continue
        
        total_time = time.time() - start_time
        
        # Calculate statistics including ALL images
        total_images = len(results)
        correct_count = sum(1 for r in results if r['correct'])
        failed_detections = sum(1 for r in results if r['detection_failed'])
        successful_detections = total_images - failed_detections
        
        # Display results
        print(f"Total images with expected directions: {total_images}")
        print(f"Images that failed to load: {failed_to_load}")
        print(f"Face detection failures: {failed_detections}")
        print(f"Successfully detected faces: {successful_detections}")
        
        if total_images > 0:
            overall_accuracy = (correct_count / total_images) * 100
            print(f"Correct predictions: {correct_count}/{total_images}")
            print(f"Overall Accuracy: {overall_accuracy:.1f}%")
            print(f"Processing time: {total_time:.1f} seconds")
            
            # Accuracy by direction (including failures)
            direction_stats = {}
            for result in results:
                direction = result['expected']
                if direction not in direction_stats:
                    direction_stats[direction] = {
                        'total': 0, 
                        'correct': 0, 
                        'failed': 0,
                        'ratios': []
                    }
                direction_stats[direction]['total'] += 1
                if result['correct']:
                    direction_stats[direction]['correct'] += 1
                if result['detection_failed']:
                    direction_stats[direction]['failed'] += 1
                else:
                    direction_stats[direction]['ratios'].append((result['h_ratio'], result['v_ratio']))
            
            print(f"\nACCURACY BY DIRECTION (including detection failures):")
            for direction, stats in direction_stats.items():
                accuracy = (stats['correct'] / stats['total']) * 100
                detection_rate = ((stats['total'] - stats['failed']) / stats['total']) * 100
                
                if stats['ratios']:
                    avg_h = np.mean([r[0] for r in stats['ratios']])
                    avg_v = np.mean([r[1] for r in stats['ratios']])
                    ratio_info = f"H_avg={avg_h:.3f}, V_avg={avg_v:.3f}"
                else:
                    ratio_info = "No successful detections"
                
                print(f"  {direction.upper()}: {stats['correct']}/{stats['total']} = {accuracy:.1f}% (Detection: {detection_rate:.1f}%, {ratio_info})")
            
            # Special analysis for DOWN images
            down_results = [r for r in results if r['expected'] == 'down']
            if down_results:
                down_total = len(down_results)
                down_correct = sum(1 for r in down_results if r['correct'])
                down_failed = sum(1 for r in down_results if r['detection_failed'])
                down_successful = down_total - down_failed
                
                down_accuracy = (down_correct / down_total) * 100
                down_detection_rate = (down_successful / down_total) * 100
                
                # Only count successful detections for threshold analysis
                successful_down = [r for r in down_results if not r['detection_failed']]
                if successful_down:
                    standard_down = sum(1 for r in successful_down if r['v_ratio'] > self.config['detection']['gaze']['down_limit'])
                    extreme_down = sum(1 for r in successful_down if r['v_ratio'] > 0.75)
                else:
                    standard_down = extreme_down = 0
                
                print(f"DOWN total images: {down_total}")
                print(f"DOWN detection rate: {down_successful}/{down_total} = {down_detection_rate:.1f}%")
                print(f"DOWN accuracy (including failures): {down_correct}/{down_total} = {down_accuracy:.1f}%")
                
                if down_successful > 0:
                    down_accuracy_detected = (down_correct / down_successful) * 100
                    print(f"DOWN accuracy (only detected faces): {down_correct}/{down_successful} = {down_accuracy_detected:.1f}%")
                    print(f"Standard threshold (>{self.config['detection']['gaze']['down_limit']}) detected: {standard_down}/{down_successful}")
                    print(f"Extreme threshold (>0.75) detected: {extreme_down}/{down_successful}")
                
                if down_accuracy > 90:
                    print("EXCELLENT: MediaPipe handles tilted heads very well!")
                elif down_accuracy > 70:
                    print("GOOD: MediaPipe handles tilted heads reasonably well!")
                else:
                    print("NEEDS IMPROVEMENT: MediaPipe struggles with tilted heads")
            
            annotated_count = sum(1 for r in results if not r['detection_failed'])
            print(f"\nAnnotated images saved to: {self.output_dir}/")
            print(f"Total annotated images: {annotated_count}")
            
        else:
            print("No images were found with expected directions!")

def main():
    try:
        analyzer = CleanMediaPipeAnalyzer()
        analyzer.analyze_all_images()
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()