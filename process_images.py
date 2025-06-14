import cv2
import os
import json
import numpy as np
from datetime import datetime
from modules.face_detector import FaceDetector

class ImageProcessor:
    def __init__(self, config_path="config.json"):
        """Initialize the image processor with configuration"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.mirror_image = self.config["camera"]["mirror_image"]
        self.face_detector = FaceDetector(self.mirror_image)
        
        # Input and output directories
        self.input_dir = "input_images"
        self.output_dir = "output_images"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def draw_gaze_info(self, frame, direction, h_ratio, v_ratio):
        """Draw gaze direction information on the frame"""
        if frame is None:
            return frame
            
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Draw gaze direction text
        direction_text = f"Direction: {direction}"
        cv2.putText(frame, direction_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw horizontal ratio
        h_text = f"H-Ratio: {h_ratio:.3f}"
        cv2.putText(frame, h_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw vertical ratio
        v_text = f"V-Ratio: {v_ratio:.3f}"
        cv2.putText(frame, v_text, (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw gaze indicator
        self.draw_gaze_indicator(frame, h_ratio, v_ratio)
        
        return frame
    
    def draw_gaze_indicator(self, frame, h_ratio, v_ratio):
        """Draw visual gaze direction indicator"""
        height, width = frame.shape[:2]
        
        # Draw gaze indicator in top-right corner
        indicator_size = 100
        indicator_x = width - indicator_size - 20
        indicator_y = 20
        
        # Draw indicator background
        cv2.rectangle(frame, 
                        (indicator_x, indicator_y), 
                        (indicator_x + indicator_size, indicator_y + indicator_size),
                        (50, 50, 50), -1)
        
        # Draw grid lines
        cv2.line(frame, 
                (indicator_x + indicator_size//2, indicator_y),
                (indicator_x + indicator_size//2, indicator_y + indicator_size),
                (100, 100, 100), 1)
        cv2.line(frame, 
                (indicator_x, indicator_y + indicator_size//2),
                (indicator_x + indicator_size, indicator_y + indicator_size//2),
                (100, 100, 100), 1)
        
        # Calculate gaze point position
        gaze_x = int(indicator_x + h_ratio * indicator_size)
        gaze_y = int(indicator_y + v_ratio * indicator_size)
        
        # Ensure point is within bounds
        gaze_x = max(indicator_x, min(gaze_x, indicator_x + indicator_size))
        gaze_y = max(indicator_y, min(gaze_y, indicator_y + indicator_size))
        
        # Draw gaze point
        cv2.circle(frame, (gaze_x, gaze_y), 5, (0, 255, 255), -1)
        cv2.circle(frame, (gaze_x, gaze_y), 8, (0, 255, 255), 2)
        
        # Draw center reference point
        center_x = indicator_x + indicator_size // 2
        center_y = indicator_y + indicator_size // 2
        cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
    
    def draw_eye_regions(self, frame):
        """Draw detected eye regions and pupils"""
        if not hasattr(self.face_detector, 'gaze_tracker'):
            return frame
            
        gaze_tracker = self.face_detector.gaze_tracker
        
        # Draw left eye region
        if gaze_tracker.eye_left and gaze_tracker.eye_left.landmark_points is not None:
            cv2.polylines(frame, [gaze_tracker.eye_left.landmark_points], 
                            True, (0, 255, 0), 1)
            
            # Draw left pupil
            if gaze_tracker.eye_left.pupil:
                pupil_x = int(gaze_tracker.eye_left.origin[0] + gaze_tracker.eye_left.pupil.x)
                pupil_y = int(gaze_tracker.eye_left.origin[1] + gaze_tracker.eye_left.pupil.y)
                cv2.circle(frame, (pupil_x, pupil_y), 3, (0, 0, 255), -1)
                cv2.circle(frame, (pupil_x, pupil_y), 6, (0, 0, 255), 1)
        
        # Draw right eye region
        if gaze_tracker.eye_right and gaze_tracker.eye_right.landmark_points is not None:
            cv2.polylines(frame, [gaze_tracker.eye_right.landmark_points], 
                            True, (0, 255, 0), 1)
            
            # Draw right pupil
            if gaze_tracker.eye_right.pupil:
                pupil_x = int(gaze_tracker.eye_right.origin[0] + gaze_tracker.eye_right.pupil.x)
                pupil_y = int(gaze_tracker.eye_right.origin[1] + gaze_tracker.eye_right.pupil.y)
                cv2.circle(frame, (pupil_x, pupil_y), 3, (0, 0, 255), -1)
                cv2.circle(frame, (pupil_x, pupil_y), 6, (0, 0, 255), 1)
        
        return frame
    
    def draw_face_landmarks(self, frame):
        """Draw facial landmarks for reference"""
        if not hasattr(self.face_detector, 'gaze_tracker'):
            return frame
            
        gaze_tracker = self.face_detector.gaze_tracker
        
        # Draw face rectangle if available
        if hasattr(gaze_tracker, 'face_rect') and gaze_tracker.face_rect:
            x, y, w, h = gaze_tracker.face_rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        return frame
    
    def process_single_image(self, image_path, output_path):
        """Process a single image and save the result"""
        print(f"Processing: {image_path}")
        
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image {image_path}")
            return False
        
        try:
            # Detect gaze direction
            direction, processed_frame, h_ratio, v_ratio = self.face_detector.detect_direction(frame)
            
            # Draw eye regions and pupils
            processed_frame = self.draw_eye_regions(processed_frame)
            
            # Draw face landmarks for reference
            processed_frame = self.draw_face_landmarks(processed_frame)
            
            # Add gaze information overlay
            processed_frame = self.draw_gaze_info(processed_frame, direction, h_ratio, v_ratio)
            
            # Add processing timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(processed_frame, f"Processed: {timestamp}", 
                        (10, processed_frame.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            
            # Save processed image
            success = cv2.imwrite(output_path, processed_frame)
            
            if success:
                print(f"✓ Saved: {output_path}")
                print(f"  Direction: {direction}, H: {h_ratio:.3f}, V: {v_ratio:.3f}")
                return True
            else:
                print(f"✗ Failed to save: {output_path}")
                return False
                
        except Exception as e:
            print(f"✗ Error processing {image_path}: {e}")
            return False
    
    def process_all_images(self):
        """Process all images in the input directory"""
        if not os.path.exists(self.input_dir):
            print(f"Error: Input directory '{self.input_dir}' does not exist!")
            return
        
        # Get list of image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for file in os.listdir(self.input_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file)
        
        if not image_files:
            print(f"No image files found in '{self.input_dir}'")
            return
        
        print(f"Found {len(image_files)} images to process:")
        for img in image_files:
            print(f"  - {img}")
        print()
        
        # Process each image
        successful = 0
        failed = 0
        
        for image_file in image_files:
            input_path = os.path.join(self.input_dir, image_file)
            
            # Create output filename with processed prefix
            name, ext = os.path.splitext(image_file)
            output_filename = f"processed_{name}{ext}"
            output_path = os.path.join(self.output_dir, output_filename)
            
            if self.process_single_image(input_path, output_path):
                successful += 1
            else:
                failed += 1
        
        print(f"\n=== Processing Complete ===")
        print(f"Successfully processed: {successful}")
        print(f"Failed: {failed}")
        print(f"Output directory: {os.path.abspath(self.output_dir)}")
    
    def create_summary_report(self):
        """Create a summary report of all processed images"""
        report_path = os.path.join(self.output_dir, "processing_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("=== Image Processing Report ===\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input Directory: {os.path.abspath(self.input_dir)}\n")
            f.write(f"Output Directory: {os.path.abspath(self.output_dir)}\n\n")
            
            # List processed files
            output_files = [f for f in os.listdir(self.output_dir) 
                            if f.startswith('processed_') and not f.endswith('.txt')]
            
            f.write(f"Processed Images ({len(output_files)}):\n")
            for i, file in enumerate(output_files, 1):
                f.write(f"{i:2d}. {file}\n")
            
            f.write(f"\nConfiguration Used:\n")
            f.write(f"Mirror Image: {self.mirror_image}\n")
            f.write(f"Left Limit: {self.config['detection']['gaze']['left_limit']}\n")
            f.write(f"Right Limit: {self.config['detection']['gaze']['right_limit']}\n")
            f.write(f"Down Limit: {self.config['detection']['gaze']['down_limit']}\n")
        
        print(f"Summary report saved: {report_path}")

def main():
    """Main function to run the image processor"""
    print("=== Anti-Plagiarism Image Processor ===")
    print("Processing images with gaze detection...\n")
    
    try:
        # Initialize processor
        processor = ImageProcessor()
        
        # Process all images
        processor.process_all_images()
        
        # Create summary report
        processor.create_summary_report()
        
        print(f"\n🎉 All done! Check the '{processor.output_dir}' directory for results.")
        
    except FileNotFoundError as e:
        print(f"Configuration file not found: {e}")
        print("Make sure 'config.json' exists in the current directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()