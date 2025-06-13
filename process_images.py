import os
import cv2
import json
from modules.gaze_tracking.gaze_tracker import GazeTracker

def process_images(input_dir, output_dir):
    """
    Simple processing of images with gaze tracking
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize gaze tracker
    gaze_tracker = GazeTracker(mirror_image=True, config_path="config.json")
    
    # Get image files
    image_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_files.append(file)
    
    image_files.sort()  # Sort to ensure consistent order
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for i, filename in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {filename}")
        
        # Load image
        input_path = os.path.join(input_dir, filename)
        frame = cv2.imread(input_path)
        
        if frame is None:
            print(f"  ERROR: Could not load {filename}")
            continue
        
        # Process with gaze tracker
        direction, annotated_frame, h_ratio, v_ratio = gaze_tracker.detect_gaze_direction(frame)
        
        # Add text with direction and ratios
        cv2.putText(annotated_frame, f"Direction: {direction}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"H-Ratio: {h_ratio:.3f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"V-Ratio: {v_ratio:.3f}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save with simple naming
        output_filename = f"result_{i+1:03d}_{filename}"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, annotated_frame)
        
        print(f"  Saved: {output_filename} - Direction: {direction}, H: {h_ratio:.3f}, V: {v_ratio:.3f}")

if __name__ == "__main__":
    # Simple configuration
    INPUT_DIR = "input_images"
    OUTPUT_DIR = "output_images"
    
    # Create config if missing
    if not os.path.exists("config.json"):
        config = {
            "detection": {
                "gaze": {
                    "left_limit": 0.6,
                    "right_limit": 0.4,
                    "down_limit": 0.6
                }
            }
        }
        with open("config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    # Check input directory
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"Put your images in {INPUT_DIR} folder")
    else:
        process_images(INPUT_DIR, OUTPUT_DIR)
