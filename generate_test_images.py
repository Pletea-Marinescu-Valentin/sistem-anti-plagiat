import cv2
import os
import numpy as np
from datetime import datetime

def capture_gaze_images():
    """Capture images with different gaze directions"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    os.makedirs("input_images", exist_ok=True)
    
    # Only 4 directions: center, left, right, down
    directions = [
        "center", "left", "right", "down"
    ]
    
    images_per_direction = 25  # 25 x 4 = 100 images
    
    print("=== Gaze Image Capture Tool ===")
    print("Instructions:")
    for i, direction in enumerate(directions):
        print(f"{i+1}. Look {direction.upper()}")
    print("\nPress SPACE to capture, Q to quit, N for next direction")
    
    current_dir_idx = 0
    captured_count = 0
    
    while current_dir_idx < len(directions):
        direction = directions[current_dir_idx]
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create a display frame with instructions (separate from capture frame)
        display_frame = frame.copy()
        
        # Add instructions only to display frame
        cv2.putText(display_frame, f"Look {direction.upper()}", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Captured: {captured_count}/{images_per_direction}", 
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(display_frame, "SPACE=Capture, N=Next Direction, Q=Quit", 
                    (50, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Show display frame with instructions
        cv2.imshow('Gaze Capture', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space to capture
            filename = f"{direction}_{captured_count+1:02d}.jpg"
            filepath = os.path.join("input_images", filename)
            
            # Save original frame WITHOUT text overlays
            cv2.imwrite(filepath, frame)
            print(f"Captured: {filename}")
            captured_count += 1
            
            if captured_count >= images_per_direction:
                current_dir_idx += 1
                captured_count = 0
                
        elif key == ord('n'):  # Next direction
            current_dir_idx += 1
            captured_count = 0
            
        elif key == ord('q'):  # Quit
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Capture completed! Check 'input_images' folder.")

if __name__ == "__main__":
    capture_gaze_images()