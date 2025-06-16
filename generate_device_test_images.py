import cv2
import os
from datetime import datetime

def capture_device_images():
    """Capture 25 phone + 25 smartwatch images for testing object_detector.py"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    os.makedirs("device_test_images", exist_ok=True)
    
    # Simple scenarios: just phone and smartwatch
    scenarios = [
        ("phone", "PHONE", 25, (0, 0, 255)),      # Red
        ("smartwatch", "SMARTWATCH", 25, (255, 0, 0))  # Blue
    ]
    
    print("PHONE: Hold phone clearly visible in different positions")
    print("SMARTWATCH: Show smartwatch on wrist in various angles")
    print("\nControls: SPACE=Capture, N=Next category, Q=Quit")
    
    current_scenario_idx = 0
    captured_count = 0
    
    while current_scenario_idx < len(scenarios):
        scenario_name, display_name, target_count, color = scenarios[current_scenario_idx]
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create display frame
        display_frame = frame.copy()
        
        # Current scenario info
        cv2.putText(display_frame, f"{display_name}", 
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Progress
        cv2.putText(display_frame, f"Captured: {captured_count}/{target_count}", 
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Instructions based on scenario
        if scenario_name == "phone":
            instructions = [
                "Hold phone clearly visible",
                "Try different angles and positions", 
                "Make sure phone is recognizable",
                "Vary hand positions and orientations"
            ]
        else:  # smartwatch
            instructions = [
                "Show smartwatch on wrist clearly",
                "Try different wrist angles",
                "Make sure watch face is visible",
                "Vary arm positions and lighting"
            ]
        
        y_pos = 120
        for instruction in instructions:
            cv2.putText(display_frame, instruction, 
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_pos += 25
        
        # Controls
        cv2.putText(display_frame, "SPACE=Capture | N=Next Category | Q=Quit", 
                    (10, display_frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Progress bar
        progress = captured_count / target_count
        bar_width = 300
        bar_height = 20
        bar_x = display_frame.shape[1] - bar_width - 20
        bar_y = 20
        
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), color, -1)
        cv2.putText(display_frame, f"{captured_count}/{target_count}", 
                    (bar_x + 10, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Device Capture', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space to capture
            timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]  # Include milliseconds
            filename = f"{scenario_name}_{captured_count+1:02d}_{timestamp}.jpg"
            filepath = os.path.join("device_test_images", filename)
            
            # Save original frame WITHOUT overlays
            cv2.imwrite(filepath, frame)
            print(f"Captured: {filename}")
            captured_count += 1
            
            if captured_count >= target_count:
                current_scenario_idx += 1
                captured_count = 0
                if current_scenario_idx < len(scenarios):
                    print(f"\n{display_name} complete! Moving to next category...")
                
        elif key == ord('n'):  # Next category
            if captured_count > 0:
                current_scenario_idx += 1
                captured_count = 0
            else:
                print("Capture at least one image before moving to next category")
                
        elif key == ord('q'):  # Quit
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Images saved to 'device_test_images' folder")
    
    # Count actual files
    if os.path.exists("device_test_images"):
        files = [f for f in os.listdir("device_test_images") if f.endswith(('.jpg', '.jpeg', '.png'))]
        phone_files = [f for f in files if f.startswith('phone_')]
        watch_files = [f for f in files if f.startswith('smartwatch_')]
        
        print(f"Final count:")
        print(f"Phone images: {len(phone_files)}")
        print(f"Smartwatch images: {len(watch_files)}")
        print(f"Total: {len(files)}")

if __name__ == "__main__":
    capture_device_images()