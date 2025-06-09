import os
import cv2
import time

# Force X11 platform usage
os.environ["QT_QPA_PLATFORM"] = "xcb"

def test_camera():
    print("Testing webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    
    if not cap.isOpened():
        print("Could not access webcam")
        return "Camera inaccessible"
    
    # Display camera resolution
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera resolution: {int(width)}x{int(height)}")

    # Display live camera preview
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame from camera")
            break

        cv2.imshow("Camera Preview", frame)

        # Close preview when user presses 'q' key or after 30 seconds
        if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time > 30):
            print("Preview closed")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    return "Test completed successfully"

if __name__ == "__main__":
    result = test_camera()
    print(result)