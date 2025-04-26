import cv2

def test_camera():
    print("Testare camera web...")
    cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    
    if not cap.isOpened():
        print("Nu s-a putut accesa camera web")
        return False
    
    ret, frame = cap.read()
    if not ret:
        print("Nu s-a putut citi frame-ul din camera")
        cap.release()
        return False
    
    # Salveaza un frame pentru a verifica
    cv2.imwrite("test_camera.jpg", frame)
    print("Imagine de test salvata in 'test_camera.jpg'")
    
    cap.release()
    return True

if __name__ == "__main__":
    test_camera()