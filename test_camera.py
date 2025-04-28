import cv2

def test_camera():
    print("Testare camera web...")
    cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    
    if not cap.isOpened():
        print("Nu s-a putut accesa camera web")
        return False
    
    # Afiseaza rezoluția camerei
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Rezoluția camerei: {int(width)}x{int(height)}")

    # Afiseaza un preview live al camerei
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Nu s-a putut citi frame-ul din camera")
            break

        cv2.imshow("Preview Camera", frame)

        # Inchide preview-ul daca utilizatorul apasa tasta 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    test_camera()