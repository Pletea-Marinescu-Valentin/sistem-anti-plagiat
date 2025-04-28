import os
import cv2
import time

# Forteaza utilizarea platformei X11
os.environ["QT_QPA_PLATFORM"] = "xcb"

def test_camera():
    print("Testare camera web...")
    cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    
    if not cap.isOpened():
        print("Nu s-a putut accesa camera web")
        return "Camera inaccesibila"
    
    # Afiseaza rezolutia camerei
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Rezolutia camerei: {int(width)}x{int(height)}")

    # Afiseaza un preview live al camerei
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Nu s-a putut citi frame-ul din camera")
            break

        cv2.imshow("Preview Camera", frame)

        # Inchide preview-ul daca utilizatorul apasa tasta 'q' sau dupa 30 de secunde
        if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time > 30):
            print("Preview inchis")
            break

    # Elibereaza resursele
    cap.release()
    cv2.destroyAllWindows()
    return "Test finalizat cu succes"

if __name__ == "__main__":
    rezultat = test_camera()
    print(rezultat)