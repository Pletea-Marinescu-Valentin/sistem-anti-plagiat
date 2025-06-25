import cv2
import mediapipe as mp
import numpy as np
import os

class MediaPipeTest:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Configurare MediaPipe
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,  # Pentru detectie mai precisa
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Indicii pentru ochi din MediaPipe (468 puncte)
        # Ochiul stang (din perspectiva subiectului)
        self.LEFT_EYE_INDICES = [
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
        ]
        
        # Ochiul drept
        self.RIGHT_EYE_INDICES = [
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
        ]
        
        # Centrul pupilei (aproximativ)
        self.LEFT_PUPIL_INDEX = 468  # Primul punct din iris landmarks
        self.RIGHT_PUPIL_INDEX = 473 # Al doilea punct din iris landmarks

    def test_single_image(self, image_path):
        """Testează MediaPipe pe o singură imagine"""
        print(f"\n=== Testând: {os.path.basename(image_path)} ===")
        
        # Încarcă imaginea
        image = cv2.imread(image_path)
        if image is None:
            print(f"Nu pot încărca imaginea: {image_path}")
            return None
            
        # Convertește la RGB pentru MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Procesează cu MediaPipe
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            print("❌ Nu s-a detectat nicio față")
            return None
            
        print("✅ Față detectată!")
        
        # Obține landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convertește landmarks la coordonate pixel
        height, width = image.shape[:2]
        landmarks_px = []
        
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmarks_px.append((x, y))
        
        # Analizează ochii
        left_eye_detected = self.analyze_eye(landmarks_px, self.LEFT_EYE_INDICES, "STÂNG")
        right_eye_detected = self.analyze_eye(landmarks_px, self.RIGHT_EYE_INDICES, "DREPT")
        
        # Calculează raporturile aproximative (simplificate pentru test)
        if left_eye_detected and right_eye_detected:
            h_ratio, v_ratio = self.calculate_gaze_ratios(landmarks_px)
            print(f"📊 H-Ratio: {h_ratio:.3f}, V-Ratio: {v_ratio:.3f}")
        else:
            print("⚠️  Nu pot calcula raporturile - ochi nedetectați")
            h_ratio, v_ratio = 0.5, 0.5
        
        # Creează imagine annotată
        annotated_image = self.create_annotated_image(image, landmarks_px, h_ratio, v_ratio)
        
        return annotated_image, h_ratio, v_ratio
    
    def analyze_eye(self, landmarks_px, eye_indices, eye_name):
        """Analizează dacă un ochi este detectat corect"""
        try:
            eye_points = [landmarks_px[i] for i in eye_indices]
            
            # Verifică dacă punctele sunt în limitele imaginii
            valid_points = 0
            for x, y in eye_points:
                if 0 <= x < 2000 and 0 <= y < 2000:  # Limite rezonabile
                    valid_points += 1
            
            detection_rate = valid_points / len(eye_points)
            print(f"   👁️  Ochi {eye_name}: {valid_points}/{len(eye_points)} puncte valide ({detection_rate*100:.1f}%)")
            
            return detection_rate > 0.7  # Considerăm detectat dacă >70% puncte sunt valide
            
        except Exception as e:
            print(f"   ❌ Eroare la analiza ochiului {eye_name}: {e}")
            return False
    
    def calculate_gaze_ratios(self, landmarks_px):
        """Calculează raporturile H și V simplificate"""
        try:
            # Folosește puncte cheie pentru calcul rapid
            # Punctul 1: vârful nasului
            nose_tip = landmarks_px[1]
            
            # Punctele 127 și 356: marginile feței
            left_face = landmarks_px[127]
            right_face = landmarks_px[356]
            
            # Calculează H-ratio
            total_width = abs(right_face[0] - left_face[0])
            if total_width > 0:
                nose_offset = nose_tip[0] - left_face[0]
                h_ratio = nose_offset / total_width
            else:
                h_ratio = 0.5
            
            # Calculează V-ratio (simplu - bazat pe poziția nasului față de ochi)
            # Punctele 10 și 152: partea de sus și jos a feței
            top_face = landmarks_px[10]
            bottom_face = landmarks_px[152]
            
            total_height = abs(bottom_face[1] - top_face[1])
            if total_height > 0:
                nose_offset_v = nose_tip[1] - top_face[1]
                v_ratio = nose_offset_v / total_height
            else:
                v_ratio = 0.5
            
            # Limitează valorile la 0-1
            h_ratio = max(0.0, min(1.0, h_ratio))
            v_ratio = max(0.0, min(1.0, v_ratio))
            
            return h_ratio, v_ratio
            
        except Exception as e:
            print(f"   ❌ Eroare la calculul raporturilor: {e}")
            return 0.5, 0.5
    
    def create_annotated_image(self, image, landmarks_px, h_ratio, v_ratio):
        """Creează imagine annotată pentru debugging"""
        annotated = image.copy()
        
        # Desenează câteva puncte cheie
        key_points = [1, 10, 152, 127, 356]  # nas, sus, jos, stânga, dreapta față
        
        for i in key_points:
            if i < len(landmarks_px):
                x, y = landmarks_px[i]
                cv2.circle(annotated, (x, y), 3, (0, 255, 0), -1)
                cv2.putText(annotated, str(i), (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Desenează ochii
        for eye_indices, color in [(self.LEFT_EYE_INDICES, (255, 0, 0)), (self.RIGHT_EYE_INDICES, (0, 0, 255))]:
            for i in eye_indices:
                if i < len(landmarks_px):
                    x, y = landmarks_px[i]
                    cv2.circle(annotated, (x, y), 1, color, -1)
        
        # Adaugă text cu ratios
        cv2.putText(annotated, f"H: {h_ratio:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(annotated, f"V: {v_ratio:.3f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        return annotated

    def test_all_images(self, input_dir="input_images", output_dir="mediapipe_test_results"):
        """Testează toate imaginile din directorul de input"""
        
        if not os.path.exists(input_dir):
            print(f"❌ Directorul {input_dir} nu există!")
            return
        
        # Creează directorul de output
        os.makedirs(output_dir, exist_ok=True)
        
        # Găsește toate imaginile
        image_files = []
        for file in os.listdir(input_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(file)
        
        if not image_files:
            print(f"❌ Nu s-au găsit imagini în {input_dir}")
            return
        
        print(f"🚀 Testez {len(image_files)} imagini cu MediaPipe...")
        
        success_count = 0
        total_count = len(image_files)
        
        for filename in sorted(image_files):
            input_path = os.path.join(input_dir, filename)
            result = self.test_single_image(input_path)
            
            if result is not None:
                annotated_image, h_ratio, v_ratio = result
                
                # Salvează imaginea annotată
                output_filename = f"mp_{filename}"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, annotated_image)
                
                success_count += 1
                print(f"💾 Salvat: {output_filename}")
            
            print("-" * 50)
        
        print(f"\n🎯 REZULTATE FINALE:")
        print(f"   Succese: {success_count}/{total_count}")
        print(f"   Rata de succes: {(success_count/total_count)*100:.1f}%")
        print(f"   Imagini salvate în: {output_dir}/")


def main():
    tester = MediaPipeTest()
    tester.test_all_images()


if __name__ == "__main__":
    main()