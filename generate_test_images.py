import sys
import os
import cv2
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

class GazeCaptureGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_camera()
        self.setup_capture_logic()
        
    def setup_ui(self):
        self.setWindowTitle("Gaze Capture Tool")
        self.setGeometry(100, 100, 1000, 700)
        self.set_dark_theme()
        
        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid #555; background-color: #222;")
        self.video_label.setAlignment(Qt.AlignCenter)
        
        # Controls panel
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_widget.setMaximumWidth(300)
        
        # Person info
        self.person_label = QLabel("PERSON #1")
        self.person_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.person_label.setAlignment(Qt.AlignCenter)
        
        # Current direction
        self.direction_label = QLabel("Look CENTER")
        self.direction_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.direction_label.setAlignment(Qt.AlignCenter)
        self.direction_label.setStyleSheet("color: #00ff00; padding: 10px;")
        
        # Progress bars
        self.current_progress = QProgressBar()
        self.current_progress.setMaximum(25)
        self.current_progress.setValue(0)
        
        self.total_progress = QProgressBar()
        self.total_progress.setMaximum(100)
        self.total_progress.setValue(0)
        
        # Buttons
        self.capture_button = QPushButton("CAPTURE (SPACE)")
        self.capture_button.clicked.connect(self.capture_image)
        self.capture_button.setFont(QFont("Arial", 12, QFont.Bold))
        
        self.next_button = QPushButton("NEXT DIRECTION (N)")
        self.next_button.clicked.connect(self.next_direction)
        
        self.quit_button = QPushButton("QUIT (Q)")
        self.quit_button.clicked.connect(self.close)
        
        # Instructions
        instructions = QLabel("""
INSTRUCTIONS:
1. Look CENTER (Green)
2. Look LEFT (Blue) 
3. Look RIGHT (Cyan)
4. Look DOWN (Red)

CONTROLS:
• SPACE or Click = Capture
• N or Next = Next Direction
• Q or Quit = Exit

25 images per direction
100 images total
        """)
        instructions.setStyleSheet("color: #ccc; font-size: 10px; padding: 10px;")
        
        # Add to controls layout
        controls_layout.addWidget(self.person_label)
        controls_layout.addWidget(self.direction_label)
        controls_layout.addWidget(QLabel("Current Direction:"))
        controls_layout.addWidget(self.current_progress)
        controls_layout.addWidget(QLabel("Total Progress:"))
        controls_layout.addWidget(self.total_progress)
        controls_layout.addWidget(self.capture_button)
        controls_layout.addWidget(self.next_button)
        controls_layout.addWidget(self.quit_button)
        controls_layout.addWidget(instructions)
        controls_layout.addStretch()
        
        # Add to main layout
        main_layout.addWidget(self.video_label, 2)
        main_layout.addWidget(controls_widget, 1)
        
    def set_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        self.setPalette(palette)
        
        self.setStyleSheet("""
            QPushButton {
                background-color: #666;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
                border: 1px solid #888;
            }
            QPushButton:hover {
                background-color: #777;
            }
            QPushButton:pressed {
                background-color: #555;
            }
            QProgressBar {
                border: 1px solid #666;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
    
    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open camera")
            return
            
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Timer for video updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms = ~33 FPS
        
    def setup_capture_logic(self):
        # Create input_images directory
        os.makedirs("input_images", exist_ok=True)
        
        # Get next person number
        self.person_number = self.get_next_person_number()
        self.person_label.setText(f"PERSON #{self.person_number}")
        
        # Directions setup
        self.directions = [
            ("center", "CENTER", "#00ff00"),    # Green
            ("left", "LEFT", "#0000ff"),        # Blue  
            ("right", "RIGHT", "#00ffff"),      # Cyan
            ("down", "DOWN", "#ff0000")         # Red
        ]
        
        self.current_dir_idx = 0
        self.captured_count = 0
        self.total_captured = 0
        self.images_per_direction = 25
        
        self.update_direction_display()
        
    def get_next_person_number(self):
        if not os.path.exists("input_images"):
            return 1
        
        files = os.listdir("input_images")
        person_numbers = set()
        
        for file in files:
            if '_' in file and file.endswith('.jpg'):
                parts = file.split('_')
                if len(parts) >= 3:
                    try:
                        person_num = int(parts[1])
                        person_numbers.add(person_num)
                    except ValueError:
                        continue
        
        return max(person_numbers) + 1 if person_numbers else 1
    
    def update_direction_display(self):
        if self.current_dir_idx < len(self.directions):
            direction, display_name, color = self.directions[self.current_dir_idx]
            self.direction_label.setText(f"Look {display_name}")
            self.direction_label.setStyleSheet(f"color: {color}; padding: 10px; font-weight: bold;")
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
            
        self.current_frame = frame.copy()
        
        # Convert BGR to RGB for Qt display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert frame to Qt format
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit display
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
    
    def capture_image(self):
        if not hasattr(self, 'current_frame'):
            return
            
        if self.current_dir_idx >= len(self.directions):
            return
            
        direction = self.directions[self.current_dir_idx][0]
        
        # Save image
        filename = f"{direction}_{self.person_number:02d}_{self.captured_count+1:02d}.jpg"
        filepath = os.path.join("input_images", filename)
        
        cv2.imwrite(filepath, self.current_frame)
        print(f"Captured: {filename}")
        
        # Update counters
        self.captured_count += 1
        self.total_captured += 1
        
        # Update progress bars
        self.current_progress.setValue(self.captured_count)
        self.total_progress.setValue(self.total_captured)
        
        # Check if direction is complete
        if self.captured_count >= self.images_per_direction:
            self.next_direction()
    
    def next_direction(self):
        self.current_dir_idx += 1
        self.captured_count = 0
        self.current_progress.setValue(0)
        
        if self.current_dir_idx < len(self.directions):
            self.update_direction_display()
            direction_name = self.directions[self.current_dir_idx][1]
            print(f"Moving to {direction_name}...")
        else:
            self.finish_session()
    
    def finish_session(self):
        print(f"\n=== SESSION COMPLETE FOR PERSON #{self.person_number} ===")
        self.show_completion_summary()
        self.close()
    
    def show_completion_summary(self):
        # Count files for this person
        files = [f for f in os.listdir("input_images") if f.endswith('.jpg')]
        person_files = [f for f in files if f.split('_')[1] == f"{self.person_number:02d}"]
        
        center_files = len([f for f in person_files if f.startswith('center_')])
        left_files = len([f for f in person_files if f.startswith('left_')])
        right_files = len([f for f in person_files if f.startswith('right_')])
        down_files = len([f for f in person_files if f.startswith('down_')])
        
        summary = f"""
Summary for Person #{self.person_number}:
CENTER: {center_files} images
LEFT: {left_files} images  
RIGHT: {right_files} images
DOWN: {down_files} images
TOTAL: {len(person_files)} images

Total images in folder: {len(files)}
Next person will be Person #{self.person_number + 1}
        """
        
        print(summary)
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.capture_image()
        elif event.key() == Qt.Key_N:
            self.next_direction()
        elif event.key() == Qt.Key_Q:
            self.close()
        else:
            super().keyPressEvent(event)
    
    def closeEvent(self, event):
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'timer'):
            self.timer.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = GazeCaptureGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()