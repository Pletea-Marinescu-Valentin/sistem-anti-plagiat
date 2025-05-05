import queue
import sys
import cv2
import json
import os
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QPushButton, QGroupBox, QCheckBox, QTextEdit, QMessageBox, QSpinBox, QFormLayout)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QColor, QPalette
from main import SistemAntiPlagiat

class VideoProcessingThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    violation_detected = pyqtSignal(dict)

    def __init__(self, config, system):
        super().__init__()
        self.config = config
        self.system = system
        self.running = False
        self.paused = False

        self.frame_buffer = queue.Queue(maxsize=5) # buffer pentru frame-uri

        # procesam unul din trei cadre pentru eficienta
        self.process_every_n_frames = 3
        self.frame_count = 0

    def run(self):
        try:
            print("Initializare camera web")
            cap = cv2.VideoCapture(0)  # Folosim direct camera web(id=0)

            if not cap.isOpened():
                print("Eroare: Nu s-a putut accesa camera web")
                return

            print("Camera web initializata cu succes")

            # rezolutie camera
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

            self.running = True

            while self.running:
                if not self.paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("Eroare: Nu s-a putut citi frame-ul din camera")
                        self.msleep(100)
                        continue

                    # procesare frame doar la anumite intervale
                    if self.frame_count % self.process_every_n_frames == 0:
                        try:
                            processed_frame = self.system.process_frame(frame)

                            # emite semnale
                            self.frame_ready.emit(processed_frame)

                            # verifica daca exista incalcari recente
                            recent_violations = self.system.get_recent_violations()
                            if recent_violations:
                                self.violation_detected.emit(recent_violations)
                        except Exception as e:
                            print(f"Eroare la procesarea frame-ului: {e}")
                            
                # o scurta pauza pentru a nu supraincarca CPU
                self.msleep(15)
        except Exception as e:
            print(f"Eroare in thread-ul de procesare video: {e}")
        finally:
            if 'cap' in locals() and cap is not None:
                cap.release()

    def stop(self):
        self.running = False
        self.wait()

    def pause(self):
        self.paused = not self.paused


class AntiPlagiatGUI(QMainWindow):
    def __init__(self, config_path="config.json"):
        super().__init__()

        # incarcare configuratie
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.initialize_system()
        self.setup_ui()

        # initializare thread pentru procesare video
        self.video_thread = VideoProcessingThread(self.config, self.system)
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.violation_detected.connect(self.handle_violation)

        self.recording = False
        self.monitoring = False
        self.current_frame = None
        self.violations_count = 0  # contor explicit pentru incalcari

    def initialize_system(self):
        self.system = SistemAntiPlagiat(self.config)

    def setup_ui(self):
        # Configurare fereastra principala
        self.setWindowTitle("Sistem Anti-Plagiat")
        self.setGeometry(100, 100, 1200, 800)
        self.set_theme("dark")

        main_layout = QHBoxLayout()
        self.video_display = QLabel()
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setMinimumSize(640, 480)
        controls_layout = QVBoxLayout()
        system_group = QGroupBox("Control Sistem")
        system_layout = QVBoxLayout()

        # configurare butoane
        self.start_button = QPushButton("Start Monitorizare")
        self.start_button.clicked.connect(self.toggle_monitoring)

        self.record_button = QPushButton("Start Inregistrare")
        self.record_button.clicked.connect(self.toggle_recording)

        self.pause_button = QPushButton("Pauza")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setEnabled(False)

        self.capture_button = QPushButton("Captura Instantanee")
        self.capture_button.clicked.connect(self.capture_snapshot)

        system_layout.addWidget(self.start_button)
        system_layout.addWidget(self.record_button)
        system_layout.addWidget(self.pause_button)
        system_layout.addWidget(self.capture_button)
        system_group.setLayout(system_layout)

        # adugare checkbox pentru a selecta modul oglindire
        self.mirror_check = QCheckBox("Oglindire Imagine")
        self.mirror_check.setChecked(self.config["camera"]["mirror_image"])
        self.mirror_check.stateChanged.connect(self.toggle_mirror)

        # grup de statistici
        stats_group = QGroupBox("Statistici")
        stats_layout = QVBoxLayout()

        # statisticile despre incalcari
        self.violations_label = QLabel("Incalcari detectate: 0")
        self.recording_time_label = QLabel("Timp inregistrare: 00:00:00")

        stats_layout.addWidget(self.violations_label)
        stats_layout.addWidget(self.recording_time_label)
        stats_group.setLayout(stats_layout)

        # zona de afisare a raportului
        report_group = QGroupBox("Raport Live")
        report_layout = QVBoxLayout()

        # text edit pentru raport live
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)

        # adaugare buton pentru a da export la raport
        self.export_button = QPushButton("Export Raport")
        self.export_button.clicked.connect(self.export_report)

        report_layout.addWidget(self.report_text)
        report_layout.addWidget(self.export_button)
        report_group.setLayout(report_layout)

        # adauga grupurile la layout-ul de controale
        controls_layout.addWidget(system_group)
        controls_layout.addWidget(stats_group)
        controls_layout.addWidget(report_group)

        # configurare limite privire
        config_group = QGroupBox("Configurare Limite Privire")
        config_layout = QFormLayout()

        self.left_limit_spinbox = QSpinBox()
        self.left_limit_spinbox.setRange(0, 100)
        self.left_limit_spinbox.setValue(int(self.config["detection"]["gaze"]["left_limit"] * 100))

        self.right_limit_spinbox = QSpinBox()
        self.right_limit_spinbox.setRange(0, 100)
        self.right_limit_spinbox.setValue(int(self.config["detection"]["gaze"]["right_limit"] * 100))

        self.down_limit_spinbox = QSpinBox()
        self.down_limit_spinbox.setRange(0, 100)
        self.down_limit_spinbox.setValue(int(self.config["detection"]["gaze"]["down_limit"] * 100))

        save_button = QPushButton("Salveaza Configuratia")
        save_button.clicked.connect(self.save_config)

        config_layout.addRow("Limita Stanga (%)", self.left_limit_spinbox)
        config_layout.addRow("Limita Dreapta (%)", self.right_limit_spinbox)
        config_layout.addRow("Limita Jos (%)", self.down_limit_spinbox)
        config_layout.addWidget(save_button)
        config_group.setLayout(config_layout)

        controls_layout.addWidget(config_group)

        # adauga layout-urile la layout-ul principal
        main_layout.addWidget(self.video_display, 2)
        main_layout.addLayout(controls_layout, 1)

        # widget central
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # timer pentru actualizarea timpului de inregistrare
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_recording_time)
        self.recording_start_time = None

        # adauga placeholder pentru afisarea video
        self.display_placeholder()

    def set_theme(self, theme_name):
        app = QApplication.instance()
        if theme_name == "dark":
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor(53, 53, 53))
            palette.setColor(QPalette.WindowText, Qt.white)
            palette.setColor(QPalette.Base, QColor(25, 25, 25))
            palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            palette.setColor(QPalette.Text, Qt.white)
            palette.setColor(QPalette.Button, QColor(53, 53, 53))
            palette.setColor(QPalette.ButtonText, Qt.black)
            palette.setColor(QPalette.BrightText, Qt.red)
            palette.setColor(QPalette.Link, QColor(42, 130, 218))
            palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            app.setPalette(palette)

            # stil butoane
            self.setStyleSheet("""
                QPushButton {
                    background-color: #bbbbbb;
                    color: black;
                    font-weight: bold;
                    padding: 5px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #cccccc;
                }
                QPushButton:pressed {
                    background-color: #aaaaaa;
                }
            """)
        else:
            app.setPalette(app.style().standardPalette())
            self.setStyleSheet("")

    def display_placeholder(self):
        """Afiseaza un placeholder in zona de afisare video"""
        placeholder = QPixmap(640, 480)
        placeholder.fill(QColor(200, 200, 200))
        self.video_display.setPixmap(placeholder)

    def update_frame(self, frame):
        """Actualizeaza frame-ul afisat"""
        if frame is not None:
            try:
                # salvam frame-ul curent
                self.current_frame = frame.copy()

                # converteste frame-ul OpenCV in QImage
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                qt_frame = convert_to_qt_format.rgbSwapped()

                # redimensioneaza pentru a se potrivi in zona
                display_size = self.video_display.size()
                scaled_frame = qt_frame.scaled(display_size, Qt.KeepAspectRatio)

                # afiseaza frame-ul
                self.video_display.setPixmap(QPixmap.fromImage(scaled_frame))
            except Exception as e:
                print(f"Eroare la actualizarea frame-ului: {e}")

    def handle_violation(self, violation_data):
        try:
            # creeaza mesajul curent
            current_message = f"<b>[{violation_data['timestamp']}]</b> {', '.join(violation_data['violations'])}"
            current_time = datetime.now().timestamp()

            # verificam timpul ultimei incalcari
            if not hasattr(self, 'last_violation_time'):
                self.last_violation_time = 0

            # verifica daca au trecut cel putin 5 secunde de la ultima incalcare
            if current_time - self.last_violation_time >= 5:
                # actualizeaza numarul de incalcari
                self.violations_count += 1
                self.violations_label.setText(f"Incalcari detectate: {self.violations_count}")

                # adauga mesajul in raport
                self.report_text.append(current_message)

                # actualizeaza timpul ultimei incalcari
                self.last_violation_time = current_time

        except Exception as e:
            print(f"Eroare la gestionarea incalcarii: {e}")

    def update_recording_time(self):
        # actualizeaza timpul de inregistrare afisat
        if self.recording_start_time:
            elapsed = datetime.now() - self.recording_start_time
            hours, remainder = divmod(elapsed.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.recording_time_label.setText(f"Timp inregistrare: {hours:02}:{minutes:02}:{seconds:02}")

    def toggle_monitoring(self):
        if not self.monitoring:
            # porneste monitorizarea
            self.start_button.setText("Stop Monitorizare")
            self.pause_button.setEnabled(True)

            # seteaza camera la 0 (camera web de pe laptop)
            self.config["camera"]["index"] = 0

            # porneste thread-ul de procesare video
            self.video_thread.start()
            self.monitoring = True
        else:
            # opreste monitorizarea
            self.start_button.setText("Start Monitorizare")
            self.pause_button.setEnabled(False)

            # opreste thread-ul de procesare video
            self.video_thread.stop()
            self.monitoring = False

            # daca inregistrarea este pornita, o oprim
            if self.recording:
                self.toggle_recording()

            self.display_placeholder()

    def toggle_recording(self):
        if not self.recording:
            # daca monitorizarea nu este pornita, o pornim
            if not self.monitoring:
                self.toggle_monitoring()

           # porneste inregistrarea
            self.system.start_recording()
            self.record_button.setText("Stop Inregistrare")
            self.recording = True

            # porneste timer-ul pentru timpul de inregistrare
            self.recording_start_time = datetime.now()
            self.timer.start(1000)  # Actualizeaza la fiecare secunda
        else:
            # opreste inregistrarea
            self.system.stop_recording()
            self.record_button.setText("Start Inregistrare")
            self.recording = False

            # opreste timer-ul
            self.timer.stop()
            self.recording_time_label.setText("Timp inregistrare: 00:00:00")

            # afiseaza mesaj de confirmare
            QMessageBox.information(self, "Inregistrare Oprita",
                                    f"Inregistrarea a fost salvata in:\n{self.system.get_recording_path()}")

    def toggle_pause(self):
        self.video_thread.pause()
        if self.video_thread.paused:
            self.pause_button.setText("Reluare")
        else:
            self.pause_button.setText("Pauza")

    def toggle_mirror(self, state):
        is_checked = state == Qt.Checked
        self.config["camera"]["mirror_image"] = is_checked
        if hasattr(self, 'system'):
            self.system.set_mirror_mode(is_checked)

    def capture_snapshot(self):
        if self.monitoring:
            path = self.system.capture_snapshot()
            if path:
                QMessageBox.information(self, "Captura Realizata", f"Imaginea a fost salvata in:\n{path}")
            else:
                QMessageBox.warning(self, "Eroare", "Nu s-a putut salva captura.")

    def export_report(self):
        if not hasattr(self.system, 'violation_monitor') or not self.system.violation_monitor.get_logs():
            QMessageBox.information(self, "Export Raport", "Nu exista incalcari de raportat.")
            return

        # timestamp pentru fișiere
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export raport in toate cele 3 formate
        report_paths = []

        try:
            # Exportam în HTML
            html_path = os.path.join(self.config["reporting"]["save_path"], f"raport_{timestamp}.html")
            self.system.export_report(html_path)
            report_paths.append(html_path)

            # Exportam în CSV
            csv_path = os.path.join(self.config["reporting"]["save_path"], f"raport_{timestamp}.csv")
            self.system.export_report(csv_path)
            report_paths.append(csv_path)

            # Exportam în JSON
            json_path = os.path.join(self.config["reporting"]["save_path"], f"raport_{timestamp}.json")
            self.system.export_report(json_path)
            report_paths.append(json_path)

            # Afisare mesaj de confirmare cu toate caile
            paths_text = "\n".join(report_paths)
            QMessageBox.information(self, "Export Raport",
                                f"Rapoartele au fost exportate cu succes in:\n{paths_text}")
        except Exception as e:
            QMessageBox.critical(self, "Eroare", f"Eroare la exportul raportului: {str(e)}")

    def save_config(self):
        self.config["detection"]["gaze"]["left_limit"] = self.left_limit_spinbox.value() / 100
        self.config["detection"]["gaze"]["right_limit"] = self.right_limit_spinbox.value() / 100
        self.config["detection"]["gaze"]["down_limit"] = self.down_limit_spinbox.value() / 100

        with open("config.json", "w") as f:
            json.dump(self.config, f, indent=4)

        # Actualizare în timp real a componentelor relevante
        self.system.face_detector.gaze_tracker.left_limit = self.config["detection"]["gaze"]["left_limit"]
        self.system.face_detector.gaze_tracker.right_limit = self.config["detection"]["gaze"]["right_limit"]
        self.system.face_detector.gaze_tracker.down_limit = self.config["detection"]["gaze"]["down_limit"]

        QMessageBox.information(self, "Configurare Salvata", "Limitele au fost salvate și aplicate cu succes!")

def main():
    app = QApplication(sys.argv)
    config_path = "config.json"

    # incarcare configuratie pentru a avea acces la caile configurate
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Creeaza toate directoarele necesare o singura data
        recording_path = config.get("recording", {}).get("save_path", "./recordings")
        reporting_path = config.get("reporting", {}).get("save_path", "./reports")
        snapshot_path = config.get("snapshots", {}).get("save_path", "./snapshots")

        os.makedirs(recording_path, exist_ok=True)
        os.makedirs(reporting_path, exist_ok=True)
        os.makedirs(snapshot_path, exist_ok=True)

        print(f"Directoare create/verificate: {recording_path}, {reporting_path}, {snapshot_path}")
    except Exception as e:
        print(f"Eroare la crearea directoarelor: {e}")
        # creare directoare cu nume implicite in caz de eroare
        os.makedirs("recordings", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        os.makedirs("snapshots", exist_ok=True)

    window = AntiPlagiatGUI(config_path)
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()