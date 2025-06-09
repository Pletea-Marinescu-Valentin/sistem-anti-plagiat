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
from main import AntiPlagiarismSystem

class VideoProcessingThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    violation_detected = pyqtSignal(dict)

    def __init__(self, config, system):
        super().__init__()
        self.config = config
        self.system = system
        self.running = False
        self.paused = False

        self.frame_buffer = queue.Queue(maxsize=5)  # frame buffer

        # process every third frame for efficiency
        self.process_every_n_frames = 3
        self.frame_count = 0

    def run(self):
        try:
            print("Initializing webcam")
            cap = cv2.VideoCapture(0)  # Use webcam directly (id=0)

            if not cap.isOpened():
                print("Error: Could not access webcam")
                return

            print("Webcam initialized successfully")

            # camera resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

            self.running = True

            while self.running:
                if not self.paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Could not read frame from camera")
                        self.msleep(100)
                        continue

                    # process frame only at certain intervals
                    if self.frame_count % self.process_every_n_frames == 0:
                        try:
                            processed_frame = self.system.process_frame(frame)

                            # emit signals
                            self.frame_ready.emit(processed_frame)

                            # check if there are recent violations
                            recent_violations = self.system.get_recent_violations()
                            if recent_violations:
                                self.violation_detected.emit(recent_violations)
                        except Exception as e:
                            print(f"Error processing frame: {e}")

                # short pause to not overload CPU
                self.msleep(15)
        except Exception as e:
            print(f"Error in video processing thread: {e}")
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

        # load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.initialize_system()
        self.setup_ui()

        # initialize thread for video processing
        self.video_thread = VideoProcessingThread(self.config, self.system)
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.violation_detected.connect(self.handle_violation)

        self.recording = False
        self.monitoring = False
        self.current_frame = None
        self.violations_count = 0  # explicit counter for violations

    def initialize_system(self):
        self.system = AntiPlagiarismSystem(self.config)

    def setup_ui(self):
        # Main window configuration
        self.setWindowTitle("Anti-Plagiarism System")
        self.setGeometry(100, 100, 1200, 800)
        self.set_theme("dark")

        main_layout = QHBoxLayout()
        self.video_display = QLabel()
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setMinimumSize(640, 480)
        controls_layout = QVBoxLayout()
        system_group = QGroupBox("System Control")
        system_layout = QVBoxLayout()

        # button configuration
        self.start_button = QPushButton("Start Monitoring")
        self.start_button.clicked.connect(self.toggle_monitoring)

        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setEnabled(False)

        self.capture_button = QPushButton("Capture Snapshot")
        self.capture_button.clicked.connect(self.capture_snapshot)

        system_layout.addWidget(self.start_button)
        system_layout.addWidget(self.record_button)
        system_layout.addWidget(self.pause_button)
        system_layout.addWidget(self.capture_button)

        # add checkbox to select mirror mode
        self.mirror_check = QCheckBox("Mirror Image")
        self.mirror_check.setChecked(self.config["camera"]["mirror_image"])
        self.mirror_check.stateChanged.connect(self.toggle_mirror)
        system_layout.addWidget(self.mirror_check)

        system_group.setLayout(system_layout)

        # statistics group
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()

        # violation statistics
        self.violations_label = QLabel("Violations detected: 0")
        self.recording_time_label = QLabel("Recording time: 00:00:00")

        stats_layout.addWidget(self.violations_label)
        stats_layout.addWidget(self.recording_time_label)
        stats_group.setLayout(stats_layout)

        # report display area
        report_group = QGroupBox("Live Report")
        report_layout = QVBoxLayout()

        # text edit for live report
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)

        # add button to export report
        self.export_button = QPushButton("Export Report")
        self.export_button.clicked.connect(self.export_report)

        report_layout.addWidget(self.report_text)
        report_layout.addWidget(self.export_button)
        report_group.setLayout(report_layout)

        # add groups to controls layout
        controls_layout.addWidget(system_group)
        controls_layout.addWidget(stats_group)
        controls_layout.addWidget(report_group)

        # gaze limits configuration
        config_group = QGroupBox("Gaze Limits Configuration")
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

        save_button = QPushButton("Save Configuration")
        save_button.clicked.connect(self.save_config)

        config_layout.addRow("Left Limit (%)", self.left_limit_spinbox)
        config_layout.addRow("Right Limit (%)", self.right_limit_spinbox)
        config_layout.addRow("Down Limit (%)", self.down_limit_spinbox)
        config_layout.addWidget(save_button)
        config_group.setLayout(config_layout)

        controls_layout.addWidget(config_group)

        # add layouts to main layout
        main_layout.addWidget(self.video_display, 2)
        main_layout.addLayout(controls_layout, 1)

        # central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # timer for recording time update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_recording_time)
        self.recording_start_time = None

        # add placeholder for video display
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

            # button and spinbox styles
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
                QSpinBox {
                    background-color: #444444;
                    color: white;
                    border: 1px solid #555555;
                    border-radius: 3px;
                    padding: 2px;
                }
                QSpinBox::up-button, QSpinBox::down-button {
                    background-color: #666666;
                }
                QMessageBox {
                    background-color: #333333;
                    color: white;
                }
                QMessageBox QLabel {
                    color: white;
                }
                QMessageBox QPushButton {
                    background-color: #bbbbbb;
                    color: black;
                    font-weight: bold;
                    padding: 5px;
                    border-radius: 3px;
                    min-width: 80px;
                }
                QMessageBox QPushButton:hover {
                    background-color: #cccccc;
                }
            """)
        else:
            app.setPalette(app.style().standardPalette())
            self.setStyleSheet("")

    def display_placeholder(self):
        """Display a placeholder in the video display area"""
        placeholder = QPixmap(640, 480)
        placeholder.fill(QColor(200, 200, 200))
        self.video_display.setPixmap(placeholder)

    def update_frame(self, frame):
        """Update the displayed frame"""
        if frame is not None:
            try:
                # save current frame
                self.current_frame = frame.copy()

                # convert OpenCV frame to QImage
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                qt_frame = convert_to_qt_format.rgbSwapped()

                # resize to fit display area
                display_size = self.video_display.size()
                scaled_frame = qt_frame.scaled(display_size, Qt.KeepAspectRatio)

                # display frame
                self.video_display.setPixmap(QPixmap.fromImage(scaled_frame))
            except Exception as e:
                print(f"Error updating frame: {e}")

    def handle_violation(self, violation_data):
        try:
            # create current message
            current_message = f"<b>[{violation_data['timestamp']}]</b> {', '.join(violation_data['violations'])}"
            current_time = datetime.now().timestamp()

            # check time of last violation
            if not hasattr(self, 'last_violation_time'):
                self.last_violation_time = 0

            # check if at least 5 seconds have passed since last violation
            if current_time - self.last_violation_time >= 5:
                # update violation count
                self.violations_count += 1
                self.violations_label.setText(f"Violations detected: {self.violations_count}")

                # add message to report
                self.report_text.append(current_message)

                # update last violation time
                self.last_violation_time = current_time

        except Exception as e:
            print(f"Error handling violation: {e}")

    def update_recording_time(self):
        # update displayed recording time
        if self.recording_start_time:
            elapsed = datetime.now() - self.recording_start_time
            hours, remainder = divmod(elapsed.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.recording_time_label.setText(f"Recording time: {hours:02}:{minutes:02}:{seconds:02}")

    def toggle_monitoring(self):
        if not self.monitoring:
            # start monitoring
            self.start_button.setText("Stop Monitoring")
            self.pause_button.setEnabled(True)

            # set camera to 0 (laptop webcam)
            self.config["camera"]["index"] = 0

            # start video processing thread
            self.video_thread.start()
            self.monitoring = True
        else:
            # stop monitoring
            self.start_button.setText("Start Monitoring")
            self.pause_button.setEnabled(False)

            # stop video processing thread
            self.video_thread.stop()
            self.monitoring = False

            # if recording is on, stop it
            if self.recording:
                self.toggle_recording()

            self.display_placeholder()

    def toggle_recording(self):
        if not self.recording:
            # if monitoring is not started, start it
            if not self.monitoring:
                self.toggle_monitoring()

           # start recording
            self.system.start_recording()
            self.record_button.setText("Stop Recording")
            self.recording = True

            # start timer for recording time
            self.recording_start_time = datetime.now()
            self.timer.start(1000)  # Update every second
        else:
            # stop recording
            self.system.stop_recording()
            self.record_button.setText("Start Recording")
            self.recording = False

            # stop timer
            self.timer.stop()
            self.recording_time_label.setText("Recording time: 00:00:00")

            # show confirmation message
            QMessageBox.information(self, "Recording Stopped",
                                    f"Recording has been saved to:\n{self.system.get_recording_path()}")

    def toggle_pause(self):
        self.video_thread.pause()
        if self.video_thread.paused:
            self.pause_button.setText("Resume")
        else:
            self.pause_button.setText("Pause")

    def toggle_mirror(self, state):
        is_checked = state == Qt.Checked
        self.config["camera"]["mirror_image"] = is_checked
        if hasattr(self, 'system'):
            self.system.set_mirror_mode(is_checked)

    def capture_snapshot(self):
        if self.monitoring:
            path = self.system.capture_snapshot()
            if path:
                QMessageBox.information(self, "Snapshot Captured", f"Image has been saved to:\n{path}")
            else:
                QMessageBox.warning(self, "Error", "Could not save snapshot.")

    def export_report(self):
        if not hasattr(self.system, 'violation_monitor') or not self.system.violation_monitor.get_logs():
            QMessageBox.information(self, "Export Report", "No violations to report.")
            return

        # timestamp for files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export report in all 3 formats
        report_paths = []

        try:
            # Export to HTML
            html_path = os.path.join(self.config["reporting"]["save_path"], f"report_{timestamp}.html")
            self.system.export_report(html_path)
            report_paths.append(html_path)

            # Export to CSV
            csv_path = os.path.join(self.config["reporting"]["save_path"], f"report_{timestamp}.csv")
            self.system.export_report(csv_path)
            report_paths.append(csv_path)

            # Export to JSON
            json_path = os.path.join(self.config["reporting"]["save_path"], f"report_{timestamp}.json")
            self.system.export_report(json_path)
            report_paths.append(json_path)

            # Show confirmation message with all paths
            paths_text = "\n".join(report_paths)
            QMessageBox.information(self, "Export Report",
                                f"Reports have been exported successfully to:\n{paths_text}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error exporting report: {str(e)}")

    def save_config(self):
        self.config["detection"]["gaze"]["left_limit"] = self.left_limit_spinbox.value() / 100
        self.config["detection"]["gaze"]["right_limit"] = self.right_limit_spinbox.value() / 100
        self.config["detection"]["gaze"]["down_limit"] = self.down_limit_spinbox.value() / 100

        with open("config.json", "w") as f:
            json.dump(self.config, f, indent=4)

        # Real-time update of relevant components
        self.system.face_detector.gaze_tracker.left_limit = self.config["detection"]["gaze"]["left_limit"]
        self.system.face_detector.gaze_tracker.right_limit = self.config["detection"]["gaze"]["right_limit"]
        self.system.face_detector.gaze_tracker.down_limit = self.config["detection"]["gaze"]["down_limit"]

        QMessageBox.information(self, "Configuration Saved", "Limits have been saved and applied successfully!")

def main():
    app = QApplication(sys.argv)
    config_path = "config.json"

    # load configuration to access configured paths
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Create all necessary directories once
        recording_path = config.get("recording", {}).get("save_path", "./recordings")
        reporting_path = config.get("reporting", {}).get("save_path", "./reports")
        snapshot_path = config.get("snapshots", {}).get("save_path", "./snapshots")

        os.makedirs(recording_path, exist_ok=True)
        os.makedirs(reporting_path, exist_ok=True)
        os.makedirs(snapshot_path, exist_ok=True)

        print(f"Directories created/verified: {recording_path}, {reporting_path}, {snapshot_path}")
    except Exception as e:
        print(f"Error creating directories: {e}")
        # create directories with default names in case of error
        os.makedirs("recordings", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        os.makedirs("snapshots", exist_ok=True)

    window = AntiPlagiatGUI(config_path)
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()