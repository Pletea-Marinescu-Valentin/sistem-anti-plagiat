import cv2
import os
import numpy as np
from datetime import datetime
from modules.face_detector import FaceDetector
from modules.object_detector import ObjectDetector
from modules.violation_monitor import ViolationMonitor
from modules.video_handler import VideoHandler
from modules.report_generator import ReportGenerator
from modules.data_exporter import DataExporter
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("anti_plagiarism_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AntiPlagiarismSystem:
    def __init__(self, config):
        self.config = config
        self.mirror_image = config["camera"]["mirror_image"]

        # timestamp for files
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.face_detector = FaceDetector(self.mirror_image)
        self.object_detector = ObjectDetector(self.config)
        self.violation_monitor = ViolationMonitor()
        self.video_handler = VideoHandler(self.mirror_image, self.config.get("recording", {}).get("save_path", "./recordings"))

        self.recording = False
        self.video_writer = None
        self.recording_path = None

    def process_frame(self, frame):
        # process a frame and detect violations
        if frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        direction, frame, h_ratio, v_ratio = self.face_detector.detect_direction(frame)
        objects, frame = self.object_detector.detect_objects(frame)
        violations = self.violation_monitor.check_violations(direction, objects)

        if violations:
            self.violation_monitor.log_violation(violations)

        display_frame = self.video_handler.prepare_frame_for_display(frame, violations, h_ratio, v_ratio)

        if self.recording:
            if self.video_writer is None and frame is not None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    self.recording_path, fourcc, 20.0, (w, h)
                )
                logger.info(f"Video recording at: {self.recording_path}")

            if self.video_writer is not None:
                self.video_writer.write(display_frame)

        # Save last processed frame for captures
        self.last_processed_frame = frame.copy()

        return display_frame

    def get_recent_violations(self):
        logs = self.violation_monitor.get_logs()
        if logs and len(logs) > 0:
            return logs[-1]  # most recent record
        return None

    def start_recording(self):
        if self.recording:
            return self.recording_path

        save_path = self.config.get("recording", {}).get("save_path", "./recordings")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_path = os.path.join(save_path, f"recording_{timestamp}.mp4")
        self.recording = True

        return self.recording_path

    def stop_recording(self):
        if not self.recording:
            return None

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        self.recording = False
        return self.recording_path

    def get_recording_path(self):
        return self.recording_path

    def capture_snapshot(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # ensure directory exists
            snapshot_dir = self.config.get("snapshots", {}).get("save_path", "./snapshots")
            os.makedirs(snapshot_dir, exist_ok=True)
            
            snapshot_path = os.path.join(snapshot_dir, f"snapshot_{timestamp}.jpg")
            
            # Check if monitoring is active
            if hasattr(self, 'last_processed_frame') and self.last_processed_frame is not None:
                # Use last processed frame instead of opening camera again
                cv2.imwrite(snapshot_path, self.last_processed_frame)
                return snapshot_path
            else:
                # Open camera only if we don't have an existing frame
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    logger.error("Could not access camera")
                    return None

                ret, frame = cap.read()
                cap.release()

                if ret:
                    cv2.imwrite(snapshot_path, frame)
                    return snapshot_path
                return None
        except Exception as e:
            logger.exception(f"Capture error: {e}")
            return None

    def export_report(self, file_path):
        logs = self.violation_monitor.get_logs()
        if not logs:
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # get save path from configuration
        report_save_path = self.config.get("reporting", {}).get("save_path", "./reports")

        try:
            if file_path.endswith(".html"):
                # pass correct path to ReportGenerator initialization
                report_gen = ReportGenerator(timestamp, save_path=report_save_path)
                report_gen.generate_html_report(logs, self.recording_path)
                return True
            elif file_path.endswith(".csv"):
                exporter = DataExporter(self.config)
                result = exporter.export_data(logs, timestamp)
                return bool(result)
            elif file_path.endswith(".json"):
                exporter = DataExporter(self.config)
                result = exporter.export_data(logs, timestamp)
                return bool(result)
            return False
        except Exception as e:
            logger.exception(f"Export error: {e}")
            return False

    def set_mirror_mode(self, mirror_mode):
        # mirror mode
        self.mirror_image = mirror_mode
        self.face_detector.mirror_image = mirror_mode
        self.video_handler.mirror_image = mirror_mode