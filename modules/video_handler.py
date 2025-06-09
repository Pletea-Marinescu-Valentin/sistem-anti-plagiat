import cv2
import time
import numpy as np
from datetime import datetime

class VideoHandler:
    def __init__(self, mirror_image=True, output_path="./recordings"):
        self.current_alert = None
        self.alert_start_time = None
        self.alert_duration = 2  # alert seconds

        # timestamp for files
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # video recording
        self.video_writer = None
        self.output_path = output_path
        self.output_video_path = f"{output_path}/recording_{self.timestamp}.mp4"

        # mirror flag
        self.mirror_image = mirror_image

        # font and text parameters
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1

        # overlay buffer
        self.info_overlay = None
        self.info_update_interval = 60  # update every 60 frames
        self.frame_count = 0

        # text dimensions
        self.text_height = None

        # alert cache
        self.alert_text_cache = {}

        # date and time
        self.current_date_str = datetime.now().strftime("%Y-%m-%d")
        self.last_time_update = time.time()
        self.current_time_str = datetime.now().strftime("%H:%M:%S")

    def initialize_video_writer(self, frame):
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.output_video_path, fourcc, 20.0, (w, h)
        )
        print(f"video recording: {self.output_video_path}")
        return self.output_video_path

    def prepare_frame_for_display(self, frame, violations, h_ratio = 0.5, v_ratio = 0.5):
        """optimized version without flickering"""
        if frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # frame counter
        self.frame_count += 1

        # mirror image
        if self.mirror_image:
            display_frame = cv2.flip(frame, 1)
        else:
            display_frame = frame.copy()

        # frame dimensions
        height, width = display_frame.shape[:2]

        # update timestamp every second
        current_time = time.time()
        if current_time - self.last_time_update >= 1.0:
            self.current_time_str = datetime.now().strftime("%H:%M:%S")
            self.last_time_update = current_time

        # update static overlay at intervals
        if self.info_overlay is None or self.frame_count % self.info_update_interval == 0:
            self.info_overlay = np.zeros((height, width, 3), dtype=np.uint8)

            # date text
            (text_width, text_height), _ = cv2.getTextSize(self.current_date_str, self.font, self.font_scale, self.font_thickness)
            self.text_height = text_height

            # date in bottom right corner
            x = width - text_width - 10
            y = height - 30
            cv2.putText(self.info_overlay, self.current_date_str, (x, y), self.font, self.font_scale, (255, 255, 255), self.font_thickness)

        # IMPORTANT: add H and V values in EVERY frame - avoid flickering
        text = f"H: {h_ratio:.2f}, V: {v_ratio:.2f}"
        cv2.putText(display_frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # check violations
        if violations:
            self.current_alert = violations
            self.alert_start_time = time.time()

        # display alert
        current_time = time.time()
        if self.current_alert and current_time - self.alert_start_time < self.alert_duration:
            # use cache
            alert_text = ", ".join(self.current_alert)

            if alert_text not in self.alert_text_cache:
                # create alert overlay
                alert_overlay = np.zeros((40, width, 3), dtype=np.uint8)
                alert_overlay[:, :] = (0, 0, 255)  # red background

                # text
                full_text = "ALERT: " + alert_text
                cv2.putText(alert_overlay, full_text, (10, 30),
                        self.font, self.font_scale, (255, 255, 255), self.font_thickness)

                # save to cache
                self.alert_text_cache[alert_text] = alert_overlay.copy()

                # limit cache
                if len(self.alert_text_cache) > 10:
                    old_key = list(self.alert_text_cache.keys())[0]
                    del self.alert_text_cache[old_key]

            # use from cache
            cached_overlay = self.alert_text_cache[alert_text]

            # apply overlay
            alpha = 0.7  # transparency
            h_alert = cached_overlay.shape[0]
            display_frame[0:h_alert, 0:width] = cv2.addWeighted(
                cached_overlay, alpha,
                display_frame[0:h_alert, 0:width], 1-alpha, 0
            )

        # add time
        cv2.putText(display_frame, self.current_time_str, (width - 80, height - 10),
                self.font, self.font_scale, (255, 255, 255), self.font_thickness)

        # apply static overlay
        if self.info_overlay is not None:
            # fast composition method
            gray = cv2.cvtColor(self.info_overlay, cv2.COLOR_BGR2GRAY)
            mask = (gray > 0).astype(np.uint8)

            # vectorized operations
            for c in range(3):
                display_frame[:, :, c] = (display_frame[:, :, c] * (1 - mask) +
                                        self.info_overlay[:, :, c] * mask)

        return display_frame

    def write_frame(self, frame):
        if self.video_writer:
            self.video_writer.write(frame)

    def release(self):
        if self.video_writer:
            self.video_writer.release()

    def get_output_path(self):
        return self.output_video_path