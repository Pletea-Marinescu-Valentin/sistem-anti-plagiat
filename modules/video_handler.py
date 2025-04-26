import cv2
import time
import numpy as np
from datetime import datetime

class VideoHandler:
    def __init__(self, mirror_image=True, output_path="./recordings"):
        self.current_alert = None
        self.alert_start_time = None
        self.alert_duration = 2  # secunde alerta

        # timestamp pentru fisiere
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # inregistrare video
        self.video_writer = None
        self.output_path = output_path
        self.output_video_path = f"{output_path}/inregistrare_{self.timestamp}.mp4"

        # flag pentru oglindire
        self.mirror_image = mirror_image

        # font si parametri text
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1

        # buffer overlay
        self.info_overlay = None
        self.info_update_interval = 60  # update la 60 frame-uri
        self.frame_count = 0

        # dimensiuni text
        self.text_height = None

        # cache alerte
        self.alert_text_cache = {}

        # data si timp
        self.current_date_str = datetime.now().strftime("%Y-%m-%d")
        self.last_time_update = time.time()
        self.current_time_str = datetime.now().strftime("%H:%M:%S")

    def initialize_video_writer(self, frame):
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.output_video_path, fourcc, 20.0, (w, h)
        )
        print(f"inregistrare video: {self.output_video_path}")
        return self.output_video_path

    def prepare_frame_for_display(self, frame, violations, h_ratio = 0.5, v_ratio = 0.5):
        """versiune optimizata fara palpaire"""
        if frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # contor frame
        self.frame_count += 1

        # oglindire imagine
        if self.mirror_image:
            display_frame = cv2.flip(frame, 1)
        else:
            display_frame = frame.copy()

        # dimensiuni frame
        height, width = display_frame.shape[:2]

        # update timestamp la fiecare secunda
        current_time = time.time()
        if current_time - self.last_time_update >= 1.0:
            self.current_time_str = datetime.now().strftime("%H:%M:%S")
            self.last_time_update = current_time

        # update overlay static la interval
        if self.info_overlay is None or self.frame_count % self.info_update_interval == 0:
            self.info_overlay = np.zeros((height, width, 3), dtype=np.uint8)

            # text data
            (text_width, text_height), _ = cv2.getTextSize(self.current_date_str, self.font, self.font_scale, self.font_thickness)
            self.text_height = text_height

            # data in colt dreapta jos
            x = width - text_width - 10
            y = height - 30
            cv2.putText(self.info_overlay, self.current_date_str, (x, y), self.font, self.font_scale, (255, 255, 255), self.font_thickness)

        # IMPORTANT: adauga valori H si V in FIECARE frame - evitam palpairea
        text = f"H: {h_ratio:.2f}, V: {v_ratio:.2f}"
        cv2.putText(display_frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # verificare incalcari
        if violations:
            self.current_alert = violations
            self.alert_start_time = time.time()

        # afisare alerta
        current_time = time.time()
        if self.current_alert and current_time - self.alert_start_time < self.alert_duration:
            # folosire cache
            alert_text = ", ".join(self.current_alert)

            if alert_text not in self.alert_text_cache:
                # creare overlay alerta
                alert_overlay = np.zeros((40, width, 3), dtype=np.uint8)
                alert_overlay[:, :] = (0, 0, 255)  # fundal rosu

                # text
                full_text = "ALERTA: " + alert_text
                cv2.putText(alert_overlay, full_text, (10, 30),
                        self.font, self.font_scale, (255, 255, 255), self.font_thickness)

                # salvare in cache
                self.alert_text_cache[alert_text] = alert_overlay.copy()

                # limitare cache
                if len(self.alert_text_cache) > 10:
                    old_key = list(self.alert_text_cache.keys())[0]
                    del self.alert_text_cache[old_key]

            # folosire din cache
            cached_overlay = self.alert_text_cache[alert_text]

            # aplicare overlay
            alpha = 0.7  # transparenta
            h_alert = cached_overlay.shape[0]
            display_frame[0:h_alert, 0:width] = cv2.addWeighted(
                cached_overlay, alpha,
                display_frame[0:h_alert, 0:width], 1-alpha, 0
            )

        # adauga timp
        cv2.putText(display_frame, self.current_time_str, (width - 80, height - 10),
                self.font, self.font_scale, (255, 255, 255), self.font_thickness)

        # aplicare overlay static
        if self.info_overlay is not None:
            # metoda rapida compunere
            gray = cv2.cvtColor(self.info_overlay, cv2.COLOR_BGR2GRAY)
            mask = (gray > 0).astype(np.uint8)

            # operatii vectorizate
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