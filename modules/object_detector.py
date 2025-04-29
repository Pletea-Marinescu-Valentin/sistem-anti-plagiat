import cv2
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, config):
        print("initializare detector obiecte yolov8...")

        self.confidence_threshold = config.get("detection", {}).get("object", {}).get("confidence_threshold", 0.5)
        self.objects_of_interest = config.get("detection", {}).get("object", {}).get("objects_of_interest", [])

        self.class_mapping = {
            "cell phone": "telefon",
            "clock": "smartwatch"
        }

        try:
            # model optimizat: YOLOv8s
            self.phone_model = YOLO("yolov8s.pt")     # pentru telefon
            self.watch_model = YOLO("yolov8s.pt")     # pentru smartwatch
            self.device = "cpu"
            print("folosim CPU pentru detectie")
        except Exception as e:
            print(f"eroare incarcare modele yolov8: {e}")

        self.frame_count = 0
        self.process_every_n_frames = 30
        self.last_detections = []
        self.last_annotated_frame = None

        print("detector obiecte initializat.")

    def detect_objects(self, frame):
        self.frame_count += 1

        if self.frame_count % self.process_every_n_frames != 0 and self.last_annotated_frame is not None:
            current_frame = frame.copy()
            return self.last_detections, current_frame

        if self.frame_count > 1000:
            self.frame_count = 0

        # rezolutie mai mare pentru acuratete buna
        resized_frame = cv2.resize(frame, (640, 640))
        return self._detect_with_yolo(resized_frame, frame)

    def _detect_with_yolo(self, resized_frame, original_frame):
        try:
            annotated_frame = original_frame.copy()
            detected_objects = []

            # detectie telefon
            phones = self._detect_with_model(self.phone_model, resized_frame, original_frame, class_ids=[67], label="telefon", filter_small_objects=True)
            detected_objects.extend(phones)

            # detectie smartwatch
            watches = self._detect_with_model(self.watch_model, resized_frame, original_frame, class_ids=[74], label="smartwatch", filter_small_objects=False)
            detected_objects.extend(watches)

            self.last_detections = detected_objects
            self.last_annotated_frame = annotated_frame

            return detected_objects, annotated_frame

        except Exception as e:
            print(f"eroare detectie yolov8: {e}")
            return [], original_frame.copy()

    def _detect_with_model(self, model, resized_frame, original_frame, class_ids, label, filter_small_objects=False):
        results = model(resized_frame, conf=self.confidence_threshold, classes=class_ids, verbose=False)
        detections = []

        result = results[0]
        h_orig, w_orig = original_frame.shape[:2]
        h_resized, w_resized = resized_frame.shape[:2]

        for box in result.boxes:
            confidence = float(box.conf)

            if confidence < self.confidence_threshold:
                continue

            # coordonate originale
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1 = int(x1 * w_orig / w_resized)
            y1 = int(y1 * h_orig / h_resized)
            x2 = int(x2 * w_orig / w_resized)
            y2 = int(y2 * h_orig / h_resized)

            width = x2 - x1
            height = y2 - y1

            # filtrare obiecte prea inguste (ex: stick negru) - doar pentru telefon
            if filter_small_objects:
                aspect_ratio = height / (width + 1e-5)
                min_width = 40  # in pixeli
                min_height = 60
                min_aspect = 1.4  # telefonul e mai inalt decat lat

                if width < min_width or height < min_height or aspect_ratio < min_aspect:
                    continue  # ignoram obiectele mici sau prea late

            detections.append((label, confidence))

            # desenare bbox
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(original_frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return detections
