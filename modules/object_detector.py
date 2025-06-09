import cv2
import torch
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, config):
        print("initializing yolov8 object detector...")

        self.confidence_thresholds = config["detection"]["object"]["confidence_thresholds"]
        self.objects_of_interest = config.get("detection", {}).get("object", {}).get("objects_of_interest", [])

        self.class_mapping = {
            "cell phone": "phone",
            "clock": "smartwatch"
        }

        try:
            # check if GPU is available
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
            print(f"using {self.device.upper()} for detection")

            self.phone_model = YOLO("yolov8s.pt")
            self.phone_model.to(self.device)

            self.watch_model = YOLO("yolov8s.pt")
            self.watch_model.to(self.device)

        except Exception as e:
            print(f"error loading yolov8 models: {e}")
            self.device = "cpu"

        self.frame_count = 0
        self.process_every_n_frames = 20
        self.last_detections = []
        self.last_annotated_frame = None

        print("object detector initialized.")


    def detect_objects(self, frame):
        self.frame_count += 1

        if self.frame_count % self.process_every_n_frames != 0 and self.last_annotated_frame is not None:
            current_frame = frame.copy()
            return self.last_detections, current_frame

        if self.frame_count > 1000:
            self.frame_count = 0

        # higher resolution for good accuracy
        resized_frame = cv2.resize(frame, (1280, 1280))
        return self._detect_with_yolo(resized_frame, frame)

    def _detect_with_yolo(self, resized_frame, original_frame):
        try:
            annotated_frame = original_frame.copy()
            detected_objects = []

            # phone detection
            phones = self._detect_with_model(self.phone_model, resized_frame, original_frame, class_ids=[67], label="phone")
            detected_objects.extend(phones)

            # smartwatch detection  
            watches = self._detect_with_model(self.watch_model, resized_frame, original_frame, class_ids=[74], label="smartwatch")
            detected_objects.extend(watches)

            self.last_detections = detected_objects
            self.last_annotated_frame = annotated_frame

            return detected_objects, annotated_frame

        except Exception as e:
            print(f"yolov8 detection error: {e}")
            return [], original_frame.copy()

    def _detect_with_model(self, model, resized_frame, original_frame, class_ids, label):
        confidence_threshold = self.confidence_thresholds.get(label, 0.5)
        results = model(resized_frame, conf=confidence_threshold, classes=class_ids, verbose=False)
        detections = []

        result = results[0]
        h_orig, w_orig = original_frame.shape[:2]
        h_resized, w_resized = resized_frame.shape[:2]

        for box in result.boxes:
            confidence = float(box.conf)

            if confidence < confidence_threshold:
                continue

            # original coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1 = int(x1 * w_orig / w_resized)
            y1 = int(y1 * h_orig / h_resized)
            x2 = int(x2 * w_orig / w_resized)
            y2 = int(y2 * h_orig / h_resized)

            width = x2 - x1
            height = y2 - y1

            current_label = label
            if label == "phone" and width < 100 and height < 100:
                current_label = "smartwatch"

            # filtering for phone
            if current_label == "phone":
                aspect_ratio = height / (width + 1e-5)
                min_width = 30  # in pixels
                min_height = 50
                min_aspect = 1.2  # phone is taller than wide

                if width < min_width or height < min_height or aspect_ratio < min_aspect:
                    continue  # ignore small or too wide objects

            detections.append((current_label, confidence))

            # draw bbox
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{current_label}: {confidence:.2f}"
            cv2.putText(original_frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return detections
