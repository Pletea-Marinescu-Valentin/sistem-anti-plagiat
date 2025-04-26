import cv2
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, config):
        print("initializare detector obiecte yolov8...")

        self.confidence_threshold = config.get("detection", {}).get("object", {}).get("confidence_threshold", 0.5)

        # obiecte de interes din config
        self.objects_of_interest = config.get("detection", {}).get("object", {}).get("objects_of_interest", [])

        # mapare clase
        self.class_mapping = {
            "cell phone": "telefon",
            "clock": "smartwatch"
        }

        # indecsi pentru telefon si ceas in COCO
        self.classes_to_detect = [67, 74]  # telefon si ceas

        # incarcare model
        try:
            # model yolo
            self.model = YOLO("yolov8s.pt")


            self.device = "cpu"
            print("folosim cpu pentru detectie")

        except Exception as e:
            print(f"eroare incarcare model yolov8: {e}")

        # procesare frame din 30 in 30
        self.frame_count = 0
        self.process_every_n_frames = 30

        # cache detectii
        self.last_detections = []
        self.last_annotated_frame = None

        print("detector obiecte initializat.")

    def detect_objects(self, frame):
        """detecteaza obiecte in frame"""
        self.frame_count += 1

        # procesam doar un frame din n
        if self.frame_count % self.process_every_n_frames != 0 and self.last_annotated_frame is not None:
            current_frame = frame.copy()
            return self.last_detections, current_frame

        # reset counter
        if self.frame_count > 1000:
            self.frame_count = 0

        # resize frame pentru detectie rapida
        resized_frame = cv2.resize(frame, (416, 416))
        return self._detect_with_yolo(resized_frame, frame)

    def _detect_with_yolo(self, resized_frame, original_frame):
        """detectie cu yolo"""
        try:
            # detectie pe frame redimensionat
            results = self.model(resized_frame,
                                conf=self.confidence_threshold,
                                classes=self.classes_to_detect,
                                verbose=False)

            # rezultate
            detected_objects = []
            annotated_frame = original_frame.copy()

            # procesare rezultate
            result = results[0]

            for box in result.boxes:
                class_id = int(box.cls)
                class_name = result.names[class_id]
                confidence = float(box.conf)

                # verificare clasa
                if class_name in self.class_mapping:
                    mapped_class = self.class_mapping[class_name]

                    # adauga obiect detectat
                    detected_objects.append((mapped_class, confidence))

                    # conversie coordonate la frame original
                    h_orig, w_orig = original_frame.shape[:2]
                    h_resized, w_resized = resized_frame.shape[:2]

                    # coordonate
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy

                    # scalare la original
                    x1 = int(x1 * w_orig / w_resized)
                    y1 = int(y1 * h_orig / h_resized)
                    x2 = int(x2 * w_orig / w_resized)
                    y2 = int(y2 * h_orig / h_resized)

                    # desenare bbox
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # text
                    text = f"{mapped_class}: {confidence:.2f}"
                    cv2.putText(annotated_frame, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # cache
            self.last_detections = detected_objects
            self.last_annotated_frame = annotated_frame

            return detected_objects, annotated_frame

        except Exception as e:
            print(f"eroare detectie yolov8: {e}")
            return [], original_frame.copy()