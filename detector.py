"""
Wildlife Detection Engine
Uses YOLOv8 to detect animals in images and video frames.
Dual-model ensemble:
1. Base YOLOv8 model for standard farm animals (Dog, Cow, Sheep, Horse, etc.)
2. Fine-tuned Roboflow YOLOv8 model for native Monkey detection with Person disambiguation.
"""
import os
import cv2
import numpy as np
from datetime import datetime, timedelta
from ultralytics import YOLO
from config import (
    BASE_DIR, YOLO_MODEL, CONFIDENCE_THRESHOLD, WILDLIFE_CLASSES,
    THREAT_LEVELS, THREAT_COLORS, ALERT_COOLDOWN_SECONDS, DEFAULT_ZONE
)


class WildlifeDetector:
    """Core detection engine for wildlife intrusion detection."""

    def __init__(self):
        print("[INFO] Loading base YOLOv8 model (COCO farm wildlife)...")
        self.model = YOLO(YOLO_MODEL)
        
        # Load fine-tuned Roboflow monkey model (trained on 5,355 images)
        self.monkey_model = None
        roboflow_weights = os.path.join(BASE_DIR, 'yolov8_monkey_roboflow.pt')
        if os.path.exists(roboflow_weights):
            try:
                print(f"[INFO] Loading fine-tuned Roboflow model: {roboflow_weights}")
                self.monkey_model = YOLO(roboflow_weights)
                print("[INFO] Roboflow Monkey detector loaded successfully!")
            except Exception as e:
                print(f"[WARN] Could not load monkey model: {e}")

        self.detection_log = []
        self.alert_history = {}  # Track last alert time per animal type
        self.total_detections = 0
        self.detection_zone = DEFAULT_ZONE.copy()
        self.stats = {
            'total_detections': 0,
            'alerts_triggered': 0,
            'animals_detected': {},
            'threat_counts': {'low': 0, 'medium': 0, 'critical': 0},
            'session_start': datetime.now().isoformat(),
        }
        print("[INFO] Dual-model detection pipeline ready!")

    def set_detection_zone(self, zone):
        """Update the detection zone boundaries."""
        self.detection_zone = zone

    def is_in_zone(self, bbox, frame_width, frame_height):
        """Check if detected object is within the defined zone."""
        x1, y1, x2, y2 = bbox
        zone = self.detection_zone
        zone_x1 = zone['x1'] * frame_width
        zone_y1 = zone['y1'] * frame_height
        zone_x2 = zone['x2'] * frame_width
        zone_y2 = zone['y2'] * frame_height

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return zone_x1 <= cx <= zone_x2 and zone_y1 <= cy <= zone_y2

    def should_trigger_alert(self, animal_name):
        """Check if alert should be triggered (respects cooldown)."""
        now = datetime.now()
        if animal_name in self.alert_history:
            last_alert = self.alert_history[animal_name]
            if (now - last_alert).total_seconds() < ALERT_COOLDOWN_SECONDS:
                return False
        self.alert_history[animal_name] = now
        return True

    def detect(self, frame):
        """
        Run wildlife detection on a single frame.
        Runs base YOLO and custom Roboflow model, cleanly merging detections.
        """
        raw_candidates = []
        suppression_boxes = []

        # 1. Base YOLO Inference for farm animals & humans
        base_results = self.model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        for result in base_results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Check for humans (COCO class 0: person)
                if cls_id == 0:
                    suppression_boxes.append([x1, y1, x2, y2])
                    continue

                if cls_id in WILDLIFE_CLASSES:
                    animal_name = WILDLIFE_CLASSES[cls_id]
                    # Confirmed livestock boxes shield from false monkey overlays
                    if animal_name in ('Cow', 'Sheep', 'Dog', 'Horse', 'Elephant', 'Zebra', 'Giraffe'):
                        suppression_boxes.append([x1, y1, x2, y2])

                    raw_candidates.append({
                        'animal': animal_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2],
                        'is_custom': False
                    })

        def compute_overlap(boxA, boxB):
            ax1, ay1, ax2, ay2 = boxA
            bx1, by1, bx2, by2 = boxB
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            inter_area = (ix2 - ix1) * (iy2 - iy1)
            areaA = (ax2 - ax1) * (ay2 - ay1)
            areaB = (bx2 - bx1) * (by2 - by1)
            min_area = min(areaA, areaB)
            return inter_area / min_area if min_area > 0 else 0.0

        # 2. Roboflow Trained Model Inference (Monkey detection with Person filter)
        if self.monkey_model is not None:
            m_results = self.monkey_model(frame, conf=0.45, verbose=False)
            for result in m_results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.monkey_model.names.get(cls_id, '')

                    # The Roboflow model specifically distinguishes 'monkey' from 'Person'
                    if cls_name.lower() == 'monkey':
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        m_box = [x1, y1, x2, y2]

                        # Ensure box doesn't overlap with confirmed human/livestock
                        conflict = any(compute_overlap(m_box, s_box) > 0.35 for s_box in suppression_boxes)
                        if not conflict:
                            raw_candidates.append({
                                'animal': 'Monkey',
                                'confidence': conf,
                                'bbox': m_box,
                                'is_custom': True
                            })

        # 3. Suppress false Bear/Cat when a verified Monkey is present
        filtered_candidates = []
        for cand in raw_candidates:
            if cand['animal'] in ('Bear', 'Cat', 'Bird'):
                overlap = False
                for m_cand in raw_candidates:
                    if m_cand['animal'] == 'Monkey':
                        if compute_overlap(cand['bbox'], m_cand['bbox']) > 0.40:
                            overlap = True
                            break
                if not overlap:
                    filtered_candidates.append(cand)
            else:
                filtered_candidates.append(cand)

        detections = []
        alerts = []
        h, w = frame.shape[:2]
        annotated = frame.copy()

        # Draw detection zone
        zone = self.detection_zone
        zx1, zy1 = int(zone['x1'] * w), int(zone['y1'] * h)
        zx2, zy2 = int(zone['x2'] * w), int(zone['y2'] * h)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), (255, 200, 0), 2)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
        cv2.putText(annotated, f"Zone: {zone.get('name', 'Detection Zone')}",
                    (zx1 + 5, zy1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        for item in filtered_candidates:
            animal_name = item['animal']
            confidence = item['confidence']
            x1, y1, x2, y2 = item['bbox']
            threat = THREAT_LEVELS.get(animal_name, 'medium')

            # Check if in detection zone
            in_zone = self.is_in_zone((x1, y1, x2, y2), w, h)

            detection = {
                'animal': animal_name,
                'confidence': round(confidence * 100, 1),
                'bbox': [x1, y1, x2, y2],
                'threat': threat,
                'in_zone': in_zone,
                'timestamp': datetime.now().isoformat(),
            }
            detections.append(detection)
            self.total_detections += 1

            # Update stats
            self.stats['total_detections'] += 1
            self.stats['animals_detected'][animal_name] = \
                self.stats['animals_detected'].get(animal_name, 0) + 1
            self.stats['threat_counts'][threat] += 1

            # Draw bounding box
            color_bgr = THREAT_COLORS.get(threat, (255, 255, 255))
            thickness = 3 if threat == 'critical' else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, thickness)

            # Draw label background
            label = f"{animal_name} {confidence*100:.0f}% [{threat.upper()}]"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - lh - 10), (x1 + lw + 10, y1), color_bgr, -1)
            cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Generate alert if in zone
            if in_zone and self.should_trigger_alert(animal_name):
                alert = {
                    'type': 'intrusion',
                    'animal': animal_name,
                    'threat': threat,
                    'confidence': round(confidence * 100, 1),
                    'message': f"⚠️ {threat.upper()} ALERT: {animal_name} detected in {zone.get('name', 'zone')}!",
                    'timestamp': datetime.now().isoformat(),
                }
                alerts.append(alert)
                self.stats['alerts_triggered'] += 1

                self.detection_log.append(alert)
                if len(self.detection_log) > 500:
                    self.detection_log = self.detection_log[-500:]

        # Draw status overlay
        status_text = f"Detections: {len(detections)} | Total: {self.total_detections}"
        cv2.putText(annotated, status_text, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    (w - 230, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated, detections, alerts

    def detect_image(self, image_path):
        """Detect wildlife in a single image file."""
        frame = cv2.imread(image_path)
        if frame is None:
            return None, [], []
        return self.detect(frame)

    def get_stats(self):
        """Return current detection statistics."""
        return self.stats.copy()

    def get_log(self, limit=50):
        """Return recent detection log entries."""
        return list(reversed(self.detection_log[-limit:]))

    def reset_stats(self):
        """Reset all statistics."""
        self.stats = {
            'total_detections': 0,
            'alerts_triggered': 0,
            'animals_detected': {},
            'threat_counts': {'low': 0, 'medium': 0, 'critical': 0},
            'session_start': datetime.now().isoformat(),
        }
        self.detection_log = []
        self.total_detections = 0
        self.alert_history = {}
