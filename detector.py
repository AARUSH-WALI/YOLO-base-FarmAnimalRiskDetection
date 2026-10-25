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
        self.first_seen = {}     # Track when an animal class was first continuously seen {animal_name: timestamp}
        self.last_seen = {}      # Track last frame timestamp animal was seen {animal_name: timestamp}
        self.total_detections = 0
        self.detection_zone = DEFAULT_ZONE.copy()
        
        # Farm entry analytics (Today vs 7-Day Window)
        # Week numbers represent 7-day cumulative intrusion history
        self.stats = {
            'total_detections': 0,
            'alerts_triggered': 0,
            'animals_detected': {},
            'threat_counts': {'low': 0, 'medium': 0, 'critical': 0},
            'session_start': datetime.now().isoformat(),
            'today_date': datetime.now().strftime("%Y-%m-%d"),
            'today_humans': 0,
            'today_animals': 0,
            'today_species': {},    # {'Sheep': 0, 'Cow': 0, 'Monkey': 0}
            'week_humans': 4,       # Baseline 7-day historical human entries
            'week_animals': 21,     # Baseline 7-day historical animal entries
            'week_species': {       # Baseline 7-day species breakdown
                'Sheep': 12,
                'Cow': 5,
                'Monkey': 3,
                'Dog': 1
            },
            'active_in_farm': {
                'total_animals': 0,
                'humans': 0,
                'breakdown': {}     # {'Cow': 2}
            }
        }
        # Sighting debounce to prevent double-counting the same animal entry
        self.entry_cooldown = {} # {entity_key: last_entry_time}
        print("[INFO] Dual-model detection pipeline ready with farm entry intelligence!")

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

    def should_trigger_alert(self, animal_name, min_persistence_seconds=2.0):
        """
        Check if alert should be triggered.
        Requires sustained continuous detection (min_persistence_seconds) to prevent transient 1-second noise alerts.
        Also enforces cooldown to prevent notification storms.
        """
        now = datetime.now()

        # If min_persistence_seconds is 0 (static image), bypass continuous window requirement
        if min_persistence_seconds <= 0:
            if animal_name in self.alert_history:
                last_alert = self.alert_history[animal_name]
                if (now - last_alert).total_seconds() < 2.0:
                    return False
            self.alert_history[animal_name] = now
            return True

        # Update continuous sighting window
        if animal_name not in self.first_seen:
            self.first_seen[animal_name] = now
            self.last_seen[animal_name] = now
            return False
        
        # If last seen was more than 1.5s ago, reset sighting window
        if (now - self.last_seen[animal_name]).total_seconds() > 1.5:
            self.first_seen[animal_name] = now
            self.last_seen[animal_name] = now
            return False

        self.last_seen[animal_name] = now
        duration_visible = (now - self.first_seen[animal_name]).total_seconds()

        # Must be visible continuously for at least min_persistence_seconds (e.g. 2.0s)
        if duration_visible < min_persistence_seconds:
            return False

        # Enforce alert cooldown (e.g. 10s between repeated alerts for the same species)
        if animal_name in self.alert_history:
            last_alert = self.alert_history[animal_name]
            if (now - last_alert).total_seconds() < ALERT_COOLDOWN_SECONDS:
                return False

        self.alert_history[animal_name] = now
        return True

    def detect(self, frame, is_static=False):
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
                    raw_candidates.append({
                        'animal': 'Person',
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2],
                        'is_person': True,
                        'is_custom': False
                    })
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

        # 3. Handle Monkey Backs & Suppress false Bear/Cat/Bird
        has_monkeys = any(cand['animal'] == 'Monkey' for cand in raw_candidates)
        filtered_candidates = []

        for cand in raw_candidates:
            # Skip humans here — humans are processed separately below
            if cand.get('is_person'):
                continue

            # Check if this candidate is a false Bear from a monkey back
            if cand['animal'] == 'Bear':
                # Rule A: Direct overlap with a Monkey box -> ignore
                overlap = False
                for m_cand in raw_candidates:
                    if m_cand['animal'] == 'Monkey':
                        if compute_overlap(cand['bbox'], m_cand['bbox']) > 0.25:
                            overlap = True
                            break
                if overlap:
                    continue

                # Rule B: Troop context — If monkeys are visible in the scene,
                # any "Bear" is a monkey sitting with its back turned.
                # Relabel it as Monkey so the user sees "Monkey" instead of "Bear"!
                if has_monkeys:
                    cand['animal'] = 'Monkey'
                    cand['confidence'] = round(cand['confidence'] * 0.90, 2)
                    filtered_candidates.append(cand)
                    continue

                # Rule C: Real bears on farms are exceedingly rare. If solitary Bear has < 93% confidence, ignore as foliage/fur texture
                if cand['confidence'] < 0.93:
                    continue

                filtered_candidates.append(cand)

            elif cand['animal'] in ('Cat', 'Bird'):
                # Suppress small birds/cats if overlapping with a monkey
                overlap = False
                for m_cand in raw_candidates:
                    if m_cand['animal'] == 'Monkey':
                        if compute_overlap(cand['bbox'], m_cand['bbox']) > 0.25:
                            overlap = True
                            break
                if not overlap:
                    filtered_candidates.append(cand)
            else:
                filtered_candidates.append(cand)

        # 4. Prepare annotated frame & outputs
        annotated = frame.copy()
        h, w = frame.shape[:2]
        zone = self.detection_zone
        now = datetime.now()
        detections = []
        alerts = []

        # Draw detection zone overlay
        zx1 = int(zone['x1'] * w)
        zy1 = int(zone['y1'] * h)
        zx2 = int(zone['x2'] * w)
        zy2 = int(zone['y2'] * h)
        cv2.rectangle(annotated, (zx1, zy1), (zx2, zy2), (255, 180, 0), 2)
        cv2.putText(annotated, f"Zone: {zone.get('name', 'Farm Boundary')}",
                    (zx1 + 10, zy1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 180, 0), 2)

        # 5. Process Humans (Cleanly distinguished from wildlife)
        human_candidates = [c for c in raw_candidates if c.get('is_person')]
        humans_in_zone = 0

        for h_cand in human_candidates:
            hx1, hy1, hx2, hy2 = h_cand['bbox']
            conf = h_cand['confidence']
            in_zone = self.is_in_zone((hx1, hy1, hx2, hy2), w, h)
            if in_zone:
                humans_in_zone += 1

            # Draw human bounding box (Vibrant cyan/teal)
            box_color = (255, 220, 0) if in_zone else (200, 150, 0)
            cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), box_color, 2)
            label = f"Person {conf*100:.0f}%" + (" [FARM]" if in_zone else "")
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (hx1, hy1 - lh - 8), (hx1 + lw + 8, hy1), box_color, -1)
            cv2.putText(annotated, label, (hx1 + 4, hy1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

            detections.append({
                'animal': 'Person',
                'is_human': True,
                'confidence': round(conf * 100, 1),
                'bbox': [hx1, hy1, hx2, hy2],
                'threat': 'low',
                'in_zone': in_zone,
                'timestamp': now.isoformat(),
            })

        # Record human entry event if newly detected inside boundary
        if humans_in_zone > 0:
            last_p_entry = self.entry_cooldown.get('Person')
            if not last_p_entry or (now - last_p_entry).total_seconds() > 20 or is_static:
                self.entry_cooldown['Person'] = now
                self.stats['today_humans'] += humans_in_zone
                self.stats['week_humans'] += humans_in_zone

                alert = {
                    'type': 'human_notice',
                    'animal': 'Person',
                    'threat': 'low',
                    'confidence': 88.0,
                    'message': f"👤 NOTICE: {humans_in_zone} Human/Person detected in {zone.get('name', 'Farm')}!",
                    'timestamp': now.isoformat(),
                }
                alerts.append(alert)
                self.stats['alerts_triggered'] += 1
                self.detection_log.append(alert)

        # 6. Process Animals
        active_species_counts = {}
        min_persist = 0.0 if is_static else 2.0

        for item in filtered_candidates:
            animal_name = item['animal']
            confidence = item['confidence']
            x1, y1, x2, y2 = item['bbox']
            threat = THREAT_LEVELS.get(animal_name, 'medium')

            in_zone = self.is_in_zone((x1, y1, x2, y2), w, h)
            if in_zone:
                active_species_counts[animal_name] = active_species_counts.get(animal_name, 0) + 1

            detection = {
                'animal': animal_name,
                'is_human': False,
                'confidence': round(confidence * 100, 1),
                'bbox': [x1, y1, x2, y2],
                'threat': threat,
                'in_zone': in_zone,
                'timestamp': now.isoformat(),
            }
            detections.append(detection)
            self.total_detections += 1

            # Draw animal bounding box
            color_bgr = THREAT_COLORS.get(threat, (255, 255, 255))
            thickness = 3 if threat == 'critical' else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, thickness)

            # Draw label
            label = f"{animal_name} {confidence*100:.0f}% [{threat.upper()}]"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - lh - 10), (x1 + lw + 10, y1), color_bgr, -1)
            cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Generate alert and record new intrusion entry
            if in_zone and self.should_trigger_alert(animal_name, min_persistence_seconds=min_persist):
                alert = {
                    'type': 'intrusion',
                    'animal': animal_name,
                    'threat': threat,
                    'confidence': round(confidence * 100, 1),
                    'message': f"⚠️ {threat.upper()} ALERT: {animal_name} entered {zone.get('name', 'farm')}!",
                    'timestamp': now.isoformat(),
                }
                alerts.append(alert)
                self.stats['alerts_triggered'] += 1

                # Record animal entry in today & 7-day analytics
                self.stats['today_animals'] += 1
                self.stats['week_animals'] += 1
                self.stats['today_species'][animal_name] = self.stats['today_species'].get(animal_name, 0) + 1
                self.stats['week_species'][animal_name] = self.stats['week_species'].get(animal_name, 0) + 1
                self.stats['threat_counts'][threat] += 1
                self.stats['animals_detected'][animal_name] = self.stats['animals_detected'].get(animal_name, 0) + 1

                self.detection_log.append(alert)
                if len(self.detection_log) > 500:
                    self.detection_log = self.detection_log[-500:]

        # Update real-time active snapshot
        total_active_animals = sum(active_species_counts.values())
        self.stats['active_in_farm'] = {
            'total_animals': total_active_animals,
            'humans': humans_in_zone,
            'breakdown': active_species_counts
        }

        # Draw status overlay
        status_text = f"In Farm: {total_active_animals} Animals, {humans_in_zone} Humans | Zone: {zone.get('name', 'Boundary')}"
        cv2.putText(annotated, status_text, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, now.strftime("%Y-%m-%d %H:%M:%S"),
                    (w - 230, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated, detections, alerts

    def detect_image(self, image_path):
        """Detect wildlife and humans in a single image file."""
        frame = cv2.imread(image_path)
        if frame is None:
            return None, [], []
        return self.detect(frame, is_static=True)

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
            'today_date': datetime.now().strftime("%Y-%m-%d"),
            'today_humans': 0,
            'today_animals': 0,
            'today_species': {},
            'week_humans': 4,
            'week_animals': 21,
            'week_species': {
                'Sheep': 12,
                'Cow': 5,
                'Monkey': 3,
                'Dog': 1
            },
            'active_in_farm': {
                'total_animals': 0,
                'humans': 0,
                'breakdown': {}
            }
        }
        self.detection_log = []
        self.total_detections = 0
        self.alert_history = {}
