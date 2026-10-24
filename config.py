"""
Configuration settings for Wildlife Intrusion Alert System
"""
import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
LOG_FOLDER = os.path.join(BASE_DIR, 'logs')
SAMPLE_FOLDER = os.path.join(BASE_DIR, 'static', 'samples')

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, LOG_FOLDER, SAMPLE_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# --- YOLO Model Settings ---
YOLO_MODEL = 'yolov8n.pt'  # Using YOLOv8 Nano for fast inference
CONFIDENCE_THRESHOLD = 0.45

# --- Wildlife Classes (COCO dataset class IDs and names) ---
# These are the animal classes available in COCO-pretrained YOLOv8
WILDLIFE_CLASSES = {
    14: 'Bird',
    15: 'Cat',
    16: 'Dog',
    17: 'Horse',
    18: 'Sheep',
    19: 'Cow',
    20: 'Elephant',
    21: 'Bear',
    22: 'Zebra',
    23: 'Giraffe',
    # Extended Wildlife Classes (Specialized Transfer / Heuristics)
    99: 'Monkey',
}

# Threat levels for each animal type
THREAT_LEVELS = {
    'Bird': 'low',
    'Cat': 'low',
    'Dog': 'medium',
    'Horse': 'medium',
    'Sheep': 'low',
    'Cow': 'medium',
    'Monkey': 'medium',  # Crop raiding, orchard damage, troop behavior
    'Elephant': 'critical',
    'Bear': 'critical',
    'Zebra': 'medium',
    'Giraffe': 'low',
}

# Colors for bounding boxes (BGR format for OpenCV)
THREAT_COLORS = {
    'low': (0, 200, 0),       # Green
    'medium': (0, 165, 255),   # Orange
    'critical': (0, 0, 255),   # Red
}

# --- Alert Settings ---
ALERT_COOLDOWN_SECONDS = 10  # Minimum seconds between repeated alerts for same animal
ENABLE_SOUND_ALERTS = True

# --- Server Settings ---
HOST = '0.0.0.0'
PORT = 5000
DEBUG = True

# --- Detection Zone ---
# Default zone coordinates (percentage of frame: x1, y1, x2, y2)
DEFAULT_ZONE = {
    'x1': 0.05,
    'y1': 0.05,
    'x2': 0.95,
    'y2': 0.95,
    'name': 'Farm Boundary'
}
