"""
Wildlife Intrusion Alert System - Main Application
Flask web server with real-time video processing and WebSocket alerts.
"""
import os
import cv2
import json
import base64
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from detector import WildlifeDetector
from config import UPLOAD_FOLDER, HOST, PORT, DEBUG, LOG_FOLDER
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'wildlife-intrusion-secret-2024'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize detector
detector = WildlifeDetector()

# Camera state
camera_state = {
    'active': False,
    'source': None,  # 'webcam', 'video', 'image'
    'cap': None,
    'thread': None,
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'mp4', 'avi', 'mov', 'mkv', 'webm'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_detection_log(alert):
    """Save detection alert to log file."""
    log_file = os.path.join(LOG_FOLDER, f"detections_{datetime.now().strftime('%Y%m%d')}.json")
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(alert)
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save log: {e}")


def process_video_feed(source):
    """Process video frames from webcam or video file."""
    if source == 'webcam':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        socketio.emit('camera_error', {'message': f'Could not open video source: {source}'})
        camera_state['active'] = False
        return

    camera_state['cap'] = cap
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = 1.0 / min(fps, 15)  # Cap at 15 FPS for processing

    while camera_state['active']:
        ret, frame = cap.read()
        if not ret:
            if source != 'webcam':
                # Loop video files
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        # Resize for faster processing
        h, w = frame.shape[:2]
        if w > 960:
            scale = 960 / w
            frame = cv2.resize(frame, (960, int(h * scale)))

        # Run detection
        annotated, detections, alerts = detector.detect(frame)

        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        # Send frame via WebSocket
        socketio.emit('video_frame', {
            'frame': frame_b64,
            'detections': detections,
        })

        # Send alerts
        for alert in alerts:
            save_detection_log(alert)
            socketio.emit('wildlife_alert', alert)

        time.sleep(frame_delay)

    cap.release()
    camera_state['cap'] = None
    camera_state['active'] = False
    socketio.emit('camera_stopped', {})


# ============== Routes ==============

@app.route('/')
def dashboard():
    """Serve the main dashboard."""
    return render_template('dashboard.html')


@app.route('/api/stats')
def get_stats():
    """Get current detection statistics."""
    return jsonify(detector.get_stats())


@app.route('/api/log')
def get_log():
    """Get recent detection log."""
    limit = request.args.get('limit', 50, type=int)
    return jsonify(detector.get_log(limit))


@app.route('/api/reset', methods=['POST'])
def reset_stats():
    """Reset all statistics and logs."""
    detector.reset_stats()
    return jsonify({'status': 'ok', 'message': 'Statistics reset successfully'})


@app.route('/api/detect', methods=['POST'])
def detect_image():
    """Detect wildlife in an uploaded image."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    ext = filename.rsplit('.', 1)[1].lower()

    if ext in {'mp4', 'avi', 'mov', 'mkv', 'webm'}:
        # Start video processing
        start_video(filepath)
        return jsonify({
            'status': 'ok',
            'message': 'Video processing started',
            'type': 'video'
        })
    else:
        # Process single image
        annotated, detections, alerts = detector.detect_image(filepath)
        if annotated is None:
            return jsonify({'error': 'Could not read image'}), 400

        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        result_b64 = base64.b64encode(buffer).decode('utf-8')

        for alert in alerts:
            save_detection_log(alert)
            socketio.emit('wildlife_alert', alert)

        return jsonify({
            'status': 'ok',
            'type': 'image',
            'image': result_b64,
            'detections': detections,
            'alerts': alerts,
            'stats': detector.get_stats(),
        })


@app.route('/api/zone', methods=['POST'])
def update_zone():
    """Update the detection zone boundaries."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No zone data provided'}), 400

    zone = {
        'x1': float(data.get('x1', 0.05)),
        'y1': float(data.get('y1', 0.05)),
        'x2': float(data.get('x2', 0.95)),
        'y2': float(data.get('y2', 0.95)),
        'name': data.get('name', 'Detection Zone'),
    }
    detector.set_detection_zone(zone)
    return jsonify({'status': 'ok', 'zone': zone})


def start_video(source):
    """Start video feed processing in a background thread."""
    stop_camera()
    camera_state['active'] = True
    camera_state['source'] = source
    camera_state['thread'] = threading.Thread(target=process_video_feed, args=(source,), daemon=True)
    camera_state['thread'].start()


def stop_camera():
    """Stop the active camera/video feed."""
    camera_state['active'] = False
    if camera_state['cap'] is not None:
        camera_state['cap'].release()
        camera_state['cap'] = None
    if camera_state['thread'] is not None:
        camera_state['thread'].join(timeout=3)
        camera_state['thread'] = None


# ============== WebSocket Events ==============

@socketio.on('connect')
def handle_connect():
    print('[INFO] Client connected')
    emit('connected', {'status': 'ok', 'stats': detector.get_stats()})


@socketio.on('disconnect')
def handle_disconnect():
    print('[INFO] Client disconnected')


@socketio.on('start_webcam')
def handle_start_webcam():
    """Start processing webcam feed."""
    start_video('webcam')
    emit('camera_started', {'source': 'webcam'})


@socketio.on('stop_camera')
def handle_stop_camera():
    """Stop the active camera feed."""
    stop_camera()
    emit('camera_stopped', {})


@socketio.on('request_stats')
def handle_request_stats():
    """Send current statistics to the client."""
    emit('stats_update', detector.get_stats())


# ============== Main ==============

if __name__ == '__main__':
    print("=" * 60)
    print("  🐾 Wildlife Intrusion Alert System")
    print("  🌐 Dashboard: http://localhost:5000")
    print("=" * 60)
    socketio.run(app, host=HOST, port=PORT, debug=DEBUG, allow_unsafe_werkzeug=True)
