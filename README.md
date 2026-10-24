# 🐾 WildGuard AI — Wildlife Intrusion Alert System

> **AI-powered real-time wildlife detection for farm and residential boundary protection using YOLOv8**

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-3.0-green?style=for-the-badge&logo=flask)
![WebSocket](https://img.shields.io/badge/WebSocket-Real--time-orange?style=for-the-badge)

---

## 📋 Problem Statement

Farmers and rural residents face significant losses due to **wildlife intrusion** into their farmland and residential areas. Animals like wild boars, stray dogs, monkeys, elephants, and bears can:

- 🌾 **Destroy crops** worth thousands of dollars
- 🏠 **Damage property** and infrastructure
- ⚠️ **Pose safety risks** to humans and livestock
- 🕐 **Go undetected** during night hours when traditional monitoring fails

Traditional solutions like fences and manual watchers are **expensive**, **unreliable**, and **not scalable**.

---

## 💡 Our Solution

**WildGuard AI** uses **YOLOv8 (You Only Look Once)** — a state-of-the-art deep learning object detection model — to provide:

| Feature | Description |
|---------|-------------|
| 🎥 **Real-time Detection** | Processes live camera feeds at 15+ FPS |
| 🐻 **Multi-species Recognition** | Detects 10+ animal species with high accuracy |
| 🔴 **Threat Classification** | Categorizes threats as Low / Medium / Critical |
| 🚨 **Instant Alerts** | Audio + visual alerts within milliseconds |
| 📍 **Zone-based Monitoring** | Define custom protection zones on camera feed |
| 📊 **Analytics Dashboard** | Track detection history, species breakdown, and trends |
| 📝 **Automatic Logging** | All detections are logged with timestamps for review |

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────┐
│                  Camera Input                    │
│         (Webcam / IP Camera / Video File)        │
└───────────────────┬────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────┐
│              YOLOv8 Detection Engine             │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Frame    │→ │ Object   │→ │ Wildlife     │  │
│  │ Capture  │  │ Detection│  │ Classification│  │
│  └──────────┘  └──────────┘  └──────────────┘  │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Zone     │→ │ Threat   │→ │ Alert        │  │
│  │ Check    │  │ Analysis │  │ Generation   │  │
│  └──────────┘  └──────────┘  └──────────────┘  │
└───────────────────┬────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────┐
│              Flask Web Server                    │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ REST API │  │ WebSocket│  │ Static Files │  │
│  │ Endpoints│  │ Server   │  │ Serving      │  │
│  └──────────┘  └──────────┘  └──────────────┘  │
└───────────────────┬────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────┐
│            Web Dashboard (Browser)               │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Live     │  │ Alert    │  │ Statistics   │  │
│  │ Video    │  │ System   │  │ & Logging    │  │
│  └──────────┘  └──────────┘  └──────────────┘  │
└────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** installed
- **Webcam** (optional, for live detection)
- **pip** package manager

### Installation

```bash
# 1. Navigate to the project directory
cd YOLO

# 2. Create a virtual environment (recommended)
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the application
python app.py
```

### Access the Dashboard

Open your browser and go to: **http://localhost:5000**

---

## 📖 How to Use

### 1. Image Detection
- Click **"Upload Image / Video"** on the dashboard
- Select an image containing animals
- The system will process the image and display results with bounding boxes

### 2. Live Webcam Detection
- Click **"Start Webcam"** to begin real-time detection
- Animals will be detected and highlighted in the video feed
- Alerts will appear as toast notifications with sound

### 3. Video File Processing
- Upload a `.mp4`, `.avi`, or `.mov` video file
- The system will process each frame and stream results in real-time

### 4. Zone Configuration
- Adjust the detection zone boundaries in the right panel
- Only animals within the defined zone will trigger alerts

---

## 🐾 Detectable Animals

| Animal | Threat Level | Example Scenario |
|--------|-------------|-----------------|
| 🐦 Bird | 🟢 Low | Crop pecking |
| 🐱 Cat | 🟢 Low | Minor poultry threat |
| 🐕 Dog (Stray) | 🟡 Medium | Livestock harassment |
| 🐴 Horse | 🟡 Medium | Crop trampling |
| 🐑 Sheep | 🟢 Low | Boundary breach |
| 🐄 Cow | 🟡 Medium | Crop damage |
| 🐘 Elephant | 🔴 Critical | Major crop/property damage |
| 🐻 Bear | 🔴 Critical | Human safety threat |
| 🦓 Zebra | 🟡 Medium | Fence damage |
| 🦒 Giraffe | 🟢 Low | Tree damage |

---

## 📁 Project Structure

```
YOLO/
├── app.py              # Flask web server with WebSocket support
├── detector.py         # YOLOv8 detection engine
├── config.py           # Configuration settings
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
├── templates/
│   └── dashboard.html  # Web dashboard template
├── static/
│   ├── css/
│   │   └── style.css   # Dashboard styling
│   └── js/
│       └── dashboard.js # Frontend JavaScript
├── uploads/            # Uploaded files (auto-created)
└── logs/               # Detection logs (auto-created)
```

---

## 🔧 Technical Details

### YOLOv8 Model
- **Model**: YOLOv8n (Nano) — optimized for speed
- **Dataset**: Pre-trained on COCO (80 classes, 10 animal classes used)
- **Inference Speed**: ~15-30 FPS on CPU, 60+ FPS on GPU
- **Accuracy**: mAP50 of 37.3% on COCO (nano variant)

### Key Technologies
| Technology | Purpose |
|-----------|---------|
| **YOLOv8** | Real-time object detection |
| **OpenCV** | Image/video processing |
| **Flask** | Web server framework |
| **Socket.IO** | Real-time WebSocket communication |
| **Web Audio API** | Browser-based alert sounds |

---

## 🔮 Future Enhancements

1. **Custom Model Training** — Train on region-specific wildlife datasets
2. **Night Vision Support** — Infrared camera integration
3. **SMS/Email Alerts** — Remote notification via Twilio/SMTP
4. **Multi-Camera Support** — Monitor multiple zones simultaneously
5. **Cloud Dashboard** — Remote monitoring from anywhere
6. **GPS Integration** — Map-based animal movement tracking
7. **Edge Deployment** — Run on Raspberry Pi / NVIDIA Jetson

---

## 📄 License

This project is developed for academic/educational purposes as part of a college seminar project.

---

## 👥 Team

- **Project**: Wildlife Intrusion Alert System
- **Domain**: Computer Vision & Deep Learning
- **Framework**: YOLOv8 by Ultralytics

---

*Built with ❤️ for protecting farms and communities using AI*
