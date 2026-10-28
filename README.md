# 🌾 WildGuard AI — Agricultural Wildlife Intrusion Alert & Census System

> **Autonomous Edge Computer Vision Framework for Agricultural Perimeter Protection, Primate Detection, and Farm Intrusion Census using Dual-Model YOLOv8 Ensemble**

[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/AARUSH-WALI/YOLO-base-FarmAnimalRiskDetection.git)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-3.0-green?style=for-the-badge&logo=flask)
![SocketIO](https://img.shields.io/badge/Socket.IO-Real--time-orange?style=for-the-badge)
![Roboflow](https://img.shields.io/badge/Dataset-Roboflow%205.3K-red?style=for-the-badge)

---

## 📋 The Real-World Agricultural Crisis (Problem Statement)

Farmers, landlords, and rural estate owners face catastrophic crop loss and financial devastation caused by **wildlife intrusion** during peak harvest periods:

- 🐒 **Primate Crop-Raiding:** Troops of monkeys and baboons invade orchards, fruit plantations, wheat, and maize fields, destroying crops in minutes.
- 🐗 **Wild Boars & Stray Livestock:** Pigs root up seedling roots; stray cows, horses, and sheep trample and consume maturing fields.
- 🐘 **Megafauna Hazards:** Elephants and bears demolish fence perimeters, destroy irrigation pipelines, and risk fatal encounters with farm laborers.
- 🌙 **The "Dumb CCTV" Problem:** Traditional surveillance cameras only record passively to DVRs. By the time a farmer watches the recording next morning, entire fields are destroyed.
- ⚡ **Hazards of Current Methods:** High-voltage electric fencing risks human and animal electrocution, while manual night patrolling is dangerous, exhausting, and unreliable.

**WildGuard AI** converts standard farm cameras into an **autonomous, proactive perimeter defense and census system** that detects intruders, alerts the farmer instantly, and tracks distinct farm entries.

---

## 💡 Dual-Model Architecture & The COCO Dataset Gap

### What was missing in standard YOLO (MS-COCO)?
Standard YOLOv8 models pre-trained on Microsoft COCO only recognize 10 animal classes (`bird`, `cat`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe`).
1. **No Primate Support:** COCO contains **zero** classes for `Monkey` or `Baboon`—the #1 agricultural pest in Asia and Africa.
2. **Back-Facing Morphological Confusion:** When monkeys sit hunched with their backs turned, standard COCO misclassified their silhouette as `Bear [CRITICAL]`, causing false emergency panic.
3. **No Human/Livestock Separation:** Standard models generated false positives when animals stood near farm machinery or workers.

### How WildGuard AI Solved This:
We implemented a **Hierarchical Dual-Model Ensemble Pipeline**:
- **Model A (Base YOLOv8n):** Handles livestock, megafauna, and identifies human workers (`COCO Class 0: Person`).
- **Model B (Custom Primate Detector `yolov8_monkey_roboflow.pt`):** Fine-tuned on **5,355 annotated Roboflow agricultural images** including negative background foliage and human counter-examples ($\text{mAP}_{50} = 75.1\%$).
- **Arbitration Engine:** Overlap suppression shields confirmed livestock and humans from false primate overlays, while dynamic troop-context logic relabels back-turned monkeys correctly.

---

## 🚀 Key Innovations (What We Built That Was Missing Earlier)

| Innovation | Earlier Limitation | WildGuard AI Solution |
|---|---|---|
| **1. Dual-Model Ensemble with Counter-Class Suppression** | Single models either missed monkeys or mislabeled humans/fans as monkeys. | Merges base COCO with fine-tuned Roboflow primate model; livestock & human boxes actively shield false primate detections. |
| **2. Dynamic Troop-Context Relabeling** | Monkeys seen from the back were misclassified as dangerous `Bear [CRITICAL]`. | Heuristic engine: If monkeys are active in the scene, co-occurring solitary bear classifications are relabeled to `Monkey [MEDIUM]`. |
| **3. Temporal Persistence Engine ($\ge 2.0$s)** | 1-frame transient noise glitches triggered false sirens. | Requires continuous spatial presence for $\ge 2.0$ seconds before sounding alarms or recording incidents. |
| **4. Agricultural Intrusion Census** | Raw video frames produced meaningless numbers (e.g. 80,000 detections in 30 minutes). | Real-time **"In Farm Right Now"** status + distinct intrusion census for **Today vs. 7 Days (Week)** with species breakdown and human worker tracking. |
| **5. Virtual Polygon Geo-Fencing** | Animals grazing peacefully outside the fence line triggered sirens. | Normalized centroid-in-boundary calculus ($X_{min} \le C_x \le X_{max}$); only true boundary breaches trigger alarms. |
| **6. Zero-Dependency Audio Synthesis** | MP3 audio files failed on network lag or missing file paths. | Web Audio API oscillator synthesis directly in browser memory (sine, triangle, sawtooth sirens) with zero network dependency. |

---

## 📊 Farmer-Centric Intrusion Census Dashboard

Rather than overwhelming the user with raw computer vision telemetry, the WildGuard AI dashboard provides actionable farm intelligence:

1. **In Farm Right Now (Top Hero Beacon):**
   - Live Active Animals badge (e.g., `2 Animals` with live threat pills: `🐄 Cow 2`, `🐒 Monkey 1`).
   - Live Human Presence badge (e.g., `0 Humans` or `👤 1 Human [Inside Boundary]`).
2. **Farm Intrusion Census (Tabbed: Today | 7 Days):**
   - 👤 **Humans Entered Today / This Week** (Tracks laborers and potential trespassers).
   - 🐾 **Animals Entered Today / This Week** (Distinct invasion events).
   - **Detailed Species Entry Breakdown:** Dynamic percentage share pills and distinct counts (e.g., `🐑 Sheep: 14`, `🐄 Cow: 2`, `🐒 Monkey: 2`).
3. **Multi-Tier Threat Matrix:**
   - 🟢 **Low Threat (520Hz Chime):** Bird, Cat, Sheep.
   - 🟡 **Medium Threat (660Hz/550Hz Buzzer):** Cow, Monkey, Stray Dog, Horse.
   - 🔴 **Critical Threat (880Hz/1100Hz Urgent Siren + Red Screen Strobe):** Elephant, Bear, Wild Boar.

---

## 🛠️ Technology Stack & Justification

- **Ultralytics YOLOv8 Nano:** Ultra-lightweight 6.2 MB weight footprint, anchor-free decoupled head, runs at 30+ FPS on edge CPUs.
- **OpenCV (`cv2`):** Ingestion of RTSP/CCTV/webcam streams, aspect-ratio preserved downscaling to 960px width, and spatial geometry.
- **Flask & Flask-SocketIO:** Asynchronous bi-directional WebSocket streaming with sub-20ms frame delivery.
- **Web Audio API:** Hardware-level audio oscillator frequency synthesis in browser memory.
- **Pure Vanilla CSS:** Dark-mode glassmorphic interface tailored for night surveillance control rooms with zero frontend framework bloat.

---

## ⚡ Quick Start & Installation

```bash
# 1. Clone the repository
git clone https://github.com/AARUSH-WALI/YOLO-base-FarmAnimalRiskDetection.git
cd YOLO-base-FarmAnimalRiskDetection

# 2. Create and activate a Python virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the application
python app.py
```

Open your browser and navigate to: **`http://localhost:5000`**

---

## 📁 Project Architecture

```
YOLO/
├── app.py                            # Flask server, WebSocket event dispatcher & video loop
├── detector.py                       # Dual-model YOLOv8 ensemble, geofencing & census engine
├── config.py                         # Threat taxonomies, boundary defaults & cooldown settings
├── requirements.txt                  # Python dependencies (ultralytics, opencv, flask-socketio)
├── README.md                         # Project documentation
├── yolov8n.pt                        # Base MS-COCO YOLOv8 nano model
├── yolov8_monkey_roboflow.pt         # Fine-tuned 5,355-image Roboflow primate model
├── templates/
│   └── dashboard.html                # Dark glassmorphism live surveillance control room
├── static/
│   ├── css/style.css                 # Responsive stylesheet with pulse beacons & threat pills
│   └── js/dashboard.js               # Socket.IO client, Web Audio API synthesis & census logic
├── uploads/                          # Temporary uploaded test footage
└── logs/                             # Time-stamped JSON intrusion incident logs
```

---

---

*Built with ❤️ for agricultural perimeter security, ethical wildlife mitigation, and protecting farmer livelihoods.*
