/**
 * WildGuard AI — Dashboard JavaScript
 * Handles WebSocket communication, real-time UI updates,
 * file uploads, alerts, and audio notifications.
 */

// ============== Socket Connection ==============
const socket = io();
let isConnected = false;
let isCameraActive = false;

// ============== Audio Alert System ==============
class AlertSound {
    constructor() {
        this.ctx = null;
        this.initialized = false;
    }

    init() {
        if (this.initialized) return;
        try {
            this.ctx = new (window.AudioContext || window.webkitAudioContext)();
            this.initialized = true;
        } catch (e) {
            console.warn('Web Audio API not available');
        }
    }

    play(threat) {
        if (!this.initialized) this.init();
        if (!this.ctx) return;

        const now = this.ctx.currentTime;
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.connect(gain);
        gain.connect(this.ctx.destination);

        if (threat === 'critical') {
            // Urgent siren: alternating high tones
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(880, now);
            osc.frequency.setValueAtTime(1100, now + 0.15);
            osc.frequency.setValueAtTime(880, now + 0.3);
            osc.frequency.setValueAtTime(1100, now + 0.45);
            gain.gain.setValueAtTime(0.15, now);
            gain.gain.exponentialRampToValueAtTime(0.01, now + 0.6);
            osc.start(now);
            osc.stop(now + 0.6);
        } else if (threat === 'medium') {
            // Warning beep: two tones
            osc.type = 'triangle';
            osc.frequency.setValueAtTime(660, now);
            osc.frequency.setValueAtTime(550, now + 0.15);
            gain.gain.setValueAtTime(0.12, now);
            gain.gain.exponentialRampToValueAtTime(0.01, now + 0.35);
            osc.start(now);
            osc.stop(now + 0.35);
        } else {
            // Info ping: soft single tone
            osc.type = 'sine';
            osc.frequency.setValueAtTime(520, now);
            gain.gain.setValueAtTime(0.08, now);
            gain.gain.exponentialRampToValueAtTime(0.01, now + 0.2);
            osc.start(now);
            osc.stop(now + 0.2);
        }
    }
}

const alertSound = new AlertSound();

// Initialize audio on first user interaction
document.addEventListener('click', () => alertSound.init(), { once: true });

// ============== Animal Emojis ==============
const animalEmojis = {
    'Bird': '🐦', 'Cat': '🐱', 'Dog': '🐕', 'Horse': '🐴',
    'Sheep': '🐑', 'Cow': '🐄', 'Elephant': '🐘', 'Bear': '🐻',
    'Zebra': '🦓', 'Giraffe': '🦒', 'Monkey': '🐒',
};

// ============== Time Display ==============
function updateTime() {
    const now = new Date();
    document.getElementById('navTime').textContent = now.toLocaleTimeString('en-US', {
        hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
}
setInterval(updateTime, 1000);
updateTime();

// ============== Socket Events ==============
socket.on('connect', () => {
    isConnected = true;
    updateConnectionStatus(true);
    console.log('[WildGuard] Connected to server');
});

socket.on('disconnect', () => {
    isConnected = false;
    updateConnectionStatus(false);
    console.log('[WildGuard] Disconnected from server');
});

socket.on('connected', (data) => {
    if (data.stats) updateStats(data.stats);
});

socket.on('video_frame', (data) => {
    const frame = document.getElementById('videoFrame');
    const placeholder = document.getElementById('videoPlaceholder');
    const overlay = document.getElementById('videoOverlay');

    frame.src = 'data:image/jpeg;base64,' + data.frame;
    frame.classList.remove('hidden');
    placeholder.classList.add('hidden');
    overlay.classList.remove('hidden');

    // Update live detection count
    const count = data.detections ? data.detections.length : 0;
    const counter = document.getElementById('liveDetectionCount');
    if (counter) {
        counter.textContent = `${count} in boundary`;
        counter.className = 'detection-counter' + (count > 0 ? ' has-detections' : '');
    }

    // Update real-time farm presence (e.g. "2 Monkeys, 1 Cow right now")
    updateLiveAnimalsInFarm(data.detections || []);
});

socket.on('wildlife_alert', (alert) => {
    showToast(alert);
    addAlertLog(alert);
    alertSound.play(alert.threat);

    // Screen flash for critical alerts
    if (alert.threat === 'critical') {
        const flash = document.createElement('div');
        flash.className = 'screen-flash';
        document.body.appendChild(flash);
        setTimeout(() => flash.remove(), 700);
    }

    // Refresh stats
    socket.emit('request_stats');
});

socket.on('stats_update', (stats) => {
    updateStats(stats);
});

socket.on('camera_started', (data) => {
    isCameraActive = true;
    updateCameraStatus(true);
    document.getElementById('btnWebcam').classList.add('hidden');
    document.getElementById('btnStop').classList.remove('hidden');
});

socket.on('camera_stopped', () => {
    isCameraActive = false;
    updateCameraStatus(false);
    document.getElementById('btnWebcam').classList.remove('hidden');
    document.getElementById('btnStop').classList.add('hidden');
});

socket.on('camera_error', (data) => {
    showToast({
        animal: 'System',
        threat: 'medium',
        message: data.message || 'Camera error occurred',
        timestamp: new Date().toISOString(),
    });
});

// ============== UI Update Functions ==============
function updateConnectionStatus(connected) {
    const el = document.getElementById('connectionStatus');
    const dot = el.querySelector('.status-dot');
    const text = el.querySelector('span:last-child');
    dot.className = 'status-dot ' + (connected ? 'online' : 'offline');
    text.textContent = connected ? 'Connected' : 'Disconnected';
}

function updateCameraStatus(active) {
    const el = document.getElementById('cameraStatus');
    const dot = el.querySelector('.status-dot');
    const text = el.querySelector('span:last-child');
    dot.className = 'status-dot ' + (active ? 'active' : 'offline');
    text.textContent = active ? 'Camera Active' : 'Camera Off';
}

function updateStats(stats) {
    animateValue('totalDetections', stats.total_detections || 0);
    animateValue('totalAlerts', stats.alerts_triggered || 0);
    animateValue('criticalCount', stats.threat_counts?.critical || 0);
    animateValue('speciesCount', Object.keys(stats.animals_detected || {}).length);

    // Update threat bars
    const total = (stats.threat_counts?.low || 0) +
                  (stats.threat_counts?.medium || 0) +
                  (stats.threat_counts?.critical || 0);

    if (total > 0) {
        setThreatBar('threatLow', 'threatLowCount', stats.threat_counts?.low || 0, total);
        setThreatBar('threatMedium', 'threatMediumCount', stats.threat_counts?.medium || 0, total);
        setThreatBar('threatCritical', 'threatCriticalCount', stats.threat_counts?.critical || 0, total);
    }

    // Update species list
    updateSpeciesList(stats.animals_detected || {});
}

// ============== Real-Time Animals in Farm Right Now ==============
function updateLiveAnimalsInFarm(detections) {
    const container = document.getElementById('liveAnimalChips');
    const badge = document.getElementById('liveTotalAnimals');
    if (!container || !badge) return;

    // Filter animals that are inside the farm zone
    const insideZone = detections.filter(d => d.in_zone !== false);
    const totalCount = insideZone.length;

    badge.textContent = `${totalCount} Active`;
    if (totalCount > 0) {
        badge.className = 'live-badge live-badge-active';
    } else {
        badge.className = 'live-badge';
    }

    if (totalCount === 0) {
        container.innerHTML = '<div class="empty-live-state">🌾 No animals currently in boundary</div>';
        return;
    }

    // Count per animal type (e.g. 2 Monkeys, 1 Cow, 14 Sheep)
    const counts = {};
    const threats = {};
    insideZone.forEach(d => {
        counts[d.animal] = (counts[d.animal] || 0) + 1;
        threats[d.animal] = d.threat || 'medium';
    });

    // Render chips
    container.innerHTML = Object.entries(counts).map(([animal, count]) => {
        const threat = threats[animal] || 'medium';
        const emoji = animalEmojis[animal] || '🐾';
        return `
            <div class="live-chip live-chip-${threat}">
                <span class="live-chip-icon">${emoji}</span>
                <span class="live-chip-name">${animal}</span>
                <span class="live-chip-count">${count}</span>
            </div>
        `;
    }).join('');
}

function animateValue(elementId, newValue) {
    const el = document.getElementById(elementId);
    const current = parseInt(el.textContent) || 0;
    if (current !== newValue) {
        el.textContent = newValue;
        el.classList.add('updated');
        setTimeout(() => el.classList.remove('updated'), 500);
    }
}

function setThreatBar(barId, countId, value, total) {
    const pct = total > 0 ? (value / total * 100) : 0;
    document.getElementById(barId).style.width = pct + '%';
    document.getElementById(countId).textContent = value;
}

function updateSpeciesList(animals) {
    const container = document.getElementById('speciesList');
    if (Object.keys(animals).length === 0) {
        container.innerHTML = '<div class="empty-state-small">No species detected yet</div>';
        return;
    }

    // Sort by count descending
    const sorted = Object.entries(animals).sort((a, b) => b[1] - a[1]);
    container.innerHTML = sorted.map(([name, count]) => `
        <div class="species-item">
            <span class="species-name">
                <span>${animalEmojis[name] || '🐾'}</span>
                ${name}
            </span>
            <span class="species-count">${count}</span>
        </div>
    `).join('');
}

// ============== Alert Log ==============
function addAlertLog(alert) {
    const container = document.getElementById('alertLog');

    // Remove empty state
    const emptyState = container.querySelector('.empty-state-small');
    if (emptyState) emptyState.remove();

    const time = new Date(alert.timestamp).toLocaleTimeString('en-US', {
        hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'
    });

    const item = document.createElement('div');
    item.className = `alert-item alert-${alert.threat}`;
    item.innerHTML = `
        <span class="alert-time">${time}</span>
        <span class="alert-message">
            ${animalEmojis[alert.animal] || '🐾'} ${alert.message}
        </span>
    `;

    container.insertBefore(item, container.firstChild);

    // Keep max 100 entries
    while (container.children.length > 100) {
        container.removeChild(container.lastChild);
    }
}

function clearLogs() {
    const container = document.getElementById('alertLog');
    container.innerHTML = '<div class="empty-state-small">No alerts yet — waiting for detections</div>';
}

// ============== Toast Notifications ==============
function showToast(alert) {
    const container = document.getElementById('toastContainer');

    const threatTitles = {
        'low': 'Low Threat Detected',
        'medium': '⚠️ Medium Threat Alert',
        'critical': '🚨 CRITICAL THREAT ALERT',
    };

    const toast = document.createElement('div');
    toast.className = `toast toast-${alert.threat}`;
    toast.innerHTML = `
        <span class="toast-icon">${animalEmojis[alert.animal] || '🐾'}</span>
        <div class="toast-body">
            <div class="toast-title">${threatTitles[alert.threat] || 'Alert'}</div>
            <div class="toast-message">${alert.message}</div>
        </div>
        <button class="toast-close" onclick="this.parentElement.remove()">✕</button>
        <div class="toast-progress"></div>
    `;

    container.appendChild(toast);

    // Auto-dismiss
    const duration = alert.threat === 'critical' ? 8000 : 6000;
    setTimeout(() => {
        toast.classList.add('toast-exit');
        setTimeout(() => toast.remove(), 400);
    }, duration);

    // Max 5 toasts
    while (container.children.length > 5) {
        container.removeChild(container.firstChild);
    }
}

// ============== Camera Controls ==============
function startWebcam() {
    if (!isConnected) {
        showToast({
            animal: 'System',
            threat: 'medium',
            message: 'Not connected to server. Please check if the server is running.',
            timestamp: new Date().toISOString(),
        });
        return;
    }
    socket.emit('start_webcam');
}

function stopCamera() {
    socket.emit('stop_camera');
    const frame = document.getElementById('videoFrame');
    const placeholder = document.getElementById('videoPlaceholder');
    const overlay = document.getElementById('videoOverlay');
    frame.classList.add('hidden');
    placeholder.classList.remove('hidden');
    overlay.classList.add('hidden');
}

// ============== File Upload ==============
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    // Show loading state
    const placeholder = document.getElementById('videoPlaceholder');
    const origContent = placeholder.innerHTML;
    placeholder.innerHTML = `
        <div class="placeholder-content">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" class="placeholder-icon" style="animation: spin 1s linear infinite;">
                <path d="M21 12a9 9 0 11-6.219-8.56"/>
            </svg>
            <h3>Processing...</h3>
            <p>Running YOLO detection on ${file.name}</p>
        </div>
    `;

    try {
        const response = await fetch('/api/detect', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (data.error) {
            placeholder.innerHTML = origContent;
            showToast({
                animal: 'System',
                threat: 'medium',
                message: data.error,
                timestamp: new Date().toISOString(),
            });
            return;
        }

        if (data.type === 'image') {
            // Display result image
            const frame = document.getElementById('videoFrame');
            const overlay = document.getElementById('videoOverlay');
            frame.src = 'data:image/jpeg;base64,' + data.image;
            frame.classList.remove('hidden');
            placeholder.classList.add('hidden');
            overlay.classList.remove('hidden');

            const count = data.detections ? data.detections.length : 0;
            const counter = document.getElementById('liveDetectionCount');
            counter.textContent = `${count} detection${count !== 1 ? 's' : ''}`;
            counter.className = 'detection-counter' + (count > 0 ? ' has-detections' : '');

            // Show detection cards and live farm presence
            updateLiveAnimalsInFarm(data.detections || []);
            if (data.detections && data.detections.length > 0) {
                showDetectionResults(data.detections);
            }

            // Update stats
            if (data.stats) updateStats(data.stats);

            showToast({
                animal: 'System',
                threat: count > 0 ? 'medium' : 'low',
                message: count > 0
                    ? `Detected ${count} animal${count !== 1 ? 's' : ''} in uploaded image!`
                    : 'No wildlife detected in the uploaded image.',
                timestamp: new Date().toISOString(),
            });
        } else if (data.type === 'video') {
            // Video processing started
            placeholder.innerHTML = origContent;
            isCameraActive = true;
            updateCameraStatus(true);
            document.getElementById('btnWebcam').classList.add('hidden');
            document.getElementById('btnStop').classList.remove('hidden');
        }
    } catch (err) {
        console.error('Upload error:', err);
        placeholder.innerHTML = origContent;
        showToast({
            animal: 'System',
            threat: 'critical',
            message: 'Upload failed. Check server connection.',
            timestamp: new Date().toISOString(),
        });
    }

    // Reset file input
    event.target.value = '';
}

function showDetectionResults(detections) {
    const container = document.getElementById('detectionResults');
    const cards = document.getElementById('detectionCards');

    cards.innerHTML = detections.map(d => `
        <div class="detection-card threat-${d.threat}">
            <span class="animal-emoji">${animalEmojis[d.animal] || '🐾'}</span>
            <span class="animal-name">${d.animal}</span>
            <span class="confidence">${d.confidence}%</span>
        </div>
    `).join('');

    container.classList.remove('hidden');
}

// ============== Zone Configuration ==============
async function updateZone() {
    const zone = {
        x1: parseInt(document.getElementById('zoneX1').value) / 100,
        y1: parseInt(document.getElementById('zoneY1').value) / 100,
        x2: parseInt(document.getElementById('zoneX2').value) / 100,
        y2: parseInt(document.getElementById('zoneY2').value) / 100,
        name: document.getElementById('zoneName').value,
    };

    try {
        const response = await fetch('/api/zone', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(zone),
        });

        const data = await response.json();
        showToast({
            animal: 'System',
            threat: 'low',
            message: `Detection zone "${zone.name}" updated successfully.`,
            timestamp: new Date().toISOString(),
        });
    } catch (err) {
        showToast({
            animal: 'System',
            threat: 'medium',
            message: 'Failed to update zone configuration.',
            timestamp: new Date().toISOString(),
        });
    }
}

// ============== Reset Stats ==============
async function resetStats() {
    try {
        await fetch('/api/reset', { method: 'POST' });
        updateStats({
            total_detections: 0,
            alerts_triggered: 0,
            animals_detected: {},
            threat_counts: { low: 0, medium: 0, critical: 0 },
        });
        clearLogs();
        showToast({
            animal: 'System',
            threat: 'low',
            message: 'All statistics have been reset.',
            timestamp: new Date().toISOString(),
        });
    } catch (err) {
        console.error('Reset error:', err);
    }
}

// ============== CSS Animation for spinner ==============
const style = document.createElement('style');
style.textContent = '@keyframes spin { to { transform: rotate(360deg); } }';
document.head.appendChild(style);

// ============== Initial Stats Fetch ==============
fetch('/api/stats')
    .then(res => res.json())
    .then(stats => updateStats(stats))
    .catch(() => {});

console.log('🐾 WildGuard AI Dashboard loaded');
