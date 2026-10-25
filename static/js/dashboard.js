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

    // Update real-time farm presence (Animals and Humans currently in farm)
    updateLiveFarmPresence(data.detections || []);
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

// ============== UI State & Census Control ==============
let currentCensusTab = 'today';
let cachedStats = null;

function switchCensusTab(period) {
    currentCensusTab = period;
    const tabToday = document.getElementById('tabToday');
    const tabWeek = document.getElementById('tabWeek');
    const humanLbl = document.getElementById('censusHumansLabel');
    const animalLbl = document.getElementById('censusAnimalsLabel');
    const speciesTitle = document.getElementById('speciesCensusTitle');

    if (period === 'today') {
        if (tabToday) tabToday.classList.add('active');
        if (tabWeek) tabWeek.classList.remove('active');
        if (humanLbl) humanLbl.textContent = 'Humans Entered (Today)';
        if (animalLbl) animalLbl.textContent = 'Animals Entered (Today)';
        if (speciesTitle) speciesTitle.textContent = 'Species Entries Today:';
    } else {
        if (tabToday) tabToday.classList.remove('active');
        if (tabWeek) tabWeek.classList.add('active');
        if (humanLbl) humanLbl.textContent = 'Humans Entered (7 Days)';
        if (animalLbl) animalLbl.textContent = 'Animals Entered (7 Days)';
        if (speciesTitle) speciesTitle.textContent = 'Species Entries (Past 7 Days):';
    }

    if (cachedStats) {
        renderCensus(cachedStats);
    }
}

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
    cachedStats = stats;
    animateValue('totalDetections', stats.total_detections || 0);
    animateValue('totalAlerts', stats.alerts_triggered || 0);
    animateValue('criticalCount', stats.threat_counts?.critical || 0);
    animateValue('speciesCount', Object.keys(stats.animals_detected || stats.today_species || {}).length);

    // Update threat bars
    const total = (stats.threat_counts?.low || 0) +
                  (stats.threat_counts?.medium || 0) +
                  (stats.threat_counts?.critical || 0);

    if (total > 0) {
        setThreatBar('threatLow', 'threatLowCount', stats.threat_counts?.low || 0, total);
        setThreatBar('threatMedium', 'threatMediumCount', stats.threat_counts?.medium || 0, total);
        setThreatBar('threatCritical', 'threatCriticalCount', stats.threat_counts?.critical || 0, total);
    }

    // Render Farm Intrusion Census (Today vs 7 Days)
    renderCensus(stats);
}

// ============== Real-Time Animals & Humans in Farm Right Now ==============
function updateLiveFarmPresence(detections) {
    const animalContainer = document.getElementById('liveAnimalChips');
    const humanContainer = document.getElementById('liveHumanChips');
    const animalBadge = document.getElementById('liveTotalAnimals');
    const humanBadge = document.getElementById('liveTotalHumans');

    // Filter only detections within farm zone
    const insideZone = (detections || []).filter(d => d.in_zone !== false);

    // Distinguish humans and animals
    const humans = insideZone.filter(d => d.is_human === true || d.animal === 'Person');
    const animals = insideZone.filter(d => d.is_human !== true && d.animal !== 'Person');

    // 1. Update Animals Badge & Chips
    if (animalBadge) {
        animalBadge.textContent = `${animals.length} Animal${animals.length === 1 ? '' : 's'}`;
        animalBadge.className = 'live-badge' + (animals.length > 0 ? ' live-badge-active' : '');
    }

    if (animalContainer) {
        if (animals.length === 0) {
            animalContainer.innerHTML = '<div class="empty-live-state">🌾 No animals in boundary</div>';
        } else {
            const counts = {};
            const threats = {};
            animals.forEach(d => {
                counts[d.animal] = (counts[d.animal] || 0) + 1;
                threats[d.animal] = d.threat || 'medium';
            });
            animalContainer.innerHTML = Object.entries(counts).map(([animal, count]) => {
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
    }

    // 2. Update Humans Badge & Chips
    if (humanBadge) {
        humanBadge.textContent = `${humans.length} Human${humans.length === 1 ? '' : 's'}`;
        humanBadge.className = 'live-badge live-badge-human' + (humans.length > 0 ? ' live-badge-human-active' : '');
    }

    if (humanContainer) {
        if (humans.length === 0) {
            humanContainer.innerHTML = '<div class="empty-live-state">👤 No humans in boundary</div>';
        } else {
            humanContainer.innerHTML = `
                <div class="live-chip live-chip-human">
                    <span class="live-chip-icon">👤</span>
                    <span class="live-chip-name">Human / Person</span>
                    <span class="live-chip-count">${humans.length}</span>
                </div>
            `;
        }
    }
}

// ============== Farm Intrusion Census Rendering ==============
function renderCensus(stats) {
    const isToday = (currentCensusTab === 'today');
    const humanCount = isToday ? (stats.today_humans || 0) : (stats.week_humans || 0);
    const animalCount = isToday ? (stats.today_animals || 0) : (stats.week_animals || 0);
    const speciesMap = isToday ? (stats.today_species || {}) : (stats.week_species || {});

    animateValue('censusHumans', humanCount);
    animateValue('censusAnimals', animalCount);

    updateSpeciesCensus(speciesMap, animalCount);
}

function updateSpeciesCensus(speciesMap, totalAnimals) {
    const container = document.getElementById('speciesList');
    if (!container) return;

    const entries = Object.entries(speciesMap).filter(([_, count]) => count > 0);
    if (entries.length === 0) {
        container.innerHTML = '<div class="empty-state-small">No animal entries recorded yet</div>';
        return;
    }

    // Sort by count descending
    entries.sort((a, b) => b[1] - a[1]);

    container.innerHTML = entries.map(([name, count]) => {
        const emoji = animalEmojis[name] || '🐾';
        const pct = totalAnimals > 0 ? Math.round((count / totalAnimals) * 100) : 0;
        return `
            <div class="species-item">
                <div class="species-name-wrap">
                    <span class="species-emoji">${emoji}</span>
                    <span class="species-name">${name}</span>
                </div>
                <div class="species-meta">
                    <span class="species-pct-pill">${pct}%</span>
                    <span class="species-count">${count}</span>
                </div>
            </div>
        `;
    }).join('');
}

function animateValue(elementId, newValue) {
    const el = document.getElementById(elementId);
    if (!el) return;
    const current = parseInt(el.textContent) || 0;
    if (current !== newValue) {
        el.textContent = newValue;
        el.classList.add('updated');
        setTimeout(() => el.classList.remove('updated'), 500);
    }
}

function setThreatBar(barId, countId, value, total) {
    const pct = total > 0 ? (value / total * 100) : 0;
    const bar = document.getElementById(barId);
    const count = document.getElementById(countId);
    if (bar) bar.style.width = pct + '%';
    if (count) count.textContent = value;
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
