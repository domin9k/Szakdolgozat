/**
 * DermAI Scan — Main Application Logic
 * ======================================
 * Handles camera, upload, cropping, quality checks, API calls, and results rendering.
 */

// --- State ---
const API_BASE = window.location.origin;
let currentImage = null;      // File or Blob to send
let selectedModel = 'both';
let cameraStream = null;
let facingMode = 'environment'; // rear camera by default

// --- Initialization ---
document.addEventListener('DOMContentLoaded', async () => {
    // Animate splash
    const splashStatus = document.getElementById('splash-status');

    splashStatus.textContent = 'Loading models...';

    // Check server health
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();

        if (data.mobilenet_ready && data.efficientnet_ready) {
            splashStatus.textContent = 'Models ready!';
            document.getElementById('server-status').classList.add('online');
        } else {
            splashStatus.textContent = 'Some models unavailable';
            document.getElementById('server-status').classList.add('online');
        }
    } catch (e) {
        splashStatus.textContent = 'Server unreachable — retrying...';
        document.getElementById('server-status').classList.add('offline');
    }

    // Wait for splash animation
    setTimeout(() => {
        const splash = document.getElementById('splash-screen');
        splash.classList.add('fade-out');
        setTimeout(() => {
            splash.style.display = 'none';
            document.getElementById('app').classList.remove('hidden');
        }, 600);
    }, 2800);

    // Register service worker
    if ('serviceWorker' in navigator) {
        try {
            await navigator.serviceWorker.register('/sw.js');
        } catch (e) {
            console.log('SW registration failed:', e);
        }
    }
});

// --- Screen Navigation ---
function showScreen(screenId) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    const target = document.getElementById(`screen-${screenId}`);
    if (target) target.classList.add('active');

    // Show/hide header on camera screen
    const header = document.querySelector('.app-header');
    if (screenId === 'camera') {
        header.style.display = 'none';
    } else {
        header.style.display = '';
    }
}

function goHome() {
    stopCamera();
    showScreen('home');
}

// --- Camera ---
async function openCamera() {
    showScreen('camera');

    try {
        const constraints = {
            video: {
                facingMode: facingMode,
                width: { ideal: 1280 },
                height: { ideal: 1280 },
            },
            audio: false,
        };

        cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
        const video = document.getElementById('camera-video');
        video.srcObject = cameraStream;
        await video.play();

        // Draw darkened overlay with square cutout
        drawCameraOverlay();
        window.addEventListener('resize', drawCameraOverlay);
    } catch (e) {
        console.error('Camera error:', e);
        alert('Could not access camera. Please check permissions or use the Upload option.');
        goHome();
    }
}

function drawCameraOverlay() {
    const canvas = document.getElementById('camera-overlay');
    const container = canvas.parentElement;
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;

    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    // Semi-transparent dark overlay
    ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
    ctx.fillRect(0, 0, w, h);

    // Calculate square guide position (centered)
    const guideSize = Math.min(w, h) * 0.65;
    const gx = (w - guideSize) / 2;
    const gy = (h - guideSize) / 2;

    // Clear the square area (transparent window)
    ctx.clearRect(gx, gy, guideSize, guideSize);

    // Subtle rounded-corner border around clear area
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.5)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    const r = 12;
    ctx.moveTo(gx + r, gy);
    ctx.lineTo(gx + guideSize - r, gy);
    ctx.arcTo(gx + guideSize, gy, gx + guideSize, gy + r, r);
    ctx.lineTo(gx + guideSize, gy + guideSize - r);
    ctx.arcTo(gx + guideSize, gy + guideSize, gx + guideSize - r, gy + guideSize, r);
    ctx.lineTo(gx + r, gy + guideSize);
    ctx.arcTo(gx, gy + guideSize, gx, gy + guideSize - r, r);
    ctx.lineTo(gx, gy + r);
    ctx.arcTo(gx, gy, gx + r, gy, r);
    ctx.closePath();
    ctx.stroke();
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
    }
    window.removeEventListener('resize', drawCameraOverlay);
}

async function switchCamera() {
    facingMode = facingMode === 'environment' ? 'user' : 'environment';
    stopCamera();
    await openCamera();
}

function capturePhoto() {
    const video = document.getElementById('camera-video');

    // Flash effect
    const flash = document.createElement('div');
    flash.className = 'camera-flash';
    document.body.appendChild(flash);
    setTimeout(() => flash.remove(), 300);

    // Capture from video to canvas
    const canvas = document.createElement('canvas');
    const vw = video.videoWidth;
    const vh = video.videoHeight;

    // Square crop from center of video
    const side = Math.min(vw, vh);
    canvas.width = side;
    canvas.height = side;

    const ctx = canvas.getContext('2d');
    const sx = (vw - side) / 2;
    const sy = (vh - side) / 2;
    ctx.drawImage(video, sx, sy, side, side, 0, 0, side, side);

    // Convert to blob
    canvas.toBlob((blob) => {
        currentImage = blob;
        stopCamera();
        showPreview(blob);
    }, 'image/jpeg', 0.92);
}

// --- File Upload ---
function triggerUpload() {
    document.getElementById('file-input').click();
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    currentImage = file;
    showPreview(file);

    // Reset input so same file can be re-selected
    event.target.value = '';
}

// --- Preview & Quality Check ---
async function showPreview(imageSource) {
    showScreen('preview');

    // Load image and draw square cropped preview
    const img = await createImageBitmap(imageSource);
    const canvas = document.getElementById('preview-canvas');
    const ctx = canvas.getContext('2d');

    // Square crop from center
    const side = Math.min(img.width, img.height);
    const sx = (img.width - side) / 2;
    const sy = (img.height - side) / 2;

    // Display at reasonable size
    const displaySize = Math.min(400, window.innerWidth - 48);
    canvas.width = displaySize;
    canvas.height = displaySize;

    ctx.drawImage(img, sx, sy, side, side, 0, 0, displaySize, displaySize);

    // Run quality check
    const qualityPanel = document.getElementById('quality-panel');
    const qualityBadge = document.getElementById('quality-badge');
    const qualityDetails = document.getElementById('quality-details');

    qualityBadge.textContent = 'Checking...';
    qualityBadge.className = 'quality-badge';
    qualityDetails.innerHTML = '';

    try {
        const formData = new FormData();
        formData.append('file', currentImage);

        const res = await fetch(`${API_BASE}/api/quality-check`, {
            method: 'POST',
            body: formData,
        });

        const data = await res.json();
        const q = data.quality;

        // Update badge
        if (q.quality_score >= 80) {
            qualityBadge.textContent = `Excellent (${q.quality_score}/100)`;
            qualityBadge.className = 'quality-badge good';
        } else if (q.quality_score >= 40) {
            qualityBadge.textContent = `Fair (${q.quality_score}/100)`;
            qualityBadge.className = 'quality-badge fair';
        } else {
            qualityBadge.textContent = `Poor (${q.quality_score}/100)`;
            qualityBadge.className = 'quality-badge poor';
        }

        // Metrics
        let metricsHtml = `
            <div class="quality-metric">
                <span class="label">Sharpness</span>
                <span class="value">${q.sharpness}</span>
            </div>
            <div class="quality-metric">
                <span class="label">Brightness</span>
                <span class="value">${q.brightness}</span>
            </div>
            <div class="quality-metric">
                <span class="label">Contrast</span>
                <span class="value">${q.contrast}</span>
            </div>
            <div class="quality-metric">
                <span class="label">Image Size</span>
                <span class="value">${data.image_info.size}</span>
            </div>
        `;

        // Warnings
        if (q.warnings.length > 0) {
            metricsHtml += q.warnings.map(w =>
                `<div class="quality-warning"><span>${w}</span></div>`
            ).join('');
        }

        qualityDetails.innerHTML = metricsHtml;

        // Disable analyze button if quality too poor
        const analyzeBtn = document.getElementById('btn-analyze');
        if (!q.is_acceptable) {
            analyzeBtn.disabled = true;
            analyzeBtn.textContent = 'Quality too low';
        } else {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                Analyze
            `;
        }
    } catch (e) {
        qualityBadge.textContent = 'Check failed';
        qualityBadge.className = 'quality-badge fair';
        qualityDetails.innerHTML = '<div class="quality-warning"><span>Could not connect to server for quality check.</span></div>';
    }
}

// --- Model Selection ---
function selectModel(model) {
    selectedModel = model;
    document.querySelectorAll('.model-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.model === model);
    });
}

// --- Analyze Image ---
async function analyzeImage() {
    if (!currentImage) return;

    showScreen('analyzing');

    const stepEl = document.getElementById('analyzing-step');
    const progressBar = document.getElementById('progress-bar');

    // Progress animation
    stepEl.textContent = 'Uploading image...';
    progressBar.style.width = '15%';

    const formData = new FormData();
    formData.append('file', currentImage);

    const url = `${API_BASE}/api/analyze?model=${selectedModel}&tta_steps=5`;

    try {
        stepEl.textContent = 'Sending to AI models...';
        progressBar.style.width = '30%';

        const startTime = performance.now();
        const res = await fetch(url, {
            method: 'POST',
            body: formData,
        });
        const networkTime = performance.now() - startTime;

        progressBar.style.width = '80%';
        stepEl.textContent = 'Processing results...';

        if (res.status === 422) {
            // Quality too low
            const data = await res.json();
            progressBar.style.width = '100%';
            setTimeout(() => showQualityError(data), 300);
            return;
        }

        if (!res.ok) {
            throw new Error(`Server error: ${res.status}`);
        }

        const data = await res.json();
        data._networkTime = Math.round(networkTime);

        progressBar.style.width = '100%';
        stepEl.textContent = 'Done!';

        setTimeout(() => showResults(data), 400);

    } catch (e) {
        console.error('Analysis failed:', e);
        progressBar.style.width = '100%';
        stepEl.textContent = 'Error!';
        setTimeout(() => {
            alert(`Analysis failed: ${e.message}\n\nPlease check that the server is running.`);
            goHome();
        }, 300);
    }
}

// --- Show Results ---
function showResults(data) {
    showScreen('results');
    const body = document.getElementById('results-body');

    let html = '';

    // Overall recommendation (when both models used)
    if (data.overall_recommendation) {
        // Determine overall risk from individual predictions
        let overallRisk = 'low';
        if (data.predictions.efficientnet) {
            overallRisk = data.predictions.efficientnet.risk_level;
        } else if (data.predictions.mobilenet) {
            overallRisk = data.predictions.mobilenet.risk_level;
        }

        const labels = { low: 'Low Risk', high: 'High Risk', uncertain: 'Uncertain' };

        html += `
            <div class="result-overall ${overallRisk}">
                <div class="result-overall-label">${labels[overallRisk]}</div>
                <div class="result-overall-desc">${data.overall_recommendation}</div>
            </div>
        `;
    }

    // Individual model results
    for (const [key, pred] of Object.entries(data.predictions)) {
        const risk = pred.risk_level;
        const scorePercent = Math.round(pred.raw_score * 100);

        html += `
            <div class="result-card">
                <div class="result-card-header">
                    <span class="result-model-name">${pred.model}</span>
                    <span class="result-time">${pred.processing_time_ms}ms</span>
                </div>
                <div class="result-card-body">
                    <div class="result-score-bar">
                        <div class="result-score-fill ${risk}" style="width: ${scorePercent}%"></div>
                        <div class="result-score-marker" style="left: calc(${scorePercent}% - 1.5px)"></div>
                    </div>
                    <div class="result-label-row">
                        <span class="result-verdict ${risk}">${pred.label}</span>
                        <span class="result-confidence">${pred.confidence_label} · ${pred.confidence_percent}%</span>
                    </div>
                    <div class="result-raw-score">
                        Raw score: ${pred.raw_score} (0 = benign, 1 = malignant) · Threshold: 0.5
                    </div>
                </div>
            </div>
        `;
    }

    // Timing
    html += `
        <div class="timing-card">
            <span class="label">Total server processing</span>
            <span class="value">${data.total_processing_time_ms}ms</span>
        </div>
        <div class="timing-card">
            <span class="label">Network round-trip</span>
            <span class="value">${data._networkTime}ms</span>
        </div>
    `;

    // Quality warnings in results
    if (data.quality && data.quality.warnings && data.quality.warnings.length > 0) {
        html += `
            <div class="result-quality">
                <div class="result-quality-title">Image Quality Notes (Score: ${data.quality.quality_score}/100)</div>
                <div class="result-quality-warnings">
                    ${data.quality.warnings.map(w => `<div class="rq-warning"><span>${w}</span></div>`).join('')}
                </div>
            </div>
        `;
    }

    // Disclaimer
    html += `
        <div class="disclaimer">
            <p><strong>Disclaimer:</strong> This is an AI-based screening tool and NOT a medical diagnosis. 
            Results should be interpreted by a qualified dermatologist. If in doubt, always seek professional medical advice.</p>
        </div>
    `;

    body.innerHTML = html;

    // Animate score bars after render
    requestAnimationFrame(() => {
        body.querySelectorAll('.result-score-fill').forEach(el => {
            const width = el.style.width;
            el.style.width = '0%';
            requestAnimationFrame(() => {
                el.style.width = width;
            });
        });
    });
}

function showQualityError(data) {
    showScreen('results');
    const body = document.getElementById('results-body');

    let html = `
        <div class="result-overall uncertain">
            <div class="result-overall-label">Image Quality Too Low</div>
            <div class="result-overall-desc">The image does not meet the minimum quality requirements for reliable analysis. Please retake or upload a better image.</div>
        </div>
    `;

    if (data.quality) {
        html += `
            <div class="result-quality">
                <div class="result-quality-title">Quality Issues (Score: ${data.quality.quality_score}/100)</div>
                <div class="result-quality-warnings">
                    ${data.quality.warnings.map(w => `<div class="rq-warning"><span>${w}</span></div>`).join('')}
                </div>
            </div>
        `;
    }

    body.innerHTML = html;
}

// --- Utilities ---
function closeCamera() {
    stopCamera();
    goHome();
}
