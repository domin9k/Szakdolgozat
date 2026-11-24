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

// --- Helper: Universal Image Loader ---
async function loadImageElement(source) {
    if (typeof createImageBitmap === 'function') {
        try {
            return await createImageBitmap(source);
        } catch (e) {
            console.warn('createImageBitmap failed, using image fallback:', e);
        }
    }
    return new Promise((resolve, reject) => {
        const img = new Image();
        const url = URL.createObjectURL(source);
        img.onload = () => {
            URL.revokeObjectURL(url);
            resolve(img);
        };
        img.onerror = () => {
            URL.revokeObjectURL(url);
            const reader = new FileReader();
            reader.onload = (re) => {
                const img2 = new Image();
                img2.onload = () => resolve(img2);
                img2.onerror = reject;
                img2.src = re.target.result;
            };
            reader.onerror = reject;
            reader.readAsDataURL(source);
        };
        img.src = url;
    });
}

// --- Initialization ---
document.addEventListener('DOMContentLoaded', async () => {
    // Setup File Upload & Drag & Drop
    setupFileUpload();
    setupDragAndDrop();

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

    // Clean up service workers and caches
    if ('serviceWorker' in navigator) {
        try {
            const registrations = await navigator.serviceWorker.getRegistrations();
            for (const r of registrations) {
                await r.unregister();
            }
        } catch (e) {
            console.log('SW unregister failed:', e);
        }
    }
    if ('caches' in window) {
        try {
            const keys = await caches.keys();
            await Promise.all(keys.map(k => caches.delete(k)));
        } catch (e) { }
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
    const fileInput = document.getElementById('file-input');
    if (fileInput) fileInput.value = '';
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
    const fileInput = document.getElementById('file-input');
    if (fileInput) {
        fileInput.value = '';
        fileInput.click();
    }
}

function setupFileUpload() {
    const fileInput = document.getElementById('file-input');
    if (!fileInput) return;
    fileInput.addEventListener('change', handleFileSelect);
}

function handleFileSelect(event) {
    const files = event.target ? event.target.files : (event.files || null);
    if (!files || files.length === 0) return;

    const file = files[0];
    const isImage = (file.type && file.type.startsWith('image/')) ||
                    /\.(jpe?g|png|webp|bmp|gif|tiff?|heic|heif)$/i.test(file.name || '');

    if (!isImage) {
        alert('Kérjük, képfájlt válasszon ki (pl. JPG, PNG, WebP)!');
        return;
    }

    currentImage = file;
    showPreview(file);
}

// --- Preview & Quality Check ---
async function showPreview(imageSource) {
    showScreen('preview');

    const canvas = document.getElementById('preview-canvas');
    const ctx = canvas.getContext('2d');

    // Display at reasonable size
    const displaySize = Math.min(400, window.innerWidth - 48);
    canvas.width = displaySize;
    canvas.height = displaySize;

    try {
        // Load image using universal loader (supports bitmap + fallback)
        const img = await loadImageElement(imageSource);

        // Square crop from center
        const width = img.naturalWidth || img.width;
        const height = img.naturalHeight || img.height;
        const side = Math.min(width, height);
        const sx = (width - side) / 2;
        const sy = (height - side) / 2;

        ctx.drawImage(img, sx, sy, side, side, 0, 0, displaySize, displaySize);
    } catch (err) {
        console.error('Failed to render preview image:', err);
    }

    // Run quality check
    const qualityPanel = document.getElementById('quality-panel');
    const qualityBadge = document.getElementById('quality-badge');
    const qualityDetails = document.getElementById('quality-details');
    const analyzeBtn = document.getElementById('btn-analyze');

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
            qualityBadge.textContent = `Warning (${q.quality_score}/100)`;
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

        // Never hard-block analysis: allow "Analyze Anyway" with warning
        analyzeBtn.disabled = false;
        if (!q.is_acceptable) {
            analyzeBtn.textContent = 'Analyze Anyway (Quality Warning)';
            analyzeBtn.onclick = () => analyzeImage(true);
        } else {
            analyzeBtn.textContent = 'Analyze';
            analyzeBtn.onclick = () => analyzeImage(false);
        }
    } catch (e) {
        qualityBadge.textContent = 'Check skipped';
        qualityBadge.className = 'quality-badge fair';
        qualityDetails.innerHTML = '<div class="quality-warning"><span>Offline / Quality check unavailable. You can still proceed with analysis.</span></div>';
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Analyze';
        analyzeBtn.onclick = () => analyzeImage(false);
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
async function analyzeImage(force = false) {
    if (!currentImage) return;

    showScreen('analyzing');

    const stepEl = document.getElementById('analyzing-step');
    const progressBar = document.getElementById('progress-bar');

    // Progress animation
    stepEl.textContent = 'Uploading image...';
    progressBar.style.width = '15%';

    const formData = new FormData();
    formData.append('file', currentImage);

    const url = `${API_BASE}/api/analyze?model=${selectedModel}&tta_steps=5&force=${force}`;

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

    // Calculate aggregated score and consensus
    const preds = Object.values(data.predictions || {});
    let primaryPred = null;
    if (data.predictions.efficientnet) {
        primaryPred = data.predictions.efficientnet;
    } else if (data.predictions.mobilenet) {
        primaryPred = data.predictions.mobilenet;
    }

    // Determine overall risk level
    let overallRisk = 'low';
    let avgRawScore = 0;
    if (preds.length > 0) {
        const sumScores = preds.reduce((acc, p) => acc + (p.raw_score !== undefined ? p.raw_score : 0.5), 0);
        avgRawScore = sumScores / preds.length;
        
        const hasHigh = preds.some(p => p.risk_level === 'high');
        const hasUncertain = preds.some(p => p.risk_level === 'uncertain');
        
        if (hasHigh) {
            overallRisk = 'high';
        } else if (hasUncertain) {
            overallRisk = 'uncertain';
        } else {
            overallRisk = 'low';
        }
    }

    // Calculate gauge position (0% - 100%)
    const gaugePercent = Math.min(Math.max(Math.round(avgRawScore * 100), 2), 98);

    // Human-friendly content configurations
    const verdictConfig = {
        low: {
            title: 'Alacsony kockázatú lelet',
            subtitle: 'Jóindulatú bőrelváltozásra jellemző mintázat (Low Risk)',
            badge: 'Megnyugtató lelet',
            description: 'A mesterséges intelligencia modellek jóindulatú (benignus) bőrelváltozásra utaló jegyeket azonosítottak. Nem észlelhető a rosszindulatú sejtburjánzásra vagy melanómára jellemző atípusos mintázat.',
            actionHeader: 'Ajánlott teendők',
            actions: [
                {
                    title: 'Rendszeres önvizsgálat (ABCD szabály)',
                    desc: 'Félévente-évente vizsgálja át anyajegyeit az A (Aszimmetria), B (Határ), C (Szín), D (Átmérő) és E (Változás) szempontok szerint.'
                },
                {
                    title: 'Figyelje a változásokat',
                    desc: 'Amennyiben a képlet alakja, színe megváltozik, növekedni kezd, viszket vagy vérzik, azonnal mutassa meg orvosnak.'
                },
                {
                    title: 'Éves szűrővizsgálat',
                    desc: 'Az évenkénti szakorvosi dermatoszkópos szűrés a leghatékonyabb módja a bőrrák korai megelőzésének.'
                }
            ]
        },
        uncertain: {
            title: 'Bizonytalan / Határeset lelet',
            subtitle: 'Kontroll vagy ismételt fotó javasolt (Inconclusive)',
            badge: 'További vizsgálat ajánlott',
            description: 'A modell határérték közeli pontszámot adott, vagy a kép sajátosságai miatt nem határozható meg kellő magabiztossággal a mintázat jellege. Javasoljuk a kép újbóli elkészítését jobb megvilágításban, vagy szakorvosi ellenőrzést.',
            actionHeader: 'Mit tegyen most?',
            actions: [
                {
                    title: 'Készítsen új képet optimális fényviszonyok közt',
                    desc: 'Kerülje a közvetlen vakut vagy a túl erős árnyékot. Ügyeljen arra, hogy a képlet éles és fókuszált legyen a kamera közepén.'
                },
                {
                    title: 'Szakorvosi dermatoszkópia',
                    desc: 'Bizonytalan vagy nehezen megítélhető képletek esetén a bőrgyógyászati dermatoszkópos mikroszkópos vizsgálat ad pontos választ.'
                },
                {
                    title: 'Ne próbálja mechanikusan kaparni vagy kezelni',
                    desc: 'Bármilyen kétség esetén bízza szakorvosra az anyajegy pontos felmérését.'
                }
            ]
        },
        high: {
            title: 'Szakorvosi konzultáció javasolt',
            subtitle: 'Gyanús vagy atípusos mintázat (Elevated Risk)',
            badge: 'Konzultáció indokolt',
            description: 'A képen olyan atípusos vagy aszimmetrikus strukturális jegyek találhatók, amelyeknél a szakorvosi kivizsgálás erősen indokolt. Fontos: ez a gépi előszűrés NEM végleges diagnózis, de szakorvosi dermatoszkópos vizsgálatot tesz szükségessé.',
            actionHeader: 'Kiemelten javasolt lépések',
            actions: [
                {
                    title: 'Keresse fel a területileg illetékes bőrgyógyászt',
                    desc: 'Javasoljuk, hogy a közeljövőben mutassa meg az elváltozást bőrgyógyász szakorvosnak dermatoszkópos vizsgálatra.'
                },
                {
                    title: 'Kerülje a napozást és az irritációt',
                    desc: 'Óvja a bőrfelületet a dörzsölődéstől, közvetlen UV-sugárzástól és mechanikai sérüléstől.'
                },
                {
                    title: 'Őrizze meg a fotót összehasonlításként',
                    desc: 'Az orvosi konzultáció során hasznos lehet bemutatni a rögzített képet az időbeli változások követésére.'
                }
            ]
        }
    };

    const currentConfig = verdictConfig[overallRisk] || verdictConfig.uncertain;

    // Consensus sentence
    let consensusNote = '';
    if (preds.length >= 2) {
        const agreement = preds.every(p => p.risk_level === preds[0].risk_level);
        if (agreement) {
            consensusNote = 'Mindkét független AI modell (MobileNet és EfficientNet) azonos kategóriába sorolta a mintázatot.';
        } else {
            consensusNote = 'A modellek eltérő valószínűségi értéket adtak; a rendszer az óvatosság elve alapján a magasabb kockázatot veszi figyelembe.';
        }
    } else if (preds.length === 1) {
        consensusNote = `Kiértékelés: ${preds[0].model} modell alapján.`;
    }

    let html = '';

    // 1. HERO VERDICT CARD
    html += `
        <div class="result-hero-card ${overallRisk}">
            <div class="result-hero-badge-wrap">
                <span class="result-hero-badge ${overallRisk}">${currentConfig.badge}</span>
                <span class="result-hero-consensus">${consensusNote}</span>
            </div>
            
            <h1 class="result-hero-title">${currentConfig.title}</h1>
            <p class="result-hero-subtitle">${currentConfig.subtitle}</p>
            <p class="result-hero-desc">${currentConfig.description}</p>

            <!-- Risk Gauge Meter -->
            <div class="risk-gauge-container">
                <div class="risk-gauge-labels">
                    <span class="gauge-lbl-low">Jóindulatú</span>
                    <span class="gauge-lbl-mid">Bizonytalan</span>
                    <span class="gauge-lbl-high">Gyanús</span>
                </div>
                <div class="risk-gauge-track">
                    <div class="risk-gauge-marker" style="left: ${gaugePercent}%" title="Becsült kockázat: ${gaugePercent}%">
                        <div class="marker-pin ${overallRisk}"></div>
                        <span class="marker-bubble">${gaugePercent}%</span>
                    </div>
                </div>
                <div class="risk-gauge-subtext">
                    Kockázati indikátor (0% = biztosan jóindulatú, 100% = erősen atípusos)
                </div>
            </div>
        </div>
    `;

    // 2. ACTIONABLE GUIDANCE (Mit tegyek most?)
    html += `
        <div class="result-guidance-card">
            <h3 class="guidance-heading">${currentConfig.actionHeader}</h3>
            <div class="guidance-list">
                ${currentConfig.actions.map((act, idx) => `
                    <div class="guidance-item">
                        <div class="guidance-num">${idx + 1}</div>
                        <div class="guidance-content">
                            <strong class="guidance-item-title">${act.title}</strong>
                            <p class="guidance-item-desc">${act.desc}</p>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;

    // 3. MODEL SUMMARIES (User-friendly view)
    html += `
        <div class="result-models-summary">
            <h3 class="summary-heading">Alkalmazott mesterséges intelligencia modellek</h3>
            <div class="models-grid">
    `;

    for (const [key, pred] of Object.entries(data.predictions)) {
        const risk = pred.risk_level;
        const confidenceText = `${pred.confidence_label} (${pred.confidence_percent}%)`;
        const readableVerdict = risk === 'low' ? 'Jóindulatú mintázat' :
                                risk === 'high' ? 'Gyanús mintázat' : 'Bizonytalan mintázat';
        
        const friendlyName = key === 'efficientnet' ? 'EfficientNet (Nagy pontosságú mélyháló)' :
                             key === 'mobilenet' ? 'MobileNetV2 (Gyors szűrőmodell)' : pred.model;

        html += `
            <div class="model-summary-card ${risk}">
                <div class="model-card-top">
                    <span class="model-friendly-name">${friendlyName}</span>
                    <span class="model-risk-pill ${risk}">${readableVerdict}</span>
                </div>
                <div class="model-confidence-row">
                    <span class="model-conf-label">Megbízhatóság:</span>
                    <span class="model-conf-val">${confidenceText}</span>
                </div>
            </div>
        `;
    }

    html += `
            </div>
        </div>
    `;

    // 4. QUALITY REPORT (IF ANY)
    if (data.quality) {
        const qScore = data.quality.quality_score;
        const qAcceptable = data.quality.is_acceptable;
        const qWarnings = data.quality.warnings || [];

        html += `
            <div class="result-quality-box">
                <div class="quality-box-header">
                    <span class="quality-box-title">Képminőség értékelése</span>
                    <span class="quality-box-score ${qAcceptable ? 'good' : 'warning'}">Minőség: ${qScore}/100</span>
                </div>
                ${qWarnings.length > 0 ? `
                    <div class="quality-box-warnings">
                        ${qWarnings.map(w => `<div class="qb-warn">Megjegyzés: ${w}</div>`).join('')}
                    </div>
                ` : `
                    <div class="quality-box-ok">A feltöltött kép élessége és megvilágítása megfelelt a vizsgálati feltételeknek.</div>
                `}
            </div>
        `;
    }

    // 5. COLLAPSIBLE TECHNICAL & TELEMETRY DETAILS
    html += `
        <details class="tech-details-accordion">
            <summary class="tech-details-summary">
                <span>Részletes technikai és diagnosztikai adatok (fejlesztőknek és orvosoknak)</span>
                <span class="tech-details-arrow">›</span>
            </summary>
            <div class="tech-details-content">
                <table class="tech-table">
                    <tbody>
                        ${Object.entries(data.predictions).map(([k, p]) => `
                            <tr>
                                <td><strong>${p.model} nyers pontszám (raw score)</strong></td>
                                <td><code>${p.raw_score}</code> (Döntési határ: 0.50)</td>
                            </tr>
                            <tr>
                                <td><strong>${p.model} számítási idő</strong></td>
                                <td>${p.processing_time_ms} ms</td>
                            </tr>
                        `).join('')}
                        <tr>
                            <td><strong>Szerveroldali teljes futási idő</strong></td>
                            <td>${data.total_processing_time_ms} ms</td>
                        </tr>
                        <tr>
                            <td><strong>Hálózati oda-vissza út (RTT)</strong></td>
                            <td>${data._networkTime} ms</td>
                        </tr>
                        ${data.image_info ? `
                            <tr>
                                <td><strong>Bemeneti felbontás</strong></td>
                                <td>${data.image_info.original_size} → ${data.image_info.processed_size}</td>
                            </tr>
                            <tr>
                                <td><strong>Digitális hajeltávolítás (DullRazor)</strong></td>
                                <td>${data.image_info.hair_removal && data.image_info.hair_removal.hair_detected ? 
                                    `Alkalmazva (${data.image_info.hair_removal.pixels_modified} képpont)` : 
                                    'Nem volt szükség szőrszál-korrekcióra'}</td>
                            </tr>
                        ` : ''}
                        ${data.quality ? `
                            <tr>
                                <td><strong>Képélesség (Laplace-variancia)</strong></td>
                                <td>${data.quality.blur ? data.quality.blur.laplacian_var : 'N/A'} (Küszöb: 50.0)</td>
                            </tr>
                        ` : ''}
                    </tbody>
                </table>
            </div>
        </details>
    `;

    // 6. MEDICAL DISCLAIMER
    html += `
        <div class="disclaimer-banner">
            <div class="disclaimer-title">Fontos orvosi tájékoztatás</div>
            <p>A DermAI Scan egy mesterséges intelligencián alapuló kísérleti előszűrő rendszer, amely <strong>nem helyettesíti a szakképzett orvosi vizsgálatot</strong> és nem ad hivatalos orvosi diagnózist. Amennyiben bármilyen kételye vagy észlelhető bőrelváltozása van, kérjük, haladéktalanul forduljon bőrgyógyász szakorvoshoz!</p>
        </div>
    `;

    body.innerHTML = html;
}

function showQualityError(data) {
    showScreen('results');
    const body = document.getElementById('results-body');

    let html = `
        <div class="result-hero-card uncertain">
            <div class="result-hero-badge-wrap">
                <span class="result-hero-badge uncertain">Figyelmeztetés</span>
            </div>
            <h1 class="result-hero-title">Gyenge képminőség észlelve</h1>
            <p class="result-hero-subtitle">Image Quality Warning</p>
            <p class="result-hero-desc">
                A feltöltött fotó élessége vagy megvilágítása elmarad az optimális vizsgálati szinttől. 
                Gyenge minőségű képen a mesterséges intelligencia modellek nem tudnak megbízható kiértékelést garantálni.
            </p>
        </div>
    `;

    if (data.quality) {
        html += `
            <div class="result-quality-box">
                <div class="quality-box-header">
                    <span class="quality-box-title">Részletes minőségi hibák</span>
                    <span class="quality-box-score warning">Pontszám: ${data.quality.quality_score}/100</span>
                </div>
                <div class="quality-box-warnings">
                    ${(data.quality.warnings || []).map(w => `<div class="qb-warn">Figyelem: ${w}</div>`).join('')}
                </div>
            </div>
        `;
    }

    html += `
        <div style="margin-top: 1rem; display: flex; flex-direction: column; gap: 0.65rem;">
            <button class="btn-primary" style="width: 100%; border-radius: 0;" onclick="analyzeImage(true)">Elemzés végrehajtása ennek ellenére</button>
            <button class="btn-secondary" style="width: 100%; border-radius: 0;" onclick="goHome()">Új, élesebb kép választása</button>
        </div>
    `;

    body.innerHTML = html;
}

// --- Drag & Drop Support ---
function setupDragAndDrop() {
    const uploadBtn = document.getElementById('btn-upload');

    ['dragenter', 'dragover'].forEach(eventName => {
        window.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (uploadBtn) uploadBtn.classList.add('drag-over');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        window.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (uploadBtn) uploadBtn.classList.remove('drag-over');
        }, false);
    });

    window.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (uploadBtn) uploadBtn.classList.remove('drag-over');

        const dt = e.dataTransfer;
        if (!dt) return;
        const files = dt.files;
        if (files && files.length > 0) {
            const file = files[0];
            const isImage = (file.type && file.type.startsWith('image/')) ||
                            /\.(jpe?g|png|webp|bmp|gif|tiff?|heic|heif)$/i.test(file.name || '');
            if (!isImage) {
                alert('Kérjük, képfájlt válasszon ki (pl. JPG, PNG, WebP)!');
                return;
            }
            currentImage = file;
            showPreview(file);
        }
    }, false);
}

// --- Utilities ---
function closeCamera() {
    stopCamera();
    goHome();
}
