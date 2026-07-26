import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js';

env.allowLocalModels  = true;
env.allowRemoteModels = false;
env.localModelPath    = '../models/';

// `ontouchstart in window` is true on iOS Safari (all modes, including iPad
// desktop-mode which fakes a Mac UA) but false on actual macOS Safari.
const isIOS = /iPhone|iPod/.test(navigator.userAgent) ||
              /iPad/.test(navigator.userAgent) ||
              (navigator.vendor === 'Apple Computer, Inc.' &&
               navigator.maxTouchPoints > 1 &&
               'ontouchstart' in window);

const SCAN_PHRASES = [
  'Looking for the perfect creature…',
  'Chasing the Cheshire Cat…',
  'Consulting the oracle of meow…',
  'Negotiating with a higher power…',
  'Asking if Schrödinger\'s cat is in…',
  'Summoning the keyboard cat…',
  'Cross-referencing with ancient Egypt…',
  'Checking if it\'s a cat or a loaf…',
  'pst pst pst pst pst…',
  'Hiding this from ALF…',
];

// ── Rarity table ──────────────────────────────────────────────────────────────
// `base` is the score floor for each colour (0–10).  Detection confidence then
// adds up to ~±1.2 points of variation, so common cats land around 4–6 and
// rare ones around 8–10.  This avoids the "always 10" problem: YOLOS is
// nearly always >0.9 confident when it finds a cat, so using confidence×10
// directly saturates the scale immediately.
const RARITY = {
  'Ginger':         { base: 4.0, label: 'Common'   },
  'Midnight Black': { base: 4.5, label: 'Common'   },
  'Charcoal':       { base: 4.0, label: 'Common'   },
  'Ash Gray':       { base: 4.5, label: 'Common'   },
  'Silver':         { base: 5.5, label: 'Uncommon' },
  'Snow White':     { base: 5.5, label: 'Uncommon' },
  'Spotted':        { base: 6.0, label: 'Uncommon' },
  'Russet':         { base: 6.5, label: 'Uncommon' },
  'Golden':         { base: 7.5, label: 'Rare'     },
  'Slate Blue':     { base: 8.5, label: 'Rare'     },
  'Lilac':          { base: 9.5, label: 'Very Rare'},
};

// ── State ─────────────────────────────────────────────────────────────────────

let detector   = null;
let stream     = null;
let facingMode = 'environment';

// ── DOM ───────────────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);

const loadingScreen   = $('loadingScreen');
const cameraScreen    = $('cameraScreen');
const listScreen      = $('listScreen');
const scanningOverlay = $('scanningOverlay');
const progressFill    = $('progressFill');
const progressLabel   = $('progressLabel');
const loadingSubtitle = $('loadingSubtitle');
const video           = $('video');
const cameraError     = $('cameraError');
const captureBtn      = $('captureBtn');
const uploadBtn       = $('uploadBtn');
const flipBtn         = $('flipBtn');
const listCaptureBtn  = $('listCaptureBtn');
const listUploadBtn   = $('listUploadBtn');
const exportBtn       = $('exportBtn');
const importBtn       = $('importBtn');
const fileInput       = $('fileInput');
const importInput     = $('importInput');
const catList         = $('catList');
const scanCanvas      = $('scanCanvas');
const scanText        = $('scanText');

// ── Screens ───────────────────────────────────────────────────────────────────

function showScreen(name) {
  loadingScreen.classList.add('hidden');
  cameraScreen.classList.add('hidden');
  listScreen.classList.add('hidden');
  scanningOverlay.classList.add('hidden');
  if (name === 'loading') loadingScreen.classList.remove('hidden');
  if (name === 'camera')  cameraScreen.classList.remove('hidden');
  if (name === 'list')    listScreen.classList.remove('hidden');
}

// ── Model loading ─────────────────────────────────────────────────────────────

function updateProgress(pct, subtitle) {
  const v = Math.min(100, Math.max(0, pct));
  progressFill.style.width = `${v}%`;
  progressLabel.textContent = `${Math.round(v)}%`;
  if (subtitle) loadingSubtitle.textContent = subtitle;
}

async function loadModel() {
  let fileProgress = {};

  // iOS Safari: force single-threaded WASM — multi-thread WASM needs
  // SharedArrayBuffer (COOP/COEP headers) and stresses the memory budget.
  if (isIOS) {
    env.backends.onnx.wasm.numThreads = 1;
  }

  // Try WebGPU on non-iOS — much faster on modern desktop hardware.
  // Skip on iOS: it exposes navigator.gpu in iOS 17+ but is fragile for ONNX.
  let device = 'wasm';
  if (!isIOS && 'gpu' in navigator) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) device = 'webgpu';
    } catch {}
  }

  const tryLoad = async (dev) => {
    loadingSubtitle.textContent = dev === 'webgpu' ? 'Loading with WebGPU…' : 'Loading with WASM…';
    return pipeline('object-detection', 'Xenova/yolos-tiny', {
      device: dev,
      dtype: 'q4',
      progress_callback: (p) => {
        if (p.status === 'progress' && p.file) {
          fileProgress[p.file] = p.progress ?? 0;
          const vals = Object.values(fileProgress);
          const avg  = vals.reduce((a, b) => a + b, 0) / vals.length;
          updateProgress(avg, `Loading… ${p.file.split('/').pop()}`);
        } else if (p.status === 'ready') {
          updateProgress(100, 'Ready!');
        }
      },
    });
  };

  if (device === 'webgpu') {
    try {
      detector = await tryLoad('webgpu');
    } catch (err) {
      console.warn('WebGPU load failed, falling back to WASM:', err);
      fileProgress = {};
      updateProgress(0);
      detector = await tryLoad('wasm');
    }
  } else {
    detector = await tryLoad('wasm');
  }
}

// ── Camera ────────────────────────────────────────────────────────────────────

async function startCamera() {
  if (stream) stream.getTracks().forEach(t => t.stop());
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode, width: { ideal: 1920 }, height: { ideal: 1080 } },
      audio: false,
    });
    video.srcObject = stream;
    cameraError.classList.add('hidden');
  } catch {
    stream = null;
    cameraError.classList.remove('hidden');
  }
}

function stopCamera() {
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
}

// ── Capture & detect ──────────────────────────────────────────────────────────

async function captureAndDetect(sourceCanvas) {
  // Show scanning overlay with a blurred preview of the shot
  const ctx = scanCanvas.getContext('2d');
  scanCanvas.width  = sourceCanvas.width;
  scanCanvas.height = sourceCanvas.height;
  ctx.drawImage(sourceCanvas, 0, 0);
  scanText.textContent = SCAN_PHRASES[Math.floor(Math.random() * SCAN_PHRASES.length)];
  scanningOverlay.classList.remove('hidden');

  stopCamera();

  const detCanvas = resizeForDetection(sourceCanvas);
  const blob      = await canvasToBlob(detCanvas);
  const blobUrl   = URL.createObjectURL(blob);
  let entry = null;

  try {
    const results = await detector(blobUrl, { threshold: 0.3 });
    const cats    = results.filter(r => r.label === 'cat');

    if (cats.length) {
      const best = cats.sort((a, b) => b.score - a.score)[0];
      // Scale box back to original canvas coordinates for colour sampling
      const sx  = sourceCanvas.width  / detCanvas.width;
      const sy  = sourceCanvas.height / detCanvas.height;
      const box = {
        xmin: best.box.xmin * sx, ymin: best.box.ymin * sy,
        xmax: best.box.xmax * sx, ymax: best.box.ymax * sy,
      };
      const color     = extractCenterColor(sourceCanvas, box);
      const thumbnail = extractPixelThumbnail(sourceCanvas, box);
      entry = buildEntry(color, best.score);
      entry.thumbnail = thumbnail;
    }
  } catch (err) {
    console.error('Detection error:', err);
  } finally {
    URL.revokeObjectURL(blobUrl);
    scanningOverlay.classList.add('hidden');
  }

  if (entry) {
    saveEntry(entry);
  } else {
    showToast('No cat detected — try again!');
    showScreen('camera');
    startCamera();
    return;
  }

  showScreen('list');
  renderList();
}

function canvasToBlob(canvas) {
  return new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.92));
}

// Shrink the bounding-box region to a tiny pixel grid and store as PNG.
// 16×16 gives a chunky pixel-art portrait; PNG of solid-ish cat colours
// typically encodes to ~300–600 bytes before base64.
function extractPixelThumbnail(canvas, box, maxPx = 32) {
  const { xmin, ymin, xmax, ymax } = box;
  const bw = xmax - xmin;
  const bh = ymax - ymin;
  const scale = maxPx / Math.max(bw, bh);
  const pw = Math.max(1, Math.round(bw * scale));
  const ph = Math.max(1, Math.round(bh * scale));
  const tiny = document.createElement('canvas');
  tiny.width  = pw;
  tiny.height = ph;
  const ctx = tiny.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(canvas, xmin, ymin, bw, bh, 0, 0, pw, ph);
  return tiny.toDataURL('image/png');
}

// YOLOS-tiny processes at 512px internally; passing a full camera frame
// (1920×1080) inflates memory ~12× for free.  Cap the detection input at 640px
// on the long edge — colour extraction still uses the original full-res canvas.
function resizeForDetection(canvas, maxPx = 640) {
  const scale = Math.min(1, maxPx / Math.max(canvas.width, canvas.height));
  if (scale === 1) return canvas;
  const out = document.createElement('canvas');
  out.width  = Math.round(canvas.width  * scale);
  out.height = Math.round(canvas.height * scale);
  out.getContext('2d').drawImage(canvas, 0, 0, out.width, out.height);
  return out;
}

// ── Color helpers ─────────────────────────────────────────────────────────────

function extractCenterColor(canvas, box) {
  const { xmin, ymin, xmax, ymax } = box;
  const w  = xmax - xmin;
  const h  = ymax - ymin;
  const sx = Math.round(xmin + w * 0.2);
  const sy = Math.round(ymin + h * 0.2);
  const sw = Math.round(w * 0.6);
  const sh = Math.round(h * 0.6);

  const ctx  = canvas.getContext('2d');
  const data = ctx.getImageData(sx, sy, sw, sh).data;
  let r = 0, g = 0, b = 0, n = 0;
  for (let i = 0; i < data.length; i += 16) {
    r += data[i]; g += data[i + 1]; b += data[i + 2]; n++;
  }
  return { r: Math.round(r / n), g: Math.round(g / n), b: Math.round(b / n) };
}

function rgbToHsl(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const l   = (max + min) / 2;
  if (max === min) return { h: 0, s: 0, l: l * 100 };
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h;
  switch (max) {
    case r: h = ((g - b) / d + (g < b ? 6 : 0)) / 6; break;
    case g: h = ((b - r) / d + 2) / 6; break;
    default: h = ((r - g) / d + 4) / 6;
  }
  return { h: h * 360, s: s * 100, l: l * 100 };
}

function getCatColorName(r, g, b) {
  const { h, s, l } = rgbToHsl(r, g, b);
  if (l > 85)  return 'Snow White';
  if (l < 12)  return 'Midnight Black';
  if (s < 14) {
    if (l < 35) return 'Charcoal';
    if (l < 60) return 'Silver';
    return 'Ash Gray';
  }
  if (h < 20 || h > 335) return 'Russet';
  if (h < 48)  return 'Ginger';
  if (h < 70)  return 'Golden';
  if (h < 150) return 'Spotted';
  if (h < 265) return 'Slate Blue';
  return 'Lilac';
}

function toHex({ r, g, b }) {
  return '#' + [r, g, b].map(v => v.toString(16).padStart(2, '0')).join('');
}

function contrastColor({ r, g, b }) {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b > 140 ? '#1a1a1a' : '#ffffff';
}

// ── Entry building & scoring ──────────────────────────────────────────────────

function buildEntry(color, confidence) {
  const colorName  = getCatColorName(color.r, color.g, color.b);
  const rarity     = RARITY[colorName] ?? { base: 5.0, label: 'Common' };
  // Confidence (typically 0.3–1.0) adds roughly ±1.2 variation around the base.
  // Centre point 0.65 maps to 0 delta; 1.0 maps to +1.225; 0.3 maps to -1.225.
  const delta      = (confidence - 0.65) * 3.5;
  const catometer  = +Math.min(10, Math.max(1, rarity.base + delta)).toFixed(1);

  return {
    id:          Date.now(),
    date:        new Date().toISOString(),
    colorName,
    colorHex:    toHex(color),
    textColor:   contrastColor(color),
    confidence:  +confidence.toFixed(3),
    rarityBase:  rarity.base,
    rarityLabel: rarity.label,
    catometer,
  };
}

// ── localStorage ──────────────────────────────────────────────────────────────

const STORE_KEY = 'un-gaaat-cats';

function loadCollection() {
  try {
    return JSON.parse(localStorage.getItem(STORE_KEY) || '[]');
  } catch {
    return [];
  }
}

function saveEntry(entry) {
  const col = loadCollection();
  col.unshift(entry);
  localStorage.setItem(STORE_KEY, JSON.stringify(col));
}

function deleteEntry(id) {
  const col = loadCollection().filter(e => e.id !== id);
  localStorage.setItem(STORE_KEY, JSON.stringify(col));
}

// ── Export / Import ───────────────────────────────────────────────────────────

function exportCollection() {
  const col  = loadCollection();
  const blob = new Blob([JSON.stringify(col, null, 2)], { type: 'application/json' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `un-gaaat-${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function importCollection(file) {
  const reader = new FileReader();
  reader.onload = e => {
    try {
      const imported = JSON.parse(e.target.result);
      if (!Array.isArray(imported)) throw new Error('Not an array');
      const existing = loadCollection();
      const ids      = new Set(existing.map(x => x.id));
      const merged   = [...imported.filter(x => !ids.has(x.id)), ...existing];
      merged.sort((a, b) => b.id - a.id);
      localStorage.setItem(STORE_KEY, JSON.stringify(merged));
      renderList();
      showToast(`Imported ${imported.length} entries`);
    } catch {
      showToast('Invalid file');
    }
  };
  reader.readAsText(file);
}

// ── List rendering ────────────────────────────────────────────────────────────

function formatDate(iso) {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

function renderList() {
  const col = loadCollection();

  const header = document.querySelector('.list-header');
  header.classList.toggle('hidden', !col.length);

  if (!col.length) {
    catList.innerHTML = `
      <div class="empty-state">
        <img src="media/icon.png" class="empty-state-logo" alt="Un Gaaat" />
        <div class="empty-state-text">Un Gaaat</div>
        <div class="empty-state-sub">
          Point your camera at a cat and tap the shutter button below.
          A q4-quantized YOLOS-tiny ONNX model running locally in your browser
          will identify it, score it by colour rarity, and add it to your
          collection — no data ever leaves your device.
        </div>
        <div class="empty-state-hint">
          <i class="ph-light ph-arrow-down"></i> tap the shutter to begin
        </div>
      </div>`;
    return;
  }

  catList.innerHTML = col.map(entry => `
    <div class="sticker-card" data-id="${entry.id}" style="background:var(--bg2); color:${entry.textColor};">
      ${entry.thumbnail
        ? `<img class="card-photo" src="${entry.thumbnail}" alt="${entry.colorName} cat" style="background-color:${entry.colorHex};">`
        : `<div class="card-photo" style="background-color:${entry.colorHex};"></div>`
      }
      <div class="card-info" style="background:${entry.colorHex}; color:${entry.textColor};">
        <div class="card-name">${entry.colorName} Cat</div>
        <div class="card-meter-row">
          <div class="card-meter-track">
            <div class="card-meter-fill" style="width:${entry.catometer * 10}%; background:${entry.textColor};"></div>
          </div>
          <div class="card-meter-value">${entry.catometer}<span style="font-size:9px;opacity:0.6;">/10</span></div>
        </div>
        <div class="card-footer">
          <div class="card-rarity-badge" style="background:${entry.textColor}22; color:${entry.textColor};">${entry.rarityLabel}</div>
          <div class="card-date" style="color:${entry.textColor};">${formatDate(entry.date)}</div>
        </div>
      </div>
      <button class="card-delete" data-id="${entry.id}" aria-label="Delete"><i class="ph-light ph-x"></i></button>
    </div>
  `).join('');

  // Delete buttons
  catList.querySelectorAll('.card-delete').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      deleteEntry(Number(btn.dataset.id));
      renderList();
    });
  });
}

// ── Toast ─────────────────────────────────────────────────────────────────────

function showToast(msg) {
  const t = document.createElement('div');
  t.className   = 'toast';
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 2700);
}

// ── Event listeners ───────────────────────────────────────────────────────────

function triggerCapture() {
  if (!video.videoWidth) return;
  const c = document.createElement('canvas');
  c.width  = video.videoWidth;
  c.height = video.videoHeight;
  c.getContext('2d').drawImage(video, 0, 0);
  captureAndDetect(c);
}

function triggerUpload() { fileInput.click(); }

captureBtn.addEventListener('click',     triggerCapture);
listCaptureBtn.addEventListener('click', () => { showScreen('camera'); startCamera(); });

uploadBtn.addEventListener('click',     triggerUpload);
listUploadBtn.addEventListener('click', triggerUpload);

flipBtn.addEventListener('click', () => {
  facingMode = facingMode === 'environment' ? 'user' : 'environment';
  startCamera();
});

fileInput.addEventListener('change', e => {
  const file = e.target.files[0];
  if (!file) return;
  fileInput.value = '';
  const reader = new FileReader();
  reader.onload = evt => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement('canvas');
      c.width  = img.naturalWidth;
      c.height = img.naturalHeight;
      c.getContext('2d').drawImage(img, 0, 0);
      captureAndDetect(c);
    };
    img.src = evt.target.result;
  };
  reader.readAsDataURL(file);
});

exportBtn.addEventListener('click', exportCollection);
importBtn.addEventListener('click', () => importInput.click());
importInput.addEventListener('change', e => {
  const file = e.target.files[0];
  if (file) { importInput.value = ''; importCollection(file); }
});

// ── Init ──────────────────────────────────────────────────────────────────────

async function init() {
  if ('ontouchstart' in window || navigator.maxTouchPoints > 0) {
    document.body.classList.add('is-touch');
  }
  showScreen('loading');
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('../sw.js').catch(() => {});
  }

  try {
    await loadModel();
  } catch (err) {
    console.error('Model load failed:', err);
    loadingSubtitle.textContent = 'Failed to load model — check your connection.';
    return;
  }

  showScreen('list');
  renderList();
}

init();
