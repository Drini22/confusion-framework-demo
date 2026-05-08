/* ===========================================================================
 * Mini-demo for the "The model" section of the framework page.
 *
 * Three small canvases:
 *   1. Field of view  — drag two sources (red = p1, blue = p2) on (x, y) plane
 *   2. Clean mu_P     — high-density grayscale of  F1*phi(x-p1) + F2*phi(x-p2)
 *   3. Pixelated noisy — 20x20 detector binning + cached Gaussian noise * sigma
 *
 * Sliders: F1, F2, sigma_noise.  Plot range +/- 3 sigma.
 *
 * Self-contained: no dependency on app.js.
 * =========================================================================== */

(() => {

const cvFov   = document.getElementById('model-fov');
const cvClean = document.getElementById('model-clean');
const cvNoisy = document.getElementById('model-noisy');
if (!cvFov || !cvClean || !cvNoisy) return;   /* not on this page */

const SIGMA_PSF = 1.0;
const RANGE     = 3.0;
const PI_FINE   = 240;
const PI_PIX    = 20;

const S = {
  p1x: -0.5, p1y: 0.0,
  p2x:  0.5, p2y: 0.0,
  F1: 1.0, F2: 1.0, sigma: 0.10,
};

const phi2 = (dx, dy) =>
  Math.exp(-(dx*dx + dy*dy) / (2.0 * SIGMA_PSF * SIGMA_PSF));

/* Cached Gaussian-noise pattern (drawn once) */
const noise = new Float64Array(PI_PIX * PI_PIX);
let noiseInit = false;
function gauss() {
  const u1 = Math.random() + 1e-12;
  const u2 = Math.random();
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}
function ensureNoise() {
  if (noiseInit) return;
  for (let i = 0; i < noise.length; i++) noise[i] = gauss();
  noiseInit = true;
}

/* mu_P arrays (computed each frame) */
const muFine = new Float64Array(PI_FINE * PI_FINE);
const muPix  = new Float64Array(PI_PIX  * PI_PIX);

function compute() {
  let muMax = 0.0;
  /* Top of image = highest y (math convention). */
  for (let i = 0; i < PI_FINE; i++) {
    const y = +RANGE - (i + 0.5) / PI_FINE * 2.0 * RANGE;
    for (let j = 0; j < PI_FINE; j++) {
      const x = -RANGE + (j + 0.5) / PI_FINE * 2.0 * RANGE;
      const v = S.F1 * phi2(x - S.p1x, y - S.p1y)
              + S.F2 * phi2(x - S.p2x, y - S.p2y);
      muFine[i * PI_FINE + j] = v;
      if (v > muMax) muMax = v;
    }
  }
  if (muMax < 1e-9) muMax = 1e-9;

  /* Bin to detector grid */
  const block = PI_FINE / PI_PIX;   /* = 12 */
  for (let pi = 0; pi < PI_PIX; pi++) {
    for (let pj = 0; pj < PI_PIX; pj++) {
      let s = 0;
      for (let di = 0; di < block; di++) {
        for (let dj = 0; dj < block; dj++) {
          s += muFine[(pi*block + di) * PI_FINE + (pj*block + dj)];
        }
      }
      muPix[pi * PI_PIX + pj] = s / (block * block);
    }
  }
  return muMax;
}

/* Canvas helpers */
function setupCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth, cssH = cv.clientHeight;
  if (cv.width  !== Math.round(cssW * dpr) ||
      cv.height !== Math.round(cssH * dpr)) {
    cv.width  = Math.round(cssW * dpr);
    cv.height = Math.round(cssH * dpr);
  }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: cssW, h: cssH, dpr };
}
function getCSSColor(name) {
  return getComputedStyle(document.body).getPropertyValue(name).trim();
}

/* Mouse tracking (absolute coords, per-canvas conversion) */
let mouseAbs = null, pressed = false, held = false, released = false;
let dragSrc = -1;
document.addEventListener('mousemove', (e) => {
  mouseAbs = { x: e.clientX, y: e.clientY };
});
document.addEventListener('mousedown', () => { pressed = true; held = true; });
document.addEventListener('mouseup',   () => { released = true; held = false; });
function localMouse(cv) {
  if (!mouseAbs) return null;
  const r = cv.getBoundingClientRect();
  return { x: mouseAbs.x - r.left, y: mouseAbs.y - r.top };
}

/* Field of view */
function draw_fov() {
  const { ctx, w, h } = setupCanvas(cvFov);
  const sz = Math.min(w, h);
  const x0 = (w - sz) / 2, y0 = (h - sz) / 2;
  const d2sx = (dx) => x0 + (dx + RANGE) / (2*RANGE) * sz;
  const d2sy = (dy) => y0 + sz - (dy + RANGE) / (2*RANGE) * sz;
  const s2dx = (sx) => (sx - x0) / sz * 2*RANGE - RANGE;
  const s2dy = (sy) => -((sy - y0) / sz * 2*RANGE - RANGE);

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = getCSSColor('--canvas-bg');
  ctx.fillRect(x0, y0, sz, sz);

  /* Gridlines at integer values within range */
  ctx.strokeStyle = getCSSColor('--canvas-grid');
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let t = -2; t <= 2; t++) {
    if (t === 0) continue;
    const gx = d2sx(t), gy = d2sy(t);
    ctx.moveTo(gx, y0); ctx.lineTo(gx, y0 + sz);
    ctx.moveTo(x0, gy); ctx.lineTo(x0 + sz, gy);
  }
  ctx.stroke();

  /* Origin axes */
  ctx.strokeStyle = getCSSColor('--canvas-axis');
  ctx.beginPath();
  ctx.moveTo(d2sx(0), y0);     ctx.lineTo(d2sx(0), y0 + sz);
  ctx.moveTo(x0, d2sy(0));     ctx.lineTo(x0 + sz, d2sy(0));
  ctx.stroke();

  ctx.strokeStyle = getCSSColor('--border');
  ctx.strokeRect(x0, y0, sz, sz);

  const ppu = sz / (2 * RANGE);
  const p1sx = d2sx(S.p1x), p1sy = d2sy(S.p1y);
  const p2sx = d2sx(S.p2x), p2sy = d2sy(S.p2y);

  /* Faint dashed PSF circle at r = sigma to hint at width */
  const red  = getCSSColor('--plot-red');
  const blue = getCSSColor('--plot-blue');
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.globalAlpha = 0.5;
  ctx.strokeStyle = red;
  ctx.beginPath(); ctx.arc(p1sx, p1sy, SIGMA_PSF * ppu, 0, Math.PI*2); ctx.stroke();
  ctx.strokeStyle = blue;
  ctx.beginPath(); ctx.arc(p2sx, p2sy, SIGMA_PSF * ppu, 0, Math.PI*2); ctx.stroke();
  ctx.setLineDash([]);
  ctx.globalAlpha = 1.0;

  /* Source dots (drag handles) */
  const mp = localMouse(cvFov);
  let r1 = 5, r2 = 5;
  let near1 = false, near2 = false;
  if (mp) {
    const d1 = (mp.x - p1sx)**2 + (mp.y - p1sy)**2;
    const d2 = (mp.x - p2sx)**2 + (mp.y - p2sy)**2;
    near1 = d1 < 100; near2 = d2 < 100;
    if (dragSrc === 0 || near1) r1 = 7;
    if (dragSrc === 1 || near2) r2 = 7;
  }
  ctx.fillStyle = red;
  ctx.beginPath(); ctx.arc(p1sx, p1sy, r1, 0, Math.PI*2); ctx.fill();
  ctx.fillStyle = blue;
  ctx.beginPath(); ctx.arc(p2sx, p2sy, r2, 0, Math.PI*2); ctx.fill();

  /* Drag interaction */
  if (mp && pressed && dragSrc === -1
      && mp.x >= x0 && mp.x <= x0 + sz
      && mp.y >= y0 && mp.y <= y0 + sz) {
    if (near1)      dragSrc = 0;
    else if (near2) dragSrc = 1;
  }
  if (mp && held && dragSrc >= 0) {
    const nx = Math.max(-RANGE, Math.min(RANGE, s2dx(mp.x)));
    const ny = Math.max(-RANGE, Math.min(RANGE, s2dy(mp.y)));
    if (dragSrc === 0) { S.p1x = nx; S.p1y = ny; }
    else               { S.p2x = nx; S.p2y = ny; }
  }
  if (released) dragSrc = -1;

  cvFov.style.cursor = (near1 || near2 || dragSrc >= 0) ? 'move' : 'default';
}

/* Clean image (high-density via ImageData) */
function draw_clean(muMax) {
  const cv = cvClean;
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth, cssH = cv.clientHeight;
  const cw = Math.round(cssW * dpr), ch = Math.round(cssH * dpr);
  if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
  const ctx = cv.getContext('2d');

  const img = ctx.createImageData(cw, ch);
  const inv = 1.0 / muMax;
  for (let cy = 0; cy < ch; cy++) {
    let i = (cy * PI_FINE / ch) | 0;
    if (i >= PI_FINE) i = PI_FINE - 1;
    const row = i * PI_FINE;
    for (let cx = 0; cx < cw; cx++) {
      let j = (cx * PI_FINE / cw) | 0;
      if (j >= PI_FINE) j = PI_FINE - 1;
      let v = muFine[row + j] * inv;
      if (cleanLog) v = logStretch(v);
      if (v < 0) v = 0; else if (v > 1) v = 1;
      const g = (v * 255) | 0;
      const idx = (cy * cw + cx) * 4;
      img.data[idx]     = g;
      img.data[idx + 1] = g;
      img.data[idx + 2] = g;
      img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);

  ctx.save();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.strokeStyle = getCSSColor('--border');
  ctx.strokeRect(0, 0, cssW, cssH);
  ctx.restore();
}

/* asinh-style log stretch (γ = 1000) — preserves both endpoints, brings up
 * faint structure such as Airy-like rings under noise. */
function logStretch(t) {
  const g = 1000;
  return Math.log1p(g * Math.max(0, t)) / Math.log1p(g);
}
let cleanLog = false, noisyLog = false;

/* Pixelated + noise */
function draw_noisy(muMax) {
  ensureNoise();
  const { ctx, w, h } = setupCanvas(cvNoisy);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = getCSSColor('--canvas-bg');
  ctx.fillRect(0, 0, w, h);
  const cell = w / PI_PIX;
  for (let pi = 0; pi < PI_PIX; pi++) {
    for (let pj = 0; pj < PI_PIX; pj++) {
      let v = (muPix[pi * PI_PIX + pj]
            +  S.sigma * noise[pi * PI_PIX + pj]) / muMax;
      if (noisyLog) v = logStretch(v);
      if (v < 0) v = 0; if (v > 1) v = 1;
      const g = (v * 255) | 0;
      ctx.fillStyle = `rgb(${g},${g},${g})`;
      ctx.fillRect(pj * cell, pi * cell, cell + 0.5, cell + 0.5);
    }
  }
  ctx.strokeStyle = getCSSColor('--border');
  ctx.strokeRect(0, 0, w, h);
}

/* Sliders */
function bindSlider(slider, valEl, key, decimals) {
  if (!slider || !valEl) return;
  slider.addEventListener('input', () => {
    S[key] = parseFloat(slider.value);
    valEl.textContent = S[key].toFixed(decimals);
  });
  valEl.textContent = S[key].toFixed(decimals);
}
bindSlider(document.getElementById('model-F1'),
           document.getElementById('model-F1-val'), 'F1', 2);
bindSlider(document.getElementById('model-F2'),
           document.getElementById('model-F2-val'), 'F2', 2);
bindSlider(document.getElementById('model-sigma'),
           document.getElementById('model-sigma-val'), 'sigma', 3);

/* "↻ new sample" button: redraw the cached Gaussian-noise pattern. */
const newNoiseBtn = document.getElementById('model-new-noise');
if (newNoiseBtn) {
  newNoiseBtn.addEventListener('click', () => {
    for (let i = 0; i < noise.length; i++) noise[i] = gauss();
    /* noiseInit stays true; the next loop tick picks up the new pattern. */
  });
}

/* Log-stretch toggles for the two intensity plots. */
const cleanLogBtn = document.getElementById('model-clean-log');
if (cleanLogBtn) cleanLogBtn.addEventListener('click', () => {
  cleanLog = !cleanLog; cleanLogBtn.classList.toggle('is-on', cleanLog);
});
const noisyLogBtn = document.getElementById('model-noisy-log');
if (noisyLogBtn) noisyLogBtn.addEventListener('click', () => {
  noisyLog = !noisyLog; noisyLogBtn.classList.toggle('is-on', noisyLog);
});

/* Main loop */
function loop() {
  const muMax = compute();
  draw_fov();
  draw_clean(muMax);
  draw_noisy(muMax);
  pressed = false; released = false;
  requestAnimationFrame(loop);
}
loop();

})();
