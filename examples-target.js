/* ===========================================================================
 * Mini-demo for the "Target confusion map" section.
 *
 *   Plot 1 (grayscale)  : mu_P, drag p1 (red), p2 (blue).
 *   Plot 2 (grayscale)  : pixelated mu_P + cached white noise * sigma.
 *                         "↻ new sample" button redraws the noise pattern.
 *   Plot 3              : D versus Delta = ||p2 - p1|| over [0, 5 sigma].
 *                         Curve is colour-segmented (red where D > 0,
 *                         green where D < 0); yellow vertical line at the
 *                         zero crossing Delta_c (with numerical readout);
 *                         white vertical line at the user's current Delta.
 *
 * Sliders: F_P1, F_P2, sigma_noise (sigma feeds plot 2 only — D is
 * independent of noise, which is the whole point of the section).
 * =========================================================================== */

(() => {

const cvP = document.getElementById('target-P');
const cvN = document.getElementById('target-noisy');
const cvD = document.getElementById('target-D');
if (!cvP || !cvN || !cvD) return;

const SIGMA_PSF  = 1.0;
const RANGE      = 3.0;
const PI_FINE    = 240;
const PI_PIX     = 20;
const DELTA_MAX  = 5.0;
const N_D        = 100;

const S = {
  P1x: -0.6, P1y: 0.0,
  P2x: +0.6, P2y: 0.0,
  FP1: 1.0,  FP2: 1.0,
  sigma: 0.10,
};

const phi2 = (dx, dy) =>
  Math.exp(-(dx*dx + dy*dy) / (2.0 * SIGMA_PSF * SIGMA_PSF));
const k1d  = (t) =>
  Math.exp(-t*t / (4.0 * SIGMA_PSF * SIGMA_PSF));

/* Buffers */
const muFine = new Float64Array(PI_FINE * PI_FINE);
const muPix  = new Float64Array(PI_PIX  * PI_PIX);
const noiseSamples = new Float64Array(PI_PIX * PI_PIX);
let noiseInit = false;

/* Log-stretch toggles for the two intensity plots. */
function logStretch(t) {
  const g = 1000;
  return Math.log1p(g * Math.max(0, t)) / Math.log1p(g);
}
let pLog = false, noisyLog = false;
let pMax = 1;

const Dvals = new Float64Array(N_D + 1);

/* ---- Compute mu_P + binned image ---- */
function compute_means() {
  pMax = 1e-9;
  for (let i = 0; i < PI_FINE; i++) {
    const y = +RANGE - (i + 0.5) / PI_FINE * 2*RANGE;
    for (let j = 0; j < PI_FINE; j++) {
      const x = -RANGE + (j + 0.5) / PI_FINE * 2*RANGE;
      const v = S.FP1 * phi2(x - S.P1x, y - S.P1y)
              + S.FP2 * phi2(x - S.P2x, y - S.P2y);
      muFine[i * PI_FINE + j] = v;
      if (v > pMax) pMax = v;
    }
  }
  const block = PI_FINE / PI_PIX;
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
}

/* ---- Cached Gaussian noise ---- */
function gauss() {
  const u1 = Math.random() + 1e-12;
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
function ensureNoise() {
  if (noiseInit) return;
  for (let i = 0; i < noiseSamples.length; i++) noiseSamples[i] = gauss();
  noiseInit = true;
}

/* ---- 1D matched-filter q* search along the segment of length Delta ---- */
function find_qstar(Delta) {
  const F1 = S.FP1, F2 = S.FP2;
  const SIG = SIGMA_PSF;
  const inv2s2 = 1.0 / (2 * SIG * SIG);

  /* Coarse grid: tight around the segment, dense enough that the seed for
   * Newton is good to ~0.02. */
  const lo = -2 * SIG;
  const hi = Delta + 2 * SIG;
  const N_GRID = 200;
  let bestQ = lo, bestF = -Infinity;
  for (let i = 0; i <= N_GRID; i++) {
    const q = lo + (hi - lo) * i / N_GRID;
    const f = F1 * k1d(q) + F2 * k1d(Delta - q);
    if (f > bestF) { bestF = f; bestQ = q; }
  }

  /* Newton on f'(q) = 0 with the CORRECT Hessian
   *   f''(q) = -F1/(2 sigma^2) * k(q)   * [1 - q^2 / (2 sigma^2)]
   *           - F2/(2 sigma^2) * k(D-q) * [1 - (D-q)^2 / (2 sigma^2)]
   * After each step we re-evaluate f and only accept if it improved —
   * this keeps the result monotone in the grid argmin's neighbourhood
   * even if Newton overshoots. */
  for (let it = 0; it < 12; it++) {
    const e1 = k1d(bestQ),  e2 = k1d(Delta - bestQ);
    const g  = -F1 * bestQ * e1 * inv2s2
             +  F2 * (Delta - bestQ) * e2 * inv2s2;
    const h  = -F1 * (1 - bestQ * bestQ * inv2s2)               * e1 * inv2s2
             -  F2 * (1 - (Delta - bestQ) * (Delta - bestQ) * inv2s2) * e2 * inv2s2;
    if (h >= -1e-12) break;
    let step = -g / h;
    if (step >  0.5) step =  0.5;
    if (step < -0.5) step = -0.5;
    const candidate = bestQ + step;
    const fNew = F1 * k1d(candidate) + F2 * k1d(Delta - candidate);
    if (fNew > bestF) {
      bestQ = candidate;
      bestF = fNew;
    } else {
      break;            /* Newton would make things worse; keep grid winner */
    }
    if (Math.abs(step) < 1e-9) break;
  }

  if (bestQ < 0) bestQ = 0;
  if (bestQ > Delta) bestQ = Delta;
  return bestQ;
}

/* ---- Sample normalised D-bar(Delta) ---- *
 * Stored values are
 *   Dbar = D / (2 J_1) + D / (2 J_2),
 * which is bounded in [-1, 1] for the smooth PSF dictionaries. */
function sample_D_curve() {
  const F1 = S.FP1, F2 = S.FP2;
  const J1 = Math.max(F1 * F1, 1e-12);
  const J2 = Math.max(F2 * F2, 1e-12);
  const norm = 1 / (2 * J1) + 1 / (2 * J2);
  for (let i = 0; i <= N_D; i++) {
    const Delta = i / N_D * DELTA_MAX;
    const q = find_qstar(Delta);
    const c1 = k1d(q), c2 = k1d(Delta - q), r = k1d(Delta);
    const Fs = F1 * c1 + F2 * c2;
    const muP_sq = F1*F1 + F2*F2 + 2*F1*F2*r;
    const JPQ = muP_sq - Fs*Fs;
    const Je  = Fs * Fs;
    const Jp1 = F1*F1 - 2*F1*Fs*c1 + Fs*Fs;
    const Jp2 = F2*F2 - 2*F2*Fs*c2 + Fs*Fs;
    const D   = Math.min(Je, Jp1, Jp2) - JPQ;
    Dvals[i] = D * norm;
  }
}

/* ---- Helpers ---- */
function getCSSColor(name) {
  return getComputedStyle(document.body).getPropertyValue(name).trim();
}

let mouseAbs = null, pressed = false, held = false, released = false;
let dragSrc = null;
document.addEventListener('mousemove', e => {
  mouseAbs = { x: e.clientX, y: e.clientY };
});
document.addEventListener('mousedown', () => { pressed = true; held = true; });
document.addEventListener('mouseup',   () => { released = true; held = false; });
function localMouse(cv) {
  if (!mouseAbs) return null;
  const r = cv.getBoundingClientRect();
  return { x: mouseAbs.x - r.left, y: mouseAbs.y - r.top };
}

/* ---- Plot 1: mu_P (grayscale, draggable sources) ---- */
function draw_P() {
  const cv = cvP;
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth, cssH = cv.clientHeight;
  const cw = Math.round(cssW * dpr), ch = Math.round(cssH * dpr);
  if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
  const ctx = cv.getContext('2d');

  const img = ctx.createImageData(cw, ch);
  const inv = 1.0 / pMax;
  for (let cy = 0; cy < ch; cy++) {
    let i = (cy * PI_FINE / ch) | 0; if (i >= PI_FINE) i = PI_FINE - 1;
    const row = i * PI_FINE;
    for (let cx = 0; cx < cw; cx++) {
      let j = (cx * PI_FINE / cw) | 0; if (j >= PI_FINE) j = PI_FINE - 1;
      let t = muFine[row + j] * inv;
      if (pLog) t = logStretch(t);
      if (t < 0) t = 0; else if (t > 1) t = 1;
      const g = (t * 255) | 0;
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

  const sz = cssW;
  const ppu = sz / (2 * RANGE);
  const RED  = 'rgb(220, 80, 80)';
  const BLUE = 'rgb(80, 130, 220)';
  const d2sx = (x) => (x + RANGE) / (2 * RANGE) * sz;
  const d2sy = (y) => sz - (y + RANGE) / (2 * RANGE) * sz;

  const p1sx = d2sx(S.P1x), p1sy = d2sy(S.P1y);
  const p2sx = d2sx(S.P2x), p2sy = d2sy(S.P2y);

  ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.globalAlpha = 0.55;
  ctx.strokeStyle = RED;
  ctx.beginPath(); ctx.arc(p1sx, p1sy, SIGMA_PSF * ppu, 0, Math.PI*2); ctx.stroke();
  ctx.strokeStyle = BLUE;
  ctx.beginPath(); ctx.arc(p2sx, p2sy, SIGMA_PSF * ppu, 0, Math.PI*2); ctx.stroke();
  ctx.setLineDash([]); ctx.globalAlpha = 1.0;

  const mp = localMouse(cv);
  let near = null;
  if (mp) {
    const d1 = (mp.x - p1sx)**2 + (mp.y - p1sy)**2;
    const d2 = (mp.x - p2sx)**2 + (mp.y - p2sy)**2;
    if (d1 < 121 && d1 <= d2) near = 'P1';
    else if (d2 < 121)        near = 'P2';
  }
  const r1 = (dragSrc === 'P1' || near === 'P1') ? 7 : 5;
  const r2 = (dragSrc === 'P2' || near === 'P2') ? 7 : 5;
  ctx.fillStyle = RED;
  ctx.beginPath(); ctx.arc(p1sx, p1sy, r1, 0, Math.PI*2); ctx.fill();
  ctx.fillStyle = BLUE;
  ctx.beginPath(); ctx.arc(p2sx, p2sy, r2, 0, Math.PI*2); ctx.fill();

  ctx.strokeStyle = getCSSColor('--border');
  ctx.strokeRect(0, 0, cssW, cssW);
  ctx.restore();

  if (mp && pressed && !dragSrc && near) dragSrc = near;
  if (mp && held && (dragSrc === 'P1' || dragSrc === 'P2')) {
    const sx = mp.x, sy = mp.y;
    const nx = Math.max(-RANGE, Math.min(RANGE,  sx / sz * 2*RANGE - RANGE));
    const ny = Math.max(-RANGE, Math.min(RANGE, -((sy / sz) * 2*RANGE - RANGE)));
    if (dragSrc === 'P1') { S.P1x = nx; S.P1y = ny; }
    else                  { S.P2x = nx; S.P2y = ny; }
  }
  cv.style.cursor = (near || dragSrc === 'P1' || dragSrc === 'P2') ? 'move' : 'default';
}

/* ---- Plot 2: pixelated noisy ---- */
function draw_noisy() {
  ensureNoise();
  const cv = cvN;
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth, cssH = cv.clientHeight;
  const cw = Math.round(cssW * dpr), ch = Math.round(cssH * dpr);
  if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.fillStyle = getCSSColor('--canvas-bg');
  ctx.fillRect(0, 0, cssW, cssH);

  const cell = cssW / PI_PIX;
  const inv = 1.0 / pMax;
  for (let pi = 0; pi < PI_PIX; pi++) {
    for (let pj = 0; pj < PI_PIX; pj++) {
      let v = (muPix[pi * PI_PIX + pj]
            +  S.sigma * noiseSamples[pi * PI_PIX + pj]) * inv;
      if (noisyLog) v = logStretch(v);
      if (v < 0) v = 0; if (v > 1) v = 1;
      const g = (v * 255) | 0;
      ctx.fillStyle = `rgb(${g},${g},${g})`;
      ctx.fillRect(pj * cell, pi * cell, cell + 0.5, cell + 0.5);
    }
  }
  ctx.strokeStyle = getCSSColor('--border');
  ctx.strokeRect(0, 0, cssW, cssH);
}

/* ---- Plot 3: D vs Delta ---- */
function draw_D() {
  const cv = cvD;
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth, cssH = cv.clientHeight;
  const cw = Math.round(cssW * dpr), ch = Math.round(cssH * dpr);
  if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.fillStyle = getCSSColor('--canvas-bg');
  ctx.fillRect(0, 0, cssW, cssH);

  const padL = 36, padR = 8, padT = 14, padB = 22;
  const px = padL, py = padT;
  const pw = cssW - padL - padR, ph = cssH - padT - padB;

  /* Y-range with margin and zero visible */
  let dmax = 0, dmin = 0;
  for (let i = 0; i <= N_D; i++) {
    if (Dvals[i] > dmax) dmax = Dvals[i];
    if (Dvals[i] < dmin) dmin = Dvals[i];
  }
  let ymax = dmax * 1.15, ymin = dmin * 1.15;
  if (ymax <  0.05) ymax =  0.05;
  if (ymin > -0.05) ymin = -0.05;

  /* Zero line */
  const zero_y = py + ph - (0 - ymin)/(ymax - ymin) * ph;
  ctx.strokeStyle = getCSSColor('--canvas-axis');
  ctx.beginPath(); ctx.moveTo(px, zero_y); ctx.lineTo(px+pw, zero_y); ctx.stroke();

  /* Border */
  ctx.strokeStyle = getCSSColor('--border');
  ctx.strokeRect(px, py, pw, ph);

  /* Find first sign change Delta_c */
  let dc = -1;
  for (let i = 1; i <= N_D; i++) {
    if (Dvals[i-1] > 0 && Dvals[i] <= 0) {
      const f = Dvals[i-1] / (Dvals[i-1] - Dvals[i]);
      dc = (i - 1) / N_D * DELTA_MAX
         + f * (1 / N_D * DELTA_MAX);
      break;
    }
  }

  /* Color-segmented curve */
  const colRed   = 'rgb(220, 80, 80)';
  const colGreen = 'rgb(80, 200, 110)';
  ctx.lineWidth = 2;
  for (let i = 1; i <= N_D; i++) {
    const Da = Dvals[i-1], Db = Dvals[i];
    const xa = px + ((i-1) / N_D) * pw;
    const xb = px + ( i    / N_D) * pw;
    const ya = py + ph - (Da - ymin)/(ymax - ymin) * ph;
    const yb = py + ph - (Db - ymin)/(ymax - ymin) * ph;
    ctx.strokeStyle = (0.5*(Da + Db) > 0) ? colRed : colGreen;
    ctx.beginPath(); ctx.moveTo(xa, ya); ctx.lineTo(xb, yb); ctx.stroke();
  }

  /* Vertical line at user's current Delta */
  const Delta_now = Math.sqrt((S.P2x - S.P1x)**2 + (S.P2y - S.P1y)**2);
  if (Delta_now >= 0 && Delta_now <= DELTA_MAX) {
    const xc = px + (Delta_now / DELTA_MAX) * pw;
    ctx.strokeStyle = getCSSColor('--text');
    ctx.globalAlpha = 0.7;
    ctx.beginPath(); ctx.moveTo(xc, py); ctx.lineTo(xc, py+ph); ctx.stroke();
    ctx.globalAlpha = 1.0;
  }

  /* Yellow line at Delta_c, with numerical readout */
  if (dc > 0) {
    const dx = px + (dc / DELTA_MAX) * pw;
    ctx.strokeStyle = 'rgb(245, 200, 70)';
    ctx.globalAlpha = 0.85;
    ctx.beginPath(); ctx.moveTo(dx, py); ctx.lineTo(dx, py+ph); ctx.stroke();
    ctx.globalAlpha = 1.0;
    ctx.fillStyle = 'rgb(245, 200, 70)';
    ctx.font = '10px monospace';
    ctx.fillText(`Δc=${dc.toFixed(3)}`, dx + 4, py + ph - 4);
  }

  /* Axis labels */
  ctx.fillStyle = getCSSColor('--text-soft');
  ctx.font = '10px monospace';
  ctx.fillText('0',                    px - 4,             py + ph + 14);
  ctx.fillText(DELTA_MAX.toFixed(0),   px + pw - 6,         py + ph + 14);
  ctx.fillText(ymax.toFixed(2),        2,                  py + 8);
  ctx.fillText(ymin.toFixed(2),        2,                  py + ph - 4);
  ctx.fillText('0',                    18,                 zero_y - 2);
  ctx.fillText('Δ',                    px + pw / 2,        py + ph + 14);
  ctx.fillText('D̄',                    4,                  py + ph / 2);
}

/* ---- Sliders ---- */
function bindSlider(slider, valEl, key, decimals) {
  if (!slider || !valEl) return;
  slider.addEventListener('input', () => {
    S[key] = parseFloat(slider.value);
    valEl.textContent = S[key].toFixed(decimals);
  });
  valEl.textContent = S[key].toFixed(decimals);
}
bindSlider(document.getElementById('target-FP1'),
           document.getElementById('target-FP1-val'), 'FP1', 2);
bindSlider(document.getElementById('target-FP2'),
           document.getElementById('target-FP2-val'), 'FP2', 2);
bindSlider(document.getElementById('target-sigma'),
           document.getElementById('target-sigma-val'), 'sigma', 3);

const newNoiseBtn = document.getElementById('target-new-noise');
if (newNoiseBtn) {
  newNoiseBtn.addEventListener('click', () => {
    for (let i = 0; i < noiseSamples.length; i++) noiseSamples[i] = gauss();
  });
}

const pLogBtn = document.getElementById('target-P-log');
if (pLogBtn) pLogBtn.addEventListener('click', () => {
  pLog = !pLog; pLogBtn.classList.toggle('is-on', pLog);
});
const nLogBtn = document.getElementById('target-noisy-log');
if (nLogBtn) nLogBtn.addEventListener('click', () => {
  noisyLog = !noisyLog; nLogBtn.classList.toggle('is-on', noisyLog);
});

/* ---- Verdict box: sign of D-bar at the user's current Delta ---- */
const verdictEl = document.getElementById('target-verdict');
const verdictText = verdictEl ? verdictEl.querySelector('.verdict-text') : null;

function update_verdict() {
  if (!verdictEl || !verdictText) return;
  const Delta_now = Math.hypot(S.P2x - S.P1x, S.P2y - S.P1y);
  /* Off-axis (Delta > DELTA_MAX) is always clearly resolved. */
  let Dbar_now;
  if (Delta_now >= DELTA_MAX) {
    Dbar_now = Dvals[N_D];
  } else {
    const t = Delta_now / DELTA_MAX * N_D;
    const i0 = Math.min(N_D - 1, Math.floor(t));
    const a  = t - i0;
    Dbar_now = (1 - a) * Dvals[i0] + a * Dvals[i0 + 1];
  }
  const EPS = 1e-4;
  if (Dbar_now > EPS) {
    verdictEl.classList.remove('is-clean');
    verdictEl.classList.add('is-contam');
    verdictText.textContent =
      'Contamination is geometrically possible — for low enough noise the data prefers a phantom Q⋆.';
  } else if (Dbar_now < -EPS) {
    verdictEl.classList.remove('is-contam');
    verdictEl.classList.add('is-clean');
    verdictText.textContent =
      'No contamination at any noise level — every simpler hypothesis beats the phantom Q⋆.';
  } else {
    verdictEl.classList.remove('is-contam');
    verdictEl.classList.remove('is-clean');
    verdictText.textContent =
      'On the boundary D̄ ≈ 0 — the critical separation Δc.';
  }
}

/* ---- Main loop ---- */
function loop() {
  compute_means();
  sample_D_curve();
  draw_P();
  draw_noisy();
  draw_D();
  update_verdict();
  if (released) dragSrc = null;
  pressed = false; released = false;
  requestAnimationFrame(loop);
}
loop();

})();
