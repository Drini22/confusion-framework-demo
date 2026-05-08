/* ===========================================================================
 * Mini-demo for the "Three conditions for contamination" section.
 *
 *   Plot 1 (grayscale)    : mu_P, drag p1 (red), p2 (blue).
 *   Plot 2 (grayscale)    : pixelated mu_P + cached white-noise scaled by sigma.
 *   Plot 3 (regions)      : (Delta, R = SNR_P / eta) classification in the
 *                           same colour scheme as the gaussian-2d demo:
 *                             GREY  = no detection  (R < R_detect)
 *                             BLUE  = single real source p_1 or p_2
 *                             RED   = phantom Q*  (contamination)
 *                             GREEN = full P = {p_1, p_2}  (resolved)
 *                           Yellow dot at the user's current (Delta, R).
 *
 * Sliders: F_P1, F_P2, sigma_noise.
 * Convention: eta_SNR is the look-elsewhere-corrected 5σ threshold computed
 * by examples-aic.js for this same 2D-Gaussian setup (≈ 4.9, exposed via
 * window.AIC_ETA_2D_GAUSSIAN; falls back to 5 if the AIC demo isn't loaded).
 * The boundary curves Rd/Rr/Rc are dimensionless ratios in units of
 * SNR(P)/eta — they're eta-invariant; eta only sets where the dot sits on
 * the y-axis.
 * =========================================================================== */

(() => {

const cvP = document.getElementById('contam-P');
const cvN = document.getElementById('contam-noisy');
const cvC = document.getElementById('contam-class');
if (!cvP || !cvN || !cvC) return;

const SIGMA_PSF  = 1.0;
const RANGE      = 3.0;
const PI_FINE    = 240;
const PI_PIX     = 20;
const SNR_THRESH = (typeof window !== 'undefined' && window.AIC_ETA_2D_GAUSSIAN)
                 ? window.AIC_ETA_2D_GAUSSIAN : 5.0;
const DELTA_MAX  = 5.0;
const N_CLASS    = 100;

const S = {
  P1x: -0.6, P1y: 0.0,
  P2x: +0.6, P2y: 0.0,
  FP1: 1.0, FP2: 1.0,
  sigma: 0.10,
};

const phi2 = (dx, dy) =>
  Math.exp(-(dx*dx + dy*dy) / (2.0 * SIGMA_PSF * SIGMA_PSF));
const k1d  = (t) =>
  Math.exp(-t*t / (4.0 * SIGMA_PSF * SIGMA_PSF));      /* unit-norm autocorr */

/* ---- Buffers ---- */
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

const Rd = new Float64Array(N_CLASS + 1);
const Rr = new Float64Array(N_CLASS + 1);
const Rc = new Float64Array(N_CLASS + 1);

/* ---- Compute mu_P and binned image ---- */
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

/* ---- Gaussian noise (Box-Muller), drawn once and reused ---- */
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

/* ---- Find q* on the segment for a given Delta (1D matched filter) ----
 * Coarse grid + Newton with the CORRECT Hessian and per-step f-improvement
 * check.  Inner curvature term is (1 - q^2/(2 sigma^2)), NOT (1 - q^2/sigma^2). */
function find_qstar(Delta) {
  const F1 = S.FP1, F2 = S.FP2;
  const SIG = SIGMA_PSF;
  const inv2s2 = 1.0 / (2 * SIG * SIG);

  /* Tight grid seed (~0.02 spacing) */
  const lo = -2 * SIG;
  const hi = Delta + 2 * SIG;
  const N_GRID = 200;
  let bestQ = lo, bestF = -Infinity;
  for (let i = 0; i <= N_GRID; i++) {
    const q = lo + (hi - lo) * i / N_GRID;
    const f = F1 * k1d(q) + F2 * k1d(Delta - q);
    if (f > bestF) { bestF = f; bestQ = q; }
  }

  /* Newton with f-tracking */
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
      break;
    }
    if (Math.abs(step) < 1e-9) break;
  }

  if (bestQ < 0) bestQ = 0;
  if (bestQ > Delta) bestQ = Delta;
  return bestQ;
}

/* ---- Sample boundary R-values across Delta ---- */
function sample_R_thresholds() {
  const F1 = S.FP1, F2 = S.FP2;
  for (let i = 0; i <= N_CLASS; i++) {
    const Delta = i / N_CLASS * DELTA_MAX;
    const q = find_qstar(Delta);
    const c1 = k1d(q),  c2 = k1d(Delta - q),  r = k1d(Delta);
    const Fs = F1 * c1 + F2 * c2;
    const muP_sq = F1*F1 + F2*F2 + 2*F1*F2*r;
    const JPQ    = muP_sq - Fs*Fs;
    const Je     = Fs*Fs;
    const Jp1    = F1*F1 - 2*F1*Fs*c1 + Fs*Fs;
    const Jp2    = F2*F2 - 2*F2*Fs*c2 + Fs*Fs;
    const minPs  = Math.min(Je, Jp1, Jp2);
    const denom_d = Math.max(muP_sq - JPQ, 1e-12);
    const denom_r = Math.max(JPQ,           1e-12);
    const denom_c = Math.max(minPs,         1e-12);
    Rd[i] = Math.sqrt(muP_sq / denom_d);
    Rr[i] = Math.sqrt(muP_sq / denom_r);
    Rc[i] = Math.sqrt(muP_sq / denom_c);
  }
}

/* ---- Canvas helpers ---- */
function getCSSColor(name) {
  return getComputedStyle(document.body).getPropertyValue(name).trim();
}

let mouseAbs = null, pressed = false, held = false, released = false;
let dragSrc = null;          /* 'P1' | 'P2' | 'CLASS' | null */
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

/* ---- Plot 1: mu_P grayscale, drag p1/p2 ---- */
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

/* ---- Plot 3: region classification ---- */
function draw_classification() {
  const cv = cvC;
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth, cssH = cv.clientHeight;
  const cw = Math.round(cssW * dpr), ch = Math.round(cssH * dpr);
  if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = getCSSColor('--canvas-bg');
  ctx.fillRect(0, 0, cssW, cssH);

  const padL = 32, padR = 8, padT = 14, padB = 22;
  const px = padL, py = padT;
  const pw = cssW - padL - padR, ph = cssH - padT - padB;

  const R_min = 0.4, R_max = 100;
  const lr_min = Math.log10(R_min), lr_max = Math.log10(R_max);

  const colGrey  = 'rgb( 80, 84, 96)';
  const colBlue  = 'rgb( 70,110,180)';
  const colRed   = 'rgb(200, 80, 80)';
  const colGreen = 'rgb( 80,180,100)';

  ctx.globalAlpha = 0.85;
  for (let xp = px; xp < px + pw; xp++) {
    const dx = (xp - px) / pw * DELTA_MAX;
    const idxF = (dx / DELTA_MAX) * N_CLASS;
    let i_lo = Math.floor(idxF); if (i_lo < 0) i_lo = 0; if (i_lo > N_CLASS - 1) i_lo = N_CLASS - 1;
    const frac = idxF - i_lo;
    const Rdp = Rd[i_lo]*(1-frac) + Rd[i_lo+1]*frac;
    const Rrp = Rr[i_lo]*(1-frac) + Rr[i_lo+1]*frac;
    const Rcp = Rc[i_lo]*(1-frac) + Rc[i_lo+1]*frac;
    function R2y(R) {
      let lr = Math.log10(R);
      if (lr < lr_min) lr = lr_min;
      if (lr > lr_max) lr = lr_max;
      return py + ph - (lr - lr_min)/(lr_max - lr_min) * ph;
    }
    const yd = R2y(Rdp), yr = R2y(Rrp), yc = R2y(Rcp);
    const yBot = py + ph, yTop = py;

    ctx.fillStyle = colGrey;  ctx.fillRect(xp, yd, 1, yBot - yd);
    if (Rcp < Rrp) {
      ctx.fillStyle = colBlue;  ctx.fillRect(xp, yc, 1, yd - yc);
      ctx.fillStyle = colRed;   ctx.fillRect(xp, yr, 1, yc - yr);
    } else {
      ctx.fillStyle = colBlue;  ctx.fillRect(xp, yr, 1, yd - yr);
    }
    ctx.fillStyle = colGreen; ctx.fillRect(xp, yTop, 1, yr - yTop);
  }
  ctx.globalAlpha = 1.0;

  /* Border */
  ctx.strokeStyle = getCSSColor('--border');
  ctx.strokeRect(px, py, pw, ph);

  /* Yellow dot at user's (Delta, R) — draggable.
   * Horizontal drag rescales (p1, p2) symmetrically about their midpoint
   * to produce the new Delta; vertical drag rescales sigma so R hits the
   * target log-y value. */
  const Delta = Math.sqrt((S.P2x - S.P1x)**2 + (S.P2y - S.P1y)**2);
  const r_now = k1d(Delta);
  const muP_sq_now = S.FP1*S.FP1 + S.FP2*S.FP2 + 2*S.FP1*S.FP2*r_now;
  /* SNR := sqrt(J(P,∅,Σ)) = sqrt(muP_sq) / σ_F̂ where σ_F̂ is the matched-
   * filter retrieval std (paper convention).  In a per-pixel-noise demo
   * with N_pix-side cell size dx_pix, σ_F̂ = σ_pix · dx_pix.  Earlier
   * versions of this plot dropped the dx_pix factor and underplotted R
   * by that amount. */
  const dx_pix = 2 * RANGE / PI_PIX;
  const safeSig = Math.max(S.sigma * dx_pix, 1e-9);
  const R_now = Math.sqrt(muP_sq_now) / (safeSig * SNR_THRESH);

  let xc = -1, yc = -1, near = false;
  if (Delta >= 0 && Delta <= DELTA_MAX) {
    xc = px + (Delta / DELTA_MAX) * pw;
    let lr = Math.log10(R_now);
    if (lr < lr_min) lr = lr_min;
    if (lr > lr_max) lr = lr_max;
    yc = py + ph - (lr - lr_min)/(lr_max - lr_min) * ph;

    const mp = localMouse(cv);
    if (mp) {
      const d2 = (mp.x - xc)**2 + (mp.y - yc)**2;
      near = d2 < 196;   /* 14 px hit radius */
    }
    const rOuter = (dragSrc === 'CLASS' || near) ? 10 : 7;
    const rInner = (dragSrc === 'CLASS' || near) ?  6 : 5;
    ctx.fillStyle = 'rgb(245, 200, 70)';
    ctx.beginPath(); ctx.arc(xc, yc, rInner, 0, Math.PI*2); ctx.fill();
    ctx.strokeStyle = 'rgb(245, 200, 70)';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(xc, yc, rOuter, 0, Math.PI*2); ctx.stroke();
  }

  /* Drag start */
  const mpcp = localMouse(cv);
  if (mpcp && pressed && !dragSrc && near) dragSrc = 'CLASS';

  /* Drag updates */
  if (mpcp && held && dragSrc === 'CLASS') {
    let mx = Math.max(px, Math.min(px + pw, mpcp.x));
    let my = Math.max(py, Math.min(py + ph, mpcp.y));

    const newDelta = (mx - px) / pw * DELTA_MAX;
    const newLr    = (py + ph - my) / ph * (lr_max - lr_min) + lr_min;
    const newR     = Math.pow(10, newLr);

    /* Rescale (p1, p2) symmetrically about their midpoint to give newDelta */
    const dxp = S.P2x - S.P1x, dyp = S.P2y - S.P1y;
    const len = Math.hypot(dxp, dyp);
    let ux = 1, uy = 0;
    if (len > 1e-9) { ux = dxp / len; uy = dyp / len; }
    const cxm = 0.5 * (S.P1x + S.P2x);
    const cym = 0.5 * (S.P1y + S.P2y);
    S.P1x = Math.max(-RANGE, Math.min(RANGE, cxm - 0.5 * newDelta * ux));
    S.P1y = Math.max(-RANGE, Math.min(RANGE, cym - 0.5 * newDelta * uy));
    S.P2x = Math.max(-RANGE, Math.min(RANGE, cxm + 0.5 * newDelta * ux));
    S.P2y = Math.max(-RANGE, Math.min(RANGE, cym + 0.5 * newDelta * uy));

    /* Set sigma so R lands on the target log-y.  Inverting
     *   R = sqrt(muP_sq) / (σ_pix · dx_pix · SNR_THRESH)
     * for σ_pix. */
    const new_kernel = k1d(newDelta);
    const muP_sq_new = S.FP1*S.FP1 + S.FP2*S.FP2 + 2*S.FP1*S.FP2*new_kernel;
    const dx_pix = 2 * RANGE / PI_PIX;
    let new_sigma = Math.sqrt(muP_sq_new)
                  / (Math.max(newR, 1e-6) * SNR_THRESH * dx_pix);
    new_sigma = Math.max(0.001, Math.min(4.0, new_sigma));
    S.sigma = new_sigma;

    /* Sync slider + label for sigma */
    const sigSlider = document.getElementById('contam-sigma');
    const sigLabel  = document.getElementById('contam-sigma-val');
    if (sigSlider) sigSlider.value = S.sigma;
    if (sigLabel)  sigLabel.textContent = S.sigma.toFixed(3);
  }

  cv.style.cursor = (near || dragSrc === 'CLASS') ? 'move' : 'default';

  /* Axes labels (R on log scale, [R_min, R_max]). */
  ctx.fillStyle = getCSSColor('--text-soft');
  ctx.font = '10px monospace';
  ctx.fillText('100', 4, py + 8);
  const y10 = py + ph - (1 - lr_min) / (lr_max - lr_min) * ph;
  ctx.fillText('10',  6, y10 - 2);
  const y1  = py + ph - (0 - lr_min) / (lr_max - lr_min) * ph;
  ctx.fillText('1',   8, y1 - 2);
  ctx.fillText('0.4', 4, py + ph - 2);
  ctx.fillText('0',   px - 4, py + ph + 14);
  ctx.fillText(DELTA_MAX.toFixed(0), px + pw - 6, py + ph + 14);
  ctx.fillText('R',   4, py + ph / 2);
  ctx.fillText('Δ',   px + pw / 2, py + ph + 14);
}

/* ---- Slider wiring ---- */
function bindSlider(slider, valEl, key, decimals) {
  if (!slider || !valEl) return;
  slider.addEventListener('input', () => {
    S[key] = parseFloat(slider.value);
    valEl.textContent = S[key].toFixed(decimals);
  });
  valEl.textContent = S[key].toFixed(decimals);
}
/* Plot label: show the SNR_THRESH that was actually used (the
 * look-elsewhere η from the AIC demo). */
const contamEtaEl = document.getElementById('contam-class-eta');
if (contamEtaEl) contamEtaEl.textContent = SNR_THRESH.toFixed(2);

bindSlider(document.getElementById('contam-FP1'),
           document.getElementById('contam-FP1-val'), 'FP1', 2);
bindSlider(document.getElementById('contam-FP2'),
           document.getElementById('contam-FP2-val'), 'FP2', 2);
bindSlider(document.getElementById('contam-sigma'),
           document.getElementById('contam-sigma-val'), 'sigma', 3);

const newNoiseBtn = document.getElementById('contam-new-noise');
if (newNoiseBtn) {
  newNoiseBtn.addEventListener('click', () => {
    for (let i = 0; i < noiseSamples.length; i++) noiseSamples[i] = gauss();
  });
}

const pLogBtn = document.getElementById('contam-P-log');
if (pLogBtn) pLogBtn.addEventListener('click', () => {
  pLog = !pLog; pLogBtn.classList.toggle('is-on', pLog);
});
const nLogBtn = document.getElementById('contam-noisy-log');
if (nLogBtn) nLogBtn.addEventListener('click', () => {
  noisyLog = !noisyLog; nLogBtn.classList.toggle('is-on', noisyLog);
});

/* ---- Main loop ---- */
function loop() {
  compute_means();
  sample_R_thresholds();
  draw_P();
  draw_noisy();
  draw_classification();
  if (released) dragSrc = null;
  pressed = false; released = false;
  requestAnimationFrame(loop);
}
loop();

})();
