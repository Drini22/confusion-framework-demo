/* ===========================================================================
 * Interactive SAR demo — sar.html.
 *
 * Adapts the framework's machinery (matched-filter Q*, three contamination
 * conditions, sign test D̄, δ_1 = √(area/π)) to a separable SAR impulse
 * response  U_p(r, x) = h_r(r - r0) · h_x(x - x0),  with the range PSF
 * h_r determined by the chosen transmit waveform and the cross-range PSF
 * h_x determined by the synthetic-aperture length.
 *
 * All quantities in **resolution cells** so δ_1 reads as a dimensionless
 * number directly comparable to the optical δ_1 ≈ 1.16 λ/d (= 0.952 δ_R).
 *
 * Self-contained: no dependency on apertures.js.
 * =========================================================================== */

(() => {

if (typeof FFT2D === 'undefined') {
  console.error('FFT2D not loaded — include fft.js before examples-sar.js');
  return;
}

/* ---- Geometry constants (all positions in resolution cells) ----
 * FOV is ±FOV cells along both range and cross-range.  PSF tabulated
 * on a 2D grid wide enough to support any drag inside the FOV.
 */
const RAYLEIGH = 1.22;                 /* keep the same δ_R reference */
const FOV      = 2.5;                  /* ±2.5 cells half-width */
const PI_FINE  = 200;
const PI_PIX   = 20;
const N_DGRID  = 100;
const DELTA_MAX = 4.0;                 /* x-axis max for plots 6 & 9 (cells) */
const N_DELTA  = 220;
const N_RATIO  = 120;
const N_CLASS  = 100;

/* PSF lookup table parameters (separable: 1D in each axis). */
const U_HALF = 64;                     /* ±64 cells worth of samples */
const U_SIZE = 2 * U_HALF + 1;
const U_DX   = 4.0 / 64;               /* 0.0625 cells per sample */

const K_HALF = 80;
const K_SIZE = 2 * K_HALF + 1;

const S = {
  P1r: -0.60, P1x: 0.0,
  P2r: +0.60, P2x: 0.0,
  A1: 0.7,  A2: 1.0,
  sigma: 0.10,
  Lsyn: 1.0,
};

/* ---- Canvases ----------------------------------------------------------- */
const cv = {
  wf_time: document.getElementById('sar-wf-time'),
  wf_spec: document.getElementById('sar-wf-spec'),
  wf_psf:  document.getElementById('sar-wf-psf'),
  az_ap:   document.getElementById('sar-az-aperture'),
  az_psf:  document.getElementById('sar-az-psf'),
  az_2d:   document.getElementById('sar-az-2dpsf'),
  /* inline section-1 demo: drag two targets, see μ_P + noisy */
  model_fov:   document.getElementById('sar-model-fov'),
  model_clean: document.getElementById('sar-model-clean'),
  model_noisy: document.getElementById('sar-model-noisy'),
  /* inline section-5 demo: pair demo */
  pair_fov:   document.getElementById('sar-pair-fov'),
  pair_clean: document.getElementById('sar-pair-clean'),
  pair_noisy: document.getElementById('sar-pair-noisy'),
  /* big demo */
  fov:     document.getElementById('sar-fov'),
  psf2d:   document.getElementById('sar-psf2d'),
  noisy:   document.getElementById('sar-noisy'),
  P:       document.getElementById('sar-P'),
  Dmap:    document.getElementById('sar-Dmap'),
  Dratio:  document.getElementById('sar-Dratio'),
  best:    document.getElementById('sar-best'),
  flux:    document.getElementById('sar-flux'),
  class:   document.getElementById('sar-class'),
};

/* ---- Waveform library --------------------------------------------------- *
 * Each waveform's range PSF h_r(τ) is computed in CELL units via the
 * waveform's autocorrelation, normalised to peak = 1.  The cell width
 * is set so the FWHM (Rayleigh-style) lands at τ = 1.
 *
 * - rect:   h_r(τ) = triangular envelope, FWHM ≈ 1 cell
 * - lfm:    after pulse compression, h_r(τ) ≈ sinc(B·τ_phys), so in cell
 *           units h_r(τ) ≈ sinc(τ).  Same FWHM, ringing sidelobes.
 * - nlfm:   sinc with a Hamming window in frequency → smoother sidelobes,
 *           slightly wider main lobe.
 * - costas: thumbtack-ish — approximated as a tighter sinc² to evoke the
 *           low-sidelobe ideal (not the actual Costas ambiguity function,
 *           which is 2D).
 */
const WAVEFORM_LIB = {
  rect: {
    name: 'rect pulse',
    /* triangular: h_r(τ) = max(0, 1 - |τ|) */
    h_range: (tau) => Math.max(0, 1 - Math.abs(tau)),
    /* time-domain waveform (for the wf_time plot, dimensionless t/T) */
    wf_time: (t) => (Math.abs(t) < 0.5) ? 1 : 0,
    /* spectrum (for wf_spec plot, dimensionless f·T) */
    wf_spec: (f) => { const x = Math.PI * f; return x === 0 ? 1 : (Math.sin(x)/x)**2; },
  },
  lfm: {
    name: 'LFM chirp',
    /* sinc after pulse compression */
    h_range: (tau) => {
      const x = Math.PI * tau;
      if (Math.abs(x) < 1e-6) return 1.0;
      return Math.abs(Math.sin(x) / x);
    },
    wf_time: (t) => (Math.abs(t) < 0.5) ? 1 : 0,
    wf_spec: (f) => (Math.abs(f) < 10) ? 1 : 0,    /* flat bandwidth */
  },
  nlfm: {
    name: 'NLFM (Taylor)',
    /* Taylor-window-tapered sinc: roughly sinc² with broader main lobe */
    h_range: (tau) => {
      const x = Math.PI * tau * 0.88;        /* slight main-lobe broadening */
      const s = (Math.abs(x) < 1e-6) ? 1.0 : Math.sin(x) / x;
      return s * s;
    },
    wf_time: (t) => (Math.abs(t) < 0.5) ? Math.cos(Math.PI * t) ** 2 : 0,
    wf_spec: (f) => {
      if (Math.abs(f) > 12) return 0;
      const w = 0.54 + 0.46 * Math.cos(Math.PI * f / 12);   /* Hamming */
      return w * w;
    },
  },
  costas: {
    name: 'Costas-like',
    /* thumbtack ideal: very narrow main lobe, ~0 sidelobes */
    h_range: (tau) => {
      const x = Math.PI * tau * 1.15;
      const s = (Math.abs(x) < 1e-6) ? 1.0 : Math.sin(x) / x;
      return Math.max(0, s * s * (1 - 0.85 * Math.min(1, Math.abs(tau)/3)));
    },
    wf_time: (t) => (Math.abs(t) < 0.5) ? (0.5 + 0.5*Math.cos(8*Math.PI*t)) : 0,
    wf_spec: (f) => (Math.abs(f) < 8) ? (0.5 + 0.5*Math.cos(Math.PI*f/8)) : 0,
  },
};
let currentWaveform = 'rect';

/* ---- 1D PSF tables (range + cross-range) ------------------------------- */
const range_table = new Float64Array(U_SIZE);   /* h_r sampled at U_DX cells */
const cross_table = new Float64Array(U_SIZE);   /* h_x */
const k_range = new Float64Array(K_SIZE);       /* autocorrelation of h_r */
const k_cross = new Float64Array(K_SIZE);       /* autocorrelation of h_x */

function rebuild_range_psf() {
  const wf = WAVEFORM_LIB[currentWaveform];
  for (let i = 0; i < U_SIZE; i++) {
    const tau = (i - U_HALF) * U_DX;
    range_table[i] = wf.h_range(tau);
  }
  /* autocorrelation: k(τ) = ∫ h_r(s) h_r(s+τ) ds, normalised k(0)=1 */
  const k_tmp = new Float64Array(K_SIZE);
  for (let dr = -K_HALF; dr <= K_HALF; dr++) {
    const tau = dr * U_DX;
    let s = 0;
    for (let i = 0; i < U_SIZE; i++) {
      const sI = (i - U_HALF) * U_DX;
      const t2 = sI + tau;
      const j  = Math.round(t2 / U_DX) + U_HALF;
      if (j >= 0 && j < U_SIZE) s += range_table[i] * range_table[j];
    }
    k_tmp[dr + K_HALF] = s;
  }
  const k0 = k_tmp[K_HALF] || 1;
  for (let i = 0; i < K_SIZE; i++) k_range[i] = k_tmp[i] / k0;
}

function rebuild_cross_psf() {
  /* h_x(x) = sinc(2 L_syn · x / (λ r_0))  --  in normalised "cells",
   * the natural unit is δ_x at L_syn = 1, so h_x(x; L) = sinc(L · x).
   * Larger L → narrower PSF → larger δ_x in cells.  In the demo we
   * keep the SHAPE at L=1 (so cells track the chosen aperture) and
   * only change the visual width on inline plots — the geometry of
   * the framework is invariant.  L just scales the physical δ_x. */
  for (let i = 0; i < U_SIZE; i++) {
    const x = (i - U_HALF) * U_DX;
    const px = Math.PI * x;
    cross_table[i] = (Math.abs(px) < 1e-6) ? 1.0 : Math.sin(px) / px;
  }
  const k_tmp = new Float64Array(K_SIZE);
  for (let dr = -K_HALF; dr <= K_HALF; dr++) {
    const tau = dr * U_DX;
    let s = 0;
    for (let i = 0; i < U_SIZE; i++) {
      const sI = (i - U_HALF) * U_DX;
      const t2 = sI + tau;
      const j  = Math.round(t2 / U_DX) + U_HALF;
      if (j >= 0 && j < U_SIZE) s += cross_table[i] * cross_table[j];
    }
    k_tmp[dr + K_HALF] = s;
  }
  const k0 = k_tmp[K_HALF] || 1;
  for (let i = 0; i < K_SIZE; i++) k_cross[i] = k_tmp[i] / k0;
}

function U_at(dr, dx) {
  /* Bilinear from 1D tables: U_p(r,x) = h_r(r) · h_x(x). */
  const fr = dr / U_DX + U_HALF;
  const fx = dx / U_DX + U_HALF;
  if (fr < 0 || fr >= U_SIZE-1 || fx < 0 || fx >= U_SIZE-1) return 0;
  const ir = Math.floor(fr), ix = Math.floor(fx);
  const ar = fr - ir, ax = fx - ix;
  const hr = (1-ar) * range_table[ir] + ar * range_table[ir+1];
  const hx = (1-ax) * cross_table[ix] + ax * cross_table[ix+1];
  return hr * hx;
}

function k_at(dr, dx) {
  /* k(τ) = (U_p ⋆ U_p)(τ) = k_range(dr) · k_cross(dx) (separable). */
  function lookup(dr, table) {
    const f = dr / U_DX + K_HALF;
    if (f < 0 || f >= K_SIZE - 1) return 0;
    const i = Math.floor(f), a = f - i;
    return (1-a) * table[i] + a * table[i+1];
  }
  return lookup(dr, k_range) * lookup(dx, k_cross);
}

/* ---- Initialize PSF tables --------------------------------------------- */
rebuild_range_psf();
rebuild_cross_psf();

/* ---- Per-waveform η_SNR (look-elsewhere-corrected) --------------------- */
const ETA_FOV  = 2.5;
const ETA_N    = 16;          /* 16x16 search grid for MC */
const ETA_DX   = 2 * ETA_FOV / ETA_N;
const ETA_NMC  = 600;         /* fewer MC samples to keep startup snappy */

function gauss_rng() {
  const u1 = Math.random() + 1e-12, u2 = Math.random();
  return Math.sqrt(-2*Math.log(u1)) * Math.cos(2*Math.PI*u2);
}
let etaVal = 5.0;
function compute_eta() {
  /* Pre-tabulate matched-filter response m_q on ETA_N × ETA_N grid. */
  const m_table = [];
  const m_norm  = new Float64Array(ETA_N * ETA_N);
  for (let qi = 0; qi < ETA_N; qi++) {
    const qr = -ETA_FOV + (qi + 0.5) * ETA_DX;
    for (let qj = 0; qj < ETA_N; qj++) {
      const qx = -ETA_FOV + (qj + 0.5) * ETA_DX;
      const m = new Float64Array(ETA_N * ETA_N);
      let nrm = 0;
      for (let i = 0; i < ETA_N; i++) {
        const y = -ETA_FOV + (i + 0.5) * ETA_DX;
        for (let j = 0; j < ETA_N; j++) {
          const x = -ETA_FOV + (j + 0.5) * ETA_DX;
          const v = U_at(y - qr, x - qx);
          m[i * ETA_N + j] = v;
          nrm += v * v;
        }
      }
      m_table.push(m);
      m_norm[qi * ETA_N + qj] = nrm;
    }
  }
  const X = new Float64Array(ETA_N * ETA_N);
  let sum = 0, sum2 = 0;
  for (let s = 0; s < ETA_NMC; s++) {
    for (let p = 0; p < X.length; p++) X[p] = gauss_rng();
    let best = 0;
    for (let q = 0; q < m_table.length; q++) {
      const mq = m_table[q];
      const nrm = Math.sqrt(m_norm[q] || 1e-12);
      let b = 0;
      for (let p = 0; p < X.length; p++) b += mq[p] * X[p];
      if (b <= 0) continue;
      const snr = b / nrm;
      if (snr > best) best = snr;
    }
    sum += best; sum2 += best*best;
  }
  const mean = sum / ETA_NMC;
  const v = Math.max(sum2/ETA_NMC - mean*mean, 0);
  const std = Math.sqrt(v);
  etaVal = mean + 5 * std;
}
compute_eta();

/* ---- Color helpers ----------------------------------------------------- */
function getCSSColor(name) {
  return getComputedStyle(document.body).getPropertyValue(name).trim();
}
function inferno_rgb(t) {
  /* compact inferno-ish colormap */
  if (t < 0) t = 0; else if (t > 1) t = 1;
  const r = Math.min(255, 255 * Math.pow(t, 0.5));
  const g = Math.min(255, 255 * Math.pow(t, 1.8));
  const b = Math.min(255, 255 * (0.2 + 0.6 * Math.pow(1 - t, 2)));
  return [r|0, g|0, b|0];
}

/* ---- Mini-demos: range PSF + waveform plots --------------------------- */
function draw_curve(canvas, fn, xRange, yRange, color, opts = {}) {
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth, cssH = canvas.clientHeight;
  const cw = Math.round(cssW * dpr), ch = Math.round(cssH * dpr);
  if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = getCSSColor('--canvas-bg');
  ctx.fillRect(0, 0, cssW, cssH);

  const padL = 32, padR = 8, padT = 10, padB = 22;
  const px = padL, py = padT;
  const pw = cssW - padL - padR;
  const ph = cssH - padT - padB;

  const [x0, x1] = xRange, [y0, y1] = yRange;
  const x2px = x => px + (x - x0)/(x1 - x0) * pw;
  const y2px = y => py + ph - (y - y0)/(y1 - y0) * ph;

  /* zero line */
  if (y0 < 0 && y1 > 0) {
    ctx.strokeStyle = getCSSColor('--canvas-axis');
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(px, y2px(0)); ctx.lineTo(px + pw, y2px(0)); ctx.stroke();
  }

  /* curve */
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  const N = 200;
  for (let i = 0; i <= N; i++) {
    const x = x0 + i/N * (x1 - x0);
    const y = fn(x);
    const sx = x2px(x), sy = y2px(y);
    if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
  }
  ctx.stroke();

  /* frame */
  ctx.strokeStyle = getCSSColor('--border');
  ctx.lineWidth = 1;
  ctx.strokeRect(px, py, pw, ph);

  /* axis labels */
  ctx.fillStyle = getCSSColor('--text-soft');
  ctx.font = '10px monospace';
  ctx.fillText(String(x0), px - 2, py + ph + 14);
  ctx.fillText(String(x1), px + pw - 14, py + ph + 14);
  if (opts.xlabel) ctx.fillText(opts.xlabel, px + pw/2 - 12, py + ph + 14);
}

function refresh_wf_demos() {
  const wf = WAVEFORM_LIB[currentWaveform];
  draw_curve(cv.wf_time, t => wf.wf_time(t), [-0.7, 0.7], [-0.05, 1.1], 'rgb(245, 200, 70)', {xlabel: 't/T'});
  draw_curve(cv.wf_spec, f => wf.wf_spec(f), [-15, 15], [-0.05, 1.1], 'rgb(80, 180, 220)', {xlabel: 'f·T'});
  draw_curve(cv.wf_psf,  t => wf.h_range(t), [-3, 3], [-0.3, 1.1], 'rgb(220, 80, 80)', {xlabel: 'τ (cells)'});
}
refresh_wf_demos();

/* Waveform selector for the inline mini-demo */
const wfSelect = document.getElementById('sar-wf-select');
if (wfSelect) wfSelect.addEventListener('change', () => {
  /* Inline section selector is independent from the big-demo palette;
   * we only update the inline plots here, not the big demo. */
  const tmp = wfSelect.value;
  const saved = currentWaveform;
  currentWaveform = tmp; rebuild_range_psf();
  refresh_wf_demos();
  currentWaveform = saved; rebuild_range_psf();
});

/* Cross-range aperture inline demo (3 plots: aperture + 1D PSF + 2D U_p) */
function refresh_az_demos() {
  const L = S.Lsyn;
  draw_curve(cv.az_ap,
             x => (Math.abs(x) <= L/2 ? 1 : 0),
             [-2, 2], [-0.05, 1.1], 'rgb(245, 200, 70)', {xlabel: 'x_p / L_ref'});
  draw_curve(cv.az_psf,
             x => { const z = Math.PI * L * x;
                    if (Math.abs(z) < 1e-6) return 1;
                    return Math.abs(Math.sin(z) / z); },
             [-3, 3], [-0.3, 1.1], 'rgb(80, 180, 220)', {xlabel: 'x (cells at L_ref)'});
  /* 2D PSF: U_p(r, x) = h_r(r) · h_x(x) centered, peak = 1. */
  if (cv.az_2d) {
    const sz = 80;
    const buf = new Float64Array(sz * sz);
    let pk = 1e-9;
    for (let i = 0; i < sz; i++) {
      const x = +2.0 - (i + 0.5)/sz * 4.0;
      for (let j = 0; j < sz; j++) {
        const r = -2.0 + (j + 0.5)/sz * 4.0;
        const v = Math.max(0, U_at(r, x));
        buf[i * sz + j] = v;
        if (v > pk) pk = v;
      }
    }
    draw_grayscale(cv.az_2d, buf, sz, pk);
  }
}
refresh_az_demos();
const lsynSlider = document.getElementById('sar-Lsyn');
const lsynLabel  = document.getElementById('sar-Lsyn-val');
if (lsynSlider) lsynSlider.addEventListener('input', () => {
  S.Lsyn = parseFloat(lsynSlider.value);
  if (lsynLabel) lsynLabel.textContent = S.Lsyn.toFixed(2);
  refresh_az_demos();
});

/* ===========================================================================
 * Inline two-target mini-demos (sections 1 and 5).
 *
 * Each demo has its own state, its own noise pattern, its own draggable
 * sources.  Reuses the global U_at / k_at PSF lookups so it always
 * reflects the *currently selected waveform* from the big demo's
 * palette.  This keeps the inline demos consistent with the rest of
 * the page without duplicating PSF code.
 * =========================================================================== */

function make_inline_pair_demo(opts) {
  /* opts = { stateKey: 'sar-model' | 'sar-pair', defaultA1, defaultA2, cv: {fov, clean, noisy} } */
  const local = {
    P1r: opts.P1r0, P1x: opts.P1x0,
    P2r: opts.P2r0, P2x: opts.P2x0,
    A1: opts.defaultA1, A2: opts.defaultA2,
    sigma: 0.10,
  };
  const PI_FINE_INL = 120;
  const PI_PIX_INL  = 18;
  const FOV_INL     = 2.5;
  const muFine = new Float64Array(PI_FINE_INL * PI_FINE_INL);
  const muPix  = new Float64Array(PI_PIX_INL  * PI_PIX_INL);
  const noiseSamp = new Float64Array(PI_PIX_INL * PI_PIX_INL);
  let noiseInit = false;
  let drag = null, pressed = false, held = false;

  function ensureNoise() {
    if (noiseInit) return;
    for (let i = 0; i < noiseSamp.length; i++) noiseSamp[i] = gauss_rng();
    noiseInit = true;
  }
  function recompute() {
    let pk = 1e-9;
    for (let i = 0; i < PI_FINE_INL; i++) {
      const x = +FOV_INL - (i + 0.5)/PI_FINE_INL * 2*FOV_INL;
      for (let j = 0; j < PI_FINE_INL; j++) {
        const r = -FOV_INL + (j + 0.5)/PI_FINE_INL * 2*FOV_INL;
        const v = local.A1 * U_at(r - local.P1r, x - local.P1x) +
                  local.A2 * U_at(r - local.P2r, x - local.P2x);
        muFine[i * PI_FINE_INL + j] = v;
        if (v > pk) pk = v;
      }
    }
    /* Block-average to detector grid */
    const block = (PI_FINE_INL / PI_PIX_INL) | 0;
    const inv_b2 = 1 / (block * block);
    for (let pi = 0; pi < PI_PIX_INL; pi++) {
      for (let pj = 0; pj < PI_PIX_INL; pj++) {
        let s = 0;
        for (let di = 0; di < block; di++) {
          const rowOff = (pi * block + di) * PI_FINE_INL;
          for (let dj = 0; dj < block; dj++) s += muFine[rowOff + pj*block + dj];
        }
        muPix[pi * PI_PIX_INL + pj] = s * inv_b2;
      }
    }
    return pk;
  }
  function draw_fov() {
    if (!opts.cv.fov) return;
    const dpr = window.devicePixelRatio || 1;
    const cssW = opts.cv.fov.clientWidth, cssH = opts.cv.fov.clientHeight;
    const cw = Math.round(cssW*dpr), ch = Math.round(cssH*dpr);
    if (opts.cv.fov.width !== cw || opts.cv.fov.height !== ch) { opts.cv.fov.width = cw; opts.cv.fov.height = ch; }
    const ctx = opts.cv.fov.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = getCSSColor('--canvas-bg');
    ctx.fillRect(0, 0, cw, ch);
    const sz = Math.min(cw, ch);
    const x0 = ((cw - sz)/2)|0, y0 = ((ch - sz)/2)|0;
    ctx.strokeStyle = getCSSColor('--canvas-axis');
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x0 + sz/2, y0); ctx.lineTo(x0 + sz/2, y0 + sz);
    ctx.moveTo(x0, y0 + sz/2); ctx.lineTo(x0 + sz, y0 + sz/2);
    ctx.stroke();
    ctx.strokeStyle = getCSSColor('--border');
    ctx.strokeRect(x0, y0, sz, sz);
    /* Draw + drag sources */
    function dot(srcR, srcX, key, color) {
      const xs = x0 + (srcR + FOV_INL) / (2*FOV_INL) * sz;
      const ys = y0 + sz - (srcX + FOV_INL) / (2*FOV_INL) * sz;
      ctx.fillStyle = color; ctx.beginPath(); ctx.arc(xs, ys, 5*dpr, 0, 2*Math.PI); ctx.fill();
      const mp = opts.cv.fov._mp;
      if (mp) {
        const mx = mp.x * dpr, my = mp.y * dpr;
        const d2 = (mx-xs)**2 + (my-ys)**2;
        const near = d2 < (10*dpr)**2;
        if (pressed && !drag && near) drag = key;
        if (held && drag === key) {
          const newR = (mx - x0)/sz * 2*FOV_INL - FOV_INL;
          const newX = FOV_INL - (my - y0)/sz * 2*FOV_INL;
          local[key + 'r'] = Math.max(-FOV_INL, Math.min(FOV_INL, newR));
          local[key + 'x'] = Math.max(-FOV_INL, Math.min(FOV_INL, newX));
          opts.cv.fov.style.cursor = 'move';
        } else if (near) opts.cv.fov.style.cursor = 'move';
        else opts.cv.fov.style.cursor = 'default';
      }
    }
    dot(local.P1r, local.P1x, 'P1', 'rgb(220, 80, 80)');
    dot(local.P2r, local.P2x, 'P2', 'rgb(80, 130, 220)');
  }
  /* Mouse hooks for the FOV canvas */
  if (opts.cv.fov) {
    opts.cv.fov.addEventListener('mousemove', e => {
      const r = opts.cv.fov.getBoundingClientRect();
      opts.cv.fov._mp = { x: e.clientX - r.left, y: e.clientY - r.top };
    });
    opts.cv.fov.addEventListener('mouseleave', () => { opts.cv.fov._mp = null; });
    opts.cv.fov.addEventListener('mousedown', () => { pressed = true; held = true; });
    opts.cv.fov.addEventListener('mouseup',   () => { pressed = false; held = false; drag = null; });
  }
  /* Slider wiring */
  function bind(id, valId, key, dec=2) {
    const sl = document.getElementById(id), vl = document.getElementById(valId);
    if (!sl) return;
    sl.addEventListener('input', () => {
      local[key] = parseFloat(sl.value);
      if (vl) vl.textContent = local[key].toFixed(dec);
    });
    if (vl) vl.textContent = local[key].toFixed(dec);
  }
  bind(opts.stateKey + '-A1', opts.stateKey + '-A1-val', 'A1');
  bind(opts.stateKey + '-A2', opts.stateKey + '-A2-val', 'A2');
  bind(opts.stateKey + '-sigma', opts.stateKey + '-sigma-val', 'sigma', 3);
  const newNoiseBtn = document.getElementById(opts.stateKey + '-new-noise');
  if (newNoiseBtn) newNoiseBtn.addEventListener('click', () => {
    for (let i = 0; i < noiseSamp.length; i++) noiseSamp[i] = gauss_rng();
  });
  /* Main loop entry */
  function tick() {
    const pk = recompute();
    draw_fov();
    draw_grayscale(opts.cv.clean, muFine, PI_FINE_INL, pk);
    ensureNoise();
    const buf = new Float64Array(PI_PIX_INL * PI_PIX_INL);
    let pkN = 1e-9;
    for (let p = 0; p < buf.length; p++) {
      const v = muPix[p] + local.sigma * noiseSamp[p];
      buf[p] = Math.max(0, v);
      if (buf[p] > pkN) pkN = buf[p];
    }
    draw_grayscale(opts.cv.noisy, buf, PI_PIX_INL, pkN);
  }
  return tick;
}

/* Section-1 demo: one off-axis pair at moderate separation */
const tick_model_demo = make_inline_pair_demo({
  stateKey: 'sar-model',
  P1r0: -0.9, P1x0: 0.0,
  P2r0: +0.9, P2x0: 0.0,
  defaultA1: 1.0, defaultA2: 1.0,
  cv: { fov: cv.model_fov, clean: cv.model_clean, noisy: cv.model_noisy },
});

/* Section-5 demo: a closer pair to show merging */
const tick_pair_demo = make_inline_pair_demo({
  stateKey: 'sar-pair',
  P1r0: -0.50, P1x0: 0.0,
  P2r0: +0.50, P2x0: 0.0,
  defaultA1: 0.7, defaultA2: 1.0,
  cv: { fov: cv.pair_fov, clean: cv.pair_clean, noisy: cv.pair_noisy },
});

/* ---- BIG INTERACTIVE DEMO --------------------------------------------- */

/* Mu_P / pixelated / noise buffers */
const muFine    = new Float64Array(PI_FINE * PI_FINE);
const muPix     = new Float64Array(PI_PIX * PI_PIX);
const noiseSamp = new Float64Array(PI_PIX * PI_PIX);
let noiseInit = false;
let pMax = 1, bestMuMax = 1;

function ensureNoise() {
  if (noiseInit) return;
  for (let i = 0; i < noiseSamp.length; i++) noiseSamp[i] = gauss_rng();
  noiseInit = true;
}

/* Compute μ_P, m1_pix, m2_pix on fine + pixel grids. */
const m1_fine = new Float64Array(PI_FINE * PI_FINE);
const m2_fine = new Float64Array(PI_FINE * PI_FINE);
const m1_pix  = new Float64Array(PI_PIX * PI_PIX);
const m2_pix  = new Float64Array(PI_PIX * PI_PIX);
const Kpix    = { K11: 1, K12: 0, K22: 1 };

function compute_means() {
  pMax = 1e-9;
  for (let i = 0; i < PI_FINE; i++) {
    const x = +FOV - (i + 0.5)/PI_FINE * 2*FOV;
    for (let j = 0; j < PI_FINE; j++) {
      const r = -FOV + (j + 0.5)/PI_FINE * 2*FOV;
      const u1 = U_at(r - S.P1r, x - S.P1x);
      const u2 = U_at(r - S.P2r, x - S.P2x);
      const v  = S.A1 * u1 + S.A2 * u2;
      m1_fine[i * PI_FINE + j] = u1;
      m2_fine[i * PI_FINE + j] = u2;
      muFine[i * PI_FINE + j]  = v;
      if (v > pMax) pMax = v;
    }
  }
  /* Block-average to detector pixels. */
  const block = (PI_FINE / PI_PIX) | 0;
  const inv_b2 = 1 / (block * block);
  for (let pi = 0; pi < PI_PIX; pi++) {
    for (let pj = 0; pj < PI_PIX; pj++) {
      let sMu = 0, s1 = 0, s2 = 0;
      for (let di = 0; di < block; di++) {
        const rowOff = (pi * block + di) * PI_FINE;
        for (let dj = 0; dj < block; dj++) {
          const idx = rowOff + (pj * block + dj);
          sMu += muFine[idx];
          s1  += m1_fine[idx];
          s2  += m2_fine[idx];
        }
      }
      muPix[pi * PI_PIX + pj] = sMu * inv_b2;
      m1_pix[pi * PI_PIX + pj] = s1 * inv_b2;
      m2_pix[pi * PI_PIX + pj] = s2 * inv_b2;
    }
  }
  let K11 = 0, K12 = 0, K22 = 0;
  for (let p = 0; p < PI_PIX*PI_PIX; p++) {
    K11 += m1_pix[p]*m1_pix[p];
    K22 += m2_pix[p]*m2_pix[p];
    K12 += m1_pix[p]*m2_pix[p];
  }
  Kpix.K11 = K11; Kpix.K22 = K22; Kpix.K12 = K12;
}

/* ---- Matched-filter machinery on pixel grid --------------------------- */
const PIX_step = 2 * FOV / PI_PIX;
const PIX_0    = -FOV + 0.5 * PIX_step;
const PIX_x0   = +FOV - 0.5 * PIX_step;
const yBuf     = new Float64Array(PI_PIX * PI_PIX);
let sumY2 = 0;

function fill_y(noiseArr) {
  let s2 = 0;
  for (let p = 0; p < PI_PIX * PI_PIX; p++) {
    const v = muPix[p] + S.sigma * noiseArr[p];
    yBuf[p] = v;
    s2 += v * v;
  }
  sumY2 = s2;
}

function compute_b_at(qr, qx, y) {
  let s = 0;
  for (let pi = 0; pi < PI_PIX; pi++) {
    const pxx = PIX_x0 - pi * PIX_step;
    const yRow = pi * PI_PIX;
    for (let pj = 0; pj < PI_PIX; pj++) {
      const pr = PIX_0 + pj * PIX_step;
      s += U_at(pr - qr, pxx - qx) * y[yRow + pj];
    }
  }
  return s;
}
function compute_K_at(qr, qx) {
  let s = 0;
  for (let pi = 0; pi < PI_PIX; pi++) {
    const pxx = PIX_x0 - pi * PIX_step;
    for (let pj = 0; pj < PI_PIX; pj++) {
      const pr = PIX_0 + pj * PIX_step;
      const v = U_at(pr - qr, pxx - qx);
      s += v * v;
    }
  }
  return s;
}
function fit_1source(q0r, q0x, y) {
  /* 1D segment search seeded between p1 and p2, then gradient ascent.
   * Same approach as the airy demo but in (r, x) cell coords. */
  let qr = q0r, qx = q0x;
  let b  = compute_b_at(qr, qx, y);
  const EPS = 0.05, LR = 0.05;
  for (let it = 0; it < 10; it++) {
    const bxp = compute_b_at(qr + EPS, qx, y);
    const bxm = compute_b_at(qr - EPS, qx, y);
    const byp = compute_b_at(qr, qx + EPS, y);
    const bym = compute_b_at(qr, qx - EPS, y);
    const sgn = (b >= 0) ? 1 : -1;
    const gr = sgn * (bxp - bxm) / (2 * EPS);
    const gx = sgn * (byp - bym) / (2 * EPS);
    const gn = Math.hypot(gr, gx);
    if (gn < 1e-7) break;
    let alpha = LR, ok = false;
    for (let bs = 0; bs < 6; bs++) {
      const nr = qr + alpha * gr / gn;
      const nx = qx + alpha * gx / gn;
      const nb = compute_b_at(nr, nx, y);
      if (nb * nb > b * b + 1e-12) { qr = nr; qx = nx; b = nb; ok = true; break; }
      alpha *= 0.5;
    }
    if (!ok) break;
  }
  const Kq = compute_K_at(qr, qx);
  const F = (Kq > 1e-12) ? Math.max(0, b / Kq) : 0;   /* F̂ ≥ 0 */
  const J = sumY2 - F * b;
  return { qr, qx, F, J };
}

/* ---- D-bar computation (paper Eq. 6 in detector form) ----------------- */
function find_qstar(p1r, p1x, p2r, p2x, F1, F2) {
  const dr = p2r - p1r, dx = p2x - p1x;
  let bestQr = p1r, bestQx = p1x, bestF = -Infinity;
  const N = 80;
  for (let i = 0; i <= N; i++) {
    const t = -0.4 + 1.8 * i / N;
    const qr = p1r + t * dr, qx = p1x + t * dx;
    const f = F1 * k_at(qr - p1r, qx - p1x) + F2 * k_at(qr - p2r, qx - p2x);
    if (f > bestF) { bestF = f; bestQr = qr; bestQx = qx; }
  }
  /* Gradient ascent (finite-diff) in 2D. */
  for (let it = 0; it < 8; it++) {
    const eps = 0.03;
    const fpr = F1 * k_at((bestQr+eps) - p1r, bestQx - p1x) + F2 * k_at((bestQr+eps) - p2r, bestQx - p2x);
    const fmr = F1 * k_at((bestQr-eps) - p1r, bestQx - p1x) + F2 * k_at((bestQr-eps) - p2r, bestQx - p2x);
    const fpx = F1 * k_at(bestQr - p1r, (bestQx+eps) - p1x) + F2 * k_at(bestQr - p2r, (bestQx+eps) - p2x);
    const fmx = F1 * k_at(bestQr - p1r, (bestQx-eps) - p1x) + F2 * k_at(bestQr - p2r, (bestQx-eps) - p2x);
    const gr = (fpr - fmr) / (2 * eps);
    const gx = (fpx - fmx) / (2 * eps);
    const gn = Math.hypot(gr, gx);
    if (gn < 1e-9) break;
    let alpha = 0.10, ok = false;
    for (let bs = 0; bs < 8; bs++) {
      const nr = bestQr + alpha * gr / gn;
      const nx = bestQx + alpha * gx / gn;
      const fn = F1 * k_at(nr - p1r, nx - p1x) + F2 * k_at(nr - p2r, nx - p2x);
      if (fn > bestF + 1e-12) { bestF = fn; bestQr = nr; bestQx = nx; ok = true; break; }
      alpha *= 0.5;
    }
    if (!ok) break;
  }
  return { qr: bestQr, qx: bestQx, fmax: bestF };
}

function compute_Dbar(p1r, p1x, p2r, p2x, F1, F2) {
  const J1 = Math.max(F1*F1, 1e-9);
  const J2 = Math.max(F2*F2, 1e-9);
  const norm = 1/(2*J1) + 1/(2*J2);
  const { qr, qx, fmax } = find_qstar(p1r, p1x, p2r, p2x, F1, F2);
  const c1 = k_at(qr - p1r, qx - p1x);
  const c2 = k_at(qr - p2r, qx - p2x);
  const r  = k_at(p2r - p1r, p2x - p1x);
  const Fs = fmax;
  const muP = F1*F1 + F2*F2 + 2*F1*F2*r;
  const JPQ = muP - Fs*Fs;
  const Je  = Fs*Fs;
  const Jp1 = F1*F1 - 2*F1*Fs*c1 + Fs*Fs;
  const Jp2 = F2*F2 - 2*F2*Fs*c2 + Fs*Fs;
  const D   = Math.min(Je, Jp1, Jp2) - JPQ;
  return D * norm;
}

/* ---- D-bar map ---- */
const Dmap = new Float64Array(N_DGRID * N_DGRID);
let dmapDirty = true;
let mcDirty   = true;     /* MC error-bar cache invalidated on slider / drag */

/* Cached MC retrieval stats (computed by refresh_mc, read by draw_Flux). */
const McCache = {
  m1: 0, std1: 0, m2: 0, std2: 0,
};
const N_MC = 200;
function refresh_mc() {
  if (!mcDirty) return;
  if (S.sigma <= 0) {
    McCache.m1 = S.A1; McCache.std1 = 0;
    McCache.m2 = S.A2; McCache.std2 = 0;
    mcDirty = false;
    return;
  }
  let s1 = 0, s12 = 0, s2 = 0, s22 = 0;
  const tmp = new Float64Array(PI_PIX * PI_PIX);
  for (let s = 0; s < N_MC; s++) {
    for (let p = 0; p < tmp.length; p++) tmp[p] = gauss_rng();
    fill_y(tmp);
    const f1 = fit_1source(S.P1r, S.P1x, yBuf);
    const f2 = fit_1source(S.P2r, S.P2x, yBuf);
    s1 += f1.F; s12 += f1.F * f1.F;
    s2 += f2.F; s22 += f2.F * f2.F;
  }
  McCache.m1   = s1 / N_MC;
  McCache.std1 = Math.sqrt(Math.max(s12 / N_MC - McCache.m1 * McCache.m1, 0));
  McCache.m2   = s2 / N_MC;
  McCache.std2 = Math.sqrt(Math.max(s22 / N_MC - McCache.m2 * McCache.m2, 0));
  mcDirty = false;
}
function compute_Dmap() {
  const F1 = S.A1, F2 = S.A2;
  const p1r = S.P1r, p1x = S.P1x;
  for (let cy = 0; cy < N_DGRID; cy++) {
    const x = +FOV - (cy + 0.5)/N_DGRID * 2*FOV;
    for (let cx = 0; cx < N_DGRID; cx++) {
      const r = -FOV + (cx + 0.5)/N_DGRID * 2*FOV;
      Dmap[cy * N_DGRID + cx] = compute_Dbar(p1r, p1x, r, x, F1, F2);
    }
  }
}

/* ---- Dratio (D-bar vs separation, flux ratio) ---- */
const Dratio = new Float64Array(N_RATIO * N_DELTA);
let dratioDirty = true;
function build_Dratio() {
  const F2 = 1.0;
  for (let jr = 0; jr < N_RATIO; jr++) {
    const ratio = (N_RATIO - 1 - jr) / (N_RATIO - 1);
    const F1 = Math.max(ratio, 1e-3) * F2;
    for (let id = 0; id < N_DELTA; id++) {
      const delta = (id / (N_DELTA - 1)) * DELTA_MAX;
      Dratio[jr * N_DELTA + id] = compute_Dbar(-0.5 * delta, 0, +0.5 * delta, 0, F1, F2);
    }
  }
}

/* ---- Classification thresholds (Rd, Rr, Rc) ---- */
const Rd = new Float64Array(N_CLASS + 1);
const Rr = new Float64Array(N_CLASS + 1);
const Rc = new Float64Array(N_CLASS + 1);
function compute_classification() {
  const F1 = S.A1, F2 = S.A2;
  for (let i = 0; i <= N_CLASS; i++) {
    const Delta = i / N_CLASS * DELTA_MAX;
    const { qr } = find_qstar(-Delta/2, 0, +Delta/2, 0, F1, F2);
    const c1 = k_at(qr + Delta/2, 0);
    const c2 = k_at(qr - Delta/2, 0);
    const r  = k_at(Delta, 0);
    const Fs = F1 * c1 + F2 * c2;
    const muP = F1*F1 + F2*F2 + 2*F1*F2*r;
    const JPQ = muP - Fs*Fs;
    const Je  = Fs*Fs;
    const Jp1 = F1*F1 - 2*F1*Fs*c1 + Fs*Fs;
    const Jp2 = F2*F2 - 2*F2*Fs*c2 + Fs*Fs;
    const minPs = Math.min(Je, Jp1, Jp2);
    Rd[i] = Math.sqrt(muP / Math.max(muP - JPQ, 1e-12));
    Rr[i] = Math.sqrt(muP / Math.max(JPQ, 1e-12));
    Rc[i] = Math.sqrt(muP / Math.max(minPs, 1e-12));
  }
}

/* ---- area-of-{D>0} for δ_1 readout ---- */
function area_above_zero(v00, v10, v01, v11) {
  const idx = (v00>0?1:0)|(v10>0?2:0)|(v11>0?4:0)|(v01>0?8:0);
  if (idx === 0) return 0;
  if (idx === 15) return 1;
  const lerp = (a,b)=>(0-a)/(b-a);
  switch(idx) {
    case  1: return 0.5*lerp(v00,v10)*lerp(v00,v01);
    case 14: return 1-0.5*lerp(v00,v10)*lerp(v00,v01);
    case  2: return 0.5*(1-lerp(v00,v10))*lerp(v10,v11);
    case 13: return 1-0.5*(1-lerp(v00,v10))*lerp(v10,v11);
    case  4: return 0.5*(1-lerp(v01,v11))*(1-lerp(v10,v11));
    case 11: return 1-0.5*(1-lerp(v01,v11))*(1-lerp(v10,v11));
    case  8: return 0.5*lerp(v01,v11)*(1-lerp(v00,v01));
    case  7: return 1-0.5*lerp(v01,v11)*(1-lerp(v00,v01));
    case  3: return 0.5*(lerp(v00,v01)+lerp(v10,v11));
    case 12: return 1-0.5*(lerp(v00,v01)+lerp(v10,v11));
    case  6: return 0.5*((1-lerp(v00,v10))+(1-lerp(v01,v11)));
    case  9: return 1-0.5*((1-lerp(v00,v10))+(1-lerp(v01,v11)));
    case  5: return 0.5*lerp(v00,v10)*lerp(v00,v01)+0.5*(1-lerp(v01,v11))*(1-lerp(v10,v11));
    case 10: return 0.5*(1-lerp(v00,v10))*lerp(v10,v11)+0.5*lerp(v01,v11)*(1-lerp(v00,v01));
  }
  return 0;
}
function compute_dC() {
  const cellArea = (2*FOV/N_DGRID)**2;
  let area = 0;
  for (let cy = 0; cy < N_DGRID-1; cy++) {
    const r0 = cy*N_DGRID, r1 = (cy+1)*N_DGRID;
    for (let cx = 0; cx < N_DGRID-1; cx++) {
      area += area_above_zero(Dmap[r0+cx], Dmap[r0+cx+1], Dmap[r1+cx], Dmap[r1+cx+1]) * cellArea;
    }
  }
  return Math.sqrt(area / Math.PI);
}

/* ---- Mouse / drag state ----------------------------------------------- */
let dragSrc = null, pressed = false, held = false;

function localMouse(canvas) {
  if (canvas._mp) return canvas._mp;
  return null;
}
function attachMouse(canvas, name) {
  canvas.addEventListener('mousemove', e => {
    const r = canvas.getBoundingClientRect();
    canvas._mp = { x: e.clientX - r.left, y: e.clientY - r.top };
  });
  canvas.addEventListener('mouseleave', () => { canvas._mp = null; });
  canvas.addEventListener('mousedown', () => { pressed = true; held = true; });
  canvas.addEventListener('mouseup',   () => { pressed = false; held = false; dragSrc = null; });
}
[cv.fov, cv.P, cv.Dmap, cv.Dratio, cv.class].forEach((c, i) => {
  if (c) attachMouse(c, ['fov','P','Dmap','Dratio','class'][i]);
});
/* Global safety net: if the user releases the mouse outside any canvas
 * (or off-window), the per-canvas mouseup handlers never fire and drag
 * state stays stuck.  Catch that at the window level. */
window.addEventListener('mouseup', () => {
  pressed = false; held = false; dragSrc = null;
});

/* ---- Drawing helpers --------------------------------------------------- */
function draw_grayscale(canvas, buf, side, peak) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth, cssH = canvas.clientHeight;
  const cw = Math.round(cssW * dpr), ch = Math.round(cssH * dpr);
  if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = getCSSColor('--canvas-bg');
  ctx.fillRect(0, 0, cw, ch);
  const sz = Math.min(cw, ch);
  const x0 = ((cw - sz)/2)|0, y0 = ((ch - sz)/2)|0;
  const img = ctx.createImageData(sz, sz);
  for (let cy = 0; cy < sz; cy++) {
    const i = Math.min(side-1, (cy * side / sz) | 0);
    const row = i * side;
    for (let cx = 0; cx < sz; cx++) {
      const j = Math.min(side-1, (cx * side / sz) | 0);
      let t = buf[row + j] / Math.max(peak, 1e-9);
      if (t < 0) t = 0; else if (t > 1) t = 1;
      const [r,g,b] = inferno_rgb(t);
      const idx = (cy * sz + cx) * 4;
      img.data[idx]   = r;
      img.data[idx+1] = g;
      img.data[idx+2] = b;
      img.data[idx+3] = 255;
    }
  }
  ctx.putImageData(img, x0, y0);
  ctx.setTransform(dpr/dpr, 0, 0, dpr/dpr, 0, 0);
  ctx.strokeStyle = getCSSColor('--border');
  ctx.strokeRect(x0, y0, sz, sz);
  return { x0, y0, sz };
}

function drag_source_circle(canvas, geom, srcManim, srcKey, color) {
  /* srcManim = (r, x) in normalised cells; geom = grayscale layout. */
  const dpr = window.devicePixelRatio || 1;
  const { x0, y0, sz } = geom;
  const xs = x0 + (srcManim[0] + FOV) / (2*FOV) * sz;
  const ys = y0 + sz - (srcManim[1] + FOV) / (2*FOV) * sz;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = color;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5 * dpr;
  ctx.beginPath(); ctx.arc(xs, ys, 5*dpr, 0, 2*Math.PI); ctx.fill();

  /* hit-test + drag */
  const mp = canvas._mp;
  if (mp) {
    const mx = mp.x * dpr, my = mp.y * dpr;
    const d2 = (mx-xs)**2 + (my-ys)**2;
    const near = d2 < (10*dpr)**2;
    if (pressed && !dragSrc && near) { dragSrc = srcKey; }
    if (held && dragSrc === srcKey) {
      const newR = (mx - x0)/sz * 2*FOV - FOV;
      const newX = FOV - (my - y0)/sz * 2*FOV;
      S[srcKey + 'r'] = Math.max(-FOV, Math.min(FOV, newR));
      S[srcKey + 'x'] = Math.max(-FOV, Math.min(FOV, newX));
      dmapDirty = true; mcDirty = true;
      canvas.style.cursor = 'move';
    } else if (near) {
      canvas.style.cursor = 'move';
    } else {
      canvas.style.cursor = 'default';
    }
  }
}

/* ---- Per-plot draws ---------------------------------------------------- */
function draw_fov() {
  if (!cv.fov) return;
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.fov.clientWidth, cssH = cv.fov.clientHeight;
  const cw = Math.round(cssW * dpr), ch = Math.round(cssH * dpr);
  if (cv.fov.width !== cw || cv.fov.height !== ch) { cv.fov.width = cw; cv.fov.height = ch; }
  const ctx = cv.fov.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = getCSSColor('--canvas-bg');
  ctx.fillRect(0, 0, cw, ch);
  const sz = Math.min(cw, ch);
  const x0 = ((cw - sz)/2)|0, y0 = ((ch - sz)/2)|0;
  /* grid */
  ctx.strokeStyle = getCSSColor('--canvas-axis');
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x0 + sz/2, y0); ctx.lineTo(x0 + sz/2, y0 + sz);
  ctx.moveTo(x0, y0 + sz/2); ctx.lineTo(x0 + sz, y0 + sz/2);
  ctx.stroke();
  ctx.strokeStyle = getCSSColor('--border');
  ctx.strokeRect(x0, y0, sz, sz);
  drag_source_circle(cv.fov, { x0, y0, sz }, [S.P1r, S.P1x], 'P1', 'rgb(220, 80, 80)');
  drag_source_circle(cv.fov, { x0, y0, sz }, [S.P2r, S.P2x], 'P2', 'rgb(80, 130, 220)');
}

function draw_psf2d() {
  if (!cv.psf2d) return;
  /* Compute U_p centred on the canvas. */
  const sz = 80;
  const buf = new Float64Array(sz * sz);
  let pk = 1e-9;
  for (let i = 0; i < sz; i++) {
    const x = +FOV - (i + 0.5)/sz * 2*FOV;
    for (let j = 0; j < sz; j++) {
      const r = -FOV + (j + 0.5)/sz * 2*FOV;
      const v = Math.max(0, U_at(r, x));
      buf[i * sz + j] = v;
      if (v > pk) pk = v;
    }
  }
  draw_grayscale(cv.psf2d, buf, sz, pk);
}

function draw_clean_mu() {
  if (!cv.P) return;
  const geom = draw_grayscale(cv.P, muFine, PI_FINE, pMax);
  drag_source_circle(cv.P, geom, [S.P1r, S.P1x], 'P1', 'rgb(220, 80, 80)');
  drag_source_circle(cv.P, geom, [S.P2r, S.P2x], 'P2', 'rgb(80, 130, 220)');
}

function draw_noisy() {
  if (!cv.noisy) return;
  ensureNoise();
  const buf = new Float64Array(PI_PIX * PI_PIX);
  let pk = 1e-9;
  for (let p = 0; p < buf.length; p++) {
    const v = muPix[p] + S.sigma * noiseSamp[p];
    buf[p] = Math.max(0, v);
    if (buf[p] > pk) pk = buf[p];
  }
  draw_grayscale(cv.noisy, buf, PI_PIX, pk);
}

function draw_Dmap() {
  if (!cv.Dmap) return;
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.Dmap.clientWidth, cssH = cv.Dmap.clientHeight;
  const cw = Math.round(cssW*dpr), ch = Math.round(cssH*dpr);
  if (cv.Dmap.width !== cw || cv.Dmap.height !== ch) { cv.Dmap.width = cw; cv.Dmap.height = ch; }
  const ctx = cv.Dmap.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = getCSSColor('--canvas-bg');
  ctx.fillRect(0, 0, cw, ch);
  const sz = Math.min(cw, ch);
  const x0 = ((cw - sz)/2)|0, y0 = ((ch - sz)/2)|0;
  /* signed colormap: red = positive (contamination), grey = negative */
  const img = ctx.createImageData(sz, sz);
  let dmax = 1e-9;
  for (let p = 0; p < Dmap.length; p++) {
    const v = Math.abs(Dmap[p]); if (v > dmax) dmax = v;
  }
  for (let cy = 0; cy < sz; cy++) {
    const i = Math.min(N_DGRID-1, (cy * N_DGRID / sz)|0);
    const row = i * N_DGRID;
    for (let cx = 0; cx < sz; cx++) {
      const j = Math.min(N_DGRID-1, (cx * N_DGRID / sz)|0);
      const v = Dmap[row + j] / dmax;
      let r, g, b;
      if (v > 0) {
        const t = Math.min(1, v);
        r = 240; g = 220 - 130*t; b = 80 - 70*t;     /* yellow → orange */
      } else {
        const t = Math.min(1, -v);
        r = 30 + 90*(1-t); g = 40 + 90*(1-t); b = 70 + 110*(1-t);  /* dark blue */
      }
      const idx = (cy * sz + cx) * 4;
      img.data[idx]   = r;
      img.data[idx+1] = g;
      img.data[idx+2] = b;
      img.data[idx+3] = 255;
    }
  }
  ctx.putImageData(img, x0, y0);
  ctx.strokeStyle = getCSSColor('--border');
  ctx.strokeRect(x0, y0, sz, sz);
  /* red contour at D=0 */
  ctx.strokeStyle = 'rgb(220, 60, 60)';
  ctx.lineWidth = 2;
  for (let cy = 0; cy < N_DGRID-1; cy++) {
    const r0 = cy * N_DGRID, r1 = (cy+1) * N_DGRID;
    for (let cx = 0; cx < N_DGRID-1; cx++) {
      const v00 = Dmap[r0+cx], v10 = Dmap[r0+cx+1], v01 = Dmap[r1+cx], v11 = Dmap[r1+cx+1];
      const idx = (v00>0?1:0)|(v10>0?2:0)|(v11>0?4:0)|(v01>0?8:0);
      if (idx === 0 || idx === 15) continue;
      const x = x0 + cx * sz / N_DGRID;
      const y = y0 + cy * sz / N_DGRID;
      ctx.beginPath();
      ctx.moveTo(x, y + sz/N_DGRID/2);
      ctx.lineTo(x + sz/N_DGRID, y + sz/N_DGRID/2);
      ctx.stroke();
    }
  }
  /* p1 = origin dot */
  ctx.fillStyle = 'rgb(220, 80, 80)';
  ctx.beginPath();
  ctx.arc(x0 + sz/2, y0 + sz/2, 4, 0, 2*Math.PI);
  ctx.fill();
}

function draw_Dratio() {
  if (!cv.Dratio) return;
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.Dratio.clientWidth, cssH = cv.Dratio.clientHeight;
  const cw = Math.round(cssW*dpr), ch = Math.round(cssH*dpr);
  if (cv.Dratio.width !== cw || cv.Dratio.height !== ch) { cv.Dratio.width = cw; cv.Dratio.height = ch; }
  const ctx = cv.Dratio.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = getCSSColor('--canvas-bg');
  ctx.fillRect(0, 0, cw, ch);
  const padL = 36, padR = 8, padT = 14, padB = 22;
  const px = padL, py = padT;
  const pw = cw - padL - padR, ph = ch - padT - padB;
  const img = ctx.createImageData(pw|0, ph|0);
  let dmax = 1e-9;
  for (let p = 0; p < Dratio.length; p++) { const v = Math.abs(Dratio[p]); if (v > dmax) dmax = v; }
  for (let cy = 0; cy < ph; cy++) {
    const jr = Math.min(N_RATIO-1, (cy * N_RATIO / ph)|0);
    const row = jr * N_DELTA;
    for (let cx = 0; cx < pw; cx++) {
      const id = Math.min(N_DELTA-1, (cx * N_DELTA / pw)|0);
      const v = Dratio[row + id] / dmax;
      let r, g, b;
      if (v > 0) { const t = Math.min(1, v); r=240; g=220-130*t; b=80-70*t; }
      else { const t = Math.min(1, -v); r=30+90*(1-t); g=40+90*(1-t); b=70+110*(1-t); }
      const idx = (cy * pw + cx) * 4;
      img.data[idx]=r; img.data[idx+1]=g; img.data[idx+2]=b; img.data[idx+3]=255;
    }
  }
  ctx.putImageData(img, px, py);
  /* Switch to CSS coordinates for the yellow dot + drag handling. */
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const cssPx = padL, cssPy = padT;
  const cssPw = cssW - padL - padR, cssPh = cssH - padT - padB;

  /* Yellow dot at current (Δ, ratio); draggable. */
  const dNow  = Math.hypot(S.P2r - S.P1r, S.P2x - S.P1x);
  const ratio = Math.min(S.A1, S.A2) / Math.max(S.A1, S.A2);
  const cxd = cssPx + Math.max(0, Math.min(1, dNow / DELTA_MAX)) * cssPw;
  const cyd = cssPy + (1 - ratio) * cssPh;

  const mpcp = cv.Dratio._mp;
  let nearD = false;
  if (mpcp) {
    const d2 = (mpcp.x - cxd)**2 + (mpcp.y - cyd)**2;
    nearD = d2 < 144;
  }
  const rOuter = (dragSrc === 'DRATIO' || nearD) ? 10 : 8;
  const rInner = (dragSrc === 'DRATIO' || nearD) ? 6 : 5;
  ctx.fillStyle   = 'rgb(245, 200, 70)';
  ctx.strokeStyle = 'rgb(245, 200, 70)';
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(cxd, cyd, rInner, 0, 2*Math.PI); ctx.fill();
  ctx.beginPath(); ctx.arc(cxd, cyd, rOuter, 0, 2*Math.PI); ctx.stroke();

  /* Drag start */
  if (mpcp && pressed && !dragSrc && nearD) dragSrc = 'DRATIO';

  /* Drag updates: x → Δ (rescale p1,p2 about midpoint); y → ratio (rescale α1,α2). */
  if (mpcp && held && dragSrc === 'DRATIO') {
    const mx = Math.max(cssPx, Math.min(cssPx + cssPw, mpcp.x));
    const my = Math.max(cssPy, Math.min(cssPy + cssPh, mpcp.y));
    const newDelta = (mx - cssPx) / cssPw * DELTA_MAX;
    let newRatio = 1 - (my - cssPy) / cssPh;
    newRatio = Math.max(0.001, Math.min(1, newRatio));

    /* Rescale p1, p2 symmetrically about midpoint. */
    const drp = S.P2r - S.P1r, dxp = S.P2x - S.P1x;
    const len = Math.hypot(drp, dxp);
    let ur = 1, ux = 0;
    if (len > 1e-9) { ur = drp / len; ux = dxp / len; }
    const cxm = 0.5 * (S.P1r + S.P2r);
    const cym = 0.5 * (S.P1x + S.P2x);
    S.P1r = Math.max(-FOV, Math.min(FOV, cxm - 0.5 * newDelta * ur));
    S.P1x = Math.max(-FOV, Math.min(FOV, cym - 0.5 * newDelta * ux));
    S.P2r = Math.max(-FOV, Math.min(FOV, cxm + 0.5 * newDelta * ur));
    S.P2x = Math.max(-FOV, Math.min(FOV, cym + 0.5 * newDelta * ux));

    /* Solve for α1, α2 preserving ‖α‖². */
    const Rnorm = Math.sqrt(S.A1*S.A1 + S.A2*S.A2);
    const Amax  = Rnorm / Math.sqrt(1 + newRatio * newRatio);
    const Amin  = Amax * newRatio;
    const cl = a => Math.max(0.05, Math.min(2, a));
    if (S.A1 >= S.A2) { S.A1 = cl(Amax); S.A2 = cl(Amin); }
    else              { S.A1 = cl(Amin); S.A2 = cl(Amax); }

    /* Sync sliders. */
    const sl1 = document.getElementById('sar-A1'), lb1 = document.getElementById('sar-A1-val');
    const sl2 = document.getElementById('sar-A2'), lb2 = document.getElementById('sar-A2-val');
    if (sl1) sl1.value = S.A1;
    if (lb1) lb1.textContent = S.A1.toFixed(2);
    if (sl2) sl2.value = S.A2;
    if (lb2) lb2.textContent = S.A2.toFixed(2);

    dmapDirty = true; dratioDirty = true; mcDirty = true;
  }

  cv.Dratio.style.cursor = (nearD || dragSrc === 'DRATIO') ? 'move' : 'default';

  /* Frame + axis labels (CSS coords). */
  ctx.strokeStyle = getCSSColor('--border'); ctx.strokeRect(cssPx, cssPy, cssPw, cssPh);
  ctx.fillStyle = getCSSColor('--text-soft');
  ctx.font = '10px monospace';
  ctx.fillText('0', cssPx - 4, cssPy + cssPh + 14);
  ctx.fillText(DELTA_MAX.toFixed(1), cssPx + cssPw - 16, cssPy + cssPh + 14);
  ctx.fillText('1.0', 6, cssPy + 8);
  ctx.fillText('0.0', 6, cssPy + cssPh - 2);
}

function draw_Best() {
  if (!cv.best) return;
  /* Draw the current-sample best fit μ̂. */
  fill_y(noiseSamp);
  const fit1 = fit_1source(S.P1r, S.P1x, yBuf);
  const fit2 = fit_1source(S.P2r, S.P2x, yBuf);
  /* Render the two-source best fit on a fine grid. */
  const sz = 100;
  const buf = new Float64Array(sz * sz);
  let pk = 1e-9;
  for (let i = 0; i < sz; i++) {
    const x = +FOV - (i + 0.5)/sz * 2*FOV;
    for (let j = 0; j < sz; j++) {
      const r = -FOV + (j + 0.5)/sz * 2*FOV;
      const v = Math.max(0, fit1.F * U_at(r - fit1.qr, x - fit1.qx) +
                            fit2.F * U_at(r - fit2.qr, x - fit2.qx));
      buf[i * sz + j] = v;
      if (v > pk) pk = v;
    }
  }
  const geom = draw_grayscale(cv.best, buf, sz, pk);
  /* Truth dots */
  const ctx = cv.best.getContext('2d');
  const f2sx = r => geom.x0 + (r + FOV) / (2*FOV) * geom.sz;
  const f2sy = x => geom.y0 + geom.sz - (x + FOV) / (2*FOV) * geom.sz;
  ctx.fillStyle = 'rgb(220, 80, 80)';
  ctx.beginPath(); ctx.arc(f2sx(S.P1r), f2sy(S.P1x), 4, 0, 2*Math.PI); ctx.fill();
  ctx.fillStyle = 'rgb(80, 130, 220)';
  ctx.beginPath(); ctx.arc(f2sx(S.P2r), f2sy(S.P2x), 4, 0, 2*Math.PI); ctx.fill();
  /* Fit diamonds */
  function diamond(x, y, col) {
    ctx.fillStyle = col; ctx.strokeStyle = 'rgb(20,22,30)'; ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(x, y-5); ctx.lineTo(x+5, y); ctx.lineTo(x, y+5); ctx.lineTo(x-5, y);
    ctx.closePath(); ctx.fill(); ctx.stroke();
  }
  diamond(f2sx(fit1.qr), f2sy(fit1.qx), 'rgb(220, 80, 80)');
  diamond(f2sx(fit2.qr), f2sy(fit2.qx), 'rgb(80, 130, 220)');
}

function draw_Flux() {
  if (!cv.flux) return;
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.flux.clientWidth, cssH = cv.flux.clientHeight;
  const cw = Math.round(cssW*dpr), ch = Math.round(cssH*dpr);
  if (cv.flux.width !== cw || cv.flux.height !== ch) { cv.flux.width = cw; cv.flux.height = ch; }
  const ctx = cv.flux.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = getCSSColor('--canvas-bg');
  ctx.fillRect(0, 0, cssW, cssH);
  const padL = 36, padR = 12, padT = 18, padB = 22;
  const px = padL, py = padT;
  const pw = cssW - padL - padR, ph = cssH - padT - padB;
  const yLo = -0.2, yHi = Math.max(2.5, Math.max(S.A1, S.A2) * 1.6);
  const y2px = y => py + ph - (y - yLo)/(yHi - yLo) * ph;
  /* zero line */
  ctx.strokeStyle = getCSSColor('--canvas-axis');
  ctx.lineWidth = 1; ctx.beginPath();
  ctx.moveTo(px, y2px(0)); ctx.lineTo(px + pw, y2px(0)); ctx.stroke();
  /* Two columns: α₁, α₂ */
  const x1 = px + pw * 0.33;
  const x2 = px + pw * 0.66;
  function bar(xc, mean, std, col) {
    const ym = y2px(mean);
    const yhi = y2px(mean + 3*std);
    const ylo = y2px(mean - 3*std);
    ctx.strokeStyle = col; ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(xc, ylo); ctx.lineTo(xc, yhi);
    ctx.moveTo(xc - 6, ylo); ctx.lineTo(xc + 6, ylo);
    ctx.moveTo(xc - 6, yhi); ctx.lineTo(xc + 6, yhi);
    ctx.stroke();
    ctx.fillStyle = col;
    ctx.beginPath(); ctx.arc(xc, ym, 3.5, 0, 2*Math.PI); ctx.fill();
  }
  function actualBox(xc, val, col) {
    const ya = y2px(val);
    ctx.strokeStyle = col; ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.rect(xc - 6, ya - 6, 12, 12); ctx.stroke();
  }
  actualBox(x1, S.A1, 'rgb(220, 80, 80)');
  actualBox(x2, S.A2, 'rgb(80, 130, 220)');
  /* Read the cached MC stats (computed by refresh_mc on mcDirty). */
  if (S.sigma > 0) {
    bar(x1 + 14, McCache.m1, McCache.std1, 'rgb(220, 80, 80)');
    bar(x2 + 14, McCache.m2, McCache.std2, 'rgb(80, 130, 220)');
  }
  ctx.strokeStyle = getCSSColor('--border'); ctx.strokeRect(px, py, pw, ph);
  ctx.fillStyle = getCSSColor('--text-soft'); ctx.font = '10px monospace';
  ctx.fillText('α', px - 16, py + ph/2);
  ctx.fillText('α₁', x1 - 8, py + ph + 14);
  ctx.fillText('α₂', x2 - 8, py + ph + 14);
  ctx.fillText(yHi.toFixed(1), 8, py + 8);
  ctx.fillText('0', 18, y2px(0) - 2);
}

function draw_Class() {
  if (!cv.class) return;
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.class.clientWidth, cssH = cv.class.clientHeight;
  const cw = Math.round(cssW*dpr), ch = Math.round(cssH*dpr);
  if (cv.class.width !== cw || cv.class.height !== ch) { cv.class.width = cw; cv.class.height = ch; }
  const ctx = cv.class.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = getCSSColor('--canvas-bg');
  ctx.fillRect(0, 0, cssW, cssH);
  const padL = 36, padR = 8, padT = 14, padB = 22;
  const px = padL, py = padT;
  const pw = cssW - padL - padR, ph = cssH - padT - padB;
  const lr_min = Math.log10(0.4), lr_max = Math.log10(100);
  const R2y = R => py + ph - (Math.log10(Math.max(R, 0.4)) - lr_min)/(lr_max - lr_min) * ph;
  const colGrey = 'rgb(80,84,96)', colBlue = 'rgb(70,110,180)',
        colRed  = 'rgb(200,80,80)', colGreen = 'rgb(80,180,100)';
  for (let xp = 0; xp < pw; xp++) {
    const Delta = xp / pw * DELTA_MAX;
    const t = Delta / DELTA_MAX * N_CLASS;
    const i_lo = Math.max(0, Math.min(N_CLASS-1, Math.floor(t)));
    const f = t - i_lo;
    const Rdp = Rd[i_lo]*(1-f) + Rd[i_lo+1]*f;
    const Rrp = Rr[i_lo]*(1-f) + Rr[i_lo+1]*f;
    const Rcp = Rc[i_lo]*(1-f) + Rc[i_lo+1]*f;
    const yp = px + xp;
    const yd = Math.max(py, Math.min(py+ph, R2y(Rdp)));
    const yr = Math.max(py, Math.min(py+ph, R2y(Rrp)));
    const yc = Math.max(py, Math.min(py+ph, R2y(Rcp)));
    ctx.fillStyle = colGrey;  ctx.fillRect(yp, yd, 1, py+ph - yd);
    if (Rcp < Rrp) {
      ctx.fillStyle = colBlue; ctx.fillRect(yp, yc, 1, yd - yc);
      ctx.fillStyle = colRed;  ctx.fillRect(yp, yr, 1, yc - yr);
    } else {
      ctx.fillStyle = colBlue; ctx.fillRect(yp, yr, 1, yd - yr);
    }
    ctx.fillStyle = colGreen; ctx.fillRect(yp, py, 1, yr - py);
  }
  ctx.strokeStyle = getCSSColor('--border'); ctx.strokeRect(px, py, pw, ph);
  /* yellow dot at current config — draggable (changes Δ + σ via R) */
  const Delta_now = Math.hypot(S.P2r - S.P1r, S.P2x - S.P1x);
  const k_now = k_at(S.P2r - S.P1r, S.P2x - S.P1x);
  const muPsq = S.A1*S.A1 + S.A2*S.A2 + 2*S.A1*S.A2*k_now;
  const dx_pix = 2*FOV/PI_PIX;
  const safeSig = Math.max(S.sigma * dx_pix, 1e-9);
  const R_now = Math.sqrt(muPsq) / (safeSig * etaVal);

  let xc = -1, yc = -1, near = false;
  if (Delta_now <= DELTA_MAX) {
    xc = px + Delta_now/DELTA_MAX * pw;
    yc = R2y(Math.max(0.4, Math.min(100, R_now)));
    const mp = cv.class._mp;
    if (mp) {
      const d2 = (mp.x - xc)**2 + (mp.y - yc)**2;
      near = d2 < 144;
    }
    const rOuter = (dragSrc === 'CLASS' || near) ? 10 : 8;
    const rInner = (dragSrc === 'CLASS' || near) ? 6 : 5;
    ctx.fillStyle   = 'rgb(245, 200, 70)';
    ctx.strokeStyle = 'rgb(245, 200, 70)';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(xc, yc, rInner, 0, 2*Math.PI); ctx.fill();
    ctx.beginPath(); ctx.arc(xc, yc, rOuter, 0, 2*Math.PI); ctx.stroke();
  }

  /* Drag start */
  const mpcp = cv.class._mp;
  if (mpcp && pressed && !dragSrc && near) dragSrc = 'CLASS';

  /* Drag updates: new x → Δ, new y → R → solve for σ. */
  if (mpcp && held && dragSrc === 'CLASS') {
    const mx = Math.max(px, Math.min(px + pw, mpcp.x));
    const my = Math.max(py, Math.min(py + ph, mpcp.y));
    const newDelta = (mx - px) / pw * DELTA_MAX;
    const newLr    = (py + ph - my) / ph * (lr_max - lr_min) + lr_min;
    const newR     = Math.pow(10, newLr);

    /* Rescale (p1, p2) symmetrically about midpoint. */
    const dxp = S.P2r - S.P1r, dyp = S.P2x - S.P1x;
    const len = Math.hypot(dxp, dyp);
    let ux = 1, uy = 0;
    if (len > 1e-9) { ux = dxp / len; uy = dyp / len; }
    const cxm = 0.5 * (S.P1r + S.P2r);
    const cym = 0.5 * (S.P1x + S.P2x);
    S.P1r = Math.max(-FOV, Math.min(FOV, cxm - 0.5 * newDelta * ux));
    S.P1x = Math.max(-FOV, Math.min(FOV, cym - 0.5 * newDelta * uy));
    S.P2r = Math.max(-FOV, Math.min(FOV, cxm + 0.5 * newDelta * ux));
    S.P2x = Math.max(-FOV, Math.min(FOV, cym + 0.5 * newDelta * uy));

    /* Solve for σ given target R. */
    const new_kernel = k_at(S.P2r - S.P1r, S.P2x - S.P1x);
    const muPsq_new = S.A1*S.A1 + S.A2*S.A2 + 2*S.A1*S.A2*new_kernel;
    let new_sigma = Math.sqrt(muPsq_new) / (Math.max(newR, 1e-6) * etaVal * dx_pix);
    new_sigma = Math.max(0.001, Math.min(4.0, new_sigma));
    S.sigma = new_sigma;

    /* Sync sliders + labels. */
    const sigSlider = document.getElementById('sar-sigma');
    const sigLabel  = document.getElementById('sar-sigma-val');
    if (sigSlider) sigSlider.value = S.sigma;
    if (sigLabel)  sigLabel.textContent = S.sigma.toFixed(3);
    dmapDirty = true; dratioDirty = true; mcDirty = true;
  }

  cv.class.style.cursor = (near || dragSrc === 'CLASS') ? 'move' : 'default';
  /* axis labels */
  ctx.fillStyle = getCSSColor('--text-soft');
  ctx.font = '10px monospace';
  ctx.fillText('100', 4, py + 8);
  ctx.fillText('10', 6, py + ph - (1 - lr_min)/(lr_max - lr_min)*ph - 2);
  ctx.fillText('1', 8, py + ph - (0 - lr_min)/(lr_max - lr_min)*ph - 2);
  ctx.fillText('0.4', 4, py + ph - 2);
  ctx.fillText('0', px - 4, py + ph + 14);
  ctx.fillText(DELTA_MAX.toFixed(1), px + pw - 16, py + ph + 14);
  ctx.fillText('R', 4, py + ph/2);
  ctx.fillText('Δ', px + pw/2, py + ph + 14);
}

/* ---- δ_C readout ------------------------------------------------------ */
const dcEl = document.getElementById('sar-deltaC-val');
function update_dC_readout() {
  if (!dcEl) return;
  const d = compute_dC();
  if (!isFinite(d) || d <= 0) { dcEl.textContent = '—'; return; }
  dcEl.innerHTML = `${d.toFixed(3)} cells&nbsp;&nbsp;(${(d/RAYLEIGH).toFixed(3)} δ<sub>R</sub>)`;
}

/* ---- η + waveform-name readout ----------------------------------------- */
function update_class_label() {
  const elE = document.getElementById('sar-class-eta');
  const elN = document.getElementById('sar-class-wfname');
  if (elE) elE.textContent = etaVal.toFixed(2);
  if (elN) elN.textContent = WAVEFORM_LIB[currentWaveform].name;
}
update_class_label();

/* ---- Waveform palette: click to switch -------------------------------- */
function setup_palette() {
  const tiles = document.querySelectorAll('[data-waveform]');
  function refreshActive() {
    tiles.forEach(t => t.classList.toggle('active', t.dataset.waveform === currentWaveform));
  }
  tiles.forEach(tile => {
    /* Render a small icon: the range PSF for this waveform */
    const icon = tile.querySelector('canvas.aperture-tile-icon');
    if (icon) {
      const w = icon.width = 36, h = icon.height = 36;
      const ctx = icon.getContext('2d');
      const wf = WAVEFORM_LIB[tile.dataset.waveform];
      ctx.fillStyle = 'rgb(22, 24, 32)'; ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = 'rgb(245, 220, 80)'; ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (let i = 0; i <= w; i++) {
        const tau = -3 + 6 * i / w;
        const v = Math.abs(wf.h_range(tau));
        const x = i;
        const y = h - 4 - v * (h - 8);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    tile.addEventListener('click', () => {
      currentWaveform = tile.dataset.waveform;
      rebuild_range_psf();
      compute_eta();
      update_class_label();
      refresh_az_demos();      /* update section-3 2D PSF panel */
      dmapDirty = true; dratioDirty = true;
      refreshActive();
    });
  });
  refreshActive();
}
setup_palette();

/* ---- Sliders ---------------------------------------------------------- */
function bindSlider(id, valId, key, decimals = 2, after = null) {
  const slider = document.getElementById(id);
  const valEl  = document.getElementById(valId);
  if (!slider) return;
  slider.addEventListener('input', () => {
    S[key] = parseFloat(slider.value);
    if (valEl) valEl.textContent = S[key].toFixed(decimals);
    /* Position-dependent: dmap + dratio invalidated.
     * Flux- or noise-dependent: MC bars invalidated too. */
    if (key === 'A1' || key === 'A2') { dmapDirty = true; dratioDirty = true; }
    mcDirty = true;
    if (after) after();
  });
  if (valEl) valEl.textContent = S[key].toFixed(decimals);
}
bindSlider('sar-A1', 'sar-A1-val', 'A1', 2);
bindSlider('sar-A2', 'sar-A2-val', 'A2', 2);
bindSlider('sar-sigma', 'sar-sigma-val', 'sigma', 3);
bindSlider('sar-Lsyn-big', 'sar-Lsyn-big-val', 'Lsyn', 2);

const newNoiseBtn = document.getElementById('sar-new-noise');
if (newNoiseBtn) newNoiseBtn.addEventListener('click', () => {
  for (let i = 0; i < noiseSamp.length; i++) noiseSamp[i] = gauss_rng();
});

/* ---- Main loop ------------------------------------------------------- */
function loop() {
  compute_means();
  if (dmapDirty && !held) {
    compute_Dmap();
    compute_classification();
    dmapDirty = false;
    update_dC_readout();
  }
  if (dratioDirty && !held) { build_Dratio(); dratioDirty = false; }
  if (mcDirty && !held) { refresh_mc(); }      /* MC bars: cached */
  draw_fov();
  draw_psf2d();
  draw_noisy();
  draw_clean_mu();
  draw_Dmap();
  draw_Dratio();
  draw_Best();
  draw_Flux();
  draw_Class();
  /* Inline section demos */
  if (tick_model_demo) tick_model_demo();
  if (tick_pair_demo)  tick_pair_demo();
  /* Refresh inline cross-range 2D PSF (waveform may have changed) */
  /* draw_grayscale is recomputed by refresh_az_demos each frame would be
   * wasteful; the only time it really needs updating is on waveform
   * change.  Skip per-frame refresh. */
  requestAnimationFrame(loop);
}
loop();

})();
