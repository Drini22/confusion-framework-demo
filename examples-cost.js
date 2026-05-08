/* ===========================================================================
 * Mini-demo for the "The cost function" section.
 *
 *   Plot 1 (violet)  : mu_P, with TWO draggable sources p1 (red) and p2 (blue)
 *   Plot 2 (green)   : mu_Q, with one draggable source q (green)
 *   Plot 3 (amber)   : loss map J(P, {(q, F_Q)}) for q sweeping the FOV
 *
 * The current Q's position appears as a green dot on plots 2 and 3 — drag
 * from either to move it.  Faint red/blue rings on plot 3 show p1, p2 so
 * the basins of low J are easy to locate.
 *
 * J is computed analytically with the Gaussian-kernel formula
 *   J(P, Q_q) = ||mu_P||^2 - 2 F_Q sum_i F_i k(p_i, q) + F_Q^2 ||U||^2,
 *   k(p, q)   = pi sigma^2 exp(-||p - q||^2 / (4 sigma^2)),
 *   ||U||^2   = pi sigma^2.
 * =========================================================================== */

(() => {

const cvP = document.getElementById('cost-P');
const cvQ = document.getElementById('cost-Q');
const cvL = document.getElementById('cost-loss');
if (!cvP || !cvQ || !cvL) return;

const SIGMA_PSF = 1.0;
const RANGE     = 3.0;
const N         = 240;     /* mu_P, mu_Q heatmap resolution */
const N_LOSS    = 120;     /* loss map sampling resolution  */

const S = {
  P1x: -0.7, P1y: 0.0,
  P2x: +0.7, P2y: 0.0,
  Qx:  0.0,  Qy:  1.2,
  FP1: 1.0, FP2: 1.0, FQ: 1.0,
};

const phi2 = (dx, dy) =>
  Math.exp(-(dx*dx + dy*dy) / (2.0 * SIGMA_PSF * SIGMA_PSF));

const PI_SIGMA_SQ = Math.PI * SIGMA_PSF * SIGMA_PSF;
function kernel(px, py, qx, qy) {
  const dx = px - qx, dy = py - qy;
  return PI_SIGMA_SQ * Math.exp(-(dx*dx + dy*dy) / (4.0 * SIGMA_PSF * SIGMA_PSF));
}

/* Buffers */
const muP   = new Float64Array(N * N);
const muQ   = new Float64Array(N * N);
const Lmap  = new Float64Array(N_LOSS * N_LOSS);

/* Per-frame statistics */
let pMax = 1, qMax = 1;
let lossMin = 0, lossMax = 1;
let Jcurrent = 0;

function compute_means() {
  pMax = 1e-9; qMax = 1e-9;
  for (let i = 0; i < N; i++) {
    const y = +RANGE - (i + 0.5) / N * 2*RANGE;
    for (let j = 0; j < N; j++) {
      const x = -RANGE + (j + 0.5) / N * 2*RANGE;
      const p = S.FP1 * phi2(x - S.P1x, y - S.P1y)
              + S.FP2 * phi2(x - S.P2x, y - S.P2y);
      const q = S.FQ  * phi2(x - S.Qx,  y - S.Qy);
      muP[i * N + j] = p;
      muQ[i * N + j] = q;
      if (p > pMax) pMax = p;
      if (q > qMax) qMax = q;
    }
  }
}

/* ||mu_P||^2 (analytic, used everywhere in this demo) */
function muP_norm_sq() {
  return S.FP1*S.FP1 * kernel(S.P1x, S.P1y, S.P1x, S.P1y)
       + S.FP2*S.FP2 * kernel(S.P2x, S.P2y, S.P2x, S.P2y)
       + 2 * S.FP1 * S.FP2 * kernel(S.P1x, S.P1y, S.P2x, S.P2y);
}

/* Analytic J(P, Q) */
function J_at(qx, qy) {
  const pp = muP_norm_sq();
  const pq = S.FQ * (S.FP1 * kernel(S.P1x, S.P1y, qx, qy)
                   + S.FP2 * kernel(S.P2x, S.P2y, qx, qy));
  const qq = S.FQ * S.FQ * PI_SIGMA_SQ;
  return pp - 2 * pq + qq;
}

function compute_loss_map() {
  const pp = muP_norm_sq();
  const qqAt0 = S.FQ * S.FQ * PI_SIGMA_SQ;
  lossMin = Infinity; lossMax = -Infinity;
  for (let i = 0; i < N_LOSS; i++) {
    const y = +RANGE - (i + 0.5) / N_LOSS * 2*RANGE;
    for (let j = 0; j < N_LOSS; j++) {
      const x = -RANGE + (j + 0.5) / N_LOSS * 2*RANGE;
      const pq = S.FQ * (S.FP1 * kernel(S.P1x, S.P1y, x, y)
                       + S.FP2 * kernel(S.P2x, S.P2y, x, y));
      const J  = pp - 2 * pq + qqAt0;
      Lmap[i * N_LOSS + j] = J;
      if (J < lossMin) lossMin = J;
      if (J > lossMax) lossMax = J;
    }
  }
  if (lossMax - lossMin < 1e-9) lossMax = lossMin + 1e-9;
}

/* CSS / canvas helpers */
function getCSSColor(name) {
  return getComputedStyle(document.body).getPropertyValue(name).trim();
}
function hexRGB(name) {
  const c = getCSSColor(name);
  if (c.startsWith('#') && c.length === 7) {
    return [parseInt(c.slice(1, 3), 16),
            parseInt(c.slice(3, 5), 16),
            parseInt(c.slice(5, 7), 16)];
  }
  return [0, 0, 0];
}

/* Mouse / drag */
let mouseAbs = null, pressed = false, held = false, released = false;
let dragSrc = null;            /* 'P1' | 'P2' | 'Q' | null */
let dragCv  = null;            /* canvas where the drag started (for coord mapping) */
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

/* Pure black -> white grayscale heatmap (matches the model demo's "Clean"
 * panel — no theme-dependent tint).  Optional log-stretch via useLog. */
function logStretch(t) {
  const g = 1000;
  return Math.log1p(g * Math.max(0, t)) / Math.log1p(g);
}
let pLog = false, qLog = false;
function paintGrayscale(cv, mu, peak, useLog) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth, cssH = cv.clientHeight;
  const cw = Math.round(cssW * dpr), ch = Math.round(cssH * dpr);
  if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
  const ctx = cv.getContext('2d');

  const inv = 1.0 / peak;
  const img = ctx.createImageData(cw, ch);
  for (let cy = 0; cy < ch; cy++) {
    let i = (cy * N / ch) | 0; if (i >= N) i = N - 1;
    const row = i * N;
    for (let cx = 0; cx < cw; cx++) {
      let j = (cx * N / cw) | 0; if (j >= N) j = N - 1;
      let t = mu[row + j] * inv;
      if (useLog) t = logStretch(t);
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
  return { ctx, dpr, cssW, cssH };
}

function paintLossMap(cv) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth, cssH = cv.clientHeight;
  const cw = Math.round(cssW * dpr), ch = Math.round(cssH * dpr);
  if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
  const ctx = cv.getContext('2d');

  const bg     = hexRGB('--canvas-bg');
  const accent = hexRGB('--accent');
  const inv = 1.0 / (lossMax - lossMin);

  const img = ctx.createImageData(cw, ch);
  for (let cy = 0; cy < ch; cy++) {
    let i = (cy * N_LOSS / ch) | 0; if (i >= N_LOSS) i = N_LOSS - 1;
    const row = i * N_LOSS;
    for (let cx = 0; cx < cw; cx++) {
      let j = (cx * N_LOSS / cw) | 0; if (j >= N_LOSS) j = N_LOSS - 1;
      let t = (Lmap[row + j] - lossMin) * inv;
      if (t < 0) t = 0; else if (t > 1) t = 1;
      const idx = (cy * cw + cx) * 4;
      img.data[idx]     = (bg[0] + (accent[0] - bg[0]) * t) | 0;
      img.data[idx + 1] = (bg[1] + (accent[1] - bg[1]) * t) | 0;
      img.data[idx + 2] = (bg[2] + (accent[2] - bg[2]) * t) | 0;
      img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return { ctx, dpr, cssW, cssH };
}

/* Coordinate helpers (always w == h for our square canvases) */
function d2sx(dx, sz) { return (dx + RANGE) / (2 * RANGE) * sz; }
function d2sy(dy, sz) { return sz - (dy + RANGE) / (2 * RANGE) * sz; }
function s2dx(sx, sz) { return  sx / sz * 2 * RANGE - RANGE; }
function s2dy(sy, sz) { return -(sy / sz * 2 * RANGE - RANGE); }

/* Plot 1: P heatmap (grayscale) with red/blue source dots */
function draw_P() {
  const { ctx, dpr, cssW, cssH } = paintGrayscale(cvP, muP, pMax, pLog);
  ctx.save();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const sz = cssW;
  const RED  = 'rgb(220, 80, 80)';
  const BLUE = 'rgb(80, 130, 220)';
  const ppu  = sz / (2 * RANGE);

  const p1sx = d2sx(S.P1x, sz), p1sy = d2sy(S.P1y, sz);
  const p2sx = d2sx(S.P2x, sz), p2sy = d2sy(S.P2y, sz);

  /* Faint dashed sigma rings */
  ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.globalAlpha = 0.55;
  ctx.strokeStyle = RED;
  ctx.beginPath(); ctx.arc(p1sx, p1sy, SIGMA_PSF * ppu, 0, Math.PI*2); ctx.stroke();
  ctx.strokeStyle = BLUE;
  ctx.beginPath(); ctx.arc(p2sx, p2sy, SIGMA_PSF * ppu, 0, Math.PI*2); ctx.stroke();
  ctx.setLineDash([]); ctx.globalAlpha = 1.0;

  /* Drag detection */
  const mp = localMouse(cvP);
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

  /* Border */
  ctx.strokeStyle = getCSSColor('--border');
  ctx.strokeRect(0, 0, cssW, cssH);
  ctx.restore();

  /* Drag start */
  if (mp && pressed && !dragSrc && near) { dragSrc = near; dragCv = cvP; }
  cvP.style.cursor = (near || dragSrc === 'P1' || dragSrc === 'P2') ? 'move' : 'default';
}

/* Plot 2: Q heatmap (grayscale) with green source dot.
 * Uses pMax (the max of mu_P) as the reference peak so that low-flux Q's
 * appear visibly dimmer than P — same vmin = 0 / vmax across both plots. */
function draw_Q() {
  const GREEN = [80, 200, 110];
  const { ctx, dpr, cssW, cssH } = paintGrayscale(cvQ, muQ, pMax, qLog);
  ctx.save();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const sz = cssW;
  const ppu = sz / (2 * RANGE);
  const qsx = d2sx(S.Qx, sz), qsy = d2sy(S.Qy, sz);
  const dotColor = `rgb(${GREEN[0]},${GREEN[1]},${GREEN[2]})`;

  ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.globalAlpha = 0.55;
  ctx.strokeStyle = dotColor;
  ctx.beginPath(); ctx.arc(qsx, qsy, SIGMA_PSF * ppu, 0, Math.PI*2); ctx.stroke();
  ctx.setLineDash([]); ctx.globalAlpha = 1.0;

  const mp = localMouse(cvQ);
  let near = false;
  if (mp) {
    const d2 = (mp.x - qsx)**2 + (mp.y - qsy)**2;
    near = d2 < 121;
  }
  const r = (dragSrc === 'Q' || near) ? 7 : 5;
  ctx.fillStyle = dotColor;
  ctx.beginPath(); ctx.arc(qsx, qsy, r, 0, Math.PI*2); ctx.fill();

  ctx.strokeStyle = getCSSColor('--border');
  ctx.strokeRect(0, 0, cssW, cssH);
  ctx.restore();

  if (mp && pressed && !dragSrc && near) { dragSrc = 'Q'; dragCv = cvQ; }
  cvQ.style.cursor = (near || dragSrc === 'Q') ? 'move' : 'default';
}

/* Plot 3: loss map J(P, Q_q) with the same green dot mirrored. */
function draw_L() {
  const { ctx, dpr, cssW, cssH } = paintLossMap(cvL);
  ctx.save();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const sz = cssW;
  const ppu = sz / (2 * RANGE);
  const p1sx = d2sx(S.P1x, sz), p1sy = d2sy(S.P1y, sz);
  const p2sx = d2sx(S.P2x, sz), p2sy = d2sy(S.P2y, sz);
  const qsx  = d2sx(S.Qx,  sz), qsy  = d2sy(S.Qy,  sz);

  /* Faint p1, p2 rings to locate the basins */
  ctx.lineWidth = 1; ctx.setLineDash([2, 3]); ctx.globalAlpha = 0.45;
  ctx.strokeStyle = 'rgb(220, 80, 80)';
  ctx.beginPath(); ctx.arc(p1sx, p1sy, SIGMA_PSF * ppu, 0, Math.PI*2); ctx.stroke();
  ctx.strokeStyle = 'rgb(80, 130, 220)';
  ctx.beginPath(); ctx.arc(p2sx, p2sy, SIGMA_PSF * ppu, 0, Math.PI*2); ctx.stroke();
  ctx.setLineDash([]); ctx.globalAlpha = 1.0;

  /* Q dot (drag handle, mirrored from plot 2) */
  const mp = localMouse(cvL);
  let near = false;
  if (mp) {
    const d2 = (mp.x - qsx)**2 + (mp.y - qsy)**2;
    near = d2 < 121;
  }
  const r = (dragSrc === 'Q' || near) ? 7 : 5;
  ctx.fillStyle = 'rgb(80, 200, 110)';
  ctx.beginPath(); ctx.arc(qsx, qsy, r, 0, Math.PI*2); ctx.fill();

  ctx.strokeStyle = getCSSColor('--border');
  ctx.strokeRect(0, 0, cssW, cssH);
  ctx.restore();

  if (mp && pressed && !dragSrc && near) { dragSrc = 'Q'; dragCv = cvL; }
  cvL.style.cursor = (near || dragSrc === 'Q') ? 'move' : 'default';
}

/* Apply drag updates using the canvas where the drag started */
function apply_drag() {
  if (!dragSrc || !dragCv || !held) return;
  const mp = localMouse(dragCv);
  if (!mp) return;
  const sz = dragCv.clientWidth;
  let nx = s2dx(mp.x, sz), ny = s2dy(mp.y, sz);
  if (nx < -RANGE) nx = -RANGE; if (nx > RANGE) nx = RANGE;
  if (ny < -RANGE) ny = -RANGE; if (ny > RANGE) ny = RANGE;
  if (dragSrc === 'P1') { S.P1x = nx; S.P1y = ny; }
  if (dragSrc === 'P2') { S.P2x = nx; S.P2y = ny; }
  if (dragSrc === 'Q')  { S.Qx  = nx; S.Qy  = ny; }
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
bindSlider(document.getElementById('cost-FP1'),
           document.getElementById('cost-FP1-val'), 'FP1', 2);
bindSlider(document.getElementById('cost-FP2'),
           document.getElementById('cost-FP2-val'), 'FP2', 2);
bindSlider(document.getElementById('cost-FQ'),
           document.getElementById('cost-FQ-val'),  'FQ',  2);

const pLogBtn = document.getElementById('cost-P-log');
if (pLogBtn) pLogBtn.addEventListener('click', () => {
  pLog = !pLog; pLogBtn.classList.toggle('is-on', pLog);
});
const qLogBtn = document.getElementById('cost-Q-log');
if (qLogBtn) qLogBtn.addEventListener('click', () => {
  qLog = !qLog; qLogBtn.classList.toggle('is-on', qLog);
});

/* Main loop */
function loop() {
  apply_drag();
  compute_means();
  compute_loss_map();
  Jcurrent = J_at(S.Qx, S.Qy);

  draw_P();
  draw_Q();
  draw_L();

  const Jel = document.getElementById('cost-J-val');
  if (Jel) Jel.textContent = Jcurrent.toFixed(3);

  if (released) { dragSrc = null; dragCv = null; }
  pressed = false; released = false;
  requestAnimationFrame(loop);
}
loop();

})();
