/* ===========================================================================
 * Mini-demo for the "Best one-source explanation Q*" section.
 *
 *   Plot 1 (grayscale)    : mu_P, with TWO draggable sources p1 (red), p2 (blue)
 *   Plot 2 (grayscale)    : mu_{Q_q} = F_q* * phi(.- q), where F_q* is the
 *                           matched-filter coefficient at the user's q.
 *                           q is draggable (green dot); F_q is auto-computed.
 *   Plot 3 (amber)        : profiled loss map J*(P, q) = ||mu_P||^2
 *                           - (F_q*)^2 * pi sigma^2 over the FOV.
 *                           Green dot = user's q.  Yellow star = Q* (argmin).
 *
 * Sliders: F_P1, F_P2 only (no F_Q — auto via matched filter).
 *
 *   F_q*     = sum_i F_P_i exp(-|p_i - q|^2 / (4 sigma^2))
 *   J*(P, q) = ||mu_P||^2 - (F_q*)^2 * pi sigma^2
 * =========================================================================== */

(() => {

const cvP = document.getElementById('qstar-P');
const cvQ = document.getElementById('qstar-Q');
const cvL = document.getElementById('qstar-loss');
if (!cvP || !cvQ || !cvL) return;

const SIGMA_PSF = 1.0;
const RANGE     = 3.0;
const N         = 240;
const N_LOSS    = 120;

const S = {
  P1x: -0.6, P1y: 0.0,
  P2x: +0.6, P2y: 0.0,
  Qx:  0.0,  Qy:  1.2,
  FP1: 1.0,  FP2: 1.0,
};

const phi2 = (dx, dy) =>
  Math.exp(-(dx*dx + dy*dy) / (2.0 * SIGMA_PSF * SIGMA_PSF));

const PI_SIGMA_SQ = Math.PI * SIGMA_PSF * SIGMA_PSF;
const FOUR_SIGMA_SQ = 4.0 * SIGMA_PSF * SIGMA_PSF;

/* Matched-filter coefficient at q. */
function F_q_star(qx, qy) {
  const d1 = (S.P1x - qx)*(S.P1x - qx) + (S.P1y - qy)*(S.P1y - qy);
  const d2 = (S.P2x - qx)*(S.P2x - qx) + (S.P2y - qy)*(S.P2y - qy);
  return S.FP1 * Math.exp(-d1 / FOUR_SIGMA_SQ)
       + S.FP2 * Math.exp(-d2 / FOUR_SIGMA_SQ);
}

function muP_norm_sq() {
  const d12 = (S.P1x - S.P2x)*(S.P1x - S.P2x) + (S.P1y - S.P2y)*(S.P1y - S.P2y);
  return PI_SIGMA_SQ * (
      S.FP1 * S.FP1
    + S.FP2 * S.FP2
    + 2 * S.FP1 * S.FP2 * Math.exp(-d12 / FOUR_SIGMA_SQ)
  );
}

function J_star_at(qx, qy) {
  const Fq = F_q_star(qx, qy);
  return muP_norm_sq() - Fq * Fq * PI_SIGMA_SQ;
}

/* Buffers */
const muP   = new Float64Array(N * N);
const muQ   = new Float64Array(N * N);
const Lmap  = new Float64Array(N_LOSS * N_LOSS);

let pMax = 1, qMax = 1, lossMin = 0, lossMax = 1, JatQ = 0, FqAtQ = 0;
let qStarX = 0, qStarY = 0, qStarJ = 0;

function compute_means() {
  pMax = 1e-9; qMax = 1e-9;
  /* mu_Q at user's q uses auto F_q* */
  FqAtQ = F_q_star(S.Qx, S.Qy);
  for (let i = 0; i < N; i++) {
    const y = +RANGE - (i + 0.5) / N * 2*RANGE;
    for (let j = 0; j < N; j++) {
      const x = -RANGE + (j + 0.5) / N * 2*RANGE;
      const p = S.FP1 * phi2(x - S.P1x, y - S.P1y)
              + S.FP2 * phi2(x - S.P2x, y - S.P2y);
      const q = FqAtQ * phi2(x - S.Qx, y - S.Qy);
      muP[i * N + j] = p;
      muQ[i * N + j] = q;
      if (p > pMax) pMax = p;
      if (q > qMax) qMax = q;
    }
  }
  JatQ = J_star_at(S.Qx, S.Qy);
}

function compute_loss_map() {
  const pp = muP_norm_sq();
  lossMin = Infinity; lossMax = -Infinity;
  let bestI = 0, bestJ = 0;
  for (let i = 0; i < N_LOSS; i++) {
    const y = +RANGE - (i + 0.5) / N_LOSS * 2*RANGE;
    for (let j = 0; j < N_LOSS; j++) {
      const x = -RANGE + (j + 0.5) / N_LOSS * 2*RANGE;
      const Fq = F_q_star(x, y);
      const J  = pp - Fq * Fq * PI_SIGMA_SQ;
      Lmap[i * N_LOSS + j] = J;
      if (J < lossMin) { lossMin = J; bestI = i; bestJ = j; }
      if (J > lossMax) { lossMax = J; }
    }
  }
  if (lossMax - lossMin < 1e-9) lossMax = lossMin + 1e-9;

  /* Q* = grid argmin (sub-grid refinement skipped — N_LOSS=120 is fine
   * for visualisation). */
  qStarX = -RANGE + (bestJ + 0.5) / N_LOSS * 2*RANGE;
  qStarY = +RANGE - (bestI + 0.5) / N_LOSS * 2*RANGE;
  qStarJ = lossMin;
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
let dragSrc = null;
let dragCv  = null;
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

/* Coordinate helpers */
function d2sx(dx, sz) { return (dx + RANGE) / (2 * RANGE) * sz; }
function d2sy(dy, sz) { return sz - (dy + RANGE) / (2 * RANGE) * sz; }
function s2dx(sx, sz) { return  sx / sz * 2 * RANGE - RANGE; }
function s2dy(sy, sz) { return -(sy / sz * 2 * RANGE - RANGE); }

/* Plot 1: mu_P (grayscale) */
function draw_P() {
  const { ctx, dpr, cssW } = paintGrayscale(cvP, muP, pMax, pLog);
  ctx.save();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const sz = cssW;
  const RED  = 'rgb(220, 80, 80)';
  const BLUE = 'rgb(80, 130, 220)';
  const ppu  = sz / (2 * RANGE);

  const p1sx = d2sx(S.P1x, sz), p1sy = d2sy(S.P1y, sz);
  const p2sx = d2sx(S.P2x, sz), p2sy = d2sy(S.P2y, sz);

  ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.globalAlpha = 0.55;
  ctx.strokeStyle = RED;
  ctx.beginPath(); ctx.arc(p1sx, p1sy, SIGMA_PSF * ppu, 0, Math.PI*2); ctx.stroke();
  ctx.strokeStyle = BLUE;
  ctx.beginPath(); ctx.arc(p2sx, p2sy, SIGMA_PSF * ppu, 0, Math.PI*2); ctx.stroke();
  ctx.setLineDash([]); ctx.globalAlpha = 1.0;

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

  ctx.strokeStyle = getCSSColor('--border');
  ctx.strokeRect(0, 0, cssW, cssW);
  ctx.restore();

  if (mp && pressed && !dragSrc && near) { dragSrc = near; dragCv = cvP; }
  cvP.style.cursor = (near || dragSrc === 'P1' || dragSrc === 'P2') ? 'move' : 'default';
}

/* Plot 2: mu_Q with auto F_q* (grayscale).
 * Shares pMax with plot 1 so the brightness is comparable: when q sits in
 * a basin of mu_P, mu_Q reaches plot 1's peak; otherwise it appears dimmer. */
function draw_Q() {
  const { ctx, dpr, cssW } = paintGrayscale(cvQ, muQ, pMax, qLog);
  ctx.save();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const sz = cssW;
  const ppu = sz / (2 * RANGE);
  const qsx = d2sx(S.Qx, sz), qsy = d2sy(S.Qy, sz);
  const GREEN = 'rgb(80, 200, 110)';

  ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.globalAlpha = 0.55;
  ctx.strokeStyle = GREEN;
  ctx.beginPath(); ctx.arc(qsx, qsy, SIGMA_PSF * ppu, 0, Math.PI*2); ctx.stroke();
  ctx.setLineDash([]); ctx.globalAlpha = 1.0;

  const mp = localMouse(cvQ);
  let near = false;
  if (mp) {
    const d2 = (mp.x - qsx)**2 + (mp.y - qsy)**2;
    near = d2 < 121;
  }
  const r = (dragSrc === 'Q' || near) ? 7 : 5;
  ctx.fillStyle = GREEN;
  ctx.beginPath(); ctx.arc(qsx, qsy, r, 0, Math.PI*2); ctx.fill();

  ctx.strokeStyle = getCSSColor('--border');
  ctx.strokeRect(0, 0, cssW, cssW);
  ctx.restore();

  if (mp && pressed && !dragSrc && near) { dragSrc = 'Q'; dragCv = cvQ; }
  cvQ.style.cursor = (near || dragSrc === 'Q') ? 'move' : 'default';
}

/* Plot 3: profiled loss map J*(q), with green dot at user's q and a yellow
 * star at Q* = argmin. */
function drawStar(ctx, x, y, R, color) {
  /* Five-pointed star, centred at (x, y), outer radius R. */
  ctx.fillStyle = color;
  ctx.beginPath();
  for (let k = 0; k < 10; k++) {
    const ang = -Math.PI / 2 + k * Math.PI / 5;
    const rk  = (k % 2 === 0) ? R : R * 0.42;
    const px = x + rk * Math.cos(ang);
    const py = y + rk * Math.sin(ang);
    if (k === 0) ctx.moveTo(px, py);
    else         ctx.lineTo(px, py);
  }
  ctx.closePath();
  ctx.fill();
}

function draw_L() {
  const { ctx, dpr, cssW } = paintLossMap(cvL);
  ctx.save();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const sz = cssW;
  const ppu = sz / (2 * RANGE);
  const p1sx = d2sx(S.P1x, sz), p1sy = d2sy(S.P1y, sz);
  const p2sx = d2sx(S.P2x, sz), p2sy = d2sy(S.P2y, sz);
  const qsx  = d2sx(S.Qx,  sz), qsy  = d2sy(S.Qy,  sz);
  const qSsx = d2sx(qStarX, sz), qSsy = d2sy(qStarY, sz);

  /* Faint p1, p2 rings to locate the basins */
  ctx.lineWidth = 1; ctx.setLineDash([2, 3]); ctx.globalAlpha = 0.45;
  ctx.strokeStyle = 'rgb(220, 80, 80)';
  ctx.beginPath(); ctx.arc(p1sx, p1sy, SIGMA_PSF * ppu, 0, Math.PI*2); ctx.stroke();
  ctx.strokeStyle = 'rgb(80, 130, 220)';
  ctx.beginPath(); ctx.arc(p2sx, p2sy, SIGMA_PSF * ppu, 0, Math.PI*2); ctx.stroke();
  ctx.setLineDash([]); ctx.globalAlpha = 1.0;

  /* Yellow star at Q* — drawn first so the user's green dot sits on top */
  drawStar(ctx, qSsx, qSsy, 9, 'rgb(245, 200, 70)');
  ctx.strokeStyle = 'rgba(20, 20, 28, 0.55)';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  for (let k = 0; k < 10; k++) {
    const ang = -Math.PI / 2 + k * Math.PI / 5;
    const rk  = (k % 2 === 0) ? 9 : 9 * 0.42;
    const px = qSsx + rk * Math.cos(ang);
    const py = qSsy + rk * Math.sin(ang);
    if (k === 0) ctx.moveTo(px, py);
    else         ctx.lineTo(px, py);
  }
  ctx.closePath();
  ctx.stroke();

  /* Green dot at user's q */
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
  ctx.strokeRect(0, 0, cssW, cssW);
  ctx.restore();

  if (mp && pressed && !dragSrc && near) { dragSrc = 'Q'; dragCv = cvL; }
  cvL.style.cursor = (near || dragSrc === 'Q') ? 'move' : 'default';
}

/* Drag updates */
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
bindSlider(document.getElementById('qstar-FP1'),
           document.getElementById('qstar-FP1-val'), 'FP1', 2);
bindSlider(document.getElementById('qstar-FP2'),
           document.getElementById('qstar-FP2-val'), 'FP2', 2);

const pLogBtn = document.getElementById('qstar-P-log');
if (pLogBtn) pLogBtn.addEventListener('click', () => {
  pLog = !pLog; pLogBtn.classList.toggle('is-on', pLog);
});
const qLogBtn = document.getElementById('qstar-Q-log');
if (qLogBtn) qLogBtn.addEventListener('click', () => {
  qLog = !qLog; qLogBtn.classList.toggle('is-on', qLog);
});

/* Main loop */
function loop() {
  apply_drag();
  compute_means();
  compute_loss_map();

  draw_P();
  draw_Q();
  draw_L();

  const fEl    = document.getElementById('qstar-Fq-val');
  const jEl    = document.getElementById('qstar-J-val');
  const jMinEl = document.getElementById('qstar-Jmin-val');
  if (fEl)    fEl.textContent    = FqAtQ.toFixed(3);
  if (jEl)    jEl.textContent    = JatQ.toFixed(3);
  if (jMinEl) jMinEl.textContent = qStarJ.toFixed(3);

  if (released) { dragSrc = null; dragCv = null; }
  pressed = false; released = false;
  requestAnimationFrame(loop);
}
loop();

})();
