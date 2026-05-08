/* ===========================================================================
 * 2D Gaussian contamination — interactive companion
 *
 * Mirrors the C demo's math (gaussian_demo.c) for the 2D Gaussian PSF case:
 *   - kernel: k(t) = exp(-||t||^2 / (4 sigma^2))
 *   - matched filter q* on the segment [p1, p2] (Gaussian FOC)
 *   - D = min over P_s of J(P_s, Q*) - J(P, Q*)
 *   - regions in (Delta, R) plane with R = SNR_P / SNR_thresh, eta = 5 sigma
 *   - clean and pixelated+noise images
 * =========================================================================== */

(() => {

const SIGMA_PSF  = 1.0;
const SNR_THRESH = 5.0;
const DELTA_C    = 2.836912;          /* Theorem 6.11 */
const PLOT_RANGE = 2.5 * DELTA_C;     /* universal half-extent: 7.092 */

/* Kernel (analytic for Gaussian, no table needed) */
const k    = (t) => Math.exp(-t*t / (4.0 * SIGMA_PSF * SIGMA_PSF));
const kd1  = (t) => -t / (2.0 * SIGMA_PSF * SIGMA_PSF) * k(t);
const kd2  = (t) => {
  const s2 = SIGMA_PSF * SIGMA_PSF;
  return (t*t / (4.0 * s2 * s2) - 1.0 / (2.0 * s2)) * k(t);
};
const phi2 = (dx, dy) => Math.exp(-(dx*dx + dy*dy) / (2.0 * SIGMA_PSF * SIGMA_PSF));

/* ----------------- State (driven by sliders) ----------------- */
const S = {
  p1x: -0.5, p1y:  0.0,
  p2x:  0.5, p2y:  0.0,
  F1: 1.0, F2: 1.0,
  sigma: 0.10,
};

/* Derived per frame */
const D = {
  delta: 0,
  uvalue: 0, vvalue: 0,            /* segment-1D positions */
  qx: 0, qy: 0,
  Fstar: 0,
  c1: 0, c2: 0, r: 0,
  J1: 0, J2: 0,
  J_PQ: 0, J_emp: 0, J_p1: 0, J_p2: 0,
  Dval: 0, Dbar: 0,
  cond_lhs: 0, cond_rhs: 0,
};

/* Find q* by 1D coarse search + Newton along the segment p1->p2.
 * The segment is parametrised by t in [-, +] (we let t freely range and
 * the FOC pins it to [0, Delta] for monotone radial Gaussian). */
function find_q_star_segment(delta) {
  /* Coarse 201-point scan over [-5, delta+5] to seed Newton */
  const lo = -5.0, hi = delta + 5.0;
  const n  = 201;
  let bestT = lo, bestF = S.F1*k(lo) + S.F2*k(delta - lo);
  for (let i = 1; i < n; i++) {
    const t = lo + (hi - lo) * i / (n - 1);
    const f = S.F1*k(t) + S.F2*k(delta - t);
    if (f > bestF) { bestF = f; bestT = t; }
  }
  /* Newton refinement */
  function newton_one(t0) {
    let t = t0;
    for (let it = 0; it < 50; it++) {
      const g = S.F1*kd1(t) - S.F2*kd1(delta - t);
      const h = S.F1*kd2(t) + S.F2*kd2(delta - t);
      let step = h < -1e-12 ? -g/h : 0.05*g;
      if (step >  1.0) step =  1.0;
      if (step < -1.0) step = -1.0;
      t += step;
      if (Math.abs(g) < 1e-12) break;
    }
    return t;
  }
  let bt = newton_one(bestT);
  let bf = S.F1*k(bt) + S.F2*k(delta - bt);
  if (bf > bestF) { bestF = bf; bestT = bt; }
  /* 5 starts inside [0, delta] */
  for (let kk = 0; kk < 5; kk++) {
    const t0 = (delta) * kk / 4.0;
    const t1 = newton_one(t0);
    const f1 = S.F1*k(t1) + S.F2*k(delta - t1);
    if (f1 > bestF) { bestF = f1; bestT = t1; }
  }
  return bestT;
}

/* Recompute everything for current state */
function recompute() {
  const dx = S.p2x - S.p1x, dy = S.p2y - S.p1y;
  D.delta = Math.sqrt(dx*dx + dy*dy);
  if (D.delta < 1e-9) D.delta = 1e-9;

  /* q* on segment, parametrised by scalar t in [0, delta] */
  const t = find_q_star_segment(D.delta);
  D.uvalue = t; D.vvalue = D.delta - t;

  /* Lift back to 2D */
  const frac = t / D.delta;
  D.qx = S.p1x + frac * dx;
  D.qy = S.p1y + frac * dy;

  /* Correlations and J's */
  D.c1 = k(t);
  D.c2 = k(D.delta - t);
  D.r  = k(D.delta);
  D.Fstar = S.F1 * D.c1 + S.F2 * D.c2;
  D.J1 = S.F1 * S.F1;
  D.J2 = S.F2 * S.F2;
  const muP_sq = D.J1 + D.J2 + 2*S.F1*S.F2*D.r;
  D.J_PQ  = muP_sq - D.Fstar*D.Fstar;
  D.J_emp = D.Fstar * D.Fstar;
  D.J_p1  = D.J1 - 2*S.F1*D.Fstar*D.c1 + D.Fstar*D.Fstar;
  D.J_p2  = D.J2 - 2*S.F2*D.Fstar*D.c2 + D.Fstar*D.Fstar;
  const minPs = Math.min(D.J_emp, D.J_p1, D.J_p2);
  D.Dval = minPs - D.J_PQ;
  D.Dbar = D.Dval / (2*Math.max(D.J1,1e-30)) + D.Dval / (2*Math.max(D.J2,1e-30));

  /* FCC condition (WLOG F1 >= F2) */
  let FA = S.F1, FB = S.F2, cA = D.c1, cB = D.c2;
  if (FB > FA) { [FA,FB] = [FB,FA]; [cA,cB] = [cB,cA]; }
  D.cond_lhs = FA * (cA*cB - D.r);
  D.cond_rhs = FB * (1 - cB*cB);
}

/* Helper: compute D at arbitrary separation for the D plot.
 * Saves and restores the global state. */
function compute_D_at_separation(delta) {
  const sx1 = S.p1x, sy1 = S.p1y, sx2 = S.p2x, sy2 = S.p2y;
  S.p1x = 0; S.p1y = 0; S.p2x = delta; S.p2y = 0;
  const t = find_q_star_segment(delta);
  const c1 = k(t), c2 = k(delta - t), r = k(delta);
  const Fs = S.F1*c1 + S.F2*c2;
  const J1 = S.F1*S.F1, J2 = S.F2*S.F2;
  const muP_sq = J1 + J2 + 2*S.F1*S.F2*r;
  const JPQ = muP_sq - Fs*Fs;
  const Je  = Fs*Fs;
  const Jp1 = J1 - 2*S.F1*Fs*c1 + Fs*Fs;
  const Jp2 = J2 - 2*S.F2*Fs*c2 + Fs*Fs;
  const Dret = Math.min(Je, Jp1, Jp2) - JPQ;
  S.p1x = sx1; S.p1y = sy1; S.p2x = sx2; S.p2y = sy2;
  return { D: Dret, JPQ, Je, Jp1, Jp2, muP_sq };
}

/* ====================== Canvas helpers ====================== */
function getCSSColor(name) {
  return getComputedStyle(document.body).getPropertyValue(name).trim();
}

/* Crisp DPI handling: set canvas pixel size = backing × dpr */
function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth, cssH = canvas.clientHeight;
  if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
    canvas.width  = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: cssW, h: cssH };
}

/* ====================== Top-view (2D) ====================== */
const cvTop = document.getElementById('canvas-topview');
let dragging = -1;       /* -1: none, 0: p1, 1: p2 */

function draw_topview() {
  const { ctx, w, h } = setupCanvas(cvTop);

  /* Square inset, centred horizontally */
  const sz = Math.min(w, h);
  const x0 = (w - sz) / 2;
  const y0 = (h - sz) / 2;
  const range = PLOT_RANGE;
  const data2screen_x = (dx) => x0 + (dx + range) / (2*range) * sz;
  const data2screen_y = (dy) => y0 + sz - (dy + range) / (2*range) * sz;
  const screen2data_x = (sx) => (sx - x0) / sz * 2*range - range;
  const screen2data_y = (sy) => -((sy - y0) / sz * 2*range - range);

  /* Clear the entire canvas first (to avoid stale pixels in the empty
   * margin if w !== h), then paint the data background over the inset. */
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = getCSSColor('--canvas-bg');
  ctx.fillRect(x0, y0, sz, sz);

  /* Gridlines at integers */
  ctx.strokeStyle = getCSSColor('--canvas-grid');
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let t = Math.ceil(-range); t <= Math.floor(range); t++) {
    if (t === 0) continue;
    const gx = data2screen_x(t), gy = data2screen_y(t);
    ctx.moveTo(gx, y0); ctx.lineTo(gx, y0+sz);
    ctx.moveTo(x0, gy); ctx.lineTo(x0+sz, gy);
  }
  ctx.stroke();

  /* Origin axes */
  ctx.strokeStyle = getCSSColor('--canvas-axis');
  const ox = data2screen_x(0), oy = data2screen_y(0);
  ctx.beginPath();
  ctx.moveTo(ox, y0); ctx.lineTo(ox, y0+sz);
  ctx.moveTo(x0, oy); ctx.lineTo(x0+sz, oy);
  ctx.stroke();

  /* Border */
  ctx.strokeStyle = getCSSColor('--border');
  ctx.strokeRect(x0, y0, sz, sz);

  const pix_per_unit = sz / (2*range);
  const p1sx = data2screen_x(S.p1x), p1sy = data2screen_y(S.p1y);
  const p2sx = data2screen_x(S.p2x), p2sy = data2screen_y(S.p2y);
  const qsx  = data2screen_x(D.qx),  qsy  = data2screen_y(D.qy);

  /* Segment line */
  ctx.strokeStyle = getCSSColor('--canvas-axis');
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(p1sx, p1sy); ctx.lineTo(p2sx, p2sy);
  ctx.stroke();

  /* Concentric Gaussian isocontour circles for p1 (red) and p2 (blue),
   * plus dashed for q* (green). */
  const levels = [0.5, 0.25, 0.125, 0.0625];
  const red    = getCSSColor('--plot-red');
  const blue   = getCSSColor('--plot-blue');
  const green  = getCSSColor('--plot-green');
  ctx.lineWidth = 1;
  for (const L of levels) {
    const r_data = SIGMA_PSF * Math.sqrt(2 * Math.log(1 / L));
    const r_pix  = r_data * pix_per_unit;
    const a = 0.35 + 0.55 * (L / levels[0]);

    /* p1 circle */
    ctx.strokeStyle = red;
    ctx.globalAlpha = a;
    ctx.beginPath(); ctx.arc(p1sx, p1sy, r_pix, 0, Math.PI*2); ctx.stroke();

    /* p2 circle */
    ctx.strokeStyle = blue;
    ctx.beginPath(); ctx.arc(p2sx, p2sy, r_pix, 0, Math.PI*2); ctx.stroke();

    /* q* circle (dashed) */
    ctx.strokeStyle = green;
    ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.arc(qsx, qsy, r_pix, 0, Math.PI*2); ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.globalAlpha = 1.0;

  /* Source dots */
  function dot(sx, sy, r, color) {
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(sx, sy, r, 0, Math.PI*2); ctx.fill();
  }
  /* Hover effect for drag handles */
  const mp = localMouse(cvTop);
  let r1 = 5, r2 = 5;
  if (mp) {
    const d1 = (mp.x - p1sx)**2 + (mp.y - p1sy)**2;
    const d2 = (mp.x - p2sx)**2 + (mp.y - p2sy)**2;
    if (dragging === 0 || d1 < 100) r1 = 7;
    if (dragging === 1 || d2 < 100) r2 = 7;
  }
  dot(p1sx, p1sy, r1, red);
  dot(p2sx, p2sy, r2, blue);
  dot(qsx,  qsy,  4,  green);

  /* Labels */
  ctx.font = '12px ' + getCSSColor('--font-ui');
  ctx.fillStyle = red;   ctx.fillText('p1', p1sx + 8, p1sy - 6);
  ctx.fillStyle = blue;  ctx.fillText('p2', p2sx + 8, p2sy - 6);
  ctx.fillStyle = green; ctx.fillText('q*', qsx + 8,  qsy - 6);

  /* Δ label at midpoint */
  ctx.fillStyle = getCSSColor('--text-soft');
  ctx.font = '11px ' + getCSSColor('--font-ui');
  const mx = (p1sx + p2sx) / 2, my = (p1sy + p2sy) / 2;
  ctx.fillText(`Δ = ${D.delta.toFixed(3)}`, mx + 10, my - 14);

  /* Mouse interactions */
  if (mp && pressed && dragging === -1
      && mp.x >= x0 && mp.x <= x0+sz && mp.y >= y0 && mp.y <= y0+sz) {
    const d1 = (mp.x - p1sx)**2 + (mp.y - p1sy)**2;
    const d2 = (mp.x - p2sx)**2 + (mp.y - p2sy)**2;
    if (d1 < 100)      dragging = 0;
    else if (d2 < 100) dragging = 1;
  }
  if (mp && held && dragging >= 0) {
    const nx = Math.max(-range, Math.min(range, screen2data_x(mp.x)));
    const ny = Math.max(-range, Math.min(range, screen2data_y(mp.y)));
    if (dragging === 0) { S.p1x = nx; S.p1y = ny;
                          document.getElementById('slider-p1x').value = nx;
                          document.getElementById('slider-p1y').value = ny;
                          updateValueLabels(); }
    else                { S.p2x = nx; S.p2y = ny;
                          document.getElementById('slider-p2x').value = nx;
                          document.getElementById('slider-p2y').value = ny;
                          updateValueLabels(); }
  }
  if (released) dragging = -1;
}

/* ====================== D vs Δ plot ====================== */
const cvD = document.getElementById('canvas-dplot');
const DP_N = 100;
const dpDeltas = new Array(DP_N + 1);
const dpDvals  = new Array(DP_N + 1);

function sample_D_curve() {
  for (let i = 0; i <= DP_N; i++) {
    const delta = (i / DP_N) * PLOT_RANGE;
    dpDeltas[i] = delta;
    dpDvals[i]  = compute_D_at_separation(delta).D;
  }
}

function draw_dplot() {
  const { ctx, w, h } = setupCanvas(cvD);
  ctx.fillStyle = getCSSColor('--canvas-bg');
  ctx.fillRect(0, 0, w, h);

  /* Inner */
  const padL = 36, padR = 10, padT = 14, padB = 22;
  const px = padL, py = padT, pw = w - padL - padR, ph = h - padT - padB;

  /* Range */
  let dmax = 0, dmin = 0;
  for (const v of dpDvals) { if (v > dmax) dmax = v; if (v < dmin) dmin = v; }
  let ymax = dmax * 1.15, ymin = dmin * 1.15;
  if (ymax < 0.05) ymax = 0.05;
  if (ymin > -0.05) ymin = -0.05;

  /* Zero line */
  const zero_y = py + ph - (0 - ymin)/(ymax - ymin) * ph;
  ctx.strokeStyle = getCSSColor('--canvas-axis');
  ctx.beginPath(); ctx.moveTo(px, zero_y); ctx.lineTo(px+pw, zero_y); ctx.stroke();

  /* Border */
  ctx.strokeStyle = getCSSColor('--border');
  ctx.strokeRect(px, py, pw, ph);

  /* Find first sign change (Delta_c) */
  let dc = -1;
  for (let i = 1; i <= DP_N; i++) {
    if (dpDvals[i-1] > 0 && dpDvals[i] <= 0) {
      const f = dpDvals[i-1] / (dpDvals[i-1] - dpDvals[i]);
      dc = dpDeltas[i-1] + f * (dpDeltas[i] - dpDeltas[i-1]);
      break;
    }
  }

  /* Curve coloured by sign */
  const red   = getCSSColor('--plot-red');
  const green = getCSSColor('--plot-green');
  ctx.lineWidth = 2;
  for (let i = 1; i <= DP_N; i++) {
    const x0 = px + (dpDeltas[i-1] / PLOT_RANGE) * pw;
    const x1 = px + (dpDeltas[i]   / PLOT_RANGE) * pw;
    const y0 = py + ph - (dpDvals[i-1] - ymin)/(ymax - ymin) * ph;
    const y1 = py + ph - (dpDvals[i]   - ymin)/(ymax - ymin) * ph;
    ctx.strokeStyle = (0.5*(dpDvals[i-1] + dpDvals[i]) > 0) ? red : green;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  }

  /* Vertical at current Delta */
  const cx = px + (D.delta / PLOT_RANGE) * pw;
  if (cx >= px && cx <= px+pw) {
    ctx.strokeStyle = getCSSColor('--text');
    ctx.globalAlpha = 0.7;
    ctx.beginPath(); ctx.moveTo(cx, py); ctx.lineTo(cx, py+ph); ctx.stroke();
    ctx.globalAlpha = 1.0;
  }

  /* Vertical at Delta_c */
  if (dc > 0) {
    const dx = px + (dc / PLOT_RANGE) * pw;
    ctx.strokeStyle = getCSSColor('--plot-yellow');
    ctx.globalAlpha = 0.85;
    ctx.beginPath(); ctx.moveTo(dx, py); ctx.lineTo(dx, py+ph); ctx.stroke();
    ctx.globalAlpha = 1.0;
    ctx.fillStyle = getCSSColor('--plot-yellow');
    ctx.font = '10px ' + getCSSColor('--font-mono');
    ctx.fillText(`Δc=${dc.toFixed(3)}`, dx + 4, py + ph - 4);
  }

  /* Axis labels */
  ctx.fillStyle = getCSSColor('--text-soft');
  ctx.font = '10px ' + getCSSColor('--font-mono');
  ctx.fillText('0',                 px - 4,  py + ph + 14);
  ctx.fillText(PLOT_RANGE.toFixed(2), px + pw - 24, py + ph + 14);
  ctx.fillText(ymax.toFixed(2),     2,       py + 8);
  ctx.fillText(ymin.toFixed(2),     2,       py + ph - 4);
  ctx.fillText('0',                 18,      zero_y - 2);
}

/* ====================== Classification plot ====================== */
const cvCP = document.getElementById('canvas-classify');
const CP_N = 100;
const Rd = new Array(CP_N + 1), Rr = new Array(CP_N + 1), Rc = new Array(CP_N + 1);

function sample_R_thresholds() {
  for (let i = 0; i <= CP_N; i++) {
    const delta = (i / CP_N) * PLOT_RANGE;
    const x = compute_D_at_separation(delta);
    const denom_d = x.muP_sq - x.JPQ;
    const minPs = Math.min(x.Je, x.Jp1, x.Jp2);
    const R_d_sq = denom_d > 1e-12 ? x.muP_sq / denom_d : 1e30;
    const R_r_sq = x.JPQ   > 1e-12 ? x.muP_sq / x.JPQ   : 1e30;
    const R_c_sq = minPs   > 1e-12 ? x.muP_sq / minPs   : 1e30;
    Rd[i] = Math.sqrt(R_d_sq);
    Rr[i] = Math.sqrt(R_r_sq);
    Rc[i] = Math.sqrt(R_c_sq);
  }
}

function draw_classification() {
  const { ctx, w, h } = setupCanvas(cvCP);
  ctx.fillStyle = getCSSColor('--canvas-bg');
  ctx.fillRect(0, 0, w, h);

  const padL = 36, padR = 10, padT = 14, padB = 22;
  const px = padL, py = padT, pw = w - padL - padR, ph = h - padT - padB;

  const R_min = 0.4, R_max = 10;
  const lr_min = Math.log10(R_min), lr_max = Math.log10(R_max);

  const colGrey  = getCSSColor('--plot-grey');
  const colBlue  = getCSSColor('--plot-blue');
  const colRed   = getCSSColor('--plot-red');
  const colGreen = getCSSColor('--plot-green');

  /* Per-pixel column fill */
  ctx.globalAlpha = 0.85;
  for (let xp = px; xp < px + pw; xp++) {
    const dx = (xp - px) / pw * PLOT_RANGE;
    const idxF = (dx / PLOT_RANGE) * CP_N;
    let i_lo = Math.floor(idxF);
    if (i_lo < 0) i_lo = 0; if (i_lo > CP_N - 1) i_lo = CP_N - 1;
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

    /* GREY */
    ctx.fillStyle = colGrey;
    ctx.fillRect(xp, yd, 1, yBot - yd);
    /* Middle */
    if (Rcp < Rrp) {
      ctx.fillStyle = colBlue;  ctx.fillRect(xp, yc, 1, yd - yc);
      ctx.fillStyle = colRed;   ctx.fillRect(xp, yr, 1, yc - yr);
    } else {
      ctx.fillStyle = colBlue;  ctx.fillRect(xp, yr, 1, yd - yr);
    }
    /* GREEN */
    ctx.fillStyle = colGreen;
    ctx.fillRect(xp, yTop, 1, yr - yTop);
  }
  ctx.globalAlpha = 1.0;

  /* Border */
  ctx.strokeStyle = getCSSColor('--border');
  ctx.strokeRect(px, py, pw, ph);

  /* D=0 vertical line (where Rc crosses Rr) */
  let dc = -1;
  for (let i = 1; i <= CP_N; i++) {
    if (Rc[i-1] < Rr[i-1] && Rc[i] >= Rr[i]) {
      const f = (Rc[i-1] - Rr[i-1]) / ((Rr[i] - Rr[i-1]) - (Rc[i] - Rc[i-1]));
      dc = (i-1)/CP_N*PLOT_RANGE + f * (1/CP_N*PLOT_RANGE);
      break;
    }
  }
  if (dc > 0) {
    const dx2 = px + (dc / PLOT_RANGE) * pw;
    ctx.strokeStyle = getCSSColor('--plot-yellow');
    ctx.globalAlpha = 0.85;
    ctx.beginPath(); ctx.moveTo(dx2, py); ctx.lineTo(dx2, py+ph); ctx.stroke();
    ctx.globalAlpha = 1.0;
  }

  /* Yellow dot at (delta, R), draggable.
   * - Horizontal drag changes Delta by rescaling p1, p2 about their midpoint.
   * - Vertical drag changes R, which rescales sigma_noise via R = ||mu_P||/(sigma * eta). */
  const muP_sq = D.J1 + D.J2 + 2*S.F1*S.F2*D.r;
  const safe_sigma = Math.max(S.sigma, 1e-6);
  const R_now = Math.sqrt(muP_sq) / (safe_sigma * SNR_THRESH);
  let xc = -1, yc = -1;
  if (D.delta >= 0 && D.delta <= PLOT_RANGE) {
    xc = px + (D.delta / PLOT_RANGE) * pw;
    let lr = Math.log10(R_now);
    if (lr < lr_min) lr = lr_min;
    if (lr > lr_max) lr = lr_max;
    yc = py + ph - (lr - lr_min)/(lr_max - lr_min) * ph;

    /* Hover halo */
    const mpcp = localMouse(cvCP);
    const hovered = mpcp
                 && Math.hypot(mpcp.x - xc, mpcp.y - yc) < 14;
    const r_outer = (hovered || draggingClassDot) ? 10 : 7;
    const r_inner = (hovered || draggingClassDot) ?  6 : 5;
    ctx.fillStyle = getCSSColor('--plot-yellow');
    ctx.beginPath(); ctx.arc(xc, yc, r_inner, 0, Math.PI*2); ctx.fill();
    ctx.strokeStyle = getCSSColor('--plot-yellow');
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(xc, yc, r_outer, 0, Math.PI*2); ctx.stroke();

    /* Cursor feedback */
    cvCP.style.cursor = (hovered || draggingClassDot) ? 'move' : 'default';

    /* Drag interaction */
    if (pressed && hovered) draggingClassDot = true;
    if (released)           draggingClassDot = false;
    if (draggingClassDot && held && mpcp) {
      /* Mouse, clamped to plot interior */
      let mx = Math.max(px, Math.min(px + pw, mpcp.x));
      let my = Math.max(py, Math.min(py + ph, mpcp.y));

      /* Inverse maps */
      const newDelta = (mx - px) / pw * PLOT_RANGE;
      const newLr    = (py + ph - my) / ph * (lr_max - lr_min) + lr_min;
      const newR     = Math.pow(10, newLr);

      /* (x): scale (p1, p2) symmetrically about their midpoint to give newDelta */
      const dxp = S.p2x - S.p1x, dyp = S.p2y - S.p1y;
      const len = Math.hypot(dxp, dyp);
      let ux = 1, uy = 0;
      if (len > 1e-9) { ux = dxp/len; uy = dyp/len; }
      const cxm = 0.5*(S.p1x + S.p2x), cym = 0.5*(S.p1y + S.p2y);
      S.p1x = cxm - 0.5*newDelta*ux;
      S.p1y = cym - 0.5*newDelta*uy;
      S.p2x = cxm + 0.5*newDelta*ux;
      S.p2y = cym + 0.5*newDelta*uy;
      /* Clamp to slider range */
      const RNG = PLOT_RANGE;
      S.p1x = Math.max(-RNG, Math.min(RNG, S.p1x));
      S.p1y = Math.max(-RNG, Math.min(RNG, S.p1y));
      S.p2x = Math.max(-RNG, Math.min(RNG, S.p2x));
      S.p2y = Math.max(-RNG, Math.min(RNG, S.p2y));

      /* (y): set sigma to give the new R (using the new Delta's ||mu_P||) */
      const new_r_kernel = Math.exp(-newDelta*newDelta / (4*SIGMA_PSF*SIGMA_PSF));
      const muP_sq_new = S.F1*S.F1 + S.F2*S.F2 + 2*S.F1*S.F2*new_r_kernel;
      let new_sigma = Math.sqrt(muP_sq_new) / (Math.max(newR, 1e-6) * SNR_THRESH);
      new_sigma = Math.max(0.001, Math.min(1.0, new_sigma));
      S.sigma = new_sigma;

      /* Sync sliders + labels */
      document.getElementById('slider-p1x').value   = S.p1x;
      document.getElementById('slider-p1y').value   = S.p1y;
      document.getElementById('slider-p2x').value   = S.p2x;
      document.getElementById('slider-p2y').value   = S.p2y;
      document.getElementById('slider-sigma').value = S.sigma;
      updateValueLabels();
    }
  }

  /* Axis labels */
  ctx.fillStyle = getCSSColor('--text-soft');
  ctx.font = '10px ' + getCSSColor('--font-mono');
  ctx.fillText('10',  4,  py + 8);
  const y1 = py + ph - (0 - lr_min)/(lr_max - lr_min) * ph;
  ctx.fillText('1',   8,  y1 - 2);
  ctx.fillText('0.4', 4,  py + ph - 2);
  ctx.fillText('0',   px - 4,  py + ph + 14);
  ctx.fillText(PLOT_RANGE.toFixed(2), px + pw - 24, py + ph + 14);
}

/* ====================== Clean and noisy images ====================== */
const cvClean = document.getElementById('canvas-clean');
const cvNoisy = document.getElementById('canvas-noisy');
const PI_FINE = 240, PI_PIX = 20;   /* clean grid 4x denser; detector still 20x20 */
const muFine = new Float64Array(PI_FINE * PI_FINE);
const muPix  = new Float64Array(PI_PIX  * PI_PIX);

/* Noise samples — drawn once, reused */
const noiseSamples = new Float64Array(PI_PIX * PI_PIX);
let noiseInit = false;
function gaussRandom() {
  const u1 = Math.random() + 1e-12;
  const u2 = Math.random();
  return Math.sqrt(-2*Math.log(u1)) * Math.cos(2*Math.PI*u2);
}
function ensureNoise() {
  if (noiseInit) return;
  for (let i = 0; i < PI_PIX*PI_PIX; i++) noiseSamples[i] = gaussRandom();
  noiseInit = true;
}

function compute_images() {
  /* Use full PLOT_RANGE for the image bounds (matches the C demo's
   * "all plots scaled to 2.5 * Delta_c" convention). */
  const range = PLOT_RANGE;
  let muMax = 0;
  /* i = 0 corresponds to the TOP of the screen (highest y). */
  for (let i = 0; i < PI_FINE; i++) {
    const y = +range - (i + 0.5) / PI_FINE * 2*range;
    for (let j = 0; j < PI_FINE; j++) {
      const x = -range + (j + 0.5) / PI_FINE * 2*range;
      const v = S.F1 * phi2(x - S.p1x, y - S.p1y)
              + S.F2 * phi2(x - S.p2x, y - S.p2y);
      muFine[i*PI_FINE + j] = v;
      if (v > muMax) muMax = v;
    }
  }
  if (muMax < 1e-9) muMax = 1e-9;
  const block = PI_FINE / PI_PIX;
  for (let pi = 0; pi < PI_PIX; pi++) {
    for (let pj = 0; pj < PI_PIX; pj++) {
      let s = 0;
      for (let di = 0; di < block; di++) {
        for (let dj = 0; dj < block; dj++) {
          s += muFine[(pi*block + di)*PI_FINE + (pj*block + dj)];
        }
      }
      muPix[pi*PI_PIX + pj] = s / (block*block);
    }
  }
  return muMax;
}

function draw_clean(muMax) {
  /* Render at native canvas pixel resolution via ImageData — at 240x240
   * sample density the cells are sub-pixel small so fillRect would alias;
   * direct pixel writes give a clean image. */
  const cv = cvClean;
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth, cssH = cv.clientHeight;
  const cw = Math.round(cssW * dpr), ch = Math.round(cssH * dpr);
  if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
  const ctx = cv.getContext('2d');

  const img = ctx.createImageData(cw, ch);
  const inv = 1 / muMax;
  for (let cy = 0; cy < ch; cy++) {
    let i = (cy * PI_FINE / ch) | 0;
    if (i >= PI_FINE) i = PI_FINE - 1;
    const row = i * PI_FINE;
    for (let cx = 0; cx < cw; cx++) {
      let j = (cx * PI_FINE / cw) | 0;
      if (j >= PI_FINE) j = PI_FINE - 1;
      let v = muFine[row + j] * inv;
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

  /* Border in CSS coords */
  ctx.save();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.strokeStyle = getCSSColor('--border');
  ctx.strokeRect(0, 0, cssW, cssH);
  ctx.restore();
}

function draw_noisy(muMax) {
  ensureNoise();
  const { ctx, w, h } = setupCanvas(cvNoisy);
  ctx.fillStyle = getCSSColor('--canvas-bg');
  ctx.fillRect(0, 0, w, h);
  const cell = w / PI_PIX;
  for (let pi = 0; pi < PI_PIX; pi++) {
    for (let pj = 0; pj < PI_PIX; pj++) {
      const val = muPix[pi*PI_PIX + pj] + S.sigma * noiseSamples[pi*PI_PIX + pj];
      let v = val / muMax;
      if (v < 0) v = 0; if (v > 1) v = 1;
      const g = (v * 255) | 0;
      ctx.fillStyle = `rgb(${g},${g},${g})`;
      ctx.fillRect(pj*cell, pi*cell, cell + 0.5, cell + 0.5);
    }
  }
  ctx.strokeStyle = getCSSColor('--border');
  ctx.strokeRect(0, 0, w, h);
}

/* ====================== Stats panel ====================== */
function update_stats() {
  const minPs = Math.min(D.J_emp, D.J_p1, D.J_p2);
  const min_label = D.J_emp <= D.J_p1 && D.J_emp <= D.J_p2 ? '∅'
                   : D.J_p1 <= D.J_p2 ? 'p1' : 'p2';
  const dColor = D.Dval > 0 ? 'verdict-bad' : 'verdict-good';
  const dLabel = D.Dval > 0 ? 'contamination POSSIBLE' : 'no contamination';
  const condHolds = (D.cond_rhs - D.cond_lhs) >= 0;

  document.getElementById('stats').innerHTML = `
    <div class="stat-row"><span>q*  </span><span>(${D.qx.toFixed(3)}, ${D.qy.toFixed(3)})</span></div>
    <div class="stat-row"><span>F*  </span><span>${D.Fstar.toFixed(4)}</span></div>
    <div class="stat-row"><span>Δ   </span><span>${D.delta.toFixed(4)}</span></div>
    <div class="stat-row"><span>Δc  </span><span>${DELTA_C.toFixed(4)} σ</span></div>
    <div class="stat-row"><span>min P_s</span><span>at ${min_label} (${minPs.toFixed(4)})</span></div>
    <div class="stat-row"><span>D    </span><span class="${dColor}">${D.Dval.toFixed(4)}</span></div>
    <div class="stat-row"><span>D̄    </span><span class="${dColor}">${D.Dbar.toFixed(4)}</span></div>
    <div class="stat-row"><span>     </span><span class="${dColor}">${dLabel}</span></div>
    <div class="stat-row" style="margin-top:6px"><span>FCC margin</span><span>${(D.cond_rhs - D.cond_lhs).toFixed(4)}</span></div>
    <div class="stat-row"><span>           </span><span class="${condHolds ? 'verdict-good' : 'verdict-bad'}">${condHolds ? 'holds' : 'fails'}</span></div>
  `;
}

/* ====================== Main loop ====================== */
let pressed = false, held = false, released = false;
let mouseAbs = null;          /* page-absolute mouse coords */
let mouse    = null;          /* alias of localMouse(cvTop) used by topview */
let draggingClassDot = false; /* are we dragging the yellow R-vs-Delta dot? */

/* Translate the absolute mouse coords into a given canvas's local space. */
function localMouse(canvas) {
  if (!mouseAbs) return null;
  const rect = canvas.getBoundingClientRect();
  return { x: mouseAbs.x - rect.left, y: mouseAbs.y - rect.top };
}

function loop() {
  recompute();
  sample_D_curve();
  sample_R_thresholds();
  const muMax = compute_images();

  draw_topview();
  draw_dplot();
  draw_classification();
  draw_clean(muMax);
  draw_noisy(muMax);
  update_stats();

  pressed = false; released = false;
  requestAnimationFrame(loop);
}

/* ====================== Sliders + input wiring ====================== */
function bindSlider(id, key, decimals = 2) {
  const el = document.getElementById('slider-' + id);
  el.addEventListener('input', () => {
    S[key] = parseFloat(el.value);
    updateValueLabels();
  });
}
function updateValueLabels() {
  const fmt = (v, d = 2) => {
    const s = v.toFixed(d);
    return v >= 0 ? '+' + s : s;
  };
  document.getElementById('val-p1x').textContent = fmt(S.p1x);
  document.getElementById('val-p1y').textContent = fmt(S.p1y);
  document.getElementById('val-p2x').textContent = fmt(S.p2x);
  document.getElementById('val-p2y').textContent = fmt(S.p2y);
  document.getElementById('val-F2').textContent  = S.F2.toFixed(2);
  document.getElementById('val-sigma').textContent = S.sigma.toFixed(3);
}
bindSlider('p1x', 'p1x'); bindSlider('p1y', 'p1y');
bindSlider('p2x', 'p2x'); bindSlider('p2y', 'p2y');
bindSlider('F2', 'F2'); bindSlider('sigma', 'sigma');
updateValueLabels();

/* Mouse for drag (absolute coords; per-canvas conversion via localMouse) */
document.addEventListener('mousemove', (e) => {
  mouseAbs = { x: e.clientX, y: e.clientY };
});
document.addEventListener('mousedown', (e) => {
  pressed = true; held = true;
});
document.addEventListener('mouseup', () => {
  released = true; held = false;
});

/* (Theme toggle is loaded separately via theme.js for cross-page reuse.) */

/* Kick off */
loop();

})();
