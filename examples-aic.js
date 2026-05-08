/* ===========================================================================
 * Mini-demo for the "A parsimony cost for model selection (AIC)" section.
 *
 * Toy 2D-Gaussian setup (σ_PSF = 1, FOV = ±3, 20×20 detector grid).  Generates
 * many noise-only realisations and, for each, finds the maximum matched-filter
 * S/N over a search grid of source positions.  Builds a histogram of those
 * max-S/N values, fits a Gaussian, and reports η_SNR = μ_fit + 5σ_fit.
 *
 *   Plot 1: a single pure-noise sample on the detector.
 *   Plot 2: histogram + Gaussian fit + vertical line at η_SNR (the 5σ tail
 *           of the fit), with the numerical η_SNR value displayed.
 * =========================================================================== */

(() => {

const cvNoise = document.getElementById('aic-noise');
const cvFit   = document.getElementById('aic-fit');
const cvHist  = document.getElementById('aic-hist');
const etaEl   = document.getElementById('aic-eta-val');
if (!cvNoise || !cvHist) return;

const SIGMA_PSF = 1.0;
const RANGE     = 3.0;
const PI_PIX    = 20;
const dx_pix    = 2 * RANGE / PI_PIX;             /* pixel size on detector */
const N_MC      = 5000;
const N_BINS    = 60;
const PSF_norm  = 1 / Math.sqrt(Math.PI * SIGMA_PSF * SIGMA_PSF);

/* ---- helpers ---- */
function gauss() {
  const u1 = Math.random() + 1e-12;
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
function getCSSColor(name) {
  return getComputedStyle(document.body).getPropertyValue(name).trim();
}
function pixelX(j) { return -RANGE + (j + 0.5) * dx_pix; }
function pixelY(i) { return +RANGE - (i + 0.5) * dx_pix; }

/* ---- Pre-tabulate the matched-filter response m(q) at a 20×20 grid of
 * source positions q (one q per pixel centre).  This avoids re-evaluating
 * the Gaussian inside the MC inner loop. */
const N_Q = PI_PIX;
const m_table   = [];     /* array of N_Q² Float64Arrays, each of size PI_PIX² */
const m_norm_sq = new Float64Array(N_Q * N_Q);
(function build_m_table() {
  for (let qi = 0; qi < N_Q; qi++) {
    const qy = pixelY(qi);
    for (let qj = 0; qj < N_Q; qj++) {
      const qx = pixelX(qj);
      const m  = new Float64Array(PI_PIX * PI_PIX);
      let nrm2 = 0;
      for (let i = 0; i < PI_PIX; i++) {
        const y = pixelY(i);
        for (let j = 0; j < PI_PIX; j++) {
          const x = pixelX(j);
          const r2 = (x - qx) * (x - qx) + (y - qy) * (y - qy);
          const v  = PSF_norm * Math.exp(-r2 / (2 * SIGMA_PSF * SIGMA_PSF));
          m[i * PI_PIX + j] = v;
          nrm2 += v * v;
        }
      }
      m_table.push(m);
      m_norm_sq[qi * N_Q + qj] = nrm2;
    }
  }
})();

/* ---- Run the MC: noise-only X, find max(⟨m_q, X⟩, 0) / ‖m_q‖ over q ----
 * One-sided maximum (the paper's model space requires F_q ≥ 0, so a
 * candidate with negative ⟨m_q, X⟩ collapses to F̂ = 0 and J = ‖X‖² —
 * indistinguishable from the empty model).
 *
 * NOTE: this MC searches the *discrete* 20×20 q-table only, while the live
 * fit shown in the middle plot also runs a continuous hill-climb refinement
 * (find_qstar() below).  The refinement can push the per-sample S/N up by
 * a few percent above the discrete max, so the displayed S/N may sit
 * slightly to the right of the histogram's bulk on average.  This is a
 * deliberate trade-off — adding refinement to N_MC=5000 samples would
 * multiply startup cost by ~50× without meaningfully changing
 * η_SNR = μ_fit + 5σ_fit (the parametric Gaussian extrapolation dominates).
 */
const snr_samples = new Float64Array(N_MC);
let snr_min = Infinity, snr_max_obs = 0;
let snr_mean = 0, snr_var = 0;
let eta_SNR  = 5;

(function run_mc() {
  const X = new Float64Array(PI_PIX * PI_PIX);
  for (let s = 0; s < N_MC; s++) {
    for (let p = 0; p < X.length; p++) X[p] = gauss();
    let best = 0;
    for (let q = 0; q < m_table.length; q++) {
      const m_q = m_table[q];
      let b = 0;
      for (let p = 0; p < X.length; p++) b += m_q[p] * X[p];
      if (b <= 0) continue;                        /* F̂ ≥ 0 constraint */
      const snr = b / Math.sqrt(m_norm_sq[q]);
      if (snr > best) best = snr;
    }
    snr_samples[s] = best;
    snr_mean += best;
    if (best < snr_min)     snr_min     = best;
    if (best > snr_max_obs) snr_max_obs = best;
  }
  snr_mean /= N_MC;
  for (let s = 0; s < N_MC; s++) snr_var += (snr_samples[s] - snr_mean) ** 2;
  snr_var /= N_MC;
  const snr_std = Math.sqrt(snr_var);
  eta_SNR = snr_mean + 5 * snr_std;
})();

/* ---- Histogram (linear bins from snr_min to snr_max_obs · 1.05) ---- */
const hist = new Int32Array(N_BINS);
const hi_edge = Math.max(snr_max_obs * 1.05, eta_SNR * 1.02);
const lo_edge = Math.max(0, snr_min * 0.95);
const bin_w   = (hi_edge - lo_edge) / N_BINS;
let hist_max  = 0;
for (let s = 0; s < N_MC; s++) {
  const idx = Math.min(N_BINS - 1, Math.max(0, Math.floor((snr_samples[s] - lo_edge) / bin_w)));
  hist[idx]++;
  if (hist[idx] > hist_max) hist_max = hist[idx];
}

/* ---- Live noise-sample state for plot 1 ---- */
const noisePix = new Float64Array(PI_PIX * PI_PIX);

/* ---- Best-fit single source for the current noise sample ---- */
const currentFit = { qx: 0, qy: 0, F: 0, snr: 0 };

/* One-sided matched-filter SNR with F̂ ≥ 0 constraint (paper's Q₁
 * model space).  When ⟨m_q, X⟩ ≤ 0 the optimal F̂ is 0 and the position
 * is indistinguishable from the empty model — score is 0. */
function eval_fit(qx, qy, X) {
  let g = 0, h = 0;
  const inv2s2 = 1 / (2 * SIGMA_PSF * SIGMA_PSF);
  for (let i = 0; i < PI_PIX; i++) {
    const yp = pixelY(i);
    const dy = yp - qy;
    for (let j = 0; j < PI_PIX; j++) {
      const xp = pixelX(j);
      const dx = xp - qx;
      const v  = PSF_norm * Math.exp(-(dx * dx + dy * dy) * inv2s2);
      g += v * X[i * PI_PIX + j];
      h += v * v;
    }
  }
  const F   = Math.max(0, g / h);
  const snr = Math.max(0, g) / Math.sqrt(h);
  return { g, h, snr, F };
}

function find_qstar(X) {
  /* Discrete grid search over the precomputed 20x20 q-table. */
  let best = 0, bestSnr = 0;
  for (let q = 0; q < m_table.length; q++) {
    const m_q = m_table[q];
    let g = 0;
    for (let p = 0; p < X.length; p++) g += m_q[p] * X[p];
    if (g <= 0) continue;                          /* F̂ ≥ 0 constraint */
    const snr = g / Math.sqrt(m_norm_sq[q]);
    if (snr > bestSnr) { bestSnr = snr; best = q; }
  }
  /* Continuous refinement: 8-direction hill climb with shrinking step. */
  const qi = (best / N_Q) | 0;
  const qj = best - qi * N_Q;
  let qx = pixelX(qj), qy = pixelY(qi);
  let r  = eval_fit(qx, qy, X);
  let step = dx_pix * 0.5;
  const dirs = [[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,1],[1,-1],[-1,-1]];
  for (let it = 0; it < 40; it++) {
    let improved = false;
    for (const [ux, uy] of dirs) {
      const r2 = eval_fit(qx + ux * step, qy + uy * step, X);
      if (r2.snr > r.snr) {
        qx += ux * step; qy += uy * step; r = r2;
        improved = true; break;
      }
    }
    if (!improved) { step *= 0.5; if (step < 1e-4) break; }
  }
  return { qx, qy, F: r.F, snr: r.snr };
}

function reroll_noise() {
  for (let p = 0; p < noisePix.length; p++) noisePix[p] = gauss();
  const f = find_qstar(noisePix);
  currentFit.qx = f.qx; currentFit.qy = f.qy;
  currentFit.F  = f.F;  currentFit.snr = f.snr;
}
reroll_noise();

const newNoiseBtn = document.getElementById('aic-new-noise');
if (newNoiseBtn) newNoiseBtn.addEventListener('click', reroll_noise);

/* ---- Plot 1: pure-noise sample on the detector ---- */
function draw_noise() {
  const cv = cvNoise;
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth, cssH = cv.clientHeight;
  const cw = Math.round(cssW * dpr), ch = Math.round(cssH * dpr);
  if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.fillStyle = getCSSColor('--canvas-bg');
  ctx.fillRect(0, 0, cssW, cssH);

  const sz_css = Math.min(cssW, cssH);
  const x0_css = (cssW - sz_css) / 2, y0_css = (cssH - sz_css) / 2;
  const cell   = sz_css / PI_PIX;

  /* Noise visualised on a [-3σ, +3σ] grayscale ramp. */
  for (let pi = 0; pi < PI_PIX; pi++) {
    for (let pj = 0; pj < PI_PIX; pj++) {
      const v = noisePix[pi * PI_PIX + pj];
      let t = (v + 3) / 6;     /* map [-3, +3] → [0, 1] */
      if (t < 0) t = 0; else if (t > 1) t = 1;
      const g = (t * 255) | 0;
      ctx.fillStyle = `rgb(${g},${g},${g})`;
      ctx.fillRect(x0_css + pj * cell, y0_css + pi * cell,
                   cell + 0.5, cell + 0.5);
    }
  }
  ctx.strokeStyle = getCSSColor('--border');
  ctx.strokeRect(x0_css, y0_css, sz_css, sz_css);
}

/* ---- Plot 2: clean Gaussian fit F̂·U(x-q⋆) on a fine grid ----
 * Same [-3, +3] greyscale ramp as the noise plot so the two are visually
 * directly comparable.  Annotations (F̂, S/N) drawn in white. */
function draw_fit() {
  const cv = cvFit;
  if (!cv) return;
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth, cssH = cv.clientHeight;
  const cw = Math.round(cssW * dpr), ch = Math.round(cssH * dpr);
  if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.fillStyle = getCSSColor('--canvas-bg');
  ctx.fillRect(0, 0, cssW, cssH);

  const sz_css = Math.min(cssW, cssH);
  const x0_css = (cssW - sz_css) / 2, y0_css = (cssH - sz_css) / 2;

  const N_FINE = 140;
  const cell   = sz_css / N_FINE;
  const inv2s2 = 1 / (2 * SIGMA_PSF * SIGMA_PSF);
  const F = currentFit.F;
  const qx = currentFit.qx, qy = currentFit.qy;
  for (let i = 0; i < N_FINE; i++) {
    const y = +RANGE - (i + 0.5) / N_FINE * 2 * RANGE;
    const dy = y - qy;
    for (let j = 0; j < N_FINE; j++) {
      const x = -RANGE + (j + 0.5) / N_FINE * 2 * RANGE;
      const dx = x - qx;
      const v  = F * PSF_norm * Math.exp(-(dx * dx + dy * dy) * inv2s2);
      let t = (v + 3) / 6;
      if (t < 0) t = 0; else if (t > 1) t = 1;
      const g = (t * 255) | 0;
      ctx.fillStyle = `rgb(${g},${g},${g})`;
      ctx.fillRect(x0_css + j * cell, y0_css + i * cell,
                   cell + 0.5, cell + 0.5);
    }
  }
  ctx.strokeStyle = getCSSColor('--border');
  ctx.strokeRect(x0_css, y0_css, sz_css, sz_css);

  /* Cross-hair reticle at q⋆ — needed because F̂·U has small amplitude on
   * noise-only data so the bright/dark spot can be hard to spot. */
  const qx_css = x0_css + (qx + RANGE) / (2 * RANGE) * sz_css;
  const qy_css = y0_css + (RANGE - qy) / (2 * RANGE) * sz_css;
  ctx.strokeStyle = 'rgb(220, 60, 60)';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.arc(qx_css, qy_css, 8, 0, Math.PI * 2);
  ctx.moveTo(qx_css - 12, qy_css); ctx.lineTo(qx_css - 4, qy_css);
  ctx.moveTo(qx_css + 4,  qy_css); ctx.lineTo(qx_css + 12, qy_css);
  ctx.moveTo(qx_css, qy_css - 12); ctx.lineTo(qx_css, qy_css - 4);
  ctx.moveTo(qx_css, qy_css + 4);  ctx.lineTo(qx_css, qy_css + 12);
  ctx.stroke();

  ctx.fillStyle = 'white';
  ctx.font = '11px monospace';
  ctx.fillText(`F̂ = ${F.toFixed(3)}`,
               x0_css + 6, y0_css + 14);
  ctx.fillText(`S/N = ${currentFit.snr.toFixed(2)}`,
               x0_css + 6, y0_css + sz_css - 6);
}

/* ---- Plot 3: histogram + Gaussian fit + 5σ line ---- */
function draw_hist() {
  const cv = cvHist;
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth, cssH = cv.clientHeight;
  const cw = Math.round(cssW * dpr), ch = Math.round(cssH * dpr);
  if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.fillStyle = getCSSColor('--canvas-bg');
  ctx.fillRect(0, 0, cssW, cssH);

  const padL = 36, padR = 12, padT = 18, padB = 26;
  const px = padL, py = padT;
  const pw = cssW - padL - padR;
  const ph = cssH - padT - padB;

  const x_to_px = (x) => px + (x - lo_edge) / (hi_edge - lo_edge) * pw;
  const c_to_py = (c) => py + ph - (c / hist_max) * ph;

  /* Histogram bars */
  const barCol = getCSSColor('--text-soft') || 'rgb(120, 120, 130)';
  ctx.fillStyle = barCol;
  ctx.globalAlpha = 0.55;
  for (let b = 0; b < N_BINS; b++) {
    if (hist[b] === 0) continue;
    const x1 = x_to_px(lo_edge + b * bin_w);
    const x2 = x_to_px(lo_edge + (b + 1) * bin_w);
    const yT = c_to_py(hist[b]);
    const yB = c_to_py(0);
    ctx.fillRect(x1, yT, Math.max(1, x2 - x1 - 1), yB - yT);
  }
  ctx.globalAlpha = 1.0;

  /* Gaussian fit overlay (scaled to the histogram peak). */
  const snr_std = Math.sqrt(snr_var);
  const N_pts   = 240;
  const y_norm  = N_MC * bin_w / (snr_std * Math.sqrt(2 * Math.PI));
  ctx.strokeStyle = 'rgb(80, 200, 110)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let k = 0; k <= N_pts; k++) {
    const x = lo_edge + k / N_pts * (hi_edge - lo_edge);
    const z = (x - snr_mean) / snr_std;
    const f = Math.exp(-0.5 * z * z) * y_norm;
    const xc = x_to_px(x);
    const yc = c_to_py(f);
    if (k === 0) ctx.moveTo(xc, yc); else ctx.lineTo(xc, yc);
  }
  ctx.stroke();

  /* 5σ vertical line at η_SNR = μ + 5σ. */
  if (eta_SNR >= lo_edge && eta_SNR <= hi_edge) {
    const xc = x_to_px(eta_SNR);
    ctx.strokeStyle = 'rgb(220, 60, 60)';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(xc, py); ctx.lineTo(xc, py + ph); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgb(220, 60, 60)';
    ctx.font = '11px monospace';
    ctx.fillText(`η_SNR = ${eta_SNR.toFixed(2)}`, xc + 6, py + 12);
  }

  /* Frame + axis labels */
  ctx.strokeStyle = getCSSColor('--border');
  ctx.lineWidth = 1;
  ctx.strokeRect(px, py, pw, ph);
  ctx.fillStyle = getCSSColor('--text-soft');
  ctx.font = '10px monospace';
  /* x ticks: integer values inside [lo, hi]. */
  const xlo = Math.ceil(lo_edge);
  const xhi = Math.floor(hi_edge);
  for (let v = xlo; v <= xhi; v++) {
    const xt = x_to_px(v);
    ctx.beginPath(); ctx.moveTo(xt, py + ph); ctx.lineTo(xt, py + ph + 3); ctx.stroke();
    ctx.fillText(String(v), xt - 3, py + ph + 14);
  }
  ctx.fillText('S/N (Q₁⋆)',  px + pw / 2 - 28, py + ph + 22);
  ctx.fillText('count', 4, py + ph / 2);
}

/* ---- Update the η readout once. */
if (etaEl) etaEl.textContent = eta_SNR.toFixed(3);

/* ---- Expose to other demos that share the 2D-Gaussian PSF setup
 * (σ_PSF=1, FOV=±3, 20×20 grid).  examples-contam.js reads this so the
 * R-classification plot uses the same look-elsewhere-corrected η as the
 * AIC histogram above. */
window.AIC_ETA_2D_GAUSSIAN = eta_SNR;

/* ---- Main loop. */
function loop() {
  draw_noise();
  draw_fit();
  draw_hist();
  requestAnimationFrame(loop);
}
loop();

})();
