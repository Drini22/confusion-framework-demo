/* ===========================================================================
 * Mini-demo for the "Specialization to a circular aperture" section.
 *
 * Two rows of three plots each, plus an aperture-tile palette on top.
 *
 *   Palette         : 5 draggable aperture tiles (disk, annulus, square,
 *                     hexagon, 6-star). Drop one onto the slot to make it
 *                     active; all plots recompute.
 *
 *   Row 1
 *     airy-fov      : Field of view with draggable p_1, p_2 markers,
 *                     gridlines, axes — no PSF render.
 *     airy-slot     : Drop zone showing the current aperture's pupil mask.
 *     airy-noisy    : Pixelated mu_P (active aperture) + sigma noise.
 *
 *   Row 2
 *     airy-P        : Clean mu_P (active aperture), draggable p_1, p_2.
 *     airy-Dmap     : D-bar({p_1, p_2 = (x, y)}) over FOV with red contour.
 *     airy-Dratio   : D-bar(delta, F_P1/F_P2) heatmap with red contour and
 *                     yellow dot at the user's current (delta, ratio).
 *                     Recomputed for the active aperture, evaluated along
 *                     the current segment angle.
 *
 *   Sliders         : F_P1, F_P2, sigma.
 *
 * Lengths in lambda/d.  delta_Rayleigh = 1.22 lambda/d.
 * =========================================================================== */

(() => {

const cvFov   = document.getElementById('airy-fov');
const cvSlot  = document.getElementById('airy-slot');
const cvNoisy = document.getElementById('airy-noisy');
const cvP     = document.getElementById('airy-P');
const cvDmap  = document.getElementById('airy-Dmap');
const cvR     = document.getElementById('airy-Dratio');
const cvBest  = document.getElementById('airy-best');
const cvFlux  = document.getElementById('airy-flux');
const cvClass = document.getElementById('airy-class');
if (!cvFov || !cvSlot || !cvNoisy || !cvP || !cvDmap || !cvR
    || !cvBest || !cvFlux || !cvClass) return;
if (typeof Apertures === 'undefined') {
  console.error('Apertures not loaded — include fft.js + apertures.js before examples-airy.js');
  return;
}

const RAYLEIGH  = 1.22;                 /* λ/d */
const FOV       = 1.5 * RAYLEIGH;       /* half-width of FOV plots, in λ/d */
const PI_FINE   = 200;
const PI_PIX    = 20;
const N_DGRID   = 100;                  /* row-2 D-map grid */
const N_DELTA   = 220;                  /* row-2 ratio map δ axis */
const N_RATIO   = 120;                  /* row-2 ratio map y axis */
const DELTA_MAX = 2.0;          /* x-axis max for plot 6 (Dratio) and plot 9 (Class), in λ/d */

/* δ_1(ratio) curve overlaid on the (δ, ratio) heatmap.  For each ratio,
 * build a 2D D-map (p_1 at origin, F_1 = ratio, F_2 = 1) and integrate the
 * area where D̄ > 0; δ_1 = √(area/π).  Computed once per aperture.  */
const N_CURVE_R   = 24;
const N_CURVE_G   = 50;
const FOV_CURVE   = 1.7 * RAYLEIGH;
const dCcurve = {
  ratios: new Float64Array(N_CURVE_R),
  deltas: new Float64Array(N_CURVE_R),       /* in δ_R units */
};

const S = {
  P1x: -0.45 * RAYLEIGH, P1y: 0.0,
  P2x: +0.45 * RAYLEIGH, P2y: 0.0,
  FP1: 0.7, FP2: 1.0,
  sigma: 0.10,
};

const muFine       = new Float64Array(PI_FINE * PI_FINE);
const muPix        = new Float64Array(PI_PIX * PI_PIX);
const noiseSamples = new Float64Array(PI_PIX * PI_PIX);
let noiseInit = false;
let pMax = 1;

/* Per-source response maps for matched-filter retrieval (row 3).
 * m_i_fine = U(x − p_i) on the fine grid; m_i_pix = block-averaged to
 * the detector grid (PI_PIX).  K_pix is the 2×2 Gram matrix of the
 * two-source pix responses.  m_q_* is the matched-filter "phantom"
 * response at the deterministic Q⋆ (computed on clean μ_P), used for
 * the N=1 (single-source) candidate model. */
const m1_fine  = new Float64Array(PI_FINE * PI_FINE);
const m2_fine  = new Float64Array(PI_FINE * PI_FINE);
const mQ_fine  = new Float64Array(PI_FINE * PI_FINE);
const m1_pix   = new Float64Array(PI_PIX * PI_PIX);
const m2_pix   = new Float64Array(PI_PIX * PI_PIX);
const mQ_pix   = new Float64Array(PI_PIX * PI_PIX);
const Kpix     = { K11: 1, K12: 0, K22: 1 };
const Qstar    = { qx: 0, qy: 0, K: 1 };

/* Best-model state.  We try 5 candidates as MLE starting points:
 *   ∅, {p_1}, {p_2}, Q⋆, P
 * Each is fitted (positions + fluxes refined to a local optimum), then
 * compared by total cost J_pix/σ² + N·η² where η² is the
 * look-elsewhere-corrected per-aperture threshold (≈ 24–26, computed by
 * apertures.js → ap.eta2; see the "Calibrating η_SNR" section).
 *
 * Two parallel structures:
 *   - currentFits: re-derived every frame from the cached noise sample.
 *     Drives plot 1 (best-fit μ̂) and the "best:" badge.
 *   - mcFits: re-derived only when mcDirty.  N_MC fresh noise samples
 *     are fit from the current-sample winner's starting candidate, and
 *     mean ± std of each fitted source's flux is accumulated.  Drives
 *     plot 2's error bars.
 * Each fit slot:
 *   { qx, qy, F, col, [Fmean, Fstd, qxMean, qyMean, qxStd, qyStd] }
 * `col ∈ { 'src1', 'src2', 'Qstar' }` is determined by the *starter
 * candidate*'s name (not the fitted position) so the badge and the bar
 * always agree on which column the retrieval lives in. */
const CAND_NAMES = ['empty', 'p1', 'p2', 'Qstar', 'P'];
const Best = {
  /* per-frame, current cached sample */
  currentN:     2,
  currentName:  'P',
  currentFits:  [],
  /* cached over N_MC fresh samples, refreshed only on mcDirty */
  mcN:          -1,
  mcName:       '',
  mcFits:       [],
};
let bestMuMax = 1;
let mcDirty   = true;

const N_MC = 200;        /* Monte-Carlo samples for the error bars */
const MLE_ITERS_1   = 10;
const MLE_ITERS_2   = 8;
const MLE_EPS       = 0.025;  /* finite-diff step in λ/d for position grad */
const MLE_LR        = 0.05;   /* gradient-descent step (λ/d) */

const N_CLASS = 100;
const Rd = new Float64Array(N_CLASS + 1);
const Rr = new Float64Array(N_CLASS + 1);
const Rc = new Float64Array(N_CLASS + 1);

const Dmap   = new Float64Array(N_DGRID * N_DGRID);
const Dratio = new Float64Array(N_RATIO * N_DELTA);

let dmapDirty   = true;
let dratioDirty = true;

/* Per-plot log/lin display toggles. */
let pLog = false, noisyLog = false;
function logStretch(t) {
  /* asinh-style logarithmic stretch with γ = 1000.  Maps t ∈ [0, 1] to
   * [0, 1]; at t = 0 returns 0, at t = 1 returns 1, with strong
   * compression of small values (the dark diffraction rings become
   * visible). */
  const g = 1000;
  return Math.log1p(g * Math.max(0, t)) / Math.log1p(g);
}

/* ---- Helpers ---- */
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
function getCSSColor(name) {
  return getComputedStyle(document.body).getPropertyValue(name).trim();
}

/* ---- Compute mu_P, m1, m2 on fine + pixelated grids ---- *
 * mu_P = F1·m1 + F2·m2 where m_i(x) = U(x − p_i).  Pixelated maps are
 * block-averages of the fine maps. */
function compute_means() {
  const ap = Apertures.active();
  pMax = 1e-9;
  for (let i = 0; i < PI_FINE; i++) {
    const y = +FOV - (i + 0.5) / PI_FINE * 2 * FOV;
    for (let j = 0; j < PI_FINE; j++) {
      const x  = -FOV + (j + 0.5) / PI_FINE * 2 * FOV;
      const u1 = Apertures.U(ap, x - S.P1x, y - S.P1y);
      const u2 = Apertures.U(ap, x - S.P2x, y - S.P2y);
      const v  = S.FP1 * u1 + S.FP2 * u2;
      m1_fine[i * PI_FINE + j] = u1;
      m2_fine[i * PI_FINE + j] = u2;
      muFine[i * PI_FINE + j]  = v;
      if (v > pMax) pMax = v;
    }
  }
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
  /* 2×2 Gram K_pix = M^T M. */
  let K11 = 0, K22 = 0, K12 = 0;
  for (let p = 0; p < PI_PIX * PI_PIX; p++) {
    K11 += m1_pix[p] * m1_pix[p];
    K22 += m2_pix[p] * m2_pix[p];
    K12 += m1_pix[p] * m2_pix[p];
  }
  Kpix.K11 = K11; Kpix.K12 = K12; Kpix.K22 = K22;
}

/* ---- MLE helpers: pixel-space inner products at arbitrary positions ----
 * compute_b_at returns ⟨m_q_pix, y⟩, compute_K_at returns ⟨m_q_pix, m_q_pix⟩,
 * and compute_K_cross returns ⟨m_q1_pix, m_q2_pix⟩.  All sample U directly
 * at PI_PIX pixel centres (no fine sub-pixel averaging — adequate for the
 * fit search; the deterministic clean response maps still use averaging). */
const PIX_step = 2 * FOV / PI_PIX;
const PIX_x0   = -FOV + 0.5 * PIX_step;
const PIX_y0   = +FOV - 0.5 * PIX_step;
const yBuf     = new Float64Array(PI_PIX * PI_PIX);
let   sumY2    = 0;

function compute_b_at(qx, qy, y) {
  const ap = Apertures.active();
  let s = 0;
  for (let pi = 0; pi < PI_PIX; pi++) {
    const py_pos = PIX_y0 - pi * PIX_step;
    const yRow   = pi * PI_PIX;
    for (let pj = 0; pj < PI_PIX; pj++) {
      const px_pos = PIX_x0 + pj * PIX_step;
      s += Apertures.U(ap, px_pos - qx, py_pos - qy) * y[yRow + pj];
    }
  }
  return s;
}
function compute_K_at(qx, qy) {
  const ap = Apertures.active();
  let s = 0;
  for (let pi = 0; pi < PI_PIX; pi++) {
    const py_pos = PIX_y0 - pi * PIX_step;
    for (let pj = 0; pj < PI_PIX; pj++) {
      const px_pos = PIX_x0 + pj * PIX_step;
      const v = Apertures.U(ap, px_pos - qx, py_pos - qy);
      s += v * v;
    }
  }
  return s;
}
function compute_K_cross(q1x, q1y, q2x, q2y) {
  const ap = Apertures.active();
  let s = 0;
  for (let pi = 0; pi < PI_PIX; pi++) {
    const py_pos = PIX_y0 - pi * PIX_step;
    for (let pj = 0; pj < PI_PIX; pj++) {
      const px_pos = PIX_x0 + pj * PIX_step;
      const v1 = Apertures.U(ap, px_pos - q1x, py_pos - q1y);
      const v2 = Apertures.U(ap, px_pos - q2x, py_pos - q2y);
      s += v1 * v2;
    }
  }
  return s;
}

function fill_y(noiseArr) {
  /* y_pix = muPix + σ · noiseArr */
  let s2 = 0;
  for (let p = 0; p < PI_PIX * PI_PIX; p++) {
    const v = muPix[p] + S.sigma * noiseArr[p];
    yBuf[p] = v;
    s2 += v * v;
  }
  sumY2 = s2;
}

/* ---- 1-source MLE: gradient ascent on ⟨m_q, y⟩², starting from (qx, qy) ----
 * Flux is eliminated analytically: F = ⟨m_q, y⟩ / ⟨m_q, m_q⟩.
 * Returns the fitted (qx, qy, F, J) where J = ‖y‖² − F·b. */
function fit_1source(qx0, qy0, y) {
  let qx = qx0, qy = qy0;
  let b  = compute_b_at(qx, qy, y);
  for (let it = 0; it < MLE_ITERS_1; it++) {
    const eps = MLE_EPS;
    const bxp = compute_b_at(qx + eps, qy, y);
    const bxm = compute_b_at(qx - eps, qy, y);
    const byp = compute_b_at(qx, qy + eps, y);
    const bym = compute_b_at(qx, qy - eps, y);
    const sign = (b >= 0) ? 1 : -1;
    const gx = sign * (bxp - bxm) / (2 * eps);
    const gy = sign * (byp - bym) / (2 * eps);
    const gn = Math.hypot(gx, gy);
    if (gn < 1e-7) break;
    let alpha = MLE_LR;
    let accepted = false;
    for (let bs = 0; bs < 6; bs++) {
      const nx = qx + alpha * gx / gn;
      const ny = qy + alpha * gy / gn;
      const nb = compute_b_at(nx, ny, y);
      if (nb * nb > b * b + 1e-12) {
        qx = nx; qy = ny; b = nb; accepted = true; break;
      }
      alpha *= 0.5;
    }
    if (!accepted) break;
  }
  const Kq = compute_K_at(qx, qy);
  const F  = (Kq > 1e-12) ? (b / Kq) : 0;
  const J  = sumY2 - F * b;
  return { qx, qy, F, J };
}

/* ---- 2-source MLE: alternating flux solve + position gradient step ---- */
function flux_solve_2(b1, b2, K11, K22, K12) {
  const det = K11 * K22 - K12 * K12;
  if (Math.abs(det) < 1e-12) return [0, 0];
  return [(K22 * b1 - K12 * b2) / det, (K11 * b2 - K12 * b1) / det];
}
function J_2source(q1x, q1y, q2x, q2y, y) {
  const b1  = compute_b_at(q1x, q1y, y);
  const b2  = compute_b_at(q2x, q2y, y);
  const K11 = compute_K_at(q1x, q1y);
  const K22 = compute_K_at(q2x, q2y);
  const K12 = compute_K_cross(q1x, q1y, q2x, q2y);
  const [F1, F2] = flux_solve_2(b1, b2, K11, K22, K12);
  return { J: sumY2 - F1 * b1 - F2 * b2, F1, F2 };
}
function fit_2source(q1x0, q1y0, q2x0, q2y0, y) {
  let q1x = q1x0, q1y = q1y0, q2x = q2x0, q2y = q2y0;
  let cur = J_2source(q1x, q1y, q2x, q2y, y);
  for (let it = 0; it < MLE_ITERS_2; it++) {
    const eps = MLE_EPS;
    const J0 = cur.J;
    const J1xp = J_2source(q1x + eps, q1y, q2x, q2y, y).J;
    const J1xm = J_2source(q1x - eps, q1y, q2x, q2y, y).J;
    const J1yp = J_2source(q1x, q1y + eps, q2x, q2y, y).J;
    const J1ym = J_2source(q1x, q1y - eps, q2x, q2y, y).J;
    const J2xp = J_2source(q1x, q1y, q2x + eps, q2y, y).J;
    const J2xm = J_2source(q1x, q1y, q2x - eps, q2y, y).J;
    const J2yp = J_2source(q1x, q1y, q2x, q2y + eps, y).J;
    const J2ym = J_2source(q1x, q1y, q2x, q2y - eps, y).J;
    const g1x = (J1xp - J1xm) / (2 * eps);
    const g1y = (J1yp - J1ym) / (2 * eps);
    const g2x = (J2xp - J2xm) / (2 * eps);
    const g2y = (J2yp - J2ym) / (2 * eps);
    const gn = Math.hypot(g1x, g1y, g2x, g2y);
    if (gn < 1e-7) break;
    let alpha = MLE_LR;
    let accepted = false;
    for (let bs = 0; bs < 6; bs++) {
      const nq1x = q1x - alpha * g1x / gn;
      const nq1y = q1y - alpha * g1y / gn;
      const nq2x = q2x - alpha * g2x / gn;
      const nq2y = q2y - alpha * g2y / gn;
      const nxt = J_2source(nq1x, nq1y, nq2x, nq2y, y);
      if (nxt.J < J0 - 1e-12) {
        q1x = nq1x; q1y = nq1y; q2x = nq2x; q2y = nq2y;
        cur = nxt; accepted = true; break;
      }
      alpha *= 0.5;
    }
    if (!accepted) break;
  }
  return { q1x, q1y, q2x, q2y, F1: cur.F1, F2: cur.F2, J: cur.J };
}

/* ---- 5-candidate AIC search ----
 * Tries each candidate's starting model, runs MLE refinement, then picks
 * the one with smallest J/σ² + N·η². */
function compute_Qstar_responses() {
  /* Compute the clean-data Q⋆ position and its fine response for plot 1 +
   * for the {Q⋆} starting candidate. */
  const ap = Apertures.active();
  const r = find_qstar_2D(S.P1x, S.P1y, S.P2x, S.P2y, S.FP1, S.FP2);
  Qstar.qx = r.qx;
  Qstar.qy = r.qy;
  for (let i = 0; i < PI_FINE; i++) {
    const y = +FOV - (i + 0.5) / PI_FINE * 2 * FOV;
    for (let j = 0; j < PI_FINE; j++) {
      const x = -FOV + (j + 0.5) / PI_FINE * 2 * FOV;
      mQ_fine[i * PI_FINE + j] = Apertures.U(ap, x - Qstar.qx, y - Qstar.qy);
    }
  }
}

/* Run the 5-candidate AIC search on the y_pix already loaded in yBuf,
 * returning an object describing the winner.  Used by both the per-frame
 * current-sample badge and the per-MC-sample inner loop. */
function aic_search() {
  const sig2 = Math.max(S.sigma * S.sigma, 1e-12);
  const eta2 = Apertures.active().eta2;
  let best = { name: 'empty', N: 0, Jtot: sumY2 / sig2, fits: [] };

  const starts1 = [
    { name: 'p1',    qx: S.P1x,    qy: S.P1y    },
    { name: 'p2',    qx: S.P2x,    qy: S.P2y    },
    { name: 'Qstar', qx: Qstar.qx, qy: Qstar.qy },
  ];
  for (const s of starts1) {
    const f = fit_1source(s.qx, s.qy, yBuf);
    const Jtot = f.J / sig2 + eta2;
    if (Jtot < best.Jtot) {
      best = { name: s.name, N: 1, Jtot,
               fits: [{ qx: f.qx, qy: f.qy, F: f.F }] };
    }
  }
  {
    const f = fit_2source(S.P1x, S.P1y, S.P2x, S.P2y, yBuf);
    const Jtot = f.J / sig2 + 2 * eta2;
    if (Jtot < best.Jtot) {
      best = { name: 'P', N: 2, Jtot,
               fits: [{ qx: f.q1x, qy: f.q1y, F: f.F1 },
                      { qx: f.q2x, qy: f.q2y, F: f.F2 }] };
    }
  }

  /* Deterministic starter→col mapping, so the badge always names the same
   * column the bar lives in. */
  if (best.N === 1) {
    const map = { p1: 'src1', p2: 'src2', Qstar: 'Qstar' };
    best.fits[0].col = map[best.name];
  } else if (best.N === 2) {
    /* P starter: lock fitted sources to {src1, src2} by closeness to truth. */
    const f1 = best.fits[0], f2 = best.fits[1];
    const d1 = (f1.qx - S.P1x)**2 + (f1.qy - S.P1y)**2;
    const d2 = (f2.qx - S.P1x)**2 + (f2.qy - S.P1y)**2;
    if (d1 <= d2) { f1.col = 'src1'; f2.col = 'src2'; }
    else          { f1.col = 'src2'; f2.col = 'src1'; }
  }
  return best;
}

/* Per-frame: AIC search on the current cached noise → drives plot 1 + badge. */
function compute_best_model() {
  ensureNoise();
  fill_y(noiseSamples);
  const w = aic_search();
  Best.currentN    = w.N;
  Best.currentName = w.name;
  Best.currentFits = w.fits;
}

/* Render the current-sample fitted μ̂ onto a fine grid for plot 1. */
const muFitFine = new Float64Array(PI_FINE * PI_FINE);
function compute_muFitFine() {
  const ap = Apertures.active();
  let pk = 1e-9;
  if (Best.currentFits.length === 0) {
    muFitFine.fill(0);
    bestMuMax = pk;
    return;
  }
  for (let i = 0; i < PI_FINE; i++) {
    const yy = +FOV - (i + 0.5) / PI_FINE * 2 * FOV;
    for (let j = 0; j < PI_FINE; j++) {
      const xx = -FOV + (j + 0.5) / PI_FINE * 2 * FOV;
      let v = 0;
      for (const f of Best.currentFits) {
        v += f.F * Apertures.U(ap, xx - f.qx, yy - f.qy);
      }
      muFitFine[i * PI_FINE + j] = v;
      if (v > pk) pk = v;
    }
  }
  bestMuMax = pk;
}

/* ---- MC error bars ----
 *
 * For each of N_MC fresh noise realisations:
 *   1. Generate y_s = μ_P + σ · n_s.
 *   2. Run the full 5-candidate AIC search on y_s.
 *   3. Take the winning model's fitted parameters.
 *
 * The most-frequent winner over N_MC samples gives the cached badge
 * (Best.mcName, Best.mcN).  Samples whose winner matches that majority
 * winner contribute to the running mean / std of the fitted positions
 * and fluxes; the bars in plot 2 are mean ± 2σ over those samples.
 *
 * yBuf is overwritten during the loop, then restored at the end so the
 * per-frame compute_best_model continues to see the current sample. */
function refresh_mc() {
  if (!mcDirty) return;
  const fresh = new Float64Array(PI_PIX * PI_PIX);

  /* First pass: histogram over winning candidate names. */
  const counts = { empty: 0, p1: 0, p2: 0, Qstar: 0, P: 0 };
  const samples = [];   /* keep winners so we can pick a majority */
  for (let s = 0; s < N_MC; s++) {
    for (let p = 0; p < PI_PIX * PI_PIX; p++) fresh[p] = gauss();
    fill_y(fresh);
    const w = aic_search();
    counts[w.name]++;
    samples.push(w);
  }
  /* Aggregate the per-starter counts into three high-level classes:
   *   empty (∅)  |  single source (any of {p₁}, {p₂}, Q⋆)  |  P.
   * The starters {p₁}, {p₂}, Q⋆ are all 1-source models with the same
   * AIC penalty — they differ only in their initial position; the
   * MLE-fitted source can converge to any local maximum of the matched
   * filter.  Treating them as one class avoids artificial fragmentation
   * of the histogram and gives one combined bar in plot 2. */
  const singleCount = counts.p1 + counts.p2 + counts.Qstar;
  const countsAgg = {
    empty:  counts.empty,
    Qstar:  singleCount,
    P:      counts.P,
  };
  let majority = 'P', majCount = -1;
  for (const k of ['empty', 'Qstar', 'P']) {
    if (countsAgg[k] > majCount) { majCount = countsAgg[k]; majority = k; }
  }

  let mcN = 0;
  let mcFits = [];
  if (majority === 'P') {
    /* Two-source aggregation: use the first P-winning sample to fix the
     * (src1, src2) column ordering, then accumulate across all P wins. */
    const seed = samples.find(w => w.name === 'P');
    if (seed) {
      mcN = 2;
      mcFits = seed.fits.map(f => ({
        col: f.col,
        sumF: 0, sumF2: 0, sumX: 0, sumX2: 0, sumY: 0, sumY2: 0, sumXY: 0, n: 0,
      }));
    }
    for (const w of samples) {
      if (w.name !== 'P') continue;
      for (let k = 0; k < w.fits.length; k++) {
        const slot = mcFits.find(s => s.col === w.fits[k].col);
        if (!slot) continue;
        const f = w.fits[k];
        slot.sumF  += f.F;   slot.sumF2 += f.F * f.F;
        slot.sumX  += f.qx;  slot.sumX2 += f.qx * f.qx;
        slot.sumY  += f.qy;  slot.sumY2 += f.qy * f.qy;
        slot.sumXY += f.qx * f.qy;
        slot.n++;
      }
    }
  } else if (majority === 'Qstar') {
    /* Single-source aggregation: sum across all 1-source starters; the
     * lone bar lives in the Q⋆ column. */
    mcN = 1;
    mcFits = [{ col: 'Qstar',
                sumF: 0, sumF2: 0, sumX: 0, sumX2: 0, sumY: 0, sumY2: 0, sumXY: 0, n: 0 }];
    for (const w of samples) {
      if (w.name !== 'p1' && w.name !== 'p2' && w.name !== 'Qstar') continue;
      const f = w.fits[0];
      const slot = mcFits[0];
      slot.sumF  += f.F;   slot.sumF2 += f.F * f.F;
      slot.sumX  += f.qx;  slot.sumX2 += f.qx * f.qx;
      slot.sumY  += f.qy;  slot.sumY2 += f.qy * f.qy;
      slot.sumXY += f.qx * f.qy;
      slot.n++;
    }
  }
  /* majority === 'empty' → no fits, nothing to compute. */

  for (const slot of mcFits) {
    const n = Math.max(slot.n, 1);
    const mF = slot.sumF / n, mX = slot.sumX / n, mY = slot.sumY / n;
    slot.Fmean  = mF;
    slot.Fstd   = Math.sqrt(Math.max(slot.sumF2  / n - mF * mF, 0));
    slot.qxMean = mX;
    slot.qxStd  = Math.sqrt(Math.max(slot.sumX2 / n - mX * mX, 0));
    slot.qyMean = mY;
    slot.qyStd  = Math.sqrt(Math.max(slot.sumY2 / n - mY * mY, 0));
    /* 2D positional covariance + eigendecomposition for the confidence
     * ellipses on plot 7.  Σ = [[a, b], [b, c]] with a, c ≥ b² (PSD). */
    const a = slot.qxStd * slot.qxStd;
    const c = slot.qyStd * slot.qyStd;
    const b = slot.sumXY / n - mX * mY;
    slot.qxyCov = b;
    const tr = a + c;
    const dt = (a - c) * (a - c) + 4 * b * b;
    const lam_max = Math.max((tr + Math.sqrt(dt)) / 2, 0);
    const lam_min = Math.max((tr - Math.sqrt(dt)) / 2, 0);
    slot.ellipseAngle = 0.5 * Math.atan2(2 * b, a - c);  /* major-axis angle (physics coords) */
    slot.ellipseSemiMajor = Math.sqrt(lam_max);          /* 1σ semi-axes */
    slot.ellipseSemiMinor = Math.sqrt(lam_min);
  }

  Best.mcN         = mcN;
  Best.mcName      = majority;
  Best.mcFits      = mcFits;
  Best.mcCounts    = counts;       /* per-starter (debug / inspection)  */
  Best.mcCountsAgg = countsAgg;    /* aggregated 3-class — drives header */
  Best.mcTotal     = N_MC;

  /* Restore yBuf to the current noise sample so per-frame draw uses it. */
  fill_y(noiseSamples);
  mcDirty = false;
}

/* ---- Region-classification thresholds (sources on x-axis) ----
 * For each Δ, compute Q*-cost-vs-data threshold ratios Rd/Rr/Rc following
 * the contam-section formulation, with k(·) from the active aperture.   */
function compute_classification() {
  const F1 = S.FP1, F2 = S.FP2;
  for (let i = 0; i <= N_CLASS; i++) {
    const Delta = i / N_CLASS * DELTA_MAX;
    /* 1D segment Q*-search via compute_Dbar's machinery. */
    const { qx } = find_qstar_2D(-Delta/2, 0, +Delta/2, 0, F1, F2);
    const ap = Apertures.active();
    const c1 = Apertures.k(ap, qx + Delta/2, 0);
    const c2 = Apertures.k(ap, qx - Delta/2, 0);
    const r  = Apertures.k(ap, Delta, 0);
    const Fs = F1 * c1 + F2 * c2;
    const muP_sq = F1 * F1 + F2 * F2 + 2 * F1 * F2 * r;
    const JPQ    = muP_sq - Fs * Fs;
    const Je     = Fs * Fs;
    const Jp1    = F1 * F1 - 2 * F1 * Fs * c1 + Fs * Fs;
    const Jp2    = F2 * F2 - 2 * F2 * Fs * c2 + Fs * Fs;
    const minPs  = Math.min(Je, Jp1, Jp2);
    const denom_d = Math.max(muP_sq - JPQ, 1e-12);
    const denom_r = Math.max(JPQ,            1e-12);
    const denom_c = Math.max(minPs,          1e-12);
    Rd[i] = Math.sqrt(muP_sq / denom_d);
    Rr[i] = Math.sqrt(muP_sq / denom_r);
    Rc[i] = Math.sqrt(muP_sq / denom_c);
  }
}

/* ---- 2D Q* search: 1D segment seed + backtracking gradient ascent ---- */
function find_qstar_2D(p1x, p1y, p2x, p2y, F1, F2) {
  const ap = Apertures.active();
  const dx = p2x - p1x, dy = p2y - p1y;

  /* 1D segment grid (extended slightly past the endpoints) */
  let bestQx = p1x, bestQy = p1y, bestF = -Infinity;
  const T_LO = -0.4, T_HI = 1.4, N_GRID = 80;
  for (let i = 0; i <= N_GRID; i++) {
    const t = T_LO + (T_HI - T_LO) * i / N_GRID;
    const qx = p1x + t * dx, qy = p1y + t * dy;
    const f  = F1 * Apertures.k(ap, qx - p1x, qy - p1y)
             + F2 * Apertures.k(ap, qx - p2x, qy - p2y);
    if (f > bestF) { bestF = f; bestQx = qx; bestQy = qy; }
  }

  /* 2D gradient ascent off the segment */
  for (let it = 0; it < 8; it++) {
    const gx1 = Apertures.kgrad(ap, bestQx - p1x, bestQy - p1y);
    const gx2 = Apertures.kgrad(ap, bestQx - p2x, bestQy - p2y);
    const gx = F1 * gx1[0] + F2 * gx2[0];
    const gy = F1 * gx1[1] + F2 * gx2[1];
    const gn = Math.hypot(gx, gy);
    if (gn < 1e-9) break;
    let alpha = 0.15, accepted = false;
    for (let bs = 0; bs < 8; bs++) {
      const qx = bestQx + alpha * gx / gn;
      const qy = bestQy + alpha * gy / gn;
      const fNew = F1 * Apertures.k(ap, qx - p1x, qy - p1y)
                 + F2 * Apertures.k(ap, qx - p2x, qy - p2y);
      if (fNew > bestF + 1e-12) {
        bestF = fNew; bestQx = qx; bestQy = qy; accepted = true; break;
      }
      alpha *= 0.5;
    }
    if (!accepted) break;
  }

  return { qx: bestQx, qy: bestQy, fmax: bestF };
}

/* ---- D-bar at a single (p_1, p_2, F_1, F_2) ---- */
function compute_Dbar(p1x, p1y, p2x, p2y, F1, F2) {
  const ap = Apertures.active();
  const J1 = Math.max(F1 * F1, 1e-9);
  const J2 = Math.max(F2 * F2, 1e-9);
  const norm = 1 / (2 * J1) + 1 / (2 * J2);
  const { qx, qy, fmax } = find_qstar_2D(p1x, p1y, p2x, p2y, F1, F2);
  const c1 = Apertures.k(ap, qx - p1x, qy - p1y);
  const c2 = Apertures.k(ap, qx - p2x, qy - p2y);
  const kr = Apertures.k(ap, p2x - p1x, p2y - p1y);
  const Fs = fmax;
  const muP_sq = F1 * F1 + F2 * F2 + 2 * F1 * F2 * kr;
  const JPQ    = muP_sq - Fs * Fs;
  const Je     = Fs * Fs;
  const Jp1    = F1 * F1 - 2 * F1 * Fs * c1 + Fs * Fs;
  const Jp2    = F2 * F2 - 2 * F2 * Fs * c2 + Fs * Fs;
  const D      = Math.min(Je, Jp1, Jp2) - JPQ;
  return D * norm;
}

/* ---- Plot 5 (D-map): p_1 fixed, p_2 sweeps the FOV ---- */
function compute_Dmap() {
  const F1 = S.FP1, F2 = S.FP2;
  const p1x = S.P1x, p1y = S.P1y;
  for (let cy = 0; cy < N_DGRID; cy++) {
    const y = +FOV - (cy + 0.5) / N_DGRID * 2 * FOV;
    for (let cx = 0; cx < N_DGRID; cx++) {
      const x = -FOV + (cx + 0.5) / N_DGRID * 2 * FOV;
      Dmap[cy * N_DGRID + cx] = compute_Dbar(p1x, p1y, x, y, F1, F2);
    }
  }
}

/* ---- Plot 6 (D-bar vs (δ, ratio)): sources fixed on the x-axis ---- *
 * Depends only on the active aperture — independent of the user's source
 * positions and of the F_1, F_2 sliders (the y-axis is the flux ratio
 * directly).  For non-isotropic apertures (hex, star) this is the cross-
 * section along the x-axis through the aperture's PSF. */
function build_Dratio() {
  const F2 = 1.0;
  for (let jr = 0; jr < N_RATIO; jr++) {
    const ratio = (N_RATIO - 1 - jr) / (N_RATIO - 1);
    const F1    = Math.max(ratio, 1e-3) * F2;
    for (let id = 0; id < N_DELTA; id++) {
      const delta = (id / (N_DELTA - 1)) * DELTA_MAX;
      Dratio[jr * N_DELTA + id] = compute_Dbar(-0.5 * delta, 0, +0.5 * delta, 0, F1, F2);
    }
  }
}

/* ---- Area-based δ_1(ratio) curve, overlaid on the heatmap ---- */
function build_dCcurve() {
  const cellArea = (2 * FOV_CURVE / N_CURVE_G) * (2 * FOV_CURVE / N_CURVE_G);
  const D = new Float64Array(N_CURVE_G * N_CURVE_G);
  const F2 = 1.0;
  for (let jr = 0; jr < N_CURVE_R; jr++) {
    const ratio = (N_CURVE_R - 1 - jr) / (N_CURVE_R - 1);
    dCcurve.ratios[jr] = ratio;
    const F1 = Math.max(ratio, 1e-3) * F2;
    /* 2D D-map at p_1 = origin */
    for (let cy = 0; cy < N_CURVE_G; cy++) {
      const y = +FOV_CURVE - (cy + 0.5) / N_CURVE_G * 2 * FOV_CURVE;
      for (let cx = 0; cx < N_CURVE_G; cx++) {
        const x = -FOV_CURVE + (cx + 0.5) / N_CURVE_G * 2 * FOV_CURVE;
        D[cy * N_CURVE_G + cx] = compute_Dbar(0, 0, x, y, F1, F2);
      }
    }
    /* Marching-squares cell-fraction area integration */
    let area = 0;
    for (let cy = 0; cy < N_CURVE_G - 1; cy++) {
      const r0 = cy * N_CURVE_G, r1 = (cy + 1) * N_CURVE_G;
      for (let cx = 0; cx < N_CURVE_G - 1; cx++) {
        area += area_above_zero(D[r0 + cx],     D[r0 + cx + 1],
                                D[r1 + cx],     D[r1 + cx + 1]) * cellArea;
      }
    }
    dCcurve.deltas[jr] = Math.sqrt(area / Math.PI) / RAYLEIGH;
  }
}

/* ---- Diverging colormap fixed to [-1, +1] ---- */
function diverging_rgb(v) {
  let t = v;
  if (t < -1) t = -1; else if (t > 1) t = 1;
  const grey = [142, 145, 130];
  const yel  = [245, 220,  80];
  const dark = [ 38,  55,  85];
  if (t >= 0) {
    return [
      (grey[0] + (yel[0] - grey[0]) * t) | 0,
      (grey[1] + (yel[1] - grey[1]) * t) | 0,
      (grey[2] + (yel[2] - grey[2]) * t) | 0,
    ];
  }
  const u = -t;
  return [
    (grey[0] + (dark[0] - grey[0]) * u) | 0,
    (grey[1] + (dark[1] - grey[1]) * u) | 0,
    (grey[2] + (dark[2] - grey[2]) * u) | 0,
  ];
}

/* ---- Marching squares iso-contour ---- */
function drawIsoContour(ctx, grid, nx, ny, level, x0, y0, dx, dy) {
  ctx.beginPath();
  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const v00 = grid[ j      * nx +  i     ];
      const v10 = grid[ j      * nx + (i + 1)];
      const v01 = grid[(j + 1) * nx +  i     ];
      const v11 = grid[(j + 1) * nx + (i + 1)];
      const idx = (v00 > level ? 1 : 0)
                | (v10 > level ? 2 : 0)
                | (v11 > level ? 4 : 0)
                | (v01 > level ? 8 : 0);
      if (idx === 0 || idx === 15) continue;
      const x = x0 + i * dx;
      const y = y0 + j * dy;
      const lerp = (a, b) => (level - a) / (b - a);
      const e0x = x + dx * lerp(v00, v10), e0y = y;
      const e1x = x + dx,                  e1y = y + dy * lerp(v10, v11);
      const e2x = x + dx * lerp(v01, v11), e2y = y + dy;
      const e3x = x,                       e3y = y + dy * lerp(v00, v01);
      switch (idx) {
        case  1: case 14: ctx.moveTo(e3x, e3y); ctx.lineTo(e0x, e0y); break;
        case  2: case 13: ctx.moveTo(e0x, e0y); ctx.lineTo(e1x, e1y); break;
        case  3: case 12: ctx.moveTo(e3x, e3y); ctx.lineTo(e1x, e1y); break;
        case  4: case 11: ctx.moveTo(e1x, e1y); ctx.lineTo(e2x, e2y); break;
        case  6: case  9: ctx.moveTo(e0x, e0y); ctx.lineTo(e2x, e2y); break;
        case  7: case  8: ctx.moveTo(e2x, e2y); ctx.lineTo(e3x, e3y); break;
        case  5:
          ctx.moveTo(e3x, e3y); ctx.lineTo(e0x, e0y);
          ctx.moveTo(e1x, e1y); ctx.lineTo(e2x, e2y); break;
        case 10:
          ctx.moveTo(e0x, e0y); ctx.lineTo(e1x, e1y);
          ctx.moveTo(e2x, e2y); ctx.lineTo(e3x, e3y); break;
      }
    }
  }
  ctx.stroke();
}

/* ---- Mouse plumbing ---- */
let mouseAbs = null, pressed = false, held = false, released = false;
let dragSrc = null, dragCv = null;
document.addEventListener('mousemove', e => { mouseAbs = { x: e.clientX, y: e.clientY }; });
document.addEventListener('mousedown', () => { pressed = true; held = true; });
document.addEventListener('mouseup',   () => {
  released = true; held = false;
  /* Source movement, classification-dot drag, and Dratio-dot drag all
   * change positions/fluxes/σ and therefore invalidate the D-map and the
   * MC bars. */
  if (dragSrc === 'P1' || dragSrc === 'P2'
   || dragSrc === 'CLASS' || dragSrc === 'DRATIO') {
    dmapDirty = true;
    mcDirty   = true;
  }
});
function localMouse(cv) {
  if (!mouseAbs) return null;
  const r = cv.getBoundingClientRect();
  return { x: mouseAbs.x - r.left, y: mouseAbs.y - r.top };
}

/* Generic FOV drag handler shared by cvFov and cvP.
 *
 * The plot area is a centred square of side sz = min(cssW, cssH) anchored
 * at (x0, y0) within the canvas — without this, a layout that gives the
 * canvas cssH < cssW (e.g. row-1 plot 1, where the row's other labels
 * stretch the row taller) makes mp.y / cssW < 1, which collapses the
 * y-update to the upper half of the FOV.  Only the canvas that owns the
 * drag (dragCv) writes to S so the *other* draw function (which sees the
 * mouse outside its own bounds) can't clobber the position. */
function handleSourceDrag(cv) {
  const cssW = cv.clientWidth, cssH = cv.clientHeight;
  if (!cssW || !cssH) {
    return { p1sx: 0, p1sy: 0, p2sx: 0, p2sy: 0, near: null,
             x0: 0, y0: 0, sz: 0 };
  }
  const sz = Math.min(cssW, cssH);
  const x0 = (cssW - sz) / 2, y0 = (cssH - sz) / 2;

  const f2sx = (x) => x0 + (x + FOV) / (2 * FOV) * sz;
  const f2sy = (y) => y0 + sz - (y + FOV) / (2 * FOV) * sz;
  const p1sx = f2sx(S.P1x), p1sy = f2sy(S.P1y);
  const p2sx = f2sx(S.P2x), p2sy = f2sy(S.P2y);

  const mp = localMouse(cv);
  const inside = mp && mp.x >= x0 && mp.x <= x0 + sz
                    && mp.y >= y0 && mp.y <= y0 + sz;
  let near = null;
  if (inside) {
    const d1 = (mp.x - p1sx) ** 2 + (mp.y - p1sy) ** 2;
    const d2 = (mp.x - p2sx) ** 2 + (mp.y - p2sy) ** 2;
    if (d1 < 121 && d1 <= d2) near = 'P1';
    else if (d2 < 121)        near = 'P2';
  }
  if (mp && pressed && !dragSrc && near && inside) {
    dragSrc = near;
    dragCv  = cv;
  }
  if (mp && held && cv === dragCv && (dragSrc === 'P1' || dragSrc === 'P2')) {
    const sx = mp.x - x0, sy = mp.y - y0;
    const nx =  Math.max(-FOV, Math.min(FOV,  sx / sz * 2 * FOV - FOV));
    const ny =  Math.max(-FOV, Math.min(FOV, -((sy / sz) * 2 * FOV - FOV)));
    if (dragSrc === 'P1') { S.P1x = nx; S.P1y = ny; }
    else                  { S.P2x = nx; S.P2y = ny; }
    dmapDirty = true;
  }
  const showMove = (cv === dragCv && (dragSrc === 'P1' || dragSrc === 'P2'))
                || (!dragSrc && near);
  cv.style.cursor = showMove ? 'move' : 'default';
  return { p1sx, p1sy, p2sx, p2sy, near, x0, y0, sz };
}

/* ---- Plot 1 (FOV): grid + axes + draggable markers ---- */
function draw_FOV() {
  const cv = cvFov;
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth, cssH = cv.clientHeight;
  const cw = Math.round(cssW * dpr), ch = Math.round(cssH * dpr);
  if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const { p1sx, p1sy, p2sx, p2sy, near, x0, y0, sz } = handleSourceDrag(cv);
  if (!sz) return;

  /* Background fills the whole canvas; the plot area is a centred square. */
  ctx.fillStyle = getCSSColor('--canvas-bg');
  ctx.fillRect(0, 0, cssW, cssH);

  /* Gridlines at integer multiples of δ_R within the FOV */
  ctx.strokeStyle = getCSSColor('--canvas-grid');
  ctx.lineWidth = 1;
  ctx.beginPath();
  const f2sx = (x) => x0 + (x + FOV) / (2 * FOV) * sz;
  const f2sy = (y) => y0 + sz - (y + FOV) / (2 * FOV) * sz;
  for (let t = -1; t <= 1; t++) {
    if (t === 0) continue;
    const gx = f2sx(t * RAYLEIGH), gy = f2sy(t * RAYLEIGH);
    ctx.moveTo(gx, y0); ctx.lineTo(gx, y0 + sz);
    ctx.moveTo(x0, gy); ctx.lineTo(x0 + sz, gy);
  }
  ctx.stroke();

  /* Origin axes */
  ctx.strokeStyle = getCSSColor('--canvas-axis');
  ctx.beginPath();
  ctx.moveTo(f2sx(0), y0); ctx.lineTo(f2sx(0), y0 + sz);
  ctx.moveTo(x0, f2sy(0)); ctx.lineTo(x0 + sz, f2sy(0));
  ctx.stroke();

  ctx.strokeStyle = getCSSColor('--border');
  ctx.strokeRect(x0, y0, sz, sz);

  /* Faint dashed circle at r = δ_R around each source */
  const ppu = sz / (2 * FOV);
  const RED = 'rgb(220, 80, 80)', BLUE = 'rgb(80, 130, 220)';
  ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.globalAlpha = 0.55;
  ctx.strokeStyle = RED;
  ctx.beginPath(); ctx.arc(p1sx, p1sy, RAYLEIGH * ppu, 0, Math.PI * 2); ctx.stroke();
  ctx.strokeStyle = BLUE;
  ctx.beginPath(); ctx.arc(p2sx, p2sy, RAYLEIGH * ppu, 0, Math.PI * 2); ctx.stroke();
  ctx.setLineDash([]); ctx.globalAlpha = 1.0;

  const r1 = (dragSrc === 'P1' || near === 'P1') ? 7 : 5;
  const r2 = (dragSrc === 'P2' || near === 'P2') ? 7 : 5;
  ctx.fillStyle = RED;
  ctx.beginPath(); ctx.arc(p1sx, p1sy, r1, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = BLUE;
  ctx.beginPath(); ctx.arc(p2sx, p2sy, r2, 0, Math.PI * 2); ctx.fill();
}

/* ---- Plot 2 (aperture slot): pupil-mask render of the active aperture ---- */
function draw_Slot() {
  const cv = cvSlot;
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth, cssH = cv.clientHeight;
  const cw = Math.round(cssW * dpr), ch = Math.round(cssH * dpr);
  if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
  const ctx = cv.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);

  const ap   = Apertures.active();
  const apFn = ap.apFn;
  /* Rotate the pupil mask to match the active θ (image-plane rotation).
   * The pupil rotates with the image plane, so we sample apFn at the
   * inverse-rotated (u, v).  cosT, sinT below are for -θ. */
  const theta = Apertures.getAngle();
  const cosT  =  Math.cos(theta);
  const sinT  =  Math.sin(theta);

  /* Pupil-plane visualisation: bright pixels where mask = 1. */
  const img = ctx.createImageData(cw, ch);
  const halfW = cw / 2, halfH = ch / 2;
  const pupilHalf = 0.91;           /* 1.3× unzoom from 0.7 — pulls aperture off the border */
  for (let cy = 0; cy < ch; cy++) {
    const v = (halfH - cy) / halfH * pupilHalf;
    for (let cx = 0; cx < cw; cx++) {
      const u = (cx - halfW) / halfW * pupilHalf;
      const ur =  cosT * u + sinT * v;
      const vr = -sinT * u + cosT * v;
      const inside = apFn(ur, vr) > 0;
      const idx = (cy * cw + cx) * 4;
      if (inside) {
        img.data[idx]     = 245; img.data[idx + 1] = 220; img.data[idx + 2] = 80;
      } else {
        img.data[idx]     = 22;  img.data[idx + 1] = 24;  img.data[idx + 2] = 32;
      }
      img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.strokeStyle = getCSSColor('--border');
  ctx.lineWidth   = 1;
  ctx.strokeRect(0, 0, cssW, cssH);

  ctx.fillStyle = 'rgb(245, 220, 80)';
  ctx.font = '12px sans-serif';
  const lbl = Apertures.apertureLabels[ap.name] || ap.name;
  ctx.fillText(lbl, 8, cssH - 8);
}

/* ---- Plot 3 (noisy): pixelated mu_P + sigma noise ---- */
function draw_Noisy() {
  ensureNoise();
  const cv = cvNoisy;
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
      let v = (muPix[pi * PI_PIX + pj] + S.sigma * noiseSamples[pi * PI_PIX + pj]) * inv;
      if (noisyLog) v = logStretch(v);
      if (v < 0) v = 0; else if (v > 1) v = 1;
      const g = (v * 255) | 0;
      ctx.fillStyle = `rgb(${g},${g},${g})`;
      ctx.fillRect(pj * cell, pi * cell, cell + 0.5, cell + 0.5);
    }
  }
  ctx.strokeStyle = getCSSColor('--border');
  ctx.strokeRect(0, 0, cssW, cssH);
}

/* ---- Plot 4 (clean mu_P): high-res render + draggable sources ---- *
 * mu_P is rendered into a centred square of side sz_css = min(cssW, cssH);
 * the rest of the canvas is filled with the bg colour.  Same pattern as
 * draw_FOV. */
function draw_P() {
  const cv = cvP;
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth, cssH = cv.clientHeight;
  const cw = Math.round(cssW * dpr), ch = Math.round(cssH * dpr);
  if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
  const ctx = cv.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);

  const sz_css = Math.min(cssW, cssH);
  const x0_css = (cssW - sz_css) / 2, y0_css = (cssH - sz_css) / 2;
  const sz_px  = Math.max(1, Math.round(sz_css * dpr));
  const x0_px  = Math.round(x0_css * dpr);
  const y0_px  = Math.round(y0_css * dpr);

  ctx.fillStyle = getCSSColor('--canvas-bg');
  ctx.fillRect(0, 0, cw, ch);

  const img = ctx.createImageData(sz_px, sz_px);
  const inv = 1.0 / pMax;
  for (let cy = 0; cy < sz_px; cy++) {
    let i = (cy * PI_FINE / sz_px) | 0; if (i >= PI_FINE) i = PI_FINE - 1;
    const row = i * PI_FINE;
    for (let cx = 0; cx < sz_px; cx++) {
      let j = (cx * PI_FINE / sz_px) | 0; if (j >= PI_FINE) j = PI_FINE - 1;
      let t = muFine[row + j] * inv;
      if (pLog) t = logStretch(t);
      if (t < 0) t = 0; else if (t > 1) t = 1;
      const g = (t * 255) | 0;
      const idx = (cy * sz_px + cx) * 4;
      img.data[idx]     = g;
      img.data[idx + 1] = g;
      img.data[idx + 2] = g;
      img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, x0_px, y0_px);

  ctx.save();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const { p1sx, p1sy, p2sx, p2sy, near, x0, y0, sz } = handleSourceDrag(cv);
  const RED = 'rgb(220, 80, 80)', BLUE = 'rgb(80, 130, 220)';
  const r1 = (dragSrc === 'P1' || near === 'P1') ? 7 : 5;
  const r2 = (dragSrc === 'P2' || near === 'P2') ? 7 : 5;
  ctx.fillStyle = RED;
  ctx.beginPath(); ctx.arc(p1sx, p1sy, r1, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = BLUE;
  ctx.beginPath(); ctx.arc(p2sx, p2sy, r2, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = getCSSColor('--border');
  ctx.strokeRect(x0, y0, sz, sz);
  ctx.restore();
}

/* ---- Plot 5 (D-map) ---- */
function draw_Dmap() {
  const cv = cvDmap;
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth, cssH = cv.clientHeight;
  const cw = Math.round(cssW * dpr), ch = Math.round(cssH * dpr);
  if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
  const ctx = cv.getContext('2d');

  const img = ctx.createImageData(cw, ch);
  for (let cy = 0; cy < ch; cy++) {
    let i = (cy * N_DGRID / ch) | 0; if (i >= N_DGRID) i = N_DGRID - 1;
    const row = i * N_DGRID;
    for (let cx = 0; cx < cw; cx++) {
      let j = (cx * N_DGRID / cw) | 0; if (j >= N_DGRID) j = N_DGRID - 1;
      const v = Dmap[row + j];
      const rgb = diverging_rgb(v);
      const idx = (cy * cw + cx) * 4;
      img.data[idx]     = rgb[0];
      img.data[idx + 1] = rgb[1];
      img.data[idx + 2] = rgb[2];
      img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);

  ctx.save();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const sz = cssW;
  const cell = sz / N_DGRID;

  ctx.strokeStyle = 'rgb(220, 60, 60)';
  ctx.lineWidth = 2;
  drawIsoContour(ctx, Dmap, N_DGRID, N_DGRID, 0,
                 cell / 2, cell / 2, cell, cell);

  const f2sx = (x) => (x + FOV) / (2 * FOV) * sz;
  const f2sy = (y) => sz - (y + FOV) / (2 * FOV) * sz;
  const p1sx = f2sx(S.P1x), p1sy = f2sy(S.P1y);
  const p2sx = f2sx(S.P2x), p2sy = f2sy(S.P2y);

  ctx.strokeStyle = 'rgb(255, 255, 255)'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(p2sx, p2sy, 7, 0, Math.PI * 2); ctx.stroke();
  ctx.fillStyle = 'rgb(80, 130, 220)';
  ctx.beginPath(); ctx.arc(p2sx, p2sy, 4, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = 'rgb(255, 255, 255)';
  ctx.beginPath(); ctx.arc(p1sx, p1sy, 7, 0, Math.PI * 2); ctx.stroke();
  ctx.fillStyle = 'rgb(220, 80, 80)';
  ctx.beginPath(); ctx.arc(p1sx, p1sy, 4, 0, Math.PI * 2); ctx.fill();

  ctx.strokeStyle = getCSSColor('--border');
  ctx.strokeRect(0, 0, cssW, cssW);
  ctx.restore();
}

/* ---- Plot 6 (D-bar vs (δ, ratio)) ---- */
function draw_DRatio() {
  const cv = cvR;
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth, cssH = cv.clientHeight;
  const cw = Math.round(cssW * dpr), ch = Math.round(cssH * dpr);
  if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
  const ctx = cv.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);

  const padL = 36, padR = 8, padT = 14, padB = 22;
  const dprPadL = padL * dpr, dprPadR = padR * dpr;
  const dprPadT = padT * dpr, dprPadB = padB * dpr;
  const px = dprPadL, py = dprPadT;
  const pw = cw - dprPadL - dprPadR;
  const ph = ch - dprPadT - dprPadB;

  ctx.fillStyle = getCSSColor('--canvas-bg');
  ctx.fillRect(0, 0, cw, ch);

  const img = ctx.createImageData(pw | 0, ph | 0);
  const pwI = img.width, phI = img.height;
  for (let cy = 0; cy < phI; cy++) {
    let jr = (cy * N_RATIO / phI) | 0; if (jr >= N_RATIO) jr = N_RATIO - 1;
    const row = jr * N_DELTA;
    for (let cx = 0; cx < pwI; cx++) {
      let id = (cx * N_DELTA / pwI) | 0; if (id >= N_DELTA) id = N_DELTA - 1;
      const v = Dratio[row + id];
      const rgb = diverging_rgb(v);
      const idx = (cy * pwI + cx) * 4;
      img.data[idx]     = rgb[0];
      img.data[idx + 1] = rgb[1];
      img.data[idx + 2] = rgb[2];
      img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, px, py);

  ctx.save();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const cssPx = padL, cssPy = padT;
  const cssPw = cssW - padL - padR;
  const cssPh = cssH - padT - padB;

  ctx.strokeStyle = 'rgb(220, 60, 60)';
  ctx.lineWidth = 2;
  const dx = cssPw / (N_DELTA - 1);
  const dy = cssPh / (N_RATIO - 1);
  drawIsoContour(ctx, Dratio, N_DELTA, N_RATIO, 0,
                 cssPx, cssPy, dx, dy);

  /* Green dashed: area-based δ_1(ratio) from the per-aperture build.
   * dCcurve.deltas[jr] is stored in δ_R units; convert to λ/d (× RAYLEIGH)
   * to match the x-axis in plot units. */
  ctx.strokeStyle = 'rgb(80, 200, 110)';
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  let pen = false;
  for (let jr = 0; jr < N_CURVE_R; jr++) {
    const r = dCcurve.ratios[jr];
    const d_lam = dCcurve.deltas[jr] * RAYLEIGH;     /* λ/d */
    const visible = d_lam > 0 && d_lam <= DELTA_MAX;
    if (visible) {
      const xPx = cssPx + (d_lam / DELTA_MAX) * cssPw;
      const yPx = cssPy + (1 - r) * cssPh;
      if (pen) ctx.lineTo(xPx, yPx);
      else { ctx.moveTo(xPx, yPx); pen = true; }
    } else {
      pen = false;
    }
  }
  ctx.stroke();
  ctx.setLineDash([]);

  const dNow  = Math.hypot(S.P2x - S.P1x, S.P2y - S.P1y);   /* λ/d */
  const ratio = Math.min(S.FP1, S.FP2) / Math.max(S.FP1, S.FP2);
  const cxd = cssPx + Math.max(0, Math.min(1, dNow / DELTA_MAX)) * cssPw;
  const cyd = cssPy + (1 - ratio) * cssPh;

  /* Mouse hit-test on the dot. */
  const mpcp = localMouse(cv);
  let nearD = false;
  if (mpcp) {
    const d2 = (mpcp.x - cxd) ** 2 + (mpcp.y - cyd) ** 2;
    nearD = d2 < 121;
  }
  const rOuter = (dragSrc === 'DRATIO' || nearD) ? 10 : 8;
  const rInner = (dragSrc === 'DRATIO' || nearD) ?  6 : 5;
  ctx.fillStyle   = 'rgb(245, 200, 70)';
  ctx.strokeStyle = 'rgb(245, 200, 70)';
  ctx.lineWidth   = 1.5;
  ctx.beginPath(); ctx.arc(cxd, cyd, rInner, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(cxd, cyd, rOuter, 0, Math.PI * 2); ctx.stroke();

  /* Drag start */
  if (mpcp && pressed && !dragSrc && nearD) dragSrc = 'DRATIO';

  /* Drag updates: x → new Δ along current segment direction; y → new
   * ratio = min(F_P1,F_P2)/max(F_P1,F_P2), preserving F_P1² + F_P2². */
  if (mpcp && held && dragSrc === 'DRATIO') {
    const mx = Math.max(cssPx, Math.min(cssPx + cssPw, mpcp.x));
    const my = Math.max(cssPy, Math.min(cssPy + cssPh, mpcp.y));

    const newDelta = (mx - cssPx) / cssPw * DELTA_MAX;   /* λ/d */
    let   newRatio    = 1 - (my - cssPy) / cssPh;
    if (newRatio < 0.001) newRatio = 0.001;
    if (newRatio > 1)     newRatio = 1;

    /* Rescale (p1, p2) symmetrically about midpoint. */
    const dxp = S.P2x - S.P1x, dyp = S.P2y - S.P1y;
    const len = Math.hypot(dxp, dyp);
    let ux = 1, uy = 0;
    if (len > 1e-9) { ux = dxp / len; uy = dyp / len; }
    const cxm = 0.5 * (S.P1x + S.P2x);
    const cym = 0.5 * (S.P1y + S.P2y);
    S.P1x = Math.max(-FOV, Math.min(FOV, cxm - 0.5 * newDelta * ux));
    S.P1y = Math.max(-FOV, Math.min(FOV, cym - 0.5 * newDelta * uy));
    S.P2x = Math.max(-FOV, Math.min(FOV, cxm + 0.5 * newDelta * ux));
    S.P2y = Math.max(-FOV, Math.min(FOV, cym + 0.5 * newDelta * uy));

    /* Solve {F_max² + (F_max·r)² = R²} → F_max = R/√(1+r²), F_min = r·F_max,
     * preserving the current ‖F‖² = F_P1² + F_P2² and which source is bright. */
    const Rnorm = Math.sqrt(S.FP1*S.FP1 + S.FP2*S.FP2);
    const Fmax  = Rnorm / Math.sqrt(1 + newRatio * newRatio);
    const Fmin  = Fmax * newRatio;
    const clampF = (f) => Math.max(0.05, Math.min(2, f));
    if (S.FP1 >= S.FP2) { S.FP1 = clampF(Fmax); S.FP2 = clampF(Fmin); }
    else                { S.FP1 = clampF(Fmin); S.FP2 = clampF(Fmax); }

    /* Sync FP1 / FP2 sliders. */
    const f1Slider = document.getElementById('airy-FP1');
    const f1Label  = document.getElementById('airy-FP1-val');
    const f2Slider = document.getElementById('airy-FP2');
    const f2Label  = document.getElementById('airy-FP2-val');
    if (f1Slider) f1Slider.value = S.FP1;
    if (f1Label)  f1Label.textContent = S.FP1.toFixed(2);
    if (f2Slider) f2Slider.value = S.FP2;
    if (f2Label)  f2Label.textContent = S.FP2.toFixed(2);
  }
  cv.style.cursor = (nearD || dragSrc === 'DRATIO') ? 'move' : 'default';

  ctx.strokeStyle = getCSSColor('--border');
  ctx.lineWidth = 1;
  ctx.strokeRect(cssPx, cssPy, cssPw, cssPh);

  ctx.fillStyle = getCSSColor('--text-soft');
  ctx.font = '10px monospace';
  ctx.fillText('0',                            cssPx - 4,             cssPy + cssPh + 14);
  ctx.fillText((DELTA_MAX / 2).toFixed(1),     cssPx + cssPw / 2 - 8, cssPy + cssPh + 14);
  ctx.fillText(DELTA_MAX.toFixed(1),           cssPx + cssPw - 12,    cssPy + cssPh + 14);
  ctx.fillText('1.0',                          cssPx - 22,            cssPy + 8);
  ctx.fillText('0.5',                          cssPx - 22,            cssPy + cssPh / 2);
  ctx.fillText('0.0',                          cssPx - 22,            cssPy + cssPh - 2);

  ctx.restore();
}

/* ---- Plot 7 (best-fit μ̂): F̂_1·m_1 + F̂_2·m_2 in a centred square ---- */
let bestLog = false;
function draw_BestFit() {
  const cv = cvBest;
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth, cssH = cv.clientHeight;
  const cw = Math.round(cssW * dpr), ch = Math.round(cssH * dpr);
  if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
  const ctx = cv.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);

  const sz_css = Math.min(cssW, cssH);
  const x0_css = (cssW - sz_css) / 2, y0_css = (cssH - sz_css) / 2;
  const sz_px  = Math.max(1, Math.round(sz_css * dpr));
  const x0_px  = Math.round(x0_css * dpr);
  const y0_px  = Math.round(y0_css * dpr);

  ctx.fillStyle = getCSSColor('--canvas-bg');
  ctx.fillRect(0, 0, cw, ch);

  const inv = 1.0 / Math.max(pMax, bestMuMax * 0.01, 1e-9);
  const img = ctx.createImageData(sz_px, sz_px);
  for (let cy = 0; cy < sz_px; cy++) {
    let i = (cy * PI_FINE / sz_px) | 0; if (i >= PI_FINE) i = PI_FINE - 1;
    const row = i * PI_FINE;
    for (let cx = 0; cx < sz_px; cx++) {
      let j = (cx * PI_FINE / sz_px) | 0; if (j >= PI_FINE) j = PI_FINE - 1;
      let t = muFitFine[row + j] * inv;
      if (bestLog) t = logStretch(t);
      if (t < 0) t = 0; else if (t > 1) t = 1;
      const g = (t * 255) | 0;
      const idx = (cy * sz_px + cx) * 4;
      img.data[idx]     = g;
      img.data[idx + 1] = g;
      img.data[idx + 2] = g;
      img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, x0_px, y0_px);

  ctx.save();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  /* Markers:
   *   - filled dots at the TRUE source positions (ground truth)
   *   - target reticle at each MC-mean fit position
   *   - 1, 2, 3σ confidence ellipses from the MC positional covariance
   *     (2D Gaussian fit to the (q_x, q_y) sample cloud) */
  const f2sx = (x) => x0_css + (x + FOV) / (2 * FOV) * sz_css;
  const f2sy = (y) => y0_css + sz_css - (y + FOV) / (2 * FOV) * sz_css;
  const colMap = { 'src1': 'rgb(220, 80, 80)',
                   'src2': 'rgb(80, 130, 220)',
                   'Qstar': 'rgb(245, 200, 70)' };
  const k_per = sz_css / (2 * FOV);   /* px per λ/d */

  /* Truth dots — always visible, regardless of fit winner. */
  ctx.fillStyle = 'rgb(220, 80, 80)';
  ctx.beginPath(); ctx.arc(f2sx(S.P1x), f2sy(S.P1y), 4.5, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = 'rgb(80, 130, 220)';
  ctx.beginPath(); ctx.arc(f2sx(S.P2x), f2sy(S.P2y), 4.5, 0, Math.PI * 2); ctx.fill();

  /* Per-fit reticle + ellipses, anchored at the MC mean position.
   * Reticle is just a small crosshair (no outer circle) so it doesn't
   * occlude small ellipses for tightly-clustered low-noise fits. */
  function drawReticle(xs, ys, col) {
    ctx.strokeStyle = col;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(xs - 7, ys); ctx.lineTo(xs - 2, ys);
    ctx.moveTo(xs + 2, ys); ctx.lineTo(xs + 7, ys);
    ctx.moveTo(xs, ys - 7); ctx.lineTo(xs, ys - 2);
    ctx.moveTo(xs, ys + 2); ctx.lineTo(xs, ys + 7);
    ctx.stroke();
  }
  function tracePath(mx_phys, my_phys, a_phys, b_phys, theta_phys) {
    const N = 64;
    const cosT = Math.cos(theta_phys), sinT = Math.sin(theta_phys);
    ctx.beginPath();
    for (let i = 0; i <= N; i++) {
      const t = (i / N) * 2 * Math.PI;
      const ct = Math.cos(t), st = Math.sin(t);
      const dx_phys = a_phys * ct * cosT - b_phys * st * sinT;
      const dy_phys = a_phys * ct * sinT + b_phys * st * cosT;
      const xs = f2sx(mx_phys + dx_phys);
      const ys = f2sy(my_phys + dy_phys);
      if (i === 0) ctx.moveTo(xs, ys); else ctx.lineTo(xs, ys);
    }
  }
  function drawEllipse(mx, my, a, b, theta, col, alpha) {
    /* Dark backing stroke for contrast on bright fit peaks, then the
     * coloured stroke on top. */
    tracePath(mx, my, a, b, theta);
    ctx.save();
    ctx.globalAlpha = 0.7 * alpha;
    ctx.strokeStyle = 'rgb(20, 22, 30)';
    ctx.lineWidth = 4.0;
    ctx.stroke();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = col;
    ctx.lineWidth = 2.2;
    ctx.stroke();
    ctx.restore();
  }

  for (const f of Best.mcFits) {
    if (!f || f.n === 0) continue;
    const col = colMap[f.col] || 'rgb(245, 200, 70)';
    const mx = f.qxMean, my = f.qyMean;
    const a1 = f.ellipseSemiMajor || 0;
    const b1 = f.ellipseSemiMinor || 0;
    const th = f.ellipseAngle || 0;
    /* Reticle first so the ellipse renders on top of it. */
    drawReticle(f2sx(mx), f2sy(my), col);
    if (a1 > 1e-9) {
      drawEllipse(mx, my, 3 * a1, 3 * b1, th, col, 1.00);
    }
  }
  void k_per;  /* reserved if we ever switch ellipses to canvas-space scaling */

  /* Diamond markers for the CURRENT noise sample's fit positions, with the
   * fitted flux F̂ as a small white label.  These jiggle on each "↻ new
   * sample" click — they show where this realisation landed relative to
   * the MC-mean reticle and the 3σ ellipse. */
  function drawDiamond(xs, ys, col) {
    ctx.save();
    ctx.fillStyle = col;
    ctx.strokeStyle = 'rgb(20, 22, 30)';
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(xs,     ys - 6);
    ctx.lineTo(xs + 6, ys);
    ctx.lineTo(xs,     ys + 6);
    ctx.lineTo(xs - 6, ys);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }
  /* When the winner is a single-source model, display in Q⋆ yellow
   * regardless of which starter (p1/p2/Qstar) won. */
  for (const f of Best.currentFits) {
    const col = (Best.currentN === 1)
              ? 'rgb(245, 200, 70)'
              : (colMap[f.col] || 'rgb(245, 200, 70)');
    drawDiamond(f2sx(f.qx), f2sy(f.qy), col);
  }

  /* Best-model badge in the top-left corner. */
  const tagMap = { 'empty': 'best: ∅', 'p1': 'best: {p₁}',
                   'p2': 'best: {p₂}', 'Qstar': 'best: Q⋆', 'P': 'best: P' };
  const tag = tagMap[Best.currentName] || 'best';
  ctx.fillStyle = 'rgba(20, 20, 30, 0.7)';
  ctx.fillRect(x0_css + 6, y0_css + 6, 80, 16);
  ctx.fillStyle = 'rgb(245, 220, 80)';
  ctx.font = '11px sans-serif';
  ctx.fillText(tag, x0_css + 11, y0_css + 18);

  ctx.strokeStyle = getCSSColor('--border');
  ctx.strokeRect(x0_css, y0_css, sz_css, sz_css);
  ctx.restore();
}

/* ---- Plot 8 (flux comparison): actual vs best-model retrieved ± 2σ ---- *
 * Three columns: src 1, src 2, Q⋆.  Actuals (open square + tick) shown
 * at src 1 and src 2.  Retrieved values (filled disk + 2σ bar) shown
 * only in the columns the winning model has parameters for:
 *   N=2 → src 1 and src 2 columns
 *   N=1 → Q⋆ column
 *   N=0 → no retrievals at all. */
function draw_Flux() {
  const cv = cvFlux;
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth, cssH = cv.clientHeight;
  const cw = Math.round(cssW * dpr), ch = Math.round(cssH * dpr);
  if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.fillStyle = getCSSColor('--canvas-bg');
  ctx.fillRect(0, 0, cssW, cssH);

  const padL = 36, padR = 12, padT = 18, padB = 24;
  const px = padL, py = padT;
  const pw = cssW - padL - padR;
  const ph = cssH - padT - padB;

  /* Y range: cover actuals and the MC retrieved ± 3σ. */
  let lo_data = Math.min(0, S.FP1, S.FP2);
  let hi_data = Math.max(S.FP1, S.FP2, 1.0);
  for (const f of Best.mcFits) {
    const m = f.Fmean ?? 0, s = f.Fstd ?? 0;
    lo_data = Math.min(lo_data, m - 3.0 * s);
    hi_data = Math.max(hi_data, m + 3.0 * s);
  }
  const margin = 0.1 * (hi_data - lo_data + 1e-3);
  const yLo = lo_data - margin;
  const yHi = hi_data + margin;
  const y2px = (y) => py + ph - (y - yLo) / (yHi - yLo) * ph;

  /* Zero line */
  if (yLo <= 0 && yHi >= 0) {
    ctx.strokeStyle = getCSSColor('--canvas-axis');
    ctx.lineWidth = 1;
    ctx.beginPath();
    const y0 = y2px(0);
    ctx.moveTo(px, y0); ctx.lineTo(px + pw, y0);
    ctx.stroke();
  }

  /* Three columns at 1/6, 3/6, 5/6 of plot width. */
  const x1 = px + pw * (1 / 6);
  const x2 = px + pw * (3 / 6);
  const xQ = px + pw * (5 / 6);

  function drawActual(xColumn, F_actual, color) {
    const yA = y2px(F_actual);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.rect(xColumn - 6, yA - 6, 12, 12);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(xColumn - 14, yA); ctx.lineTo(xColumn + 14, yA);
    ctx.stroke();
  }
  function drawErrorBar(xColumn, F_mean, F_std, color) {
    const yMean = y2px(F_mean);
    const yLo = y2px(F_mean - 3 * F_std);
    const yHi = y2px(F_mean + 3 * F_std);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(xColumn, yLo);     ctx.lineTo(xColumn, yHi);
    ctx.moveTo(xColumn - 6, yLo); ctx.lineTo(xColumn + 6, yLo);
    ctx.moveTo(xColumn - 6, yHi); ctx.lineTo(xColumn + 6, yHi);
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(xColumn, yMean, 3.5, 0, Math.PI * 2); ctx.fill();
  }

  /* Actuals always shown at src 1, src 2.  MC error bars placed at the
   * column corresponding to each fitted source's `col` field. */
  drawActual(x1, S.FP1, 'rgb(220, 80, 80)');
  drawActual(x2, S.FP2, 'rgb(80, 130, 220)');
  const colX     = { 'src1': x1, 'src2': x2, 'Qstar': xQ };
  const colColor = { 'src1': 'rgb(220, 80, 80)',
                     'src2': 'rgb(80, 130, 220)',
                     'Qstar': 'rgb(245, 200, 70)' };
  for (const f of Best.mcFits) {
    const baseX = colX[f.col]     ?? xQ;
    const color = colColor[f.col] ?? 'rgb(245, 200, 70)';
    /* Offset retrieved bar by +18 from actual marker for src1/src2 so they
     * don't overlap; place Q⋆ retrieval centred on its column. */
    const xBar  = (f.col === 'Qstar') ? baseX : baseX + 18;
    drawErrorBar(xBar, f.Fmean ?? 0, f.Fstd ?? 0, color);
  }

  /* Diamond markers for the CURRENT noise sample's retrieved fluxes — same
   * column logic as the error bars, slightly offset to the right of the
   * 3σ bar so they don't overlap. */
  function drawFluxDiamond(xCol, F_now, color) {
    const yc = y2px(F_now);
    ctx.save();
    ctx.fillStyle = color;
    ctx.strokeStyle = 'rgb(20, 22, 30)';
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(xCol,     yc - 6);
    ctx.lineTo(xCol + 6, yc);
    ctx.lineTo(xCol,     yc + 6);
    ctx.lineTo(xCol - 6, yc);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }
  /* Single-source winners always render in Q⋆'s column + yellow, even
   * when the per-frame starter was {p₁} or {p₂}. */
  for (const f of Best.currentFits) {
    const colKey = (Best.currentN === 1) ? 'Qstar' : f.col;
    const baseX  = colX[colKey]     ?? xQ;
    const color  = colColor[colKey] ?? 'rgb(245, 200, 70)';
    const xDia   = (colKey === 'Qstar') ? baseX + 14 : baseX + 18 + 14;
    drawFluxDiamond(xDia, f.F, color);
  }

  /* Frame */
  ctx.strokeStyle = getCSSColor('--border');
  ctx.lineWidth = 1;
  ctx.strokeRect(px, py, pw, ph);

  /* Axis labels */
  ctx.fillStyle = getCSSColor('--text-soft');
  ctx.font = '10px monospace';
  ctx.fillText(yHi.toFixed(2), 4, py + 8);
  if (yLo <= 0 && yHi >= 0) ctx.fillText('0', 24, y2px(0) - 2);
  ctx.fillText(yLo.toFixed(2), 4, py + ph - 2);
  ctx.fillText('src 1', x1 - 12, py + ph + 14);
  ctx.fillText('src 2', x2 - 12, py + ph + 14);
  ctx.fillText('Q⋆',    xQ - 8,  py + ph + 14);

  /* Header — aggregated MC histogram across three model classes:
   *   ∅ (no source) | Q⋆ (any 1-source) | P (2 sources).
   * The {p₁}, {p₂} and Q⋆ starters are pooled because they're all
   * 1-source models with the same parsimony cost; their probabilities
   * sum together under "Q⋆".  The bars below show the distribution of
   * the aggregated single-source fit. */
  const labelMap = { 'empty': '∅', 'Qstar': 'Q⋆', 'P': 'P' };
  const order    = ['empty', 'Qstar', 'P'];
  const countsAgg = Best.mcCountsAgg || {};
  const total     = Best.mcTotal     || N_MC;
  const items = order
    .filter(k => (countsAgg[k] || 0) > 0)
    .sort((a, b) => countsAgg[b] - countsAgg[a])
    .map(k => `${labelMap[k]} (${(100 * countsAgg[k] / total).toFixed(1)}%)`);
  const tag = items.length > 0 ? items.join('   ')
                               : `${total} samples`;
  ctx.fillStyle = getCSSColor('--text');
  ctx.font = '11px sans-serif';
  ctx.fillText(tag, px + 4, py - 4);
}

/* ---- Plot 9 (region classification): copy of contam's, with active-aperture k ---- *
 * Renders a column of vertical pixels at each Δ.  Bands top→bottom are
 * grey (no detection) → blue (single source) → red (phantom Q*) → green
 * (full P).  Yellow dot at the user's (Δ, R) point. */
function draw_Class() {
  const cv = cvClass;
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
  const pw = cssW - padL - padR;
  const ph = cssH - padT - padB;

  /* Log-scale R in [0.4, 10]. */
  const lr_min = Math.log10(0.4);
  const lr_max = Math.log10(100);
  const R2y = (R) => py + ph - (Math.log10(Math.max(R, 0.4)) - lr_min)
                              / (lr_max - lr_min) * ph;

  const colGrey  = 'rgb( 80, 84, 96)';
  const colBlue  = 'rgb( 70,110,180)';
  const colRed   = 'rgb(200, 80, 80)';
  const colGreen = 'rgb( 80,180,100)';

  /* Clamp threshold y-pixels to the plot area so coloured columns don't
   * leak into the top/bottom padding when R-values fall off the [0.4, 10]
   * range (R2y maps Rd/Rr/Rc to negative y when the threshold is above
   * the plotted range). */
  const yTop = py;
  const yBot = py + ph;
  const clip = (y) => Math.max(yTop, Math.min(yBot, y));

  for (let xPix = 0; xPix < pw; xPix++) {
    const Delta = (xPix / pw) * DELTA_MAX;
    const tt = Delta / DELTA_MAX * N_CLASS;
    const i_lo = Math.max(0, Math.min(N_CLASS - 1, Math.floor(tt)));
    const frac = tt - i_lo;
    const Rdp = Rd[i_lo] * (1 - frac) + Rd[i_lo + 1] * frac;
    const Rrp = Rr[i_lo] * (1 - frac) + Rr[i_lo + 1] * frac;
    const Rcp = Rc[i_lo] * (1 - frac) + Rc[i_lo + 1] * frac;

    const xp = px + xPix;
    const yd = clip(R2y(Rdp));
    const yr = clip(R2y(Rrp));
    const yc = clip(R2y(Rcp));

    /* Bottom-up: grey (no detect), then {blue, red} or {red only}, then green. */
    ctx.fillStyle = colGrey;  ctx.fillRect(xp, yd, 1, yBot - yd);
    if (Rcp < Rrp) {
      ctx.fillStyle = colBlue; ctx.fillRect(xp, yc, 1, yd - yc);
      ctx.fillStyle = colRed;  ctx.fillRect(xp, yr, 1, yc - yr);
    } else {
      ctx.fillStyle = colBlue; ctx.fillRect(xp, yr, 1, yd - yr);
    }
    ctx.fillStyle = colGreen; ctx.fillRect(xp, yTop, 1, yr - yTop);
  }

  ctx.strokeStyle = getCSSColor('--border');
  ctx.strokeRect(px, py, pw, ph);

  /* Current δ_1 (D̄ = 0 boundary) for the active flux ratio — vertical line.
   * Read from the live Dmap via the same area integral as the readout. */
  const dC_lam_d = compute_dC_area();
  if (isFinite(dC_lam_d) && dC_lam_d > 0 && dC_lam_d <= DELTA_MAX) {
    const xd = px + (dC_lam_d / DELTA_MAX) * pw;
    ctx.save();
    ctx.strokeStyle = 'rgb(220, 120, 220)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(xd, py); ctx.lineTo(xd, py + ph); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgb(220, 120, 220)';
    ctx.font = '10px monospace';
    ctx.fillText('δ₁', xd + 3, py + 10);
    ctx.restore();
  }

  /* User's current (Δ, R) — draggable yellow dot.  R = SNR(P)/η_SNR with
   *   SNR(P) = sqrt(J(P,∅,Σ)) = sqrt(muP_sq) / σ_F̂,
   * and σ_F̂ = σ_pix · dx_pix is the matched-filter retrieval std for a
   * unit-norm PSF in per-pixel-noise space. */
  const F1 = S.FP1, F2 = S.FP2;
  const ap = Apertures.active();
  const eta = ap.eta;
  const Delta_now = Math.hypot(S.P2x - S.P1x, S.P2y - S.P1y);
  const k_now = Apertures.k(ap, Delta_now, 0);
  const muP_sq_now = F1*F1 + F2*F2 + 2 * F1 * F2 * k_now;
  const dx_pix = 2 * FOV / PI_PIX;
  const safeSig = Math.max(S.sigma * dx_pix, 1e-9);
  const R_now = Math.sqrt(muP_sq_now) / (safeSig * eta);

  let xc = -1, yc = -1, near = false;
  if (Delta_now <= DELTA_MAX) {
    xc = px + (Delta_now / DELTA_MAX) * pw;
    yc = R2y(R_now);
    const mp = localMouse(cv);
    if (mp) {
      const d2 = (mp.x - xc) ** 2 + (mp.y - yc) ** 2;
      near = d2 < 121;
    }
    const rOuter = (dragSrc === 'CLASS' || near) ? 10 : 8;
    const rInner = (dragSrc === 'CLASS' || near) ?  6 : 5;
    ctx.fillStyle   = 'rgb(245, 200, 70)';
    ctx.strokeStyle = 'rgb(245, 200, 70)';
    ctx.lineWidth   = 1.5;
    ctx.beginPath(); ctx.arc(xc, yc, rInner, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(xc, yc, rOuter, 0, Math.PI * 2); ctx.stroke();
  }

  /* Drag start */
  const mpcp = localMouse(cv);
  if (mpcp && pressed && !dragSrc && near) dragSrc = 'CLASS';

  /* Drag updates: new Δ from x, new R from y → solve for new σ_pix. */
  if (mpcp && held && dragSrc === 'CLASS') {
    const mx = Math.max(px, Math.min(px + pw, mpcp.x));
    const my = Math.max(py, Math.min(py + ph, mpcp.y));

    const newDelta = (mx - px) / pw * DELTA_MAX;
    const newLr    = (py + ph - my) / ph * (lr_max - lr_min) + lr_min;
    const newR     = Math.pow(10, newLr);

    /* Rescale (p1, p2) symmetrically about their midpoint to give newDelta. */
    const dxp = S.P2x - S.P1x, dyp = S.P2y - S.P1y;
    const len = Math.hypot(dxp, dyp);
    let ux = 1, uy = 0;
    if (len > 1e-9) { ux = dxp / len; uy = dyp / len; }
    const cxm = 0.5 * (S.P1x + S.P2x);
    const cym = 0.5 * (S.P1y + S.P2y);
    S.P1x = Math.max(-FOV, Math.min(FOV, cxm - 0.5 * newDelta * ux));
    S.P1y = Math.max(-FOV, Math.min(FOV, cym - 0.5 * newDelta * uy));
    S.P2x = Math.max(-FOV, Math.min(FOV, cxm + 0.5 * newDelta * ux));
    S.P2y = Math.max(-FOV, Math.min(FOV, cym + 0.5 * newDelta * uy));

    /* Set σ_pix so R lands on the target log-y.
     *   R = sqrt(muP_sq) / (σ_pix · dx_pix · η_SNR)
     * → σ_pix = sqrt(muP_sq) / (R · η_SNR · dx_pix) */
    const new_kernel = Apertures.k(ap, newDelta, 0);
    const muP_sq_new = S.FP1*S.FP1 + S.FP2*S.FP2
                     + 2 * S.FP1 * S.FP2 * new_kernel;
    let new_sigma = Math.sqrt(muP_sq_new)
                  / (Math.max(newR, 1e-6) * eta * dx_pix);
    new_sigma = Math.max(0.001, Math.min(4.0, new_sigma));
    S.sigma = new_sigma;

    /* Sync σ slider + label. */
    const sigSlider = document.getElementById('airy-sigma');
    const sigLabel  = document.getElementById('airy-sigma-val');
    if (sigSlider) sigSlider.value = S.sigma;
    if (sigLabel)  sigLabel.textContent = S.sigma.toFixed(3);
  }

  cv.style.cursor = (near || dragSrc === 'CLASS') ? 'move' : 'default';

  /* Axis labels (R on log scale, [0.4, 100]). */
  ctx.fillStyle = getCSSColor('--text-soft');
  ctx.font = '10px monospace';
  ctx.fillText('100',    4,             py + 8);
  const y10 = py + ph - (1 - lr_min) / (lr_max - lr_min) * ph;
  ctx.fillText('10',     6,             y10 - 2);
  const y1 = py + ph - (0 - lr_min) / (lr_max - lr_min) * ph;
  ctx.fillText('1',      8,             y1 - 2);
  ctx.fillText('0.4',    4,             py + ph - 2);
  ctx.fillText('0',      px - 4,        py + ph + 14);
  ctx.fillText(DELTA_MAX.toFixed(1),
                            px + pw - 16, py + ph + 14);
  ctx.fillText('R',      4,             py + ph / 2);
  ctx.fillText('Δ',      px + pw / 2,   py + ph + 14);
}

/* ---- Slider wiring ---- */
function bindSlider(slider, valEl, key, decimals) {
  if (!slider || !valEl) return;
  slider.addEventListener('input', () => {
    S[key] = parseFloat(slider.value);
    valEl.textContent = S[key].toFixed(decimals);
    if (key === 'FP1' || key === 'FP2') dmapDirty = true;
    /* Best-model + MC depend on fluxes AND σ. */
    mcDirty = true;
  });
  valEl.textContent = S[key].toFixed(decimals);
}
bindSlider(document.getElementById('airy-FP1'),
           document.getElementById('airy-FP1-val'), 'FP1', 2);
bindSlider(document.getElementById('airy-FP2'),
           document.getElementById('airy-FP2-val'), 'FP2', 2);
bindSlider(document.getElementById('airy-sigma'),
           document.getElementById('airy-sigma-val'), 'sigma', 3);

/* Aperture rotation slider (degrees → radians).  All cached
 * geometry-dependent quantities must be recomputed. */
(function bindThetaSlider() {
  const slider = document.getElementById('airy-theta');
  const valEl  = document.getElementById('airy-theta-val');
  if (!slider || !valEl) return;
  slider.addEventListener('input', () => {
    const deg = parseFloat(slider.value);
    valEl.textContent = `${deg.toFixed(0)}°`;
    Apertures.setAngle(deg * Math.PI / 180);
    dmapDirty = true; dratioDirty = true; mcDirty = true;
  });
  valEl.textContent = '0°';
  Apertures.setAngle(0);
})();

const newNoiseBtn = document.getElementById('airy-new-noise');
if (newNoiseBtn) {
  newNoiseBtn.addEventListener('click', () => {
    for (let i = 0; i < noiseSamples.length; i++) noiseSamples[i] = gauss();
    /* Don't trigger mcDirty: the bars represent the noise *distribution*
     * over many samples, which is unchanged by drawing a single new
     * realisation.  Only plot 1 (current sample) updates. */
  });
}

const btnPlog = document.getElementById('airy-P-log');
if (btnPlog) {
  btnPlog.addEventListener('click', () => {
    pLog = !pLog;
    btnPlog.classList.toggle('is-on', pLog);
  });
}
const btnNlog = document.getElementById('airy-noisy-log');
if (btnNlog) {
  btnNlog.addEventListener('click', () => {
    noisyLog = !noisyLog;
    btnNlog.classList.toggle('is-on', noisyLog);
  });
}
const btnBestLog = document.getElementById('airy-best-log');
if (btnBestLog) {
  btnBestLog.addEventListener('click', () => {
    bestLog = !bestLog;
    btnBestLog.classList.toggle('is-on', bestLog);
  });
}

/* ---- Plot-9 label: show the active aperture's η_SNR + name. ---- */
const APERTURE_DISPLAY_NAMES = {
  disk: 'disk', annulus: 'annulus', square: 'square',
  hexagon: 'hexagon', star6: '6-star', slot2: 'double slit',
  slot1: 'single slit', disk4: '4-disk array', hubble: 'Hubble',
};
function refreshClassLabelEta() {
  const ap = Apertures.active();
  const etaEl = document.getElementById('airy-class-eta');
  const apEl  = document.getElementById('airy-class-apname');
  if (etaEl) etaEl.textContent = ap.eta.toFixed(2);
  if (apEl)  apEl.textContent  = APERTURE_DISPLAY_NAMES[ap.name] || ap.name;
}


/* ---- Aperture palette: click-to-select ---- */
function setupAperturePalette() {
  const tiles = document.querySelectorAll('.aperture-tile');
  function refreshActive() {
    const activeName = Apertures.active().name;
    tiles.forEach(t => t.classList.toggle('active', t.dataset.aperture === activeName));
    refreshClassLabelEta();
  }
  tiles.forEach(tile => {
    /* Render the tile's icon by drawing the aperture's pupil mask. */
    const icon = tile.querySelector('canvas.aperture-tile-icon');
    if (icon) {
      const w = icon.width  = 36;
      const h = icon.height = 36;
      const ctx = icon.getContext('2d');
      const apFn = Apertures.apertureFns[tile.dataset.aperture];
      const img  = ctx.createImageData(w, h);
      for (let cy = 0; cy < h; cy++) {
        const v = (h / 2 - cy) / (h / 2) * 0.7;
        for (let cx = 0; cx < w; cx++) {
          const u = (cx - w / 2) / (w / 2) * 0.7;
          const inside = apFn(u, v) > 0;
          const idx = (cy * w + cx) * 4;
          if (inside) { img.data[idx] = 245; img.data[idx+1] = 220; img.data[idx+2] = 80; }
          else        { img.data[idx] =  22; img.data[idx+1] =  24; img.data[idx+2] = 32; }
          img.data[idx + 3] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);
    }
    tile.addEventListener('click', () => {
      const name = tile.dataset.aperture;
      if (Apertures.apertureNames.includes(name) && name !== Apertures.active().name) {
        Apertures.setActive(name);
        dmapDirty = true; dratioDirty = true; mcDirty = true;
        refreshActive();
      }
    });
  });
  refreshActive();
}

/* ---- δ_C readout via the area of the D̄ > 0 region in the D-map ---- *
 * For each cell of the D-map, compute the fraction of the cell inside
 * {D̄ > 0} via the same marching-squares geometry that draws the contour
 * (signs at corners → triangle / trapezoid / saddle inside-area).  The
 * total area gives ρ = area / π, and δ_1 = √ρ (paper definition).
 *
 * Note: this *is* the contour-line area integral the user asked for —
 * we're using the bilinear contour fragment per cell.  The bottom-right
 * (Dratio) plot is unaffected; only the readout switches calculation. */
function area_above_zero(v00, v10, v01, v11) {
  const idx = (v00 > 0 ? 1 : 0) | (v10 > 0 ? 2 : 0)
            | (v11 > 0 ? 4 : 0) | (v01 > 0 ? 8 : 0);
  if (idx === 0)  return 0;
  if (idx === 15) return 1;
  const lerp = (a, b) => (0 - a) / (b - a);
  switch (idx) {
    case  1: return 0.5 *      lerp(v00, v10)       *      lerp(v00, v01);
    case 14: return 1 - 0.5 *  lerp(v00, v10)       *      lerp(v00, v01);
    case  2: return 0.5 * (1 - lerp(v00, v10))      *      lerp(v10, v11);
    case 13: return 1 - 0.5 * (1 - lerp(v00, v10))  *      lerp(v10, v11);
    case  4: return 0.5 * (1 - lerp(v01, v11))      * (1 - lerp(v10, v11));
    case 11: return 1 - 0.5 * (1 - lerp(v01, v11))  * (1 - lerp(v10, v11));
    case  8: return 0.5 *      lerp(v01, v11)       * (1 - lerp(v00, v01));
    case  7: return 1 - 0.5 *  lerp(v01, v11)       * (1 - lerp(v00, v01));
    case  3: return 0.5 * (lerp(v00, v01) + lerp(v10, v11));
    case 12: return 1 - 0.5 * (lerp(v00, v01) + lerp(v10, v11));
    case  6: return 0.5 * ((1 - lerp(v00, v10)) + (1 - lerp(v01, v11)));
    case  9: return 1 - 0.5 * ((1 - lerp(v00, v10)) + (1 - lerp(v01, v11)));
    case  5: return 0.5 * lerp(v00, v10) * lerp(v00, v01)
                  + 0.5 * (1 - lerp(v01, v11)) * (1 - lerp(v10, v11));
    case 10: return 0.5 * (1 - lerp(v00, v10)) * lerp(v10, v11)
                  + 0.5 * lerp(v01, v11) * (1 - lerp(v00, v01));
  }
  return 0;
}
function compute_dC_area() {
  /* Cell area in image space (λ/d). */
  const cellArea = (2 * FOV / N_DGRID) * (2 * FOV / N_DGRID);
  let area = 0;
  for (let cy = 0; cy < N_DGRID - 1; cy++) {
    const r0 = cy * N_DGRID, r1 = (cy + 1) * N_DGRID;
    for (let cx = 0; cx < N_DGRID - 1; cx++) {
      area += area_above_zero(Dmap[r0 + cx],     Dmap[r0 + cx + 1],
                              Dmap[r1 + cx],     Dmap[r1 + cx + 1]) * cellArea;
    }
  }
  /* δ_1 = √(area / π).  Returns λ/d. */
  return Math.sqrt(area / Math.PI);
}

/* ---- Extended-FOV δ_C: separate grid covering ±FOV_DC_EXT to capture
 * far-field contamination islands for sparse apertures (slot2, disk4)
 * which sit beyond the visible Dmap's ±FOV.  Computed lazily on
 * dmapDirty (so once per slider release / aperture change) and cached. */
const FOV_DC_EXT  = 4.0;          /* λ/d — converges for all 7 apertures */
const N_DC_EXT    = 120;          /* 0.067 λ/d resolution */
let dCExtCached   = NaN;
let dCExtNeedsUpdate = true;
function compute_dC_extended() {
  const F1 = S.FP1, F2 = S.FP2;
  const p1x = S.P1x, p1y = S.P1y;
  const cellSide = 2 * FOV_DC_EXT / N_DC_EXT;
  const cellArea = cellSide * cellSide;
  /* Build extended D-bar grid centred on origin (sources span small Δ
   * around there; the contamination set translates with p_1 but its
   * area is invariant). */
  const Dext = new Float64Array(N_DC_EXT * N_DC_EXT);
  for (let cy = 0; cy < N_DC_EXT; cy++) {
    const y = +FOV_DC_EXT - (cy + 0.5) * cellSide;
    for (let cx = 0; cx < N_DC_EXT; cx++) {
      const x = -FOV_DC_EXT + (cx + 0.5) * cellSide;
      Dext[cy * N_DC_EXT + cx] = compute_Dbar(p1x, p1y, x + p1x, y + p1y, F1, F2);
    }
  }
  let area = 0;
  for (let cy = 0; cy < N_DC_EXT - 1; cy++) {
    const r0 = cy * N_DC_EXT, r1 = (cy + 1) * N_DC_EXT;
    for (let cx = 0; cx < N_DC_EXT - 1; cx++) {
      area += area_above_zero(Dext[r0+cx], Dext[r0+cx+1],
                              Dext[r1+cx], Dext[r1+cx+1]) * cellArea;
    }
  }
  return Math.sqrt(area / Math.PI);
}

const deltaCEl = document.getElementById('airy-deltaC-val');
function update_deltaC_readout() {
  if (!deltaCEl) return;
  if (dCExtNeedsUpdate) {
    dCExtCached = compute_dC_extended();
    dCExtNeedsUpdate = false;
  }
  const dC_lam_d = dCExtCached;
  if (!isFinite(dC_lam_d) || dC_lam_d <= 0) {
    deltaCEl.textContent = '—';
    return;
  }
  const dC_R = dC_lam_d / RAYLEIGH;
  deltaCEl.innerHTML =
    `${dC_R.toFixed(3)} δ<sub>R</sub>&nbsp;&nbsp;≈&nbsp;&nbsp;${dC_lam_d.toFixed(3)} λ/d`;
}

/* ---- Boot: precompute all apertures (blocking the first frame, ~250 ms) ---- */
Apertures.precomputeAll();
setupAperturePalette();

/* ---- Main loop ---- */
function loop() {
  compute_means();
  draw_FOV();
  draw_Slot();
  draw_Noisy();
  draw_P();

  if (dmapDirty && !held)   {
    compute_Dmap();
    dmapDirty = false;
    /* δ_C readout depends on the same configuration as Dmap, but is
     * computed on a 4× larger FOV to capture far-field contamination
     * (matters for the 4-disk array, double slit, etc.). */
    dCExtNeedsUpdate = true;
  }
  if (dratioDirty && !held) {
    build_Dratio();
    build_dCcurve();
    dratioDirty = false;
  }

  draw_Dmap();
  draw_DRatio();

  /* Row 3: 5-candidate MLE best-model selection + cached MC + classification.
   * compute_best_model runs each frame on the current cached noise so
   * "↻ new sample" / source drag changes plot 1 + badge live.  refresh_mc
   * runs the AIC search on N_MC fresh samples and stores the majority
   * winner's mean ± 2σ retrieval — only when the configuration is dirty
   * (aperture / sources / sliders). */
  if (mcDirty && !held) {
    compute_Qstar_responses();
    refresh_mc();
  }
  compute_best_model();
  compute_muFitFine();
  compute_classification();
  draw_BestFit();
  draw_Flux();
  draw_Class();

  update_deltaC_readout();

  if (released) { dragSrc = null; dragCv = null; }
  pressed = false; released = false;
  requestAnimationFrame(loop);
}
loop();

})();
