/* ===========================================================================
 * Aperture library: pupil-plane masks for several shapes (disk, annulus,
 * square, hexagon, 6-point star), plus FFT-based image-plane lookups for the
 * intensity PSF U(x, y) and the autocorrelation k(Δx, Δy) of the unit-norm
 * 2D dictionary element.
 *
 * Conventions:
 *   - Pupil plane sampled in [-U_PUPIL, U_PUPIL]² with N×N grid.  Aperture
 *     mask returns 1 inside, 0 outside, and is centred on (0, 0).  Apertures
 *     fit inside the unit disk of radius 0.5, so the pupil "diameter" is 1.
 *   - Image plane sampled at dx = 1 / (2 U_PUPIL) (in λ/d) over [-X_max, X_max]
 *     where X_max = N · dx / 2.
 *   - U(x, y) is the squared modulus of F[aperture], normalised so its peak
 *     equals 1 (NOT unit L²).  k(Δx, Δy) is the autocorrelation of the
 *     unit-L²-norm dictionary element, normalised so k(0, 0) = 1.
 *   - Tables are cropped to a small window around (0, 0) in image space:
 *     U is needed only inside the FOV, k inside the q*-search radius.
 * =========================================================================== */

(function (global) {

const N        = 512;            /* FFT grid size (must be a power of 2) */
const U_PUPIL  = 8.0;            /* pupil domain half-width */
const dx       = 1.0 / (2 * U_PUPIL);   /* image-plane sampling (λ/d) — = 0.0625 */

const U_HALF   = 32;             /* ±2 in λ/d */
const U_size   = 2 * U_HALF + 1; /* 65 */
const K_HALF   = 80;             /* ±5 in λ/d */
const K_size   = 2 * K_HALF + 1; /* 161 */

/* ---- Aperture masks (defined in pupil plane) ----
 *
 * All apertures are rescaled to the same area (π/4 ≈ 0.7854) as the disk
 * of radius 0.5, so the demo compares shapes at equal light-collecting
 * area.  Linear dimensions scale by √(π/4 / orig_area).  Notably the
 * sparser apertures (slot2, 6-star) end up larger than radius 0.5 in
 * pupil; the visualisation window in the slot panel and tile icons is
 * widened to accommodate them. */
const sqrt3 = Math.sqrt(3);

/* Annulus: keep inner/outer ratio ε = 0.3.  area = πR²(1−ε²) = π/4 → R²=0.25/0.91. */
const ANNULUS_RO = Math.sqrt(0.25 / 0.91);          /* ≈ 0.5241 */
const ANNULUS_RI = 0.3 * ANNULUS_RO;                /* ≈ 0.1572 */
/* Square half-side: (2a)² = π/4 → a = √π/4. */
const SQUARE_HALF = Math.sqrt(Math.PI) / 4;         /* ≈ 0.4431 */
/* Regular hexagon vertex radius (vertex on x-axis): area = (3√3/2)R² = π/4. */
const HEX_R       = Math.sqrt(Math.PI / (6 * sqrt3));   /* ≈ 0.5498 */
/* Star of David: union of two equilateral triangles, both with circumradius R.
 * Each triangle has area (3√3/4)R²; their intersection is a regular hexagon
 * with vertex radius R/√3, so area = √3·R².  Set √3·R² = π/4 → R² = π/(4√3). */
const STAR_R      = Math.sqrt(Math.PI / (4 * sqrt3));   /* ≈ 0.6734 */
/* Double slit: keep proportions (w=0.2, h=0.6, d=0.5 → orig area 0.24).
 * Scale factor = √(π/4 / 0.24). */
const SLOT_SCALE  = Math.sqrt((Math.PI / 4) / 0.24);    /* ≈ 1.8090 */
const SLOT_HW     = 0.10 * SLOT_SCALE;              /* slot half-width ≈ 0.1809 */
const SLOT_HH     = 0.30 * SLOT_SCALE;              /* slot half-height ≈ 0.5427 */
const SLOT_C      = 0.25 * SLOT_SCALE;              /* slot centre offset ≈ 0.4523 */
/* Single slit: 1:3 aspect.  4·HW·HH = π/4 with HH = 3·HW → HW = √(π/48). */
const SLOT1_HW    = Math.sqrt(Math.PI / 48);        /* ≈ 0.2557 */
const SLOT1_HH    = 3 * SLOT1_HW;                   /* ≈ 0.7672 */
/* Four-disk array: 4 disks of radius r at (±d, ±d).  4·π·r² = π/4 → r = 0.25.
 * d = 0.42 keeps the bounding extent ≤ 0.67, with a clear gap between disks. */
const DISK4_R     = 0.25;
const DISK4_D     = 0.42;
const DISK4_R2    = DISK4_R * DISK4_R;
/* Hubble-style: primary disk, central secondary obstruction, 4 spider vanes
 * forming a plus.  Inner-to-outer ratio ≈ 0.33 (HST), vane half-width 0.020.
 * Outer radius solved so transmitted area = π/4:
 *   π R_o²·(1−0.33²) − 4·(R_o−R_i)·w = π/4 → R_o ≈ 0.549. */
const HUBBLE_RO   = 0.549;
const HUBBLE_RI   = 0.33 * HUBBLE_RO;            /* ≈ 0.181 */
const HUBBLE_RO2  = HUBBLE_RO * HUBBLE_RO;
const HUBBLE_RI2  = HUBBLE_RI * HUBBLE_RI;
const HUBBLE_W2   = 0.020;                       /* spider vane half-width */

const apertureFns = {
  disk: (u, v) => (u*u + v*v <= 0.25) ? 1 : 0,

  annulus: (u, v) => {
    const r2 = u*u + v*v;
    return (r2 >= ANNULUS_RI * ANNULUS_RI && r2 <= ANNULUS_RO * ANNULUS_RO) ? 1 : 0;
  },

  square: (u, v) => (Math.abs(u) <= SQUARE_HALF && Math.abs(v) <= SQUARE_HALF) ? 1 : 0,

  hexagon: (u, v) => {
    /* Regular hexagon, vertices at (±HEX_R, 0) and (±HEX_R/2, ±HEX_R√3/2).
     * Three pairs of parallel sides. */
    if (Math.abs(v) > HEX_R * sqrt3 * 0.5) return 0;
    if (Math.abs(u + v / sqrt3)   > HEX_R) return 0;
    if (Math.abs(u - v / sqrt3)   > HEX_R) return 0;
    return 1;
  },

  star6: (u, v) => {
    const R = STAR_R;
    const inUp = (v >= -R/2) && (v <= -sqrt3 * u + R) && (v <=  sqrt3 * u + R);
    const inDn = (v <=  R/2) && (v >=  sqrt3 * u - R) && (v >= -sqrt3 * u - R);
    return (inUp || inDn) ? 1 : 0;
  },

  slot2: (u, v) => {
    if (Math.abs(v) > SLOT_HH) return 0;
    const au = Math.abs(u);
    return (au >= SLOT_C - SLOT_HW && au <= SLOT_C + SLOT_HW) ? 1 : 0;
  },

  slot1: (u, v) => (Math.abs(u) <= SLOT1_HW && Math.abs(v) <= SLOT1_HH) ? 1 : 0,

  disk4: (u, v) => {
    /* Four disks centred at (±D, ±D), each of radius DISK4_R. */
    const au = Math.abs(u) - DISK4_D;
    const av = Math.abs(v) - DISK4_D;
    return (au * au + av * av <= DISK4_R2) ? 1 : 0;
  },

  hubble: (u, v) => {
    /* Annular pupil minus a 4-vane spider (plus-shaped). */
    const r2 = u * u + v * v;
    if (r2 > HUBBLE_RO2)         return 0;       /* outside primary */
    if (r2 < HUBBLE_RI2)         return 0;       /* secondary obstruction */
    if (Math.abs(u) <= HUBBLE_W2) return 0;      /* vertical spider */
    if (Math.abs(v) <= HUBBLE_W2) return 0;      /* horizontal spider */
    return 1;
  },
};

const apertureNames = ['disk', 'annulus', 'square', 'hexagon', 'star6', 'slot2', 'slot1', 'disk4', 'hubble'];
const apertureLabels = {
  disk:    'Disk',
  annulus: 'Annulus',
  square:  'Square',
  hexagon: 'Hexagon',
  star6:   '6-star',
  slot2:   'Double slit',
  slot1:   'Single slit',
  disk4:   '4-disk array',
  hubble:  'Hubble',
};

/* ---- Bessel J_0 (Numerical Recipes / A&S 9.4) — used for the disk's
 * analytical autocorrelation, which matches the paper's δ_C ≈ 0.9520
 * exactly (the FFT path drifts by ~0.005 δR around the Q*-cusp). */
function besselJ0(x) {
  const ax = Math.abs(x);
  if (ax < 8.0) {
    const y = x * x;
    const ans1 = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7
                + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
    const ans2 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718
                + y * (59272.64853 + y * (267.8532712 + y))));
    return ans1 / ans2;
  }
  const z = 8.0 / ax;
  const y = z * z;
  const ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4
              + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
  const ans2 = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5
              + y * (0.7621095161e-6 + y * (-0.934935152e-7))));
  const xx = ax - 0.785398164;
  return Math.sqrt(0.636619772 / ax)
       * (Math.cos(xx) * ans1 - z * Math.sin(xx) * ans2);
}

/* 1D radial autocorrelation table for isotropic apertures.  We need a much
 * finer sampling than the 2D bilinear table here because the equal-flux Q*
 * search bifurcates discontinuously at the cusp; second derivatives of k
 * control where the bifurcation lands, and bilinear interp at dx ≈ 0.06
 * drifts δ_C by ~0.005 δR (down to 0.945 vs the paper's 0.952). */
const K_RAD_DR   = 0.005;                                 /* λ/d */
const K_RAD_RMAX = 5.0;                                   /* λ/d */
const K_RAD_N    = Math.round(K_RAD_RMAX / K_RAD_DR);     /* = 1000 */

/* Helper: Hankel-inverse-of-MTF² builder.  Given a sampler M2u[j] = M²(j·du)·u
 * over u ∈ [0, u_cutoff], returns the radial autocorrelation table k(r). */
function hankelKRadialTable(M2u, du, u_cutoff) {
  function k_at(r) {
    let s = M2u[0] * besselJ0(0) + M2u[M2u.length - 1] * besselJ0(2 * Math.PI * u_cutoff * r);
    const N_QUAD = M2u.length - 1;
    for (let j = 1; j < N_QUAD; j += 2) s += 4 * M2u[j] * besselJ0(2 * Math.PI * j * du * r);
    for (let j = 2; j < N_QUAD; j += 2) s += 2 * M2u[j] * besselJ0(2 * Math.PI * j * du * r);
    return 2 * Math.PI * s * du / 3;
  }
  const tab = new Float64Array(K_RAD_N + 1);
  for (let i = 0; i <= K_RAD_N; i++) tab[i] = k_at(i * K_RAD_DR);
  const k0 = tab[0];
  for (let i = 0; i <= K_RAD_N; i++) tab[i] /= k0;
  return tab;
}

/* Disk's analytical k(r) = 2π ∫_0^1 M(u)² J_0(2πur) u du (unit-disk MTF). */
function diskKRadialTable_analytical() {
  const N_QUAD = 1024;
  const du = 1.0 / N_QUAD;
  const M2u = new Float64Array(N_QUAD + 1);
  for (let j = 0; j <= N_QUAD; j++) {
    const u  = j * du;
    const om = Math.sqrt(Math.max(0.0, 1.0 - u * u));
    const M  = (2.0 / Math.PI) * (Math.acos(u) - u * om);
    M2u[j]   = M * M * u;
  }
  return hankelKRadialTable(M2u, du, 1.0);
}

/* Annulus k(r) via the Hankel transform of MTF², where the annular MTF is
 * the autocorrelation of (outer-disk minus inner-disk):
 *   AC(u) = AC_disk(u; R_o) - 2 cross(u; R_o, R_i) + AC_disk(u; R_i),
 * with the cross-term being the lens-overlap of two disks of unequal radii. */
function annulusKRadialTable_analytical(R_o, R_i) {
  function diskAC(u, R) {
    if (u >= 2 * R) return 0;
    return 2 * R * R * Math.acos(u / (2 * R)) - 0.5 * u * Math.sqrt(4 * R * R - u * u);
  }
  function crossAC(u, R1, R2) {
    if (u >= R1 + R2) return 0;
    if (u <= Math.abs(R1 - R2)) return Math.PI * Math.min(R1, R2) ** 2;
    const a = R1 * R1 * Math.acos((u * u + R1 * R1 - R2 * R2) / (2 * u * R1));
    const b = R2 * R2 * Math.acos((u * u + R2 * R2 - R1 * R1) / (2 * u * R2));
    const c = 0.5 * Math.sqrt((-u + R1 + R2) * (u + R1 - R2)
                            * (u - R1 + R2) * (u + R1 + R2));
    return a + b - c;
  }
  function annulusAC(u) { return diskAC(u, R_o) - 2 * crossAC(u, R_o, R_i) + diskAC(u, R_i); }

  const u_cutoff = 2 * R_o;
  const N_QUAD = 1024;
  const du = u_cutoff / N_QUAD;
  const M0 = annulusAC(0);
  const M2u = new Float64Array(N_QUAD + 1);
  for (let j = 0; j <= N_QUAD; j++) {
    const u = j * du;
    const M = annulusAC(u) / M0;
    M2u[j]  = M * M * u;
  }
  return hankelKRadialTable(M2u, du, u_cutoff);
}

/* Square aperture (side L): the PSF is sinc²(πLu)·sinc²(πLv) — separable.
 * The 2D autocorrelation is then K1(Δx) · K1(Δy), with
 *   K1(Δ) = (2/L²) ∫_0^L (L−u)² cos(2π u Δ) du.
 * Tabulate K1 on a fine 1D grid; lookup as the product. */
function squareK1Table_analytical(L) {
  const N_QUAD = 1024;
  const du = L / N_QUAD;
  function K1_at(D) {
    let s = 0;
    for (let j = 0; j <= N_QUAD; j++) {
      const u = j * du;
      const Lu2 = (L - u) * (L - u);
      const v = Lu2 * Math.cos(2 * Math.PI * u * D);
      const w = (j === 0 || j === N_QUAD) ? 1 : (j & 1) ? 4 : 2;
      s += w * v;
    }
    return (2 / (L * L)) * s * du / 3;
  }
  const tab = new Float64Array(K_RAD_N + 1);
  for (let i = 0; i <= K_RAD_N; i++) tab[i] = K1_at(i * K_RAD_DR);
  const k0 = tab[0];
  for (let i = 0; i <= K_RAD_N; i++) tab[i] /= k0;
  return tab;
}

/* ---- Build U and k tables for a single aperture, via FFT ---- */
function buildAperture(name) {
  const apFn = apertureFns[name];
  if (!apFn) throw new Error(`Unknown aperture: ${name}`);

  const re = new Float64Array(N * N);
  const im = new Float64Array(N * N);

  /* (1) Aperture mask in pupil plane, centred. */
  for (let r = 0; r < N; r++) {
    const v = (N / 2 - r) / N * 2 * U_PUPIL;
    for (let c = 0; c < N; c++) {
      const u = (c - N / 2) / N * 2 * U_PUPIL;
      re[r * N + c] = apFn(u, v);
    }
  }

  /* (2) FFT → spectral amplitude. */
  FFT2D.fftshift2d(re, N);
  FFT2D.fft2d(re, im, N, -1);

  /* (3) |F[A]|² = PSF intensity (still in DFT order). */
  for (let i = 0; i < N * N; i++) {
    re[i] = re[i] * re[i] + im[i] * im[i];
    im[i] = 0;
  }
  FFT2D.fftshift2d(re, N);     /* peak now at array centre */

  /* (4) Crop U_table to ±U_HALF around centre, normalise peak to 1. */
  let peak = 0;
  for (let i = 0; i < N * N; i++) if (re[i] > peak) peak = re[i];
  const U_table = new Float64Array(U_size * U_size);
  for (let dr = -U_HALF; dr <= U_HALF; dr++) {
    for (let dc = -U_HALF; dc <= U_HALF; dc++) {
      U_table[(dr + U_HALF) * U_size + (dc + U_HALF)] =
        re[(N / 2 + dr) * N + (N / 2 + dc)] / peak;
    }
  }

  /* (5) Autocorrelation k = F⁻¹[|F[U]|²].  Reuse re/im. */
  for (let i = 0; i < N * N; i++) im[i] = 0;
  FFT2D.fftshift2d(re, N);     /* origin back to corner */
  FFT2D.fft2d(re, im, N, -1);
  for (let i = 0; i < N * N; i++) {
    re[i] = re[i] * re[i] + im[i] * im[i];
    im[i] = 0;
  }
  FFT2D.fft2d(re, im, N, +1);
  const inv = 1.0 / (N * N);
  for (let i = 0; i < N * N; i++) re[i] *= inv;
  FFT2D.fftshift2d(re, N);     /* lag (0,0) at array centre */

  /* (6) Crop k_table to ±K_HALF around centre, normalise k(0, 0) = 1. */
  const k0 = re[(N / 2) * N + N / 2];
  let k_table = new Float64Array(K_size * K_size);
  for (let dr = -K_HALF; dr <= K_HALF; dr++) {
    for (let dc = -K_HALF; dc <= K_HALF; dc++) {
      k_table[(dr + K_HALF) * K_size + (dc + K_HALF)] =
        re[(N / 2 + dr) * N + (N / 2 + dc)] / k0;
    }
  }

  /* Analytical fast paths.  The FFT-based 2D bilinear table has too much
   * sub-cell noise near the Q*-cusp; for apertures with closed-form
   * autocorrelation we tabulate at dr = 0.005 instead. */
  let k_radial = null;
  let separable = false;
  if (name === 'disk') {
    k_radial = diskKRadialTable_analytical();
  } else if (name === 'annulus') {
    k_radial = annulusKRadialTable_analytical(ANNULUS_RO, ANNULUS_RI);
  } else if (name === 'square') {
    /* Side length L = 2 · half-side. */
    k_radial  = squareK1Table_analytical(2 * SQUARE_HALF);
    separable = true;
  }

  const ap = { name, dx, U_HALF, U_size, U_table,
               K_HALF, K_size, k_table, k_radial, separable, apFn };

  /* Compute the look-elsewhere-corrected η_SNR for this aperture via
   * Monte Carlo on noise-only data (see "AIC" section's demo).  Stored
   * on the aperture so the airy demo can read it back in plot 9 + AIC. */
  const etaStats = computeEta(ap);
  ap.eta_mean = etaStats.mean;
  ap.eta_std  = etaStats.std;
  ap.eta      = etaStats.eta;
  ap.eta2     = etaStats.eta * etaStats.eta;

  return ap;
}

/* ---- Per-aperture η_SNR via MC on noise-only data ----
 *
 * Mirrors the airy demo's pixel-space matched-filter setup: FOV = ±1.83 λ/d
 * (= ±1.5 δ_R), 20×20 detector grid.  For each MC sample we generate
 * noise X ~ N(0, 1) per pixel, evaluate the matched-filter S/N at every
 * pixel-centred candidate q, and record the maximum.  Mean + 5·std of that
 * distribution is η_SNR. */
const ETA_FOV_LAM_D = 1.5 * 1.22;        /* matches the airy demo's FOV */
const ETA_PIX       = 20;
const ETA_DX        = 2 * ETA_FOV_LAM_D / ETA_PIX;
const ETA_N_MC      = 2000;

function _pxX(j) { return -ETA_FOV_LAM_D + (j + 0.5) * ETA_DX; }
function _pxY(i) { return +ETA_FOV_LAM_D - (i + 0.5) * ETA_DX; }
function _gauss() {
  const u1 = Math.random() + 1e-12;
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function computeEta(ap) {
  /* Build matched-filter response m_q on 20×20 search positions. */
  const NQ = ETA_PIX, NP = ETA_PIX;
  const m_table   = [];
  const m_norm_sq = new Float64Array(NQ * NQ);
  for (let qi = 0; qi < NQ; qi++) {
    const qy = _pxY(qi);
    for (let qj = 0; qj < NQ; qj++) {
      const qx = _pxX(qj);
      const m  = new Float64Array(NP * NP);
      let nrm2 = 0;
      for (let i = 0; i < NP; i++) {
        const y = _pxY(i);
        for (let j = 0; j < NP; j++) {
          const x = _pxX(j);
          const v = U(ap, x - qx, y - qy);
          m[i * NP + j] = v;
          nrm2 += v * v;
        }
      }
      m_table.push(m);
      m_norm_sq[qi * NQ + qj] = nrm2;
    }
  }

  /* MC: max(⟨m_q, X⟩, 0) / ‖m_q‖ over q for each fresh X ~ N(0, 1)^N.
   * One-sided maximum because the paper's Q₁ model space requires F_q ≥ 0
   * (a candidate with negative ⟨m_q, X⟩ collapses to F̂ = 0 and matches
   * the empty model — it doesn't compete for Q*). */
  const X = new Float64Array(NP * NP);
  let sum = 0, sum2 = 0;
  for (let s = 0; s < ETA_N_MC; s++) {
    for (let p = 0; p < X.length; p++) X[p] = _gauss();
    let best = 0;
    for (let q = 0; q < m_table.length; q++) {
      const mq   = m_table[q];
      const norm = Math.sqrt(m_norm_sq[q] || 1e-12);
      let b = 0;
      for (let p = 0; p < X.length; p++) b += mq[p] * X[p];
      if (b <= 0) continue;                        /* F̂ ≥ 0 constraint */
      const snr = b / norm;
      if (snr > best) best = snr;
    }
    sum  += best;
    sum2 += best * best;
  }
  const mean = sum / ETA_N_MC;
  const v    = Math.max(sum2 / ETA_N_MC - mean * mean, 0);
  const std  = Math.sqrt(v);
  return { mean, std, eta: mean + 5 * std };
}

/* ---- Active-aperture rotation ----
 * Rotation is applied at lookup time — the U/K tables stay fixed; we
 * rotate the query (x, y) by -θ before sampling.  Rotation is unitary,
 * so the look-elsewhere η_SNR is rotation-invariant; we don't recompute
 * it on angle change. */
let activeAngle = 0;        /* radians */
let activeCos   = 1;
let activeSin   = 0;
function setAngle(theta) {
  activeAngle = theta;
  activeCos   = Math.cos(theta);
  activeSin   = Math.sin(theta);
}
function getAngle() { return activeAngle; }

/* ---- Bilinear lookup helpers ----
 * (x, y) is image-plane in λ/d, with (0, 0) at table centre and y up.
 * Outside the table we return 0 (the PSF/autocorrelation is small there). */
function U(ap, x, y) {
  /* rotate query by -θ before sampling the unrotated table */
  const xr =  activeCos * x + activeSin * y;
  const yr = -activeSin * x + activeCos * y;
  x = xr; y = yr;
  const fc = x / ap.dx + ap.U_HALF;
  const fr = -y / ap.dx + ap.U_HALF;
  if (fc < 0 || fc >= ap.U_size - 1 || fr < 0 || fr >= ap.U_size - 1) return 0;
  const ic = Math.floor(fc), ir = Math.floor(fr);
  const ac = fc - ic, ar = fr - ir;
  const W = ap.U_size;
  const t = ap.U_table;
  return (1 - ar) * ((1 - ac) * t[ir * W + ic] + ac * t[ir * W + ic + 1])
       +     ar  * ((1 - ac) * t[(ir + 1) * W + ic] + ac * t[(ir + 1) * W + ic + 1]);
}

/* Catmull-Rom cubic Hermite interpolation between p1 and p2, with neighbours
 * p0 (left) and p3 (right).  C¹ continuous → smooth gradients across cells,
 * which the Q*-search near the cusp needs. */
function cubicHermite(p0, p1, p2, p3, t) {
  const a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
  const b =        p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
  const c = -0.5 * p0 +              0.5 * p2;
  const d =                  p1;
  return ((a * t + b) * t + c) * t + d;
}

/* Separable-aperture fast path: k(Δx, Δy) = K1(|Δx|) · K1(|Δy|).  A 1D
 * Catmull-Rom interp on the K1 table, with bilinear at the very edge. */
function k_separable(ap, dx_, dy_) {
  function eval1D(z) {
    const az = Math.abs(z);
    if (az >= K_RAD_RMAX) return 0;
    const t = az / K_RAD_DR;
    const i = Math.floor(t);
    const a = t - i;
    const tab = ap.k_radial;        /* same table — re-used as 1D K1 */
    if (i < 1 || i >= tab.length - 2) {
      return (1 - a) * tab[i] + a * tab[i + 1];
    }
    return cubicHermite(tab[i - 1], tab[i], tab[i + 1], tab[i + 2], a);
  }
  return eval1D(dx_) * eval1D(dy_);
}

function k(ap, dx_, dy_) {
  /* Rotate input by -θ so we sample the unrotated K table at the
   * rotated coordinates. */
  {
    const dxr =  activeCos * dx_ + activeSin * dy_;
    const dyr = -activeSin * dx_ + activeCos * dy_;
    dx_ = dxr; dy_ = dyr;
  }
  /* Isotropic 1D-radial fast path with cubic interp. */
  if (ap.k_radial && !ap.separable) {
    const r = Math.hypot(dx_, dy_);
    if (r >= K_RAD_RMAX) return 0;
    const t = r / K_RAD_DR;
    const i = Math.floor(t);
    const a = t - i;
    if (i < 1 || i >= ap.k_radial.length - 2) {
      return (1 - a) * ap.k_radial[i] + a * ap.k_radial[i + 1];
    }
    return cubicHermite(ap.k_radial[i - 1], ap.k_radial[i],
                        ap.k_radial[i + 1], ap.k_radial[i + 2], a);
  }
  /* Separable (square): k(Δx, Δy) = K1(Δx) · K1(Δy). */
  if (ap.separable) return k_separable(ap, dx_, dy_);

  /* General 2D bicubic for non-isotropic, non-separable apertures (hex, star). */
  const fc = dx_ / ap.dx + ap.K_HALF;
  const fr = -dy_ / ap.dx + ap.K_HALF;
  const W = ap.K_size;
  if (fc < 0 || fc >= W - 1 || fr < 0 || fr >= W - 1) return 0;
  const ic = Math.floor(fc), ir = Math.floor(fr);
  const ac = fc - ic, ar = fr - ir;
  const t = ap.k_table;
  if (ic < 1 || ic >= W - 2 || ir < 1 || ir >= W - 2) {
    /* edge: bilinear fallback */
    return (1 - ar) * ((1 - ac) * t[ir * W + ic] + ac * t[ir * W + ic + 1])
         +     ar  * ((1 - ac) * t[(ir + 1) * W + ic] + ac * t[(ir + 1) * W + ic + 1]);
  }
  const off0 = (ir - 1) * W;
  const off1 = (ir    ) * W;
  const off2 = (ir + 1) * W;
  const off3 = (ir + 2) * W;
  const r0 = cubicHermite(t[off0 + ic - 1], t[off0 + ic], t[off0 + ic + 1], t[off0 + ic + 2], ac);
  const r1 = cubicHermite(t[off1 + ic - 1], t[off1 + ic], t[off1 + ic + 1], t[off1 + ic + 2], ac);
  const r2 = cubicHermite(t[off2 + ic - 1], t[off2 + ic], t[off2 + ic + 1], t[off2 + ic + 2], ac);
  const r3 = cubicHermite(t[off3 + ic - 1], t[off3 + ic], t[off3 + ic + 1], t[off3 + ic + 2], ac);
  return cubicHermite(r0, r1, r2, r3, ar);
}

function kgrad(ap, dx_, dy_) {
  const eps = (ap.k_radial || ap.separable) ? K_RAD_DR : ap.dx;
  const kxp = k(ap, dx_ + eps, dy_);
  const kxm = k(ap, dx_ - eps, dy_);
  const kyp = k(ap, dx_, dy_ + eps);
  const kym = k(ap, dx_, dy_ - eps);
  return [(kxp - kxm) / (2 * eps), (kyp - kym) / (2 * eps)];
}

/* ---- Aperture cache and active selection ---- */
const cache = {};
function getAperture(name) {
  if (cache[name]) return cache[name];
  cache[name] = buildAperture(name);
  return cache[name];
}
function precomputeAll() {
  for (const n of apertureNames) getAperture(n);
}

let activeName = 'disk';
function setActive(name) { activeName = name; }
function active() { return getAperture(activeName); }

global.Apertures = {
  precomputeAll, setActive, active, getAperture,
  apertureNames, apertureLabels, apertureFns,
  U, k, kgrad,
  setAngle, getAngle,
};

})(typeof window !== 'undefined' ? window : globalThis);
