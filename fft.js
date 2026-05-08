/* ===========================================================================
 * Tiny 2D radix-2 Cooley–Tukey FFT (decimation-in-time, in-place).
 *
 *   FFT2D.fft1d(re, im, sign)            — 1D, length must be a power of 2
 *   FFT2D.fft2d(re, im, n, sign)         — 2D n×n, n must be a power of 2
 *   FFT2D.fftshift2d(arr, n)             — quadrant swap
 *
 * Conventions: sign = -1 forward, sign = +1 inverse.  Neither direction is
 * normalised; for the inverse divide by N (or N² in 2D).  Buffers are
 * row-major Float64Array of length N * N.
 * =========================================================================== */

(function (global) {

function fft1d(re, im, sign) {
  const n = re.length;

  /* bit-reverse permutation */
  for (let i = 0, j = 0; i < n; i++) {
    if (i < j) {
      let t;
      t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
    let m = n >> 1;
    while (m >= 1 && j >= m) { j -= m; m >>= 1; }
    j += m;
  }

  /* butterflies */
  for (let s = 1; s < n; s <<= 1) {
    const m = s << 1;
    const angle = sign * 2 * Math.PI / m;
    const wm_re = Math.cos(angle);
    const wm_im = Math.sin(angle);
    for (let k = 0; k < n; k += m) {
      let w_re = 1, w_im = 0;
      for (let j = 0; j < s; j++) {
        const ti_re = re[k + j + s] * w_re - im[k + j + s] * w_im;
        const ti_im = re[k + j + s] * w_im + im[k + j + s] * w_re;
        const ui_re = re[k + j];
        const ui_im = im[k + j];
        re[k + j]     = ui_re + ti_re;
        im[k + j]     = ui_im + ti_im;
        re[k + j + s] = ui_re - ti_re;
        im[k + j + s] = ui_im - ti_im;
        const nw_re = w_re * wm_re - w_im * wm_im;
        const nw_im = w_re * wm_im + w_im * wm_re;
        w_re = nw_re;
        w_im = nw_im;
      }
    }
  }
}

function fft2d(re, im, n, sign) {
  const tre = new Float64Array(n);
  const tim = new Float64Array(n);

  /* row FFTs */
  for (let r = 0; r < n; r++) {
    const off = r * n;
    for (let c = 0; c < n; c++) { tre[c] = re[off + c]; tim[c] = im[off + c]; }
    fft1d(tre, tim, sign);
    for (let c = 0; c < n; c++) { re[off + c] = tre[c]; im[off + c] = tim[c]; }
  }

  /* column FFTs */
  for (let c = 0; c < n; c++) {
    for (let r = 0; r < n; r++) { tre[r] = re[r * n + c]; tim[r] = im[r * n + c]; }
    fft1d(tre, tim, sign);
    for (let r = 0; r < n; r++) { re[r * n + c] = tre[r]; im[r * n + c] = tim[r]; }
  }
}

/* Quadrant swap so DC moves between corner ↔ center. */
function fftshift2d(arr, n) {
  const half = n >> 1;
  for (let r = 0; r < half; r++) {
    for (let c = 0; c < half; c++) {
      const a = r * n + c;
      const b = (r + half) * n + (c + half);
      const t1 = arr[a]; arr[a] = arr[b]; arr[b] = t1;
      const a2 = r * n + (c + half);
      const b2 = (r + half) * n + c;
      const t2 = arr[a2]; arr[a2] = arr[b2]; arr[b2] = t2;
    }
  }
}

global.FFT2D = { fft1d, fft2d, fftshift2d };

})(typeof window !== 'undefined' ? window : globalThis);
