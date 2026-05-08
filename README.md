# A confusion framework for direct imaging

Interactive companion to **Cacaj et al. 2025**, *Photobombing for the Large
Interferometer For Exoplanets (LIFE)*.

🔭 **Live demo:** https://drini22.github.io/confusion-framework-demo/

A noise-independent geometric criterion for when two faint sources blur
into one.  The framework reduces three contamination conditions
(detect / resolve fails / contaminate) to a single sign test
$\bar{\mathcal{D}} \gtrless 0$, defines an instrument-agnostic spatial
resolution $\delta_1$, and recovers the Rayleigh order of magnitude
($\delta_1 \approx 1.16\,\lambda/d$ for the Airy disk) while
generalising to nullers, segmented apertures, and arbitrary PSFs.

## Sections

The page is structured as a sequence of interactive demos that build the
framework from first principles:

1. The model — point-source configuration, PSF dictionary, additive
   Gaussian noise.
2. The cost function — Mahalanobis $J(P, Q)$.
3. The best one-source explanation $Q^\star$ — matched-filter argmax.
4. **A parsimony cost for model selection (AIC)** — defines
   $\eta_{\mathrm{SNR}}$ via Monte Carlo (Dannert et al. 2022 calibration),
   with a live noise + fit + histogram demo.
5. Three conditions for contamination — Eqs. 7–9 from the paper.
6. The target confusion map $\bar{\mathcal{D}}$ — sign test, normalisation.
7. When is $\bar{\mathcal{D}}$ bounded?  Finite-correlation condition.
8. The spatial resolution $\delta_1$ — area-of-$\{\bar{\mathcal{D}} > 0\}$ definition.
9. **$\delta_1$ for an aperture telescope** — full nine-plot demo with
   eight aperture shapes (disk, annulus, square, hexagon, 6-star,
   single/double slit, 4-disk array, Hubble), per-aperture
   look-elsewhere-corrected $\eta_{\mathrm{SNR}}$, drag-and-drop
   classification, live $\delta_C$ readout on a 4× extended FOV.

All compute (FFTs, Monte Carlo, matched-filter retrievals,
$\bar{\mathcal{D}}$-area integrals) runs in JavaScript in the visitor's
browser — no backend.

## Run locally

```bash
cd /path/to/this/repo
python3 -m http.server 8000
# open http://localhost:8000/
```

Or just open `index.html` directly in a browser, but a local HTTP server
is more reliable (some browsers block JavaScript module-style imports
over `file://`).

## Files

| file | purpose |
|---|---|
| `index.html` | main page — all sections + interactive demos |
| `gaussian-2d.html` | standalone 2-D Gaussian PSF demo with proofs |
| `style.css` | site theme + per-demo layout |
| `apertures.js` | pupil masks, FFT-based PSF/autocorrelation tables, per-aperture look-elsewhere $\eta_{\mathrm{SNR}}$, rotation hooks |
| `fft.js` | tiny 2D Cooley–Tukey FFT (no external deps) |
| `examples-model.js` | section 1 demo |
| `examples-cost.js` | section 2 demo |
| `examples-qstar.js` | section 3 demo |
| `examples-aic.js` | section 4 ($\eta_{\mathrm{SNR}}$ calibration MC) |
| `examples-contam.js` | section 5 demo |
| `examples-target.js` | section 6 ($\bar{\mathcal{D}}$ map) demo |
| `examples-airy.js` | section 9 (aperture-telescope) demo — the largest scene |
| `theme.js` | dark-light toggle |

## Citation

If you use the framework, please cite:

> Cacaj, D., Angerhausen, D., Saxena, P., Laugier, R., Kammerer, J.,
> Alei, E., Quanz, S. P. *Photobombing for the Large Interferometer For
> Exoplanets (LIFE): A New Criterion for Target Confusion and
> Application to a Mid-infrared Rotating Nulling Interferometer*.
> The Astronomical Journal, **169**, 244 (18 pp.), May 2025.
> [doi:10.3847/1538-3881/adbefc](https://doi.org/10.3847/1538-3881/adbefc)

## License

Source code: MIT. Paper figures (`life_dmap_*.png`,
`cacaj_2025_*.png`): © 2025 The American Astronomical Society, used
with attribution.
