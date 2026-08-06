# Summer Project — TESS + LOFAR Stellar Activity & Radio Burst Study

Cross-matches nearby stars (within 100 pc) with both TESS optical light
curves and LOFAR/LoTSS radio pointings, to study stellar flare activity
and search for radio bursts (Type II / Type III) that may coincide with
optical flares.

## Files

### `catalog_utils.py`
Shared catalog loader. Loads `gaia_info_100pc_lc_info_lotss.csv` (Gaia +
TESS + LoTSS cross-matched catalog) into `tab`, and provides
`get_catalog_row(tic_id)` to look up a star's info by TIC ID. Both other
scripts import from this file, so the catalog is only ever loaded once.

### `tess_lotss-project.py`
Radio/LOFAR side. Defines `DynamicSpectrum`, a wrapper around a LOFAR
dynamic-spectrum FITS cube (`[stokes, freq, time]`), with methods to plot
the dynamic spectrum, light curve, and spectrum, get basic statistics, and
`dedisperse()` to correct for frequency-dependent dispersive delay.

**Note:** this file cannot be imported as a module (its filename contains
a hyphen), so other scripts that need `DynamicSpectrum` expect it to
already be defined in the session — run this file first (F5 / Run File)
before running `burst_detection.py`.

### `flare_pipeline.py`
Optical/TESS side. Contains Ivey's `Star` and `Flares` classes (downloads
TESS light curves, detects flares via iterative sigma-clipping, computes
flare energy/duration/rate). Adds:
- `get_tic_radius(tic_id)` — queries the TESS Input Catalog for stellar
  radius, since the Gaia/LoTSS catalog doesn't include it.
- `process_star_flares(tic_id)` — full per-star pipeline: catalog lookup
  → build `Star` → run `Flares.FindAllFlares()` → save results to disk.
- `run_batch()` — runs `process_star_flares()` over every star in the
  catalog, adds `n_flares`, `flare_rate`, `peak_flare_energy` columns, and
  writes a new CSV.

Per-star output (light curve sections, flare table, metadata) is saved to
`flare_results/TIC<id>/` — **not tracked in git** (see `.gitignore`), since
it's regenerable output data, not source code.

### `burst_detection.py`
Detects and classifies Type II / Type III radio bursts in a
`DynamicSpectrum` object: flags excess emission above the noise floor,
groups connected pixels, fits a drift rate, classifies by drift rate +
duration. Includes `recover_burst()` to dedisperse a detected burst's
region and recover a clean time profile.

## Setup

```
conda activate astro
pip install astroquery   # needed for get_tic_radius()
```

Update the hardcoded paths at the top of `catalog_utils.py` and
`tess_lotss-project.py` (`env1`, `env2`) to match where the catalog CSV
and LOFAR FITS file live on your machine.

## Usage

```python
# Radio side
run tess_lotss-project.py            # defines DynamicSpectrum, loads fn_fits
ds = DynamicSpectrum(fn_fits)
ds.plot_dyn_spec(stokes="i")

# Burst detection (after the above)
from burst_detection import detect_bursts, recover_burst
bursts = detect_bursts(ds, sigma_threshold=5.0)

# Optical/flare side
run flare_pipeline.py
summary = process_star_flares(470085072, table=tab, sectors=None)
```

## Status / known issues

- Flare pipeline tested end-to-end on one star (TIC 470085072, 0 flares
  found, radius 0.753 R_sun) — not yet run across the full catalog
  (`run_batch()`).
- `sectors=None` should be used (not the default `range(14)`) to search
  all TESS sectors, not just the first ~14.
- Known bug in `Star.SplitLightCurve`: length filter uses `window` instead
  of `eclipse_window`, which can zero-pad short light-curve sections
  during flagging. Not yet fixed upstream in Ivey's code.
- `LightCurvetoTab`'s ascii `.tab` writer is slow on large light curves
  (can take several minutes); consider switching to FITS binary tables.
- Burst detection (`burst_detection.py`) and injection/recovery testing
  (`burst_pipeline.py`) are both untested — the LOFAR FITS file
  (`L2023107_21_48_11.743_+79_13_02.884.fits`) needs to be re-sourced.
- Type II/III classification thresholds in `classify_burst()` are
  literature starting points, not validated against confirmed real bursts
  yet.