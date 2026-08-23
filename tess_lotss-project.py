"""
LoTSS / LOFAR Stokes-I light-curve and trial-DM analysis.

This script:
1. Loads a LoTSS/LOFAR dynamic-spectrum FITS cube.
2. Extracts Stokes I.
3. Performs frequency-channel quality control.
4. Builds a frequency-averaged light curve.
5. Measures baseline, peak, noise, excess signal and S/N.
6. Produces diagnostic plots.
7. Searches a grid of trial dispersion measures (DMs).
8. Produces the best trial-DM dedispersed dynamic spectrum.
9. Optionally looks up a TIC ID in a Gaia/LoTSS catalogue and retrieves TESS SPOC light curves.

Expected FITS data shape:
    (n_stokes, n_frequency, n_time)
"""

from pathlib import Path
from warnings import warn

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time


# ===============================================================
# 1. USER SETTINGS
# ===============================================================

CATALOG_FILE = Path(
    r"C:/Users/ADMIN/Downloads/gaia_info_100pc_lc_info_lotss.csv"
)

FITS_FILE = Path(
    r"C:/Users/ADMIN/python_projects/"
    r"L2023107_21_48_11.743_+79_13_02.884.fits"
)

OUTPUT_DIR = Path(
    r"C:/Users/ADMIN/python_projects/lotss_stokes_i_results"
)

STOKES = "i"

# None = use the complete observing band.
# Example: (120.0, 160.0)
FREQUENCY_RANGE_MHZ = None

# None = use the complete observation.
# Example: (0.0, 300.0)
TIME_RANGE_SECONDS = None

SAVE_RESULTS = True

# Trial-DM search settings.
TRIAL_DM_MIN = 0.0
TRIAL_DM_MAX = 100.0
TRIAL_DM_STEP = 0.5

# Optional catalogue lookup.
TIC_ID = 470085072


# ===============================================================
# 2. OPTIONAL TESS HELPER
# ===============================================================

def get_lightcurve(idx: int, table):
    """Download all available SPOC 2-minute TESS light curves for a catalogue row."""
    try:
        tic_id = int(table["ID"][idx])
    except Exception as exc:
        print(f"Could not read TIC ID at catalogue row {idx}: {exc}")
        return None

    try:
        import lightkurve as lk
    except ImportError:
        warn("lightkurve is not installed; TESS lookup skipped.")
        return None

    tic_name = f"TIC {tic_id}"
    search = lk.search_lightcurve(
        tic_name,
        author="SPOC",
        exptime=120,
    )

    if len(search) == 0:
        print(f"No TESS SPOC light curves found for {tic_name}.")
        return None

    return search.download_all()


def get_catalog_row(tic_value, table):
    """
    Find a catalogue row using a TIC ID.

    Returns
    -------
    idx : int
        Index of the matching row.
    row : astropy.table.Row
        Complete matching catalogue row.
    tic_name : str
        String such as 'TIC 470085072'.
    row_info : dict
        Catalogue row converted to a normal dictionary.
    """
    if isinstance(tic_value, str):
        tic_value = tic_value.upper().replace("TIC", "").strip()

    tic_value = int(tic_value)

    if "ID" not in table.colnames:
        raise KeyError("Catalogue does not contain an 'ID' column.")

    # Convert values to integers where possible to avoid string/integer
    # comparison problems.
    ids = np.asarray(table["ID"])

    try:
        matches = np.where(ids.astype(np.int64) == tic_value)[0]
    except (TypeError, ValueError):
        matches = np.where(ids.astype(str) == str(tic_value))[0]

    if len(matches) == 0:
        raise ValueError(f"TIC {tic_value} not found in the catalogue.")

    if len(matches) > 1:
        warn(
            f"{len(matches)} catalogue rows match TIC {tic_value}; "
            "using the first match."
        )

    idx = int(matches[0])
    row = table[idx]
    tic_name = f"TIC {tic_value}"
    row_info = {column: row[column] for column in table.colnames}

    return idx, row, tic_name, row_info


# ===============================================================
# 3. ROBUST STATISTICS
# ===============================================================

def robust_sigma(values):
    """
    Estimate Gaussian-equivalent sigma using the median absolute deviation.

    sigma = 1.4826 * MAD
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan

    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))

    return 1.4826 * mad


# ===============================================================
# 4. DYNAMIC SPECTRUM CLASS
# ===============================================================

class DynamicSpectrum:
    """
    Wrapper around a LOFAR/LoTSS dynamic-spectrum FITS cube.

    Expected data shape:
        (n_stokes, n_frequency, n_time)
    """

    STOKES_MAP = {
        "i": 0,
        "q": 1,
        "u": 2,
        "v": 3,
    }

    def __init__(self, filename):
        self.filename = Path(filename)

        if not self.filename.exists():
            raise FileNotFoundError(
                f"FITS file not found:\n{self.filename}"
            )

        with fits.open(self.filename) as hdul:
            self.data = np.asarray(hdul[0].data, dtype=float)
            self.header = hdul[0].header.copy()

        if self.data.ndim != 3:
            raise ValueError(
                "Expected FITS data with shape "
                "(n_stokes, n_frequency, n_time). "
                f"Found shape: {self.data.shape}"
            )

        self.n_stokes, self.n_freq, self.n_time = self.data.shape

        if self.n_stokes < 1:
            raise ValueError("No Stokes parameters found.")

        self._read_time_axis()
        self._read_frequency_axis()

    def _read_time_axis(self):
        """Read observation start/stop times and construct the MJD/time axes."""
        start_isot = self.header.get("OBS-STAR")
        end_isot = self.header.get("OBS-STOP")

        if start_isot is None or end_isot is None:
            raise KeyError(
                "OBS-STAR and/or OBS-STOP are missing from the FITS header."
            )

        self.start_time_isot = str(start_isot)
        self.end_time_isot = str(end_isot)

        self.start_time_mjd = Time(
            self.start_time_isot,
            format="isot",
        ).mjd

        self.end_time_mjd = Time(
            self.end_time_isot,
            format="isot",
        ).mjd

        if self.n_time == 1:
            self.time_axis = np.array([self.start_time_mjd])
            self.time_seconds = np.array([0.0])
            self.time_resolution = np.nan
            return

        self.time_axis = np.linspace(
            self.start_time_mjd,
            self.end_time_mjd,
            self.n_time,
        )

        self.time_seconds = (
            self.time_axis - self.time_axis[0]
        ) * 86400.0

        self.time_resolution = float(
            np.nanmedian(np.diff(self.time_seconds))
        )

    def _read_frequency_axis(self):
        """Read frequency limits from the FITS header and convert Hz to MHz."""
        freq_max_hz = self.header.get("FRQ-MAX")
        freq_min_hz = self.header.get("FRQ-MIN")

        if freq_max_hz is None or freq_min_hz is None:
            raise KeyError(
                "FRQ-MAX and/or FRQ-MIN are missing from the FITS header."
            )

        self.freq_min_mhz = float(freq_min_hz) / 1e6
        self.freq_max_mhz = float(freq_max_hz) / 1e6

        self.freq_axis = np.linspace(
            self.freq_min_mhz,
            self.freq_max_mhz,
            self.n_freq,
        )

    def _get_stokes(self, stokes="i"):
        """Return a 2-D frequency x time array for the requested Stokes parameter."""
        if isinstance(stokes, str):
            stokes = stokes.lower()
            if stokes not in self.STOKES_MAP:
                raise ValueError("Stokes must be i, q, u, or v.")
            index = self.STOKES_MAP[stokes]
        else:
            index = int(stokes)

        if index < 0 or index >= self.n_stokes:
            raise IndexError(
                f"Stokes index {index} is unavailable. "
                f"Number of Stokes parameters = {self.n_stokes}."
            )

        return self.data[index, :, :]

    def get_stokes(self, stokes="i"):
        """Public interface for retrieving a Stokes parameter."""
        return self._get_stokes(stokes)

    def get_statistics(self, time_idx=None, freq_idx=None, stokes="i"):
        """Return standard deviation, maximum, minimum and median."""
        data = self._get_stokes(stokes)

        if time_idx is None and freq_idx is None:
            sub = data
        elif time_idx is not None and freq_idx is None:
            sub = data[:, time_idx]
        elif freq_idx is not None and time_idx is None:
            sub = data[freq_idx, :]
        else:
            sub = data[freq_idx, time_idx]

        finite = np.isfinite(sub)

        if not np.any(finite):
            return np.nan, np.nan, np.nan, np.nan

        values = sub[finite]

        return (
            float(np.nanstd(values)),
            float(np.nanmax(values)),
            float(np.nanmin(values)),
            float(np.nanmedian(values)),
        )

    def plot_dyn_spec(self, stokes="i", cmap="inferno"):
        """Plot a dynamic spectrum."""
        data = self._get_stokes(stokes)

        fig, ax = plt.subplots(figsize=(12, 7))

        vmin = np.nanpercentile(data, 5)
        vmax = np.nanpercentile(data, 99)

        image = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=[
                self.time_axis[0],
                self.time_axis[-1],
                self.freq_axis[0],
                self.freq_axis[-1],
            ],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        fig.colorbar(
            image,
            ax=ax,
            label=f"Stokes {str(stokes).upper()} Flux Density (Jy)",
        )

        ax.set_xlabel("Time (MJD)")
        ax.set_ylabel("Frequency (MHz)")
        ax.set_title(
            f"LoTSS Stokes-{str(stokes).upper()} Dynamic Spectrum",
            fontweight="bold",
        )

        plt.tight_layout()
        return fig, ax

    def plot_lightcurve(
        self,
        freq_idx=None,
        freq_range=None,
        stokes="i",
    ):
        """Plot flux versus time."""
        data = self._get_stokes(stokes)

        if freq_idx is not None:
            lc = data[freq_idx, :]
            label = (
                f"channel {freq_idx} "
                f"({self.freq_axis[freq_idx]:.1f} MHz)"
            )

        elif freq_range is not None:
            i0, i1 = freq_range

            if i0 < 0 or i1 > self.n_freq or i0 >= i1:
                raise ValueError("Invalid frequency-channel range.")

            lc = np.nanmean(data[i0:i1, :], axis=0)

            label = (
                f"{self.freq_axis[i0]:.1f}-"
                f"{self.freq_axis[i1 - 1]:.1f} MHz averaged"
            )

        else:
            lc = np.nanmean(data, axis=0)
            label = "full-band averaged"

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(self.time_axis, lc)

        ax.set_xlabel("Time (MJD)")
        ax.set_ylabel(
            f"Stokes {str(stokes).upper()} Flux Density (Jy)"
        )
        ax.set_title(f"Light Curve - {label}")

        plt.tight_layout()
        return fig, ax

    def plot_spectrum(
        self,
        time_idx=None,
        time_range=None,
        stokes="i",
    ):
        """Plot flux versus frequency."""
        data = self._get_stokes(stokes)

        if time_idx is not None:
            spec = data[:, time_idx]
            label = f"t = {self.time_axis[time_idx]:.5f} MJD"

        elif time_range is not None:
            i0, i1 = time_range

            if i0 < 0 or i1 > self.n_time or i0 >= i1:
                raise ValueError("Invalid time-sample range.")

            spec = np.nanmean(data[:, i0:i1], axis=1)

            label = (
                f"t = {self.time_axis[i0]:.5f}-"
                f"{self.time_axis[i1 - 1]:.5f} MJD averaged"
            )

        else:
            spec = np.nanmean(data, axis=1)
            label = "full-duration averaged"

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.freq_axis, spec)

        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel(
            f"Stokes {str(stokes).upper()} Flux Density (Jy)"
        )
        ax.set_title(f"Spectrum - {label}")

        plt.tight_layout()
        return fig, ax

    def get_tess_lightcurve(self):
        """Retrieve TESS SPOC 2-minute light curves using the FITS NAME keyword."""
        name = self.header.get("NAME")

        if name is None:
            print(
                "No source-name keyword found in the FITS header; "
                "inspect ds.header."
            )
            return None

        try:
            import lightkurve as lk
        except ImportError:
            warn("lightkurve is not installed; TESS lookup skipped.")
            return None

        search = lk.search_lightcurve(
            name,
            author="SPOC",
            exptime=120,
        )

        if len(search) == 0:
            print(f"No TESS SPOC light curves found for {name}.")
            return None

        self.lcs = search.download_all()
        return self.lcs

    def dedisperse(self, a, alpha, stokes="i"):
        """
        Apply the original power-law time-shift model used in the project.

        delta_t =
            1 / [a * (alpha - 1)] *
            [nu^(1-alpha) - nu0^(1-alpha)]

        This method is retained for compatibility with the earlier workflow.
        For the trial-DM search, use dedisperse_dynamic_spectrum().
        """
        if self.n_time < 2:
            raise ValueError("At least two time samples are required.")

        if a == 0 or alpha == 1:
            raise ValueError("a must be non-zero and alpha must not equal 1.")

        data = self._get_stokes(stokes)
        freqs = np.asarray(self.freq_axis, dtype=float)

        # Work from high frequency to low frequency.
        freqs_flip = np.flip(freqs)
        shifted = np.flip(data.copy(), axis=0)

        dt = self.time_resolution
        nu0 = freqs_flip[0]
        exponent = 1.0 - alpha

        for j, nu in enumerate(freqs_flip):
            delta_t = (
                1.0 / (a * (alpha - 1.0))
                * (nu**exponent - nu0**exponent)
            )

            shift_bins = int(delta_t / dt)

            if shift_bins == 0:
                continue

            shifted[j] = np.roll(shifted[j], -shift_bins)

            if shift_bins > 0:
                shifted[j, -shift_bins:] = np.nan

        return np.flip(shifted, axis=0)


# ===============================================================
# 5. FREQUENCY-CHANNEL QUALITY CONTROL
# ===============================================================

def find_good_channels(
    data,
    min_finite_fraction=0.80,
    sigma_clip_factor=10.0,
):
    """
    Identify usable frequency channels.

    A channel is rejected when:
    - too many values are non-finite, or
    - its robust noise is much larger than the typical channel noise.
    """
    data = np.asarray(data, dtype=float)

    finite_fraction = np.mean(
        np.isfinite(data),
        axis=1,
    )

    channel_sigma = np.full(
        data.shape[0],
        np.nan,
        dtype=float,
    )

    for i in range(data.shape[0]):
        channel_sigma[i] = robust_sigma(data[i])

    good_finite = finite_fraction >= min_finite_fraction

    if np.any(good_finite):
        typical_sigma = np.nanmedian(
            channel_sigma[good_finite]
        )
    else:
        typical_sigma = np.nan

    if not np.isfinite(typical_sigma) or typical_sigma <= 0:
        good_noise = np.isfinite(channel_sigma)
    else:
        good_noise = (
            np.isfinite(channel_sigma)
            & (
                channel_sigma
                <= sigma_clip_factor * typical_sigma
            )
        )

    good_channels = good_finite & good_noise

    return good_channels, channel_sigma


# ===============================================================
# 6. CONSTRUCT STOKES-I LIGHT CURVE
# ===============================================================

def make_lightcurve(
    ds,
    stokes="i",
    frequency_range_mhz=None,
    time_range_seconds=None,
):
    """
    Construct a frequency-averaged light curve.

    Returns
    -------
    time_mjd
    time_seconds
    lightcurve
    selected_frequency_mask
    time_mask
    channel_sigma
    """
    data = ds.get_stokes(stokes)

    # -----------------------------------------------------------
    # Frequency selection
    # -----------------------------------------------------------

    if frequency_range_mhz is None:
        freq_mask = np.ones(
            ds.n_freq,
            dtype=bool,
        )
    else:
        f0, f1 = frequency_range_mhz

        if f0 > f1:
            f0, f1 = f1, f0

        freq_mask = (
            (ds.freq_axis >= f0)
            & (ds.freq_axis <= f1)
        )

    if np.sum(freq_mask) == 0:
        raise ValueError(
            "No frequency channels fall inside the requested range."
        )

    # -----------------------------------------------------------
    # Frequency-channel quality control
    # -----------------------------------------------------------

    good_channels, channel_sigma = find_good_channels(data)

    selected_freq_mask = freq_mask & good_channels

    if np.sum(selected_freq_mask) == 0:
        raise ValueError(
            "No usable frequency channels remain after quality control."
        )

    # -----------------------------------------------------------
    # Time selection
    # -----------------------------------------------------------

    if time_range_seconds is None:
        time_mask = np.ones(
            ds.n_time,
            dtype=bool,
        )
    else:
        t0, t1 = time_range_seconds

        if t0 > t1:
            t0, t1 = t1, t0

        time_mask = (
            (ds.time_seconds >= t0)
            & (ds.time_seconds <= t1)
        )

    if np.sum(time_mask) == 0:
        raise ValueError(
            "No time samples fall inside the requested time range."
        )

    # -----------------------------------------------------------
    # Frequency average
    # -----------------------------------------------------------

    selected_data = data[selected_freq_mask, :]

    lightcurve_full = np.nanmean(
        selected_data,
        axis=0,
    )

    lightcurve = lightcurve_full[time_mask]

    time_mjd = ds.time_axis[time_mask]

    time_seconds = ds.time_seconds[time_mask]
    time_seconds = time_seconds - time_seconds[0]

    return (
        time_mjd,
        time_seconds,
        lightcurve,
        selected_freq_mask,
        time_mask,
        channel_sigma,
    )


# ===============================================================
# 7. OBSERVED PEAK ANALYSIS
# ===============================================================

def analyze_peak(
    time_mjd,
    time_seconds,
    lightcurve,
):
    """
    Measure baseline, peak, noise, excess signal and S/N.

    Baseline:
        median of the observed light curve.

    Noise:
        Sample standard deviation of the observed light curve.
        The standard deviation is calculated with ddof=1.
    """
    time_mjd = np.asarray(time_mjd, dtype=float)
    time_seconds = np.asarray(time_seconds, dtype=float)
    lightcurve = np.asarray(lightcurve, dtype=float)

    valid = (
        np.isfinite(time_mjd)
        & np.isfinite(time_seconds)
        & np.isfinite(lightcurve)
    )

    if np.sum(valid) < 3:
        raise ValueError(
            "Not enough valid light-curve points."
        )

    t_mjd = time_mjd[valid]
    t_sec = time_seconds[valid]
    flux = lightcurve[valid]

    baseline = np.nanmedian(flux)

    residuals = flux - baseline

    # Standard deviation of the observed light curve.
    # ddof=1 gives the sample standard deviation:
    #
    # sigma = sqrt( sum((x_i - mean(x))^2) / (N - 1) )
    #
    # This is the noise estimate used for the peak S/N.
    noise_sigma = np.std(flux, ddof=1)

    peak_idx = int(np.nanargmax(flux))

    peak_flux = flux[peak_idx]
    peak_time_mjd = t_mjd[peak_idx]
    peak_time_seconds = t_sec[peak_idx]

    peak_signal = peak_flux - baseline

    if np.isfinite(noise_sigma) and noise_sigma > 0:
        peak_snr = peak_signal / noise_sigma
    else:
        peak_snr = np.nan

    if np.isfinite(baseline) and baseline != 0:
        normalized = (flux - baseline) / baseline
    else:
        normalized = np.full_like(flux, np.nan)

    peak_fraction = normalized[peak_idx]
    peak_percent = 100.0 * peak_fraction

    rms = np.sqrt(np.nanmean(residuals**2))

    return {
        "time_mjd": t_mjd,
        "time_seconds": t_sec,
        "flux": flux,
        "baseline_jy": float(baseline),
        "peak_flux_jy": float(peak_flux),
        "peak_signal_jy": float(peak_signal),
        "peak_time_mjd": float(peak_time_mjd),
        "peak_time_seconds": float(peak_time_seconds),
        "noise_sigma_jy": float(noise_sigma),
        "noise_method": "sample standard deviation (ddof=1)",
        "rms_jy": float(rms),
        "peak_snr": float(peak_snr),
        "normalized": normalized,
        "peak_fraction": float(peak_fraction),
        "peak_percent": float(peak_percent),
    }


def calculate_snr_lightcurve(
    flux,
    baseline,
    noise_sigma,
):
    """Calculate baseline-subtracted S/N at every light-curve sample."""
    flux = np.asarray(flux, dtype=float)

    if (
        not np.isfinite(noise_sigma)
        or noise_sigma <= 0
    ):
        return np.full_like(
            flux,
            np.nan,
            dtype=float,
        )

    return (flux - baseline) / noise_sigma


# ===============================================================
# 8. STANDARD PLOTS
# ===============================================================

def plot_dynamic_spectrum(
    ds,
    stokes="i",
    frequency_mask=None,
    cmap="inferno",
):
    """Plot the selected dynamic spectrum."""
    data = ds.get_stokes(stokes)

    if frequency_mask is None:
        frequency_mask = np.ones(
            ds.n_freq,
            dtype=bool,
        )

    plot_data = data[frequency_mask, :]
    freq = ds.freq_axis[frequency_mask]

    if plot_data.size == 0:
        raise ValueError("No data available for dynamic-spectrum plot.")

    vmin = np.nanpercentile(plot_data, 5)
    vmax = np.nanpercentile(plot_data, 99)

    fig, ax = plt.subplots(figsize=(12, 7))

    image = ax.imshow(
        plot_data,
        aspect="auto",
        origin="lower",
        extent=[
            ds.time_axis[0],
            ds.time_axis[-1],
            freq[0],
            freq[-1],
        ],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(
        f"Stokes {str(stokes).upper()} Flux Density (Jy)"
    )

    ax.set_xlabel("Time (MJD)")
    ax.set_ylabel("Frequency (MHz)")
    ax.set_title(
        f"LoTSS Stokes-{str(stokes).upper()} Dynamic Spectrum",
        fontweight="bold",
    )

    plt.tight_layout()

    return fig, ax


def plot_peak_analysis(results):
    """Plot the observed light curve with baseline, noise and peak."""
    t = results["time_seconds"]
    flux = results["flux"]

    baseline = results["baseline_jy"]
    noise = results["noise_sigma_jy"]
    peak_time = results["peak_time_seconds"]
    peak_flux = results["peak_flux_jy"]
    peak_signal = results["peak_signal_jy"]
    peak_snr = results["peak_snr"]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        t,
        flux,
        linewidth=1.2,
        label="Observed Stokes I",
    )

    ax.axhline(
        baseline,
        linestyle="--",
        linewidth=2,
        label=f"Baseline = {baseline:.4f} Jy",
    )

    ax.axhline(
        baseline + noise,
        linestyle=":",
        linewidth=1.5,
        label=f"+1σ = {noise:.4f} Jy",
    )

    ax.axhline(
        baseline - noise,
        linestyle=":",
        linewidth=1.5,
    )

    ax.scatter(
        peak_time,
        peak_flux,
        s=90,
        marker="o",
        zorder=5,
        label=f"Peak = {peak_flux:.4f} Jy",
    )

    ax.axvline(
        peak_time,
        linestyle="--",
        linewidth=1.5,
    )

    ax.annotate(
        "",
        xy=(peak_time, peak_flux),
        xytext=(peak_time, baseline),
        arrowprops=dict(
            arrowstyle="<->",
            linewidth=1.5,
        ),
    )

    ax.text(
        peak_time,
        (peak_flux + baseline) / 2,
        f"Peak signal = {peak_signal:.4f} Jy",
        ha="left",
        va="center",
    )

    text = (
        f"Peak time = {peak_time:.3f} s\n"
        f"Peak flux = {peak_flux:.4f} Jy\n"
        f"Baseline = {baseline:.4f} Jy\n"
        f"Noise = {noise:.4f} Jy\n"
        f"Peak signal = {peak_signal:.4f} Jy\n"
        f"Peak S/N = {peak_snr:.2f}σ"
    )

    ax.text(
        0.02,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        bbox=dict(
            boxstyle="round",
            alpha=0.85,
        ),
    )

    ax.set_xlabel("Time since start of observation (s)")
    ax.set_ylabel("Stokes I Flux Density (Jy)")
    ax.set_title(
        "Observed Stokes-I Light Curve: Peak and Signal-to-Noise",
        fontweight="bold",
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    return fig, ax


def plot_normalized_signal(results):
    """Plot the baseline-normalized signal."""
    t = results["time_seconds"]

    normalized_percent = (
        results["normalized"] * 100.0
    )

    peak_time = results["peak_time_seconds"]
    peak_percent = results["peak_percent"]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        t,
        normalized_percent,
        linewidth=1.2,
        label="Baseline-normalized signal",
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1.5,
    )

    ax.axvline(
        peak_time,
        linestyle="--",
        linewidth=1.5,
    )

    ax.scatter(
        peak_time,
        peak_percent,
        s=90,
        marker="o",
        zorder=5,
        label=f"Peak increase = {peak_percent:.2f}%",
    )

    ax.set_xlabel("Time since start of observation (s)")
    ax.set_ylabel("Flux change relative to baseline (%)")
    ax.set_title(
        "Baseline-Normalized Stokes-I Signal",
        fontweight="bold",
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    return fig, ax


def plot_snr(results):
    """Plot the S/N light curve."""
    t = results["time_seconds"]

    snr = calculate_snr_lightcurve(
        results["flux"],
        results["baseline_jy"],
        results["noise_sigma_jy"],
    )

    peak_time = results["peak_time_seconds"]
    peak_snr = results["peak_snr"]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        t,
        snr,
        linewidth=1.2,
        label="S/N",
    )

    ax.axhline(
        3,
        linestyle="--",
        linewidth=1.2,
        label="3σ",
    )

    ax.axhline(
        5,
        linestyle="--",
        linewidth=1.2,
        label="5σ",
    )

    ax.axhline(
        0,
        linestyle=":",
        linewidth=1,
    )

    ax.scatter(
        peak_time,
        peak_snr,
        s=90,
        marker="o",
        zorder=5,
        label=f"Peak S/N = {peak_snr:.2f}σ",
    )

    ax.axvline(
        peak_time,
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_xlabel("Time since start of observation (s)")
    ax.set_ylabel("Signal-to-noise ratio (σ)")
    ax.set_title(
        "Stokes-I Signal-to-Noise Ratio",
        fontweight="bold",
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    return fig, ax


# ===============================================================
# 9. TRIAL-DM DEDISPERSION
# ===============================================================

def dedisperse_dynamic_spectrum(
    dynamic_spectrum,
    frequency_mhz,
    time_seconds,
    dispersion_measure,
):
    """
    Dedisperse a frequency-time dynamic spectrum.

    The delay relative to the highest observing frequency is:

        delay_ms = 4.148808 * DM *
                   (nu_GHz^-2 - nu_ref_GHz^-2)

    where:
        DM is in pc cm^-3
        frequency is in GHz.

    Returns
    -------
    dedispersed : ndarray
        Dedispersed dynamic spectrum, shape (frequency, time).
    delays_seconds : ndarray
        Delay for each frequency channel relative to the reference frequency.
    """
    data = np.asarray(
        dynamic_spectrum,
        dtype=float,
    )

    freq = np.asarray(
        frequency_mhz,
        dtype=float,
    )

    time = np.asarray(
        time_seconds,
        dtype=float,
    )

    if data.ndim != 2:
        raise ValueError(
            "Dynamic spectrum must be a 2-D array "
            "(frequency, time)."
        )

    if data.shape[0] != freq.size:
        raise ValueError(
            "Frequency array length does not match "
            "dynamic-spectrum frequency dimension."
        )

    if data.shape[1] != time.size:
        raise ValueError(
            "Time array length does not match "
            "dynamic-spectrum time dimension."
        )

    if time.size < 2:
        raise ValueError(
            "At least two time samples are required."
        )

    if not np.all(np.isfinite(time)):
        raise ValueError("Time array contains non-finite values.")

    dt = np.nanmedian(np.diff(time))

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Time samples must be strictly increasing.")

    reference_frequency_mhz = np.nanmax(freq)

    freq_ghz = freq / 1000.0
    reference_frequency_ghz = (
        reference_frequency_mhz / 1000.0
    )

    delays_ms = (
        4.148808
        * float(dispersion_measure)
        * (
            freq_ghz ** (-2.0)
            - reference_frequency_ghz ** (-2.0)
        )
    )

    delays_seconds = delays_ms / 1000.0

    dedispersed = np.full_like(
        data,
        np.nan,
        dtype=float,
    )

    for channel_index in range(data.shape[0]):
        channel = data[channel_index]
        finite = np.isfinite(channel)

        if np.count_nonzero(finite) < 2:
            continue

        # Observed signal at t + delay corresponds to the
        # dedispersed signal at t.
        dedispersed[channel_index] = np.interp(
            time + delays_seconds[channel_index],
            time[finite],
            channel[finite],
            left=np.nan,
            right=np.nan,
        )

    return dedispersed, delays_seconds


def trial_dm_detection_statistic(
    dedispersed_dynamic_spectrum,
):
    """
    Calculate a simple statistic for ranking trial DMs.

    The frequency channels are averaged to produce a time series.
    The statistic is:

        max(abs(signal - baseline)) / std(signal - baseline)

    This is a trial-DM ranking statistic, not a calibrated
    astrophysical significance.
    """
    frequency_averaged = np.nanmean(
        dedispersed_dynamic_spectrum,
        axis=0,
    )

    finite = np.isfinite(frequency_averaged)

    if np.count_nonzero(finite) < 3:
        return np.nan

    baseline = np.nanmedian(
        frequency_averaged[finite]
    )

    residual = (
        frequency_averaged - baseline
    )

    noise = np.nanstd(
        residual[finite],
        ddof=1,
    )

    if not np.isfinite(noise) or noise <= 0:
        return np.nan

    peak_signal = np.nanmax(
        np.abs(residual[finite])
    )

    return float(peak_signal / noise)


def search_trial_dms(
    dynamic_spectrum,
    frequency_mhz,
    time_seconds,
    dm_min,
    dm_max,
    dm_step,
):
    """Search a grid of trial DMs and select the highest-ranking DM."""
    if dm_step <= 0:
        raise ValueError("TRIAL_DM_STEP must be > 0.")

    if dm_max < dm_min:
        raise ValueError(
            "TRIAL_DM_MAX must be >= TRIAL_DM_MIN."
        )

    trial_dms = np.arange(
        dm_min,
        dm_max + 0.5 * dm_step,
        dm_step,
    )

    statistics = np.full(
        trial_dms.size,
        np.nan,
        dtype=float,
    )

    best_dm = np.nan
    best_statistic = -np.inf
    best_dynamic = None
    best_delays = None

    print()
    print("=" * 70)
    print("TRIAL-DM SEARCH")
    print("=" * 70)
    print(
        f"DM range : {dm_min:.3f} to "
        f"{dm_max:.3f} pc cm^-3"
    )
    print(f"DM step  : {dm_step:.3f} pc cm^-3")
    print(f"Number of trial DMs: {trial_dms.size}")

    for i, dm in enumerate(trial_dms):
        dedispersed, delays = dedisperse_dynamic_spectrum(
            dynamic_spectrum,
            frequency_mhz,
            time_seconds,
            dm,
        )

        statistic = trial_dm_detection_statistic(
            dedispersed
        )

        statistics[i] = statistic

        if (
            np.isfinite(statistic)
            and statistic > best_statistic
        ):
            best_statistic = statistic
            best_dm = dm
            best_dynamic = dedispersed
            best_delays = delays

    if best_dynamic is None:
        raise RuntimeError(
            "Trial-DM search failed: no valid DM "
            "produced a finite detection statistic."
        )

    print("-" * 70)
    print(
        f"Best trial DM       : "
        f"{best_dm:.6f} pc cm^-3"
    )
    print(
        f"Best DM statistic   : "
        f"{best_statistic:.6f}"
    )
    print("=" * 70)

    return (
        trial_dms,
        statistics,
        best_dm,
        best_statistic,
        best_dynamic,
        best_delays,
    )


def plot_dm_search(
    trial_dms,
    statistics,
    best_dm,
):
    """Plot the trial-DM detection statistic."""
    fig, ax = plt.subplots(
        figsize=(10, 5),
    )

    ax.plot(
        trial_dms,
        statistics,
        linewidth=1.5,
    )

    ax.axvline(
        best_dm,
        linestyle="--",
        linewidth=1.5,
        label=f"Best DM = {best_dm:.3f}",
    )

    ax.set_xlabel(
        "Trial DM (pc cm$^{-3}$)"
    )

    ax.set_ylabel(
        "Dedispersion detection statistic"
    )

    ax.set_title("Trial-DM Search")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    return fig, ax


def plot_dedispersed_dynamic_spectrum(
    dedispersed_data,
    frequency_mhz,
    time_seconds,
    best_dm,
    cmap="inferno",
):
    """Plot the final dedispersed Stokes-I dynamic spectrum."""
    finite = np.isfinite(dedispersed_data)

    if np.count_nonzero(finite) == 0:
        raise RuntimeError(
            "The dedispersed dynamic spectrum contains "
            "no finite values."
        )

    vmin = np.nanpercentile(
        dedispersed_data,
        5,
    )

    vmax = np.nanpercentile(
        dedispersed_data,
        99,
    )

    fig, ax = plt.subplots(
        figsize=(12, 7),
    )

    image = ax.imshow(
        dedispersed_data,
        aspect="auto",
        origin="lower",
        extent=[
            time_seconds[0],
            time_seconds[-1],
            frequency_mhz[0],
            frequency_mhz[-1],
        ],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(
        image,
        ax=ax,
    )

    cbar.set_label(
        "Stokes I Flux Density (Jy)"
    )

    ax.set_xlabel(
        "Time since start of observation (s)"
    )

    ax.set_ylabel(
        "Frequency (MHz)"
    )

    ax.set_title(
        "Dedispersed Stokes-I Dynamic Spectrum\n"
        f"Best trial DM = {best_dm:.4f} pc cm$^{{-3}}",
        fontweight="bold",
    )

    plt.tight_layout()

    return fig, ax


# ===============================================================
# 10. SAVE NUMERICAL RESULTS
# ===============================================================

def save_peak_results(results, output_dir):
    """Save observed peak-analysis results to a text file."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_file = (
        output_dir
        / "stokes_i_peak_results.txt"
    )

    with open(
        result_file,
        "w",
        encoding="utf-8",
    ) as f:
        f.write("LoTSS Stokes-I Peak Analysis\n")
        f.write("============================\n\n")
        f.write(
            f"Baseline flux (Jy): "
            f"{results['baseline_jy']:.8f}\n"
        )
        f.write(
            f"Peak flux (Jy): "
            f"{results['peak_flux_jy']:.8f}\n"
        )
        f.write(
            f"Peak signal above baseline (Jy): "
            f"{results['peak_signal_jy']:.8f}\n"
        )
        f.write(
            f"Peak time (MJD): "
            f"{results['peak_time_mjd']:.10f}\n"
        )
        f.write(
            f"Peak time since start (s): "
            f"{results['peak_time_seconds']:.6f}\n"
        )
        f.write(
            f"Noise sigma (Jy): "
            f"{results['noise_sigma_jy']:.8f}\n"
        )
        f.write(
            f"RMS about baseline (Jy): "
            f"{results['rms_jy']:.8f}\n"
        )
        f.write(
            f"Peak S/N: "
            f"{results['peak_snr']:.4f}\n"
        )
        f.write(
            f"Peak fractional increase: "
            f"{results['peak_fraction']:.8f}\n"
        )
        f.write(
            f"Peak percentage increase: "
            f"{results['peak_percent']:.4f}%\n"
        )

    return result_file


def save_dm_results(
    output_dir,
    trial_dms,
    dm_statistics,
    best_dm,
    best_dm_statistic,
    selected_freq,
    best_delays,
):
    """Save trial-DM search results to a text file."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_file = (
        output_dir
        / "trial_dm_search_results.txt"
    )

    with open(
        result_file,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "LoTSS Stokes-I Trial-DM Search Results\n"
        )
        f.write(
            "=====================================\n\n"
        )
        f.write(
            f"Trial DM minimum (pc cm^-3): "
            f"{TRIAL_DM_MIN:.8f}\n"
        )
        f.write(
            f"Trial DM maximum (pc cm^-3): "
            f"{TRIAL_DM_MAX:.8f}\n"
        )
        f.write(
            f"Trial DM step (pc cm^-3): "
            f"{TRIAL_DM_STEP:.8f}\n"
        )
        f.write(
            f"Best trial DM (pc cm^-3): "
            f"{best_dm:.8f}\n"
        )
        f.write(
            f"Best detection statistic: "
            f"{best_dm_statistic:.8f}\n"
        )
        f.write(
            f"Reference frequency (MHz): "
            f"{np.nanmax(selected_freq):.8f}\n"
        )
        f.write(
            f"Minimum delay (s): "
            f"{np.nanmin(best_delays):.8f}\n"
        )
        f.write(
            f"Maximum delay (s): "
            f"{np.nanmax(best_delays):.8f}\n"
        )
        f.write(
            "\nTrial DM, Detection Statistic\n"
        )

        for dm, statistic in zip(
            trial_dms,
            dm_statistics,
        ):
            f.write(
                f"{dm:.8f}, "
                f"{statistic:.8f}\n"
            )

    return result_file


# ===============================================================
# 11. MAIN ANALYSIS
# ===============================================================

def main():
    print()
    print("=" * 70)
    print("LoTSS / LOFAR STOKES-I ANALYSIS")
    print("=" * 70)

    # -----------------------------------------------------------
    # Validate and load FITS data
    # -----------------------------------------------------------

    ds = DynamicSpectrum(FITS_FILE)

    print("\nFITS file:")
    print(ds.filename)

    print("\nData shape:")
    print(ds.data.shape)

    print(
        f"\nFrequency range: "
        f"{ds.freq_min_mhz:.3f} - "
        f"{ds.freq_max_mhz:.3f} MHz"
    )

    print(
        f"Number of frequency channels: "
        f"{ds.n_freq}"
    )

    print(
        f"Number of time samples: "
        f"{ds.n_time}"
    )

    print(
        f"Time resolution: "
        f"{ds.time_resolution:.4f} s"
    )

    # -----------------------------------------------------------
    # Construct light curve
    # -----------------------------------------------------------

    (
        time_mjd,
        time_seconds,
        lightcurve,
        frequency_mask,
        time_mask,
        channel_sigma,
    ) = make_lightcurve(
        ds,
        stokes=STOKES,
        frequency_range_mhz=FREQUENCY_RANGE_MHZ,
        time_range_seconds=TIME_RANGE_SECONDS,
    )

    selected_freq = ds.freq_axis[frequency_mask]

    print(
        f"\nUsable frequency channels: "
        f"{np.sum(frequency_mask)} / "
        f"{ds.n_freq}"
    )

    print(
        f"Used frequency range: "
        f"{selected_freq[0]:.3f} - "
        f"{selected_freq[-1]:.3f} MHz"
    )

    # -----------------------------------------------------------
    # Analyze observed peak
    # -----------------------------------------------------------

    results = analyze_peak(
        time_mjd,
        time_seconds,
        lightcurve,
    )

    print()
    print("=" * 70)
    print("PEAK ANALYSIS")
    print("=" * 70)

    print(
        f"Baseline flux       : "
        f"{results['baseline_jy']:.6f} Jy"
    )

    print(
        f"Peak flux           : "
        f"{results['peak_flux_jy']:.6f} Jy"
    )

    print(
        f"Peak signal         : "
        f"{results['peak_signal_jy']:.6f} Jy"
    )

    print(
        f"Peak time           : "
        f"{results['peak_time_mjd']:.8f} MJD"
    )

    print(
        f"Time from start     : "
        f"{results['peak_time_seconds']:.4f} s"
    )

    print(
        f"Noise (1 sigma)     : "
        f"{results['noise_sigma_jy']:.6f} Jy"
    )

    print(
        f"RMS                 : "
        f"{results['rms_jy']:.6f} Jy"
    )

    print(
        f"Peak S/N            : "
        f"{results['peak_snr']:.2f} sigma"
    )

    print(
        f"Peak increase       : "
        f"{results['peak_percent']:.2f} %"
    )

    print("=" * 70)

    # -----------------------------------------------------------
    # Make observed-data plots
    # -----------------------------------------------------------

    fig1, ax1 = plot_dynamic_spectrum(
        ds,
        stokes=STOKES,
        frequency_mask=frequency_mask,
        cmap="inferno",
    )

    fig2, ax2 = plot_peak_analysis(results)
    fig3, ax3 = plot_normalized_signal(results)
    fig4, ax4 = plot_snr(results)

    # -----------------------------------------------------------
    # Trial-DM search
    # -----------------------------------------------------------

    dynamic_spectrum = ds.get_stokes(STOKES)

    # Apply the same frequency quality-control mask used by the
    # observed light-curve analysis.
    dynamic_spectrum_selected = (
        dynamic_spectrum[frequency_mask, :]
    )

    frequency_selected = ds.freq_axis[frequency_mask]

    (
        trial_dms,
        dm_statistics,
        best_dm,
        best_dm_statistic,
        best_dynamic,
        best_delays,
    ) = search_trial_dms(
        dynamic_spectrum_selected,
        frequency_selected,
        ds.time_seconds,
        TRIAL_DM_MIN,
        TRIAL_DM_MAX,
        TRIAL_DM_STEP,
    )

    fig5, ax5 = plot_dm_search(
        trial_dms,
        dm_statistics,
        best_dm,
    )

    fig6, ax6 = plot_dedispersed_dynamic_spectrum(
        best_dynamic,
        frequency_selected,
        ds.time_seconds,
        best_dm,
        cmap="inferno",
    )

    # -----------------------------------------------------------
    # Optional catalogue lookup
    # -----------------------------------------------------------

    if CATALOG_FILE.exists():
        try:
            catalog = Table.read(CATALOG_FILE)

            idx, row, tic_name, row_info = get_catalog_row(
                TIC_ID,
                catalog,
            )

            print()
            print("=" * 70)
            print("CATALOGUE LOOKUP")
            print("=" * 70)
            print(f"Catalogue index: {idx}")
            print(f"TIC name: {tic_name}")
            print(f"Catalogue row: {row_info}")
            print("=" * 70)

        except Exception as exc:
            warn(
                f"Catalogue lookup skipped because of an error: {exc}"
            )
    else:
        warn(
            f"Catalogue file not found; skipping catalogue lookup:\n"
            f"{CATALOG_FILE}"
        )

    # -----------------------------------------------------------
    # Save output
    # -----------------------------------------------------------

    if SAVE_RESULTS:
        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig1.savefig(
            OUTPUT_DIR / "01_stokes_i_dynamic_spectrum.png",
            dpi=300,
            bbox_inches="tight",
        )

        fig2.savefig(
            OUTPUT_DIR / "02_stokes_i_peak_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )

        fig3.savefig(
            OUTPUT_DIR / "03_stokes_i_normalized_signal.png",
            dpi=300,
            bbox_inches="tight",
        )

        fig4.savefig(
            OUTPUT_DIR / "04_stokes_i_snr.png",
            dpi=300,
            bbox_inches="tight",
        )

        fig5.savefig(
            OUTPUT_DIR / "05_trial_dm_search.png",
            dpi=300,
            bbox_inches="tight",
        )

        fig6.savefig(
            OUTPUT_DIR / "06_stokes_i_dedispersed_dynamic_spectrum.png",
            dpi=300,
            bbox_inches="tight",
        )

        peak_results_file = save_peak_results(
            results,
            OUTPUT_DIR,
        )

        dm_results_file = save_dm_results(
            OUTPUT_DIR,
            trial_dms,
            dm_statistics,
            best_dm,
            best_dm_statistic,
            frequency_selected,
            best_delays,
        )

        print("\nResults saved to:")
        print(OUTPUT_DIR)

        print("\nPeak-analysis results:")
        print(peak_results_file)

        print("\nTrial-DM results:")
        print(dm_results_file)

        print("\nBest dedispersion plot:")
        print(
            OUTPUT_DIR
            / "06_stokes_i_dedispersed_dynamic_spectrum.png"
        )

    plt.show()


# ===============================================================
# 12. RUN
# ===============================================================

if __name__ == "__main__":
    main()
    