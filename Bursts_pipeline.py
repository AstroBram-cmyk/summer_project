
import json
import numpy as np
from scipy import ndimage
from warnings import warn


# ===========================================================================
# Step 1: burst detection
# ===========================================================================
def detect_bursts(ds, stokes="i", sigma_threshold=5.0,
                   min_pixels=15, freq_axis=None, time_axis=None):
    """
    Detects candidate bursts in a DynamicSpectrum instance and classifies
    each as Type II, Type III or unclassified.

    Parameters
    ----------
    ds : DynamicSpectrum
        An already-constructed DynamicSpectrum instance (ds = DynamicSpectrum(fn)).
    stokes : str
        Which Stokes parameter to search, defaults to "i".
    sigma_threshold : float
        How many std devs above the per-channel median a pixel must be to
        be flagged as part of a burst. Higher = fewer, more confident
        detections; lower = more detections but more false positives (RFI).
    min_pixels : int
        Minimum connected-pixel count for a candidate to be kept -- filters
        out single-pixel noise spikes / RFI blips.

    Returns
    -------
    list of dict
        One dict per detected burst, with keys:
        'burst_id', 'type', 'drift_rate_MHz_per_s', 'duration_s',
        'freq_min_MHz', 'freq_max_MHz', 't_start_mjd', 't_end_mjd',
        'peak_snr', 'n_pixels'
    """
    data = ds._get_stokes(stokes)  # shape: [n_freq, n_time]

    # --- Step 1a: per-channel background + noise estimate ---
    # median/std per frequency channel, across the whole time axis -- this
    # accounts for the fact that different frequencies have different
    # baseline noise levels (bandpass shape, RFI environment, etc.)
    bg_median = np.nanmedian(data, axis=1, keepdims=True)
    bg_std = np.nanstd(data, axis=1, keepdims=True)
    bg_std[bg_std == 0] = np.nan  # avoid divide-by-zero on dead channels

    snr = (data - bg_median) / bg_std

    # --- Step 1b: threshold mask ---
    mask = snr >= sigma_threshold

    # --- Step 1c: group connected pixels into candidate bursts ---
    # a burst spans multiple time AND frequency bins, so we look for
    # 8-connected blobs (diagonal connectivity included, since a drifting
    # burst moves diagonally through freq-time space)
    structure = np.ones((3, 3))
    labeled, n_features = ndimage.label(mask, structure=structure)

    bursts = []
    for label_id in range(1, n_features + 1):
        freq_inds, time_inds = np.where(labeled == label_id)

        if len(freq_inds) < min_pixels:
            continue  # too small -- likely noise/RFI, not a real burst

        # --- Step 1d: fit a drift rate (freq vs time) to this blob ---
        # for each time sample the burst occupies, find the peak-SNR
        # frequency channel -- this traces the burst's frequency track
        # over time, which is what we fit a line to
        t_unique = np.unique(time_inds)
        peak_freqs = []
        for t in t_unique:
            f_at_t = freq_inds[time_inds == t]
            snr_at_t = snr[f_at_t, t]
            peak_freqs.append(f_at_t[np.argmax(snr_at_t)])
        peak_freqs = np.array(peak_freqs)

        t_sec = (ds.time_axis[t_unique] - ds.time_axis[t_unique[0]]) * 86400.0
        f_mhz = ds.freq_axis[peak_freqs]

        if len(t_unique) >= 2:
            # linear fit: frequency (MHz) vs time (s) -- slope is the
            # drift rate in MHz/s (negative = drifting from high to low
            # frequency, the typical direction for both burst types)
            drift_rate, _ = np.polyfit(t_sec, f_mhz, 1)
        else:
            drift_rate = np.nan

        duration_s = (ds.time_axis[time_inds.max()] - ds.time_axis[time_inds.min()]) * 86400.0
        freq_min = ds.freq_axis[freq_inds.min()]
        freq_max = ds.freq_axis[freq_inds.max()]
        peak_snr = float(np.nanmax(snr[freq_inds, time_inds]))

        # --- Step 1e: classify by drift rate + duration ---
        burst_type = classify_burst(drift_rate, duration_s)

        bursts.append({
            "burst_id": int(label_id),
            "type": burst_type,
            "drift_rate_MHz_per_s": float(drift_rate),
            "duration_s": float(duration_s),
            "freq_min_MHz": float(freq_min),
            "freq_max_MHz": float(freq_max),
            "t_start_mjd": float(ds.time_axis[time_inds.min()]),
            "t_end_mjd": float(ds.time_axis[time_inds.max()]),
            "peak_snr": peak_snr,
            "n_pixels": int(len(freq_inds)),
        })

    return bursts


def classify_burst(drift_rate, duration_s):
    """
    Classifies a burst as Type II, Type III, or unclassified, based on
    its fitted drift rate and duration.

    Rough literature-typical ranges (meter-wavelength / LOFAR-band bursts):
      Type III: |drift rate| roughly 5-100+ MHz/s, duration ~ a few seconds
                to a few tens of seconds. Fast, short, broadband.
      Type II:  |drift rate| roughly 0.05-1 MHz/s, duration ~ tens of
                seconds to several minutes. Slow, long-lived.

    These are STARTING POINTS, not fixed physical constants -- tune them
    against real, confirmed bursts in your own data before trusting the
    classification blindly.
    """
    if np.isnan(drift_rate):
        return "unclassified"

    abs_rate = abs(drift_rate)

    if abs_rate >= 5.0 and duration_s <= 60:
        return "Type III"
    elif 0.02 <= abs_rate <= 1.0 and duration_s >= 20:
        return "Type II"
    else:
        return "unclassified"


# ===========================================================================
# Step 2: burst recovery -- dedisperse just this burst's region
# ===========================================================================
def recover_burst(ds, burst, stokes="i", pad_freq_channels=5, pad_time_bins=10):
    """
    "Recovers" a clean, non-drifting time profile for one detected burst by
    dedispersing just its local region of the dynamic spectrum, using the
    burst's own fitted drift rate.

    This re-uses ds.dedisperse(), but converts the fitted linear drift rate
    (MHz/s) into the (a, alpha) power-law parameters that function expects.
    For a burst whose drift is close to linear over its own short duration,
    alpha=2 (a common default in the literature for kHz/s-scale fits) with
    a solved so the model's slope matches the fitted drift rate is a
    reasonable local approximation -- NOT a substitute for a proper
    physically-motivated dispersion fit if you need publication-grade
    parameters.

    Parameters
    ----------
    ds : DynamicSpectrum
    burst : dict
        One entry from detect_bursts()'s output.
    pad_freq_channels, pad_time_bins : int
        Extra margin (in pixels) around the burst's bounding box, so the
        recovered profile includes a bit of quiet baseline on each side.

    Returns
    -------
    dict with:
        'time_profile'  : 1D array, the dedispersed, frequency-averaged
                           flux vs time for this burst's region -- this is
                           the "recovered" burst signal.
        'time_axis_mjd' : matching time axis for the profile
        'freq_range_MHz': (freq_min, freq_max) used for the recovery
    """
    data = ds._get_stokes(stokes)

    # find the pixel indices of this burst's bounding box in freq/time
    freq_idx_min = int(np.searchsorted(ds.freq_axis, burst["freq_min_MHz"])) - pad_freq_channels
    freq_idx_max = int(np.searchsorted(ds.freq_axis, burst["freq_max_MHz"])) + pad_freq_channels
    time_idx_min = int(np.searchsorted(ds.time_axis, burst["t_start_mjd"])) - pad_time_bins
    time_idx_max = int(np.searchsorted(ds.time_axis, burst["t_end_mjd"])) + pad_time_bins

    freq_idx_min = max(freq_idx_min, 0)
    time_idx_min = max(time_idx_min, 0)
    freq_idx_max = min(freq_idx_max, ds.n_freq)
    time_idx_max = min(time_idx_max, ds.n_time)

    sub_data = data[freq_idx_min:freq_idx_max, time_idx_min:time_idx_max]
    sub_freq_axis = ds.freq_axis[freq_idx_min:freq_idx_max]
    sub_time_axis = ds.time_axis[time_idx_min:time_idx_max]

    if sub_data.shape[0] < 2 or sub_data.shape[1] < 2:
        warn(f"Burst {burst['burst_id']}: region too small to recover.")
        return None

    # convert the fitted linear drift rate (MHz/s) into an approximate
    # (a, alpha) pair for ds.dedisperse()'s power-law model, using alpha=2
    # as a standard default and solving for a from the measured slope at
    # the burst's central frequency
    alpha = 2.0
    nu0 = sub_freq_axis[len(sub_freq_axis) // 2]
    drift_rate = burst["drift_rate_MHz_per_s"]
    if drift_rate == 0 or np.isnan(drift_rate):
        warn(f"Burst {burst['burst_id']}: invalid drift rate, cannot recover.")
        return None

    # d(delta_t)/d(nu) = 1/a * nu^(-alpha)  =>  drift_rate = d(nu)/d(t) = a * nu^alpha
    a = drift_rate / (nu0 ** (1 - alpha)) if alpha != 1 else drift_rate / nu0

    dt = (sub_time_axis[1] - sub_time_axis[0]) * 86400.0
    freqs_flip = np.flip(sub_freq_axis)
    ds_local = np.flip(sub_data.copy(), axis=0)
    exp = 1 - alpha
    nu0_local = freqs_flip[0]

    for j, nu in enumerate(freqs_flip):
        deltat = 1 / (a * (alpha - 1)) * (nu ** exp - nu0_local ** exp)
        dn = int(deltat / dt)
        ds_local[j] = np.roll(ds_local[j], -dn)
        if dn != 0:
            ds_local[j, -dn:] = 0

    ds_local = np.flip(ds_local, axis=0)

    time_profile = np.nanmean(ds_local, axis=0)

    return {
        "time_profile": time_profile,
        "time_axis_mjd": sub_time_axis,
        "freq_range_MHz": (float(sub_freq_axis[0]), float(sub_freq_axis[-1])),
    }


# ===========================================================================
# Step 3: save / load the burst catalog
# ===========================================================================
def save_burst_catalog(bursts, out_path):
    """Writes the list of detected bursts (from detect_bursts) to a JSON file."""
    with open(out_path, "w") as f:
        json.dump(bursts, f, indent=2)


def load_burst_catalog(in_path):
    """Reloads a previously saved burst catalog from JSON."""
    with open(in_path) as f:
        return json.load(f)
