
"""
@Astro Bramuel
"""

# %% Cell 0: imports ---------------------------------------------------------
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
import lightkurve as lk
import pandas as pd

# %% Cell 1: load the Gaia/LoTSS catalog ------------------------------------
env1 = "C:/Users/ADMIN/Downloads/gaia_info_100pc_lc_info_lotss.csv"  
tab = Table.read(env1)
ids = tab["ID"]


# %% Cell 2: helper to pull a TESS light curve for a row of the catalog -----
def get_lightcurve(idx: int, table=tab):
    """Download all available SPOC 2-min light curves for catalog row idx."""
    try:
        tic_id = table["ID"][idx]
        tic_name = f"TIC {tic_id}"
    except Exception as e:
        print(e)
        return None

    l = lk.search_lightcurve(tic_name, author="SPOC", exptime=120)
    if len(l) > 0:
        lcs = l.download_all()
    else:
        return None
    return lcs


# %% Cell 3: load the local catalog + dynamic-spectrum FITS cube ------------
env2 = "C:/Users/ADMIN/OneDrive/Desktop/python_codes/"
fn_cat = f"{env2}Catalog.npy"
fn_fits = f"{env2}L2023107_21_48_11.743_+79_13_02.884.fits"

data_cat = np.load(fn_cat, allow_pickle=True)


# %% Cell 4: quick look at the Stokes I plane --------------------------------
with fits.open(fn_fits) as hdul:          # FIX: use a context manager so the file handle is closed properly
    data_fits = hdul[0]
    stokes_i = data_fits.data[0, :, :]


# %% Cell 5: the DynamicSpectrum class ---------------------------------------
# any time you update the class, you need to re-initialize the class instance
# class instance is when ds = DynamicSpectrum(fn)

class DynamicSpectrum:
    """
    Wraps a LOFAR-style dynamic-spectrum FITS cube of shape
    (n_stokes, n_freq, n_time) plus its header metadata, and gives you
    plotting + simple statistics helpers.
    """

    def __init__(self, fn: str):
        self.fn = fn
        f = fits.open(self.fn)
        self.data = f[0].data
        self.hdr = f[0].header
        f.close()

        self.start_time_isot = self.hdr["OBS-STAR"]
        self.end_time_isot = self.hdr["OBS-STOP"]
        self.freq_max = self.hdr["FRQ-MAX"]
        self.freq_min = self.hdr["FRQ-MIN"]

        self.start_time_mjd = Time(self.start_time_isot, format="isot").mjd
        self.end_time_mjd = Time(self.end_time_isot, format="isot").mjd

        # axes derived from the data shape, used by plot_lightcurve/spectrum
        self.n_stokes, self.n_freq, self.n_time = self.data.shape
        self.freq_axis = np.linspace(self.freq_min, self.freq_max, self.n_freq)
        self.time_axis = np.linspace(
            self.start_time_mjd, self.end_time_mjd, self.n_time
        )

    def _get_stokes(self, stokes):
        if isinstance(stokes, str):
            stokes_str = stokes.lower()
            assert stokes_str in ["i", "q", "u", "v"], "stokes must be i/q/u/v"
            stokes = {"i": 0, "q": 1, "u": 2, "v": 3}[stokes_str]
        data = self.data[stokes, :, :]
        return data

    def plot_dyn_spec(self, stokes="i"):
        data = self._get_stokes(stokes)
        fig, ax = plt.subplots()
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=[
                self.time_axis[0],
                self.time_axis[-1],
                self.freq_axis[0],
                self.freq_axis[-1],
            ],
        )
        fig.colorbar(im, ax=ax, label=f"Stokes {stokes.upper()}")
        ax.set_xlabel("Time (MJD)")
        ax.set_ylabel("Frequency (MHz)")
        ax.set_title(f"Dynamic spectrum - Stokes {stokes.upper()}")
        return fig, ax

    def get_statistics(self, time_idx=None, freq_idx=None, stokes="i"):
        data = self._get_stokes(stokes)
        if time_idx is None and freq_idx is None:
            sub = data
        elif time_idx is not None and freq_idx is None:
            sub = data[:, time_idx]
        elif freq_idx is not None and time_idx is None:
            sub = data[freq_idx, :]
        else:
            sub = data[freq_idx, time_idx]

        std = np.nanstd(sub)
        maxv = np.nanmax(sub)
        minv = np.nanmin(sub)
        med = np.nanmedian(sub)
        return std, maxv, minv, med
    def get_tic_name(self) ->str:
        lc = self.get_tess_lightcurve()
        name = lc[0].meta["TARGETID"]
        self.tic_name = name
        
    def plot_lightcurve(self, freq_idx=None, freq_range=None, stokes="i"):
        """
        Plot flux vs. time.
        - freq_idx: plot a single frequency channel
        - freq_range: (i0, i1) tuple, average channels i0:i1
        - if neither given, average over the whole band
        """
        data = self._get_stokes(stokes)

        if freq_idx is not None:
            lc = data[freq_idx, :]
            label = f"channel {freq_idx} ({self.freq_axis[freq_idx]:.1f} MHz)"
        elif freq_range is not None:
            i0, i1 = freq_range
            lc = np.nanmean(data[i0:i1, :], axis=0)
            label = (
                f"{self.freq_axis[i0]:.1f}-{self.freq_axis[i1 - 1]:.1f} MHz averaged"
            )
        else:
            lc = np.nanmean(data, axis=0)
            label = "full band averaged"

        fig, ax = plt.subplots()
        ax.plot(self.time_axis, lc)
        ax.set_xlabel("Time (MJD)")
        ax.set_ylabel(f"Stokes {stokes.upper()} flux")
        ax.set_title(f"Light curve - {label}")
        return fig, ax

    def plot_spectrum(self, time_idx=None, time_range=None, stokes="i"):
        """
        Plot flux vs. frequency.
        - time_idx: plot a single time sample
        - time_range: (i0, i1) tuple, average samples i0:i1
        - if neither given, average over the whole time axis
        """
        data = self._get_stokes(stokes)

        if time_idx is not None:
            spec = data[:, time_idx]
            label = f"t = {self.time_axis[time_idx]:.5f} MJD"
        elif time_range is not None:
            i0, i1 = time_range
            spec = np.nanmean(data[:, i0:i1], axis=1)
            label = f"t = {self.time_axis[i0]:.5f}-{self.time_axis[i1 - 1]:.5f} MJD averaged"
        else:
            spec = np.nanmean(data, axis=1)
            label = "full duration averaged"

        fig, ax = plt.subplots()
        ax.plot(self.freq_axis, spec)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel(f"Stokes {stokes.upper()} flux")
        ax.set_title(f"Spectrum - {label}")
        return fig, ax

    def get_tess_lightcurve(self):
        name = None
        try:
           name = self.hdr['NAME']
        except:
            pass
        if name is None:
            print("No source-name keyword found in header; inspect self.hdr.")
            return None

        search = lk.search_lightcurve(name, author="SPOC", exptime=120)
        if len(search) == 0:
            print(f"No TESS SPOC light curves found for {name}.")
            return None

        lcs = search.download_all()
        self.lcs = lcs
        return lcs


# %% Cell 6: example usage ----------------------------------------------------
if __name__ == "__main__":
    ds = DynamicSpectrum(fn_fits)

    fig1, ax1 = ds.plot_dyn_spec(stokes="i")
    fig2, ax2 = ds.plot_lightcurve(stokes="i")          # full-band light curve
    fig3, ax3 = ds.plot_spectrum(stokes="i")            # time-averaged spectrum

    print(ds.get_statistics(stokes="i"))

    plt.show()
    #%%

def get_catalog_row(tic_value, table=tab):
    """
    Find a catalog row using a TIC ID.

    Parameters
    ----------
    tic_value : int
        TIC ID (e.g. 470085072)
    table : astropy.table.Table
        Catalog table.

    Returns
    -------
    idx : int
        Index of the matching row.
    row : astropy.table.Row
        Complete row containing all catalog information.
    """
    matches = np.where(table["ID"] == tic_value)[0]

    if len(matches) == 0:
        raise ValueError(f"TIC {tic_value} not found in the catalog.")

    idx = matches[0]
    row = table[idx]

    return idx, row