"""
flare_pipeline.py

eThis code measures stellar activity from TESS light curves using Ivey's flare-finding
algorithm (Star / Flares classes). For each star in the catalog, this:

    1. looks up the star's info (Teff, radius, etc.) from the catalog,
    2. builds a Star object (downloads its TESS light curves),
    3. runs Ivey's FindAllFlares() on it,
    4. saves the star + flare info to disk (flare_table + JSON metadata),
    5. records a summary (n_flares, flare_rate, peak_flare_energy) back
       into new columns on the catalog table.

I have included Ivey's Star/Flares code  below largely unmodified -- only the
radius-fetching + connector logic at the bottom is new.
"""

from warnings import warn
import os
import json
import glob
import shutil

import numpy as np
import pandas as pd
import lightkurve as lk
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.time import Time
from astropy import units as un, constants as const
from scipy.signal import medfilt
from scipy.integrate import quad

# shared catalog + lookup function -- see catalog_utils.py
from catalog_utils import tab, get_catalog_row


# ---------------------------------------------------------------------------
# lightkurve cache directory (needed by Star.clear_cache)
# ---------------------------------------------------------------------------
cache_dir = None
try:
    cache_dir = lk.config.get_cache_dir()
except Exception:
    conf = lk.config.get_config_dir()
    cache_dir = f'{conf.rstrip("/config")}-cache'
    warn(f"Could not get the cache directory. Assuming legacy location {cache_dir}")


# ===========================================================================
# Star class (Ivey) -- downloads + holds TESS light curves for one star
# ===========================================================================
class Star:
    """
    Class for holding information about a star observed by TESS
    """
    def __init__(self, id_number: int = None, star_name: str = None,
                 radius: 'Solar radii' = 1, temperature: 'Kelvin' = 5700,
                 lcs: lk.LightCurveCollection = None, period=None,
                 sectors=range(14), exp_time: 'sec' = 120, clear_cache=True,
                 cache=cache_dir, instrument="TESS"):
        allowed_instruments = ["tess", "kepler"]
        assert(instrument.lower() in allowed_instruments), f"Allowable instruments are {allowed_instruments}"
        assert(not (id_number is None and star_name is None)), "Specify either name or KIC/TIC number of star"
        assert(not (id_number is not None and star_name is not None)), "Specify either name or KIC/TIC number of star, not both"

        assert(float(radius))
        assert(float(temperature))
        if period is not None:
            assert(float(period))

        self.id_number = id_number
        self.radius = radius
        self.temperature = temperature
        self.cache_dir = cache
        self.instrument = instrument

        # --- download light curves from TESS/Kepler if not supplied ---
        if lcs is None:
            if instrument.lower() == 'tess':
                if id_number is not None:
                    lcs = lk.search_lightcurve(f'TIC {id_number}', exptime=exp_time, author='SPOC', sector=sectors).download_all()
                elif star_name is not None:
                    lcs = lk.search_lightcurve(star_name, exptime=exp_time, author='SPOC', sector=sectors).download_all()

            if instrument.lower() == 'kepler':
                if id_number is not None:
                    lcs = lk.search_lightcurve(f'KIC {id_number}', exptime=exp_time, quarter=sectors).download_all()
                elif star_name is not None:
                    lcs = lk.search_lightcurve(star_name, exptime=exp_time, author='K2SFF', quarter=sectors).download_all()

            if self.id_number is not None:
                lc = []
                for l in lcs:
                    if int(l.meta['LABEL'].split(' ')[-1]) == id_number:
                        lc.append(l.normalize())
                self.lcs = lk.LightCurveCollection(lc)

            elif self.id_number is None:
                self.lcs = lcs
                self.id_number = int(lcs[0].meta['LABEL'].split(' ')[-1])

        if lcs is not None:
            self.lcs = lcs
            try:
                self.lcs[0].flux.mask
            except Exception:
                for i in range(len(lcs)):
                    l = lcs[i]
                    mask = np.isnan(l.flux)
                    l.flux = lk.LightCurve.MaskedColumn(data=l.flux, name='flux', mask=mask)

            exp_time = np.nanmedian((lcs[0].time - np.roll(lcs[0].time, 1)).to('s'))

            # --- estimate rotation period via BLS if not supplied ---
            if period is None:
                periods = []
                mps = []
                for lc in self.lcs:
                    p = lc.to_periodogram('bls')
                    periods.append(p.period_at_max_power.to('day').value)
                    mps.append(p.max_power.value)
                mp = np.nanmean(mps)
                if mp >= 1e3:
                    period = np.average(periods)
                elif mp < 1e3:
                    period = (np.nanmean([len(lc) for lc in lcs]) * exp_time.to('d')).value
                self.period = period

            elif period is not None:
                assert(float(period))
                self.period = period

            if clear_cache:
                self.clear_cache()
            return

    def clear_cache(self):
        for lc in self.lcs:
            fn = lc.meta['FILENAME']
            path = os.path.dirname(fn)
            if os.path.isdir(path):
                try:
                    assert(self.cache_dir in fn), f"File {fn} is not in the cache directory {cache_dir}."
                    shutil.rmtree(path)
                except Exception:
                    warn(f"Could not remove directory {fn}")


# ===========================================================================
# Flares class (Ivey) -- detects + characterizes flares in a Star's light curve
# ===========================================================================
class Flares:
    """
    Class for flagging flares from a star observed by TESS and storing that data
    """
    def __init__(self, star: Star, process: bool = True):
        self.star = star
        self.int_time = np.nanmedian((star.lcs[0].time - np.roll(star.lcs[0].time, 1)).to('min'))

        if process:
            # remove flagged (bad) points, stitch sectors into one curve
            lc = star.lcs.stitch()
            keep_inds = np.where(lc.flux.mask == False)[0]
            self.lc = lc[keep_inds]

            # smoothing window = 1/15th of rotation period, min 100 min
            window_time = (self.star.period * 1440 / 15)
            if window_time < 100:
                window_time = 100

            window = int((window_time / self.int_time.to('min').value))
            if window % 2 == 0:
                window -= 1

            self.window = window
            self.eclipse_window = 2 * window - 1

        elif not process:
            self.window = None
            self.eclipse_window = None
            self.lc_arr = None
            self.lc = None

        self.lc_arr = None
        self.lc_flagged = None
        self.lc_median = None
        self.flares = None
        self.lc_norm = None
        self.std = None
        self.flare_table = None
        self.flare_rate = None
        self._n_pts = 0
        return

    def SplitLightCurve(self, min_gap=None):
        """Splits the light curve into sections wherever there's a large time gap."""
        lcs = []
        s = pd.Series(self.lc.time.value)
        d = s.diff()
        if min_gap is None:
            min_gap = np.nanmedian(d) * 15 * un.day

        inds = np.where(d > min_gap.to('d').value)[0]

        i0 = 0
        for i in range(len(inds)):
            l = self.lc[i0:inds[i] - 1]
            i0 = inds[i]
            if len(l) > self.window:
                lcs.append(l)

        if len(self.lc[i0:]) > self.window:
            lcs.append(self.lc[i0:])

        self.lc_arr = lcs

    def __SplitFlares(self, lc, min_gap=6 * un.min):
        """Splits a light curve of flare candidate points into individual flare events."""
        lcs = []
        s = pd.Series(lc.time.value)
        d = s.diff()
        inds = np.where(d > min_gap.to('d').value)[0]

        i0 = 0
        for i in range(len(inds)):
            l = lc[i0:inds[i] - 1]
            i0 = inds[i]
            lcs.append(l)

        lcs.append(lc[i0:])
        return lcs

    def __FlagPoints(self, lc: lk.LightCurve, max_iter: int = 20):
        """Iteratively flags points >3 sigma from the median-filtered baseline."""
        lc_new = lk.LightCurve(lc)
        flux = lc_new.flux

        flag_inds = [0]
        max_iter_count = 0

        while len(flag_inds) != 0 and max_iter_count <= max_iter:
            lc_medfilt = medfilt(flux, self.window)
            lc_norm = flux / lc_medfilt

            sig = np.nanstd(lc_norm)
            med = np.nanmedian(lc_norm)
            flag_inds = np.where(lc_norm >= med + 3 * sig)[0]

            lc_medfilt = medfilt(flux, self.eclipse_window)
            lc_norm = flux / lc_medfilt

            sig = np.std(lc_norm)
            med = np.median(lc_norm)
            eclipse_inds = np.where(lc_norm <= med - 3 * sig)[0]
            flag_inds = np.concatenate((flag_inds, eclipse_inds))

            if len(flag_inds) > 0:
                new_mask = flux.mask
                new_mask[flag_inds] = True
                flux.mask = new_mask

            max_iter_count += 1

        lc_new.flux = flux
        return lc_new

    def FlagLightCurves(self, max_iter: int = 20):
        """Flags outliers in every section of lc_arr; builds detrended/normalized curves."""
        if self.lc_arr is None:
            warn("The light curve has not been split into sections yet. They will now be split with the default gap of 30 min.")
            self.SplitLightCurve()

        flagged_lcs = []
        lc_medfilts = []
        norm_lc = []
        medians = []

        for l in self.lc_arr:
            flagged_lc = self.__FlagPoints(l, max_iter)
            flagged_lcs.append(flagged_lc)

            med_lc = medfilt(flagged_lc.flux, self.window)
            lc_medfilts.append(med_lc)

            norm_lc.append(l / med_lc)
            medians.append(np.nanmedian(flagged_lc.flux / med_lc))

        self.median = np.nanmedian(medians)
        self.lc_flagged = flagged_lcs
        self.lc_median = lc_medfilts
        self.lc_norm = norm_lc
        self.CalculateSTD()

    def __FindFlares(self, lc_raw: lk.LightCurve, lc_flagged: lk.LightCurve,
                      clip_size: int = None, sigma_max_threshold=3, sigma_min_threshold=2,
                      n_points_max=3, n_points_min=3, min_gap=6 * un.min):
        """Identifies individual flare events within one light-curve section."""
        assert(float(sigma_min_threshold))
        assert(float(sigma_max_threshold))

        if clip_size is None:
            if self.window >= 209:
                clip_size = int(self.window / 3)
            else:
                clip_size = 70

        lc_raw_copy = lk.LightCurve(lc_raw)
        lc_flagged_copy = lk.LightCurve(lc_flagged)

        lc_medfilt = medfilt(lc_flagged_copy.flux, self.window)
        norm_flux = (lc_raw_copy.flux / lc_medfilt)[clip_size:-clip_size]

        lc_raw_copy = lk.LightCurve(lc_raw)[clip_size:-clip_size]
        lc_raw_copy.flux = norm_flux

        norm_flux_flagged = (lc_flagged_copy.flux / lc_medfilt)[clip_size:-clip_size]
        sig = np.nanstd(norm_flux_flagged)
        med = np.nanmedian(norm_flux_flagged)

        flag_inds = np.where(lc_raw_copy.flux >= med + sigma_min_threshold * sig)[0]
        flare_candidates = lc_raw_copy[flag_inds]

        flare_lcs = self.__SplitFlares(flare_candidates, min_gap)
        final_flare_lcs = []

        if len(flare_lcs) > 0:
            for i in flare_lcs:
                if len(i) >= n_points_min:
                    max_sig_inds = np.where(i.flux >= med + sigma_max_threshold * sig)[0]
                    mean_sig_inds = np.where(i.flux >= med + (sigma_max_threshold + sigma_min_threshold) * sig / 2)[0]

                    if len(max_sig_inds) >= n_points_max and len(mean_sig_inds) - n_points_max >= n_points_max / 2:
                        t0_ind = np.where(lc_raw_copy.time == i.time[0])[0][0]
                        te_ind = np.where(lc_raw_copy.time == i.time[-1])[0][0]
                        final_flare_lcs.append(lk.LightCurve(lc_raw_copy[t0_ind:te_ind + 1]))
        n_pts = len(norm_flux)
        return final_flare_lcs, n_pts

    def FindAllFlares(self, clip_size: int = None, sigma_max_threshold=3, sigma_min_threshold=2,
                       n_points_max=3, n_points_min=3, min_gap=6 * un.min, T_flare=10_000 * un.K):
        """Runs flare-finding on every section, builds flare_table + flare_rate."""
        if self.lc_flagged is None:
            warn("FlagLightCurves hasn't be run yet. Running it now with default values.")
            self.FlagLightCurves()

        if clip_size is None:
            if self.window >= 209:
                clip_size = int(self.window / 3)
            else:
                clip_size = 70
        all_flares = []
        n_pts_tot = 0

        for i in range(len(self.lc_arr)):
            lc_orig = self.lc_arr[i]
            lc_flagged = self.lc_flagged[i]
            flares, n_pts = self.__FindFlares(lc_orig, lc_flagged, clip_size=clip_size,
                                               sigma_max_threshold=sigma_max_threshold,
                                               sigma_min_threshold=sigma_min_threshold,
                                               n_points_max=n_points_max, n_points_min=n_points_min,
                                               min_gap=min_gap)
            n_pts_tot += n_pts
            if flares is not None:
                for f in flares:
                    all_flares.append(f)

        self.flares = all_flares
        self._n_pts = n_pts_tot
        self.MakeFlareTable()
        self.CalculateFlareRate()
        return

    def FlagFlares(self, bad_flare_list: list):
        """Manually flags misidentified flares as invalid."""
        if self.flares == None:
            raise Warning("There are no flares in the flare property. There are either no flares for this star, or you need to run FindAllFlares")
        for index in bad_flare_list:
            self.flare_table[index]['flag'] = True
            self.flares[index].flux.mask[:] = True
        self.CalculateFlareRate()
        return

    def UnflagFlares(self, good_flare_list: list):
        """Reverses FlagFlares for the given indices."""
        if self.flares == None:
            raise Warning("There are no flares in the flare property. There are either no flares for this star, or you need to run FindAllFlares")
        for index in good_flare_list:
            self.flare_table[index]['flag'] = False
            self.flares[index].flux.mask[:] = False
        self.CalculateFlareRate()
        return

    def MakeFlareTable(self, T_flare=10_000 * un.K):
        """Builds an Astropy Table with start time, duration, energy, peak luminosity, flag."""
        integration_time = self.int_time
        if self.flares == None:
            self.FindAllFlares()

        flare_table = Table(names=['flare_start', 'duration', 'energy', 'max_L', 'flag'],
                             dtype=[float, float, float, float, bool])

        for f in self.flares:
            flare_start = f.time[0].value
            flare_duration = (f.time[-1] - f.time[0]).to('s')

            L = self.CalculateFlareLuminosity(f.flux, T_flare)
            energy = (L * integration_time).to('erg')
            E = np.nansum(energy)
            max_luminosity = np.max(L)
            flare_table.add_row([flare_start, flare_duration, E, max_luminosity, False])

        self.flare_table = flare_table
        self.CalculateFlareRate()
        return

    def ConsolidateComplexFlares(self, max_time_spacing=2.4 * un.hr):
        """Merges flares occurring close together in time into single complex events."""
        if self.flare_table == None:
            raise Warning("There is no flare table. Run FindAllFlares")

        if len(self.flare_table) > 1:
            flare_tab = self.flare_table
            flare_tab.sort('flare_start')

            flare_starts = Time(flare_tab['flare_start'], format='btjd')
            flare_ends = flare_starts + flare_tab['duration'] * un.s

            flares = self.flares
            consolidated_flares = [flares[0]]
            start_ind = 0
            end_ind = start_ind + 1

            while end_ind <= len(flares) - 1:
                dt = flare_ends[end_ind] - flare_starts[start_ind]

                if dt <= max_time_spacing:
                    consolidated_flares[-1] = consolidated_flares[-1].append(flares[end_ind])
                    start_ind += 1
                    end_ind += 1
                elif dt > max_time_spacing:
                    consolidated_flares.append(flares[end_ind])
                    start_ind = end_ind
                    end_ind = start_ind + 1
        else:
            print('Not enough flares to consolidate')
            return
        print(f'{len(self.flares)} flares consolidated to {len(consolidated_flares)}')
        self.flares = consolidated_flares
        self.MakeFlareTable()

    def CalculateFlareRate(self):
        """Flares per year = (unflagged flare count) / (total observed time)."""
        duration = self._n_pts * self.int_time
        n_flares = len(np.where(self.flare_table['flag'] == False)[0])
        flare_rate = (n_flares / duration).to('yr**(-1)')
        self.flare_rate = flare_rate
        return

    def PlotLightCurves(self, show_flares: bool = True):
        """Two-panel plot: raw light curve (top), detrended + flares marked (bottom)."""
        fig, ax = plt.subplots(2, 1, sharex=True)
        for l in self.lc_arr:
            l.scatter(ax=ax[0], c='k')
        for l in self.lc_norm:
            l.scatter(ax=ax[1], c='k')
        leg = ax[0].get_legend()
        leg.remove()
        leg = ax[1].get_legend()
        leg.remove()
        y_top1, y_bottom1 = ax[0].get_ybound()
        y_top2, y_bottom2 = ax[1].get_ybound()
        if show_flares == True:
            if self.flare_table is None:
                self.MakeFlareTable()
            for i in range(len(self.flare_table)):
                f = self.flare_table[i]
                if f['flag'] == False:
                    t0 = f['flare_start']
                    ax[0].plot(np.linspace(t0, t0), np.linspace(y_top1, y_bottom1), c='r', alpha=0.4)
                    ax[1].plot(np.linspace(t0, t0), np.linspace(y_top2, y_bottom2), c='r', alpha=0.4)
                    self.flares[i].scatter(ax=ax[1], c='r')
            leg = ax[1].get_legend()
            try:
                leg.remove()
            except Exception:
                pass
        ax[0].set_ylim(y_top1, y_bottom1)
        ax[1].set_ylim(y_top2, y_bottom2)
        plt.tight_layout()
        return fig, ax

    def CalculateFlareLuminosity(self, flare_lc, T_flare=10_000 * un.K):
        """Bolometric luminosity of a flux point, via blackbody flare-vs-star intensity ratio."""
        c = flare_lc - self.median
        area = 4 * np.pi * (self.star.radius * const.R_sun) ** 2
        prefac = c * area * const.sigma_sb * T_flare ** 4

        I_star = quad(PlanckFunction, 600, 1000, args=self.star.temperature)[0] * un.W / un.m ** 2
        I_fl = quad(PlanckFunction, 600, 1000, args=T_flare.value)[0] * un.W / un.m ** 2
        L_fl = (prefac * I_star / I_fl).to('erg/s')
        return L_fl

    def CalculateFlareEnergy(self, flare_lc: lk.LightCurve(), T_flare=10_000 * un.K):
        """Bolometric energy of an entire flare event (sum over its light curve)."""
        E = 0
        for pt in flare_lc.flux:
            L = self.CalculateFlareLuminosity(pt, T_flare=T_flare)
            E = np.nansum((E + L * self.int_time).to('erg'))
        return E

    def CalculateSTD(self):
        """Std dev of the detrended, flagged light curve (excluding edge effects)."""
        if self.lc_arr == None:
            self.FlagLightCurves()
        std = []
        for i in range(len(self.lc_arr)):
            std.append(np.std(self.lc_flagged[i].flux[self.window:-self.window] / self.lc_median[i][self.window:-self.window]))
        self.std = np.nanmedian(std)
        return self.std

    def __MakeMiscInfoDict__(self):
        """Packages star + flare-analysis summary stats into a plain dict (for JSON)."""
        d = {}
        d.update({'eclipse_window': int(self.eclipse_window)})
        d.update({'flare_rate': float(self.flare_rate.value)})
        d.update({'window': int(self.window)})
        d.update({'std': float(self.std)})
        d.update({'id_number': int(self.star.id_number)})
        d.update({'radius': float(self.star.radius)})
        d.update({'temperature': float(self.star.temperature)})
        d.update({'period': float(self.star.period)})
        d.update({'median': float(self.median)})
        d.update({'n_pts': float(self._n_pts)})
        return d

    def __MakeMetaDataDict__(self):
        """Packages the light curve's metadata into a JSON-safe dict."""
        d = dict(self.lc.meta)
        try:
            d.pop('PDC_VAR')
            d.pop('PDC_VARP')
            d.pop('PDC_EPT')
            d.pop('PDC_EPTP')
            d.pop('QUALITY_MASK')
        except Exception:
            warn("Could not remove json-incompatible dictionary items.")
        return d

    def WriteOutData(self, base_dir: str = None):
        """
        Saves everything to disk under <base_dir>/TIC<id>/ :
          - star_information.json  (n_flares-relevant stats, radius, period, etc.)
          - metadata.json
          - flare_table.tab        <-- the per-star flare table you need saved
          - raw/flagged/normalized light-curve sections
          - each individual flare's light curve
        """
        try:
            if base_dir is None:
                if self.star.instrument.lower() == 'tess':
                    fp = 'TIC' + str(self.star.id_number) + r'/'
                elif self.star.instrument.lower() == 'kepler':
                    fp = 'KIC' + str(self.star.id_number) + r'/'
            elif base_dir is not None:
                if base_dir[-1] != '/':
                    base_dir += r'/'
                if self.star.instrument.lower() == 'tess':
                    fp = base_dir + 'TIC' + str(self.star.id_number) + r'/'
                elif self.star.instrument.lower() == 'kepler':
                    fp = base_dir + 'KIC' + str(self.star.id_number) + r'/'
            os.makedirs(fp, exist_ok=True)
        except Exception:
            raise Exception("Couldn't make directory " + fp)

        d = self.__MakeMiscInfoDict__()
        m = self.__MakeMetaDataDict__()
        with open(fp + 'star_information.json', 'w') as f:
            json.dump(d, f)
        with open(fp + 'metadata.json', 'w') as f:
            json.dump(m, f)

        for i in range(len(self.lc_arr)):
            lc_arr_n = fp + 'raw_lc_section_' + "{:02d}".format(i)
            lc_flag_n = fp + 'flag_lc_section_' + "{:02d}".format(i)
            lc_norm_n = fp + 'norm_lc_section_' + "{:02d}".format(i)
            lc_median_n = fp + 'lc_median_section_' + "{:02d}".format(i)

            LightCurvetoTab(self.lc_arr[i], lc_arr_n)
            LightCurvetoTab(self.lc_flagged[i], lc_flag_n)
            LightCurvetoTab(self.lc_norm[i], lc_norm_n)
            np.savez(lc_median_n, self.lc_median[i], overwrite=True)

        lc_n = fp + 'full_lc'
        LightCurvetoTab(self.lc, lc_n)

        flare_table_fn = fp + 'flare_table.tab'
        self.flare_table.write(flare_table_fn, format='ascii', overwrite=True)

        for i in range(len(self.flares)):
            fn = fp + 'flare_' + "{:02d}".format(i) + ''
            LightCurvetoTab(self.flares[i], fn)
        return


# ===========================================================================
# Free functions (Ivey)
# ===========================================================================
def PlanckFunction(lam, T):
    """Planck function: specific intensity [W/m^2/nm] at wavelength lam [nm], temperature T [K]."""
    lam = lam * un.nm
    T = T * un.K
    h = const.h
    c = const.c
    k = const.k_B
    B_num = 2 * h * c ** 2 / lam ** 5
    B_den = np.exp((h * c / (lam * k * T)).to('')) - 1
    B = (B_num / B_den).to('W/(m**2*nm)')
    return B.value


def TabtoLightCurve(tab_n: str, meta_n=None):
    """Reconstructs a LightCurve object from a saved .tab + _meta.json pair."""
    if meta_n is None:
        meta_n = tab_n.replace('.tab', '_meta.json')

    tab_data = Table.read(tab_n, format='ascii')

    with open(meta_n) as f:
        meta_dat = json.load(f)

    lc = lk.LightCurve(tab_data, meta=meta_dat)
    return lc


def LightCurvetoTab(lc_obj: lk.LightCurve, out_name: str, overwrite=True):
    """Writes a LightCurve object out to an ascii .tab file + a JSON metadata sidecar."""
    acceptable_types = [float, int, bool, str, np.ndarray]

    table = lc_obj.to_table()
    table.write(out_name + ".tab", format='ascii', overwrite=overwrite)

    meta_dict = lc_obj.meta.copy()
    bad_keys = []
    array_keys = []

    for key in meta_dict.keys():
        if type(meta_dict[key]) not in acceptable_types:
            bad_keys.append(key)
        if type(meta_dict[key]) == np.ndarray:
            array_keys.append(key)

    for key in bad_keys:
        meta_dict.__delitem__(key)

    for key in array_keys:
        arr = list(meta_dict[key].astype(list))
        meta_dict[key] = arr

    meta_out_name = out_name + "_meta.json"
    with open(meta_out_name, 'w') as f:
        json.dump(meta_dict, f, indent=4)
    return


def LoadInStar(fp: str = ''):
    """Reconstructs a full Flares (+ Star) object from a directory written by WriteOutData."""
    fp = os.path.join(fp, '')
    lc_arr_fns = glob.glob(fp + 'raw_lc_section_*.tab')
    lc_flag_fns = glob.glob(fp + 'flag_lc_section_*.tab')
    lc_norm_fns = glob.glob(fp + 'norm_lc_section_*.tab')
    lc_median_fns = glob.glob(fp + 'lc_median_section_*.npz')
    flare_fns = glob.glob(fp + 'flare*.tab')

    if f"{fp}flare_table.tab" in flare_fns:
        flare_fns.remove(f"{fp}flare_table.tab")

    flare_fns.sort()
    lc_arr_fns.sort()
    lc_flag_fns.sort()
    lc_norm_fns.sort()
    lc_median_fns.sort()
    if not (len(lc_arr_fns) == len(lc_flag_fns) == len(lc_norm_fns) == len(lc_median_fns)):
        warn("There are missing light curve sections. Cannot load in all data.")

    lc_arr = []
    lc_flag = []
    lc_norm = []
    lc_median = []
    flares = []

    misc = json.load(open(fp + 'star_information.json'))

    for i in range(len(lc_arr_fns)):
        try:
            lc_arr.append(TabtoLightCurve(lc_arr_fns[i]))
        except Exception:
            pass
        try:
            lc_flag.append(TabtoLightCurve(lc_flag_fns[i]))
        except Exception:
            pass
        try:
            lc_norm.append(TabtoLightCurve(lc_norm_fns[i]))
        except Exception:
            pass
        try:
            lc_median.append(np.load(lc_median_fns[i], allow_pickle=True)['arr_0'])
        except Exception:
            pass

    for i in range(len(flare_fns)):
        flares.append(TabtoLightCurve(flare_fns[i]))

    lcs = lk.LightCurveCollection(lc_arr)

    star = Star(id_number=misc['id_number'], radius=misc['radius'], temperature=misc['temperature'],
                period=misc['period'], lcs=lcs, clear_cache=False)

    fl = Flares(star, process=False)

    fl.eclipse_window = misc['eclipse_window']
    fl.flare_rate = misc['flare_rate'] / un.yr
    fl.window = misc['window']
    fl.median = misc['median']
    fl.std = misc['std']
    fl._n_pts = misc['n_pts']

    try:
        flare_table = Table.read(fp + 'flare_table.tab', format='ascii')
        flags = []
        for f in flare_table:
            if f['flag'] == 'True':
                flags.append(True)
            elif f['flag'] == 'False':
                flags.append(False)
        flare_table['flag'] = flags
        fl.flare_table = flare_table
    except Exception as e:
        warn(f"Could not read in flare table: {e}")

    fl.lc_arr = lc_arr
    fl.lc_flagged = lc_flag
    fl.lc_norm = lc_norm
    fl.lc_median = lc_median
    fl.lc = fl.star.lcs.stitch()
    fl.flares = flares
    return fl


# ===========================================================================
# NEW: radius lookup (catalog has no radius column -- pull it from the TIC)
# ===========================================================================
def get_tic_radius(tic_id):
    """
    Looks up a star's radius (in solar radii) from the TESS Input Catalog (TIC)
    via astroquery, since the Gaia/LoTSS catalog you have does not include
    radius directly.

    Returns
    -------
    float or None
        Radius in solar radii, or None if not found / query failed.
    """
    try:
        from astroquery.mast import Catalogs
    except ImportError:
        warn("astroquery is not installed. Run: pip install astroquery")
        return None

    try:
        result = Catalogs.query_criteria(catalog="Tic", ID=int(tic_id))
        if len(result) == 0:
            warn(f"TIC {tic_id}: no entry found in the TESS Input Catalog.")
            return None
        rad = result["rad"][0]  # TIC's stellar radius column, in solar radii
        if np.ma.is_masked(rad) or rad is None or np.isnan(rad):
            warn(f"TIC {tic_id}: TIC entry found but radius is missing.")
            return None
        return float(rad)
    except Exception as e:
        warn(f"TIC {tic_id}: radius query failed: {e}")
        return None


# ===========================================================================
# NEW: connector -- catalog row -> Star -> Flares -> saved results
# ===========================================================================
def process_star_flares(tic_value, table=tab, sectors=range(14),
                         base_dir='flare_results', find_flares_kwargs=None):
    """
    Looks up a star in the catalog, builds a Star + Flares object, runs
    flare detection, and saves the results to disk.

    Returns a dict summary (or None if the star couldn't be processed).
    """
    find_flares_kwargs = find_flares_kwargs or {}

    # --- Step 1: look up the star's row in the catalog ---
    idx, row, tic_name, row_info = get_catalog_row(tic_value, table=table)
    tic_id = int(row_info["ID"])

    # --- Step 2: Teff -- fall back to Star's default if missing, but warn ---
    teff = row_info.get("Teff", None)
    if teff is None or np.ma.is_masked(teff):
        warn(f"TIC {tic_id}: no Teff in catalog, using Star's default 5700 K.")
        teff = 5700

    # --- Step 3: radius -- catalog has none, so query the TIC directly ---
    radius = row_info.get("radius", None)
    if radius is None or (np.ma.is_masked(radius) if hasattr(radius, 'mask') else False):
        radius = get_tic_radius(tic_id)
    if radius is None:
        warn(f"TIC {tic_id}: no radius available from catalog or TIC query. Skipping star.")
        return None

    # --- Step 4: build the Star object (downloads TESS light curves) ---
    try:
        star = Star(id_number=tic_id, radius=float(radius),
                    temperature=float(teff), sectors=sectors)
    except Exception as e:
        warn(f"TIC {tic_id}: could not build Star (likely no light curves found): {e}")
        return None

    # --- Step 5: run flare detection ---
    fl = Flares(star)
    try:
        fl.FindAllFlares(**find_flares_kwargs)
    except Exception as e:
        warn(f"TIC {tic_id}: FindAllFlares failed: {e}")
        return None

    # --- Step 6: save to disk (flare_table.tab + star_information.json etc.) ---
    fl.WriteOutData(base_dir=base_dir)

    # --- Step 7: build the summary to feed back into the catalog table ---
    n_flares = int(np.sum(fl.flare_table["flag"] == False))
    if n_flares > 0:
        good = fl.flare_table[fl.flare_table["flag"] == False]
        peak_energy = float(np.nanmax(good["energy"]))
    else:
        peak_energy = np.nan

    return {
        "idx": idx,
        "tic_id": tic_id,
        "radius_used": float(radius),
        "n_flares": n_flares,
        "flare_rate_per_yr": float(fl.flare_rate.value),
        "peak_flare_energy_erg": peak_energy,
    }


# ===========================================================================
# NEW: batch driver -- adds summary columns to tab, writes a new CSV
# ===========================================================================
def run_batch(table=tab, base_dir='flare_results',
              out_csv="gaia_info_100pc_lc_info_lotss_with_flares.csv",
              checkpoint_every=20):
    """Runs process_star_flares() over every star in the table, records results."""
    table["n_flares"] = np.full(len(table), -1, dtype=int)          # -1 = not processed
    table["flare_rate"] = np.full(len(table), np.nan)
    table["peak_flare_energy"] = np.full(len(table), np.nan)

    results = []
    for i, tic_id in enumerate(table["ID"]):
        summary = process_star_flares(tic_id, table=table, base_dir=base_dir)
        if summary is None:
            continue

        table["n_flares"][summary["idx"]] = summary["n_flares"]
        table["flare_rate"][summary["idx"]] = summary["flare_rate_per_yr"]
        table["peak_flare_energy"][summary["idx"]] = summary["peak_flare_energy_erg"]
        results.append(summary)

        if i % checkpoint_every == 0:
            table.write(out_csv, overwrite=True)

    table.write(out_csv, overwrite=True)
    return results


if __name__ == "__main__":
    # test on one known TIC ID before running the whole catalog
    summary = process_star_flares(470085072, table=tab, sectors=None)
    print(summary)
    
    