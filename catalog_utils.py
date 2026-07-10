"""
catalog_utils.py

Shared catalog-loading and lookup utilities for the Gaia/LoTSS cross-matched
target list (nearby stars within 100 pc, with TESS + LOFAR coverage).

Both the radio pipeline (DynamicSpectrum) and the optical pipeline
(Star/Flares) import from this file, so the catalog only gets loaded once,
in one place -- instead of every script re-reading the CSV independently.
"""

from astropy.table import Table
import numpy as np

# ---------------------------------------------------------------------------
# Load the catalog once, at import time.
# This runs automatically the moment anything does `import catalog_utils`
# or `from catalog_utils import tab`.
# ---------------------------------------------------------------------------
env1 = "C:/Users/ADMIN/Downloads/gaia_info_100pc_lc_info_lotss.csv"

# Table.read parses the CSV into an Astropy Table -- like a DataFrame, but
# with better support for units, masked (missing) values, and metadata.
tab = Table.read(env1)

# Pull out the TIC ID column on its own, in case anything wants to loop
# over all IDs directly without going through the full table.
ids = tab["ID"]


def get_catalog_row(tic_value, table=tab):
    """
    Find a catalog row using a TIC ID.

    Parameters
    ----------
    tic_value : int or str
        TIC ID, e.g. 470085072 or "TIC 470085072".
    table : astropy.table.Table
        Catalog table (must have an "ID" column). Defaults to the module-
        level `tab` loaded above, but can be overridden (e.g. for testing
        on a smaller table).

    Returns
    -------
    idx : int
        Index of the matching row in the table (its position, 0-based).
    row : astropy.table.Row
        The full row object -- lets you do row['Teff'], row['ra'], etc.
    tic_name : str
        TIC name string, e.g. "TIC 470085072" -- this is the exact format
        lightkurve's search functions expect.
    row_info : dict
        Same row's data as a plain Python dict, {column_name: value}.
        Easier to work with than an Astropy Row in some contexts (e.g.
        .get() with a default, or json.dump-ing it later).
    """
    # --- Step 1: normalize whatever was passed in to a plain integer ID ---
    if isinstance(tic_value, str):
        tic_value = tic_value.upper().replace("TIC", "").strip()
    tic_value = int(tic_value)

    # --- Step 2: build the "TIC <id>" string used by lightkurve searches ---
    tic_name = f"TIC {tic_value}"

    # --- Step 3: search the table's ID column for a matching row ---
    matches = np.where(table["ID"] == tic_value)[0]

    # --- Step 4: handle "not found" and "found more than once" cases ---
    if len(matches) == 0:
        raise ValueError(f"TIC {tic_value} not found in the catalog.")
    if len(matches) > 1:
        print(f"Warning: {len(matches)} rows match TIC {tic_value}, using the first.")

    # --- Step 5: pull out the row and package it up ---
    idx = int(matches[0])
    row = table[idx]

    row_info = {col: row[col] for col in table.colnames}

    return idx, row, tic_name, row_info