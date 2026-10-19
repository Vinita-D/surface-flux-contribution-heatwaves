"""
extract_backtime_peak.py -- HEAVY step (needs the 100+ GB grid_time file, not on Zenodo).
Computes the flux contribution as a function of days before release for the peak-day
releases and saves flux_by_backtime_long_2007_peak.csv. fig04_timeseries_peak.py plots it.

    python extract_backtime_peak.py
Then copy the csv into <data>/inputs/ (that is where paths.BACKTIME_CSV points).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from coupled_SF import get_release_info, _open_fluxes
from paths import (GRID_TIME_2007, ERA5_FLUX_2007, PART_IC_2007, PART_IC_RUN_END_2007, PEAK_DAYS_2007,
                   CP, LATENT_FACTOR, BACK_DAYS)

LV = 2.2e6              # passed to _open_fluxes, as in your original script (converts W m-2 to the e / ss fields)
ZERO_TOL = 1e-30
OUT_CSV = "flux_by_backtime_long_2007_peak.csv"


def compute_flux_by_backward_day(grid_time_file, flux_path, part_ic_file, part_ic_run_end,
                                 cp=CP, lv=LV, back_days=BACK_DAYS, peak_days=None):
    grid_time_file = Path(grid_time_file)
    release_info = get_release_info(part_ic=True, grid_time_file=str(grid_time_file),
                                    part_ic_file=part_ic_file, part_ic_run_end=part_ic_run_end)
    e_all, ss_all = _open_fluxes(flux_path, lv)
    e_all, ss_all = e_all.load(), ss_all.load()
    rows = []

    with xr.open_dataset(grid_time_file) as ds:
        nrel = ds.sizes["pointspec"]
        if len(release_info) != nrel:
            raise ValueError(f"Number of releases ({len(release_info)}) does not match pointspec ({nrel})")
        rel_dates = pd.to_datetime(release_info["release_time"]).dt.normalize()
        idx = np.where(rel_dates.isin(pd.to_datetime(peak_days)))[0] if peak_days is not None else np.arange(nrel)
        n_run = len(idx)
        print(f"Releases to process: {n_run}")

        for k, i in enumerate(idx):
            sen = ds["spec001_mr"].isel(nageclass=0, pointspec=i).sel(height=slice(None, 100)).mean("height")
            release_id = int(release_info.iloc[i]["release"])
            release_time = pd.to_datetime(release_info.iloc[i]["release_time"])

            sen_t_1d = sen.sum(("latitude", "longitude")).values
            nonzero = np.where(np.abs(sen_t_1d) > ZERO_TOL)[0]
            if len(nonzero) == 0:
                print(f"[{k+1}/{n_run}] release {release_id}: no nonzero steps, skipped", flush=True)
                continue
            sen_block = sen.isel(time=slice(nonzero.min(), nonzero.max() + 1))
            times_block = pd.to_datetime(sen_block["time"].values)

            hours_back = (times_block.max() - times_block).total_seconds() / 3600.0
            day_bin = np.clip(np.round(hours_back / 24.0).astype(int), 0, back_days)

            e = e_all.sel(valid_time=times_block.values).rename({"valid_time": "time"})
            ss = ss_all.sel(valid_time=times_block.values).rename({"valid_time": "time"})
            dq = (e * sen_block) / 100.0
            dT = (ss * sen_block) / (cp * 100.0)
            dq_t = dq.sum(("latitude", "longitude")).values * LATENT_FACTOR      # J/kg per step
            dT_t = dT.sum(("latitude", "longitude")).values * cp                 # J/kg per step

            rows.append(pd.DataFrame({"release": release_id, "release_time": release_time,
                                      "day_back": day_bin, "dT_flux": dT_t, "dq_flux": dq_t}))
            print(f"[{k+1}/{n_run}] release {release_id} ({release_time}) -- window {len(nonzero)} steps", flush=True)

    if not rows:
        raise ValueError("No releases found on the peak days -- check PEAK_DAYS_2007 and PART_IC_RUN_END_2007.")
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    df = compute_flux_by_backward_day(GRID_TIME_2007, ERA5_FLUX_2007, PART_IC_2007, PART_IC_RUN_END_2007,
                                      peak_days=PEAK_DAYS_2007)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}  -> copy it to the inputs folder (paths.BACKTIME_CSV)")
