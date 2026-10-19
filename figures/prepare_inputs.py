"""
prepare the heavy ERA data inputs avaiable 

  synoptic_2007_*.nc        2 m temperature anomaly, z500 and heatwave mask for the 4 dates of Fig. 1
  era5_flux_window_2007.nc  ERA5 sensible / latent heat flux averaged over the pathway (Fig. 2 b, c)
  region_masks.nc           the region masks of the regional bar figures; heatwave_mask_greece.nc = full heatwave mask

    python prepare_inputs.py
"""

import shutil
import sys

import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path

from paths import (FIG8_EXTENT, FIG8_EPISODES, REF_PERIOD, SMOOTH_DAYS, SYNOPTIC_EVENTS_FILE,
                   GEOPT_CLIM_FILE, Z500_EVENTS, INPUTS, SYNOPTIC_FILES, SYNOPTIC_DATES, ERA5_WINDOW_2007, PEAK_DAYS_2007, BACK_DAYS,
                   T2M_LAND, Z500_2007, HW_MASK, ERA5_FLUX_2007)

INPUTS.mkdir(parents=True, exist_ok=True)


def synoptic_2007():
    # --- 2 m temperature: daily maximum anomaly relative to the same calendar days of all years
    ds = xr.open_dataset(T2M_LAND, chunks={"time": 500})
    t = ds["2t"]
    t = t.assign_coords(lon=((t.lon + 180) % 360) - 180).sortby("lon")
    t = t.sel(lat=slice(20, 80), lon=slice(-80, 80))
    target = t.sel(time=slice("2007-06-21", "2007-07-25"))
    doys = np.unique(target.time.dt.dayofyear.values)
    clim_input = t.where(t.time.dt.dayofyear.isin(doys), drop=True)
    target_dailymax = target.resample(time="1D").max()
    clim_dailymax = clim_input.resample(time="1D").max()
    clim = clim_dailymax.groupby("time.dayofyear").mean("time")
    t_anom = (target_dailymax.groupby("time.dayofyear") - clim).rename("t_anom")
    t_anom.sel(time=SYNOPTIC_DATES).load().to_netcdf(SYNOPTIC_FILES["t2m_anom"])
    print("saved", SYNOPTIC_FILES["t2m_anom"])

    # --- 500 hPa geopotential height (z / 100, as used in the figure), nearest time to each date
    z500 = xr.open_dataset(Z500_2007)
    z500 = z500.assign_coords(longitude=((z500.longitude + 180) % 360) - 180).sortby("longitude")
    z_height = (z500["z"].isel(pressure_level=0) / 100).rename({"valid_time": "time"}).rename("z500_height")
    z_height.sel(time=SYNOPTIC_DATES, method="nearest").load().to_netcdf(SYNOPTIC_FILES["z500"])
    print("saved", SYNOPTIC_FILES["z500"])

    # --- heatwave mask from the detection step
    mask = xr.open_dataset(HW_MASK)
    mask[["heatwave_mask"]].sel(time=SYNOPTIC_DATES).load().to_netcdf(SYNOPTIC_FILES["mask"])
    print("saved", SYNOPTIC_FILES["mask"])


def era5_window_2007():
    era5 = xr.open_dataset(ERA5_FLUX_2007)
    if float(era5.longitude.max()) > 180:                     # same longitude convention as FLEXPART
        era5 = era5.assign_coords(longitude=((era5.longitude + 180) % 360) - 180).sortby("longitude")
    first, last = PEAK_DAYS_2007[0], PEAK_DAYS_2007[-1]
    win_start = np.datetime64(first) - np.timedelta64(BACK_DAYS, "D")
    win_end = np.datetime64(last) + np.timedelta64(1, "D")
    win = era5.sel(valid_time=slice(win_start, win_end))
    out = xr.Dataset({"shf_mean": (-win["avg_ishf"]).mean("valid_time"),     # W m-2, positive = upward
                      "lhf_mean": (-win["avg_slhtf"]).mean("valid_time")})
    out.attrs["window"] = f"{pd.Timestamp(win_start).date()} to {pd.Timestamp(win_end).date()}"
    out.load().to_netcdf(ERA5_WINDOW_2007)
    print("saved", ERA5_WINDOW_2007, "| averaging window:", out.attrs["window"])



def heatwave_mask():
    shutil.copy2(HW_MASK, INPUTS / "heatwave_mask_greece.nc")
    print("copied", INPUTS / "heatwave_mask_greece.nc")


# ---------------------------------------------------------------------------------------
# Fig. 8: 500 hPa height anomaly of every event, averaged over the transport window
# ---------------------------------------------------------------------------------------
G_STD = 9.80665


def to_dam(da):
    """Geopotential (m2 s-2) or geopotential height (m) -> decametres (probes one 2-D slice)."""
    sl = da
    for d in da.dims:
        if d not in ("latitude", "longitude"):
            sl = sl.isel({d: 0})
    probe = float(np.abs(sl).max())
    if probe > 2e4:
        return da / (G_STD * 10.0)
    if probe > 2e3:
        return da / 10.0
    return da


def tidy(da):
    """Squeeze the level dimension, call the time dimension 'time', wrap longitude to -180..180."""
    for d in ("level", "pressure_level"):
        if d in da.dims:
            da = da.isel({d: 0}, drop=True)
    if "valid_time" in da.dims:
        da = da.rename({"valid_time": "time"})
    da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180)
    return da.sortby("longitude")


def crop(da, pad=5):
    """Cut to the plotting domain (+pad). Latitude is descending in ERA5."""
    return da.sel(longitude=slice(FIG8_EXTENT[0] - pad, FIG8_EXTENT[1] + pad),
                  latitude=slice(FIG8_EXTENT[3] + pad, FIG8_EXTENT[2] - pad))


def synoptic_events():
    cache = Path("z500_clim_dam_1991_2020.nc")                 # reused if it already exists in this folder
    if cache.exists():
        clim = xr.open_dataarray(cache)
        print("climatology read from", cache)
    else:
        print("building the 1991-2020 climatology (slow) ...", flush=True)
        ds = xr.open_dataset(GEOPT_CLIM_FILE, chunks={"time": 500})
        z = to_dam(crop(tidy(ds["geopt"]))).sel(time=slice(*REF_PERIOD))
        clim = z.resample(time="1D").mean().groupby("time.dayofyear").mean("time")
        clim = clim.rolling(dayofyear=2 * SMOOTH_DAYS + 1, center=True, min_periods=1).mean().compute()
        clim.name = "z500_clim"
        clim.to_netcdf(cache)

    if SYNOPTIC_EVENTS_FILE.exists():
        SYNOPTIC_EVENTS_FILE.unlink()
    for yr, (ep0, ep1) in FIG8_EPISODES.items():
        w0 = np.datetime64(ep0) - np.timedelta64(BACK_DAYS - 1, "D")
        w1 = np.datetime64(ep1)
        zds = xr.open_dataset(Z500_EVENTS[yr])
        z = to_dam(crop(tidy(zds["z" if "z" in zds else "geopt"])))
        zd = z.resample(time="1D").mean()
        c = clim
        if not np.array_equal(zd.longitude.values, c.longitude.values):
            c = c.interp(longitude=zd.longitude, latitude=zd.latitude)
        anom = (zd.groupby("time.dayofyear") - c).sel(time=slice(w0, w1)).mean("time").rename("z500_anomaly")
        anom.attrs.update(episode=f"{ep0} to {ep1}", window=f"{str(w0)[:10]} to {str(w1)[:10]}", units="dam")
        anom.load().to_netcdf(SYNOPTIC_EVENTS_FILE, group=yr, mode="a" if SYNOPTIC_EVENTS_FILE.exists() else "w")
        print("saved", yr, "| episode", ep0, "to", ep1, "| window", str(w0)[:10], "to", str(w1)[:10], flush=True)
    print("saved", SYNOPTIC_EVENTS_FILE)


if __name__ == "__main__":
    parts = sys.argv[1:] or ["synoptic", "era5", "mask", "fig08"]   # e.g.  python prepare_inputs.py fig08
    if "synoptic" in parts:
        synoptic_2007()
    if "era5" in parts:
        era5_window_2007()
    if "mask" in parts:
        heatwave_mask()
    if "fig08" in parts:
        synoptic_events()
    print("\nDONE. Also copy flux_by_backtime_long_2007_peak.csv into", INPUTS,
          "(made by extract_backtime_peak.py).")