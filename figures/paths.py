"""
paths.py -- all paths, constants and dates live here; every figure script imports from this file.

THE WHOLE SET-UP
    flux_maps.nc, ic_maps.nc, merged_hourly.csv   required for figures 3, 5, 6, 7
     six small files, already made                  required for  figures 1, 2, 4
  ERA5 files (public, Copernicus Climate Data Store)                     to re-make inputs/

"""
import os
from pathlib import Path

# ---- 1. data folder ------------------------------------------------------------
DATA = Path(os.environ.get("HEATWAVE_DATA", "/jetfs/scratch/vdeshmukh/Heatwaves_Greek_MSE_data"))
INPUTS = DATA / "inputs"
YEARS = ["1982", "1987", "1988", "2007", "2021", "2024", "2025"]

FILES = {y: str(DATA / y / "flux_maps.nc") for y in YEARS}           # per-release flux maps
IC_FILES = {y: str(DATA / y / "ic_maps.nc") for y in YEARS}          # per-release initial-condition maps
CSV_FILES = {y: str(DATA / y / "merged_hourly.csv") for y in YEARS}  # per-release budget table

SYNOPTIC_FILES = {"t2m_anom": INPUTS / "synoptic_2007_t2m_anom.nc",   # Fig. 1
                  "z500": INPUTS / "synoptic_2007_z500.nc",
                  "mask": INPUTS / "synoptic_2007_heatwave_mask.nc"}
ERA5_WINDOW_2007 = INPUTS / "era5_flux_window_2007.nc"               # Fig. 2 b, c
BACKTIME_CSV = INPUTS / "flux_by_backtime_long_2007_peak.csv"        # Fig. 4

# ---- 2. constants ---------------------------------------------------------------
CP = 1004.7              # J kg-1 K-1
G = 9.81                 # m s-2
LV_CSV = 1e6             # J/kg = LV_CSV * MF_flux        (merged_hourly.csv; the factor 2.2 is already contained in MF_flux)
LV_IC = 2.2e6            # J/kg = LV_IC * IC_q_grid  
BACK_DAYS = 10           # length of the backward window (days)

# ---- 3. heatwave days (both days included) ---------------------------------------
HW_PERIODS = {
    "1982": [("1982-06-24", "1982-06-28")],
    "1987": [("1987-07-20", "1987-07-27")],
    "1988": [("1988-07-05", "1988-07-09")],
    "2007": [("2007-06-21", "2007-06-28"), ("2007-07-17", "2007-07-26"), ("2007-08-22", "2007-08-26")],
    "2021": [("2021-06-22", "2021-06-26"), ("2021-07-27", "2021-08-06")],
    "2024": [("2024-06-09", "2024-06-13"), ("2024-06-18", "2024-06-23"),
             ("2024-07-10", "2024-07-19"), ("2024-08-12", "2024-08-16")],
    "2025": [("2025-06-24", "2025-06-28"), ("2025-07-21", "2025-07-27")],
}

# ---- 4. the 2007 example used in figures 1-4 ---------------------------------------
PEAK_DAYS_2007 = ["2007-06-25", "2007-06-26", "2007-06-27"]    #for figure 2,3,4     
SYNOPTIC_DATES = ["2007-06-22", "2007-06-26", "2007-07-21", "2007-07-24"]   #for figure 1 
RECEPTOR_BOX = dict(lon1=19.0, lat1=34.0, width=8.0, height=8.5)    # blue box on the maps

# ---- 5. input files from outside -------------------------------------------------
# Only needed to RE-MAKE the six small files in inputs/ (prepare_inputs.py). .
_S = "/jetfs/scratch/vdeshmukh/"
ERA5_FLUX_2007 = _S + "flux_ERA5/flux_files_res/ERA5_mean_fluxes_2007_MJJA_regridded_0p5.nc"   # ERA5 heat fluxes for Fig. 2 b, c
Z500_2007 = _S + "z_pv_era5_files/z500_2007_JJA.nc"                                             # ERA5 500 hPa height  for Fig. 1
T2M_LAND = _S + "2m_data_temp/ERA5_t2m_q2m_sp_1979_2025_0p5_land.nc"                            # ERA5 2 m temperature for  Fig. 1, land/sea mask
HW_MASK = "/jetfs/home/vdeshmukh/Post_processing_files/heatwave_mask_greece.nc"                 # heatwave mask of the detection step for Fig. 1

# Only needed by extract_backtime_peak.py: takes longer time to compute - generates csv file 
GRID_TIME_2007 = _S + "2007_grecce_heatwave2/summer_10daysback/grid_time_20070831180000.nc"
PART_IC_2007 = _S + "2007_grecce_heatwave2/options/part_ic.nc"
PART_IC_RUN_END_2007 = "2007-08-31 18:00:00"


# ---- helper ---------------------------------------------------------------------
def hw_mask(times, year):
    """True for release times that fall on heatwave days of that year."""
    import numpy as np
    import pandas as pd
    t = pd.to_datetime(np.asarray(times))
    m = np.zeros(len(t), dtype=bool)
    for start, end in HW_PERIODS[str(year)]:
        m |= (t >= pd.Timestamp(start)) & (t < pd.Timestamp(end) + pd.Timedelta(days=1))
    return m


# ---- Fig. 8: synoptic situation of all events - to get anomolous weather pattern ----------------------------------------
FIG8_EXTENT = [-35, 50, 18, 72]                
FIG8_EPISODES = {                               # strongest episode per year
    "1982": ("1982-06-24", "1982-06-28"),
    "1987": ("1987-07-20", "1987-07-27"),
    "1988": ("1988-07-05", "1988-07-09"),
    "2007": ("2007-06-21", "2007-06-28"),
    "2021": ("2021-06-22", "2021-06-26"),
    "2024": ("2024-08-12", "2024-08-16"),
    "2025": ("2025-07-21", "2025-07-27"),
}
REF_PERIOD = ("1991-01-01", "2020-12-31")       # climatology of the 500 hPa height
SMOOTH_DAYS = 5                                 # +/- days of smoothing over each day 
SYNOPTIC_EVENTS_FILE = INPUTS / "synoptic_events_z500_anomaly.nc"    # small file made by prepare_inputs.py

# ERA5 500 hPa geopotential, only needed to RE-MAKE the file above (not needed to reproduce Fig. 8)
GEOPT_CLIM_FILE = _S + "geopt_regrid.nc"                              # 1991-2020, used for the climatology
Z500_EVENTS = {y: _S + f"z_pv_era5_files/z500_{y}_JJA.nc" for y in FIG8_EPISODES}
Z500_EVENTS["1988"] = _S + "z_pv_era5_files/z500_MJJA_1988_regrid.nc"


