import xarray as xr
import numpy as np
import xarray as xr
import numpy as np
import pandas as pd
from scipy.ndimage import label, convolve
from scipy.ndimage import binary_closing
from scipy.ndimage import convolve, label, binary_closing





FILE_PATH = "/jetfs/scratch/vdeshmukh/2m_data_temp/ERA5_t2m_1979_2025_0p5_land_MJJA.nc"
VAR_NAME = "2t"   # change to "t2m" if needed

# Greece-centered region
LAT_RANGE = slice(34.0, 42.5)
LON_RANGE = slice(19.0, 27.0)
#LAT_RANGE = slice(15.0, 45)
#LON_RANGE = slice(10, 60)



OUT_MASK_FILE = "heatwave_mask_greece.nc"
OUT_EVENTS_FILE = "heatwave_events_greece.csv"

OUT_ANOM_FILE = "t_anom_greece.nc"

# Climatology
CLIM_START = "1992-06-01"
CLIM_END   = "2025-08-31"

# Heatwave detection settings
SUMMER_MONTHS = [6, 7, 8]
PERCENTILE = 0.95
WINDOW_DAYS = 8
L_DEG = 1.5
ALPHA = 0.70
MIN_DURATION = 5
OVERLAP_THRESHOLD = 0.60


print("Opening dataset...", flush=True)
ds = xr.open_dataset(FILE_PATH)
ds = ds.assign_coords(
    lon=((ds.lon + 180) % 360) - 180
).sortby("lon")

ds = ds.sel(lat=LAT_RANGE, lon=LON_RANGE)

if VAR_NAME not in ds:
    raise KeyError(f"Variable '{VAR_NAME}' not found. Available variables: {list(ds.data_vars)}")

t = ds[VAR_NAME]

print("Attributes:", t.attrs)

raw_min = float(t.min())
raw_max = float(t.max())
print(f"Raw min/max: {raw_min:.3f}, {raw_max:.3f}")

if raw_max > 200:
    t = t - 273.15
    print("Converted from Kelvin to Celsius.", flush=True)

print(f"Post-conversion min/max: {float(t.min()):.3f}, {float(t.max()):.3f}")

# Land mask from first timestep
land_mask = t.isel(time=0).notnull()

# Remove Feb 29 for 365-day climatology
is_feb29 = (t.time.dt.month == 2) & (t.time.dt.day == 29)
t = t.sel(time=~is_feb29)

# ============================================================
# 3. DAILY MAXIMUM TEMPERATURE
# ============================================================

print("Computing daily Tmax...", flush=True)
t_daily = t.resample(time="1D").max()

is_feb29_daily = (t_daily.time.dt.month == 2) & (t_daily.time.dt.day == 29)
t_daily = t_daily.sel(time=~is_feb29_daily)

print(f"Daily Tmax min/max: {float(t_daily.min()):.3f}, {float(t_daily.max()):.3f}")

# ============================================================
# 4. DAILY CLIMATOLOGICAL THRESHOLD
# ============================================================

print("Building climatological threshold...", flush=True)
t_clim = t_daily.sel(time=slice(CLIM_START, CLIM_END))

if t_clim.time.size == 0:
    raise ValueError("No data found in the selected climatology period.")

clim_doy = t_clim.time.dt.dayofyear.values

def circular_day_distance(a, b):
    d = np.abs(a - b)
    return np.minimum(d, 365 - d)

threshold_list = []

for target_doy in range(1, 366):
    keep = circular_day_distance(clim_doy, target_doy) <= WINDOW_DAYS
    thresh = np.nanpercentile(t_clim.values[keep, :, :], 100 * PERCENTILE, axis=0)
    threshold_list.append(thresh)

threshold = xr.DataArray(
    np.stack(threshold_list, axis=0),
    coords={
        "dayofyear": np.arange(1, 366),
        "lat": t_daily.lat,
        "lon": t_daily.lon
    },
    dims=("dayofyear", "lat", "lon"),
    name="threshold"
) 

print("Threshold computed.", flush=True)

# ============================================================
# 5. JJA HOT-DAY MASK + DAILY ANOMALY
# ============================================================

print("Selecting JJA and computing exceedance mask...", flush=True)

t_jja = t_daily.where(t_daily.time.dt.month.isin(SUMMER_MONTHS), drop=True)
jja_doy = t_jja.time.dt.dayofyear

threshold_for_jja = threshold.sel(dayofyear=xr.DataArray(jja_doy.values, dims="time"))

# --- THIS IS THE VARIABLE YOU WANTED ---
t_anom = t_jja - threshold_for_jja
t_anom.name = "t_anom"
t_anom.attrs["description"] = "Daily Tmax anomaly relative to moving climatological percentile threshold"
t_anom.attrs["units"] = "degC"

# optional save
#print(f"Saved: {OUT_ANOM_FILE}", flush=True)

hot = (t_jja > threshold_for_jja) & land_mask
hot.name = "hot_exceedance"

# ============================================================
# 6. SPATIAL COHERENCE FILTER
# ============================================================

print("Applying spatial coherence filter...", flush=True)

dlat = float(abs(t_jja.lat[1] - t_jja.lat[0]))
dlon = float(abs(t_jja.lon[1] - t_jja.lon[0]))

nlat_win = max(1, int(round(L_DEG / dlat)))
nlon_win = max(1, int(round(L_DEG / dlon)))

if nlat_win % 2 == 0:
    nlat_win += 1
if nlon_win % 2 == 0:
    nlon_win += 1

kernel = np.ones((nlat_win, nlon_win), dtype=float)

lat_weights_1d = np.cos(np.deg2rad(t_jja.lat.values))
lat_weights_2d = lat_weights_1d[:, None] * np.ones((len(t_jja.lat), len(t_jja.lon)))

hot_np = hot.values.astype(float)
land_np = land_mask.values.astype(float)

coherent_np = np.zeros_like(hot_np, dtype=bool)

for tt in range(hot_np.shape[0]):
    if tt % 20 == 0:
        print(f"Spatial filter day {tt+1}/{hot_np.shape[0]}", flush=True)

    hot_weighted = hot_np[tt] * lat_weights_2d
    land_weighted = land_np * lat_weights_2d

    num = convolve(hot_weighted, kernel, mode="constant", cval=0.0)
    den = convolve(land_weighted, kernel, mode="constant", cval=0.0)

    frac = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    coherent_np[tt] = (frac >= ALPHA) & land_mask.values

coherent = xr.DataArray(
    coherent_np,
    coords=hot.coords,
    dims=hot.dims,
    name="coherent_hot_mask"
)

# ============================================================
# 7. LABEL CONNECTED REGIONS PER DAY
# ============================================================

structure = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]], dtype=int)

structure_close = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=bool)

coherent_merged_np = np.zeros_like(coherent_np, dtype=bool)

for tt in range(coherent_np.shape[0]):
    merged = binary_closing(coherent_np[tt], structure=structure_close, iterations=1)
    coherent_merged_np[tt] = merged & land_mask.values

# ============================================================
# 8. TRACK EVENTS THROUGH TIME
# ============================================================

print("Tracking events...", flush=True)

events = []
active_events = {}
next_event_id = 1

for tt in range(len(t_jja.time)):
    date_now = pd.to_datetime(str(t_jja.time.values[tt])).date()
    lbl, nreg = label(coherent_merged_np[tt], structure=structure)

    current_regions = {}

    for reg_id in range(1, nreg + 1):
        mask = (lbl == reg_id)
        area = int(mask.sum())
        if area > 0:
            current_regions[reg_id] = {
                "mask": mask,
                "area": area
            }

    new_active_events = {}
    used_region_ids = set()

    for ev_id, ev in active_events.items():
        best_match_region = None
        best_overlap = 0.0

        prev_mask = ev["last_mask"]
        prev_area = ev["last_area"]

        for reg_id, reg in current_regions.items():
            if reg_id in used_region_ids:
                continue

            curr_mask = reg["mask"]
            curr_area = reg["area"]

            overlap = np.logical_and(prev_mask, curr_mask).sum()
            overlap_fraction = overlap / min(prev_area, curr_area)

            if overlap_fraction > best_overlap:
                best_overlap = overlap_fraction
                best_match_region = reg_id

        if best_match_region is not None and best_overlap > OVERLAP_THRESHOLD:
            reg = current_regions[best_match_region]
            used_region_ids.add(best_match_region)

            ev["end_date"] = date_now
            ev["duration"] += 1
            ev["last_mask"] = reg["mask"]
            ev["last_area"] = reg["area"]
            ev["dates"].append(date_now)
            ev["areas"].append(reg["area"])
            ev["daily_masks"].append(reg["mask"].copy())
            ev["union_mask"] = np.logical_or(ev["union_mask"], reg["mask"])

            new_active_events[ev_id] = ev
        else:
            events.append(ev)

    for reg_id, reg in current_regions.items():
        if reg_id in used_region_ids:
            continue

        event = {
            "event_id": next_event_id,
            "start_date": date_now,
            "end_date": date_now,
            "duration": 1,
            "last_mask": reg["mask"],
            "last_area": reg["area"],
            "dates": [date_now],
            "areas": [reg["area"]],
            "daily_masks": [reg["mask"].copy()],
            "union_mask": reg["mask"].copy(),
        }

        new_active_events[next_event_id] = event
        next_event_id += 1

    active_events = new_active_events

for ev in active_events.values():
    events.append(ev)

# ============================================================
# 9. MINIMUM DURATION FILTER
# ============================================================

print("Applying minimum duration filter...", flush=True)
events = [ev for ev in events if ev["duration"] >= MIN_DURATION]
print(f"Number of events kept: {len(events)}", flush=True)

# ============================================================
# 10. BUILD FINAL MASK
# ============================================================

print("Building final heatwave mask...", flush=True)

final_mask_np = np.zeros_like(coherent_np, dtype=np.int8)

for ev in events:
    for d, mask in zip(ev["dates"], ev["daily_masks"]):
        idx = np.where(pd.to_datetime(t_jja.time.values).date == d)[0]
        if len(idx) == 0:
            continue
        day_index = idx[0]
        final_mask_np[day_index][mask] = 1

final_mask = xr.DataArray(
    final_mask_np,
    coords=coherent.coords,
    dims=coherent.dims,
    name="heatwave_mask"
)

final_mask.attrs["description"] = " regional heatwave mask"
final_mask.to_netcdf(OUT_MASK_FILE)
#print(f"Saved: {OUT_MASK_FILE}", flush=True)

# ============================================================
# 11. EVENT SUMMARY TABLE
# ============================================================

print("Creating event summary table...", flush=True)

rows = []

for ev in events:
    union_mask = ev["union_mask"]
    lat_idx, lon_idx = np.where(union_mask)

    if len(lat_idx) > 0:
        lat_vals = t_jja.lat.values[lat_idx]
        lon_vals = t_jja.lon.values[lon_idx]

        lat_min = float(lat_vals.min())
        lat_max = float(lat_vals.max())
        lon_min = float(lon_vals.min())
        lon_max = float(lon_vals.max())
        centroid_lat = float(lat_vals.mean())
        centroid_lon = float(lon_vals.mean())
    else:
        lat_min = lat_max = lon_min = lon_max = np.nan
        centroid_lat = centroid_lon = np.nan

    peak_i = int(np.argmax(ev["areas"]))
    peak_day = ev["dates"][peak_i]
    max_area = int(np.max(ev["areas"]))
    peak_day_mask = ev["daily_masks"][peak_i]

    peak_idx = np.where(pd.to_datetime(t_jja.time.values).date == peak_day)[0]
    if len(peak_idx) == 0:
        peak_day_max_temp = np.nan
        peak_day_max_temp_anom = np.nan
    else:
        day_index = peak_idx[0]

        field_t = t_jja.isel(time=day_index).values
        field_thr = threshold_for_jja.isel(time=day_index).values
        field_anom = field_t - field_thr

        vals_t = field_t[peak_day_mask]
        vals_anom = field_anom[peak_day_mask]

        peak_day_max_temp = float(np.nanmax(vals_t)) if vals_t.size > 0 else np.nan
        peak_day_max_temp_anom = float(np.nanmax(vals_anom)) if vals_anom.size > 0 else np.nan

    rows.append({
        "event_id": ev["event_id"],
        "start_date": ev["start_date"],
        "end_date": ev["end_date"],
        "duration_days": ev["duration"],
        "peak_day": peak_day,
        "max_gridcells": max_area,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
        "peak_day_max_temp": peak_day_max_temp,
        "peak_day_max_temp_anom": peak_day_max_temp_anom
    })

events_df = pd.DataFrame(rows).sort_values(["start_date", "event_id"])
#events_df.to_csv(OUT_EVENTS_FILE, index=False)

print(f"Saved: {OUT_EVENTS_FILE}", flush=True)
print("Done.")


events_df = events_df[events_df["max_gridcells"] >50] # spatially large 


df = events_df.copy()

df["start_date"] = pd.to_datetime(df["start_date"])
df["end_date"] = pd.to_datetime(df["end_date"])
df["peak_day"] = pd.to_datetime(df["peak_day"])

# sort for easier merging
df = df.sort_values(["start_date", "end_date"]).reset_index(drop=True)

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def time_gap_days(row1, row2):
    """
    Gap in days between two events.
    Negative or zero means overlap/touching.
    """
    return (row2["start_date"] - row1["end_date"]).days

def boxes_overlap(row1, row2):
    """
    True if the lat-lon bounding boxes overlap.
    """
    lat_overlap = not (row1["lat_max"] < row2["lat_min"] or row2["lat_max"] < row1["lat_min"])
    lon_overlap = not (row1["lon_max"] < row2["lon_min"] or row2["lon_max"] < row1["lon_min"])
    return lat_overlap and lon_overlap



def add_peak_positive_anom_area(df_events, t_anom, anom_threshold=0.0):
    df = df_events.copy()

    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])

    peak_days = []
    peak_areas = []

    for _, row in df.iterrows():
        event_anom = t_anom.sel(
            time=slice(row["start_date"], row["end_date"])
        )
        area_ts = (event_anom > anom_threshold).sum(dim=("lat", "lon"))
        peak_idx = area_ts.argmax(dim="time")

        peak_day = pd.to_datetime(
            area_ts.time.isel(time=peak_idx).values
        )

        peak_area = int(
            area_ts.isel(time=peak_idx).values
        )

        peak_days.append(peak_day)
        peak_areas.append(peak_area)

    df["peak_area_day_tanom_gt0"] = peak_days
    df["max_area_tanom_gt0"] = peak_areas

    return df




def centroids_close(row1, row2, max_dist_deg=6.0):
    """
    Simple centroid distance in degree space.
    Good enough for post-processing.
    """
    dlat = row1["centroid_lat"] - row2["centroid_lat"]
    dlon = row1["centroid_lon"] - row2["centroid_lon"]
    dist = np.sqrt(dlat**2 + dlon**2)
    return dist <= max_dist_deg

def should_merge(row1, row2, max_gap_days=1, max_centroid_dist_deg=6.0):
    """
    Merge if:
    - same year
    - time gap small
    - and either boxes overlap or centroids are close
    """
    same_year = row1["start_date"].year == row2["start_date"].year
    gap = time_gap_days(row1, row2)

    time_ok = gap <= max_gap_days
    space_ok = boxes_overlap(row1, row2) or centroids_close(row1, row2, max_centroid_dist_deg)

    return same_year and time_ok and space_ok

# ============================================================
# 3. MERGE EVENTS INTO EPISODES
# ============================================================

episodes = []
current = None

for _, row in df.iterrows():
    row = row.copy()

    if current is None:
        current = row
        current["source_event_ids"] = [row["event_id"]]
        continue

    if should_merge(current, row, max_gap_days=3, max_centroid_dist_deg=8.0):
        # merge into current episode
        current["end_date"] = max(current["end_date"], row["end_date"])
        current["start_date"] = min(current["start_date"], row["start_date"])

        # recompute duration from merged dates
        current["duration_days"] = (current["end_date"] - current["start_date"]).days + 1

        # expand spatial box
        current["lat_min"] = min(current["lat_min"], row["lat_min"])
        current["lat_max"] = max(current["lat_max"], row["lat_max"])
        current["lon_min"] = min(current["lon_min"], row["lon_min"])
        current["lon_max"] = max(current["lon_max"], row["lon_max"])

        # keep the larger area
        if row["max_gridcells"] > current["max_gridcells"]:
            current["max_gridcells"] = row["max_gridcells"]

        # keep the hotter peak-day metrics
        if row["peak_day_max_temp"] > current["peak_day_max_temp"]:
            current["peak_day_max_temp"] = row["peak_day_max_temp"]

        if row["peak_day_max_temp_anom"] > current["peak_day_max_temp_anom"]:
            current["peak_day_max_temp_anom"] = row["peak_day_max_temp_anom"]
            current["peak_day"] = row["peak_day"]

        # recompute centroid as midpoint of merged box
        current["centroid_lat"] = 0.5 * (current["lat_min"] + current["lat_max"])
        current["centroid_lon"] = 0.5 * (current["lon_min"] + current["lon_max"])

        # keep trace of which original rows were merged
        current["source_event_ids"].append(row["event_id"])

    else:
        episodes.append(current)
        current = row
        current["source_event_ids"] = [row["event_id"]]

# append last one
if current is not None:
    episodes.append(current)

episodes_df = pd.DataFrame(episodes).reset_index(drop=True)

# ============================================================
# 4. CLEAN FORMATTING
# ============================================================
episodes_df = episodes_df.sort_values(["start_date"]).reset_index(drop=True)

print("heatwaves detected:", len(episodes_df))
print("list of h=detected events:", episodes_df)

episodes_df2 = add_peak_positive_anom_area(
    episodes_df,
    t_anom,
    anom_threshold=0.0
)

episodes_df2[
    [
        "start_date",
        "end_date",
        "duration_days",
        "peak_day",
        "peak_area_day_tanom_gt0",
        "max_area_tanom_gt0",
        "peak_day_max_temp",
        "peak_day_max_temp_anom",
    ]
]

df_hw_periods = episodes_df2.copy()

df_hw_periods["start_date"] = pd.to_datetime(df_hw_periods["start_date"])
df_hw_periods["end_date"] = pd.to_datetime(df_hw_periods["end_date"])
df_hw_periods["year"] = df_hw_periods["start_date"].dt.year

df_hw_periods = df_hw_periods#[df_hw_periods["year"].isin(years_keep)]

HW_PERIODS = {}

for year, g in df_hw_periods.groupby("year"):
    periods = []

    for _, row in g.iterrows():
        periods.append((
            row["start_date"].strftime("%Y-%m-%d"),
            row["end_date"].strftime("%Y-%m-%d")
        ))

    HW_PERIODS[str(year)] = periods

print(HW_PERIODS)


df = episodes_df.copy()

df["start_date"] = pd.to_datetime(df["start_date"])
df["end_date"] = pd.to_datetime(df["end_date"])
df["peak_day"] = pd.to_datetime(df["peak_day"])
df["year"] = df["start_date"].dt.year

# --------------------------------------------------
# Count unique heatwave days in each year
# --------------------------------------------------
year_rows = []
year_to_rows = {}

for year, g in df.groupby("year"):
    heatwave_days = set()

    for _, row in g.iterrows():
        days = pd.date_range(row["start_date"], row["end_date"], freq="D")
        heatwave_days.update(days.date)

    heatwave_days = sorted(heatwave_days)

    year_rows.append({
        "year": year,
        "n_heatwave_days": len(heatwave_days),
        "n_events_rows": len(g),
        "first_heatwave_day": heatwave_days[0],
        "last_heatwave_day": heatwave_days[-1]
    })

    year_to_rows[year] = g.sort_values("start_date").copy()

year_df = pd.DataFrame(year_rows).sort_values(
    ["n_heatwave_days", "year"], ascending=[False, True]
).reset_index(drop=True)

print(year_df)





