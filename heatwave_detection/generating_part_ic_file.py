#########################################


# ============================================================
# SETTINGS- simple release in the box with lan mask 
# ============================================================
input_file = "/jetfs/scratch/vdeshmukh/2m_data_temp/ERA5_t2m_1979_2025_0p5_land_MJJA.nc"
output_file = "part_ic_2025.nc"

start_date = "2025-06-01 06:00:00"
end_date   = "2025-07-31 18:00:00"

lat_range = slice(34, 42.5)
lon_range = slice(19, 27)

keep_hours = [6, 9, 12, 15, 18]

particle_per_grid = 500
height_m = 50.0

species_ids = np.array([24], dtype=np.int64)
kindz = np.int64(1)

# ============================================================
# OPEN + SELECT
# ============================================================
ds = xr.open_dataset(input_file)

ds_sel = ds.sel(
    lat=lat_range,
    lon=lon_range,
    time=slice(start_date, end_date)
)

ds_sel = ds_sel.sel(time=ds_sel.time.dt.hour.isin(keep_hours))

if ds_sel.time.size == 0:
    raise ValueError("No time steps found in selected period.")

# ============================================================
# LAND MASK
# ============================================================
first_var = list(ds_sel.data_vars)[0]
land_mask = ds_sel[first_var].isel(time=0).notnull()

lon2d, lat2d = np.meshgrid(ds_sel.lon.values, ds_sel.lat.values)
mask = land_mask.values

land_lon = lon2d[mask]
land_lat = lat2d[mask]

if land_lon.size == 0:
    raise ValueError("No land points found in selected box.")

# ============================================================
# BACKWARD TIME: 0 at end_date, earlier times negative
# ============================================================
time_values = ds_sel.time.values
run_reference = np.datetime64(end_date)
times_sec = ((time_values - run_reference) / np.timedelta64(1, "s")).astype(np.int64)

nland = len(land_lon)
ntimes = len(times_sec)

# ============================================================
# BUILD PARTICLES
# ============================================================
part_lon = np.tile(land_lon, ntimes)
part_lat = np.tile(land_lat, ntimes)
part_height = np.full(ntimes * nland, height_m, dtype=np.float64)

part_time = np.repeat(times_sec, nland).astype(np.int64)

# release number per timestep
part_release = np.repeat(np.arange(1, ntimes + 1, dtype=np.int64), nland)

# repeat each grid-cell release particle_per_grid times
part_lon = np.repeat(part_lon, particle_per_grid).astype(np.float64)
part_lat = np.repeat(part_lat, particle_per_grid).astype(np.float64)
part_height = np.repeat(part_height, particle_per_grid).astype(np.float64)
part_time = np.repeat(part_time, particle_per_grid).astype(np.int64)
part_release = np.repeat(part_release, particle_per_grid).astype(np.int64)

num_part = len(part_lon)
nspecies = len(species_ids)

part_mass = np.ones((nspecies, num_part), dtype=np.float64)

# ============================================================
# OUTPUT DATASET
# ============================================================
ds_out = xr.Dataset(
    data_vars={
        "longitude": (["particle"], part_lon),
        "latitude":  (["particle"], part_lat),
        "height":    (["particle"], part_height),
        "time":      (["particle"], part_time),
        "mass":      (["species", "particle"], part_mass),
        "release":   (["particle"], part_release),
    },
    coords={
        "particle": np.arange(num_part, dtype=np.int64),
        "species": np.arange(nspecies, dtype=np.int64),
    }
)

ds_out.attrs["nspecies"] = np.int64(nspecies)
ds_out.attrs["species"] = species_ids
ds_out.attrs["kindz"] = kindz

ds_out.to_netcdf(output_file)

release_times = pd.to_datetime(ds_sel.time.values)

ds_out.close()
ds.close()

print("Saved:", output_file)
print("Number of land points:", nland)
print("Number of times:", ntimes)
print("Total particles:", num_part)
print("First release time:", release_times.min())
print("Last release time :", release_times.max())
print("Unique time values:", np.unique(part_time)[:10], "...")