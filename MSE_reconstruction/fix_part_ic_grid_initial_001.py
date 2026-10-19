import numpy as np
import pandas as pd
import xarray as xr


def get_release_times_from_part_ic_file(part_ic_file: str, end_date: str):
    with xr.open_dataset(part_ic_file) as ds:
        rel_sec = np.unique(ds["time"].values)

    rel_sec = np.sort(rel_sec)
    end_time = pd.to_datetime(end_date)

    release_times = end_time + pd.to_timedelta(rel_sec, unit="s")
    return pd.to_datetime(release_times)


def correct_grid_initial_time(
    grid_initial_file: str,
    output_file: str,
    part_ic_file: str,
    end_date: str,
):
    age_seconds = 10 * 24 * 3600  # 10 days backward

    release_times = get_release_times_from_part_ic_file(part_ic_file, end_date)
    correct_times = release_times - pd.to_timedelta(age_seconds, unit="s")

    with xr.open_dataset(grid_initial_file) as ds:
        if ds.sizes["time"] != len(correct_times):
            raise ValueError(
                f"time dimension = {ds.sizes['time']}, but release times = {len(correct_times)}"
            )

        ds_fixed = ds.assign_coords(time=correct_times)
        ds_fixed.to_netcdf(output_file)

    print("Saved:", output_file)
    print("First 5 corrected times:", correct_times[:5])
    print("Last 5 corrected times:", correct_times[-5:])


correct_grid_initial_time(
    grid_initial_file="/jetfs/scratch/vdeshmukh/2007_grecce_heatwave2/output/grid_initial_001.nc",
    output_file="/jetfs/scratch/vdeshmukh/2007_grecce_heatwave2/output/grid_initial_001_fixed.nc",
    part_ic_file="/jetfs/scratch/vdeshmukh/2007_grecce_heatwave2/options/part_ic.nc",
    end_date="2007-07-30 18:00:00",
)
