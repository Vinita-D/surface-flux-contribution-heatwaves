import os
from pathlib import Path

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
    grid_initial_file: Path,
    part_ic_file: Path,
    end_date: str,
):
    age_seconds = 10 * 24 * 3600  # 10 days backward

    release_times = get_release_times_from_part_ic_file(
        part_ic_file=str(part_ic_file),
        end_date=end_date,
    )

    correct_times = release_times - pd.to_timedelta(age_seconds, unit="s")

    tmp_file = grid_initial_file.with_name("grid_initial_001_tmp.nc")
    backup_file = grid_initial_file.with_name("grid_initial_001_before_timefix.nc")

    with xr.open_dataset(grid_initial_file) as ds:
        if ds.sizes["time"] != len(correct_times):
            raise ValueError(
                f"time dimension = {ds.sizes['time']}, "
                f"but release times = {len(correct_times)}"
            )

        ds_fixed = ds.assign_coords(time=correct_times)
        ds_fixed.to_netcdf(tmp_file)

    # keep one backup of the original file
    if not backup_file.exists():
        grid_initial_file.rename(backup_file)
    else:
        grid_initial_file.unlink()

    # rename fixed file to the original name
    tmp_file.rename(grid_initial_file)

    print("Corrected file:", grid_initial_file, flush=True)
    print("Backup saved as:", backup_file, flush=True)
    print("First 5 corrected times:", correct_times[:5], flush=True)
    print("Last 5 corrected times:", correct_times[-5:], flush=True)


def main():
    flexpart_dir = Path(os.environ["flexpart_dir"]).expanduser().resolve()
    part_ic_file = Path(os.environ["part_ic_file"]).expanduser().resolve()

    # this matches your run_all_function.py variable
    end_date = os.environ["part_ic_run_end"]

    grid_initial_file = flexpart_dir / "grid_initial_001.nc"

    if not grid_initial_file.is_file():
        raise FileNotFoundError(f"Cannot find: {grid_initial_file}")

    if not part_ic_file.is_file():
        raise FileNotFoundError(f"Cannot find: {part_ic_file}")

    correct_grid_initial_time(
        grid_initial_file=grid_initial_file,
        part_ic_file=part_ic_file,
        end_date=end_date,
    )


if __name__ == "__main__":
    main()