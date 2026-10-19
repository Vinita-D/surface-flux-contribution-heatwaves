import os
from pathlib import Path
import numpy as np

import function_binary_to_ncdf_flexpart as binary_ncdf
import read_grid_init_001_Marina as read_grid


in_dir = Path(os.environ["flexpart_dir"])
out_dir = in_dir
out_dir.mkdir(parents=True, exist_ok=True)

###########################################################################
# New case: if just one combined binary grid_initial_001 + grid_time already in nc
###########################################################################
combined_grid_initial = in_dir / "grid_initial_001"

if combined_grid_initial.is_file():
    print(f"For a combined grid_initial_001  ", flush=True)
    print("only converting grid_initial_001", flush=True)

    ds_ini = read_grid.read_gridini(str(in_dir), "grid_initial_001")
    out_nc = out_dir / "grid_initial_001.nc"
    ds_ini.to_netcdf(out_nc)

    print(f"wrote {out_nc}", flush=True)

###########################################################################
# Old case: multiple binary grid_time and grid_initial files
###########################################################################
else:
    print("For LINVERSION mode -  multiple-file binary conversion.", flush=True)

    #######################################################################
    # For grid_time files
    #######################################################################
    hdr = binary_ncdf.read_header_min(in_dir / "header_grid_time")

    for bin_path in sorted(in_dir.glob("grid_time_*_001")):
        times, grids = binary_ncdf.read_grid_time_debug(bin_path, hdr, max_steps=None)

        if len(grids) == 0:
            print(
                f"Skipping {bin_path.name}: file contains no timesteps "
                f"(size={bin_path.stat().st_size} bytes)",
                flush=True,
            )
            continue

        ds = binary_ncdf.make_xarray_from_grids(times, grids, hdr)
        out_nc = out_dir / (bin_path.name.replace("_001", "") + "_python.nc")
        ds.to_netcdf(out_nc)
        print("wrote", out_nc, flush=True)

    #######################################################################
    # For grid_initial files
    #######################################################################
    hdr = binary_ncdf.read_header_min(in_dir / "header")
    lage_seconds = binary_ncdf.read_lage_seconds(in_dir / "AGECLASSES.namelist")
    print("LAGE (s) =", lage_seconds, flush=True)

    for grid_init_path in sorted(in_dir.glob("grid_initial_*_001")):
        t_rec = binary_ncdf.time_from_grid_file_name(grid_init_path)
        t_init = t_rec - np.timedelta64(lage_seconds, "s")

        yyyymmdd, hhmiss, grid3d = binary_ncdf.read_grid_initial(grid_init_path, hdr)
        ds = binary_ncdf.make_xarray_from_grid_initial(t_init, grid3d, hdr)

        out_nc = out_dir / (grid_init_path.name.replace("_001", "") + "_python.nc")
        ds.to_netcdf(out_nc)

        print(f"wrote {out_nc}  (t_rec={t_rec}, t_init={t_init})", flush=True)