import re
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset

# ============================================================
# HELPERS FOR PER-RELEASE FILES
# ============================================================
TAG_RE = re.compile(r"_(\d{14})")


def _get_tag(path: Path) -> str:
    m = TAG_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse 14-digit tag from {path.name}")
    return m.group(1)


def _tag_to_datetime(tag: str) -> pd.Timestamp:
    return pd.to_datetime(tag, format="%Y%m%d%H%M%S")


# ============================================================
# RELEASE TIMES FROM ACTUAL part_ic.nc
# ============================================================
def get_release_info_from_part_ic_file(part_ic_file: str, part_ic_run_end: str) -> pd.DataFrame:

    if part_ic_file is None:
        raise ValueError("part_ic_file is required when part_ic=True")
    if part_ic_run_end is None:
        raise ValueError("part_ic_run_end is required when part_ic=True")

    run_end = pd.Timestamp(part_ic_run_end)

    with xr.open_dataset(part_ic_file) as ds:
        df = pd.DataFrame({
            "release": ds["release"].values.astype(np.int64),
            "time": ds["time"].values.astype(np.int64),
        })

    release_df = (
        df.drop_duplicates()
          .sort_values("release")
          .reset_index(drop=True)
    )

    release_df["release_time"] = run_end + pd.to_timedelta(release_df["time"], unit="s")

    return release_df


# ============================================================
# RELEASE TIMES FROM COMBINED grid_time FILE
# ============================================================
def get_release_info_from_grid_time_file(grid_time_file: str) -> pd.DataFrame:
    """
    Read release times from RELSTART / RELEND in a combined grid_time_*.nc file.
    """
    with Dataset(grid_time_file) as f:
        tag = Path(grid_time_file).stem.split("_")[-1]
        ref_time = datetime.strptime(tag, "%Y%m%d%H%M%S")

        relstart_raw = f.variables["RELSTART"][:]
        relend_raw = f.variables["RELEND"][:]

        relstart = np.ma.asarray(relstart_raw, dtype=np.float64).filled(np.nan)
        relend = np.ma.asarray(relend_raw, dtype=np.float64).filled(np.nan)

        relative_seconds = (relstart + relend) / 2.0

        bad = ~np.isfinite(relative_seconds)
        if np.any(bad):
            raise ValueError(
                f"Invalid RELSTART/RELEND values at indices {np.where(bad)[0].tolist()}"
            )

        release_times = [
            ref_time + timedelta(seconds=float(sec))
            for sec in relative_seconds
        ]

    return pd.DataFrame({
        "release": np.arange(1, len(release_times) + 1, dtype=int),
        "release_time": pd.to_datetime(release_times),
    })


def get_release_info(
    *,
    part_ic: bool,
    grid_time_file: str | None = None,
    part_ic_file: str | None = None,
    part_ic_run_end: str | None = None,
) -> pd.DataFrame:
    """
    Unified entry point:
    - if part_ic=True  -> read actual release times from part_ic.nc
    - if part_ic=False -> read release times from combined grid_time file
    """
    if part_ic:
        return get_release_info_from_part_ic_file(
            part_ic_file=part_ic_file,
            part_ic_run_end=part_ic_run_end,
        )

    if grid_time_file is None:
        raise ValueError("grid_time_file is required when part_ic=False")

    return get_release_info_from_grid_time_file(grid_time_file)


# ============================================================
# OPEN FLUX FILE
# ============================================================
def _open_fluxes(flux_path: str, lv: float):
    flux_file = xr.open_dataset(flux_path)

    lhf_up = -flux_file["avg_slhtf"]
    shf_up = -flux_file["avg_ishf"]

    e_all = lhf_up / lv
    ss_all = shf_up

    return e_all, ss_all


# ============================================================
# CASE 1: PER-RELEASE grid_time*_python.nc FILES
# ============================================================
def compute_flux_term_multiple_files_df(
    folder: str,
    flux_path: str,
    lv: float = 2.2,
    cp: float = 1004.7,
) -> pd.DataFrame:
    folder = Path(folder)
    grid_files = sorted(folder.glob("grid_time*_python.nc"))
    if not grid_files:
        raise FileNotFoundError(f"No grid_time*_python.nc found in {folder}")

    e_all, ss_all = _open_fluxes(flux_path, lv)
    rows = []

    for grid_time_path in grid_files:
        tag = _get_tag(grid_time_path)
        release_time = _tag_to_datetime(tag)

        print(f"\nProcessing file: {grid_time_path.name}", flush=True)
        print(f"Release time: {release_time}", flush=True)

        with xr.open_dataset(grid_time_path) as ds:
            sen = ds["grid"]
            times = sen["time"].values

            e = e_all.sel(valid_time=times).rename({"valid_time": "time"})
            ss = ss_all.sel(valid_time=times).rename({"valid_time": "time"})

            dq = (e * sen) / 100.0
            dT = (ss * sen) / (cp * 100.0)

            dq_int = float(dq.sum(("time", "latitude", "longitude")).values)
            dT_int = float(dT.sum(("time", "latitude", "longitude")).values)
            dh = float(cp * dT_int + lv * dq_int)

        rows.append(
            {
                "release_time": release_time.to_pydatetime(),
                "MF_flux": dq_int,
                "SHF_flux": dT_int,
                "h_flux": dh,
            }
        )

    return pd.DataFrame(rows).sort_values("release_time").reset_index(drop=True)


# ============================================================
# CASE 2: ONE COMBINED grid_time_*.nc FILE
# ============================================================
def compute_flux_term_combined_file_df(
    grid_time_file: str,
    flux_path: str,
    lv: float = 2.2,
    cp: float = 1004.7,
    part_ic: bool = True,
    part_ic_file: str | None = None,
    part_ic_run_end: str | None = None,
):

    grid_time_file = Path(grid_time_file)

    release_info = get_release_info(
        part_ic=part_ic,
        grid_time_file=str(grid_time_file),
        part_ic_file=part_ic_file,
        part_ic_run_end=part_ic_run_end,
    )

    e_all, ss_all = _open_fluxes(flux_path, lv)
    rows = []

    sens_maps = []
    dq_maps = []
    dT_maps = []
    dh_maps = []
    release_list = []
    release_ids = []

    with xr.open_dataset(grid_time_file) as ds:
        nrel = ds.sizes["pointspec"]

        if len(release_info) != nrel:
            raise ValueError(
                f"Number of releases ({len(release_info)}) does not match pointspec ({nrel})"
            )

        print(f"grid_time file: {grid_time_file}", flush=True)
        print(f"part_ic = {part_ic}", flush=True)
        print(f"{nrel} releases in pointspec", flush=True)

        for i in range(nrel):
            print(f"\n[{i+1}/{nrel}] release", flush=True)

            sen_all = ds["spec001_mr"].isel(nageclass=0, pointspec=i)
            sen_100 = sen_all.sel(height=slice(None, 100))

            if sen_100.sizes.get("height", 0) == 0:
                raise ValueError(
                    f"No height levels <= 100 m found. Available heights: {sen_all['height'].values}"
                )

            # mean over 0–100 m
            sen = sen_100.mean("height")
            times = sen["time"].values

            release_id = int(release_info.iloc[i]["release"])
            release_time = pd.to_datetime(release_info.iloc[i]["release_time"])

            print(f"Release ID: {release_id}", flush=True)
            print(f"Release time: {release_time}", flush=True)

            e = e_all.sel(valid_time=times).rename({"valid_time": "time"})
            ss = ss_all.sel(valid_time=times).rename({"valid_time": "time"})

            dq = (e * sen) / 100.0
            dT = (ss * sen) / (cp * 100.0)

            # integrated scalar values for CSV
            dq_int = float(dq.sum(("time", "latitude", "longitude")).values)
            dT_int = float(dT.sum(("time", "latitude", "longitude")).values)
            dh = float(cp * dT_int + lv * dq_int)

            rows.append(
                {
                    "release": release_id,
                    "release_time": release_time.to_pydatetime(),
                    "MF_flux": dq_int,
                    "SHF_flux": dT_int,
                    "h_flux": dh,
                }
            )

            # 2D maps summed over time
            sens_map = sen.sum("time")
            dq_map = dq.sum("time")
            dT_map = dT.sum("time")
            dh_map = cp * dT_map + lv * dq_map

            sens_maps.append(sens_map)
            dq_maps.append(dq_map)
            dT_maps.append(dT_map)
            dh_maps.append(dh_map)
            release_list.append(release_time.to_datetime64())
            release_ids.append(release_id)

    # concatenate into 3D arrays: release, latitude, longitude
    sens_3d = xr.concat(sens_maps, dim="release")
    dq_3d = xr.concat(dq_maps, dim="release")
    dT_3d = xr.concat(dT_maps, dim="release")
    dh_3d = xr.concat(dh_maps, dim="release")

    sens_3d = sens_3d.assign_coords(
        release=("release", release_ids),
        release_time=("release", release_list),
    ).rename("sensitivity_map")

    dq_3d = dq_3d.assign_coords(
        release=("release", release_ids),
        release_time=("release", release_list),
    ).rename("dq_map")

    dT_3d = dT_3d.assign_coords(
        release=("release", release_ids),
        release_time=("release", release_list),
    ).rename("dT_map")

    dh_3d = dh_3d.assign_coords(
        release=("release", release_ids),
        release_time=("release", release_list),
    ).rename("dh_map")

    ds_maps = xr.Dataset(
        {
            "sensitivity_map": sens_3d,
            "dq_map": dq_3d,
            "dT_map": dT_3d,
            "dh_map": dh_3d,
        }
    )

    ds_maps["sensitivity_map"].attrs["description"] = "0-100 m mean sensitivity summed over time"
    ds_maps["dq_map"].attrs["description"] = "Moisture-flux contribution summed over time"
    ds_maps["dT_map"].attrs["description"] = "Sensible-heat contribution in K summed over time"
    ds_maps["dh_map"].attrs["description"] = "Total enthalpy-like contribution summed over time"

    return pd.DataFrame(rows).sort_values("release_time").reset_index(drop=True), ds_maps


# ============================================================
# TOP-LEVEL RUN FUNCTION
# ============================================================
def run_flux_term_to_csv(
    folder: str,
    flux_path: str,
    out_dir: str | None = None,
    lv: float = 2.2,
    cp: float = 1004.7,
    part_ic: bool = True,
    part_ic_file: str | None = None,
    part_ic_run_end: str | None = None,
) -> str:
    folder = Path(folder).resolve()

    out_path = Path(out_dir).resolve() if out_dir is not None else folder
    out_path.mkdir(parents=True, exist_ok=True)

    # Case A: many per-release files
    python_files = sorted(folder.glob("grid_time*_python.nc"))
    if python_files:
        print(f"Detected per-release grid_time*_python.nc files in {folder}", flush=True)

        df = compute_flux_term_multiple_files_df(
            folder=str(folder),
            flux_path=flux_path,
            lv=lv,
            cp=cp,
        )

        csv_path = out_path / f"flux_term_{folder.name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}", flush=True)
        return str(csv_path)

    # Case B: one combined file
    grid_time_files = sorted(folder.glob("grid_time_*.nc"))
    if len(grid_time_files) != 1:
        raise RuntimeError(
            f"Need either per-release grid_time*_python.nc files "
            f"or exactly one combined grid_time_*.nc file in {folder}"
        )

    print(f"Detected combined grid_time file: {grid_time_files[0]}", flush=True)
    print(f"Running with part_ic = {part_ic}", flush=True)

    df, ds_maps = compute_flux_term_combined_file_df(
        grid_time_file=str(grid_time_files[0]),
        flux_path=flux_path,
        lv=lv,
        cp=cp,
        part_ic=part_ic,
        part_ic_file=part_ic_file,
        part_ic_run_end=part_ic_run_end,
    )

    csv_path = out_path / f"flux_term_{folder.name}.csv"
    nc_path = out_path / f"flux_maps_{folder.name}.nc"

    df.to_csv(csv_path, index=False)
    ds_maps.to_netcdf(nc_path)

    print(f"Saved CSV: {csv_path}", flush=True)
    print(f"Saved maps: {nc_path}", flush=True)

    return str(csv_path)