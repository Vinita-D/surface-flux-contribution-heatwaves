import os
import re
import time
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import contextmanager

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr

import metview as mv
import metpy.calc as mpcalc
from metpy.units import units

from numba import njit, prange, set_num_threads

from eccodes import (
    codes_grib_new_from_file,
    codes_get_array,
    codes_release,
)

from netCDF4 import Dataset


g = 9.80665
Rd = 287.05
Cp = 1004.0
Lv = 2.2e6

BASE_ERA5 = "/jetfs/shared-data/ECMWF/ERA5_glob_0.5deg_1h"


@contextmanager
def timer(label: str):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"[TIMER] {label}: {dt:.2f} s", flush=True)


def tag_from_grid_initial(grid_initial_nc: str) -> str:
    m = re.search(r"grid_initial_(\d{14})", Path(grid_initial_nc).name)
    if not m:
        raise ValueError(f"Could not parse timestamp from {grid_initial_nc}")
    return m.group(1)


def era5_datetime_from_grid_initial(grid_initial_nc: str) -> pd.Timestamp:
    with xr.open_dataset(grid_initial_nc) as ds:
        t = ds["time"].values
    return pd.to_datetime(np.asarray(t).item())


def era5_path_from_dt(dt: pd.Timestamp, base_dir=BASE_ERA5) -> str:
    y = dt.strftime("%Y")
    m = dt.strftime("%m")
    d = dt.strftime("%d")
    hh = dt.strftime("%H")
    return os.path.join(base_dir, y, m, f"EA{y}{m}{d}{hh}")


def read_hyai_hybi_from_grib(grib_path: str):
    with open(grib_path, "rb") as f:
        gid = codes_grib_new_from_file(f)
        if gid is None:
            raise RuntimeError("Empty GRIB file")
        pv = np.array(codes_get_array(gid, "pv"), dtype=float)
        codes_release(gid)

    n = pv.size // 2
    hyai = pv[:n]
    hybi = pv[n:2 * n]
    return hyai, hybi


def mv_to_da(fieldset, name: str) -> xr.DataArray:
    ds = fieldset.to_dataset()
    var = list(ds.data_vars.keys())[0]
    da = ds[var].rename(name)

    if "time" in da.dims:
        da = da.isel(time=0, drop=True)
    if "hybrid" in da.dims:
        da = da.rename({"hybrid": "lev"})
    if "level" in da.dims and "lev" not in da.dims:
        da = da.rename({"level": "lev"})

    return da


def get_lat_lon_names_from_grid(grid_ds):
    lat_name = "latitude" if "latitude" in grid_ds.coords else "lat"
    lon_name = "longitude" if "longitude" in grid_ds.coords else "lon"
    return lat_name, lon_name


def regrid_latlon_da(da3d: xr.DataArray, grid_initial_nc: str) -> xr.DataArray:
    with xr.open_dataset(grid_initial_nc) as grid:
        lat_name, lon_name = get_lat_lon_names_from_grid(grid)
        lat_g = grid[lat_name]
        lon_g = grid[lon_name]

    return da3d.interp(latitude=lat_g, longitude=lon_g, method="linear")


@njit(cache=True)
def interp_1d(xp, fp, x):
    n = xp.size
    if x <= xp[0]:
        return fp[0]
    if x >= xp[n - 1]:
        return fp[n - 1]

    for i in range(n - 1):
        if xp[i] <= x <= xp[i + 1]:
            x0 = xp[i]
            x1 = xp[i + 1]
            y0 = fp[i]
            y1 = fp[i + 1]
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    return fp[n - 1]


@njit(cache=True)
def bin_dp_weighted_numba(z_lev, x_lev, w_lev, H_tops):
    nH = H_tops.size
    out = np.full(nH, np.nan, dtype=np.float32)

    nlev = z_lev.size

    count = 0
    for i in range(nlev):
        if np.isfinite(z_lev[i]) and np.isfinite(x_lev[i]) and np.isfinite(w_lev[i]) and (w_lev[i] > 0.0):
            count += 1

    if count < 2:
        return out

    z = np.empty(count, dtype=np.float64)
    x = np.empty(count, dtype=np.float64)
    w = np.empty(count, dtype=np.float64)

    j = 0
    for i in range(nlev):
        if np.isfinite(z_lev[i]) and np.isfinite(x_lev[i]) and np.isfinite(w_lev[i]) and (w_lev[i] > 0.0):
            z[j] = z_lev[i]
            x[j] = x_lev[i]
            w[j] = w_lev[i]
            j += 1

    order = np.argsort(z)
    z = z[order]
    x = x[order]
    w = w[order]

    for k in range(nH):
        z0 = 0.0 if k == 0 else H_tops[k - 1]
        z1 = H_tops[k]

        sxw = 0.0
        sw = 0.0
        found = False

        for i in range(count):
            zz = z[i]
            if k == 0:
                inside = (zz >= z0) and (zz <= z1)
            else:
                inside = (zz > z0) and (zz <= z1)

            if inside:
                sxw += x[i] * w[i]
                sw += w[i]
                found = True

        if found and sw > 0.0:
            out[k] = np.float32(sxw / sw)
        else:
            zm = 0.5 * (z0 + z1)
            out[k] = np.float32(interp_1d(z, x, zm))

    return out


@njit(parallel=True, cache=True)
def bin_3d_columns(z3, x3, w3, H_tops):
    nlev, nlat, nlon = z3.shape
    nH = H_tops.size
    out = np.empty((nH, nlat, nlon), dtype=np.float32)

    for idx in prange(nlat * nlon):
        j = idx // nlon
        i = idx % nlon
        out[:, j, i] = bin_dp_weighted_numba(
            z3[:, j, i],
            x3[:, j, i],
            w3[:, j, i],
            H_tops,
        )

    return out


def to_lev_lat_lon_numpy(da: xr.DataArray) -> np.ndarray:
    lat_dim = "latitude" if "latitude" in da.dims else "lat"
    lon_dim = "longitude" if "longitude" in da.dims else "lon"
    return np.ascontiguousarray(da.transpose("lev", lat_dim, lon_dim).values)


def coords_from_da(da: xr.DataArray):
    lat_name = "latitude" if "latitude" in da.coords else "lat"
    lon_name = "longitude" if "longitude" in da.coords else "lon"
    return lat_name, lon_name, da[lat_name].values, da[lon_name].values


# ============================================================
# RELEASE-TIME HELPERS
# ============================================================
def release_times_from_grid_time_ncfile(grid_time_file: str):
    with Dataset(grid_time_file) as f:
        tag = Path(grid_time_file).stem.split("_")[-1]
        ref_time = datetime.strptime(tag, "%Y%m%d%H%M%S")

        relative_seconds = (f.variables["RELSTART"][:] + f.variables["RELEND"][:]) / 2.0
        release_times = [ref_time + timedelta(seconds=float(sec)) for sec in relative_seconds]

    return pd.to_datetime(release_times)


def release_times_from_part_ic_file(part_ic_file: str, part_ic_run_end: str):
    """
    Read actual release times from part_ic.nc:
        release_time = run_end + time_offset_seconds
    """
    if part_ic_file is None:
        raise ValueError("part_ic_file is required when part_ic=True")
    if part_ic_run_end is None:
        raise ValueError("part_ic_run_end is required when part_ic=True")

    part_ic_path = Path(part_ic_file)
    if not part_ic_path.exists():
        raise FileNotFoundError(f"part_ic file does not exist: {part_ic_path}")
    if not part_ic_path.is_file():
        raise ValueError(f"part_ic path is not a file: {part_ic_path}")

    run_end = pd.Timestamp(part_ic_run_end)

    with xr.open_dataset(part_ic_path, engine="netcdf4") as ds:
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
    return pd.to_datetime(release_df["release_time"].values)


def get_release_time(
    initial_time,
    time_index,
    part_ic: bool = True,
    age_days: int = 10,
    grid_time_file: str | None = None,
    part_ic_file: str | None = None,
    part_ic_run_end: str | None = None,
):
    """
    Get the release time corresponding to a combined-file time index.

    - if part_ic=False: use grid_time file
    - if part_ic=True : use actual part_ic.nc
    """
    if part_ic:
        release_times = release_times_from_part_ic_file(
            part_ic_file=part_ic_file,
            part_ic_run_end=part_ic_run_end,
        )
        return pd.to_datetime(release_times[time_index])

    if grid_time_file is None:
        raise ValueError("grid_time_file is required when part_ic=False")

    release_times = release_times_from_grid_time_ncfile(grid_time_file)
    return pd.to_datetime(release_times[time_index])


# ============================================================
# COMBINED-FILE VERSION
# ============================================================
def make_ic_from_combined_file(
    combined_grid_initial_nc: str,
    time_index: int,
    grid_time_file: str | None,
    out_fields_nc: str | None = None,
    part_ic: bool = True,
    age_days: int = 10,
    part_ic_file: str | None = None,
    part_ic_run_end: str | None = None,
):
    combined_grid_initial_nc = Path(combined_grid_initial_nc)

    with xr.open_dataset(combined_grid_initial_nc) as grid:
        initial_time = pd.to_datetime(grid["time"].isel(time=time_index).values)
        H = grid["lev"].values.astype(np.float64)
        w_init = grid["sensitivity"].isel(time=time_index).load()
        lat_g = grid["lat"]
        lon_g = grid["lon"]

    release_time = get_release_time(
        initial_time=initial_time,
        time_index=time_index,
        part_ic=part_ic,
        age_days=age_days,
        grid_time_file=grid_time_file,
        part_ic_file=part_ic_file,
        part_ic_run_end=part_ic_run_end,
    )

    print(f"Release time: {release_time}", flush=True)
    print(f"Initial time used for ERA5 GRIB: {initial_time}", flush=True)

    grib = era5_path_from_dt(initial_time)
    print(f"GRIB path: {grib}", flush=True)

    if not os.path.exists(grib):
        raise FileNotFoundError(grib)

    with timer("read GRIB from metview and compute geopotential at ML"):
        fs = mv.read(grib)
        t_fs = mv.read(data=fs, param="t")
        q_fs = mv.read(data=fs, param="q")
        sp_fs = mv.read(data=fs, param="sp")
        zsf_fs = mv.read(data=fs, param="z")
        phi_fs = mv.mvl_geopotential_on_ml(t_fs, q_fs, mv.log(sp_fs), zsf_fs)
        
        
        
    
        
        
        
        

    with timer("store variables from GRIB, compute Z_AGL"):
        T = mv_to_da(t_fs, "t")
        Q = mv_to_da(q_fs, "q")
        SP = mv_to_da(sp_fs, "sp")
        PHI_ML = mv_to_da(phi_fs, "phi_ml")
        PHI_SFC = mv_to_da(zsf_fs, "phi_sfc")

    Z_AGL = (PHI_ML / g) - (PHI_SFC / g)

    with timer("calculate dp and pressure mid-level"):
        hyai_1d, hybi_1d = read_hyai_hybi_from_grib(grib)
        hyai = xr.DataArray(hyai_1d, dims=("levi",))
        hybi = xr.DataArray(hybi_1d, dims=("levi",))

        P_int = hyai + hybi * SP
        dp = np.abs(P_int.diff("levi")).rename({"levi": "lev"})

        P_mid = 0.5 * (
            P_int.isel(levi=slice(0, -1)).rename({"levi": "lev"}) +
            P_int.isel(levi=slice(1, None)).rename({"levi": "lev"})
        )

    with timer("regrid ERA5 fields to combined grid"):
        Z_AGL_r = Z_AGL.interp(latitude=lat_g, longitude=lon_g, method="linear")
        T_r = T.interp(latitude=lat_g, longitude=lon_g, method="linear")
        Q_r = Q.interp(latitude=lat_g, longitude=lon_g, method="linear")
        dp_r = dp.interp(latitude=lat_g, longitude=lon_g, method="linear")
        P_mid_r = P_mid.interp(latitude=lat_g, longitude=lon_g, method="linear")

    with timer("prepare numpy arrays for numba"):
        Z_np = to_lev_lat_lon_numpy(Z_AGL_r).astype(np.float64, copy=False)
        T_np = to_lev_lat_lon_numpy(T_r).astype(np.float64, copy=False)
        Q_np = to_lev_lat_lon_numpy(Q_r).astype(np.float64, copy=False)
        dp_np = to_lev_lat_lon_numpy(dp_r).astype(np.float64, copy=False)
        P_np = to_lev_lat_lon_numpy(P_mid_r).astype(np.float64, copy=False)
        H_np = np.ascontiguousarray(H.astype(np.float64))

        lat_name, lon_name, lat_vals, lon_vals = coords_from_da(T_r)

    #with timer("bin T, q, p to FLEXPART levels"):
    
        #T_out = bin_3d_columns(Z_np, T_np, dp_np, H_np)
        #Q_out = bin_3d_columns(Z_np, Q_np, dp_np, H_np)
        #P_out = bin_3d_columns(Z_np, P_np, dp_np, H_np)

        #T_flex = xr.DataArray(
            #T_out,
            #dims=("lev", lat_name, lon_name),
            #coords={"lev": H, lat_name: lat_vals, lon_name: lon_vals},
            #name="t",
        #)
        #Q_flex = xr.DataArray(
            #Q_out,
            #dims=("lev", lat_name, lon_name),
            #coords={"lev": H, lat_name: lat_vals, lon_name: lon_vals},
            #name="q",
        #)
        #P_flex = xr.DataArray(
            #P_out,
            #dims=("lev", lat_name, lon_name),
            #coords={"lev": H, lat_name: lat_vals, lon_name: lon_vals},
            #name="p",
        #)

        #fields = xr.Dataset({"t": T_flex, "q": Q_flex, "p": P_flex})
    #new code block 
    with timer("bin T, q, p to FLEXPART levels"):
        T_out = bin_3d_columns(Z_np, T_np, dp_np, H_np)
        Q_out = bin_3d_columns(Z_np, Q_np, dp_np, H_np)
        P_out = bin_3d_columns(Z_np, P_np, dp_np, H_np)
        Z_eff_out = bin_3d_columns(Z_np, Z_np, dp_np, H_np)   # <-- ADD
        # ... existing T_flex, Q_flex, P_flex 
        T_flex = xr.DataArray(
            T_out,
            dims=("lev", lat_name, lon_name),
            coords={"lev": H, lat_name: lat_vals, lon_name: lon_vals},
            name="t",
        )
        Q_flex = xr.DataArray(
            Q_out,
            dims=("lev", lat_name, lon_name),
            coords={"lev": H, lat_name: lat_vals, lon_name: lon_vals},
            name="q",
        )
        P_flex = xr.DataArray(
            P_out,
            dims=("lev", lat_name, lon_name),
            coords={"lev": H, lat_name: lat_vals, lon_name: lon_vals},
            name="p",
        )

        Z_eff = xr.DataArray(                                  #new
            Z_eff_out,
            dims=("lev", lat_name, lon_name),
            coords={"lev": H, lat_name: lat_vals, lon_name: lon_vals},
            name="z_eff",
        )
        fields = xr.Dataset({"t": T_flex, "q": Q_flex, "p": P_flex,"z_eff": Z_eff})                   #new   
    

    with timer("compute theta and theta_e"):
        T_u = fields["t"].values * units.kelvin
        P_u = fields["p"].values * units.pascal
        q_u = fields["q"].values * units("kg/kg")

        theta = mpcalc.potential_temperature(P_u, T_u)
        r = mpcalc.mixing_ratio_from_specific_humidity(q_u)
        e = mpcalc.vapor_pressure(P_u, r)
        Td = mpcalc.dewpoint(e)
        theta_e = mpcalc.equivalent_potential_temperature(P_u, T_u, Td)

        fields["theta"] = (fields["t"].dims, theta.m.astype(np.float32))
        fields["theta_e"] = (fields["t"].dims, theta_e.m.astype(np.float32))

    if out_fields_nc is not None:
        with timer("write full 3D fields to netcdf"):
            fields.to_netcdf(out_fields_nc)
        print("Wrote fields:", out_fields_nc, flush=True)

    with timer("couple ERA5 fields to sensitivity weights"):
        if "lat" in w_init.dims and lat_name == "latitude":
            w_init = w_init.rename({"lat": "latitude"})
        if "lon" in w_init.dims and lon_name == "longitude":
            w_init = w_init.rename({"lon": "longitude"})

        sum_dims = ("lev", lat_name, lon_name)
        denom = w_init.sum(sum_dims)

        
        # --- layer mid-points instead of tops ---
        H_tops = w_init["lev"].values.astype(float)
        H_bots = np.concatenate([[0.0], H_tops[:-1]])
        H_mid  = 0.5 * (H_bots + H_tops)
        z_mid  = xr.DataArray(H_mid, dims=["lev"], coords={"lev": w_init["lev"]})
        
        IC_T = ((fields["t"] * w_init).sum(sum_dims) / denom).item()
        IC_q = ((fields["q"] * w_init).sum(sum_dims) / denom).item()
        IC_theta = ((fields["theta"] * w_init).sum(sum_dims) / denom).item()
        IC_theta_e = ((fields["theta_e"] * w_init).sum(sum_dims) / denom).item()
        #IC_z = ((w_init * w_init["lev"]).sum(sum_dims) / denom).item()
        IC_z = ((w_init * z_mid).sum(sum_dims) / denom).item()


    IC_mse = Cp * IC_T + Lv * IC_q + g * IC_z
    
    
    with timer("couple ERA5 fields to sensitivity weights"):
        if "lat" in w_init.dims and lat_name == "latitude":
            w_init = w_init.rename({"lat": "latitude"})
        if "lon" in w_init.dims and lon_name == "longitude":
            w_init = w_init.rename({"lon": "longitude"})

        sum_dims = ("lev", lat_name, lon_name)
        denom = w_init.sum(sum_dims)

        IC_T = ((fields["t"] * w_init).sum(sum_dims) / denom).item()
        IC_q = ((fields["q"] * w_init).sum(sum_dims) / denom).item()
        # ... theta, theta_e ...

        # --- mid-point version (your fix) ---
        H_tops = w_init["lev"].values.astype(float)
        H_bots = np.concatenate([[0.0], H_tops[:-1]])
        z_mid = xr.DataArray(0.5 * (H_bots + H_tops), dims=["lev"],
                             coords={"lev": w_init["lev"]})
        IC_z = ((w_init * z_mid).sum(sum_dims) / denom).item()

        # --- dp-weighted version (consistent with how T and q were binned) ---
        IC_z_eff = ((fields["z_eff"] * w_init).sum(sum_dims) / denom).item()

        print(f"IC_z (geometric mid) : {IC_z:8.1f} m")
        print(f"IC_z (dp-weighted)   : {IC_z_eff:8.1f} m")
        print(f"difference           : {IC_z - IC_z_eff:8.1f} m "
              f"= {9.81*(IC_z - IC_z_eff):6.0f} J/kg")

    with timer("build 2D weighted IC maps"):
        denom_2d = w_init.sum("lev")

        IC_T_map = ((fields["t"] * w_init).sum("lev") / denom_2d).rename("IC_T_map")
        IC_q_map = ((fields["q"] * w_init).sum("lev") / denom_2d).rename("IC_q_map")
        IC_theta_map = ((fields["theta"] * w_init).sum("lev") / denom_2d).rename("IC_theta_map")
        IC_theta_e_map = ((fields["theta_e"] * w_init).sum("lev") / denom_2d).rename("IC_theta_e_map")
        #IC_z_map = ((w_init * w_init["lev"]).sum("lev") / denom_2d).rename("IC_z_map")
        IC_z_map = ((w_init * z_mid).sum("lev") / denom_2d).rename("IC_z_map")
        IC_mse_map = (Cp * IC_T_map + Lv * IC_q_map + g * IC_z_map).rename("IC_mse_map")
        sens_column_map = denom_2d.rename("sensitivity_column")

        maps_ds = xr.Dataset({
            "IC_T_map": IC_T_map,
            "IC_q_map": IC_q_map,
            "IC_theta_map": IC_theta_map,
            "IC_theta_e_map": IC_theta_e_map,
            "IC_z_map": IC_z_map,
            "IC_mse_map": IC_mse_map,
            "sensitivity_column": sens_column_map,
        })

    result = {
        "release_time": pd.to_datetime(release_time).to_pydatetime(),
        "IC_T_grid": float(IC_T),
        "IC_q_grid": float(IC_q),
        "IC_theta_grid": float(IC_theta),
        "IC_theta_e_grid": float(IC_theta_e),
        "IC_z_grid": float(IC_z),
        "IC_mse_grid": float(IC_mse),
    }

    return result, maps_ds


def process_combined_file_to_csv(
    combined_grid_initial_nc: str,
    out_fields_dir: str | None = None,
    csv_name: str = "ic_from_grid_initial.csv",
    maps_nc_name: str = "ic_maps_from_grid_initial.nc",
    part_ic: bool = True,
    age_days: int = 10,
    part_ic_file: str | None = None,
    part_ic_run_end: str | None = None,
):
    combined_grid_initial_nc = Path(combined_grid_initial_nc)

    grid_time_file = None
    if not part_ic:
        grid_time_files = sorted(combined_grid_initial_nc.parent.glob("grid_time_*.nc"))
        if len(grid_time_files) != 1:
            raise RuntimeError(f"Expected 1 grid_time_*.nc file, found {len(grid_time_files)}")
        grid_time_file = str(grid_time_files[0])

    with xr.open_dataset(combined_grid_initial_nc) as ds:
        ntimes = ds.sizes["time"]

    print(f"Found combined file: {combined_grid_initial_nc}", flush=True)
    print(f"Found {ntimes} releases in time dimension", flush=True)
    print(f"part_ic = {part_ic}", flush=True)
    if grid_time_file is not None:
        print(f"Using grid_time file: {grid_time_file}", flush=True)
    if part_ic:
        print(f"Using part_ic file: {part_ic_file}", flush=True)
        print(f"Using part_ic run end: {part_ic_run_end}", flush=True)

    rows = []
    map_list = []
    release_times = []

    for i in range(ntimes):
        print(f"\n[{i+1}/{ntimes}] Processing combined-file release", flush=True)
        try:
            out_nc = None
            if out_fields_dir is not None:
                out_fields_dir_path = Path(out_fields_dir)
                out_fields_dir_path.mkdir(parents=True, exist_ok=True)
                out_nc = str(out_fields_dir_path / f"combined_release_{i:03d}_fields.nc")

            result, maps_ds = make_ic_from_combined_file(
                combined_grid_initial_nc=str(combined_grid_initial_nc),
                time_index=i,
                grid_time_file=grid_time_file,
                out_fields_nc=out_nc,
                part_ic=part_ic,
                age_days=age_days,
                part_ic_file=part_ic_file,
                part_ic_run_end=part_ic_run_end,
            )

            rows.append(result)
            map_list.append(maps_ds)
            release_times.append(np.datetime64(result["release_time"]))
            print("OK", flush=True)

        except Exception as e:
            print(f"FAILED in time index {i}", flush=True)
            print("Reason:", repr(e), flush=True)

    if not rows:
        raise RuntimeError("No releases were processed successfully.")

    df = pd.DataFrame(rows).sort_values("release_time").reset_index(drop=True)
    out_csv = combined_grid_initial_nc.parent / csv_name
    df.to_csv(out_csv, index=False)
    print("Wrote CSV:", out_csv, flush=True)

    maps_3d = xr.concat(map_list, dim="release")
    maps_3d = maps_3d.assign_coords(
        release=np.arange(1, len(map_list) + 1),
        release_time=("release", release_times),
    )

    out_nc_maps = combined_grid_initial_nc.parent / maps_nc_name
    maps_3d.to_netcdf(out_nc_maps)
    print("Wrote maps:", out_nc_maps, flush=True)

    return str(out_csv)


def process_grid_initial_to_csv(
    folder: str,
    out_fields_dir: str | None = None,
    csv_name: str = "ic_from_grid_initial.csv",
    maps_nc_name: str = "ic_maps_from_grid_initial.nc",
    part_ic: bool = True,
    age_days: int = 10,
    part_ic_file: str | None = None,
    part_ic_run_end: str | None = None,
):
    folder = Path(folder)

    combined_files = sorted(folder.glob("grid_initial_001.nc"))
    if combined_files:
        if len(combined_files) > 1:
            raise RuntimeError(f"Expected only one combined file, found {len(combined_files)}")
        return process_combined_file_to_csv(
            combined_grid_initial_nc=str(combined_files[0]),
            out_fields_dir=out_fields_dir,
            csv_name=csv_name,
            maps_nc_name=maps_nc_name,
            part_ic=part_ic,
            age_days=age_days,
            part_ic_file=part_ic_file,
            part_ic_run_end=part_ic_run_end,
        )

    raise RuntimeError("This corrected version is intended for the combined grid_initial_001.nc workflow.")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=os.environ.get("flexpart_dir", "").strip())
    parser.add_argument("--csv", default="ic_from_grid_initial.csv")
    parser.add_argument("--maps_nc", default="ic_maps_from_grid_initial.nc")
    parser.add_argument("--out_fields_dir", default=None)
    parser.add_argument("--part_ic", action="store_true", help="Use actual part_ic.nc release times")
    parser.add_argument("--age_days", type=int, default=10)
    parser.add_argument("--part_ic_file", default=None)
    parser.add_argument("--part_ic_run_end", default=None)
    parser.add_argument(
        "--numba_threads",
        type=int,
        default=0,
        help="Number of threads for numba parallel region; 0 means numba default",
    )
    args = parser.parse_args()

    if not args.folder:
        raise RuntimeError("Provide --folder or set env var flexpart_dir")

    if args.numba_threads > 0:
        set_num_threads(args.numba_threads)
        print(f"Using Numba threads: {args.numba_threads}", flush=True)
    else:
        print("Using Numba default thread count", flush=True)

    print(f"Running with part_ic = {args.part_ic}", flush=True)

    process_grid_initial_to_csv(
        folder=args.folder,
        out_fields_dir=args.out_fields_dir,
        csv_name=args.csv,
        maps_nc_name=args.maps_nc,
        part_ic=args.part_ic,
        age_days=args.age_days,
        part_ic_file=args.part_ic_file,
        part_ic_run_end=args.part_ic_run_end,
    )


if __name__ == "__main__":
    main()