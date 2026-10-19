import os
import gc
import time
import warnings
import argparse  # NEW: For command-line arguments


import numpy as np
import pandas as pd
import xarray as xr


import metview as mv
import metpy.calc as mpcalc
from metpy.units import units


from eccodes import (
    codes_grib_new_from_file,
    codes_get_array,
    codes_release,
)


warnings.filterwarnings("ignore", category=FutureWarning, module="cfgrib")


from pathlib import Path


# ============================================================
# TEMP DIRECTORY
# ============================================================


tmpdir = Path("/jetfs/scratch/vdeshmukh/metview_tmp")
tmpdir.mkdir(parents=True, exist_ok=True)


os.environ["TMPDIR"] = str(tmpdir)
os.environ["TMP"] = str(tmpdir)
os.environ["TEMP"] = str(tmpdir)


print("Using temp dir:", tmpdir, flush=True)


# ============================================================
# CONSTANTS
# ============================================================


G = 9.80665
CP = 1004.0
LV = 2.2e6
BASE_ERA5 = "/jetfs/shared-data/ECMWF/ERA5_glob_0.5deg_1h"


_HYAI_CACHE = None
_HYBI_CACHE = None


# ============================================================
# SMALL HELPERS
# ============================================================


def tic():
    return time.perf_counter()


def toc(t0, label):
    print(f"{label}: {time.perf_counter() - t0:.2f} s", flush=True)


def era5_path_from_dt(dt: pd.Timestamp, base_dir: str = BASE_ERA5) -> str:
    y = dt.strftime("%Y")
    m = dt.strftime("%m")
    d = dt.strftime("%d")
    h = dt.strftime("%H")
    return os.path.join(base_dir, y, m, f"EA{y}{m}{d}{h}")


# ============================================================
# HYBRID COEFFICIENTS
# ============================================================


def read_hyai_hybi_from_grib(grib_path: str):
    with open(grib_path, "rb") as f:
        gid = codes_grib_new_from_file(f)
        if gid is None:
            raise RuntimeError(f"Could not open GRIB: {grib_path}")
        pv = np.array(codes_get_array(gid, "pv"), dtype=float)
        codes_release(gid)


    n = pv.size // 2
    hyai = pv[:n]
    hybi = pv[n:]
    return hyai, hybi


def get_hybrid_coeffs(grib_path: str):
    global _HYAI_CACHE, _HYBI_CACHE
    if _HYAI_CACHE is None or _HYBI_CACHE is None:
        hyai, hybi = read_hyai_hybi_from_grib(grib_path)
        _HYAI_CACHE = xr.DataArray(hyai, dims=("levi",))
        _HYBI_CACHE = xr.DataArray(hybi, dims=("levi",))
    return _HYAI_CACHE, _HYBI_CACHE


# ============================================================
# METVIEW TO XARRAY
# ============================================================


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


# ============================================================
# LOAD ERA5 MODEL-LEVEL FIELDS
# ============================================================


def load_era5_hour(dt: pd.Timestamp):
    grib_path = era5_path_from_dt(dt)
    print(f"Reading ERA5: {grib_path}", flush=True)


    if not os.path.exists(grib_path):
        raise FileNotFoundError(f"ERA5 file not found: {grib_path}")


    fs = mv.read(grib_path)


    t_fs = mv.read(data=fs, param="t")
    q_fs = mv.read(data=fs, param="q")
    sp_fs = mv.read(data=fs, param="sp")
    z_sfc_fs = mv.read(data=fs, param="z")


    phi_ml_fs = mv.mvl_geopotential_on_ml(t_fs, q_fs, mv.log(sp_fs), z_sfc_fs)


    T = mv_to_da(t_fs, "T")
    Q = mv_to_da(q_fs, "Q")
    SP = mv_to_da(sp_fs, "SP")
    PHI_ML = mv_to_da(phi_ml_fs, "PHI_ML")
    PHI_SFC = mv_to_da(z_sfc_fs, "PHI_SFC")


    Z_AGL = (PHI_ML / G) - (PHI_SFC / G)


    hyai, hybi = get_hybrid_coeffs(grib_path)
    p_int = hyai + hybi * SP


    P_MID = 0.5 * (
        p_int.isel(levi=slice(0, -1)).rename({"levi": "lev"})
        + p_int.isel(levi=slice(1, None)).rename({"levi": "lev"})
    )


    return T, Q, P_MID, Z_AGL


# ============================================================
# INTERPOLATE 3D FIELD TO ONE HEIGHT
# ============================================================


def interp_to_height_2d(Z3d, X3d, target_z=50.0):
    Z = Z3d.values.astype(np.float64)
    X = X3d.values.astype(np.float64)


    nlev, nlat, nlon = Z.shape
    out = np.full((nlat, nlon), np.nan, dtype=np.float64)


    Zf = Z.reshape(nlev, -1)
    Xf = X.reshape(nlev, -1)


    for col in range(Zf.shape[1]):
        z_prof = Zf[:, col]
        x_prof = Xf[:, col]


        good = np.isfinite(z_prof) & np.isfinite(x_prof)
        if good.sum() < 2:
            continue


        z_prof = z_prof[good]
        x_prof = x_prof[good]


        order = np.argsort(z_prof)
        z_prof = z_prof[order]
        x_prof = x_prof[order]


        out.flat[col] = np.interp(target_z, z_prof, x_prof, left=np.nan, right=np.nan)


    return xr.DataArray(
        out,
        coords={"latitude": X3d["latitude"], "longitude": X3d["longitude"]},
        dims=("latitude", "longitude"),
    )


# ============================================================
# NEW: COMPUTE LAYER AVERAGE 2D FIELDS
# ============================================================


def compute_layer_average_fields(Z3d, T3d, Q3d, P3d, z_top, num_levels, z_bot=10.0):
    """
    Interpolates T, Q, and P to multiple heights from z_bot to z_top,
    then computes the vertical average.

    z_bot defaults to 10 m because ERA5's lowest model level is ~10 m AGL;
    interpolating below that returns NaN.
    """
    heights = np.linspace(z_bot, z_top, num_levels)
    print(f"Averaging over {len(heights)} levels from {z_bot} to {z_top} m: "
          f"{heights.round(1)}", flush=True)

    T_list, Q_list, P_list = [], [], []
    for z in heights:
        T_list.append(interp_to_height_2d(Z3d, T3d, target_z=z))
        Q_list.append(interp_to_height_2d(Z3d, Q3d, target_z=z))
        P_list.append(interp_to_height_2d(Z3d, P3d, target_z=z))

    T_avg = xr.concat(T_list, dim="height").mean("height", skipna=True)
    Q_avg = xr.concat(Q_list, dim="height").mean("height", skipna=True)
    P_avg = xr.concat(P_list, dim="height").mean("height", skipna=True)

    z_mean = float(np.mean(heights))          # <-- actual mean height used
    return T_avg, Q_avg, P_avg, z_mean


# ============================================================
# COMPUTE THERMO ON 2D FIELDS
# ============================================================


def compute_thermo_2d(T2d, q2d, p2d, z2d):
    T = T2d.values.astype(np.float64)
    q = q2d.values.astype(np.float64)
    p = p2d.values.astype(np.float64)
    z = z2d.values.astype(np.float64)


    T_u = T * units.kelvin
    q_u = q * units("kg/kg")
    p_u = p * units.pascal


    r = mpcalc.mixing_ratio_from_specific_humidity(q_u)
    e = mpcalc.vapor_pressure(p_u, r)
    Td = mpcalc.dewpoint(e)


    theta = mpcalc.potential_temperature(p_u, T_u).magnitude
    theta_e = mpcalc.equivalent_potential_temperature(p_u, T_u, Td).magnitude
    mse = CP * T + LV * q + G * z


    theta = xr.DataArray(theta, coords=T2d.coords, dims=T2d.dims)
    theta_e = xr.DataArray(theta_e, coords=T2d.coords, dims=T2d.dims)
    mse = xr.DataArray(mse, coords=T2d.coords, dims=T2d.dims)


    return theta, theta_e, mse


# ============================================================
# READ FULL part_ic CONTENT
# ============================================================


def read_part_ic_full(part_ic_file, run_end):
    run_end = pd.Timestamp(run_end)


    ds = xr.open_dataset(part_ic_file)


    kindz = int(ds.attrs.get("kindz", -999))
    if kindz != 1:
        ds.close()
        raise ValueError(f"part_ic kindz={kindz}, but this script expects kindz=1")


    df = pd.DataFrame({
        "longitude": ds["longitude"].values,
        "latitude": ds["latitude"].values,
        "height": ds["height"].values,
        "time": ds["time"].values.astype(np.int64),
        "release": ds["release"].values.astype(np.int64),
    })
    ds.close()


    df["release_time"] = run_end + pd.to_timedelta(df["time"], unit="s")


    heights = np.sort(df["height"].unique())
    if len(heights) != 1:
        # This is a warning, not an error, as we might override the height
        print(f"Warning: Expected one unique receptor height, got {heights}", flush=True)


    target_z_from_file = float(heights[0])


    release_df = (
        df[["release", "time", "release_time"]]
        .drop_duplicates()
        .sort_values("release")
        .reset_index(drop=True)
    )


    return df, release_df, target_z_from_file


# ============================================================
# BUILD RELEASE-SPECIFIC MASK ON ERA5 GRID
# ============================================================


def nearest_index_1d(grid, values):
    grid = np.asarray(grid)
    values = np.asarray(values)


    descending = grid[0] > grid[-1]
    if descending:
        grid2 = grid[::-1]
        idx = np.searchsorted(grid2, values, side="right")
        idx = np.clip(idx, 1, len(grid2) - 1)
        left = grid2[idx - 1]
        right = grid2[idx]
        choose_right = np.abs(values - right) <= np.abs(values - left)
        out = idx - 1 + choose_right.astype(int)
        return (len(grid) - 1) - out


    idx = np.searchsorted(grid, values, side="right")
    idx = np.clip(idx, 1, len(grid) - 1)
    left = grid[idx - 1]
    right = grid[idx]
    choose_right = np.abs(values - right) <= np.abs(values - left)
    return idx - 1 + choose_right.astype(int)


def build_mask_on_era5_grid(mask_df, lat_vals, lon_vals):
    jj = nearest_index_1d(lat_vals, mask_df["latitude"].to_numpy())
    ii = nearest_index_1d(lon_vals, mask_df["longitude"].to_numpy())


    mask2d = np.zeros((len(lat_vals), len(lon_vals)), dtype=bool)
    mask2d[jj, ii] = True


    return xr.DataArray(
        mask2d,
        coords={"latitude": lat_vals, "longitude": lon_vals},
        dims=("latitude", "longitude"),
    )


# ============================================================
# MAIN COMPUTATION (MODIFIED)
# ============================================================


def compute_release_means_2d(part_ic_file, run_end, mode, z_top, num_levels):
    part_df, release_df, target_z_from_file = read_part_ic_full(part_ic_file, run_end)


    if part_df.empty:
        raise ValueError("part_ic file contains no particles.")
    if release_df.empty:
        raise ValueError("No release times found in part_ic file.")
    
    # --- Determine calculation height/layer ---
    if mode == "single_level":
        print(f"Using single_level mode at z={target_z_from_file:.1f} m (from part_ic file)", flush=True)
    elif mode == "layer_average":
        print(f"Using layer_average mode for 0-{z_top:.1f} m using {num_levels} levels", flush=True)
    else:
        raise ValueError(f"Invalid mode: '{mode}'. Choose 'single_level' or 'layer_average'.")

    # load first ERA5 hour only to get grid
    first_release_time = pd.Timestamp(release_df.iloc[0]["release_time"])
    t0 = tic()
    T0, _, _, _ = load_era5_hour(first_release_time)
    toc(t0, f"Loaded ERA5 for {first_release_time} to get grid")


    lat_vals = T0["latitude"].values
    lon_vals = T0["longitude"].values
    del T0
    gc.collect()


    rows = []


    for row in release_df.itertuples(index=False):
        t_release = tic()


        release_id = int(row.release)
        release_sec = int(row.time)
        release_time = pd.Timestamp(row.release_time)


        print(f"\nProcessing release={release_id}, release_time={release_time}", flush=True)


        # Build mask for this release
        release_points = (
            part_df.loc[part_df["release"] == release_id, ["longitude", "latitude"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )


        if release_points.empty:
            print(f"Release {release_id} has no points. Skipping.", flush=True)
            continue


        mask2d = build_mask_on_era5_grid(release_points, lat_vals, lon_vals)
        ngrid = int(mask2d.sum().values)


        # Load ERA5 for this release time
        t0 = tic()
        T3d, Q3d, P3d, Z3d = load_era5_hour(release_time)
        toc(t0, f"Loaded ERA5 for {release_time}")


        # --- MODIFIED: Interpolate based on selected mode ---
        t0 = tic()
        if mode == 'single_level':
            target_z = target_z_from_file
            T_2d = interp_to_height_2d(Z3d, T3d, target_z=target_z)
            q_2d = interp_to_height_2d(Z3d, Q3d, target_z=target_z)
            p_2d = interp_to_height_2d(Z3d, P3d, target_z=target_z)
            z_2d = xr.full_like(T_2d, target_z)
            toc(t0, f"Built {target_z:.1f} m fields for release {release_id}")
        
        elif mode == 'layer_average':
            T_2d, q_2d, p_2d, z_mean = compute_layer_average_fields(
                Z3d, T3d, Q3d, P3d, z_top=z_top, num_levels=num_levels
            )
            z_2d = xr.full_like(T_2d, z_mean)     # was z_top / 2.0
            toc(t0, f"Built {z_mean:.1f} m layer-average fields for release {release_id}")


        # Compute thermo (this part is unchanged)
        t0 = tic()
        theta_2d, theta_e_2d, mse_2d = compute_thermo_2d(T_2d, q_2d, p_2d, z_2d)
        toc(t0, f"Computed thermo fields for release {release_id}")


        # Average over this release mask only
        rec_T = float(T_2d.where(mask2d).mean().values)
        rec_q = float(q_2d.where(mask2d).mean().values)
        rec_p = float(p_2d.where(mask2d).mean().values)
        rec_z = float(z_2d.where(mask2d).mean().values) # This will be the single height or the layer midpoint
        rec_theta = float(theta_2d.where(mask2d).mean().values)
        rec_theta_e = float(theta_e_2d.where(mask2d).mean().values)
        rec_mse = float(mse_2d.where(mask2d).mean().values)


        rows.append({
            "release": release_id,
            "release_time": release_time,
            "time_sec": release_sec,
            "ngrid": ngrid,
            "rec_T": rec_T,
            "rec_q": rec_q,
            "rec_p": rec_p,
            "rec_z": rec_z,
            "rec_theta": rec_theta,
            "rec_theta_e": rec_theta_e,
            "rec_mse": rec_mse,
        })


        del T3d, Q3d, P3d, Z3d
        del T_2d, q_2d, p_2d, z_2d
        del theta_2d, theta_e_2d, mse_2d
        del mask2d, release_points
        gc.collect()


        toc(t_release, f"Finished release {release_id}")


    return pd.DataFrame(rows).sort_values("release_time").reset_index(drop=True)


# ============================================================
# SAVE TO CSV (MODIFIED)
# ============================================================


def run_to_csv(part_ic_file, run_end, out_csv, mode, z_top, num_levels):
    df = compute_release_means_2d(part_ic_file, run_end, mode, z_top, num_levels)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}", flush=True)
    return df


# ============================================================
# RUN (MODIFIED with command-line parsing)
# ============================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute thermodynamic properties at FLEXPART receptor locations from a part_ic.nc file."
    )
    parser.add_argument(
        "--part_ic_file",
        type=str,
        required=True,
        help="Path to the part_ic.nc file."
    )
    parser.add_argument(
        "--run_end",
        type=str,
        required=True,
        help="End time of the FLEXPART run in 'YYYY-MM-DD HH:MM:SS' format."
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Path to save the output CSV file."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['single_level', 'layer_average'],
        default='single_level',
        help="Calculation mode: 'single_level' (uses height from part_ic file) or 'layer_average' (averages over a column)."
    )
    parser.add_argument(
        "--z_top",
        type=float,
        default=50.0,
        help="Top of the layer in meters for 'layer_average' mode (e.g., 50 or 100)."
    )
    parser.add_argument(
        "--num_levels",
        type=int,
        default=3,
        help="Number of interpolation levels for 'layer_average' mode (e.g., 3 for 0-50m, 5 for 0-100m)."
    )


    args = parser.parse_args()


    df = run_to_csv(
        part_ic_file=args.part_ic_file,
        run_end=args.run_end,
        out_csv=args.out_csv,
        mode=args.mode,
        z_top=args.z_top,
        num_levels=args.num_levels,
    )
    print("\n--- Output Head ---")
    print(df.head())