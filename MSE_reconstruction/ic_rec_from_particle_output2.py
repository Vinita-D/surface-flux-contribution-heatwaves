import os
import time
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from multiprocessing import get_context

from metpy.units import units
import metpy.calc as mpcalc

Cp = 1004.0
Lv = 2.2e6
g = 9.80665
#timer
def _print_timer(label, t0):
    print(f"[TIMER] {label}: {time.perf_counter() - t0:.2f} s", flush=True)


#####################################################
# first valid value along time 
###################################################
def first_valid_value(arr2d):
    arr = arr2d[:, ::-1]   #  flip the axis for backward run
    valid = np.isfinite(arr)
    has_any = valid.any(axis=1)

    first_idx = valid.argmax(axis=1)
    out = np.full(arr.shape[0], np.nan, dtype=np.float64)

    ii = np.arange(arr.shape[0])
    out[has_any] = arr[ii[has_any], first_idx[has_any]]
    return out


###################################################
# each release conditioned to avaiable partoutput file 
###################################################
def compute_one_release(partinit, partoutput, idx, rt):
    # same particle indices for partinit
    pi_sel = partinit.isel(particle=idx)
    # receptor values from partinit
    sh = pi_sel["sh"].values.astype(np.float64)
    T = pi_sel["T"].values.astype(np.float64)
    z = pi_sel["z"].values.astype(np.float64)
    p = pi_sel["prs"].values.astype(np.float64)

    rec_q = float(np.nanmean(sh))
    rec_T = float(np.nanmean(T))
    rec_z = float(np.nanmean(z))
    rec_mse = float(np.nanmean(Cp * T + Lv * sh + g * z))

    T_rec = T * units.kelvin
    p_rec = p * units.pascal
    q_rec = sh * units("kg/kg")

    r_rec = mpcalc.mixing_ratio_from_specific_humidity(q_rec)
    e_rec = mpcalc.vapor_pressure(p_rec, r_rec)
    Td_rec = mpcalc.dewpoint(e_rec)

    rec_theta = float(np.nanmean(mpcalc.potential_temperature(p_rec, T_rec).magnitude))
    rec_theta_e = float(np.nanmean(mpcalc.equivalent_potential_temperature(p_rec, T_rec, Td_rec).magnitude))

    # if no partoutput file found in the folder  skip IC compute
    if partoutput is None:
        return {
            "release_time": pd.to_datetime(rt),
            "npart": int(len(idx)),
            "ic_valid_particles": 0,

            "IC_q": np.nan,
            "IC_T": np.nan,
            "IC_z": np.nan,
            "IC_theta": np.nan,
            "IC_theta_e": np.nan,
            "IC_mse": np.nan,

            "rec_q": rec_q,
            "rec_T": rec_T,
            "rec_z": rec_z,
            "rec_theta": rec_theta,
            "rec_theta_e": rec_theta_e,
            "rec_mse": rec_mse,
        }

    # IC values from partoutput
    po_sel = partoutput.isel(particle=idx)

    sh2 = po_sel["sh"].values.astype(np.float64)
    T2 = po_sel["T"].values.astype(np.float64)
    z2 = po_sel["z"].values.astype(np.float64)
    p2 = po_sel["prs"].values.astype(np.float64)

    sh_ic = first_valid_value(sh2)
    T_ic = first_valid_value(T2)
    z_ic = first_valid_value(z2)
    prs_ic = first_valid_value(p2)

    ic_valid_mask = np.isfinite(sh_ic)
    ic_valid_particles = int(ic_valid_mask.sum())

    if ic_valid_particles > 0:
        IC_q = float(np.nanmean(sh_ic))
        IC_T = float(np.nanmean(T_ic))
        IC_z = float(np.nanmean(z_ic))
        IC_mse = float(np.nanmean(Cp * T_ic + Lv * sh_ic + g * z_ic))

        T_in = T_ic * units.kelvin
        p_in = prs_ic * units.pascal
        q_in = sh_ic * units("kg/kg")

        r_in = mpcalc.mixing_ratio_from_specific_humidity(q_in)
        e_in = mpcalc.vapor_pressure(p_in, r_in)
        Td_in = mpcalc.dewpoint(e_in)

        IC_theta = float(np.nanmean(mpcalc.potential_temperature(p_in, T_in).magnitude))
        IC_theta_e = float(np.nanmean(mpcalc.equivalent_potential_temperature(p_in, T_in, Td_in).magnitude))
    else:
        print('no valid IC found in the particle output file')
        IC_q = np.nan
        IC_T = np.nan
        IC_z = np.nan
        IC_theta = np.nan
        IC_theta_e = np.nan
        IC_mse = np.nan

    return {
        "release_time": pd.to_datetime(rt),
        "npart": int(len(idx)),
        "ic_valid_particles": ic_valid_particles,
        "IC_q": IC_q,
        "IC_T": IC_T,
        "IC_z": IC_z,
        "IC_theta": IC_theta,
        "IC_theta_e": IC_theta_e,
        "IC_mse": IC_mse,
        "rec_q": rec_q,
        "rec_T": rec_T,
        "rec_z": rec_z,
        "rec_theta": rec_theta,
        "rec_theta_e": rec_theta_e,
        "rec_mse": rec_mse,
    }



# For paralle computation
def _worker_release_batch(args):
    pi_path, po_path, release_times_batch, release_index = args

    partinit = xr.open_dataset(pi_path)

    if po_path is not None and Path(po_path).exists():
        partoutput = xr.open_dataset(po_path)
    else:
        partoutput = None
    rows = []
    for rt in release_times_batch:
        idx = release_index[rt]
        print(rt)
        rows.append(compute_one_release(partinit, partoutput, idx, rt))

    partinit.close()
    if partoutput is not None:
        partoutput.close()
    
    return rows





def compute_release_dataframe(pi_path, po_path=None, n_workers=1, batch_size=4):
    t0 = time.perf_counter()
    
    partinit = xr.open_dataset(pi_path)

    if po_path is not None and Path(po_path).exists():
        partoutput = xr.open_dataset(po_path)
        N = min(partinit.sizes["particle"], partoutput.sizes["particle"])
        partinit = partinit.isel(particle=slice(0, N))
        partoutput = partoutput.isel(particle=slice(0, N))
    else:
        partoutput = None
        N = partinit.sizes["particle"]
        partinit = partinit.isel(particle=slice(0, N))
        has_po = False
        print("No partoutput file found - computing receptor values ", flush=True)
    # assign release time per particle from partinit to partoutput 
    release_time_vals = partinit["time"].values
    release_times = np.unique(release_time_vals)

    # particle indices for each release
    release_index = {rt: np.where(release_time_vals == rt)[0] for rt in release_times}

    _print_timer("indexing/setup", t0)

    t1 = time.perf_counter()

    # terminal mode
    if n_workers == 1:
        rows = []
        for rt in release_times:
            print("release:", pd.to_datetime(rt), flush=True)
            idx = release_index[rt]
            rows.append(compute_one_release(partinit, partoutput, idx, rt))

        partinit.close()
        if partoutput is not None:
            partoutput.close()

    # parallel computaion  mode
    else:
        partinit.close()
        if partoutput is not None:
            partoutput.close()

        release_times_list = list(release_times)
        batches = [release_times_list[i:i + batch_size]
            for i in range(0, len(release_times_list), batch_size)]

        tasks = [(pi_path, po_path, batch, release_index) for batch in batches]

        rows = []
        ctx = get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            for out in pool.imap_unordered(_worker_release_batch, tasks):
                rows.extend(out)

    _print_timer("computation", t1)

    df_release = (
        pd.DataFrame(rows)
        .sort_values("release_time")
        .reset_index(drop=True)
    )

    return df_release





# =========================================================
# Hourly average from release-level table
# Weighted by particle counts
# =========================================================
def make_hourly_average(df_release):
    df = df_release.copy()
    df["release_hour"] = pd.to_datetime(df["release_time"]).dt.floor("h")

    value_cols = [
        "IC_q", "IC_T", "IC_z", "IC_theta", "IC_theta_e", "IC_mse",
        "rec_q", "rec_T", "rec_z", "rec_theta", "rec_theta_e", "rec_mse"
    ]

    rows = []
    for hr, sub in df.groupby("release_hour"):
        row = {
            "release_time": hr,
            "npart": int(sub["npart"].sum()),
            "ic_valid_particles": int(sub["ic_valid_particles"].sum()),
        }

        for col in value_cols:
            if col.startswith("IC_"):
                weights = sub["ic_valid_particles"].values.astype(float)
            else:
                weights = sub["npart"].values.astype(float)

            vals = sub[col].values.astype(float)
            mask = np.isfinite(vals) & np.isfinite(weights) & (weights > 0)

            if mask.any():
                row[col] = float(np.average(vals[mask], weights=weights[mask]))
            else:
                row[col] = np.nan

        rows.append(row)

    df_hourly = (
        pd.DataFrame(rows)
        .sort_values("release_time")
        .reset_index(drop=True)
    )
    return df_hourly



# =========================================================
# Save both CSVs
# =========================================================
def run_to_csv(pi_path, po_path=None, out_dir=None, n_workers=1, batch_size=4):
    pi_path = Path(pi_path)
    po_path = Path(po_path) if po_path is not None else None

    if out_dir is None:
        out_dir = pi_path.parent
    else:
        out_dir = Path(out_dir)

    df_release = compute_release_dataframe(
        pi_path=pi_path,
        po_path=po_path,
        n_workers=n_workers,
        batch_size=batch_size,
    )

    df_hourly = make_hourly_average(df_release)

    if po_path is not None:
        stem = f"rec_init_{po_path.stem}"
    else:
        stem = f"rec_init_{pi_path.stem}"

    out_release = out_dir / f"{stem}.csv"
    out_hourly = out_dir / f"{stem}_hourly.csv"

    df_release.to_csv(out_release, index=False)
    df_hourly.to_csv(out_hourly, index=False)

    print("wrote:", out_release, flush=True)
    print("wrote:", out_hourly, flush=True)

    return df_release, df_hourly