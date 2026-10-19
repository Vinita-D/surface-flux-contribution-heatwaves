import struct
import numpy as np
from pathlib import Path
import xarray as xr
import numpy as np
# ----------------------------
# Fortran record reader with prints
# ----------------------------
def read_record(f):
    head = f.read(4)
    if len(head) < 4:
        raise EOFError("EOF before record marker")
    n = struct.unpack("<i", head)[0]

    if n <= 0 or n > 10**8:
        raise EOFError(f"Suspicious record length {n}, stopping read_record")
    # --------------------------------------------------------------------
    payload = f.read(n)
    tail = f.read(4)
    if len(payload) < n or len(tail) < 4:
        raise EOFError("EOF inside record payload")
    n2 = struct.unpack("<i", tail)[0]
    if n2 != n:
        raise ValueError(f"Record marker mismatch: {n} vs {n2}")
    return payload



def read_i4(f, label=""):
    rec = read_record(f)
    if len(rec) < 4:
        raise EOFError(f"{label}: record <4 bytes")
    val = struct.unpack("<i", rec[:4])[0]
    #print(f"{label} int32 = {val} (record bytes={len(rec)})")
    return val

def read_i4arr(f, n, label=""):
    rec = read_record(f)
    arr = np.frombuffer(rec, dtype="<i4", count=n).copy()
    print(f"{label} int32 array n={n} (record bytes={len(rec)})")
    return arr

def read_f4arr(f, n, label=""):
    rec = read_record(f)
    arr = np.frombuffer(rec, dtype="<f4", count=n).copy()
    #print(f"{label} float32 array n={n} (record bytes={len(rec)})")
    return arr

# ------------------------------------------------------------------------------------
# Read header with prints - can be used for header(grid_initial) or header_grid_time
# ------------------------------------------------------------------------------------
def read_header_min(header_path):
    with open(header_path, "rb") as f:
        rec = read_record(f)
        ibdate, ibtime = struct.unpack("<ii", rec[:8])
        flexversion = rec[8:8+29].decode("ascii", errors="ignore").strip()

        rec = read_record(f)
        loutstep, loutaver, loutsample = struct.unpack("<iii", rec[:12])

        rec = read_record(f)
        bndx, bndy, numx, numy, delx, dely = struct.unpack("<ffii ff".replace(" ", ""), rec[:24])


        rec = read_record(f)
        numz = struct.unpack("<i", rec[:4])[0]
        outheight = np.frombuffer(rec[4:], dtype="<f4", count=numz).copy()
        # ----------------------------------

        _ = read_record(f)  # jjjjmmdd, ihmmss
        rec = read_record(f)  # nspec, numz
        nspec_raw, _ = struct.unpack("<ii", rec[:8])
        nspec = nspec_raw // 3

        for _ in range(nspec):
            read_record(f); read_record(f); read_record(f)

        rec = read_record(f)
        numpoint = struct.unpack("<i", rec[:4])[0]

        for _ in range(numpoint):
            read_record(f); read_record(f); read_record(f); read_record(f)
            for __ in range(nspec):
                read_record(f); read_record(f); read_record(f)

        read_record(f) 

        rec = read_record(f)
        nageclass = struct.unpack("<i", rec[:4])[0]
        lage = np.frombuffer(rec[4:], dtype="<i4", count=nageclass)
        ageclass = int(lage[0])

    if float(numx) * float(delx) == 360.0:
        nxshift = int((bndx + 180.0) / delx)
    else:
        nxshift = 0
    

    return dict(
        nx=numx, ny=numy, nz=numz,
        outheight=outheight,  
        bndx=bndx, bndy=bndy, delx=delx, dely=dely,
        loutstep=abs(int(loutstep)), ageclass=ageclass, nxshift=nxshift
    )

# ----------------------------
# Read grid_time 
# ----------------------------
def decode_sparse_2d(sparse_i, sparse_r, nx, ny):
    
    SMALL = np.finfo(np.float32).tiny
    g = np.zeros((ny, nx), np.float32)
    ii = 0
    fact = 1.0
    n = 0
    for r in sparse_r:
        if (r * fact) > SMALL:
            n = int(sparse_i[ii]); ii += 1
            fact *= -1.0
        else:
            n += 1
        jy = int((n - nx*ny) / nx)
        ix = int(n - nx*ny - nx*jy)
        if 0 <= ix < nx and 0 <= jy < ny:
            g[jy, ix] = abs(r)
    return g



def decode_sparse_gridtime_3d(sparse_i, sparse_r, nx, ny, nz): # same as 2d 
    SMALL = np.finfo(np.float32).tiny
    g = np.zeros((nz, ny, nx), dtype=np.float32)  # (z, y, x)

    ii = 0
    fact = 1.0
    n = 0

    for r in sparse_r:
        if (r * fact) > SMALL:
            n = int(sparse_i[ii]); ii += 1
            fact *= -1.0
        else:
            n += 1

        kz = int(n / (nx * ny))
        jy = int((n - kz * nx * ny) / nx)
        ix = int(n - kz * nx * ny - nx * jy)

        ix0 = ix
        jy0 = jy
        kz0 = kz - 1  # Fortran level index is 1-based

        if 0 <= ix0 < nx and 0 <= jy0 < ny and 0 <= kz0 < nz:
            g[kz0, jy0, ix0] = abs(r)

    return g




def read_grid_time_debug(bin_path, hdr, max_steps=None):
    nx, ny = hdr["nx"], hdr["ny"]
    nxshift = hdr["nxshift"]

    times = []
    grids = []

    with open(bin_path, "rb") as f:
        step = 0
        while True:
            if max_steps is not None and step >= max_steps:
                break

            try:
                yyyymmdd = read_i4(f, "yyyymmdd")
                hhmiss   = read_i4(f, "hhmiss")

                sp_i = read_i4(f, "sp_count_i")
                rec_i = read_record(f)
                sparse_i = (
                    np.frombuffer(rec_i, dtype="<i4", count=sp_i).copy()
                    if sp_i > 0 else np.empty(0, np.int32)
                )

                sp_r = read_i4(f, "sp_count_r")
                rec_r = read_record(f)
                sparse_r = (
                    np.frombuffer(rec_r, dtype="<f4", count=sp_r).copy()
                    if sp_r > 0 else np.empty(0, np.float32)
                )

            except (EOFError, ValueError) as e:
                # ValueError here most likely means broken / truncated last record.
                print(f"Stopping {bin_path.name} at step {step} due to {e}")
                break

            g2d = decode_sparse_2d(sparse_i, sparse_r, nx, ny)
            if nxshift:
                g2d = np.roll(g2d, shift=-nxshift, axis=1)

            print("grid stats: min", g2d.min(), "max", g2d.max(), "sum", g2d.sum())
            times.append((yyyymmdd, hhmiss))
            grids.append(g2d)
            step += 1

    print("\nRead timesteps:", len(grids))
    return times, grids




# ----------------------------
# Sparse decode for grid_init
# ----------------------------
def decode_sparse_3d(sparse_i, sparse_r, nx, ny, nz):
    SMALL = np.finfo(np.float32).tiny
    g = np.zeros((nz, ny, nx), dtype=np.float32)  # (z, y, x)

    ii = 0
    fact = 1.0
    n = 0

    for r in sparse_r:
        if (r * fact) > SMALL:
            n = int(sparse_i[ii]); ii += 1
            fact *= -1.0
        else:
            n += 1

        kz = int(n / (nx * ny))
        jy = int((n - kz * nx * ny) / nx)
        ix = int(n - kz * nx * ny - nx * jy)

        
        # So we subtract 1 to convert to Python 0-based:
        ix0 = ix
        jy0 = jy
        kz0 = kz - 1

        if 0 <= ix0 < nx and 0 <= jy0 < ny and 0 <= kz0 < nz:
            g[kz0, jy0, ix0] = abs(r) 

    return g

# ----------------------------
# Read grid_initial with prints
# ----------------------------
def read_grid_initial(bin_path, hdr):
    nx, ny = hdr["nx"], hdr["ny"]
    nz = hdr["nz"]
    nxshift = hdr["nxshift"]

    with open(bin_path, "rb") as f:
        try:
            yyyymmdd = read_i4(f, "yyyymmdd")
            hhmiss   = read_i4(f, "hhmiss")

            print(f"{bin_path.name} binary time = {yyyymmdd} {hhmiss:06d}")  # <--- HERE

            sp_i = read_i4(f, "sp_count_i")
            rec_i = read_record(f)
            sparse_i = np.frombuffer(rec_i, dtype="<i4", count=sp_i).copy() if sp_i > 0 else np.empty(0, np.int32)

            sp_r = read_i4(f, "sp_count_r")
            rec_r = read_record(f)
            sparse_r = np.frombuffer(rec_r, dtype="<f4", count=sp_r).copy() if sp_r > 0 else np.empty(0, np.float32)

        except EOFError:
            return None, None, np.zeros((nz, ny, nx), dtype=np.float32)

    g3d = decode_sparse_3d(sparse_i, sparse_r, nx, ny, nz)
    if nxshift:
        g3d = np.roll(g3d, shift=-nxshift, axis=2)

    return yyyymmdd, hhmiss, g3d


# --------------------------------------------------
# assigning time corrdinate for grid_time files
# --------------------------------------------------
def times_to_datetime64(times):
    # times is list of (yyyymmdd, hhmiss)
    out = []
    for yyyymmdd, hhmiss in times:
        yyyy = yyyymmdd // 10000
        mm   = (yyyymmdd % 10000) // 100
        dd   = (yyyymmdd % 100)
        hh   = hhmiss // 10000
        mi   = (hhmiss % 10000) // 100
        ss   = hhmiss % 100
        out.append(np.datetime64(f"{yyyy:04d}-{mm:02d}-{dd:02d}T{hh:02d}:{mi:02d}:{ss:02d}"))
    return np.array(out, dtype="datetime64[ns]")



# --------------------------------------------------
# to assign right time coordinate to grid_initial 
# --------------------------------------------------
def read_last_time_from_grid_time(grid_time_path: Path) -> np.datetime64:
    """Return the LAST (oldest) timestamp stored in a grid_time binary file."""
    last = None
    with open(grid_time_path, "rb") as f:
        while True:
            try:
                yyyymmdd = read_i4(f)
                hhmiss   = read_i4(f)

                sp_i = read_i4(f)
                _ = read_record(f)   
                sp_r = read_i4(f)
                _ = read_record(f)  
                last = (yyyymmdd, hhmiss)
            except EOFError:
                break
    return times_to_datetime64([last])[0]





# convert to xarray 
def make_xarray_from_grids(times, grids, hdr):
    if len(grids) == 0:
        raise ValueError("No grids to stack: this grid_time file has 0 timesteps.")
    data = np.stack(grids, axis=0).astype(np.float32)   # (time, ny, nx)
    t = times_to_datetime64(times)

    lon = (hdr["bndx"] + 0.5*hdr["delx"] + hdr["delx"] * np.arange(hdr["nx"])).astype(np.float32)
    if hdr["nxshift"]:
        lon = np.roll(lon, shift=-hdr["nxshift"])

    lat = (hdr["bndy"] + 0.5*hdr["dely"] + hdr["dely"] * np.arange(hdr["ny"])).astype(np.float32)

    ds = xr.Dataset(
        data_vars={"grid": (("time", "latitude", "longitude"), data)},
        coords={"time": t, "latitude": lat, "longitude": lon},
    )
    return ds


# ----------------------------
# For 4D grid_time files - a
# ----------------------------
def make_xarray_from_grid_initial(t_end_release, grid3d, hdr):
    lon = (hdr["bndx"] + 0.5*hdr["delx"] + hdr["delx"] * np.arange(hdr["nx"])).astype(np.float32)
    if hdr["nxshift"]:
        lon = np.roll(lon, shift=-hdr["nxshift"])
    lat = (hdr["bndy"] + 0.5*hdr["dely"] + hdr["dely"] * np.arange(hdr["ny"])).astype(np.float32)
    z   = hdr["outheight"].astype(np.float32)

    ds = xr.Dataset(
        data_vars={"grid": (("height","latitude","longitude"), grid3d.astype(np.float32))},
        coords={"time": t_end_release, "height": z, "latitude": lat, "longitude": lon},
    )
    ds["height"].attrs["units"] = "m"
    return ds



def read_grid_time_4d(bin_path, hdr, max_steps=None):
    nx, ny, nz = hdr["nx"], hdr["ny"], hdr["nz"]
    nxshift = hdr["nxshift"]

    times = []
    grids = []

    with open(bin_path, "rb") as f:
        step = 0
        while True:
            if max_steps is not None and step >= max_steps:
                break
            try:
                yyyymmdd = read_i4(f, "yyyymmdd")
                hhmiss   = read_i4(f, "hhmiss")

                sp_i = read_i4(f, "sp_count_i")
                rec_i = read_record(f)  # always consume
                sparse_i = (
                    np.frombuffer(rec_i, dtype="<i4", count=sp_i).copy()
                    if sp_i > 0 else np.empty(0, np.int32)
                )

                sp_r = read_i4(f, "sp_count_r")
                rec_r = read_record(f)  # always consume
                sparse_r = (
                    np.frombuffer(rec_r, dtype="<f4", count=sp_r).copy()
                    if sp_r > 0 else np.empty(0, np.float32)
                )

            except (EOFError, ValueError) as e:
                print(f"Stopping {bin_path.name} at step {step}: {e}")
                break

            g3d = decode_sparse_gridtime_3d(sparse_i, sparse_r, nx, ny, nz)
            if nxshift:
                g3d = np.roll(g3d, shift=-nxshift, axis=2)  # roll lon axis

            times.append((yyyymmdd, hhmiss))
            grids.append(g3d)
            step += 1

    print("Read timesteps (4D):", len(grids))
    return times, grids


def make_xarray_from_grids_4d(times, grids, hdr):
    data = np.stack(grids, axis=0).astype(np.float32)  # (time, z, y, x)
    t = times_to_datetime64(times)

    lon = (hdr["bndx"] + 0.5*hdr["delx"] + hdr["delx"] * np.arange(hdr["nx"])).astype(np.float32)
    if hdr["nxshift"]:
        lon = np.roll(lon, shift=-hdr["nxshift"])

    lat = (hdr["bndy"] + 0.5*hdr["dely"] + hdr["dely"] * np.arange(hdr["ny"])).astype(np.float32)
    z   = hdr["outheight"].astype(np.float32)

    ds = xr.Dataset(
        data_vars={"grid": (("time","height","latitude","longitude"), data)},
        coords={"time": t, "height": z, "latitude": lat, "longitude": lon},
    )
    ds["height"].attrs["units"] = "m"
    return ds



# ------------------------------------------------------------------------------------
# tAutomatcic loader to decide to use 3d spare and 2d spare decode
# ------------------------------------------------------------------------------------
def load_grid_time_dataset(bin_path: Path, hdr: dict, max_steps=None) -> xr.Dataset:
    """
    Read a FLEXPART grid_time_*_001 binary file as xarray.Dataset.
    - If hdr["nz"] == 1  -> interpret as 2D (time, lat, lon)
    - If hdr["nz"] > 1   -> interpret as 3D (time, height, lat, lon)
    """
    nz = hdr["nz"]

    if nz == 1:
        # 2D case: use your existing 2D reader
        print(f"{bin_path.name}: detected 2D (nz=1) grid_time")
        times, grids = read_grid_time_debug(bin_path, hdr, max_steps=max_steps)
        ds = make_xarray_from_grids(times, grids, hdr)
    else:
        # 3D case: use 4D reader
        print(f"{bin_path.name}: detected 3D (nz={nz}) grid_time")
        times, grids = read_grid_time_4d(bin_path, hdr, max_steps=max_steps)
        ds = make_xarray_from_grids_4d(times, grids, hdr)

    return ds

# ------------------------------------------------------------------------------------
# function to open the file and decode which debug spare code to use 
# ------------------------------------------------------------------------------------
def load_grid_time_dataset_with_fallback(bin_path: Path, hdr: dict, max_steps=None) -> xr.Dataset:
    nz = hdr["nz"]

    # If nz==1, stick with 2D
    if nz == 1:
        times, grids = read_grid_time_debug(bin_path, hdr, max_steps=max_steps)
        return make_xarray_from_grids(times, grids, hdr)

    # If nz>1, we *expect* 3D
    try:
        times, grids = read_grid_time_4d(bin_path, hdr, max_steps=max_steps)
        return make_xarray_from_grids_4d(times, grids, hdr)
    except (EOFError, ValueError) as e:
        print(f"3D read failed for {bin_path.name} ({e}), trying 2D reader as fallback.")
        # reopen implicitly inside read_grid_time_debug
        times, grids = read_grid_time_debug(bin_path, hdr, max_steps=max_steps)
        return make_xarray_from_grids(times, grids, hdr)





# ------------------------------------------------------------------------------------
# to obatain and assign correct date and time for grid_initial files 
# ------------------------------------------------------------------------------------
def time_from_grid_file_name(p: Path) -> np.datetime64:
    # e.g. grid_initial_20230717190000_001  or grid_time_20230717190000_001
    s = p.name.split("_")[2]          # '20230717190000'
    yyyymmdd = int(s[:8])
    hhmiss   = int(s[8:])

    yyyy = yyyymmdd // 10000
    mm   = (yyyymmdd % 10000) // 100
    dd   = yyyymmdd % 100
    hh   = hhmiss // 10000
    mi   = (hhmiss % 10000) // 100
    ss   = hhmiss % 100
    return np.datetime64(f"{yyyy:04d}-{mm:02d}-{dd:02d}T{hh:02d}:{mi:02d}:{ss:02d}")

# ------------------------------------------------------------------------------------
# Reading ageclass
# ------------------------------------------------------------------------------------
def read_lage_seconds(namelist_path: Path) -> int:

    with namelist_path.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith("LAGE"):
                # e.g. "LAGE=21600      ,"
                rhs = line.split("=", 1)[1]
                val = rhs.split(",")[0].strip()
                return int(val)