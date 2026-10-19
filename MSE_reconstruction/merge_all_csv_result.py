




import os
import pandas as pd
from pathlib import Path
from make_hourly_and_merge import standardize_time, hourly_mean

FLEXPART_DIR = Path(os.environ["flexpart_dir"])

def latest_or_none(pattern):
    files = sorted(FLEXPART_DIR.glob(pattern))
    return files[-1] if files else None

rec_csv = latest_or_none("rec_init_*.csv")
flux_csv = latest_or_none("flux_term_*.csv")
ic_grid_csv = FLEXPART_DIR / "ic_from_grid_initial.csv"

OUT = FLEXPART_DIR / "merged_hourly.csv"

dfs_to_merge = []

# -----------------------------
# flux
# -----------------------------
if flux_csv is not None:
    df_flux = standardize_time(pd.read_csv(flux_csv))
    dfs_to_merge.append(df_flux)
    print("Used:", flux_csv.name)
else:
    print("WARNING: flux_term_*.csv not found, skipping flux.")

# -----------------------------
# receptor values from part_ic
# -----------------------------
if rec_csv is not None:
    df_part = standardize_time(pd.read_csv(rec_csv))

    part_cols = [
        c for c in df_part.columns
        if c.startswith("IC_") or c.startswith("rec_")
    ]

    if part_cols:
        df_part_hour = hourly_mean(df_part, part_cols)
        dfs_to_merge.append(df_part_hour)
        print("Used:", rec_csv.name)
    else:
        print("WARNING: rec_init file found but no IC_/rec_ columns found, skipping.")
else:
    print("WARNING: rec_init_*.csv not found, skipping receptor file.")

# -----------------------------
# grid initial
# -----------------------------
if ic_grid_csv.exists():
    df_ic = standardize_time(pd.read_csv(ic_grid_csv))

    ic_cols = [
        c for c in df_ic.columns
        if c.startswith("IC_")
    ]

    if ic_cols:
        df_ic_hour = hourly_mean(df_ic, ic_cols)
        dfs_to_merge.append(df_ic_hour)
        print("Used:", ic_grid_csv.name)
    else:
        print("WARNING: ic_from_grid_initial.csv found but no IC_ columns found, skipping.")
else:
    print("WARNING: ic_from_grid_initial.csv not found, skipping grid IC.")

# -----------------------------
# merge whatever exists
# -----------------------------
if not dfs_to_merge:
    raise RuntimeError("No CSV files found to merge.")

df = dfs_to_merge[0]

for other in dfs_to_merge[1:]:
    df = df.merge(other, on="release_time", how="outer")

df = df.sort_values("release_time")
df.to_csv(OUT, index=False)

print("Wrote:", OUT)