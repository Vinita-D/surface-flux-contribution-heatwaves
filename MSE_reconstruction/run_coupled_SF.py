import os
from pathlib import Path
import coupled_SF as fs

FLEXPART_DIR = Path(os.environ["flexpart_dir"]).expanduser().resolve()
FLUX_FILE = Path(os.environ["flux_file"]).expanduser().resolve()
PART_IC_FILE = Path(os.environ["part_ic_file"]).expanduser().resolve()
PART_IC_RUN_END = os.environ["part_ic_run_end"]

OUT_DIR = FLEXPART_DIR
PART_IC = True

csv_path = fs.run_flux_term_to_csv(
    folder=str(FLEXPART_DIR),
    flux_path=str(FLUX_FILE),
    out_dir=str(OUT_DIR),
    part_ic=PART_IC,
    part_ic_file=str(PART_IC_FILE),
    part_ic_run_end=PART_IC_RUN_END,
)

print("Saved CSV to:", csv_path, flush=True)

if PART_IC:
    print("For combined grid_time runs, the NetCDF 2D maps are also saved in:", OUT_DIR, flush=True)