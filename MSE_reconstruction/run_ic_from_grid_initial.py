import os
from pathlib import Path
import initial_conditions_grid_initial as fs

folder = Path(os.environ["flexpart_dir"]).expanduser().resolve()
PART_IC_FILE = Path(os.environ["part_ic_file"]).expanduser().resolve()
PART_IC_RUN_END = os.environ["part_ic_run_end"]

csv_path = fs.process_grid_initial_to_csv(
    folder=folder,
    out_fields_dir=None,
    csv_name="ic_from_grid_initial.csv",
    maps_nc_name="ic_maps_from_grid_initial.nc",
    part_ic=True,
    age_days=10,  # ignored when part_ic=True
    part_ic_file=str(PART_IC_FILE),
    part_ic_run_end=PART_IC_RUN_END,
)

print("Wrote:", csv_path, flush=True)