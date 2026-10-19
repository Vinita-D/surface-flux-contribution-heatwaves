import os
from pathlib import Path
import part_ic_rec_masked as fs # This is your main logic script

# =========================================================================
#  EDIT THIS SECTION TO CONFIGURE THE CALCULATION ---
# =========================================================================

# 'single_level' or 'layer_average'
REC_MODE = 'single_level'
REC_Z_TOP = 100.0
REC_NUM_LEVELS = 5 # (e.g., 3 for 0, 25, 50, 75,100)

FLEXPART_DIR = Path(os.environ["flexpart_dir"]).expanduser().resolve()
PART_IC_FILE = Path(os.environ["part_ic_file"]).expanduser().resolve()
PART_IC_RUN_END = os.environ["part_ic_run_end"]


if REC_MODE == 'single_level':
    out_csv_name = "rec_init_single_level.csv"
else:
    out_csv_name = f"rec_init_0-{int(REC_Z_TOP)}m_avg.csv"

out_csv = FLEXPART_DIR / out_csv_name


print(f"Starting receptor calculation with mode: {REC_MODE}", flush=True)
if REC_MODE == 'layer_average':
    print(f"  -> Layer: 0-{REC_Z_TOP} m, using {REC_NUM_LEVELS} levels", flush=True)
    
df = fs.run_to_csv(
    part_ic_file=str(PART_IC_FILE),
    run_end=PART_IC_RUN_END,
    out_csv=str(out_csv),
    mode=REC_MODE,
    z_top=REC_Z_TOP,
    num_levels=REC_NUM_LEVELS,
)

print("\nWrote:", out_csv, flush=True)
print(df.head(), flush=True)