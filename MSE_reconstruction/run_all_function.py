import os
import sys
import subprocess
from pathlib import Path

# EDIT THESE PATHS FOR EACH SIMULATION
flexpart_dir = "/jetfs/scratch/vdeshmukh/1982_grecce_heatwave/output_summer/"
flux_file = "/jetfs/scratch/vdeshmukh/flux_ERA5/flux_files_res/ERA5_mean_fluxes_1982_MJJA_regridded_0p5.nc"
part_ic_file = "/jetfs/scratch/vdeshmukh/1982_grecce_heatwave/options/part_ic.nc"
part_ic_run_end = "1982-08-31 18:00:00"

REPO_ROOT = "/jetfs/home/vdeshmukh/flexpart_flux_attribution"
PYTHON = sys.executable


def run(cmd, label):
    print(f"\n{label}", flush=True)
    subprocess.run(cmd, check=True)


def main():
    repo_root = Path(REPO_ROOT).resolve()
    fp_dir = Path(flexpart_dir).resolve()
    flux = Path(flux_file).resolve()
    part_ic = Path(part_ic_file).resolve()

    if not repo_root.is_dir():
        raise FileNotFoundError(f"Repo root not found: {repo_root}")
    if not fp_dir.is_dir():
        raise FileNotFoundError(f"FLEXPART dir not found: {fp_dir}")
    if not flux.is_file():
        raise FileNotFoundError(f"Flux file not found: {flux}")
    if not part_ic.is_file():
        raise FileNotFoundError(f"part_ic file not found: {part_ic}")

    os.chdir(repo_root)

    os.environ["flexpart_dir"] = str(fp_dir)
    os.environ["flux_file"] = str(flux)
    os.environ["part_ic_file"] = str(part_ic)
    os.environ["part_ic_run_end"] = part_ic_run_end

    print("Repo root       :", repo_root, flush=True)
    print("FLEXPART_DIR    :", fp_dir, flush=True)
    print("FLUX_FILE       :", flux, flush=True)
    print("PART_IC_FILE    :", part_ic, flush=True)
    print("PART_IC_RUN_END :", part_ic_run_end, flush=True)


    # Comment out any step you want to skip
    run([PYTHON, "run_binary_to_ncdf.py"], "1) Convert binary files to NetCDF")
    run([PYTHON, "run_fix_grid_initial_time.py"], "1b) Fix grid_initial_001.nc time coordinate")
    #run([PYTHON, "run_ic_rec_from_particle.py"], "2) IC + receptor from particle output")
    run([PYTHON, "run_rec_partic.py"], "2) rece values")
    run([PYTHON, "run_coupled_SF.py"], "3) Couple fluxes")
    run([PYTHON, "run_ic_from_grid_initial.py"], "4) IC from grid_initial")
    run([PYTHON, "merge_all_csv_result.py"], "5) Merge CSV results")

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
