#!/bin/bash
#SBATCH --time=01-15:00:00
#SBATCH --job-name=1982_flux_attrs
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --mem=100G
#SBATCH --output=ic_grid_one_%j.log

module load micromamba/2.2.0



micromamba run -n zarr_env_micro python -u /jetfs/home/vdeshmukh/flexpart_flux_attribution/run_all_function.py
