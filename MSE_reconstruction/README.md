# MSE reconstruction using SSR framework
Info: a simple tool for conducting Lagrangian diagonisis with FLEXPART transport model.
Flexpart model produces outputs based on tracking of airmass particle - particle outputs and gridded airmass sensitivity outputs : for more details refer https://flexpart.img.univie.ac.at/docs/ 
This model is further used to diagnose the influence of surface turbulent fluxes (heat and moisture) surface sensible and mositure fluxes on the airmass which evolves to become a heatwave airmass. 
thermodynamic budget of an air mass:
Initial thermodynamic state + surface flux contributions = conditions at the receptor
The receptor represents the location and time where airmass are released, which corresponding to the region on heatwave event or a weather extreme.

The objective is to quantify how surface sensible heat and moisture fluxes contribute to the development and intensification of heatwaves and other weather extremes.

This framework helps determine:
how much of the air-mass thermodynamic state is explained by initial conditions
how much is explained by surface heat and moisture fluxes along trajectories
 

## Installation 
Ideally install it in a virtual environment using micromamba
```bash
git clone https://gitlab.phaidra.org/vinitad98/flexpart_flux_attribution.git
cd flexpart_flux_attribution
```

## Environment Setup
```bash
micromamba env create -f environment.yml
micromamba activate flexpart_flux_attribution
```

### Option B — Python venv + pip

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Set input paths accordingly  for example :
```bash
export flexpart_dir=/jetfs/scratch/vdeshmukh/1hr_back/output_ncdf_5days/
export flux_file=/jetfs/scratch/vdeshmukh/ERA5_mean_fluxes_2023_JJ_1ded_reg.nc
```

# FLEXPART  set up  (can be worked with Linversion=1 or 0 mode)
LINVERSION=1 and SURF_ONLY=1 - check  test_1hr_backwards_run_2release/option/COMMAND 
---
Flexpart simulation output requirements:
- partinit_YYYYMMDDHHSSMM.nc ncdf file (receptor values)
- partoutput_YYYYMMDDHHSSMM_init.nc files or grid_initial_YYYYMMDDHHSSMM (for each release) 
- grid_time_YYYYMMDDHHSSMM 

# Codes  provides tools for:
- Converting FLEXPART binary output to NetCDF.  
- Compute initial conditions (IC) and receptor (rec) values using particle output files.  
- coupling of surface fluxes with FLEXPART grid_time files (converted ncdf).  
- All the intgrated values gets stored in csv file for each release.  
- Merge all the resulted values: flux_contribution, IC, (optional- IC_from grid_initial) and rec ) to hourly time resolution and generate plots  

sutiable HPC environments 
If using ERA5 GRIB files, additional dependencies such as `cfgrib`, `cdo` and `eccodes` is required.  
---


## Test run: `test_1hr_backwards_run_2release`

This folder contains a small test case check that the full post-processing runs correctly.

### Folder contents
- `test_1hr_backwards_run_2release/`
  - `flexpart_config/` : FLEXPART configuration files used for the test case (AVAILABLE, pathnames, options, run.job / slurm)
  - `post_processing/` : optional helper scripts for running this specific test case
  - `output/` : FLEXPART output directory. Create it locally and run FLEXPART there.

### Run flexpart (create a output dir)
```bash
sbatch ./run.job 
```

--- 

## run the post-processing - read the readme.txt /post_processing folder
Edit the run_all_function.py file with correct path and directory 4 steps of postprocessing.
Run in the terminal 
```bash
micromamba activate zarr_env_micro (name of you env)
python run_all_function.py
```
Run in parallel mode :
```bash
sbatch ./slurm_run_all_main.sh
```



## Workflow
### Convert FLEXPART binary to NetCDF

```bash
python run_binary_to_ncdf.py 
```
Core script:
- `function_binary_to_ncdf_flexpart.py`

---

### Compute Initial condition(IC) and receptor(rec) values (partinit / partoutput)

```bash
python run_ic_rec_from_particle.py
```
Core script:
- `ic_rec_from_particle_output_parallel.py` (can use in parallel or normal mode)

---
### Couple surface fluxes with grid_time output
```bash
python coupled_SF.py (this required ERA5 surfaces fluxes which can be downloaded and regridded using the following scripts 
- `download_era5_flux_tmp.py`
- `download_regrid_flux_data.sh`)
```
---

### Compute IC from grid_initial

```bash
python run_ic_from_grid_initial.py (can be run in parallel or normal mode)
```

Core script:
- `initial_conditions_grid_initial.py`

---

### Convert outputs to hourly time series and merge since there can be more number of releases each hour

```bash
python make_hourly_and_merge.py
```

---

### Plot diagnostics

Open:

```
merge_plots_script.ipynb
```

---

## Outputs

The workflow produces CSV files containing:

- Initial conditions (IC)
- Receptor values
- Surface flux contributions
- Hourly merged diagnostics


---

## Repository Structure

Main scripts to run:

- `run_binary_to_ncdf.py`
- `run_ic_rec_from_particle.py`
- `run_ic_from_grid_initial.py`
- `coupled_SF.py`
- `make_hourly_and_merge.py`

Core processing codes:

- `function_binary_to_ncdf_flexpart.py`
- `ic_rec_from_particle_output.py`
- `initial_conditions_grid_initial.py`

Download flux data scripts:

- `download_era5_flux_tmp.py`
- `download_regrid_flux_data.sh`


---
## Licence and acknowledgements

MIT licence (`LICENSE`).
Parts of the figure and data processing codes and scripts this documentation were written with the assistance of Claude (Anthropic) and U:ai chatgbt.
