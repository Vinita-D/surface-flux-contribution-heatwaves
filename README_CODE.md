# Lagrangian surface-flux attribution of the MSE of Greek heatwaves

Code for "Surface flux contributions to Mediterranean heatwaves: a new Lagrangian diagnostic. Data: "

An air mass arriving at the receptor (a heatwave over Greece) is traced back 10 days with FLEXPART. Its moist static energy (MSE) at the receptor is split into:
**the MSE it carried at the start of the pathway + the contribution of ERA5 surface sensible and latent heat
fluxes along the pathway**.

`flexpart_setup/FLEXPART_ETA` is the FLEXPART v11.0 executable (commit e8e6f14) used for the runs. FLEXPART is distributed under the GNU General Public License; its source code is available at https://www.flexpart.eu. This file is covered by that licence, not by the MIT licence of the rest of this archive.

## Folders

| folder | content |
|---|---|
| `flexpart_setup/` | FLEXPART v11.0 configuration (`options/`, `pathnames`, `run.job`) |
| `heatwave_detection/` | heatwave detection and the script that writes the release file the part_ic.nc file are large are not included here. (`generating_part_ic_file.py`) |
| `MSE_reconstruction/` | post-processing: converts the output, couples the fluxes, computes initial conditions, merges |
| `figures/` | one script per figure of the paper, plus `paths.py` |

## Steps to run : 
1. **Release file:** `heatwave_detection/generating_part_ic_file.py`
2. **FLEXPART:** run backwards with the files in `flexpart_setup/` (FLEXPART v11.0, commit e8e6f14,
   <https://www.flexpart.eu>; the program itself is not included).
3. **Post-processing:** edit the paths at the top of `MSE_reconstruction/run_all_function.py` and run it. It runs
   `run_binary_to_ncdf.py`, `run_fix_grid_initial_time.py`, `run_rec_partic.py`, `run_coupled_SF.py`,
   `run_ic_from_grid_initial.py` and `merge_all_csv_result.py`. The result is, for each release,
   the initial MSE, the receptor MSE and the flux contributions (`merged_hourly.csv`, `flux_maps*.nc`,
   `ic_maps*.nc`).
4. **Figures:** see below.

## Reproducing  the figures with some compressed avaiable data 


```bash
micromamba env create -f environment.yml
micromamba activate zarr_env_micro
export HEATWAVE_DATA=/path/to/unpacked/data       # the data record, unpacked into one folder
cd figures
python fig06_boxplot_ic_flux.py                    # one script per figure
```

| script | content | paper figure |
|---|---|---|
| `fig01_synoptic_2007.py` | 2 m temperature anomaly, z500 and heatwave mask, 2007 | Fig. 1 |
| `fig02_decomposition_2007.py` | SRR, ERA5 fluxes and flux contribution, 2007 peak releases | Fig. 2 |
| `fig03_ic_flux_maps_2007.py` | initial-condition, sensible and latent contribution maps | <<Fig.>> |
| `fig04_timeseries_peak.py` | flux contribution versus days before release | <<Fig.>> |
| `fig05_closure_2007.py` | MSE budget and scatter against the receptor MSE | <<Fig.>> |
| `fig06_boxplot_ic_flux.py` | initial condition and fluxes, heatwave vs summer days | Fig. 6 |
| `fig07_regional_bars.py` | regional shares with the region map (also writes two supplement figures) | Fig. 7 |
| `fig08_regional_bars.py` | synoptic situation of all events | Fig. 8 | needs a large ERA5 data here 

`fig07_regional_bars.py` needs the ERA5 land-sea mask to build the region masks to compute regional contribution. The two masks needed
here are included in `figures/region_masks_cache/`.

Scripts that needs ERA5 data can be downloaded, to re-make the small inputs: `figures/prepare_inputs.py` (ERA5 2 m temperature, z500, ERA5 fluxes; public, Copernicus Climate Data Store). `figures/extract_backtime_peak.py`
needs the FLEXPART `grid_time` output (not included).

## Software

Python environment: `environment.yml` (`requirements.txt` for pip, without the plotting packages).
Main packages: numpy 2.2.4, pandas 2.2.3, xarray 2025.3.1, dask 2025.3.0, netCDF4 1.7.2, scipy 1.15.2,
matplotlib 3.10.8, Cartopy 0.24.1, shapely 2.1.2. FLEXPART v11.0 (Bakels et al., 2024).
ERA5: Hersbach et al. (2020).

## Licence and acknowledgements

MIT licence (`LICENSE`).
Parts of the figure and data processing codes and scripts this documentation were written with the assistance of Claude (Anthropic) and U:ai chatgbt.