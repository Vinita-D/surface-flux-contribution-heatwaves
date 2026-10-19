# Figures

All figures of the paper can be reproduced from the data record on Zenodo (DOI: https://doi.org/10.5281/zenodo.23405520)
with one command. Nothing outside this folder and the data folder is needed.

## 1. Get the data
Download the data available from Zenodo, unpack the yearly data into one folder, e.g.

```bash
mkdir data && for f in data_*.tar; do tar -xf $f -C data; done
export HEATWAVE_DATA=$PWD/data
```

On the original cluster, the data folder is the default in `paths.py`. This gives the constants and heatwave dates etc.

## 2. Environment
Create your own enivornment

```bash
micromamba env create -f environment.yml      # or: conda env create -f environment.yml
micromamba activate <name of the environment>
```

## 3. Run
Run each figure code seperately.
```bash
cd figures
python fig01_synoptic_2007
```

Each script runs separately; the runner prints OK / FAILED for each and the files it wrote.
The scripts need the Natural Earth borders, which cartopy downloads on first use.

## Scripts

| script | content | paper figure |
|---|---|---|
| `fig01_synoptic_2007.py` | 2 m temperature anomaly, z500 and heatwave mask, 4 dates of 2007 | Fig. 1 |
| `fig02_decomposition_2007.py` | SRR, ERA5 fluxes and flux contribution, 2007 peak releases | Fig. 2 |
| `fig03_ic_flux_maps_2007.py` | initial-condition, sensible and latent contribution maps (per grid cell and per (100 km)^2) | |
| `fig04_timeseries_peak.py` | cumulative flux contribution versus days before release | |
| `fig05_closure_2007.py` | MSE budget time series and scatter plot, 2007 | |
| `fig06_boxplot_ic_flux.py` | initial condition and flux contribution, heatwave days vs summer days | Fig. 6 |
| `fig07_regional_bars.py` | regional shares (SRR vs flux; flux and initial MSE, summer vs heatwave days) with the region map | Fig. 7 |
| `fig08_synoptic_all_heatwaves.py` | requires large ERA5 data- produces anomalous weather pattern maps around the peak heatwave days of each year | Fig. 8 |

Support files: `paths.py` (all paths, constants and heatwave dates), `regions.py` (region masks and colours).

## Steps that need data that is not in the record

These were run once; their small results are in the record (folder `inputs/`).

* `prepare_inputs.py` -- reads the ERA5 2 m temperature (1979-2025), z500 and ERA5 fluxes (public, Copernicus
  Climate Data Store) and the heatwave mask; writes the small files in `inputs/`.
* `extract_backtime_peak.py` -- reads the FLEXPART `grid_time` output (~120 GB per year, which is not included;
  regenerate with the FLEXPART set-up in `flexpart_setup/`); writes `flux_by_backtime_long_2007_peak.csv`.

The 5 days x 5 releases per day, the land mask, the heatwave detection and the MSE reconstruction that
produce the data files are in `flexpart_setup/`, `heatwave_detection/` and `MSE_reconstruction/`.


## Licence and acknowledgements

MIT licence (`LICENSE`).
Parts of the figure and data processing codes and scripts this documentation were written with the assistance of Claude (Anthropic) and U:ai chatgbt.