# surface-flux-contribution-heatwaves
Lagrangian framework that splits the energy of heatwave air masses over Greece into its initial state and the contribution of surface heat fluxes along 10-day pathways (FLEXPART + ERA5, seven events, 1982-2025). Result: heatwave air starts with more energy and gains slightly less from the surface, mostly through less latent heating.
## What this is
An air mass that arrives over Greece during a heatwave carries moist static energy (MSE). We split the MSE at
arrival into two parts:
1. the MSE the air already had 10 days earlier (initial state), and
2. the energy it gained from the surface sensible and latent heat fluxes along the way.

## How it works
1. FLEXPART v11.0 runs backwards for 10 days from a box over Greece, five releases per day, June-August.
2. ERA5 surface heat fluxes are coupled to the FLEXPART source-receptor relations.
3. Results are collected per release and compared between heatwave days and other summer days.

Seven events: 1982, 1987, 1988, 2007, 2021, 2024 and 2025.

## Main result
Heatwave air starts with more energy than other summer air, and it gains slightly less energy from the surface,
mostly because of lower latent heating. <<Check against the final abstract.>>

## Folders
| folder | content |
|---|---|
| `flexpart_setup/` | FLEXPART configuration |
| `heatwave_detection/` | heatwave detection and release files |
| `MSE_reconstruction/` | post-processing and flux coupling |
| `figures/` | one script per figure of the paper |

## Data
Processed output: https://doi.org/10.5281/zenodo.23405520
Code archive: https://doi.org/10.5281/zenodo.23478028

## Reproduce the figures
1. Download the data, unpack it into one folder, and set `export HEATWAVE_DATA=/path/to/data`.
2. `micromamba env create -f environment.yml`
3. `cd figures` and run one script, for example `python fig06_boxplot_ic_flux.py`.

## Licence
MIT. Data: CC-BY 4.0.
