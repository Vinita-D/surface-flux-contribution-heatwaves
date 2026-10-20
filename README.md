# surface-flux-contribution-heatwaves
To evaluate the role of surface turbulent fluxes for heatwaves, we build a Lagrangian framework in which the
moist static energy (MSE) of an air mass at arrival is split into two parts: the MSE it already had at the start
of a 10-day backward pathway, and the energy it gained from surface sensible and latent heat fluxes along the
way. FLEXPART provides the transport, and ERA5 provides the surface fluxes.
We apply it to seven Greek heatwaves (1982, 1987, 1988, 2007, 2021, 2024 and 2025). Over the ten days, the
surface adds 5.4-6.5 kJ kg-1 to the heatwave air masses, of which sensible heating provides about 57% and
latent heating the rest. On other summer days, the surface adds 6.1-6.8 kJ kg-1. In every event, the heatwave
air therefore gains somewhat less energy from the surface (by 0.2-1.2 kJ kg-1), mainly because of lower latent
heating, while it starts with more energy. The surface fluxes thus do not amplify the energy of heatwave air
relative to other summer days, and the initial state matters even more. 
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
