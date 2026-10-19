import os
import cdsapi

OUTDIR = "/jetfs/scratch/vdeshmukh/"
YEARS = [ 1982,1988,2025]
MONTHS = ["05", "06", "07", "08"]

VARIABLES = [
    "mean_evaporation_rate",
    "mean_surface_latent_heat_flux",
    "mean_surface_sensible_heat_flux",
]

DAYS = [
    "01","02","03","04","05","06","07","08","09","10",
    "11","12","13","14","15","16","17","18","19","20",
    "21","22","23","24","25","26","27","28","29","30","31"
]

TIMES = [
    "00:00","01:00","02:00","03:00","04:00","05:00",
    "06:00","07:00","08:00","09:00","10:00","11:00",
    "12:00","13:00","14:00","15:00","16:00","17:00",
    "18:00","19:00","20:00","21:00","22:00","23:00"
]

AREA = [90, -180, -90, 180]

os.makedirs(OUTDIR, exist_ok=True)

c = cdsapi.Client()

for year in YEARS:
    for month in MONTHS:
        outfile = os.path.join(
            OUTDIR,
            f"ERA5_mean_fluxes_{year}_{month}.nc"
        )

        print(f"Downloading {year}-{month} -> {outfile}")

        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": ["reanalysis"],
                "variable": VARIABLES,
                "year": [str(year)],
                "month": [month],
                "day": DAYS,
                "time": TIMES,
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": AREA,
            },
            outfile,
        )

        print(f"Finished: {outfile}")

print("All downloads complete.")
