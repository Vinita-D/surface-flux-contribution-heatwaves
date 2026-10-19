"""Fig. 1: synoptic situation of the 2007 heatwave (2 m temperature anomaly, z500, heatwave mask).
Reads the small files made by prepare_inputs.py."""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from paths import SYNOPTIC_FILES, SYNOPTIC_DATES, RECEPTOR_BOX

PROJ = ccrs.Orthographic(central_longitude=-5, central_latitude=10)
DATA_CRS = ccrs.PlateCarree()
map_extent = [-80, 40, 15, 90]
labels = ["a)", "b)", "c)", "d)"]

t_anom = xr.open_dataarray(SYNOPTIC_FILES["t2m_anom"])          # degC anomaly of the daily maximum
z500 = xr.open_dataarray(SYNOPTIC_FILES["z500"])                # geopotential height / 100
mask = xr.open_dataset(SYNOPTIC_FILES["mask"])["heatwave_mask"]

vmax = float(abs(t_anom).max())
vmin = -vmax

fig, axes = plt.subplots(2, 2, figsize=(14, 11), subplot_kw={"projection": PROJ})
axes = axes.ravel()

for ax, date, label in zip(axes, SYNOPTIC_DATES, labels):
    da = t_anom.sel(time=date)
    mask_i = mask.sel(time=date)
    z_i = z500.sel(time=date, method="nearest")

    ax.set_extent(map_extent, crs=DATA_CRS)
    ax.add_feature(cfeature.OCEAN, facecolor="aliceblue")
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":")

    im = da.plot(ax=ax, transform=DATA_CRS, cmap="RdBu_r", vmin=vmin, vmax=vmax, add_colorbar=False)
    mask_i.plot.contour(ax=ax, transform=DATA_CRS, levels=[0.5], colors="black", linewidths=1.8,
                        add_colorbar=False)
    cs = ax.contour(z_i.longitude, z_i.latitude, z_i, levels=np.linspace(400, 600, 41),
                    colors="maroon", linewidths=0.9, transform=DATA_CRS)
    ax.add_patch(mpatches.Rectangle((RECEPTOR_BOX["lon1"], RECEPTOR_BOX["lat1"]),
                                    RECEPTOR_BOX["width"], RECEPTOR_BOX["height"],
                                    linewidth=1.2, edgecolor="blue", facecolor="none", transform=DATA_CRS))
    ax.clabel(cs, fontsize=12, inline=True, fmt="%d")
    ax.set_title(f"{label} {date}", fontsize=14)

plt.subplots_adjust(wspace=0.05, hspace=0.10, bottom=0.12)
cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.06, shrink=0.80, aspect=45)
cbar.set_label("Temperature anomaly (°C)", fontsize=15)
cbar.ax.tick_params(labelsize=14)

fig.savefig("fig01_synoptic_2007.png", dpi=350, bbox_inches="tight")
plt.show()
