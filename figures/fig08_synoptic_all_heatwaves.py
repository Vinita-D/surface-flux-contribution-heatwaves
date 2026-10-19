"""
Fig. 8: 500 hPa height anomaly (contours) and mean surface-flux contribution (shading) for the strongest
episode of every event. Both are averaged over the 10-day transport window of that episode.
IMP: The anomaly is read from the small file made by prepare_inputs.py (it comes from ERA5 geopotential at 500 hPa); the flux contribution is computed here from the maps in the data folder.
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap

from paths import FILES, FIG8_EPISODES, FIG8_EXTENT, SYNOPTIC_EVENTS_FILE, CP, LV_CSV, RECEPTOR_BOX

LABELS = list("abcdefg")
VMAX_FLUX = 10                                   # J/kg per grid cell, upper end of the colour scale
DATA_CRS = ccrs.PlateCarree()
PROJ = ccrs.Orthographic(central_longitude=-5, central_latitude=20)
CLEVS = np.arange(-24, 25, 3)
CLEVS = CLEVS[CLEVS != 0]


def event_flux(yr):
    """Mean over the releases of the episode of (Cp*dT + L*dq), J/kg per grid cell."""
    ep0, ep1 = FIG8_EPISODES[yr]
    with xr.open_dataset(FILES[yr]) as ds:
        rt = pd.to_datetime(ds["release_time"].values)
        keep = (rt >= pd.Timestamp(ep0)) & (rt < pd.Timestamp(ep1) + pd.Timedelta(days=1))
        n = int(keep.sum())
        sub = ds.isel(release=np.where(keep)[0])
        flux = (CP * sub["dT_map"] + LV_CSV * sub["dq_map"]).fillna(0).sum("release") / max(n, 1)
        flux = flux.load()
        if "dh_map" in sub:                      # check against the total stored in the file
            dh = float(sub["dh_map"].fillna(0).sum("release").sum()) / max(n, 1)
            print(f"{yr}: {n} releases | total of this map {float(flux.sum()):7.0f} J/kg | "
                  f"total of dh_map {dh:7.0f} J/kg | ratio {float(flux.sum()) / dh:.2f}")
        return flux.longitude.values, flux.latitude.values, flux, n


def make_cmap(name="plasma_r"):
    c = plt.get_cmap(name)(np.linspace(0.08, 0.92, 256))
    c[0] = [1, 1, 1, 1]
    return ListedColormap(c)


FLUX_CMAP = make_cmap()


def base_map(ax):
    ax.set_extent(FIG8_EXTENT, crs=DATA_CRS)
    ax.add_feature(cfeature.OCEAN, facecolor="aliceblue", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":", zorder=3)
    ax.add_patch(mpatches.Rectangle((RECEPTOR_BOX["lon1"], RECEPTOR_BOX["lat1"]), RECEPTOR_BOX["width"],
                                    RECEPTOR_BOX["height"], linewidth=1.6, edgecolor="blue",
                                    facecolor="none", transform=DATA_CRS, zorder=5))


fig, axes = plt.subplots(2, 4, figsize=(22, 10.5), subplot_kw={"projection": PROJ})
axes = axes.ravel()
plt.subplots_adjust(wspace=0.05, hspace=0.17, bottom=0.18)

for i, (ax, yr) in enumerate(zip(axes, FIG8_EPISODES)):
    lon, lat, flux, nrel = event_flux(yr)
    anom = xr.open_dataarray(SYNOPTIC_EVENTS_FILE, group=yr)
    base_map(ax)
    im = ax.pcolormesh(lon, lat, flux, transform=DATA_CRS, cmap=FLUX_CMAP, shading="auto",
                       vmin=0, vmax=VMAX_FLUX, zorder=1)
    cs = ax.contour(anom.longitude, anom.latitude, anom, levels=CLEVS, transform=DATA_CRS,
                    colors="black", linewidths=1.0, zorder=2)
    ax.clabel(cs, fontsize=8, inline=True, fmt="%d")
    ep0, ep1 = FIG8_EPISODES[yr]
    ax.set_title(f"{yr}\n{ep0} to {ep1}", fontsize=14, pad=8)
    ax.text(0.00, 1.07, f"{LABELS[i]})", transform=ax.transAxes, fontsize=14, va="bottom", ha="left", clip_on=False)

for j in range(len(FIG8_EPISODES), len(axes)):          # remove the unused 8th panel
    axes[j].remove()

cbar = fig.colorbar(im, ax=axes[:len(FIG8_EPISODES)], orientation="horizontal", fraction=0.05, pad=0.06,
                    shrink=0.65, aspect=50, extend="max")
cbar.set_label("Total surface-flux contribution (J kg$^{-1}$)", fontsize=15)
cbar.ax.tick_params(labelsize=12)

fig.savefig("synoptic_anomaly_flux_all_events.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.show()