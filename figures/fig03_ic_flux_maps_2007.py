"""Fig. 3: initial-condition, sensible and latent contributions as maps (2007 peak releases).
Making two versions: per grid cell, and per (100 km)^2 (divided by the cell area)."""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize, TwoSlopeNorm

from paths import FILES, IC_FILES, PEAK_DAYS_2007, CP, LATENT_FACTOR, RECEPTOR_BOX

PROJ = ccrs.Orthographic(central_longitude=-5, central_latitude=10)
DATA_CRS = ccrs.PlateCarree()
map_extent = [-80, 40, 15, 90]
START, END = PEAK_DAYS_2007[0], PEAK_DAYS_2007[-1]

# ---- load the same releases from both files
def pick(ds):
    t = ds["release_time"].values
    keep = (t >= np.datetime64(START)) & (t < np.datetime64(END) + np.timedelta64(1, "D"))
    return ds.isel(release=np.where(keep)[0])

ds_ic = pick(xr.open_dataset(IC_FILES["2007"]))
ds_flux = pick(xr.open_dataset(FILES["2007"]))
assert np.array_equal(ds_ic.release_time.values, ds_flux.release_time.values), "release mismatch"
print("Releases:", ds_ic.sizes["release"])

# ---- contributions per grid cell (J/kg), mean over the releases
w, ic = ds_ic["sensitivity_column"], ds_ic["IC_mse_map"]
print("Weight sum per release (should be ~1):", w.sum(["latitude", "longitude"]).values.round(3)[:5])
ic_cell = (w * ic).fillna(0).mean("release")
dT_cell = (CP * ds_flux["dT_map"]).fillna(0).mean("release")
dq_cell = (LATENT_FACTOR * ds_flux["dq_map"]).fillna(0).mean("release")
tot_ic, tot_dT, tot_dq = float(ic_cell.sum()) / 1000, float(dT_cell.sum()), float(dq_cell.sum())
print(f"IC = {tot_ic:.1f} kJ/kg, sensible = {tot_dT:.0f} J/kg, latent = {tot_dq:.0f} J/kg")

# ---- grid-cell area in km2 (depends only on latitude)
dlat = np.deg2rad(abs(float(ic_cell.latitude[1] - ic_cell.latitude[0])))
dlon = np.deg2rad(abs(float(ic_cell.longitude[1] - ic_cell.longitude[0])))
area_km2 = 6371.0 ** 2 * dlat * dlon * np.cos(np.deg2rad(ic_cell.latitude))


def draw(fig, ax, data, title, total, cmap, norm, unit, ticks=None):
    data = data.where(data != 0)                       # no air mass -> left blank
    ax.set_extent(map_extent, crs=DATA_CRS)
    ax.add_feature(cfeature.OCEAN, facecolor="#eef3f7", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#f7f6f2", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":", zorder=3)
    im = ax.pcolormesh(data.longitude, data.latitude, data, transform=DATA_CRS, cmap=cmap, norm=norm,
                       shading="auto", zorder=1)
    ax.add_patch(mpatches.Rectangle((RECEPTOR_BOX["lon1"], RECEPTOR_BOX["lat1"]), RECEPTOR_BOX["width"],
                                    RECEPTOR_BOX["height"], linewidth=1.6, edgecolor="blue",
                                    facecolor="none", transform=DATA_CRS, zorder=5))
    ax.set_title(f"{title}\n{total}", fontsize=14)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, extend="both", ticks=ticks)
    cb.set_label(unit, fontsize=13)
    cb.ax.tick_params(labelsize=12)


def area_mean(density):
    cells = density != 0
    return float((density * area_km2).where(cells).sum() / area_km2.where(cells).sum())


suptitle = f"Mean over the releases {START} to {END}"

# ---- version 1: per grid cell
unit = r"J kg$^{-1}$ per grid cell"
ic_max = float(np.nanpercentile(ic_cell.where(ic_cell > 0).values, 99))
fig, axes = plt.subplots(3, 1, figsize=(8, 15), dpi=300, subplot_kw={"projection": PROJ}, constrained_layout=True)
draw(fig, axes[0], ic_cell, "a) Initial condition contribution", f"total = {tot_ic:.1f} kJ kg$^{{-1}}$", "Reds",
     Normalize(vmin=2.5, vmax=ic_max), unit)
draw(fig, axes[1], dT_cell, "b) Sensible heat flux contribution", f"total = {tot_dT:.0f} J kg$^{{-1}}$", "RdBu_r",
     TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1.5), unit, ticks=[-1, 0, 0.5, 1.0, 1.5])
draw(fig, axes[2], dq_cell, "c) Moisture flux contribution", f"total = {tot_dq:.0f} J kg$^{{-1}}$", "BrBG",
     TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1.0), unit, ticks=[-1, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
fig.suptitle(suptitle, fontsize=15, y=1.02)
plt.savefig("fig03_ic_flux_per_grid_cell.png", dpi=350, bbox_inches="tight", facecolor="white")
plt.show()

# ---- version 2: per (100 km)^2, i.e. divided by the cell area
FACTOR = 1e4
ic_a, dT_a, dq_a = (m / area_km2 * FACTOR for m in (ic_cell, dT_cell, dq_cell))
unit = r"J kg$^{-1}$ (100 km)$^{-2}$"
fig, axes = plt.subplots(3, 1, figsize=(7.5, 13.5), dpi=300, subplot_kw={"projection": PROJ}, constrained_layout=True)
draw(fig, axes[0], ic_a, "a) Initial condition contribution", f"area mean = {area_mean(ic_a):.2f} {unit}", "Reds",
     Normalize(vmin=10, vmax=80), unit)
draw(fig, axes[1], dT_a, "b) Sensible heat flux contribution", f"area mean = {area_mean(dT_a):.3f} {unit}", "RdBu_r",
     TwoSlopeNorm(vmin=-5, vcenter=0, vmax=8), unit, ticks=[-5, 0, 2, 4, 6, 8])
draw(fig, axes[2], dq_a, "c) Moisture flux contribution", f"area mean = {area_mean(dq_a):.3f} {unit}", "BrBG",
     TwoSlopeNorm(vmin=-1, vcenter=0, vmax=8), unit, ticks=[-1, -0.5, 0, 2, 4, 6, 8])
fig.suptitle(suptitle, fontsize=15, y=1.02)
plt.savefig("fig03_ic_flux_per_100km2.png", dpi=350, bbox_inches="tight", facecolor="white")
plt.show()
