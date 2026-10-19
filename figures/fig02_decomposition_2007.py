"""Fig. 2: SRR, ERA5 fluxes and flux contribution for the 2007 peak releases (per grid cell)."""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LogNorm, TwoSlopeNorm

from paths import FILES, ERA5_WINDOW_2007, PEAK_DAYS_2007, CP, LATENT_FACTOR, RECEPTOR_BOX

TITLE_FS, LABEL_FS, TICK_FS = 14, 13, 12
PROJ = ccrs.Orthographic(central_longitude=-5, central_latitude=10)
DATA_CRS = ccrs.PlateCarree()
map_extent = [-80, 40, 15, 90]
START, END = PEAK_DAYS_2007[0], PEAK_DAYS_2007[-1]

# ---- data: mean over the releases of the peak days
ds = xr.open_dataset(FILES["2007"])
t = ds["release_time"].values
keep = (t >= np.datetime64(START)) & (t < np.datetime64(END) + np.timedelta64(1, "D"))
ds_sel = ds.isel(release=np.where(keep)[0])
n_rel = ds_sel.sizes["release"]
print("Releases:", n_rel)

srr_map = ds_sel["sensitivity_map"].fillna(0).mean("release")                                   # s m3 kg-1
dh_map = (CP * ds_sel["dT_map"] + LATENT_FACTOR * ds_sel["dq_map"]).fillna(0).mean("release")   # J/kg
tot_dh = float(dh_map.sum())
print(f"Total flux contribution (sum over cells) = {tot_dh:.0f} J/kg")

era5 = xr.open_dataset(ERA5_WINDOW_2007)
shf_mean, lhf_mean = era5["shf_mean"], era5["lhf_mean"]                                         # W m-2

srr_hi = float(np.nanpercentile(srr_map.where(srr_map > 0).values, 99))
dh_nz = dh_map.where(dh_map != 0).values
dh_lo = min(float(np.nanpercentile(dh_nz, 1)), -1e-6)
dh_hi = max(float(np.nanpercentile(dh_nz, 99)), 1e-6)
print(f"SRR 99th pct: {srr_hi:.3g}   flux contribution 1st-99th pct: {dh_lo:.3g} to {dh_hi:.3g}")


def draw_panel(fig, ax, data, title, cmap, cbar_label, norm, extend="both", ticks=None, total=None):
    ax.set_extent(map_extent, crs=DATA_CRS)
    ax.add_feature(cfeature.OCEAN, facecolor="#eef3f7", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#f7f6f2", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":", zorder=3)
    ax.gridlines(draw_labels=False, linestyle="--", linewidth=0.4, alpha=0.4)
    im = ax.pcolormesh(data.longitude, data.latitude, data, transform=DATA_CRS, cmap=cmap, norm=norm,
                       shading="auto", zorder=1)
    ax.add_patch(mpatches.Rectangle((RECEPTOR_BOX["lon1"], RECEPTOR_BOX["lat1"]), RECEPTOR_BOX["width"],
                                    RECEPTOR_BOX["height"], linewidth=1.6, edgecolor="blue",
                                    facecolor="none", transform=DATA_CRS, zorder=5))
    ax.set_title(title if total is None else f"{title}\n{total}", fontsize=TITLE_FS)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.85, pad=0.06, extend=extend, ticks=ticks)
    cb.set_label(cbar_label, fontsize=LABEL_FS)
    cb.ax.tick_params(labelsize=TICK_FS)


fig, axes = plt.subplots(2, 2, figsize=(14, 11), subplot_kw={"projection": PROJ}, constrained_layout=True)
axes = axes.ravel()

draw_panel(fig, axes[0], srr_map.where(srr_map > 0), "a) Source–Receptor Relation (SRR)", "magma_r",
           r"s m$^3$ kg$^{-1}$ per grid cell", LogNorm(vmin=srr_hi / 1000, vmax=srr_hi), extend="both")
draw_panel(fig, axes[1], shf_mean, "b) ERA5 sensible heat flux (SSHF)", "RdBu_r", r"W m$^{-2}$",
           TwoSlopeNorm(vmin=-50, vcenter=0, vmax=150), ticks=[-50, 0, 50, 100, 150])
draw_panel(fig, axes[2], lhf_mean, "c) ERA5 latent heat flux (SLHF)", "BrBG", r"W m$^{-2}$",
           TwoSlopeNorm(vmin=-50, vcenter=0, vmax=200), ticks=[-50, 0, 50, 100, 150, 200])
draw_panel(fig, axes[3], dh_map.where(dh_map != 0),
           r"d) $\Delta h_{\mathrm{flux}}$ = SRR $\times$ (SSHF + SLHF) / $\Delta z$", "PuOr",
           r"J kg$^{-1}$ per grid cell", TwoSlopeNorm(vmin=dh_lo, vcenter=0, vmax=dh_hi),
           total=f"total = {tot_dh:.0f} J kg$^{{-1}}$")

what = f"peak heatwave day {START}" if START == END else f"releases {START} to {END}"
fig.suptitle(f"Mean over {n_rel} {what}", fontsize=15, y=1.02)
plt.savefig("fig02_decomposition_2007.png", dpi=350, bbox_inches="tight", facecolor="white")
plt.show()
