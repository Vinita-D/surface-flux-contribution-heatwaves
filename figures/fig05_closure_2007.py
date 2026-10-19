"""Fig. 5: MSE reconstruction for 2007 budget closure for each release(a) and scatter against
the receptor MSE (b). Reads merged_hourly.csv for 2007 heatwave cases"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from paths import CSV_FILES, HW_PERIODS, CP, hw_mask, LV_CSV

window = 10 # for smoothing each releases times axis for plot a)
LABEL_FS, TICK_FS, LEGEND_FS, PANEL_FS = 12, 11, 10, 13
periods = HW_PERIODS["2007"]

df = pd.read_csv(CSV_FILES["2007"])
df["release_time"] = pd.to_datetime(df["release_time"])
df = df.sort_values("release_time").reset_index(drop=True)
df["is_hw"] = hw_mask(df["release_time"], "2007")

# ---- budget (J/kg): reconstructed receptor MSE = initial MSE + total flux contribution (h_flux)
base      = df["IC_mse_grid"]             # IC MSE
h_flux    = df["h_flux"]                  # total surface-flux contribution
sens_K    = CP * df["SHF_flux"]           # sensible part
latent_K  = df['MF_flux'] *LV_CSV              # latent part = the rest of h_flux

recon     = base + sens_K +  latent_K                # reconstructed receptor MSE
rec_mse_K = df["rec_mse"]                 # receptor MSE (same rows as recon)

# smoothing
base_smooth   = base.rolling(window=window, center=True, min_periods=1).mean()
sens_smooth   = sens_K.rolling(window=window, center=True, min_periods=1).mean()
latent_smooth = latent_K.rolling(window=window, center=True, min_periods=1).mean()
recon_smooth  = recon.rolling(window=window, center=True, min_periods=1).mean()
rec_smooth    = rec_mse_K.rolling(window=window, center=True, min_periods=1).mean()

x_time     = df["release_time"].to_numpy()
base_arr   = base_smooth.to_numpy()
sens_arr   = sens_smooth.to_numpy()
latent_arr = latent_smooth.to_numpy()
recon_arr  = recon_smooth.to_numpy()
rec_arr    = rec_smooth.to_numpy()

# ---- scatter statistics
x_raw =  base + df["h_flux"]
y_raw = rec_mse_K
finite = np.isfinite(x_raw) & np.isfinite(y_raw)

x = x_raw[finite].to_numpy()
y = y_raw[finite].to_numpy()
hw_scatter = df["is_hw"].values[finite]

r = np.corrcoef(x, y)[0, 1]
rmse = np.sqrt(np.mean((y - x) ** 2))
coef = np.polyfit(x, y, 1)

xmin = min(x.min(), y.min())
xmax = max(x.max(), y.max())
pad = 0.04 * (xmax - xmin)
lim = (xmin - pad, xmax + pad)

xx = np.linspace(lim[0], lim[1], 100)
yy_fit = coef[0] * xx + coef[1]


resid = (recon - rec_mse_K)[finite].to_numpy()    # positive = overshoot
print("=== MSE reconstruction closure, 2007 ===")
print(f"all releases : r = {r:.2f}, RMSE = {rmse:.0f} J/kg, mean residual = {resid.mean():+.0f} J/kg")
print(f"heatwave     : mean residual = {resid[hw_scatter].mean():+.0f} J/kg (n={hw_scatter.sum()})")
print(f"summer days  : mean residual = {resid[~hw_scatter].mean():+.0f} J/kg (n={(~hw_scatter).sum()})")

# ---- figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4.3), dpi=250,
                               gridspec_kw={"width_ratios": [2.3, 1.27], "wspace": 0.01})

for i, (start, end) in enumerate(periods):
    ax1.axvspan(
        pd.Timestamp(start),
        pd.Timestamp(end) + pd.Timedelta(hours=23, minutes=59),
        color="lightgrey", alpha=0.45, linewidth=0, zorder=0,
        label="Heatwave period" if i == 0 else None
    )

ax1.plot(x_time, rec_arr,   label="Receptor MSE",               color="green",  linewidth=1.8)
ax1.plot(x_time, base_arr,  label="IC MSE",                     color="black",  linestyle="--", linewidth=1.0)
ax1.plot(x_time, recon_arr, label="IC MSE + flux contribution", color="maroon", linewidth=1.8)

ax1.fill_between(x_time, base_arr, base_arr + sens_arr,
                 color="darkorange", alpha=0.45, label="Sensible contribution")
ax1.fill_between(x_time, base_arr + sens_arr, base_arr + sens_arr + latent_arr,
                 color="lightblue", alpha=0.55, label="Latent contribution")

ax1.text(-0.06, 1.04, "a)", transform=ax1.transAxes, fontsize=PANEL_FS, va="bottom", ha="left", clip_on=False)

ax1.set_ylabel("J kg$^{-1}$", fontsize=LABEL_FS)
ax1.set_xlabel("Release time", fontsize=LABEL_FS)
ax1.tick_params(axis="both", labelsize=TICK_FS)
ax1.tick_params(axis="x", rotation=45)
ax1.grid(linewidth=0.4, alpha=0.5)
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=3, frameon=False, fontsize=LEGEND_FS)
ax1.set_xlim(df["release_time"].min(), df["release_time"].max())
ax1.margins(x=0)

fmt_sci = ScalarFormatter(useMathText=True)
fmt_sci.set_powerlimits((0, 0))
ax1.yaxis.set_major_formatter(fmt_sci)
ax1.yaxis.get_offset_text().set_size(TICK_FS)


ax2.scatter(x[~hw_scatter], y[~hw_scatter], s=20, facecolor="steelblue", edgecolor="lightgrey",
            linewidth=0.5, alpha=0.65, label="warm period")
ax2.scatter(x[hw_scatter], y[hw_scatter], s=25, facecolor="steelblue", edgecolor="black",
            linewidth=0.6, alpha=0.65, label="Heatwave period")

ax2.plot(lim, lim, linestyle="--", color="black", linewidth=1.0, label="1:1 line")
ax2.plot(xx, yy_fit, color="steelblue", linewidth=1.5, linestyle="-.")

ax2.text(-0.06, 1.04, "b)", transform=ax2.transAxes, fontsize=PANEL_FS, va="bottom", ha="left", clip_on=False)
ax2.text(0.04, 0.94, f"r = {r:.2f}\nRMSE = {rmse:.2f} J kg$^{{-1}}$", transform=ax2.transAxes,
         va="top", fontsize=LABEL_FS, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

ax2.set_xlim(lim)
ax2.set_ylim(lim)
ax2.set_aspect("equal")

ax2.set_xlabel("Reconstructed MSE J kg$^{-1}$", fontsize=LABEL_FS)
ax2.set_ylabel("Receptor MSE J kg$^{-1}$", fontsize=LABEL_FS)
ax2.tick_params(axis="both", labelsize=TICK_FS)
ax2.grid(linewidth=0.4, alpha=0.5)
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=2, frameon=False, fontsize=LEGEND_FS)

fmt2 = ScalarFormatter(useMathText=True); fmt2.set_powerlimits((0, 0))
ax2.xaxis.set_major_formatter(fmt2); ax2.yaxis.set_major_formatter(fmt2)
ax2.xaxis.get_offset_text().set_size(TICK_FS); ax2.yaxis.get_offset_text().set_size(TICK_FS)

plt.subplots_adjust(top=0.78)
plt.savefig("fig05_closure_2007.png", dpi=300, bbox_inches="tight")
plt.show()
