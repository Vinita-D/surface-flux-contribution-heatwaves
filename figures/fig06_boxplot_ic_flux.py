"""Fig. 6: initial-condition terms and flux contributions, heatwave days vs summer days (all events).
Reads the merged_hourly.csv files only."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D

from paths import CSV_FILES, YEARS, CP, G,  LV_IC, hw_mask

YEAR_MARKERS = {"1982": "o", "1987": "s", "1988": "^", "2007": "D", "2021": "*", "2024": "v", "2025": "P"}
JITTER = {y: j for y, j in zip(YEAR_MARKERS, np.linspace(-0.15, 0.15, 7))}
KEYS = ["CpT", "gz", "Lvq", "sens", "lat"]

# ---- terms per release (J/kg), split into heatwave and summer-day releases
raw_by_year = {k: {"hw": {}, "sum": {}} for k in KEYS}
for yr in YEARS:
    df = pd.read_csv(CSV_FILES[yr])
    df["release_time"] = pd.to_datetime(df["release_time"])
    df["is_hw"] = hw_mask(df["release_time"], yr)
    df = df.dropna(subset=["IC_T_grid", "IC_q_grid", "IC_z_grid", "SHF_flux", "MF_flux"])
    hw = df["is_hw"].values
    raw = {"CpT": CP * df["IC_T_grid"].values, "gz": G * df["IC_z_grid"].values,
           "Lvq": LV_IC * df["IC_q_grid"].values,
           "sens": CP * df["SHF_flux"].values, "lat": LATENT_FACTOR * df["MF_flux"].values}
    for k, v in raw.items():
        raw_by_year[k]["hw"][yr], raw_by_year[k]["sum"][yr] = v[hw], v[~hw]

# ---- deviations from ONE shared reference: the mean of all summer days of all years
reference = {k: np.concatenate([raw_by_year[k]["sum"][y] for y in YEARS]).mean() for k in KEYS}
pooled = {k: {"hw": [], "sum": []} for k in KEYS}
year_means = {k: {"hw": {}, "sum": {}} for k in KEYS}
for yr in YEARS:
    for k in KEYS:
        dev_hw, dev_sum = raw_by_year[k]["hw"][yr] - reference[k], raw_by_year[k]["sum"][yr] - reference[k]
        pooled[k]["hw"].append(dev_hw); pooled[k]["sum"].append(dev_sum)
        year_means[k]["hw"][yr], year_means[k]["sum"][yr] = dev_hw.mean(), dev_sum.mean()
for k in KEYS:
    pooled[k]["hw"], pooled[k]["sum"] = np.concatenate(pooled[k]["hw"]), np.concatenate(pooled[k]["sum"])


def draw_group(ax, key, x_center, width=0.7):
    x_sum, x_hw = x_center - 0.45, x_center + 0.45
    ax.boxplot([pooled[key]["sum"]], positions=[x_sum], widths=width, patch_artist=False, showfliers=False,
               boxprops=dict(color="black", linewidth=1.3, linestyle="--"), medianprops=dict(color="black", linewidth=1.6),
               whiskerprops=dict(color="black", linestyle="--"), capprops=dict(color="black"))
    ax.boxplot([pooled[key]["hw"]], positions=[x_hw], widths=width, patch_artist=False, showfliers=False,
               boxprops=dict(color="black", linewidth=1.6), medianprops=dict(color="black", linewidth=1.8),
               whiskerprops=dict(color="black"), capprops=dict(color="black"))
    for yr, marker in YEAR_MARKERS.items():       # open = summer days, filled = heatwave days
        ax.plot(x_sum + JITTER[yr], year_means[key]["sum"][yr], marker=marker, markerfacecolor="white",
                markeredgecolor="black", markeredgewidth=1.0, ms=7, zorder=5)
        ax.plot(x_hw + JITTER[yr], year_means[key]["hw"][yr], marker=marker, markerfacecolor="black",
                markeredgecolor="black", markeredgewidth=1.0, ms=7, zorder=5)


fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)
ax = axes[0]
centers_a = [0, 2.2, 4.4]
for c, key in zip(centers_a, ["CpT", "gz", "Lvq"]):
    draw_group(ax, key, c)
ax.axhline(0, color="gray", lw=0.8, alpha=0.6)
ax.set_xticks(centers_a)
ax.set_xticklabels([r"Enthalpy" + "\n" + r"$C_p T$", r"Geopotential" + "\n" + r"$gz$",
                    r"Latent heat" + "\n" + r"$L_v q$"], fontsize=12)
ax.set_ylabel(r"Deviation from summer-day mean (J kg$^{-1}$)", fontsize=12)
ax.set_title("(a) Initial condition", fontsize=12)
ax.grid(axis="y", lw=0.3, alpha=0.4)

ax = axes[1]
centers_b = [0, 2.2]
for c, key in zip(centers_b, ["sens", "lat"]):
    draw_group(ax, key, c)
ax.axhline(0, color="gray", lw=0.8, alpha=0.6)
ax.set_xticks(centers_b)
ax.set_xticklabels(["Sensible heat\nflux contribution", "Latent heat\nflux contribution"], fontsize=12)
ax.set_ylabel(r"Deviation from summer-day mean (J kg$^{-1}$)", fontsize=12)
ax.set_title("(b) Surface flux", fontsize=14)
ax.grid(axis="y", lw=0.3, alpha=0.4)

group_handles = [
    Line2D([0], [0], color="black", linestyle="--", linewidth=1.3, marker="o", markerfacecolor="white",
           markeredgecolor="black", ms=8, label="Summer days"),
    Line2D([0], [0], color="black", linestyle="-", linewidth=1.6, marker="o", markerfacecolor="black",
           markeredgecolor="black", ms=8, label="Heatwave days"),
]
year_handles = [Line2D([0], [0], marker=m, color="black", linestyle="None", markerfacecolor="gray",
                       markeredgecolor="black", ms=7, label=yr) for yr, m in YEAR_MARKERS.items()]
fig.legend(handles=group_handles, loc="lower left", ncol=2, frameon=False, bbox_to_anchor=(0.05, -0.07),
           fontsize=10, handlelength=3.5)
fig.legend(handles=year_handles, loc="lower right", ncol=7, frameon=False, bbox_to_anchor=(0.98, -0.07),
           fontsize=10, title="Year (mean)")
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig("fig06_boxplot_ic_flux.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.show()

# ---- numbers for the text
n_hw, n_sum = len(pooled["gz"]["hw"]), len(pooled["gz"]["sum"])
print(f"\nn heatwave = {n_hw}, n summer = {n_sum}")
print("\nMean deviation of heatwave days from the summer-day mean (J/kg):")
for key, name in [("CpT", "Enthalpy"), ("gz", "Geopotential"), ("Lvq", "Latent (IC)"),
                  ("sens", "Sensible flux"), ("lat", "Latent flux")]:
    print(f"  {name:15s} {pooled[key]['hw'].mean():+9.0f}")
print("\nYearly means of the heatwave-day deviation (J/kg):")
print(pd.DataFrame({k: year_means[k]["hw"] for k in KEYS}).round(0).to_string())
