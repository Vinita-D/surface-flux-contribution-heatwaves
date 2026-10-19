"""Fig. 4: flux contribution versus days before release (2007 peak releases).
Reads the csv made by extract_backtime_peak.py """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

from paths import BACKTIME_CSV, PEAK_DAYS_2007, BACK_DAYS

TITLE = f"Peak heatwave releases, {PEAK_DAYS_2007[0]} to {PEAK_DAYS_2007[-1]}"    # "" for no title
FONT_TITLE, FONT_LABEL, FONT_TICK, FONT_LEGEND = 15, 14, 13, 13

df = pd.read_csv(BACKTIME_CSV, parse_dates=["release_time"])
n_rel = df["release"].nunique()
by_day = (df.groupby("day_back")[["dT_flux", "dq_flux"]].sum().reindex(range(BACK_DAYS + 1), fill_value=0.0) / n_rel)

# cumulative from day 10 towards day 0 -> the value at day 0 is the total at arrival
by_day = by_day.sort_index(ascending=False)
by_day["cum_sens"] = by_day["dT_flux"].cumsum()
by_day["cum_lat"] = by_day["dq_flux"].cumsum()
by_day["cum_total"] = by_day["cum_sens"] + by_day["cum_lat"]
by_day = by_day.sort_index()

print(f"Releases: {n_rel}")
print("Daily increment (J/kg) and cumulative contribution (J/kg), mean per release:")
print(by_day[["dT_flux", "dq_flux", "cum_sens", "cum_lat", "cum_total"]].round(1).to_string())

days = by_day.index.values.astype(float)
days_fine = np.linspace(days.min(), days.max(), 300)
curve = lambda col: PchipInterpolator(days, by_day[col].values)(days_fine)

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
ax.plot(days_fine, curve("cum_sens"), color="darkorange", lw=2.2, label="Sensible heat flux")
ax.plot(days_fine, curve("cum_lat"), color="steelblue", lw=2.2, label="Latent heat flux")
ax.plot(days_fine, curve("cum_total"), color="black", lw=2.3, linestyle="--", label="Total")
ax.axhline(0, color="gray", lw=0.8, alpha=0.6)
ax.set_xlabel("Days before release", fontsize=FONT_LABEL)
ax.set_ylabel("Cumulative flux contribution (J kg$^{-1}$)", fontsize=FONT_LABEL)
if TITLE:
    ax.set_title(TITLE, fontsize=FONT_TITLE)
ax.invert_xaxis()
ax.set_xticks(range(BACK_DAYS + 1))
ax.tick_params(axis="both", labelsize=FONT_TICK)
ax.grid(lw=0.3, alpha=0.4)
ax.legend(frameon=False, loc="upper left", fontsize=FONT_LEGEND)
fig.tight_layout()
fig.savefig("fig04_timeseries_peak.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.show()
