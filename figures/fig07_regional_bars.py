# ============================================================
# Regional bar figures 
# ============================================================
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from shapely.geometry import box
from paths import CSV_FILES, HW_PERIODS, CP, hw_mask, LV_CSV

# ============================================================
# SETTINGS
# ============================================================
TOP_IN_KJ = True             # numbers above the bars: kJ/kg (True) or J/kg (False)

# ERA5 land/sea mask (only used to build the region masks; the masks are then cached, see MASK_CACHE)
LAND_FILE = "/jetfs/scratch/vdeshmukh/2m_data_temp/ERA5_t2m_q2m_sp_1979_2025_0p5_land.nc"
MASK_CACHE = Path(__file__).resolve().parent / "region_masks"   # first run builds the masks (slow), later runs load them

YEARS = ["1982", "1987", "1988", "2007", "2021", "2024", "2025"]

# Where the maps are. Option 1: the collected data folder  <DATA>/<year>/flux_maps.nc and ic_maps.nc
DATA = "/jetfs/scratch/vdeshmukh/Heatwaves_Greek_MSE_data"
FILES = {y: f"{DATA}/{y}/flux_maps.nc" for y in YEARS}
IC_FILES = {y: f"{DATA}/{y}/ic_maps.nc" for y in YEARS}

# Option 2: the original run folders (replace the two dictionaries above by these lines)
# ROOT = "/jetfs/scratch/vdeshmukh/"
# FILES = {"1982": ROOT + "1982_grecce_heatwave/output_summer/flux_maps_output_summer.nc",
#          "1987": ROOT + "1987_grecce_heatwave/output_summer/flux_maps_output_summer.nc",
#          "1988": ROOT + "1988_grecce_heatwave/output_summer/flux_maps_output_summer.nc",
#          "2007": ROOT + "2007_grecce_heatwave2/output_summer/flux_maps_summer_10daysback.nc",
#          "2021": ROOT + "2021_grecce_heatwave/output_summer/flux_maps_output_summer.nc",
#          "2024": ROOT + "2024_grecce_heatwave/output_summer/flux_maps_output_summer.nc",
#          "2025": ROOT + "2025_grecce_heatwave/output_summer/flux_maps_output_box.nc"}
# IC_FILES = {y: f.rsplit("/", 1)[0] + "/ic_maps_from_grid_initial.nc" for y, f in FILES.items()}

HW_PERIODS = {                # heatwave days, both days included
    "1982": [("1982-06-24", "1982-06-28")],
    "1987": [("1987-07-20", "1987-07-27")],
    "1988": [("1988-07-05", "1988-07-09")],
    "2007": [("2007-06-21", "2007-06-28"), ("2007-07-17", "2007-07-26"), ("2007-08-22", "2007-08-26")],
    "2021": [("2021-06-22", "2021-06-26"), ("2021-07-27", "2021-08-06")],
    "2024": [("2024-06-09", "2024-06-13"), ("2024-06-18", "2024-06-23"),
             ("2024-07-10", "2024-07-19"), ("2024-08-12", "2024-08-16")],
    "2025": [("2025-06-24", "2025-06-28"), ("2025-07-21", "2025-07-27")],
}

# ============================================================
# REGIONS: colours, country groups and masks (as in the original region script)
# ============================================================
REGION_COLORS = {
    "Greece (receptor)":      "#8b0000",
    "Black Sea":              "#6b4c9a",
    "Mediterranean Sea":      "#27c9d8",
    "Red Sea":                "#f4a6c1",
    "Baltic Sea":             "lightblue",
    "Arctic Ocean":           "#b7d4f0",
    "Caspian Sea":            "lightgreen",
    "North Atlantic":         "#8e63b8",
    "Western Europe":         "#ff7f0e",
    "Balkans / SE Europe":    "#808000",
    "NE Europe and Russia":   "#d4d75a",
    "Nordic countries":       "#6c757d",
    "North Africa / Sahara":  "#2ca02c",
    "Turkey / Middle East":   "#d62728",
    "North America":          "#e377c2",
}

COUNTRY_GROUPS = {
    "Western Europe": ["France", "Spain", "Portugal", "Italy", "Switzerland", "Austria", "Germany",
                       "Belgium", "Netherlands", "United Kingdom", "Ireland"],
    "Balkans / SE Europe": ["Slovenia", "Croatia", "Bosnia and Herz.", "Bosnia and Herzegovina", "Serbia",
                            "Republic of Serbia", "Kosovo", "Montenegro", "Albania", "North Macedonia",
                            "Macedonia", "Bulgaria", "Romania", "Hungary", "Moldova"],
    "NE Europe and Russia": ["Poland", "Czechia", "Slovakia", "Ukraine", "Belarus", "Russia"],
    "Nordic countries": ["Norway", "Sweden", "Finland", "Denmark", "Iceland", "Estonia", "Latvia", "Lithuania"],
    "North Africa / Sahara": ["Morocco", "Algeria", "Tunisia", "Libya", "Egypt", "W. Sahara", "Mauritania",
                              "Mali", "Niger", "Chad", "Sudan"],
    "Turkey / Middle East": ["Turkey", "Cyprus", "Syria", "Lebanon", "Israel", "Palestine", "Jordan", "Iraq",
                             "Iran", "Saudi Arabia", "Kuwait", "United Arab Emirates", "Qatar", "Bahrain",
                             "Oman", "Yemen", "Armenia", "Georgia", "Azerbaijan", "Afghanistan", "Pakistan",
                             "Turkmenistan", "Uzbekistan", "Kazakhstan"],
    "North America": ["United States of America", "Canada", "Greenland"],
}


def to_lon180(da, lon_name):
    lon_new = ((da[lon_name] + 180) % 360) - 180
    return da.assign_coords({lon_name: lon_new}).sortby(lon_name)


def get_lon_lat_names(obj):
    lon = "longitude" if "longitude" in obj.coords else "lon"
    lat = "latitude" if "latitude" in obj.coords else "lat"
    return lon, lat


def region_box(ds, lon_min, lon_max, lat_min, lat_max):
    lon2d, lat2d = xr.broadcast(ds["longitude"], ds["latitude"])
    lon2d = lon2d.transpose("latitude", "longitude")
    lat2d = lat2d.transpose("latitude", "longitude")
    return (lon2d >= lon_min) & (lon2d <= lon_max) & (lat2d >= lat_min) & (lat2d <= lat_max)


def country_group_mask(ds, country_names):
    """Mask from real country polygons (Natural Earth), respecting coastlines and borders."""
    shpfile = shpreader.natural_earth(resolution="50m", category="cultural", name="admin_0_countries")
    records = list(shpreader.Reader(shpfile).records())
    lon2d, lat2d = xr.broadcast(ds["longitude"], ds["latitude"])
    LON = lon2d.transpose("latitude", "longitude").values
    LAT = lat2d.transpose("latitude", "longitude").values
    mask_np = np.zeros(LON.shape, dtype=bool)
    try:
        from shapely import contains_xy
        for rec in records:
            if rec.attributes.get("ADMIN", rec.attributes.get("NAME")) in country_names:
                mask_np |= contains_xy(rec.geometry, LON, LAT)
    except ImportError:
        from shapely.vectorized import contains
        for rec in records:
            if rec.attributes.get("ADMIN", rec.attributes.get("NAME")) in country_names:
                mask_np |= contains(rec.geometry, LON, LAT)
    return xr.DataArray(mask_np, coords={"latitude": ds["latitude"], "longitude": ds["longitude"]},
                        dims=("latitude", "longitude"))


def build_land_ocean_mask(land_file, target_ds, source_var=None):
    ds_land = xr.open_dataset(land_file)
    if source_var is None:
        source_var = list(ds_land.data_vars)[0]
    da = ds_land[source_var]
    if "time" in da.dims:
        da = da.isel(time=0, drop=True)
    src_lon, src_lat = get_lon_lat_names(da)
    da = to_lon180(da, src_lon)
    if src_lon != "longitude":
        da = da.rename({src_lon: "longitude"})
    if src_lat != "latitude":
        da = da.rename({src_lat: "latitude"})
    land_src = da.notnull().astype(float)
    land_tgt = land_src.interp(longitude=target_ds["longitude"], latitude=target_ds["latitude"], method="nearest")
    land_mask = land_tgt > 0.5
    ocean_mask = ~land_mask
    ds_land.close()
    return land_mask, ocean_mask


def build_region_masks(ds, land_mask, ocean_mask):
    masks = {}
    assigned = xr.zeros_like(land_mask, dtype=bool)

    def add(name, mask):
        nonlocal assigned
        mask = mask & ~assigned
        masks[name] = mask
        assigned = assigned | mask

    add("Greece (receptor)", region_box(ds, 18.0, 27.5, 34.0, 42.5) & land_mask)       # receptor first
    add("Black Sea", region_box(ds, 27.0, 42.5, 40.0, 47.5) & ocean_mask)
    add("Mediterranean Sea", region_box(ds, -6.0, 37.0, 28.0, 44.0) & ocean_mask)
    add("Red Sea", region_box(ds, 32.0, 44.0, 12.0, 30.0) & ocean_mask)
    add("Baltic Sea", region_box(ds, 9.0, 31.0, 53.0, 66.0) & ocean_mask)
    add("Arctic Ocean", region_box(ds, -85.0, 80.0, 66.0, 90.0) & ocean_mask)
    add("Caspian Sea", region_box(ds, 46.0, 55.5, 36.0, 48.0) & ocean_mask)
    add("North Atlantic", region_box(ds, -85.0, 30.0, 0.0, 66.0) & ocean_mask)
    for region_name, country_names in COUNTRY_GROUPS.items():
        add(region_name, country_group_mask(ds, country_names) & land_mask)
    return masks


# ============================================================
# PLOT SETTINGS, HEATWAVE-DAY MASK, REGION-MASK CACHE
# ============================================================
PROJ = ccrs.Orthographic(central_longitude=-5, central_latitude=20)
DATA_CRS = ccrs.PlateCarree()
HATCH, BAR_W, OFF = "///", 0.34, 0.20

SEA_BOXES = {                 # same boxes as in build_region_masks
    "North Atlantic": (-85.0, 30.0, 0.0, 66.0),
    "Arctic Ocean": (-85.0, 80.0, 66.0, 90.0),
    "Mediterranean Sea": (-6.0, 37.0, 28.0, 44.0),
    "Black Sea": (27.0, 42.5, 40.0, 47.5),
    "Red Sea": (32.0, 44.0, 12.0, 30.0),
    "Baltic Sea": (9.0, 31.0, 53.0, 66.0),
    "Caspian Sea": (46.0, 55.5, 36.0, 48.0),
}

def hw_mask(ds, year):
    """True for releases on heatwave days."""
    t = pd.to_datetime(ds["release_time"].values)
    m = np.zeros(len(t), dtype=bool)
    for start, end in HW_PERIODS[year]:
        m |= (t >= pd.Timestamp(start)) & (t < pd.Timestamp(end) + pd.Timedelta(days=1))
    return m

# ---- masks are slow to build, so they are built once per grid and cached on disk
def _grid_tag(ds):
    lat, lon = ds["latitude"].values, ds["longitude"].values
    return f"{lat.size}x{lon.size}_lat{lat[0]:g}to{lat[-1]:g}_lon{lon[0]:g}to{lon[-1]:g}"


def get_masks(ds):
    f = MASK_CACHE / f"masks_{_grid_tag(ds)}.nc"
    if f.exists():
        ids = xr.open_dataset(f)["region_id"].load()
        return {n: ids == i for i, n in enumerate(ids.attrs["region_names"].split(" | "))}
    land, ocean = build_land_ocean_mask(LAND_FILE, ds)
    masks = build_region_masks(ds, land, ocean)
    ids = xr.full_like(masks[next(iter(masks))], -1, dtype="int8")          # -1 = no region
    for i, n in enumerate(masks):
        ids = ids.where(~masks[n], i)
    out = ids.astype("int8").to_dataset(name="region_id")
    out["region_id"].attrs["region_names"] = " | ".join(masks)
    MASK_CACHE.mkdir(exist_ok=True)
    out.to_netcdf(f)
    print("masks built and cached:", f.name, flush=True)
    return masks


# ============================================================
# COMPUTATION
# ============================================================
def share_table(field, masks):
    """Regional share (%) of a map field summed over the releases."""
    df = pd.DataFrame([dict(region=r, v=float(field.where(m).sum())) for r, m in masks.items()])
    df["pct"] = 100 * df["v"] / df["v"].sum()
    return df


def flux_table(ds, masks):
    """Regional shares of the flux contribution (positive contributions only)"""
    n = ds.sizes["release"]
    sens = CP * ds["dT_map"]
    lat = LV_CSV * ds["dq_map"]

    s_pos = sens.where(sens > 0, 0).sum("release") / n
    l_pos = lat.where(lat > 0, 0).sum("release") / n
    df = pd.DataFrame([dict(region=r, s=float(s_pos.where(m).sum()),
                            l=float(l_pos.where(m).sum())) for r, m in masks.items()])
    tot = (df["s"] + df["l"]).sum()
    df["pct_s"] = 100 * df["s"] / tot
    df["pct_l"] = 100 * df["l"] / tot
    df["pct"] = df["pct_s"] + df["pct_l"]

    s_net = float(sens.sum(("latitude", "longitude")).mean("release"))
    l_net = float(lat.sum(("latitude", "longitude")).mean("release"))
    info = dict(n=n, sens=s_net, lat=l_net, total=s_net + l_net,
                S=100 * s_net / (s_net + l_net), L=100 * l_net / (s_net + l_net))
    return df, info


def ic_table(ds, masks):
    """Regional shares of the initial-condition MSE contribution and the"""
    mse, w = ds["IC_mse_map"], ds["sensitivity_column"]
    valid = np.isfinite(mse) & np.isfinite(w) & (w > 0)
    contrib = (mse * w).where(valid)
    weight = w.where(valid)
    df = share_table(contrib.mean("release"), masks)
    per_release = contrib.sum(("latitude", "longitude")) / weight.sum(("latitude", "longitude"))
    return df, dict(n=ds.sizes["release"], total=float(per_release.mean("release")))


res = {}
for yr in YEARS:
    r = res[yr] = {}
    with xr.open_dataset(FILES[yr]) as ds:
        masks = get_masks(ds)
        hw = hw_mask(ds, yr)
        ds_hd, ds_sd = ds.isel(release=np.where(hw)[0]), ds.isel(release=np.where(~hw)[0])
        r["HD"] = flux_table(ds_hd, masks) if hw.any() else None
        r["SD"] = flux_table(ds_sd, masks) if (~hw).any() else None
        r["SRR"] = share_table(ds_hd["sensitivity_map"].fillna(0).sum("release"), masks) if hw.any() else None
    with xr.open_dataset(IC_FILES[yr]) as ds:
        masks = get_masks(ds)
        hw = hw_mask(ds, yr)
        r["IC_HD"] = ic_table(ds.isel(release=np.where(hw)[0]), masks) if hw.any() else None
        r["IC_SD"] = ic_table(ds.isel(release=np.where(~hw)[0]), masks) if (~hw).any() else None
    print(f"{yr}: done  (HD n = {hw.sum()}, SD n = {(~hw).sum()})")


# ============================================================
# 3) PRINT ONCE
# ============================================================
def regional_pivot(res, key, name, with_split):
    """Regions x years table for one group (HD or SD)."""
    rows = []
    for yr in YEARS:
        if res[yr][key] is None:
            continue
        for _, r in res[yr][key][0].iterrows():
            rows.append(dict(year=yr, region=r["region"], pct=r["pct"],
                             s=r["pct_s"] if with_split else np.nan,
                             l=r["pct_l"] if with_split else np.nan))
    t = pd.DataFrame(rows)
    t["cell"] = (t["s"].round(0).astype("Int64").astype(str) + "+" + t["l"].round(0).astype("Int64").astype(str)
                 if with_split else t["pct"].round(0).astype("Int64").astype(str))
    order = t.groupby("region")["pct"].mean().sort_values(ascending=False)
    piv = t.pivot(index="region", columns="year", values="cell").loc[order[order >= 0.5].index]
    print(f"\n{name}")
    print(piv.to_string())
    #t.drop(columns="cell").to_csv(f"regional_shares_{key}.csv", index=False)


def print_tables(res):
    # --- flux totals and sensible / latent split
    rows = []
    for yr in YEARS:
        for g in ("SD", "HD"):
            if res[yr][g] is None:
                continue
            i = res[yr][g][1]
            rows.append(dict(year=yr, group=g, n_releases=i["n"], total_Jkg=i["total"],
                             sensible_Jkg=i["sens"], latent_Jkg=i["lat"],
                             sensible_pct=i["S"], latent_pct=i["L"]))
    tot = pd.DataFrame(rows)
    print("\nMEAN FLUX CONTRIBUTION PER RELEASE (net, J/kg) AND SENSIBLE / LATENT SHARE (%)")
    print(tot.round(1).to_string(index=False))
    #tot.to_csv("flux_totals.csv", index=False)

    # --- initial-condition MSE totals
    ic = pd.DataFrame([dict(year=yr, group=g, mean_IC_MSE_Jkg=res[yr][f"IC_{g}"][1]["total"])
                       for yr in YEARS for g in ("SD", "HD") if res[yr][f"IC_{g}"] is not None])
    piv = ic.pivot(index="year", columns="group", values="mean_IC_MSE_Jkg")
    piv["HD-SD"] = piv["HD"] - piv["SD"]
    print("\nMEAN INITIAL-CONDITION MSE (J/kg)")
    print(piv.round(0).to_string())
#piv.to_csv("ic_mse_totals.csv")

    # --- regional shares
    regional_pivot(res, "HD", "FLUX, REGIONAL SHARES, HEATWAVE DAYS  (% sensible + % latent)", True)
    regional_pivot(res, "SD", "FLUX, REGIONAL SHARES, SUMMER DAYS  (% sensible + % latent)", True)
    regional_pivot(res, "IC_HD", "INITIAL MSE, REGIONAL SHARES, HEATWAVE DAYS (%)", False)
    regional_pivot(res, "IC_SD", "INITIAL MSE, REGIONAL SHARES, SUMMER DAYS (%)", False)


print_tables(res)


# ============================================================
# 4) PLOTTING
# ============================================================
def pick_regions(dfs, min_pct=1.0):
    """Regions above min_pct in at least one bar, ordered small -> large."""
    s = pd.concat([d.set_index("region")["pct"] for d in dfs], axis=1).fillna(0)
    keep = s[(s > min_pct).any(axis=1)]
    return keep.mean(axis=1).sort_values().index.tolist()


def draw_bar(ax, x, spec, regs):

    kind, df, info = spec
    bottom = 0.0
    for reg in regs:
        row = df[df["region"] == reg].iloc[0]
        parts = [(row["pct_s"], None), (row["pct_l"], HATCH)] if kind == "flux" else [(row["pct"], None)]
        for h, hatch in parts:
            if h <= 0:
                continue
            ax.bar(x, h, bottom=bottom, width=BAR_W, color=REGION_COLORS.get(reg, "lightgrey"),
                   hatch=hatch, alpha=0.75 if hatch else 1.0, edgecolor="white",
                   linewidth=0.4, zorder=2)
            if h >= 5:
                ax.text(x, bottom + h / 2, f"{h:.0f}", ha="center", va="center", fontsize=8,
                        color="black" if hatch else "white")
            bottom += h
    if info is not None:                                   # number (+ S/L split) above the bar
        val = info["total"] / (1000 if TOP_IN_KJ else 1)
        txt = f"{val:{'.1f' if TOP_IN_KJ else '.0f'}}"
        if "S" in info:
            txt += f"\nS {info['S']:.0f}%\nL {info['L']:.0f}%"
        ax.text(x, bottom + 1.5, txt, ha="center", va="bottom", fontsize=9, linespacing=1.25)


def draw_region_map(ax, extent=(-55, 45, 20, 72)):
    ax.set_extent(extent, crs=DATA_CRS)
    ax.add_feature(cfeature.OCEAN, facecolor="aliceblue", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)
    for name, (x0, x1, y0, y1) in SEA_BOXES.items():
        ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=REGION_COLORS[name],
                                        edgecolor="none", alpha=0.85, transform=DATA_CRS, zorder=1))
    country_to_region = {c: r for r, cs in COUNTRY_GROUPS.items() for c in cs}
    receptor = box(18.0, 34.0, 27.5, 42.5)
    records = shpreader.Reader(shpreader.natural_earth("50m", "cultural", "admin_0_countries")).records()
    for rec in records:
        name = rec.attributes.get("ADMIN", rec.attributes.get("NAME"))
        if name in country_to_region:
            ax.add_geometries([rec.geometry], DATA_CRS, edgecolor="none", alpha=0.95, zorder=2,
                              facecolor=REGION_COLORS[country_to_region[name]])
        if name in ("Greece", "Turkey", "Albania", "North Macedonia", "Bulgaria"):
            g = rec.geometry.intersection(receptor)
            if not g.is_empty:
                ax.add_geometries([g], DATA_CRS, edgecolor="none", alpha=0.95, zorder=3,
                                  facecolor=REGION_COLORS["Greece (receptor)"])
    ax.add_patch(mpatches.Rectangle((18.0, 34.0), 9.5, 8.5, fill=False, edgecolor="black",
                                    linewidth=1.6, transform=DATA_CRS, zorder=5))
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.35, linestyle=":", zorder=4)


def panel_label(ax, text):
  
    ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(0, 6),
                textcoords="offset points", fontsize=14, ha="left", va="bottom")


def make_figure(left, right, left_label, right_label, type_handles, out, ylabel="Share of total (%)"):
   
    regs = pick_regions([s[1] for d in (left, right) for s in d.values() if s is not None])

    fig = plt.figure(figsize=(17, 8.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1], height_ratios=[1, 1.4], wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[:, 0])
    axm = fig.add_subplot(gs[0, 1], projection=PROJ)
    axl = fig.add_subplot(gs[1, 1])
    axm.set_anchor("N")                                   # map top = top of the bar panel

    x = np.arange(len(YEARS))
    for xi, yr in enumerate(YEARS):
        for d, dx in ((left, -OFF), (right, OFF)):
            if d[yr] is not None:
                draw_bar(ax, xi + dx, d[yr], regs)
        ax.text(xi - OFF, -5, left_label, ha="center", va="top", fontsize=13)
        ax.text(xi + OFF, -5, right_label, ha="center", va="top", fontsize=13)
    ax.set_xlim(-0.5, len(YEARS) - 0.5)                   # every year gets the same width
    ax.set_xticks(x)
    ax.set_xticklabels(YEARS, fontsize=12)
    ax.tick_params(axis="x", pad=36)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_ylim(0, 132)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    for xv in x[:-1] + 0.5:
        ax.axvline(xv, color="k", linewidth=0.5, alpha=0.2)

    draw_region_map(axm)
    panel_label(ax, "a)")
    panel_label(axm, "b)")

    axl.axis("off")
    reg_handles = [mpatches.Patch(facecolor=REGION_COLORS.get(r, "lightgrey"), edgecolor="grey",
                                  linewidth=0.4, label=r) for r in reversed(regs)]
    leg = axl.legend(handles=reg_handles, loc="upper left", bbox_to_anchor=(0, 1.0),
                     fontsize=10, title="Region", title_fontsize=11, frameon=True)
    axl.add_artist(leg)
    axl.legend(handles=type_handles, loc="lower left", bbox_to_anchor=(0, 0.0), fontsize=10)

    fig.savefig(out + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    
    plt.show()
    return fig


def spec(kind, table):
    """(kind, table, info) for draw_bar, or None if the group has no releases."""
    if table is None:
        return None
    if kind == "ssr":
        return ("ssr", table, None)
    return (kind, table[0], table[1])


grey = dict(facecolor="grey", edgecolor="white")
sd_hd = [mpatches.Patch(facecolor="white", edgecolor="white", label="SD: summer days"),
         mpatches.Patch(facecolor="white", edgecolor="white", label="HD: heatwave days")]
ylab_flux = "Share of total flux contribution (%)"

# ---- Figure A: SRR vs flux, heatwave releases
make_figure({yr: spec("ssr", res[yr]["SRR"]) for yr in YEARS},
            {yr: spec("flux", res[yr]["HD"]) for yr in YEARS},
            "SRR", "Flux",
            [mpatches.Patch(**grey, label="SRR (transport)"),
             mpatches.Patch(**grey, label="Flux: sensible"),
             mpatches.Patch(**grey, hatch=HATCH, alpha=0.75, label="Flux: latent")],
            "fig_A_ssr_vs_flux_map", ylabel="Share of total (%)")

# ---- Figure B: flux, summer days vs heatwave days
make_figure({yr: spec("flux", res[yr]["SD"]) for yr in YEARS},
            {yr: spec("flux", res[yr]["HD"]) for yr in YEARS},
            "SD", "HD",
            [mpatches.Patch(**grey, label="Sensible"),
             mpatches.Patch(**grey, hatch=HATCH, alpha=0.75, label="Latent")] + sd_hd,
            "fig_B_flux_sd_vs_hd_map", ylabel=ylab_flux)

# ---- Figure C: initial-condition MSE, summer days vs heatwave days
make_figure({yr: spec("ic", res[yr]["IC_SD"]) for yr in YEARS},
            {yr: spec("ic", res[yr]["IC_HD"]) for yr in YEARS},
            "SD", "HD", sd_hd,
            "fig_C_ic_mse_sd_vs_hd_map", ylabel="Share of initial-condition MSE (%)")