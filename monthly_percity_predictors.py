"""
For each city and season, extract the vegetation series and predictors during & after the heat wave (the same for compound hot and dry).

Separately, for each season, predictors:

// - Pre-existing vegetation state is already in the normalized metric

The resistance & recovery metrics are at the beginning.

PIXEL LEVEL | CITY LEVEL

* Predictors that only vary from city to city

  (fid)
  * Size of the city (log-transform for regression)

  use x (season x fid)
  * Seasonal background climatology of precipitation & temperature (1981-2010)

* Predictors that are static but vary by pixels

  use x (fid x start x end x row x col) | use x (fid x start x end x [urban/rural])
  * Pixel temperature anomalies relative to the city average (UHI)

  use x (fid x start x end x row x col) | use x (fid x start x end x [urban/rural])
  * Fraction of impervious area of the pixel
  * Fraction of land cover types of the pixel (excluding "Developed")

  (fid x row x col) | (fid x diff)
  * Elevation of the pixel

* Predictors that are different for each heat wave

  use x (fid x start x end)
  * Intensity & duration of the heat wave

  use x (fid x start x end x row x col) | use x (fid x start x end x [urban/rural])
  * Pixel level SPI/VPD during/after the heat wave

* Extras

  (season x fid x row x col x use)
  * Gap to optimal temperature

  (season x fid x row x col) | (season x fid x [urban/rural])
  * Pixel level sensitivity to water limitation of different vegetation types and across the regions

"""
import pandas as pd
import xarray as xr
import os
import numpy as np
from utils.analysis import *
from utils.constants import *
from utils.paths import *
from utils.extremes import *
from utils.get_monthly_data import *
from utils.plotting import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
import seaborn as sns
from tqdm import tqdm
import cartopy.crs as ccrs
import itertools as it
from scipy.stats import mannwhitneyu


rcParams["font.size"] = 6.0
rcParams["axes.titlesize"] = 6.0


######################################################################################
# PIXEL LEVEL
######################################################################################
# ------------------------------------------------------------------------------------
# Size of the city
# ------------------------------------------------------------------------------------'
data = pd.DataFrame(np.nan, index=range(85), columns=["pixel"])
for fid in range(85):
    mask = get_mask(fid, "core", True, Setup().extent)
    data.loc[fid] = np.nansum(mask)
data.astype(int).to_csv(os.path.join(path_out, "uhi", "urban_size.csv"))


# ------------------------------------------------------------------------------------
# Seasonal background climatology of precipitation & temperature (1981-2010, daymet)
# ------------------------------------------------------------------------------------
# First generate the time series
for fid, use in tqdm(it.product(range(85), ["daymet", "topowx", "yyz"])):
    b = BackgroundClim(fid, use)
    data = b.calc()
    data.to_csv(os.path.join(path_out, "clim", "time_series", f"{use}_{fid}.csv"))

# Then convert to climatology (Daymet)
setup = Setup("daymet")
data = pd.DataFrame(
    np.nan,
    index=pd.MultiIndex.from_product([["DJF", "MAM", "JJA", "SON"], range(85)]),
    columns=["prcp", "tmean"],
)
for fid in range(85):
    temp = pd.read_csv(
        os.path.join(path_out, "clim", "time_series", f"daymet_{fid}.csv"),
        index_col=0,
        header=[0, 1],
        parse_dates=True,
    )
    temp = temp.loc[(temp.index.year >= 1981) & (temp.index.year <= 2010), :]
    temp = temp.groupby(temp.index.to_period("Q-NOV").quarter).mean()
    for i, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
        data.loc[(season, fid), "prcp"] = temp.loc[i + 1, ("prcp", "all")]
        data.loc[(season, fid), "tmean"] = (
            temp.loc[i + 1, ("tmax", "all")] + temp.loc[i + 1, ("tmin", "all")]
        ) / 2
data.to_csv(
    os.path.join(path_out, "clim", f"clim_{setup.format_prefix_noveg_noevent()}.csv")
)


fig, axes = plt.subplots(
    4,
    2,
    figsize=(10, 10),
    sharex=True,
    sharey=True,
    subplot_kw={
        "projection": ccrs.AlbersEqualArea(central_longitude=-100, central_latitude=35)
    },
)
fig.subplots_adjust(hspace=0.01, wspace=0.01)
for i, varname in enumerate(["tmean", "prcp"]):
    for j, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
        ax = axes[j, i]
        if varname == "tmean":
            vmin = 5
            vmax = 25
        else:
            vmin = 0.3
            vmax = 4.7
        map = MapOfColors("point", data.loc[season, varname])
        cf = map.plot(
            inset_bar=False, ax=ax, map_args=dict(vmin=vmin, vmax=vmax, cmap=cmap_div())
        )
        ax.set_title(varname)
        plt.colorbar(cf, ax=ax, orientation="horizontal")
fig.savefig(
    os.path.join(
        path_out, "clim", "plots", f"clim_{setup.format_prefix_noveg_noevent()}.png"
    ),
    dpi=600.0,
    bbox_inches="tight",
)
plt.close(fig)


# ------------------------------------------------------------------------------------
# Fraction of impervious area of the pixel
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Fraction of land cover types of the pixel
# ------------------------------------------------------------------------------------
for use in ["daymet", "topowx", "yyz"]:
    setup = Setup(use)
    with pd.HDFStore(
        os.path.join(
            path_out, "luc", f"percity_per_pixel_{setup.format_prefix_noveg()}.h5"
        ),
        mode="w",
    ) as h:
        with pd.HDFStore(
            os.path.join(
                path_out, "extreme_events", f"percity_{setup.format_prefix_noveg()}.h5"
            ),
            mode="r",
        ) as hx:
            for ex in ["heat_wave", "hot_and_dry"]:
                extreme_events = hx.select(ex)
                for fid in tqdm(range(85)):
                    u = LUC(fid)
                    data = u.get_pixel(extreme_events.loc[[fid], :])
                    h.append(ex, data)

    # Plot the per pixel impervious area and land cover percenages.
    with pd.HDFStore(
        os.path.join(
            path_out, "luc", f"percity_per_pixel_{setup.format_prefix_noveg()}.h5"
        ),
        mode="r",
    ) as h:
        for fid in range(85):
            df = (
                h.select("heat_wave")
                .loc[fid, :]
                .reset_index()
                .groupby(["row", "col"])
                .mean()
            )

            fig, axes = plt.subplots(
                3, 3, figsize=(15, 15), subplot_kw={"projection": crs_daymet}
            )
            for i, varname in enumerate(
                ["impervious_frac"] + sorted(modis_luc_agg_names.keys())
            ):
                ax = axes.flat[i]

                if varname == "impervious_frac":
                    temp = (
                        df[[varname]] / 100.0
                    )  # convert to fraction for ease of plotting
                    vmin = 0
                    vmax = 0.8
                else:
                    temp = df[[varname]]
                    vmin = 0
                    vmax = 0.8

                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

                temp = add_grid_coords(fid, temp, setup.extent)

                cf = ax.scatter(
                    temp["x"].values,
                    temp["y"].values,
                    c=temp[varname].values,
                    s=10,
                    lw=0.0,
                    marker="s",
                    transform=crs_daymet,
                    cmap="Reds",
                    norm=norm,
                )
                add_core_boundary(ax, fid, "k")
                if varname == "impervious_frac":
                    ax.set_title(varname)
                else:
                    ax.set_title(modis_luc_agg_names[varname])
            cax = fig.add_axes([0.1, 0.05, 0.8, 0.01])
            cb = plt.colorbar(
                cf, cax=cax, shrink=0.5, label=varname, orientation="horizontal"
            )
            fig.savefig(
                os.path.join(
                    path_out,
                    "luc",
                    "plots",
                    f"percity_per_pixel_{setup.format_prefix_noveg()}_{fid}.png",
                ),
                dpi=600.0,
                bbox_inches="tight",
            )
            plt.close(fig)


# ------------------------------------------------------------------------------------
# Elevation of the pixel
# ------------------------------------------------------------------------------------
setup = Setup("daymet")
with pd.HDFStore(
    os.path.join(path_out, "elev", f"percity_per_pixel_{setup.extent}.h5"), mode="w"
) as h:
    for fid in tqdm(range(85)):
        u = Elevation(fid)
        data = u.get_pixel()
        h.append("elev", data)

with pd.HDFStore(
    os.path.join(path_out, "elev", f"percity_per_pixel_{setup.extent}.h5"), mode="r"
) as h:
    for fid in range(85):
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": crs_daymet})
        temp = add_grid_coords(fid, h.select("elev").loc[fid, :], setup.extent)

        norm = mpl.colors.SymLogNorm(
            linthresh=100, vmin=temp["elev"].min(), vmax=temp["elev"].max()
        )

        cf = ax.scatter(
            temp["x"].values,
            temp["y"].values,
            c=temp["elev"].values,
            s=10,
            lw=0.0,
            marker="s",
            transform=crs_daymet,
            cmap="copper_r",
            norm=norm,
        )
        add_core_boundary(ax, fid, "k")
        ax.set_title("Elevation (m)")
        cax = fig.add_axes([0.1, 0.05, 0.8, 0.01])
        cb = plt.colorbar(cf, cax=cax, shrink=0.5, orientation="horizontal")
        fig.savefig(
            os.path.join(
                path_out, "elev", "plots", f"percity_per_pixel_{setup.extent}_{fid}.png"
            ),
            dpi=600.0,
            bbox_inches="tight",
        )
        plt.close(fig)


# ------------------------------------------------------------------------------------
# Intensity & duration of the heat wave
# ------------------------------------------------------------------------------------
for use in ["daymet", "topowx", "yyz"]:
    setup = Setup(use)
    with pd.HDFStore(
        os.path.join(
            path_out, "extreme_events", f"percity_{setup.format_prefix_noveg()}.h5"
        ),
        mode="w",
    ) as h:
        for fid in tqdm(range(85)):
            m = Events(fid)
            heat_wave, hot_and_dry = m.get_spatial_avg()

            h.append("heat_wave", heat_wave)
            h.append("hot_and_dry", hot_and_dry)

    # Plot the statistical distribution of the duration and intensity of the extreme events.
    with pd.HDFStore(
        os.path.join(
            path_out, "extreme_events", f"percity_{setup.format_prefix_noveg()}.h5"
        ),
        mode="r",
    ) as h:
        for ex in ["heat_wave", "hot_and_dry"]:
            df = h.select(ex)

            df["season"] = pd.DatetimeIndex(df.reset_index()["end"]).month.map(
                month_to_season
            )

            temp = []
            for i in df.drop(["duration", "season"], axis=1).columns:
                temp.append(
                    pd.DataFrame({"val": df[i], "metric": i, "season": df["season"]})
                )

            new_df = pd.concat(
                temp
                + [
                    pd.DataFrame(
                        {
                            "val": df["duration"],
                            "season": df["season"],
                            "metric": "duration",
                        }
                    )
                ],
                axis=0,
            ).reset_index(drop=True)

            fgd = sns.displot(
                data=new_df, x="val", col="metric", row="season", kde=True
            )
            fgd.savefig(
                os.path.join(
                    path_out,
                    "extreme_events",
                    "plots",
                    f"percity_{setup.format_prefix_noveg()}_dist_of_{ex}_metrics.png",
                ),
                dpi=600.0,
            )

    # Plot the statistical distribution of the length of inter-event intervals of the extreme events.
    with pd.HDFStore(
        os.path.join(
            path_out, "extreme_events", f"percity_{setup.format_prefix_noveg()}.h5"
        ),
        mode="r",
    ) as h:
        fig, axes = plt.subplots(4, 2, figsize=(10, 15), sharex=False, sharey=False)
        for i, ex in enumerate(["heat_wave", "hot_and_dry"]):
            df["season"] = pd.DatetimeIndex(df.reset_index()["end"]).month.map(
                month_to_season
            )

            for j, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
                ax = axes[j, i]

                temp = df.loc[df["season"] == season, :]

                interval = []
                for fid in range(85):
                    if fid in temp.index.get_level_values(0):
                        interval += [
                            relativedelta(a, b).years * 12 + relativedelta(a, b).months
                            for a, b in zip(
                                pd.DatetimeIndex(
                                    temp.loc[fid, :].reset_index()["start"]
                                )[1:],
                                pd.DatetimeIndex(temp.loc[fid, :].reset_index()["end"])[
                                    :-1
                                ],
                            )
                        ]

                sns.histplot(x=interval, kde=True, ax=ax)
                if i == 0:
                    ax.set_ylabel(season)
                if j == 0:
                    ax.set_title(ex)
        fig.savefig(
            os.path.join(
                path_out,
                "extreme_events",
                "plots",
                f"percity_{setup.format_prefix_noveg()}_dist_of_intervals.png",
            ),
            dpi=600.0,
        )
        plt.close(fig)

    # Plot map of the number of city-level extreme events that have occurred.
    lab = "abcdefghijkopqrstuvwxyz"

    ex = "heat_wave"
    with pd.HDFStore(
        os.path.join(
            path_out, "extreme_events", f"percity_{setup.format_prefix_noveg()}.h5"
        ),
        mode="r",
    ) as h:
        df = h.select(ex)
        df["season"] = pd.DatetimeIndex(df.reset_index()["end"]).month.map(
            month_to_season
        )

    fig, axes = plt.subplots(
        4,
        3,
        figsize=(10, 10),
        sharex=True,
        sharey=True,
        subplot_kw={
            "projection": ccrs.AlbersEqualArea(
                central_longitude=-100, central_latitude=35
            )
        },
    )
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    for j, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
        temp = df.loc[df["season"] == season, :]

        # number of events
        n_events = temp.groupby(level=0).count().iloc[:, 0]
        for fid in range(85):
            if not fid in n_events.index:
                n_events.loc[fid] = 0
        ax = axes[j, 0]
        vmin = -0.5
        vmax = 7.5
        map = MapOfColors("point", n_events)
        cf = map.plot(
            inset_bar=False,
            ax=ax,
            point_scale=0.4,
            map_args=dict(vmin=vmin, vmax=vmax, cmap="RdYlBu_r"),
        )
        if j == 0:
            ax.set_title("Number of events")
        ax.text(0.05, 1.05, lab[j * 3], fontweight="bold", transform=ax.transAxes)
        ax.text(-0.05, 0.5, season, rotation=90, transform=ax.transAxes)
        cax = fig.add_axes([0.1, 0.1, 0.25, 0.005])
        plt.colorbar(cf, cax=cax, orientation="horizontal", ticks=range(8))

        # intensity of the events
        intensity = temp.groupby(level=0).mean()["intensity"]
        for fid in range(85):
            if not fid in intensity.index:
                intensity.loc[fid] = 0
        ax = axes[j, 1]
        vmin = 0.0
        vmax = 4.0
        map = MapOfColors("point", intensity)
        cf = map.plot(
            inset_bar=False,
            ax=ax,
            point_scale=0.4,
            map_args=dict(vmin=vmin, vmax=vmax, cmap="RdYlBu_r"),
        )
        if j == 0:
            ax.set_title("Intensity ($^o$C)")
        ax.text(0.05, 1.05, lab[j * 3 + 1], fontweight="bold", transform=ax.transAxes)
        cax = fig.add_axes([0.38, 0.1, 0.25, 0.005])
        plt.colorbar(cf, cax=cax, orientation="horizontal")

        # duration of the events
        duration = temp.groupby(level=0).mean()["duration"]
        for fid in range(85):
            if not fid in duration.index:
                duration.loc[fid] = 0
        ax = axes[j, 2]
        vmin = 0
        vmax = 6
        map = MapOfColors("point", duration)
        cf = map.plot(
            inset_bar=False,
            ax=ax,
            point_scale=0.4,
            map_args=dict(vmin=vmin, vmax=vmax, cmap="RdYlBu_r"),
        )
        if j == 0:
            ax.set_title("Duration (months)")
        ax.text(0.05, 1.05, lab[j * 3 + 2], fontweight="bold", transform=ax.transAxes)
        cax = fig.add_axes([0.67, 0.1, 0.25, 0.005])
        plt.colorbar(cf, cax=cax, orientation="horizontal")
    fig.savefig(
        os.path.join(
            path_out,
            "extreme_events",
            "plots",
            f"percity_{setup.format_prefix_noveg()}_events_map_{ex}.png",
        ),
        dpi=600.0,
        bbox_inches="tight",
    )
    plt.close(fig)

    ex = "hot_and_dry"
    with pd.HDFStore(
        os.path.join(
            path_out, "extreme_events", f"percity_{setup.format_prefix_noveg()}.h5"
        ),
        mode="r",
    ) as h:
        df = h.select(ex)
        df["season"] = pd.DatetimeIndex(df.reset_index()["end"]).month.map(
            month_to_season
        )

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(17, 12),
        sharex=True,
        sharey=True,
        subplot_kw={
            "projection": ccrs.AlbersEqualArea(
                central_longitude=-100, central_latitude=35
            )
        },
    )
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    for j, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
        temp = df.loc[df["season"] == season, :]

        # number of events
        n_events = temp.groupby(level=0).count().iloc[:, 0]
        for fid in range(85):
            if not fid in n_events.index:
                n_events.loc[fid] = 0
        ax = axes[j, 0]
        vmin = -0.5
        vmax = 7.5
        map = MapOfColors("point", n_events)
        cf = map.plot(
            inset_bar=False, ax=ax, map_args=dict(vmin=vmin, vmax=vmax, cmap="RdYlBu_r")
        )
        if j == 0:
            ax.set_title("Number of events")
        ax.text(-0.2, 0.5, season, rotation=90, transform=ax.transAxes)
        cax = fig.add_axes([0.05, 0.1, 0.2, 0.005])
        plt.colorbar(cf, cax=cax, orientation="horizontal", ticks=range(10))

        # intensity of the events
        intensity = temp.groupby(level=0).mean()["hot_intensity"]
        for fid in range(85):
            if not fid in intensity.index:
                intensity.loc[fid] = 0
        ax = axes[j, 1]
        vmin = 0.0
        vmax = 4.0
        map = MapOfColors("point", intensity)
        cf = map.plot(
            inset_bar=False, ax=ax, map_args=dict(vmin=vmin, vmax=vmax, cmap="RdYlBu_r")
        )
        if j == 0:
            ax.set_title("Hot intensity ($^o$C)")
        cax = fig.add_axes([0.27, 0.1, 0.2, 0.005])
        plt.colorbar(cf, cax=cax, orientation="horizontal")

        intensity = temp.groupby(level=0).mean()["dry_intensity"]
        for fid in range(85):
            if not fid in intensity.index:
                intensity.loc[fid] = 0
        ax = axes[j, 2]
        vmin = -3
        vmax = 3
        map = MapOfColors("point", intensity)
        cf = map.plot(
            inset_bar=False, ax=ax, map_args=dict(vmin=vmin, vmax=vmax, cmap="RdYlBu")
        )
        if j == 0:
            ax.set_title("Dry intensity (1)")
        cax = fig.add_axes([0.51, 0.1, 0.2, 0.005])
        plt.colorbar(cf, cax=cax, orientation="horizontal")

        # duration of the events
        duration = temp.groupby(level=0).mean()["duration"]
        for fid in range(85):
            if not fid in duration.index:
                duration.loc[fid] = 0
        ax = axes[j, 3]
        vmin = 0
        vmax = 10
        map = MapOfColors("point", duration)
        cf = map.plot(
            inset_bar=False, ax=ax, map_args=dict(vmin=vmin, vmax=vmax, cmap="RdYlBu_r")
        )
        if j == 0:
            ax.set_title("Duration (months)")
        cax = fig.add_axes([0.75, 0.1, 0.2, 0.005])
        plt.colorbar(cf, cax=cax, orientation="horizontal")

    fig.savefig(
        os.path.join(
            path_out,
            "extreme_events",
            "plots",
            f"percity_{setup.format_prefix_noveg()}_events_map_{ex}.png",
        ),
        dpi=600.0,
        bbox_inches="tight",
    )
    plt.close(fig)

    #
    lab = "abcdefghijkopqrstuvwxyz"

    ex = "heat_wave"
    season = "JJA"

    with pd.HDFStore(
        os.path.join(
            path_out, "extreme_events", f"percity_{setup.format_prefix_noveg()}.h5"
        ),
        mode="r",
    ) as h:
        df = h.select(ex)
        df["season"] = pd.DatetimeIndex(df.reset_index()["end"]).month.map(
            month_to_season
        )
    temp = df.loc[df["season"] == season, :]

    fig, ax = plt.subplots(
        figsize=(4, 3),
        sharex=True,
        sharey=True,
        subplot_kw={
            "projection": ccrs.AlbersEqualArea(
                central_longitude=-100, central_latitude=35
            )
        },
    )
    # count the number of events that are intensity >= 2 degC, and duration >= 4 months.
    pct_long_and_intense_events = (
        (temp["intensity"] >= 2) & (temp["duration"] >= 4)
    ).groupby(level=0).mean() * 100.0
    for fid in range(85):
        if not fid in pct_long_and_intense_events.index:
            pct_long_and_intense_events.loc[fid] = 0
    map = MapOfColors("point", pct_long_and_intense_events)
    cf = map.plot(
        inset_bar=False,
        ax=ax,
        point_scale=0.4,
        map_args=dict(vmin=0, vmax=60.0, cmap="viridis"),
    )
    ax.text(-0.05, 0.5, season, rotation=90, transform=ax.transAxes)
    cax = fig.add_axes([0.1, 0.1, 0.8, 0.01])
    plt.colorbar(
        cf,
        cax=cax,
        orientation="horizontal",
        label="Percentage long and intense events",
    )
    fig.savefig(
        os.path.join(
            path_out,
            "extreme_events",
            "plots",
            f"percity_{setup.format_prefix_noveg()}_long_and_intense_events_map_{ex}.png",
        ),
        dpi=600.0,
        bbox_inches="tight",
    )
    plt.close(fig)


# ------------------------------------------------------------------------------------
# Pixel temperature anomalies relative to the city average (of_season)
# Pixel level SPI & VPD during/after the heat wave (in_event, post_event)
# ------------------------------------------------------------------------------------
# > ~ 1 hour for each variable
for varname, use in it.product(
    ["tmax", "tmin", "spi", "vpd"], ["daymet", "topowx", "yyz"]
):
    setup = Setup(use)
    with pd.HDFStore(
        os.path.join(
            path_out,
            "clim",
            f"percity_per_pixel_{varname}_{setup.format_prefix_noveg()}.h5",
        ),
        mode="w",
    ) as h:
        with pd.HDFStore(
            os.path.join(
                path_out, "extreme_events", f"percity_{setup.format_prefix_noveg()}.h5"
            ),
            mode="r",
        ) as hx:
            for ex in ["heat_wave", "hot_and_dry"]:
                extreme_events = hx.select(ex)
                for fid in tqdm(range(85)):
                    m = Events(fid)
                    var = m.get_aux_pixel(varname, extreme_events.loc[[fid], :])
                    h.append(ex, var)


# Plot the per pixel day- and nighttime temperature during the heat waves and of the average of the season.
for use, fid in it.product(["daymet", "topowx", "yyz"], range(85)):
    setup = Setup(use)
    fig, axes = plt.subplots(
        4, 4, figsize=(20, 20), subplot_kw={"projection": crs_daymet}
    )
    for i, varname in enumerate(["tmax", "tmin", "vpd", "spi"]):
        if varname in ["tmax", "tmin"]:
            vmin = -1.5
            vmax = 1.5
        elif varname == "spi":
            vmin = -3
            vmax = 3
        elif varname == "vpd":
            vmin = -250.0
            vmax = 250
        else:
            raise "Not implemented"

        with pd.HDFStore(
            os.path.join(
                path_out,
                "clim",
                f"percity_per_pixel_{varname}_{setup.format_prefix_noveg()}.h5",
            ),
            mode="r",
        ) as h:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

            for j, ex in enumerate(["heat_wave", "hot_and_dry"]):
                for k, tt in enumerate(["in_event", "post_event"]):
                    ax = axes[i, j * 2 + k]

                    temp = (
                        h.select(ex)
                        .loc[fid, [tt]]
                        .reset_index()
                        .groupby(["row", "col"])
                        .mean()
                    )
                    temp = add_grid_coords(fid, temp, setup.extent)

                    cf = ax.scatter(
                        temp["x"].values,
                        temp["y"].values,
                        c=temp[tt].values,
                        s=10,
                        lw=0.0,
                        marker="s",
                        transform=crs_daymet,
                        cmap="RdYlBu_r",
                        norm=norm,
                    )
                    add_core_boundary(ax, fid, "k")
                    if i == 0:
                        ax.set_title(f"{ex} {tt}")
                    if (j == 0) & (k == 0):
                        ax.text(
                            -0.2, 0.5, f"{varname}", transform=ax.transAxes, rotation=90
                        )
                    elif (j * 2 + k) == 3:
                        cax = fig.add_axes([0.05 + 0.24 * i, 0.05, 0.22, 0.01])
                        cb = plt.colorbar(
                            cf,
                            cax=cax,
                            shrink=0.5,
                            label=varname,
                            orientation="horizontal",
                        )
    fig.savefig(
        os.path.join(
            path_out,
            "clim",
            "plots",
            f"percity_per_pixel_{setup.format_prefix_noveg()}_{fid}.png",
        ),
        dpi=600.0,
        bbox_inches="tight",
    )
    plt.close(fig)


# ------------------------------------------------------------------------------------
# Pixel level optimal temperature, and seasonal gap to optimal temperatures
# ------------------------------------------------------------------------------------
s = SeasonalAvgTemperaturePixel()
s.calc()

s = OptimalTPixel()
s.calc()

# ------------------------------------------------------------------------------------
# Pixel level sensitivity to SPI & VPD in each season
# ------------------------------------------------------------------------------------
s = SensitivityWaterPixel()
s.calc()

lab = "abcdefghijklmnopqrst"
h = pd.HDFStore(
    os.path.join(
        path_out, "veg", f"{s.prefix}_{s.extent}_{s.name}_sensitivity_water.h5"
    ),
    mode="r",
)
for varname in ["vpd", "spi"]:
    corr = h.select(f"corr_{varname}").copy(deep=True)
    pval = h.select(f"pval_{varname}").copy(deep=True)

    corr = pd.DataFrame(
        np.where(pval <= 0.05, corr, 0.0), index=corr.index, columns=corr.columns
    )
    corr = add_urlabel_all(corr, s.extent).reset_index()
    corr_mean = corr.groupby(["location", "fid"]).mean()

    pval = add_urlabel_all(pval <= 0.05, s.extent).reset_index()
    corr_sig = (
        pval.groupby(["location", "fid"]).mean() >= 0.5
    )  # at least 50% significant

    def sig(gdf):
        pvalues = pd.Series(np.nan, ["DJF", "MAM", "JJA", "SON"])
        for season in ["DJF", "MAM", "JJA", "SON"]:
            pvalues.loc[season] = mannwhitneyu(
                gdf.loc[gdf["location"] == "urban", season].values,
                gdf.loc[gdf["location"] == "rural", season].values,
            ).pvalue
        return pvalues

    sig_diff = corr.groupby("fid").apply(sig)

    fig, axes = plt.subplots(
        4,
        3,
        figsize=(15, 13),
        subplot_kw={
            "projection": ccrs.AlbersEqualArea(
                central_longitude=-100, central_latitude=35
            )
        },
    )
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    for j, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
        for i, location in enumerate(["urban", "rural", "urban-rural"]):
            ax = axes[j, i]

            bag = {"color": ["#313695", "#4575b4", "#a50026", "#d73027"]}

            if i < 2:
                general = pd.DataFrame(
                    {
                        "value": corr_mean.loc[location, season],
                        "sig": corr_sig.loc[location, season],
                    }
                )
                if varname == "spi":
                    mag = {
                        "cmap": "RdBu",
                        "norm": BoundaryNorm(
                            np.linspace(-0.2, 0.2, 9), ncolors=256, extend="both"
                        ),
                    }
                else:
                    mag = {
                        "cmap": "RdBu",
                        "norm": BoundaryNorm(
                            np.linspace(-0.2, 0.2, 9), ncolors=256, extend="both"
                        ),
                    }
            else:
                general = pd.DataFrame(
                    {
                        "value": corr_mean.loc["urban", season]
                        - corr_mean.loc["rural", season],
                        "sig": sig_diff.loc[:, season] <= 0.05,
                    }
                )
                mag = {
                    "cmap": "RdBu",
                    "norm": BoundaryNorm(
                        np.linspace(-0.12, 0.12, 7), ncolors=256, extend="both"
                    ),
                }

            # the differnce needs to be combined with rural sign for interpretation
            # For SPI (VPD should judge by the opposite)
            #   if rural < 0, urban > rural means less sensitive to flooding, and urban < rural means more sensitive to flooding
            #   if rural > 0, urban > rural means more water limited, and urban < rural means less water limited
            map = MapOfColors("point", general)
            cf = map.plot(ax=ax, inset_bar=True, map_args=mag, bar_args=bag)

            if i == 0:
                ax.text(
                    -0.1,
                    0.5,
                    season,
                    rotation=90,
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
            if j == 0:
                if i < 2:
                    ax.set_title(location.capitalize())
                else:
                    ax.set_title("Urban $-$ Rural")
            ax.text(
                0.02, 0.93, lab[j * 3 + i], fontweight="bold", transform=ax.transAxes
            )

            if j == 3:
                cax = fig.add_axes([0.1 + 0.27 * i, 0.08, 0.26, 0.01])
                plt.colorbar(cf, cax=cax, orientation="horizontal")

            if i == 2:
                res = wilcoxon(general["value"])
                if res.pvalue <= 0.05:
                    fontweight = "bold"
                else:
                    fontweight = "normal"
                ax.text(
                    0.55,
                    0.85,
                    f"p = {res.pvalue:.2f}",
                    fontweight=fontweight,
                    transform=ax.transAxes,
                )

    fig.savefig(
        os.path.join(
            path_out, "veg", f"{s.prefix}_{s.extent}_{s.name}_sensitivity_{varname}.png"
        ),
        dpi=600.0,
        bbox_inches="tight",
    )
    plt.close(fig)

h.close()


######################################################################################
# CITY LEVEL
######################################################################################
# ------------------------------------------------------------------------------------
# Size of the city (done at PIXEL LEVEL)
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Average impervious area fraction inside the city
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Percentage land cover type of the rural area <exclude the pixels that have > 50% crop lands>
# ------------------------------------------------------------------------------------
for use in ["daymet", "topowx", "yyz"]:
    setup = Setup(use)
    with pd.HDFStore(
        os.path.join(
            path_out, "luc", f"percity_spatial_avg_{setup.format_prefix_noveg()}.h5"
        ),
        mode="w",
    ) as h:
        with pd.HDFStore(
            os.path.join(
                path_out, "extreme_events", f"percity_{setup.format_prefix_noveg()}.h5"
            ),
            mode="r",
        ) as hx:
            for ex in ["heat_wave", "hot_and_dry"]:
                extreme_events = hx.select(ex)
                for fid in tqdm(range(85)):
                    u = LUC(fid)
                    data = u.get_spatial_avg(extreme_events.loc[[fid], :])
                    h.append(ex, data)

    # Plot the per pixel impervious area and land cover percenages and dominant land covers of each city
    rcParams["font.size"] = 6
    rcParams["axes.titlesize"] = 6

    # Plot the percentage land covers of the urban and the rural areas.
    lab = "abcdefghijkopqrstuvwxyz"
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(6.5, 7),
        subplot_kw={
            "projection": ccrs.AlbersEqualArea(
                central_longitude=-100, central_latitude=35
            )
        },
    )
    fig.subplots_adjust(hspace=0.1)
    for i, nn in enumerate(["urban", "rural"]):
        ax = axes.flat[i]
        with pd.HDFStore(
            os.path.join(
                path_out, "luc", f"percity_spatial_avg_{setup.format_prefix_noveg()}.h5"
            ),
            mode="r",
        ) as h:
            df = (
                h.select("heat_wave")
                .loc[(slice(None), slice(None), slice(None), nn), :]
                .reset_index()
                .groupby("fid")
                .mean()
                .drop(["impervious_frac", "impervious_size"], axis=1)
            )
            map = MapOfColors("pie", df)
            if i == 1:
                legend = True
            else:
                legend = False
            map.plot(
                ax=ax,
                inset_bar=False,
                legend=legend,
                legend_args={"ncol": 1, "loc": "right", "bbox_to_anchor": (1.5, 0.5)},
            )
            ax.set_title(nn.capitalize())
        ax.text(0.02, 1.05, lab[i], fontweight="bold", transform=ax.transAxes)

    ax = axes.flat[2]
    with pd.HDFStore(
        os.path.join(
            path_out, "luc", f"percity_spatial_avg_{setup.format_prefix_noveg()}.h5"
        ),
        mode="r",
    ) as h:
        df = (
            h.select("heat_wave")
            .loc[(slice(None), slice(None), slice(None), "rural"), :]
            .reset_index()
            .groupby("fid")
            .mean()
            .drop(["impervious_frac", "impervious_size"], axis=1)
        )
        df2 = pd.DataFrame(
            np.where(df > 0.2, df, np.nan), index=df.index, columns=df.columns
        )
        for fid in range(85):
            if np.all(np.isnan(df2.loc[fid, :].values)):
                df2.loc[fid, df.loc[fid, :].idxmax()] = df.loc[
                    fid, df.loc[fid, :].idxmax()
                ]
        df2["Developed"] = 0.0  # ignore this type
        df = df2.fillna(0.0)

        # Skip the following cities
        df.loc[[0, 1, 2, 3, 26, 28, 29, 32, 34], :] = 0.0
        df.loc[
            [
                51,
                58,
                6,
                5,
                8,
                19,
                27,
                30,
                43,
                48,
                49,
                45,
                46,
                38,
                33,
                68,
                74,
                77,
                78,
                79,
                80,
                81,
                82,
                83,
                31,
                20,
            ],
            "Grass",
        ] = 0.0
        df.loc[84, "Crop"] = 0.0
        df.loc[[23, 25, 35, 54, 47], "Deciduous forest"] = 0.0
        df.loc[[12, 40], "Wetland"] = 0.0
        df.loc[:, "Mixed forest"] = 0.0

    map = MapOfColors("pie", df)
    map.plot(ax=ax, annotate=False, inset_bar=False, legend=False)
    ax.set_title("Selected cities and land cover for pixel-level regression")
    ax.text(0.02, 1.05, lab[i + 1], fontweight="bold", transform=ax.transAxes)

    fig.savefig(
        os.path.join(
            path_out,
            "luc",
            "plots",
            f"percity_spatial_avg_luc_{setup.format_prefix_noveg()}.png",
        ),
        dpi=600.0,
        bbox_inches="tight",
    )
    plt.close(fig)

# ------------------------------------------------------------------------------------
# Urban-rural difference in elevation
# ------------------------------------------------------------------------------------
data = {}
for fid in range(85):
    b = Elevation(fid)
    data[fid] = b.get_spatial_avg()
data = pd.DataFrame(data).T
data.index.name = "fid"
data.to_csv(
    os.path.join(path_out, "elev", f"urban_rural_difference_{setup.extent}.csv")
)

# Plot the urban-rural elevation difference
data = pd.read_csv(
    os.path.join(path_out, "elev", f"urban_rural_difference_{setup.extent}.csv"),
    index_col=0,
)
map = MapOfColors("point", data.loc[:, "diff"])
fig, ax, cf = map.plot(
    inset_bar=False, map_args=dict(vmin=-220, vmax=5, cmap="viridis")
)
ax.set_title("Elevation difference (meters)")
plt.colorbar(cf, ax=ax, orientation="horizontal", shrink=0.7, aspect=50, pad=0.05)
fig.savefig(
    os.path.join(
        path_out, "elev", "plots", f"urban_rural_difference_{setup.extent}.png"
    ),
    dpi=600.0,
    bbox_inches="tight",
)
plt.close(fig)

# ------------------------------------------------------------------------------------
# Seasonal background climatology of precipitation & temperature (1981-2010, daymet) (done at PIXEL LEVEL)
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Intensity & duration of the heat wave (done at PIXEL LEVEL)
# ------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------
# Day & nighttime UHI (average urban-rural difference in tmax & tmin)
# Average SPI & VPD during/after the heat wave (in_event, post_event, of_season)
# ------------------------------------------------------------------------------------
for varname, use in it.product(
    ["tmax", "tmin", "vpd", "spi"], ["daymet", "topowx", "yyz"]
):
    if (varname == "tmax") | ((varname == "tmin") & (use == "daymet")):
        continue

    setup = Setup(use)
    with pd.HDFStore(
        os.path.join(
            path_out,
            "clim",
            f"percity_spatial_avg_{varname}_{setup.format_prefix_noveg()}.h5",
        ),
        mode="w",
    ) as h:
        with pd.HDFStore(
            os.path.join(
                path_out, "extreme_events", f"percity_{setup.format_prefix_noveg()}.h5"
            ),
            mode="r",
        ) as hx:
            for ex in ["heat_wave", "hot_and_dry"]:
                extreme_events = hx.select(ex)
                for fid in tqdm(range(85)):
                    m = Events(fid, use)
                    var = m.get_aux_spatial_avg(varname, extreme_events.loc[[fid], :])
                    h.append(ex, var)


for use in ["daymet", "topowx", "yyz"]:
    setup = Setup(use)
    # Plot the urban-rural day- and nighttime temperature difference during the heat waves and of the average of the season (SPI is the same)
    fig, axes = plt.subplots(
        4,
        4,
        figsize=(20, 20),
        sharex=True,
        sharey=True,
        subplot_kw={
            "projection": ccrs.AlbersEqualArea(
                central_longitude=-100, central_latitude=35
            )
        },
    )
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    for i, varname in enumerate(["tmax", "tmin", "spi", "vpd"]):
        if varname in ["tmin", "tmax"]:
            vmin = -1
            vmax = 1
        elif varname == "spi":
            vmin = -0.1
            vmax = 0.1
        elif varname == "vpd":
            vmin = -250.0
            vmax = 250
        else:
            raise "Not implemented"
        with pd.HDFStore(
            os.path.join(
                path_out,
                "clim",
                f"percity_spatial_avg_{varname}_{setup.format_prefix_noveg()}.h5",
            ),
            mode="r",
        ) as h:
            for j, ex in enumerate(["heat_wave", "hot_and_dry"]):
                for k, tt in enumerate(["in_event", "post_event"]):
                    ax = axes[i, j * 2 + k]

                    df = (
                        h.select(ex)
                        .reset_index()
                        .groupby(["fid", "location"])
                        .mean()[tt]
                        .unstack()
                    )
                    df = df["urban"] - df["rural"]

                    map = MapOfColors("point", df)
                    cf = map.plot(
                        inset_bar=True,
                        ax=ax,
                        map_args=dict(vmin=vmin, vmax=vmax, cmap=cmap_div()),
                    )

                    if i == 0:
                        ax.set_title(f"{ex} {tt}")
                    if j == 0:
                        ax.text(
                            -0.1, 0.5, f"{varname}", transform=ax.transAxes, rotation=90
                        )
                    plt.colorbar(cf, ax=ax, orientation="horizontal")
    fig.savefig(
        os.path.join(
            path_out,
            "clim",
            "plots",
            f"percity_spatial_avg_{setup.format_prefix()}.png",
        ),
        dpi=600.0,
        bbox_inches="tight",
    )
    plt.close(fig)


######################################################################################
# The resilience metrics themselves
# Save the vegetation index prior to & during & after the heat waves to HDF file
######################################################################################
# ------------------------------------------------------------------------------------
# per pixel values
# ~30 min
# ------------------------------------------------------------------------------------
with pd.HDFStore(
    os.path.join(
        path_out, "extreme_events", f"percity_{setup.format_prefix_noveg()}.h5"
    ),
    mode="r",
) as hx:
    with pd.HDFStore(
        os.path.join(
            path_out, "veg_response", f"percity_per_pixel_{setup.format_prefix()}.h5"
        ),
        mode="w",
    ) as h:
        for ex in ["heat_wave", "hot_and_dry"]:
            extreme_events = hx.select(ex)
            for fid in tqdm(range(85)):
                v = Veg(fid)
                veg_response = v.get_pixel(extreme_events.loc[[fid], :])
                h.append(ex, veg_response)


# Plot the resistance and resilience as sanity check
with pd.HDFStore(
    os.path.join(
        path_out, "veg_response", f"percity_per_pixel_{setup.format_prefix()}.h5"
    ),
    mode="r",
) as h:
    df = h.select("heat_wave")[["Recovery", "Resistance"]]
    df["season"] = pd.DatetimeIndex(df.index.get_level_values(2)).month.map(
        month_to_season
    )
    df = df.reset_index()

    for fid in tqdm(range(85)):
        fig, axes = plt.subplots(
            4, 2, figsize=(10, 20), subplot_kw={"projection": crs_daymet}
        )
        for i, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
            for j, metric in enumerate(["Resistance", "Recovery"]):
                ax = axes[i, j]

                if j == 0:
                    norm = mpl.colors.Normalize(vmin=-0.25, vmax=0.25)
                else:
                    norm = mpl.colors.Normalize(vmin=-1.2, vmax=1.2)

                temp = (
                    df.loc[
                        (df["fid"] == fid) & (df["season"] == season),
                        ["start", "end", "row", "col"] + [metric],
                    ]
                    .groupby(["row", "col"])
                    .mean()
                )
                temp = add_grid_coords(fid, temp, setup.extent)

                cf = ax.scatter(
                    temp["x"].values,
                    temp["y"].values,
                    c=temp[metric].values,
                    s=10,
                    lw=0.0,
                    marker="s",
                    transform=crs_daymet,
                    cmap="RdYlBu",
                    norm=norm,
                )
                add_core_boundary(ax, fid, "k")
                if j == 0:
                    ax.text(-0.2, 0.5, season, rotation=90, transform=ax.transAxes)
                if i == 0:
                    ax.set_title(metric)
                elif i == 3:
                    cax = fig.add_axes([0.05 + 0.5 * j, 0.05, 0.4, 0.01])
                    cb = plt.colorbar(
                        cf, cax=cax, shrink=0.5, label=metric, orientation="horizontal"
                    )

        fig.savefig(
            os.path.join(
                path_out,
                "veg_response",
                "plots",
                f"percity_per_pixel_{setup.format_prefix()}_metrics_{fid}.png",
            ),
            dpi=600.0,
            bbox_inches="tight",
        )
        plt.close(fig)


# ------------------------------------------------------------------------------------
# City average optimal temperature, and seasonal gap to optimal temperatures
# ------------------------------------------------------------------------------------
s = SeasonalAvgTemperaturePixel()
s.calc_mean()
seasonal_average = s.seasonal_average

s = OptimalTPixel()
s.calc_mean()
optimum = pd.read_csv(
    os.path.join(path_out, "veg", f"{s.prefix}_{s.extent}_{s.name}_optimalT.csv"),
    index_col=0,
    header=[0, 1],
)

gap_to_optimalT = pd.DataFrame(
    np.nan, index=seasonal_average.index, columns=seasonal_average.columns
)
for varname, location, season, use in it.product(
    ["tmax", "tmin"],
    ["urban", "rural"],
    ["DJF", "MAM", "JJA", "SON"],
    ["daymet", "topowx", "yyz"],
):
    gap_to_optimalT.loc[:, (location, varname, season, use)] = (
        seasonal_average.loc[:, (location, varname, season, use)]
        - optimum.loc[:, (varname, location)].values
    )
gap_to_optimalT.to_csv(
    os.path.join(path_out, "veg", f"{s.prefix}_{s.extent}_{s.name}_gap_to_optimalT.csv")
)

lab = "abcdefghijklmnopqrstuvwxyz"
for varname in ["tmax", "tmin"]:
    fig, axes = plt.subplots(
        4,
        3,
        figsize=(15, 13),
        subplot_kw={
            "projection": ccrs.AlbersEqualArea(
                central_longitude=-100, central_latitude=35
            )
        },
    )
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    for i, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
        for j, location in enumerate(["urban", "rural"]):
            ax = axes[i, j]
            # (plot averages over the three meteorological data sets)
            m = MapOfColors(
                "point",
                seasonal_average.loc[:, (location, varname, season)].mean(axis=1)
                - optimum.loc[:, (varname, location)],
            )
            cf = m.plot(
                ax=ax,
                inset_bar=False,
                map_args={
                    "norm": BoundaryNorm(np.arange(-20, 20.1, 2.5), 256, extend="both"),
                    "cmap": "RdBu",
                },
            )
            if i == 0:
                ax.set_title("$\Delta$" + location.capitalize())
            ax.text(0.02, 0.9, lab[i * 3 + j], transform=ax.transAxes, weight="bold")

            if j == 0:
                ax.text(
                    -0.1,
                    0.5,
                    season,
                    verticalalignment="center",
                    rotation=90,
                    transform=ax.transAxes,
                )

            if i == 3:
                cax = fig.add_axes([0.1 + 0.27 * j, 0.08, 0.26, 0.01])
                plt.colorbar(cf, cax=cax, orientation="horizontal")

        ax = axes[i, 2]
        result = (
            seasonal_average.loc[:, ("urban", varname, season)].mean(axis=1)
            - optimum.loc[:, (varname, "urban")]
        ) - (
            seasonal_average.loc[:, ("rural", varname, season)].mean(axis=1)
            - optimum.loc[:, (varname, "rural")]
        )
        m = MapOfColors("point", result)
        cf = m.plot(
            ax=ax,
            inset_bar=True,
            map_args={
                "norm": BoundaryNorm(np.arange(-2, 2.01, 0.5), 256, extend="both"),
                "cmap": "RdBu",
            },
            bar_args={"color": ["#313695", "#4575b4", "#a50026", "#d73027"]},
        )
        if i == 0:
            ax.set_title("$\Delta$(Urban \u2212 Rural)")
        ax.text(0.02, 0.9, lab[i * 3 + 2], transform=ax.transAxes, weight="bold")
        res = wilcoxon(result)
        if res.pvalue <= 0.05:
            fontweight = "bold"
        else:
            fontweight = "normal"
        ax.text(
            0.55,
            0.85,
            f"p = {res.pvalue:.2f}",
            fontweight=fontweight,
            transform=ax.transAxes,
        )

        if i == 3:
            cax = fig.add_axes([0.1 + 0.27 * 2, 0.08, 0.26, 0.01])
            plt.colorbar(cf, cax=cax, orientation="horizontal")

        if season == "JJA":
            # Calculate the fraction of cities where the UHI size was smaller than the urban increase in optimal temperature
            diff = (
                seasonal_average.loc[:, ("urban", varname, season)].mean(axis=1)
                - optimum.loc[:, (varname, "urban")]
            ) - (
                seasonal_average.loc[:, ("rural", varname, season)].mean(axis=1)
                - optimum.loc[:, (varname, "rural")]
            )

            diff_west = diff.loc[modis_luc_city_groups["West"]]
            diff_east = diff.loc[~diff.index.isin(modis_luc_city_groups["West"])]

            print(varname, "West count", sum(diff_west < 0), len(diff_west))
            print(varname, "East count", sum(diff_east < 0), len(diff_east))

    fig.savefig(
        os.path.join(
            path_out,
            "veg",
            f"{s.prefix}_{s.extent}_{s.name}_{varname}_gap_to_optimalT.png",
        ),
        dpi=600.0,
        bbox_inches="tight",
    )
    plt.close(fig)


lab = "abcdefghijklmnopqrstuvwxyz"
fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 8),
    subplot_kw={
        "projection": ccrs.AlbersEqualArea(central_longitude=-100, central_latitude=35)
    },
)
fig.subplots_adjust(hspace=0.01, wspace=0.01)
for i, varname in enumerate(["tmax", "tmin"]):
    for j, location in enumerate(["urban", "rural"]):
        ax = axes[i, j]
        m = MapOfColors("point", optimum.loc[:, (varname, location)])
        cf = m.plot(
            ax=ax,
            inset_bar=False,
            map_args={
                "norm": BoundaryNorm(np.arange(7.5, 32.6, 2.5), 256, extend="both"),
                "cmap": "RdYlBu",
            },
        )
        if i == 0:
            ax.set_title(location.capitalize())
        ax.text(0.02, 0.9, lab[i * 3 + j], transform=ax.transAxes, weight="bold")

        if j == 0:
            ax.text(
                -0.1,
                0.5,
                f"{varname}" + "$_{optimum}$",
                verticalalignment="center",
                rotation=90,
                transform=ax.transAxes,
            )

    if i == 1:
        cax = fig.add_axes([0.1, 0.08, 0.5, 0.01])
        plt.colorbar(cf, cax=cax, orientation="horizontal")

    result = optimum.loc[:, varname]["urban"] - optimum.loc[:, varname]["rural"].values
    ax = axes[i, 2]
    m = MapOfColors("point", result)
    cf = m.plot(
        ax=ax,
        inset_bar=True,
        map_args={
            "norm": BoundaryNorm(np.arange(-2, 2.1, 0.1), 256, extend="both"),
            "cmap": "RdBu",
        },
        bar_args={"color": ["#313695", "#4575b4", "#a50026", "#d73027"]},
    )
    if i == 0:
        ax.set_title("Urban \u2212 Rural")
    ax.text(0.02, 0.9, lab[i * 3 + 2], transform=ax.transAxes, weight="bold")
    res = wilcoxon(result)
    if res.pvalue <= 0.05:
        fontweight = "bold"
    else:
        fontweight = "normal"
    ax.text(
        0.55,
        0.85,
        f"p = {res.pvalue:.2f}",
        fontweight=fontweight,
        transform=ax.transAxes,
    )

    if i == 1:
        cax = fig.add_axes([0.62, 0.08, 0.3, 0.01])
        plt.colorbar(cf, cax=cax, orientation="horizontal")

fig.savefig(
    os.path.join(path_out, "veg", f"{s.prefix}_{s.extent}_{s.name}_optimalT.png"),
    dpi=600.0,
    bbox_inches="tight",
)
plt.close(fig)


# ------------------------------------------------------------------------------------
# City average sensitivity to SPI & VPD in each season
# ------------------------------------------------------------------------------------
s = SensitivityWaterPixel()
s.calc_mean()
seasonal_average = s.seasonal_average
seasonal_average.to_csv(
    os.path.join(path_out, "veg", f"{s.prefix}_{s.extent}_{s.name}_sensWater.csv")
)

lab = "abcdefghijklmnopqrstuvwxyz"
for varname in ["spi", "vpd"]:
    fig, axes = plt.subplots(
        4,
        3,
        figsize=(15, 13),
        subplot_kw={
            "projection": ccrs.AlbersEqualArea(
                central_longitude=-100, central_latitude=35
            )
        },
    )
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    for i, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
        for j, location in enumerate(["urban", "rural"]):
            ax = axes[i, j]
            # (plot averages over the three meteorological data sets)
            m = MapOfColors(
                "point",
                pd.Series(
                    seasonal_average.loc[
                        (season, slice(None), location), f"corr_{varname}"
                    ].values,
                    index=range(85),
                ),
            )
            cf = m.plot(
                ax=ax,
                inset_bar=False,
                map_args={
                    "norm": BoundaryNorm(
                        np.arange(-0.6, 0.6, 0.05), 256, extend="both"
                    ),
                    "cmap": "RdBu",
                },
            )
            if i == 0:
                ax.set_title("$\Delta$" + location.capitalize())
            ax.text(0.02, 0.9, lab[i * 3 + j], transform=ax.transAxes, weight="bold")

            if j == 0:
                ax.text(
                    -0.1,
                    0.5,
                    season,
                    verticalalignment="center",
                    rotation=90,
                    transform=ax.transAxes,
                )

            if i == 3:
                cax = fig.add_axes([0.1 + 0.27 * j, 0.08, 0.26, 0.01])
                plt.colorbar(cf, cax=cax, orientation="horizontal")

        ax = axes[i, 2]
        result = pd.Series(
            seasonal_average.loc[
                (season, slice(None), "urban"), f"corr_{varname}"
            ].values
            - seasonal_average.loc[
                (season, slice(None), "rural"), f"corr_{varname}"
            ].values,
            index=range(85),
        )
        m = MapOfColors("point", result)
        cf = m.plot(
            ax=ax,
            inset_bar=True,
            map_args={
                "norm": BoundaryNorm(np.arange(-0.6, 0.6, 0.05), 256, extend="both"),
                "cmap": "RdBu",
            },
            bar_args={"color": ["#313695", "#4575b4", "#a50026", "#d73027"]},
        )
        if i == 0:
            ax.set_title("$\Delta$(Urban \u2212 Rural)")
        ax.text(0.02, 0.9, lab[i * 3 + 2], transform=ax.transAxes, weight="bold")
        res = wilcoxon(result)
        if res.pvalue <= 0.05:
            fontweight = "bold"
        else:
            fontweight = "normal"
        ax.text(
            0.55,
            0.85,
            f"p = {res.pvalue:.2f}",
            fontweight=fontweight,
            transform=ax.transAxes,
        )

        if i == 3:
            cax = fig.add_axes([0.1 + 0.27 * 2, 0.08, 0.26, 0.01])
            plt.colorbar(cf, cax=cax, orientation="horizontal")

    fig.savefig(
        os.path.join(
            path_out,
            "veg",
            f"{s.prefix}_{s.extent}_{s.name}_{varname}_sensitivity_water.png",
        ),
        dpi=600.0,
        bbox_inches="tight",
    )
    plt.close(fig)
