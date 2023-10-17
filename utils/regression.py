import pandas as pd
import numpy as np
import os
from .paths import *
from .constants import *
from .analysis import *
from tqdm import tqdm
from scipy.stats import mannwhitneyu, spearmanr, pearsonr, wilcoxon, linregress
from statsmodels.stats.proportion import proportions_ztest
import itertools as it


class Plotter:
    def __init__(self, extreme):
        self.extreme = extreme
        self.prefix = "percity_per_pixel"
        self.extent = "tiff_3x"
        self.name = "MOD09Q1G_EVI"
        self.heat_wave_thres = 90
        self.hot_and_dry_thres = 85  # these consistent with Setup()

    def _get_summary(self, no_crop=False):
        if no_crop:
            suffix = "_nocrop"
        else:
            suffix = ""

        h = pd.HDFStore(
            os.path.join(
                path_out,
                "measures",
                "regression_per_pixel",
                "summary",
                f"{self.prefix}_{self.extent}_{self.name}_{self.heat_wave_thres}_{self.hot_and_dry_thres}_{self.extreme}_city_average_events{suffix}.h5",
            ),
            mode="r",
        )
        data = h.select("data").copy()

        summary_varname_diff_sign = pd.DataFrame(
            "         ", index=data.loc["urban", :].index, columns=data.columns
        )
        for stat in ["Resistance", "Recovery"]:
            temp = pd.concat(
                [
                    (data.loc["urban", stat] > 0).map({True: "P", False: "N"}),
                    (data.loc["rural", stat] > 0).map({True: "P", False: "N"}),
                ],
                axis=1,
            ).apply(lambda x: x.values[0] + x.values[1], axis=1)
            summary_varname_diff_sign.loc[:, stat] = temp
        summary_varname_diff_mag = (
            data.loc["urban", :].abs() - data.loc["rural", :].abs()
        )  # difference in magnitude
        summary_varname_diff = data.loc["urban", :] - data.loc["rural", :]
        summary_varname_urban = data.loc["urban", :]
        summary_varname_rural = data.loc["rural", :]
        h.close()
        return (
            summary_varname_diff_sign,
            summary_varname_diff_mag,
            summary_varname_diff,
            summary_varname_urban,
            summary_varname_rural,
        )

    def _get_predictors(self, use_gap_to_optim):
        """Collect all the city level predictors
        Sensitivity to SPI and VPD might be an important factor that controls the sign
        """
        ###################################################################
        # Read in the input data
        ###################################################################
        size = pd.read_csv(os.path.join(path_out, "uhi", "urban_size.csv"), index_col=0)
        size = np.log(
            size.iloc[:, 0]
        )  # log-transform to get closer to normal distribution

        clim = pd.read_csv(
            os.path.join(path_out, "clim", f"clim_{self.extent}_daymet.csv"),
            index_col=[0, 1],
            header=0,
        ).rename({"prcp": "background_prcp", "tmean": "background_tmean"}, axis=1)

        tmax = {}  # urban heat island sizes
        tmin = {}
        spi = {}  # moisture conditions
        vpd = {}
        for use in ["daymet", "topowx", "yyz"]:
            with pd.HDFStore(
                os.path.join(
                    path_out,
                    "clim",
                    f"percity_spatial_avg_tmax_{self.extent}_{use}_90_85.h5",
                ),
                mode="r",
            ) as h:
                tmax[use] = h.select(self.extreme).rename(
                    {
                        "in_event": "dtmax_in_event",
                        "post_event": "dtmax_post_event",
                        "of_season": "dtmax_of_season",
                    },
                    axis=1,
                )
                tmax[use] = tmax[use].unstack()
                tmax[use].columns = tmax[use].columns.reorder_levels([1, 0])
                tmax[use] = (
                    tmax[use].loc[:, "urban"] - tmax[use].loc[:, "rural"]
                ).stack()

            with pd.HDFStore(
                os.path.join(
                    path_out,
                    "clim",
                    f"percity_spatial_avg_tmin_{self.extent}_{use}_90_85.h5",
                ),
                mode="r",
            ) as h:
                tmin[use] = h.select(self.extreme).rename(
                    {
                        "in_event": "dtmin_in_event",
                        "post_event": "dtmin_post_event",
                        "of_season": "dtmin_of_season",
                    },
                    axis=1,
                )
                tmin[use] = tmin[use].unstack()
                tmin[use].columns = tmin[use].columns.reorder_levels([1, 0])
                tmin[use] = (
                    tmin[use].loc[:, "urban"] - tmin[use].loc[:, "rural"]
                ).stack()

            with pd.HDFStore(
                os.path.join(
                    path_out,
                    "clim",
                    f"percity_spatial_avg_spi_{self.extent}_{use}_90_85.h5",
                ),
                mode="r",
            ) as h:
                spi[use] = h.select(self.extreme).rename(
                    {
                        "in_event": "spi_in_event",
                        "post_event": "spi_post_event",
                        "of_season": "spi_of_season",
                    },
                    axis=1,
                )
                spi[use] = spi[use].unstack()
                spi[use].columns = spi[use].columns.reorder_levels([1, 0])
                spi[use] = spi[use].loc[:, "total"].stack()

            with pd.HDFStore(
                os.path.join(
                    path_out,
                    "clim",
                    f"percity_spatial_avg_vpd_{self.extent}_{use}_90_85.h5",
                ),
                mode="r",
            ) as h:
                vpd[use] = h.select(self.extreme).rename(
                    {
                        "in_event": "vpd_in_event",
                        "post_event": "vpd_post_event",
                        "of_season": "vpd_of_season",
                    },
                    axis=1,
                )
                vpd[use] = vpd[use].unstack()
                vpd[use].columns = vpd[use].columns.reorder_levels([1, 0])
                vpd[use] = vpd[use].loc[:, "total"].stack()
        tmax = pd.DataFrame(tmax).stack()
        tmax.index = tmax.index.reorder_levels([0, 1, 2, 4, 3])
        tmax = tmax.unstack()
        tmax.index.names = ["fid", "start", "end", "use"]
        tmin = pd.DataFrame(tmin).stack()
        tmin.index = tmin.index.reorder_levels([0, 1, 2, 4, 3])
        tmin = tmin.unstack()
        tmin.index.names = ["fid", "start", "end", "use"]
        spi = pd.DataFrame(spi).stack()
        spi.index = spi.index.reorder_levels([0, 1, 2, 4, 3])
        spi = spi.unstack()
        spi.index.names = ["fid", "start", "end", "use"]
        vpd = pd.DataFrame(vpd).stack()
        vpd.index = vpd.index.reorder_levels([0, 1, 2, 4, 3])
        vpd = vpd.unstack()
        vpd.index.names = ["fid", "start", "end", "use"]

        luc = {}
        impervious_frac = {}
        for use in ["daymet", "topowx", "yyz"]:
            with pd.HDFStore(
                os.path.join(
                    path_out, "luc", f"percity_spatial_avg_{self.extent}_{use}_90_85.h5"
                ),
                mode="r",
            ) as h:
                luc[use] = (
                    h.select(self.extreme)
                    .loc[(slice(None), slice(None), slice(None), "rural"), :]
                    .drop(["impervious_size", "impervious_frac"], axis=1)
                    .stack()
                )
                # luc[use] = h.select(extreme).loc[(slice(None), slice(None), slice(None), 'rural'), 'Shrub']
                luc[use].index = luc[use].index.droplevel(3)
                impervious_frac[use] = h.select(self.extreme).loc[
                    (slice(None), slice(None), slice(None), "urban"), "impervious_frac"
                ]
                impervious_frac[use].index = impervious_frac[use].index.droplevel(3)
        luc = pd.DataFrame(luc).stack()
        luc.index = luc.index.reorder_levels([0, 1, 2, 4, 3])
        luc = luc.unstack()
        luc.index.names = ["fid", "start", "end", "use"]
        impervious_frac = (
            pd.DataFrame(impervious_frac).stack().to_frame("impervious_frac")
        )
        impervious_frac.index.names = ["fid", "start", "end", "use"]

        intensity = {}
        duration = {}
        for use in ["daymet", "topowx", "yyz"]:
            with pd.HDFStore(
                os.path.join(
                    path_out, "extreme_events", f"percity_{self.extent}_{use}_90_85.h5"
                ),
                mode="r",
            ) as h:
                intensity[use] = h.select(self.extreme)["intensity"]
                duration[use] = h.select(self.extreme)["duration"]
        intensity = pd.DataFrame(intensity).stack().to_frame("event_intensity")
        intensity.index.names = ["fid", "start", "end", "use"]
        duration = pd.DataFrame(duration).stack().to_frame("event_duration")
        duration.index.names = ["fid", "start", "end", "use"]

        elev = pd.read_csv(
            os.path.join(path_out, "elev", f"urban_rural_difference_{self.extent}.csv"),
            index_col=0,
        ).rename({"diff": "elev_diff"}, axis=1)

        if use_gap_to_optim:
            gap_to_optimalT = pd.read_csv(
                os.path.join(
                    path_out,
                    "veg",
                    f"{self.prefix}_{self.extent}_MOD09Q1G_EVI_gap_to_optimalT.csv",
                ),
                index_col=0,
                header=[0, 1, 2, 3],
            )
        else:
            optimalT = pd.read_csv(
                os.path.join(
                    path_out,
                    "veg",
                    f"{self.prefix}_{self.extent}_MOD09Q1G_EVI_optimalT.csv",
                ),
                index_col=0,
                header=[0, 1],
            )

        sensWater = pd.read_csv(
            os.path.join(
                path_out,
                "veg",
                f"percity_per_pixel_{self.extent}_MOD09Q1G_EVI_sensWater.csv",
            ),
            index_col=[0, 1, 2],
            header=0,
        )

        ###################################################################
        # Collect into the same dataframe
        ###################################################################
        if use_gap_to_optim:
            collect = pd.DataFrame(
                np.nan,
                index=tmax.index,
                columns=[
                    "city_size_log",
                    "background_prcp",
                    "background_tmean",
                    "dtmax_in_event",
                    "dtmax_post_event",
                    "dtmin_in_event",
                    "dtmin_post_event",
                    "spi_in_event",
                    "spi_post_event",
                    "vpd_in_event",
                    "vpd_post_event",
                ]
                + list(luc.columns)
                + [
                    "impervious_frac",
                    "event_intensity",
                    "event_duration",
                    "elev_diff",
                    "gap_to_optimal_tmax_diff",
                    "gap_to_optimal_tmin_diff",
                    "corr_spi_diff",
                    "corr_vpd_diff",
                ],
            )
        else:
            collect = pd.DataFrame(
                np.nan,
                index=tmax.index,
                columns=[
                    "city_size_log",
                    "background_prcp",
                    "background_tmean",
                    "dtmax_in_event",
                    "dtmax_post_event",
                    "dtmin_in_event",
                    "dtmin_post_event",
                    "spi_in_event",
                    "spi_post_event",
                    "vpd_in_event",
                    "vpd_post_event",
                ]
                + list(luc.columns)
                + [
                    "impervious_frac",
                    "event_intensity",
                    "event_duration",
                    "elev_diff",
                    "optimal_tmax_diff",
                    "optimal_tmin_diff",
                    "corr_spi_diff",
                    "corr_vpd_diff",
                ],
            )

        for fid in range(85):
            collect.loc[fid, "city_size_log"] = size.loc[fid]

            for varname, season in it.product(
                ["background_prcp", "background_tmean"], ["DJF", "MAM", "JJA", "SON"]
            ):
                collect.loc[
                    (collect.index.get_level_values("fid") == fid)
                    & (
                        collect.index.get_level_values("end").month.map(month_to_season)
                        == season
                    ),
                    varname,
                ] = clim.loc[(season, fid), varname]

        for prefix, var in zip(
            ["dtmax", "dtmin", "spi", "vpd"], [tmax, tmin, spi, vpd]
        ):
            ind = collect.index.intersection(var.index)
            for suffix in ["in_event", "post_event"]:
                collect.loc[ind, f"{prefix}_{suffix}"] = var.loc[
                    ind, f"{prefix}_{suffix}"
                ]

        ind = collect.index.intersection(luc.index)
        for varname in luc.columns:
            collect.loc[ind, varname] = luc.loc[ind, varname]

        for varname, var in zip(
            ["impervious_frac", "event_intensity", "event_duration"],
            [impervious_frac, intensity, duration],
        ):
            ind = collect.index.intersection(var.index)
            collect.loc[ind, varname] = var.loc[ind, varname]

        for fid in range(85):
            collect.loc[
                collect.index.get_level_values("fid") == fid, "elev_diff"
            ] = elev.loc[fid, "elev_diff"]

        if use_gap_to_optim:
            for varname, fid, season, use in it.product(
                ["tmax", "tmin"],
                range(85),
                ["DJF", "MAM", "JJA", "SON"],
                ["daymet", "topowx", "yyz"],
            ):
                collect.loc[
                    (collect.index.get_level_values("fid") == fid)
                    & (
                        collect.index.get_level_values("end").month.map(month_to_season)
                        == season
                    )
                    & (collect.index.get_level_values("use") == use),
                    f"gap_to_optimal_{varname}_diff",
                ] = (
                    gap_to_optimalT.loc[fid, ("urban", varname, season, use)]
                    - gap_to_optimalT.loc[fid, ("rural", varname, season, use)]
                )
        else:
            for varname, fid in it.product(["tmax", "tmin"], range(85)):
                collect.loc[
                    (collect.index.get_level_values("fid") == fid),
                    f"optimal_{varname}_diff",
                ] = (
                    optimalT.loc[fid, (varname, "urban")]
                    - optimalT.loc[fid, (varname, "rural")]
                )

        for season, varname in it.product(
            ["DJF", "MAM", "JJA", "SON"], ["corr_spi", "corr_vpd"]
        ):
            for fid in range(85):
                collect.loc[
                    (collect.index.get_level_values("fid") == fid)
                    & (
                        collect.index.get_level_values("end").month.map(month_to_season)
                        == season
                    ),
                    f"{varname}_diff",
                ] = (
                    sensWater.loc[(season, fid, "urban"), varname]
                    - sensWater.loc[(season, fid, "rural"), varname]
                )

        ###################################################################
        # Return the data
        ###################################################################
        luc_clist = {
            "Evergreen forest": "#33a02c",
            "Deciduous forest": "#ff7f00",
            "Mixed forest": "#6a3d9a",
            "Shrub": "#b15928",
            "Grass": "#e31a1c",
            "Crop": "#e7298a",
            "Wetland": "#1f78b4",
        }

        # modis_fid_to_region = dict([(k,i) for i,j in modis_luc_city_groups.items() for k in j])
        # collect['region'] = pd.Series(collect.index.get_level_values('fid')).map(modis_fid_to_region).values
        # region_names = ['Northeast', 'Southeast_top', 'Southeast_hot', 'West']

        return collect, luc_clist  # region_names

    def get_city_summary(self, no_crop=False):
        if no_crop:
            suffix = "_nocrop"
        else:
            suffix = ""

        result = pd.DataFrame(
            np.nan,
            index=pd.MultiIndex.from_product(
                [["DJF", "MAM", "JJA", "SON"], ["daymet", "topowx", "yyz"], range(85)],
                names=["season", "use", "fid"],
            ),
            columns=pd.MultiIndex.from_product(
                [
                    ["Resistance", "Recovery"],
                    [
                        "median_urban",
                        "median_urban_pval",
                        "median_rural",
                        "median_rural_pval",
                        "median_abs_urban",
                        "median_abs_rural",
                        "median_abs_pval",
                        "frac_pos_urban",
                        "frac_pos_rural",
                        "frac_pos_pval",
                    ],
                ]
            ),
        )

        for season, use in tqdm(
            it.product(["DJF", "MAM", "JJA", "SON"], ["daymet", "topowx", "yyz"])
        ):
            with pd.HDFStore(
                os.path.join(
                    path_out,
                    "veg_response",
                    f"percity_per_pixel_{self.extent}_{use}_{self.name}_{self.heat_wave_thres}_{self.hot_and_dry_thres}.h5",
                ),
                mode="r",
            ) as h:
                data = h.select(self.extreme).loc[:, ["Resistance", "Recovery"]]
                data = data.loc[
                    data.index.get_level_values("end").month.map(month_to_season)
                    == season,
                    :,
                ]
                if no_crop:
                    h2 = pd.HDFStore(
                        os.path.join(
                            path_out,
                            "luc",
                            f"percity_per_pixel_{self.extent}_{use}_{self.heat_wave_thres}_{self.hot_and_dry_thres}.h5",
                        ),
                        mode="r",
                    )
                    # Get crop fraction
                    crop = h2.select("heat_wave")[7]
                    non_crop = crop.index[crop < 0.5]
                    del crop
                    h2.close()
                    data = data.loc[data.index.intersection(non_crop), :].reset_index()
                    data = add_urlabel_all(data, self.extent)
                else:
                    data = data.reset_index()
                data = data.set_index(["location", "fid", "start", "end", "row", "col"])

                for stat, fid in it.product(["Resistance", "Recovery"], range(85)):
                    x = data.loc[("urban", fid), stat]
                    y = data.loc[("rural", fid), stat]

                    _, pabs = mannwhitneyu(np.abs(x), np.abs(y))
                    _, pabs_pos = mannwhitneyu(
                        np.abs(x.loc[x > 0]), np.abs(y.loc[y > 0])
                    )
                    _, pabs_neg = mannwhitneyu(
                        np.abs(x.loc[x < 0]), np.abs(y.loc[y < 0])
                    )
                    _, psign = proportions_ztest(
                        ((x > 0).sum(), (y > 0).sum()),
                        nobs=(len(x), len(y)),
                        value=0,
                        alternative="two-sided",
                    )

                    result.loc[(season, use, fid), (stat, "median_urban")] = np.median(
                        x
                    )
                    result.loc[
                        (season, use, fid), (stat, "median_urban_pval")
                    ] = wilcoxon(x).pvalue
                    result.loc[(season, use, fid), (stat, "median_rural")] = np.median(
                        y
                    )
                    result.loc[
                        (season, use, fid), (stat, "median_rural_pval")
                    ] = wilcoxon(y).pvalue

                    result.loc[
                        (season, use, fid), (stat, "median_abs_urban")
                    ] = np.median(np.abs(x))
                    result.loc[
                        (season, use, fid), (stat, "median_abs_rural")
                    ] = np.median(np.abs(y))
                    result.loc[(season, use, fid), (stat, "median_abs_pval")] = pabs

                    result.loc[
                        (season, use, fid), (stat, "median_abs_urban, pos")
                    ] = np.median(np.abs(x.loc[x > 0]))
                    result.loc[
                        (season, use, fid), (stat, "median_abs_rural, pos")
                    ] = np.median(np.abs(y.loc[y > 0]))
                    result.loc[
                        (season, use, fid), (stat, "median_abs_pval, pos")
                    ] = pabs_pos

                    result.loc[
                        (season, use, fid), (stat, "median_abs_urban, neg")
                    ] = np.median(np.abs(x.loc[x < 0]))
                    result.loc[
                        (season, use, fid), (stat, "median_abs_rural, neg")
                    ] = np.median(np.abs(y.loc[y < 0]))
                    result.loc[
                        (season, use, fid), (stat, "median_abs_pval, neg")
                    ] = pabs_neg

                    result.loc[(season, use, fid), (stat, "frac_pos_urban")] = (
                        x > 0
                    ).sum() / len(x)
                    result.loc[(season, use, fid), (stat, "frac_pos_rural")] = (
                        y > 0
                    ).sum() / len(y)
                    result.loc[(season, use, fid), (stat, "frac_pos_pval")] = psign

                    print(result.loc[(season, use, fid), stat])
        result.to_csv(
            os.path.join(
                path_out,
                "measures",
                "regression_per_pixel",
                "summary",
                f"{self.prefix}_{self.extent}_{self.name}_{self.heat_wave_thres}_{self.hot_and_dry_thres}_{self.extreme}_city_average{suffix}.csv",
            )
        )

    def get_city_summary_by_event(self, no_crop=False):
        """Calculate the city-level median in the per pixel resistance and recovery per event."""
        if no_crop:
            suffix = "_nocrop"
        else:
            suffix = ""

        hsave = pd.HDFStore(
            os.path.join(
                path_out,
                "measures",
                "regression_per_pixel",
                "summary",
                f"{self.prefix}_{self.extent}_{self.name}_{self.heat_wave_thres}_{self.hot_and_dry_thres}_{self.extreme}_city_average_events{suffix}.h5",
            ),
            mode="w",
        )

        for season in ["DJF", "MAM", "JJA", "SON"]:
            y_test_all = {}
            for use in ["daymet", "topowx", "yyz"]:
                with pd.HDFStore(
                    os.path.join(
                        path_out,
                        "veg_response",
                        f"percity_per_pixel_{self.extent}_{use}_{self.name}_{self.heat_wave_thres}_{self.hot_and_dry_thres}.h5",
                    ),
                    mode="r",
                ) as h:
                    data = h.select(self.extreme).loc[:, ["Resistance", "Recovery"]]
                    data = data.loc[
                        data.index.get_level_values("end").month.map(month_to_season)
                        == season,
                        :,
                    ]
                    if no_crop:
                        h2 = pd.HDFStore(
                            os.path.join(
                                path_out,
                                "luc",
                                f"percity_per_pixel_{self.extent}_{use}_{self.heat_wave_thres}_{self.hot_and_dry_thres}.h5",
                            ),
                            mode="r",
                        )
                        # Get crop fraction
                        crop = h2.select("heat_wave")[7]
                        non_crop = crop.index[crop < 0.5]
                        del crop
                        h2.close()
                        data = data.loc[
                            data.index.intersection(non_crop), :
                        ].reset_index()
                        data = add_urlabel_all(data, self.extent)
                    else:
                        data = data.reset_index()
                    data = data.set_index(
                        ["location", "fid", "start", "end", "row", "col"]
                    )
                y_test_all[use] = data.stack()

            y_test_all = pd.DataFrame(y_test_all).unstack()
            y_test_all.columns = y_test_all.columns.reorder_levels([1, 0])
            y_test_all = y_test_all.stack()
            y_test_all.index.names = [
                "location",
                "fid",
                "start",
                "end",
                "row",
                "col",
                "use",
            ]

            y_test_all_less_mean = y_test_all.groupby(
                ["location", "fid", "start", "end", "use"]
            ).median()
            hsave.append("data", y_test_all_less_mean)

        hsave.close()
