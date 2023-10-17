import os
from re import L
import numpy as np
import pandas as pd
import rasterio as rio
import xarray as xr
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from .constants import *
from .paths import *
from .analysis import *
from .extremes import *
from scipy.stats import linregress
import itertools as it
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, spearmanr
from matplotlib.colors import BoundaryNorm
from tqdm import tqdm


class Setup:
    def __init__(self, use=["daymet", "topowx", "yyz"]):
        self.extent = "tiff_3x"
        self.name = "MOD09Q1G_EVI"
        self.heat_wave_thres = 90  # 90th percentile for heat wave events
        self.hot_and_dry_thres = 85  # 85th percentile for compound hot & dry events
        self.use = use

    def __repr__(self):
        return f"extent = {self.extent}\nuse = {self.use}\nname = {self.name}\nheat_wave_thres = {self.heat_wave_thres}\nhot_and_dry_thres = {self.hot_and_dry_thres}"

    def __str__(self):
        return f"extent = {self.extent}\nuse = {self.use}\nname = {self.name}\nheat_wave_thres = {self.heat_wave_thres}\nhot_and_dry_thres = {self.hot_and_dry_thres}"

    def format_prefix(self):
        return f"{self.extent}_{self.use}_{self.name}_{self.heat_wave_thres}_{self.hot_and_dry_thres}"

    def format_prefix_noveg(self):
        return (
            f"{self.extent}_{self.use}_{self.heat_wave_thres}_{self.hot_and_dry_thres}"
        )

    def format_prefix_noveg_noevent(self):
        return f"{self.extent}_{self.use}"

    def set_use(self, use):
        self.use = use


class Filter(Setup):
    def __init__(self, fid, use):
        super().__init__(use)
        self.fid = fid
        self.impervious_threshold = 0.8

    def apply(self, da, season=None):
        """Remove unwanted pixels, which include
        (1) dominated by water or impervious area > 80%
        (2) EVI is too low during the season
        """
        da = mask_low_evi_seasonal(
            mask_crop(
                self.fid,
                mask_water(
                    self.fid,
                    mask_impervious(
                        self.fid, da, self.impervious_threshold, self.extent, "both"
                    ),
                    self.extent,
                    "both",
                ),
                self.extent,
                "both",
            ),
            self.fid,
            self.name,
            self.extent,
            season,
        )
        return da


class Events(Setup):
    def __init__(self, fid, use, clim_period=[2003, 2016]):
        super().__init__(use)
        self.fid = fid
        self.clim_period = clim_period

    def format_prefix(self):
        """overwrite"""
        return (
            f"{self.extent}_{self.use}_{self.heat_wave_thres}_{self.hot_and_dry_thres}"
        )

    def get_spatial_avg(self):
        if self.use == "daymet":
            tmax = read_daymet(self.fid, "tmax", self.extent)
        elif self.use == "topowx":
            tmax, _ = read_topowx(self.fid, self.extent)
            # Remove Feb 29
            tmax = tmax[
                (tmax["time"].to_index().month != 2)
                | (tmax["time"].to_index().day != 29),
                :,
                :,
            ]
        elif self.use == "yyz":
            tmax, _ = read_yyz(self.fid, self.extent)
        else:
            raise "Not implemented"

        with xr.open_dataset(
            os.path.join(
                path_intrim, "Daymet", self.extent, "spi_" + str(self.fid) + ".nc"
            )
        ) as hr:
            if self.use != "yyz":
                spi = hr["spi"][30:, :, :].copy(deep=True)
            else:
                spi = hr["spi"].copy(deep=True)

        # Defining the heat wave & hot and dry events at city level does not require filtering at all.

        # Convert to monthly average
        tmax = tmax.resample({"time": "1M"}).mean()
        spi = spi.resample(
            {"time": "1M"}
        ).last()  # since SPI was calculated on 30 day rolling windows

        # Remove the same monthly climatology for the three temperature datasets
        tmax_clim = (
            tmax[
                (tmax["time"].to_index().year >= self.clim_period[0])
                & (tmax["time"].to_index().year <= self.clim_period[1]),
                :,
                :,
            ]
            .groupby("time.month")
            .mean("time")
        )
        tmax = tmax.groupby("time.month") - tmax_clim

        # Incomplete subset of the time series
        if (self.clim_period[0] == 2003) & (self.clim_period[1] == 2016):
            ymin = max(2001, tmax["time"].to_index().year[0])  # start year of EVI data
            ymax = min(2019, tmax["time"].to_index().year[-1])  # end year of EVI data
            tmax = tmax[
                (tmax["time"].to_index().year >= ymin)
                & (tmax["time"].to_index().year <= ymax),
                :,
                :,
            ]
            spi = spi[
                (spi["time"].to_index().year >= ymin)
                & (spi["time"].to_index().year <= ymax),
                :,
                :,
            ]
        else:
            tmax = tmax[
                (tmax["time"].to_index().year >= self.clim_period[0])
                & (tmax["time"].to_index().year <= self.clim_period[1]),
                :,
                :,
            ]
            spi = spi[
                (spi["time"].to_index().year >= self.clim_period[0])
                & (spi["time"].to_index().year <= self.clim_period[1]),
                :,
                :,
            ]

        # Convert meteorological factors to spatial average
        tmax = tmax.mean(dim=["row", "col"])
        spi = spi.mean(dim=["row", "col"])

        # Identify the monthly extremes
        is_heat_wave = np.full(len(tmax.values), False)
        is_hot_and_dry = np.full(len(tmax.values), False)

        for mon in range(12):
            is_heat_wave[mon::12] = identify_high_extremes(
                tmax.values[mon::12], self.heat_wave_thres
            )

            temp1 = identify_high_extremes(tmax.values[mon::12], self.hot_and_dry_thres)
            temp2 = identify_low_extremes(
                spi.values[mon::12], 100 - self.hot_and_dry_thres
            )
            is_hot_and_dry[mon::12] = temp1 & temp2

        # Attach time vector to day 1 of the month
        tvec = pd.DatetimeIndex(
            [date(t.year, t.month, 1) for t in tmax["time"].to_index()]
        )

        heat_wave_start, heat_wave_end = get_events(is_heat_wave, tvec)
        hot_and_dry_start, hot_and_dry_end = get_events(is_hot_and_dry, tvec)

        # Get the duration & intensity of each heat wave event
        heat_wave_intensity = np.empty(len(heat_wave_start))
        heat_wave_duration = np.empty(len(heat_wave_start))
        for i in range(len(heat_wave_start)):
            heat_wave_intensity[i] = tmax.values[
                (tvec >= heat_wave_start[i]) & (tvec <= heat_wave_end[i])
            ].mean()
            heat_wave_duration[i] = (
                relativedelta(heat_wave_end[i], heat_wave_start[i]).months + 1
            )

        # Get the duration & intensity of each hot and dry event
        hot_intensity = np.empty(len(hot_and_dry_start))
        dry_intensity = np.empty(len(hot_and_dry_start))
        hot_and_dry_duration = np.empty(len(hot_and_dry_start))
        for i in range(len(hot_and_dry_start)):
            hot_intensity[i] = tmax.values[
                (tvec >= hot_and_dry_start[i]) & (tvec <= hot_and_dry_end[i])
            ].mean()
            dry_intensity[i] = spi.values[
                (tvec >= hot_and_dry_start[i]) & (tvec <= hot_and_dry_end[i])
            ].mean()
            hot_and_dry_duration[i] = (
                relativedelta(hot_and_dry_end[i], hot_and_dry_start[i]).months + 1
            )

        # Save all relevant info in dataframe
        self.heat_wave = pd.DataFrame(
            {
                "fid": self.fid,
                "start": heat_wave_start,
                "end": heat_wave_end,
                "intensity": heat_wave_intensity,
                "duration": heat_wave_duration,
            }
        ).set_index(["fid", "start", "end"])
        self.hot_and_dry = pd.DataFrame(
            {
                "fid": self.fid,
                "start": hot_and_dry_start,
                "end": hot_and_dry_end,
                "hot_intensity": hot_intensity,
                "dry_intensity": dry_intensity,
                "duration": hot_and_dry_duration,
            }
        ).set_index(["fid", "start", "end"])

        return self.heat_wave, self.hot_and_dry

    def get_aux_spatial_avg(self, varname, extreme_events):
        """Get the average urban, rural, and total average tmax and SPI during and after the heat wave,
        and averaged over the same season as the end of the heat wave."""
        if varname == "spi":
            with xr.open_dataset(
                os.path.join(
                    path_intrim, "Daymet", self.extent, "spi_" + str(self.fid) + ".nc"
                )
            ) as hr:
                var = hr["spi"].copy(deep=True)
        elif varname == "vpd":
            var = read_daymet(self.fid, "vpd", self.extent)
        elif varname == "tmax":
            if self.use == "daymet":
                var = read_daymet(self.fid, "tmax", self.extent)
            elif self.use == "topowx":
                var, _ = read_topowx(self.fid, self.extent)
                # Remove Feb 29
                var = var[
                    (var["time"].to_index().month != 2)
                    | (var["time"].to_index().day != 29),
                    :,
                    :,
                ]
            elif self.use == "yyz":
                var, _ = read_yyz(self.fid, self.extent)
            else:
                raise "Not implemented"
        elif varname == "tmin":
            if self.use == "daymet":
                var = read_daymet(self.fid, "tmin", self.extent)
            elif self.use == "topowx":
                _, var = read_topowx(self.fid, self.extent)
                # Remove Feb 29
                var = var[
                    (var["time"].to_index().month != 2)
                    | (var["time"].to_index().day != 29),
                    :,
                    :,
                ]
            elif self.use == "yyz":
                _, var = read_yyz(self.fid, self.extent)
            else:
                raise "Not implemented"

        if varname == "spi":
            var = var.resample({"time": "1M"}).last()
        else:
            var = var.resample({"time": "1M"}).mean()

        F = Filter(self.fid, self.use)
        var = F.apply(var)

        # Remove the monthly climatology over the same period for the three temperature datasets
        # over the whole city in order to retain the urban heat island effect.
        # For VPD, also remove this climatology; For SPI, there is no need because it's already calculated on a monthly basis.
        if varname in ["tmax", "tmin", "vpd"]:
            var_clim = (
                var[
                    (var["time"].to_index().year >= self.clim_period[0])
                    & (var["time"].to_index().year <= self.clim_period[1]),
                    :,
                    :,
                ]
                .groupby("time.month")
                .mean()
            )
            var = var.groupby("time.month") - var_clim.mean(dim=["row", "col"])

        # Attach time vector to day 1 of the month
        tvec = pd.DatetimeIndex(
            [date(t.year, t.month, 1) for t in var["time"].to_index()]
        )

        urban_mask = get_mask(self.fid, "core", True, self.extent)
        rural_mask = get_mask(self.fid, "rural", True, self.extent)

        var_series = pd.DataFrame(
            {
                "urban": var.where(urban_mask).mean(dim=["row", "col"]).values,
                "rural": var.where(rural_mask).mean(dim=["row", "col"]).values,
                "total": var.mean(dim=["row", "col"]).values,
            },
            index=tvec,
        )

        var = pd.DataFrame(
            np.nan,
            index=pd.MultiIndex.from_tuples(
                [
                    list(i) + [j]
                    for i in extreme_events.index
                    for j in ["urban", "rural", "total"]
                ],
                names=["fid", "start", "end", "location"],
            ),
            columns=["in_event", "post_event", "of_season"],
        )

        for i, ind in enumerate(extreme_events.index):
            in_event = var_series.loc[ind[1] : ind[2], :].mean(axis=0)
            post_event = var_series.loc[
                (ind[2] + relativedelta(months=1)) : (ind[2] + relativedelta(months=3)),
                :,
            ].mean(axis=0)

            of_season = var_series.loc[
                var_series.index.month.isin(
                    season_to_month[month_to_season[ind[2].month]]
                ),
                :,
            ].mean(axis=0)

            for j in ["urban", "rural", "total"]:
                var.loc[ind + (j,), "in_event"] = in_event[j]
                var.loc[ind + (j,), "post_event"] = post_event[j]
                var.loc[ind + (j,), "of_season"] = of_season[j]

        return var

    def get_aux_pixel(self, varname, extreme_events):
        """Get the pixel-level average tmax and SPI during and after the heat wave."""
        if varname == "spi":
            with xr.open_dataset(
                os.path.join(
                    path_intrim, "Daymet", self.extent, "spi_" + str(self.fid) + ".nc"
                )
            ) as hr:
                var = hr["spi"].copy(deep=True)
                # remove the preceding NaN because it messes up the da_to_df
                var = var.loc[var["time"].to_index().year >= 2000, :, :]
        elif varname == "vpd":
            var = read_daymet(self.fid, "vpd", self.extent)
        elif varname == "tmax":
            if self.use == "daymet":
                var = read_daymet(self.fid, "tmax", self.extent)
            elif self.use == "topowx":
                var, _ = read_topowx(self.fid, self.extent)
                # Remove Feb 29
                var = var[
                    (var["time"].to_index().month != 2)
                    | (var["time"].to_index().day != 29),
                    :,
                    :,
                ]
            elif self.use == "yyz":
                var, _ = read_yyz(self.fid, self.extent)
            else:
                raise "Not implemented"
        elif varname == "tmin":
            if self.use == "daymet":
                var = read_daymet(self.fid, "tmin", self.extent)
            elif self.use == "topowx":
                _, var = read_topowx(self.fid, self.extent)
                # Remove Feb 29
                var = var[
                    (var["time"].to_index().month != 2)
                    | (var["time"].to_index().day != 29),
                    :,
                    :,
                ]
            elif self.use == "yyz":
                _, var = read_yyz(self.fid, self.extent)
            else:
                raise "Not implemented"

        # SPI is already monthly
        if varname != "spi":
            var = var.resample({"time": "1M"}).mean()

        F = Filter(self.fid, self.use)
        var = F.apply(var)

        if varname in ["tmax", "tmin"]:
            # Monthly anomalies, relative to the city-average temperature, in order to retain only the
            # urban heat island effect.
            var = var - var.mean(dim=["row", "col"])
        elif varname == "vpd":
            # Remove the monthly climatology in order to de-seasonalize.
            var_clim = (
                var[
                    (var["time"].to_index().year >= self.clim_period[0])
                    & (var["time"].to_index().year <= self.clim_period[1]),
                    :,
                    :,
                ]
                .groupby("time.month")
                .mean()
            )
            var = var.groupby("time.month") - var_clim

        # No need to apply the filter here; the vegetation filter will be auto applied when combining the predictors
        # F = Filter(self.fid)
        # var_df = da_to_df(F.apply(var))
        var_df = da_to_df(var)

        # Attach time vector to day 1 of the month
        tvec = pd.DatetimeIndex([date(t.year, t.month, 1) for t in var_df.index])
        var_df.index = tvec

        var = pd.DataFrame(
            np.nan,
            index=pd.MultiIndex.from_tuples(
                [
                    list(i) + list(j)
                    for i in extreme_events.index
                    for j in var_df.columns
                ],
                names=["fid", "start", "end", "row", "col"],
            ),
            columns=["in_event", "post_event", "of_season"],
        )

        for i, ind in enumerate(extreme_events.index):
            in_event = var_df.loc[ind[1] : ind[2], :].mean(axis=0)
            post_event = var_df.loc[
                (ind[2] + relativedelta(months=1)) : (ind[2] + relativedelta(months=3)),
                :,
            ].mean(axis=0)

            # ensure consistency between the time period of the datasets for the seasonal mean
            of_season = var_df.loc[
                var_df.index.month.isin(season_to_month[month_to_season[ind[2].month]])
                & (var_df.index.year >= self.clim_period[0])
                & (var_df.index.year <= self.clim_period[1]),
                :,
            ].mean(axis=0)

            try:
                var.loc[ind, "in_event"] = in_event.loc[var.loc[ind, :].index].values
            except:
                import pdb

                pdb.set_trace()

            var.loc[ind, "post_event"] = post_event.loc[var.loc[ind, :].index].values
            var.loc[ind, "of_season"] = of_season.loc[var.loc[ind, :].index].values

        return var


class Veg(Setup):
    def __init__(self, fid, use):
        super().__init__(use)
        self.fid = fid

        # number of months to extract prior to and after heat wave
        # use fairly large numbers to make sure all heat waves can be accomodated; isn't a problem with HDF file.
        self.bracket = (-18, 36)

    def get_spatial_avg(self, extreme_events):
        """Get the average urban and rural vegetation level before, during and after the heat wave.
        Before: the mirror approach; ignore whether there is a prior event because its effect is already in the prior condition of the EVI.
        After : min(36 months, next event)
        """
        veg0 = read_evi(self.fid, self.name, self.extent)

        F = Filter(self.fid)
        veg0 = F.apply(veg0)

        # Remove the seasonal average but add back the annual average to keep magnitude # the climatology
        ## Do not de-seasonalized the standard deviation because it makes no sense for spatial averaging
        ##mean_monthly = veg0.groupby('time.month').mean('time')
        ##mean_mean    = veg0.mean('time')
        ##std_monthly  = veg0.groupby('time.month').std('time')
        ##std_monthly  = std_monthly.where(std_monthly > 0.)
        ##std_mean     = veg0.std('time')
        ##std_mean     = std_mean.where(std_mean > 0.)
        ##veg0 = (veg0.groupby('time.month') - mean_monthly).groupby('time.month') / std_monthly * std_mean + mean_mean
        veg0 = (
            veg0.groupby("time.month")
            - veg0.groupby("time.month").mean("time")
            + veg0.mean("time")
        )
        mean_mean = veg0.mean("time")
        std_mean = veg0.std("time")
        std_mean = std_mean.where(std_mean > 0.0)

        # Convert vegetation to urban/rural average
        urban_mask = get_mask(self.fid, "core", True, self.extent)
        rural_mask = get_mask(self.fid, "rural", True, self.extent)
        tvec = pd.DatetimeIndex(
            [date(t.year, t.month, 1) for t in veg0["time"].to_index()]
        )
        veg = pd.DataFrame(
            {
                "urban": veg0.where(urban_mask).mean(dim=["row", "col"]).values,
                "rural": veg0.where(rural_mask).mean(dim=["row", "col"]).values,
            },
            index=tvec,
        )
        veg_mean_mean = {
            "urban": float(mean_mean.where(urban_mask).mean()),
            "rural": float(mean_mean.where(rural_mask).mean(dim=["row", "col"])),
        }
        veg_std_mean = {
            "urban": float(std_mean.where(urban_mask).mean()),
            "rural": float(std_mean.where(rural_mask).mean(dim=["row", "col"])),
        }
        del veg0

        # Some extreme events may be skipped because they do not fall inside the time range of the EVI data
        # or if there is not any vegetation
        self.veg_response = pd.DataFrame(
            np.nan,
            index=pd.MultiIndex.from_tuples(
                [
                    list(i) + [j]
                    for i in extreme_events.index
                    for j in ["urban", "rural"]
                ],
                names=["fid", "start", "end", "location"],
            ),
            columns=[f"Month {i}" for i in range(*self.bracket)]
            + ["pre_veg_norm", "Resistance", "Resilience", "Recovery"],
        )

        for i, ind in enumerate(extreme_events.index):
            dur = relativedelta(ind[2], ind[1]).months + 1
            mirror = ind[1] - relativedelta(months=dur)
            if mirror < veg.index[0]:
                continue  # extreme event starts too early
            if ind[2] >= veg.index[-1]:
                continue  # extreme event ends too late

            start = max(ind[1] + relativedelta(months=self.bracket[0]), veg.index[0])
            end = min(
                ind[1] + relativedelta(months=self.bracket[-1] - 1), veg.index[-1]
            )
            veg_temp = veg.loc[start:end, :].copy()

            # nullify the values not mirrored prior to the event
            if mirror > start:
                veg_temp = veg_temp.loc[mirror:, :]
                start = mirror

            # nullify the values that overlap with the next event
            if i < (len(extreme_events.index) - 1):
                start_next = min(
                    extreme_events.index[i + 1][1] - relativedelta(months=1), end
                )
                if end > start_next:
                    veg_temp = veg_temp.loc[:start_next, :]
                end = start_next

            days = [
                f"Month {i}"
                for i in range(
                    int(relativedelta(start, ind[1]).years) * 12
                    + int(relativedelta(start, ind[1]).months),
                    int(relativedelta(end, ind[1]).years) * 12
                    + int(relativedelta(end, ind[1]).months)
                    + 1,
                )
            ]
            self.veg_response.loc[ind, days] = veg_temp.T.loc[
                ["urban", "rural"], :
            ].values

            pre_event = veg_temp.loc[
                mirror : (ind[1] - relativedelta(months=1)), :
            ].mean(axis=0)

            # the sign * the maximum magnitude
            in_event = veg_temp.loc[ind[1] : ind[2], :].apply(
                lambda x: x[np.argmax(np.abs(x))], axis=0
            )

            # The vegetation recovery process is somewhat jagged. Therefore, it seems reasonable to
            # consider the average vegetation level at least three months after the heat wave ends.
            # Two heat waves separated by < 3 months have been merged into the same event.
            post_event = veg_temp.loc[
                (ind[2] + relativedelta(months=1)) : (ind[2] + relativedelta(months=3)),
                :,
            ].mean(axis=0)

            for loc in ["urban", "rural"]:
                # Use the inverse of Isbell et al. 2015, in order to have better stability (prevent the resistance from soaring into infinity)
                self.veg_response.loc[ind + (loc,), "Resistance"] = (
                    2
                    * (in_event[loc] - pre_event[loc])
                    / (pre_event[loc] + abs(in_event[loc] - pre_event[loc]))
                )
                self.veg_response.loc[ind + (loc,), "Resilience"] = (
                    2
                    * (post_event[loc] - in_event[loc])
                    / (
                        abs(post_event[loc] - in_event[loc])
                        + abs(pre_event[loc] - in_event[loc])
                    )
                )  # larger = speedier recovery
                self.veg_response.loc[ind + (loc,), "Recovery"] = (
                    2
                    * (post_event[loc] - pre_event[loc])
                    / (pre_event[loc] + abs(pre_event[loc] - post_event[loc]))
                )

                # Normalize the pre-event vegetation during the mirrored period
                self.veg_response.loc[ind + (loc,), "pre_veg_norm"] = (
                    pre_event[loc] - veg_mean_mean[loc]
                ) / veg_std_mean[loc]

        return self.veg_response

    def get_luc_avg(self, extreme_events):
        """Get the average urban and rural vegetation level before, during and after the heat wave.
        Before: the mirror approach; ignore whether there is a prior event because its effect is already in the prior condition of the EVI.
        After : min(36 months, next event)
        """
        veg0 = read_evi(self.fid, self.name, self.extent)

        F = Filter(self.fid)
        veg0 = F.apply(veg0)

        # Remove the seasonal average but add back the annual average to keep magnitude # the climatology
        ## Do not de-seasonalized the standard deviation because it makes no sense for spatial averaging
        ##mean_monthly = veg0.groupby('time.month').mean('time')
        ##mean_mean    = veg0.mean('time')
        ##std_monthly  = veg0.groupby('time.month').std('time')
        ##std_monthly  = std_monthly.where(std_monthly > 0.)
        ##std_mean     = veg0.std('time')
        ##std_mean     = std_mean.where(std_mean > 0.)
        ##veg0 = (veg0.groupby('time.month') - mean_monthly).groupby('time.month') / std_monthly * std_mean + mean_mean
        veg0 = (
            veg0.groupby("time.month")
            - veg0.groupby("time.month").mean("time")
            + veg0.mean("time")
        )
        mean_mean = veg0.mean("time")
        std_mean = veg0.std("time")
        std_mean = std_mean.where(std_mean > 0.0)

        # Convert vegetation to per land cover average
        urban_mask = get_mask(self.fid, "core", True, self.extent)
        rural_mask = get_mask(self.fid, "rural", True, self.extent)

        luc = agg_nlcd(read_nlcd(self.fid, "both", self.extent))
        luc = luc.mean("year").idxmax("band")

        tvec = pd.DatetimeIndex(
            [date(t.year, t.month, 1) for t in veg0["time"].to_index()]
        )

        veg = pd.DataFrame(np.nan, index=tvec, columns=modis_luc_short_list)
        veg_mean_mean = pd.Series(np.nan, index=modis_luc_short_list)
        veg_std_mean = pd.Series(np.nan, index=modis_luc_short_list)
        for ln in modis_luc_short_list:
            which = ln.split("_")[0]
            luc_id = modis_luc_agg_id[ln.split("_")[1]]

            if which == "urban":
                mask = (np.abs(luc - luc_id) < 1e-6) & urban_mask
            elif which == "rural":
                mask = (np.abs(luc - luc_id) < 1e-6) & rural_mask

            # Skip the LUC if the coverage is < 10 km2, because 10 pixels is prone to error
            if len(np.where(mask)[0]) < 10:
                continue

            veg.loc[:, ln] = veg0.where(mask).mean(dim=["row", "col"]).values
            veg_mean_mean.loc[ln] = (
                mean_mean.where(mask).mean(dim=["row", "col"]).values
            )
            veg_std_mean.loc[ln] = std_mean.where(mask).mean(dim=["row", "col"]).values

        del veg0

        # Some extreme events may be skipped because they do not fall inside the time range of the EVI data
        # or if there is not any vegetation
        self.veg_response = pd.DataFrame(
            np.nan,
            index=pd.MultiIndex.from_tuples(
                [
                    list(i) + [j]
                    for i in extreme_events.index
                    for j in modis_luc_short_list
                ],
                names=["fid", "start", "end", "location"],
            ),
            columns=[f"Month {i}" for i in range(*self.bracket)]
            + ["pre_veg_norm", "Resistance", "Resilience", "Recovery"],
        )

        for i, ind in enumerate(extreme_events.index):
            dur = relativedelta(ind[2], ind[1]).months + 1
            mirror = ind[1] - relativedelta(months=dur)
            if mirror < veg.index[0]:
                continue  # extreme event starts too early
            if ind[2] >= veg.index[-1]:
                continue  # extreme event ends too late

            start = max(ind[1] + relativedelta(months=self.bracket[0]), veg.index[0])
            end = min(
                ind[1] + relativedelta(months=self.bracket[-1] - 1), veg.index[-1]
            )
            veg_temp = veg.loc[start:end, :].copy()

            # nullify the values not mirrored prior to the event
            if mirror > start:
                veg_temp = veg_temp.loc[mirror:, :]
                start = mirror

            # nullify the values that overlap with the next event
            if i < (len(extreme_events.index) - 1):
                start_next = min(
                    extreme_events.index[i + 1][1] - relativedelta(months=1), end
                )
                if end > start_next:
                    veg_temp = veg_temp.loc[:start_next, :]
                end = start_next

            days = [
                f"Month {i}"
                for i in range(
                    int(relativedelta(start, ind[1]).years) * 12
                    + int(relativedelta(start, ind[1]).months),
                    int(relativedelta(end, ind[1]).years) * 12
                    + int(relativedelta(end, ind[1]).months)
                    + 1,
                )
            ]
            self.veg_response.loc[ind, days] = veg_temp.T.loc[
                modis_luc_short_list, :
            ].values

            pre_event = veg_temp.loc[
                mirror : (ind[1] - relativedelta(months=1)), :
            ].mean(axis=0)

            # the sign * the maximum magnitude
            in_event = veg_temp.loc[ind[1] : ind[2], :].apply(
                lambda x: x[np.argmax(np.abs(x))], axis=0
            )

            # The vegetation recovery process is somewhat jagged. Therefore, it seems reasonable to
            # consider the average vegetation level at least three months after the heat wave ends.
            # Two heat waves separated by < 3 months have been merged into the same event.
            post_event = veg_temp.loc[
                (ind[2] + relativedelta(months=1)) : (ind[2] + relativedelta(months=3)),
                :,
            ].mean(axis=0)

            for loc in modis_luc_short_list:
                # Use the inverse of Isbell et al. 2015, in order to have better stability (prevent the resistance from soaring into infinity)
                self.veg_response.loc[ind + (loc,), "Resistance"] = (
                    2
                    * (in_event[loc] - pre_event[loc])
                    / (pre_event[loc] + abs(in_event[loc] - pre_event[loc]))
                )
                self.veg_response.loc[ind + (loc,), "Resilience"] = (
                    2
                    * (post_event[loc] - in_event[loc])
                    / (
                        abs(post_event[loc] - in_event[loc])
                        + abs(pre_event[loc] - in_event[loc])
                    )
                )  # larger = speedier recovery
                self.veg_response.loc[ind + (loc,), "Recovery"] = (
                    2
                    * (post_event[loc] - pre_event[loc])
                    / (pre_event[loc] + abs(post_event[loc] - pre_event[loc]))
                )

                # Normalize the pre-event vegetation during the mirrored period
                self.veg_response.loc[ind + (loc,), "pre_veg_norm"] = (
                    pre_event[loc] - veg_mean_mean[loc]
                ) / veg_std_mean[loc]

        return self.veg_response

    def get_pixel(self, extreme_events):
        veg = read_evi(self.fid, self.name, self.extent)

        F = Filter(self.fid)
        veg = F.apply(veg)

        # De-seasonalize the average and the standard deviation, but keep the annual climatological magnitude and standard deviation
        mean_monthly = veg.groupby("time.month").mean("time")
        mean_mean = veg.mean("time")
        std_monthly = veg.groupby("time.month").std("time")
        std_monthly = std_monthly.where(std_monthly > 0.0)
        std_mean = veg.std("time")
        std_mean = std_mean.where(std_mean > 0.0)
        veg = (veg.groupby("time.month") - mean_monthly).groupby(
            "time.month"
        ) / std_monthly * std_mean + mean_mean
        veg = da_to_df(veg)

        # For later converting pre-veg to relative standard deviation units
        veg_std_mean = da_to_df(std_mean)
        veg_mean_mean = da_to_df(mean_mean)

        # Attach time vector to day 1 of the month
        tvec = pd.DatetimeIndex([date(t.year, t.month, 1) for t in veg.index])
        veg.index = tvec

        # Some extreme events may be skipped because they do not fall inside the time range of the EVI data
        # or if there is not any vegetation; These skipped events cause NaN in the vegetation response.
        self.veg_response = pd.DataFrame(
            np.nan,
            index=pd.MultiIndex.from_tuples(
                [list(i) + list(j) for i in extreme_events.index for j in veg.columns],
                names=["fid", "start", "end", "row", "col"],
            ),
            columns=[f"Month {i}" for i in range(*self.bracket)]
            + ["pre_veg_norm", "Resistance", "Resilience", "Recovery"],
        )

        for i, ind in enumerate(extreme_events.index):
            dur = relativedelta(ind[2], ind[1]).months + 1
            mirror = ind[1] - relativedelta(months=dur)
            if mirror < veg.index[0]:
                continue  # extreme event starts too early
            if ind[2] >= veg.index[-1]:
                continue  # extreme event ends too late

            start = max(ind[1] + relativedelta(months=self.bracket[0]), veg.index[0])
            end = min(
                ind[1] + relativedelta(months=self.bracket[-1] - 1), veg.index[-1]
            )
            veg_temp = veg.loc[start:end, :].copy()

            # nullify the values not mirrored prior to the event
            if mirror > start:
                veg_temp = veg_temp.loc[mirror:, :]
                start = mirror

            # nullify the values that overlap with the next event
            if i < (len(extreme_events.index) - 1):
                start_next = min(
                    extreme_events.index[i + 1][1] - relativedelta(months=1), end
                )
                if end > start_next:
                    veg_temp = veg_temp.loc[:start_next, :]
                end = start_next

            #
            pre_event = veg_temp.loc[
                mirror : (ind[1] - relativedelta(months=1)), :
            ].mean(axis=0)

            # the sign * the maximum magnitude
            in_event = veg_temp.loc[ind[1] : ind[2], :].apply(
                lambda x: x[np.argmax(np.abs(x))], axis=0
            )

            # the vegetation recovery process is somewhat jagged. Therefore, it seems reasonable to
            # consider the average vegetation level at least three months after the heat wave ends.
            # Two heat waves separated by < 3 months have been merged into the same event.
            post_event = veg_temp.loc[
                (ind[2] + relativedelta(months=1)) : (ind[2] + relativedelta(months=3)),
                :,
            ].mean(axis=0)

            # nullify the rows where pre-event vegetation is <= 0.05, because negative values interfere with the calculation of metrics
            filt = pre_event <= 0.05  # might be problematic!!!!!!!!!!! Too strict?
            pre_event.loc[filt] = np.nan
            in_event.loc[filt] = np.nan
            post_event.loc[filt] = np.nan

            #
            days = [
                f"Month {i}"
                for i in range(
                    int(relativedelta(start, ind[1]).years) * 12
                    + int(relativedelta(start, ind[1]).months),
                    int(relativedelta(end, ind[1]).years) * 12
                    + int(relativedelta(end, ind[1]).months)
                    + 1,
                )
            ]
            veg_temp.loc[:, filt] = np.nan
            self.veg_response.loc[ind, days] = veg_temp.T.loc[
                self.veg_response.loc[ind, :].index, :
            ].values

            # Use the inverse of Isbell et al. 2015, in order to have better stability (prevent the resistance from soaring into infinity)
            resistance = (
                2 * (in_event - pre_event) / (pre_event + abs(in_event - pre_event))
            )
            resilience = (
                2
                * (post_event - in_event)
                / (abs(post_event - in_event) + abs(pre_event - in_event))
            )  # larger = speedier recovery
            recovery = (
                2 * (post_event - pre_event) / (pre_event + abs(post_event - pre_event))
            )

            self.veg_response.loc[ind, "Resistance"] = resistance.loc[
                self.veg_response.loc[ind, :].index
            ].values
            self.veg_response.loc[ind, "Resilience"] = resilience.loc[
                self.veg_response.loc[ind, :].index
            ].values
            self.veg_response.loc[ind, "Recovery"] = recovery.loc[
                self.veg_response.loc[ind, :].index
            ].values

            # Normalize the pre-event vegetation during the mirrored period
            self.veg_response.loc[ind, "pre_veg_norm"] = (
                ((pre_event - veg_mean_mean) / veg_std_mean)
                .loc[self.veg_response.loc[ind, :].index]
                .values
            )

        self.veg_response = self.veg_response.dropna(how="all", axis=0)

        return self.veg_response


class BackgroundClim(Setup):
    """Get the city level 1981-2010 P & T climatology based on Daymet"""

    def __init__(self, fid, use):
        super().__init__(use)
        self.fid = fid

    def calc(self):
        if self.use == "daymet":
            tmax = (
                read_daymet(self.fid, "tmax", self.extent)
                .resample({"time": "1MS"})
                .mean()
            )
            # tmax = tmax.loc[tmax['time'].to_index().year >= 2000, :, :]

            tmin = (
                read_daymet(self.fid, "tmin", self.extent)
                .resample({"time": "1MS"})
                .mean()
            )
            # tmin = tmin.loc[tmin['time'].to_index().year >= 2000, :, :]

            prcp = (
                read_daymet(self.fid, "prcp", self.extent)
                .resample({"time": "1MS"})
                .mean()
            )
            # prcp = prcp.loc[prcp['time'].to_index().year >= 2000, :, :]

            with xr.open_dataset(
                os.path.join(path_intrim, "Daymet", self.extent, f"spi_{self.fid}.nc")
            ) as hr:
                spi = hr["spi"].copy(deep=True)
                # spi = spi.loc[spi['time'].to_index().year >= 2000, :, :].load()

            vpd = (
                read_daymet(self.fid, "vpd", self.extent)
                .resample({"time": "1MS"})
                .mean()
            )
            # vpd = vpd.loc[vpd['time'].to_index().year >= 2000, :, :].load()
        elif self.use == "topowx":
            tmax, tmin = read_topowx(self.fid, self.extent)
            # Remove Feb 29
            tmax = tmax[
                (tmax["time"].to_index().month != 2)
                | (tmax["time"].to_index().day != 29),
                :,
                :,
            ]
            tmin = tmin[
                (tmin["time"].to_index().month != 2)
                | (tmin["time"].to_index().day != 29),
                :,
                :,
            ]
            tmax = tmax.resample({"time": "1MS"}).mean()
            tmin = tmin.resample({"time": "1MS"}).mean()
        elif self.use == "yyz":
            tmax, tmin = read_yyz(self.fid, self.extent)
            tmax = tmax.resample({"time": "1MS"}).mean()
            tmin = tmin.resample({"time": "1MS"}).mean()

        F = Filter(self.fid, self.use)
        tmax = F.apply(tmax)
        tmin = F.apply(tmin)
        if self.use == "daymet":
            prcp = F.apply(prcp)
            spi = F.apply(spi)
            vpd = F.apply(vpd)

        mask_urban = get_mask(self.fid, "core", clip=True, opt=self.extent)
        mask_rural = get_mask(self.fid, "rural", clip=True, opt=self.extent)

        tmax_all = tmax.mean(dim=["row", "col"])
        tmax_all = pd.Series(tmax_all.values, index=tmax_all["time"].to_index())

        tmax_urban = tmax.where(mask_urban).mean(dim=["row", "col"])
        tmax_urban = pd.Series(tmax_urban.values, index=tmax_urban["time"].to_index())

        tmax_rural = tmax.where(mask_rural).mean(dim=["row", "col"])
        tmax_rural = pd.Series(tmax_rural.values, index=tmax_rural["time"].to_index())

        tmin_all = tmin.mean(dim=["row", "col"])
        tmin_all = pd.Series(tmin_all.values, index=tmin_all["time"].to_index())

        tmin_urban = tmin.where(mask_urban).mean(dim=["row", "col"])
        tmin_urban = pd.Series(tmin_urban.values, index=tmin_urban["time"].to_index())

        tmin_rural = tmin.where(mask_rural).mean(dim=["row", "col"])
        tmin_rural = pd.Series(tmin_rural.values, index=tmin_rural["time"].to_index())

        if self.use == "daymet":
            prcp_all = prcp.mean(dim=["row", "col"])
            prcp_all = pd.Series(prcp_all.values, index=prcp_all["time"].to_index())

            prcp_urban = prcp.where(mask_urban).mean(dim=["row", "col"])
            prcp_urban = pd.Series(
                prcp_urban.values, index=prcp_urban["time"].to_index()
            )

            prcp_rural = prcp.where(mask_rural).mean(dim=["row", "col"])
            prcp_rural = pd.Series(
                prcp_rural.values, index=prcp_rural["time"].to_index()
            )

            spi_all = spi.mean(dim=["row", "col"])
            spi_all = pd.Series(spi_all.values, index=spi_all["time"].to_index())

            spi_urban = spi.where(mask_urban).mean(dim=["row", "col"])
            spi_urban = pd.Series(spi_urban.values, index=spi_urban["time"].to_index())

            spi_rural = spi.where(mask_rural).mean(dim=["row", "col"])
            spi_rural = pd.Series(spi_rural.values, index=spi_rural["time"].to_index())

            vpd_all = vpd.mean(dim=["row", "col"])
            vpd_all = pd.Series(vpd_all.values, index=vpd_all["time"].to_index())

            vpd_urban = vpd.where(mask_urban).mean(dim=["row", "col"])
            vpd_urban = pd.Series(vpd_urban.values, index=vpd_urban["time"].to_index())

            vpd_rural = vpd.where(mask_rural).mean(dim=["row", "col"])
            vpd_rural = pd.Series(vpd_rural.values, index=vpd_rural["time"].to_index())

            data = pd.DataFrame(
                {
                    ("prcp", "all"): prcp_all,
                    ("prcp", "urban"): prcp_urban,
                    ("prcp", "rural"): prcp_rural,
                    ("spi", "all"): spi_all,
                    ("spi", "urban"): spi_urban,
                    ("spi", "rural"): spi_rural,
                    ("vpd", "all"): spi_all,
                    ("vpd", "urban"): spi_urban,
                    ("vpd", "rural"): spi_rural,
                    ("tmax", "all"): tmax_all,
                    ("tmax", "urban"): tmax_urban,
                    ("tmax", "rural"): tmax_rural,
                    ("tmin", "all"): tmin_all,
                    ("tmin", "urban"): tmin_urban,
                    ("tmin", "rural"): tmin_rural,
                }
            )
        else:
            data = pd.DataFrame(
                {
                    ("tmax", "all"): tmax_all,
                    ("tmax", "urban"): tmax_urban,
                    ("tmax", "rural"): tmax_rural,
                    ("tmin", "all"): tmin_all,
                    ("tmin", "urban"): tmin_urban,
                    ("tmin", "rural"): tmin_rural,
                }
            )

        return data


class LUC(Setup):
    """Match the annual size of the urban area and the percentage land cover of the rural area
    to individual extreme events.
    """

    def __init__(self, fid):
        super().__init__()
        self.fid = fid

    def get_spatial_avg(self, extreme_events):
        impervious = mask_crop(
            self.fid,
            read_impervious(self.fid, "core", self.extent),
            self.extent,
            "urban",
        )
        impervious_urban_frac = pd.Series(
            impervious.mean(dim=["row", "col"]).values, index=impervious["year"]
        )
        impervious_urban_size = pd.Series(
            impervious.sum(dim=["row", "col"]).values, index=impervious["year"]
        )

        impervious = mask_crop(
            self.fid,
            read_impervious(self.fid, "rural", self.extent),
            self.extent,
            "rural",
        )
        impervious_rural_frac = pd.Series(
            impervious.mean(dim=["row", "col"]).values, index=impervious["year"]
        )
        impervious_rural_size = pd.Series(
            impervious.sum(dim=["row", "col"]).values, index=impervious["year"]
        )

        luc_urban = mask_crop(
            self.fid, read_nlcd(self.fid, "core", self.extent), self.extent, "urban"
        )
        luc_urban = agg_nlcd(luc_urban).mean(dim=["row", "col"])

        luc_rural = mask_crop(
            self.fid, read_nlcd(self.fid, "rural", self.extent), self.extent, "rural"
        )
        luc_rural = agg_nlcd(luc_rural).mean(dim=["row", "col"])

        luc = pd.DataFrame(
            np.concatenate([luc_urban.values, luc_rural.values], axis=0),
            index=pd.MultiIndex.from_product(
                [["urban", "rural"], luc_urban["year"].values]
            ),
            columns=[modis_luc_agg_names[i] for i in luc_urban["band"].values],
        )

        data = pd.DataFrame(
            np.nan,
            index=pd.MultiIndex.from_tuples(
                [
                    list(i) + [j]
                    for i in extreme_events.index
                    for j in ["urban", "rural"]
                ],
                names=["fid", "start", "end", "location"],
            ),
            columns=["impervious_frac", "impervious_size"] + list(luc.columns),
        )

        for year in impervious["year"].values:
            filt = data.index.get_level_values(2).year == year
            data.loc[
                filt & (data.index.get_level_values(3) == "urban"), "impervious_frac"
            ] = impervious_urban_frac.loc[year]
            data.loc[
                filt & (data.index.get_level_values(3) == "urban"), "impervious_size"
            ] = impervious_urban_size.loc[year]
            data.loc[
                filt & (data.index.get_level_values(3) == "rural"), "impervious_frac"
            ] = impervious_rural_frac.loc[year]
            data.loc[
                filt & (data.index.get_level_values(3) == "rural"), "impervious_size"
            ] = impervious_rural_size.loc[year]
            for loc in ["urban", "rural"]:
                data.loc[
                    filt & (data.index.get_level_values(3) == loc), luc.columns
                ] = luc.loc[(loc, year), :].values

        return data

    def get_pixel(self, extreme_events):
        impervious = mask_crop(
            self.fid,
            read_impervious(self.fid, "both", self.extent),
            self.extent,
            "both",
        )
        impervious = da_to_df(impervious, impervious["year"].to_index())

        luc = mask_crop(
            self.fid, read_nlcd(self.fid, "both", self.extent), self.extent, "both"
        )
        luc = agg_nlcd(luc)

        data = pd.DataFrame(
            np.nan,
            index=pd.MultiIndex.from_tuples(
                [
                    list(i) + list(j)
                    for i in extreme_events.index
                    for j in impervious.columns
                ],
                names=["fid", "start", "end", "row", "col"],
            ),
            columns=["impervious_frac"] + list(luc["band"].values),
        )
        for year in luc["year"].values:
            filt = data.index.get_level_values(2).year == year
            data.loc[filt, "impervious_frac"] = impervious.loc[
                year,
                pd.MultiIndex.from_tuples(
                    zip(data.index.get_level_values(3), data.index.get_level_values(4))
                )[filt],
            ].values
            luc_temp = da_to_df(luc.loc[year, :, :, :], luc["band"].to_index())
            data.loc[filt, luc_temp.index] = luc_temp.loc[
                :,
                pd.MultiIndex.from_tuples(
                    zip(data.index.get_level_values(3), data.index.get_level_values(4))
                )[filt],
            ].values.T
        return data


class Elevation(Setup):
    """Urban-rural elevation difference"""

    def __init__(self, fid):
        super().__init__()
        self.fid = fid

    def get_spatial_avg(self):
        elev_urban = float(read_elevation(self.fid, "urban", self.extent).mean())
        elev_rural = float(read_elevation(self.fid, "rural", self.extent).mean())
        return pd.Series(
            {"urban": elev_urban, "rural": elev_rural, "diff": elev_urban - elev_rural}
        )

    def get_pixel(self):
        elev = da_to_df(read_elevation(self.fid, "both", self.extent)).to_frame("elev")
        elev["fid"] = self.fid
        elev = elev.reset_index().set_index(["fid", "row", "col"])
        return elev


class SeasonalAvgTemperaturePixel:
    def __init__(self):
        self.prefix = "percity_per_pixel"
        self.extent = Setup().extent
        self.name = Setup().name  # veg data
        self.heat_wave_thres = Setup().heat_wave_thres
        self.hot_and_dry_thres = Setup().hot_and_dry_thres

    def calc(self):
        """Calculate the urban & rural time series of tmax and tmin
        Takes > 1 hour but can segment that top for loop to run separately
        This is different from in class Sensitivity because we cannot remove the seasonality.

        Calculate the by-pixel seasonal average tmax and tmin post-2001.
        """
        h = pd.HDFStore(
            os.path.join(
                path_out,
                "clim",
                f"seasonal_average_{self.extent}_{self.heat_wave_thres}_{self.hot_and_dry_thres}.h5",
            ),
            mode="w",
        )
        for use in ["daymet", "topowx", "yyz"]:
            for fid in tqdm(range(85)):
                if use == "daymet":
                    tmax = read_daymet(fid, "tmax", self.extent)
                    tmin = read_daymet(fid, "tmin", self.extent)
                elif use == "topowx":
                    tmax, tmin = read_topowx(fid, self.extent)
                    # Remove Feb 29
                    tmax = tmax[
                        (tmax["time"].to_index().month != 2)
                        | (tmax["time"].to_index().day != 29),
                        :,
                        :,
                    ]
                    tmin = tmin[
                        (tmin["time"].to_index().month != 2)
                        | (tmin["time"].to_index().day != 29),
                        :,
                        :,
                    ]
                elif use == "yyz":
                    tmax, tmin = read_yyz(fid, self.extent)
                tmax = (
                    tmax[tmax["time"].to_index().year >= 2001, :, :]
                    .resample({"time": "Q-NOV"})
                    .mean()
                    .groupby("time.month")
                    .mean()
                )
                tmin = (
                    tmin[tmin["time"].to_index().year >= 2001, :, :]
                    .resample({"time": "Q-NOV"})
                    .mean()
                    .groupby("time.month")
                    .mean()
                )

                F = Filter(fid, use)
                for ind, season in zip([2, 5, 8, 11], ["DJF", "MAM", "JJA", "SON"]):
                    tmax.loc[ind, :, :] = F.apply(tmax.loc[ind, :, :], season).values
                    tmin.loc[ind, :, :] = F.apply(tmin.loc[ind, :, :], season).values

                tmax = da_to_df(tmax, time=tmax["month"].values)
                tmax.index = tmax.index.map({2: "DJF", 5: "MAM", 8: "JJA", 11: "SON"})
                tmax.index.name = "season"
                tmax = tmax.stack().stack().reset_index()
                tmax["fid"] = fid
                tmax = tmax.set_index(["season", "fid", "col", "row"]).iloc[:, 0]

                tmin = da_to_df(tmin, time=tmin["month"].values)
                tmin.index = tmin.index.map({2: "DJF", 5: "MAM", 8: "JJA", 11: "SON"})
                tmin.index.name = "season"
                tmin = tmin.stack().stack().reset_index()
                tmin["fid"] = fid
                tmin = tmin.set_index(["season", "fid", "col", "row"]).iloc[:, 0]

                temp = pd.DataFrame({"tmax": tmax, "tmin": tmin})
                h.append(use, temp)
        h.close()

    def calc_mean(self):
        seasonal_average = pd.DataFrame(
            np.nan,
            index=range(85),
            columns=pd.MultiIndex.from_product(
                [
                    ["urban", "rural"],
                    ["tmax", "tmin"],
                    ["DJF", "MAM", "JJA", "SON"],
                    ["daymet", "topowx", "yyz"],
                ]
            ),
        )

        h = pd.HDFStore(
            os.path.join(
                path_out,
                "clim",
                f"seasonal_average_{self.extent}_{self.heat_wave_thres}_{self.hot_and_dry_thres}.h5",
            ),
            mode="r",
        )
        for use in ["daymet", "topowx", "yyz"]:
            temp = (
                add_urlabel_all(h.select(use), self.extent)
                .groupby(["location", "season", "fid"])
                .mean()
            )
            # Calculate the urban & rural average temperature in each season
            for fid, varname in it.product(range(85), ["tmax", "tmin"]):
                for season in ["DJF", "MAM", "JJA", "SON"]:
                    for location in ["urban", "rural"]:
                        seasonal_average.loc[
                            fid, (location, varname, season, use)
                        ] = temp.loc[(location, season, fid), varname]
        self.seasonal_average = seasonal_average
        h.close()


class OptimalTPixel:
    def __init__(self):
        self.prefix = "percity_per_pixel"
        self.extent = Setup().extent
        self.name = Setup().name  # veg data

    def calc(self):
        """Calculate the optimal temperature for each individual pixel.
        This is different from in class Sensitivity because we cannot remove the seasonality.
        """

        def _find_optima(var_sr1, var_sr2, var_sr3, evi_sr, tvec):
            if len(var_sr1) == 0:
                return np.nan
            elif (
                np.isnan(evi_sr[0])
                | np.isnan(var_sr1[0])
                | np.isnan(var_sr2[0])
                | np.isnan(var_sr3[0])
            ):
                return np.nan
            else:
                """For each year, the start of the growing season = monthly EVI > 20% of the seasonal amplitude

                Yin, G., Verger, A., Descals, A., Filella, I., and Peñuelas, J.: Nonlinear thermal responses
                    outweigh water limitation in the attenuated effect of climatic warming on photosynthesis
                    in northern ecosystems, Geophys. Res. Lett., 49, https://doi.org/10.1029/2022GL100096, 2022.
                """
                filt = evi_sr > 0.05
                var_sr1 = var_sr1[filt]
                var_sr2 = var_sr2[filt]
                var_sr3 = var_sr3[filt]
                evi_sr = evi_sr[filt]
                tvec = tvec[filt]

                if len(var_sr1) < 10:
                    # not enough to calculate q90
                    return np.nan

                evi_mask = pd.Series(evi_sr, index=tvec)
                evi_mask = evi_mask.groupby(evi_mask.index.year).apply(
                    lambda ts: ts >= 0.2 * (ts.max() - ts.min())
                )

                var_sr1 = var_sr1[evi_mask]
                var_sr2 = var_sr2[evi_mask]
                var_sr3 = var_sr3[evi_mask]
                evi_sr = evi_sr[evi_mask]

                """ Create 1-degree temperature bins, get the 90th quantile within each temperature bin,
                    calculate the running mean of every 3 temperature bins, and find the bin of maxima.
                """
                edges = np.arange(-5.5, 35.6, 1.0)
                mid = (0.5 * (edges[:-1] + edges[1:])).astype(int)
                q90 = np.full(len(edges) - 1, np.nan)

                for i in range(len(edges) - 1):
                    pool = np.array([])
                    for sr in [var_sr1, var_sr2, var_sr3]:
                        pool = np.concatenate(
                            [pool, evi_sr[(sr >= edges[i]) & (sr < edges[i + 1])]]
                        )
                    if len(pool) > 0:
                        q90[i] = np.percentile(pool, 90)

                mid = mid[~np.isnan(q90)]
                q90 = q90[~np.isnan(q90)]

                # print(mid)
                # print(q90)

                try:
                    smooth = np.convolve(q90, [1 / 3, 1 / 3, 1 / 3], mode="same")
                except:
                    import pdb

                    pdb.set_trace()
                ind_max = np.argmax(smooth)
                if (ind_max == 0) | (ind_max == (len(smooth) - 1)):
                    return np.nan
                else:
                    return mid[ind_max]

        hsave = pd.HDFStore(
            os.path.join(
                path_out, "veg", f"{self.prefix}_{self.extent}_{self.name}_optimalT.h5"
            ),
            mode="w",
        )
        # optima_tmax, optima_tmin

        for fid in tqdm(range(85)):
            evi = read_evi(fid, self.name, self.extent).load()

            tmax = {}
            tmin = {}

            tmax["daymet"] = read_daymet(fid, "tmax", self.extent)
            tmin["daymet"] = read_daymet(fid, "tmin", self.extent)

            tmax["topowx"], tmin["topowx"] = read_topowx(fid, self.extent)
            tmax["topowx"] = tmax["topowx"][
                (tmax["topowx"]["time"].to_index().month != 2)
                | (tmax["topowx"]["time"].to_index().day != 29),
                :,
                :,
            ]
            tmin["topowx"] = tmin["topowx"][
                (tmin["topowx"]["time"].to_index().month != 2)
                | (tmin["topowx"]["time"].to_index().day != 29),
                :,
                :,
            ]

            tmax["yyz"], tmin["yyz"] = read_yyz(fid, self.extent)

            for use in ["daymet", "topowx", "yyz"]:
                F = Filter(fid, use)
                tmax[use] = F.apply(tmax[use].resample({"time": "1MS"}).mean())
                tmin[use] = F.apply(tmin[use].resample({"time": "1MS"}).mean())

            for varname, var in zip(["tmax", "tmin"], [tmax, tmin]):
                temp = evi["time"].to_index()
                for use in ["daymet", "topowx", "yyz"]:
                    temp = var[use]["time"].to_index().intersection(temp)
                evi_temp = evi.loc[temp, :, :]
                var1_temp = var["daymet"].loc[temp, :, :]
                var2_temp = var["topowx"].loc[temp, :, :]
                var3_temp = var["yyz"].loc[temp, :, :]

                # import pdb; pdb.set_trace()
                # _find_optima(var1_temp[:, 0, 73], var2_temp[:, 0, 73], var3_temp[:, 0, 73], evi_temp[:, 0, 73], temp)

                optima = xr.apply_ufunc(
                    _find_optima,
                    var1_temp,
                    var2_temp,
                    var3_temp,
                    evi_temp,
                    input_core_dims=[["time"], ["time"], ["time"], ["time"]],
                    output_core_dims=[[]],
                    kwargs={"tvec": temp},
                    vectorize=True,
                    dask="parallel",
                )

                optima = da_to_df(optima)
                optima = optima.reset_index()
                optima["fid"] = fid
                optima = optima.set_index(["fid", "row", "col"])

                hsave.append(f"optima_{varname}", optima)

        hsave.close()

    def calc_mean(self):
        optimum = pd.DataFrame(
            np.nan,
            index=range(85),
            columns=pd.MultiIndex.from_product([["tmax", "tmin"], ["urban", "rural"]]),
        )
        h = pd.HDFStore(
            os.path.join(
                path_out, "veg", f"{self.prefix}_{self.extent}_{self.name}_optimalT.h5"
            ),
            mode="r",
        )
        for varname in ["tmax", "tmin"]:
            result = add_urlabel_all(h.select(f"optima_{varname}"), self.extent)
            result = result.groupby(["fid", "location"]).mean().iloc[:, 0].unstack()
            for loc in ["urban", "rural"]:
                optimum.loc[:, (varname, loc)] = result.loc[:, loc]
        h.close()
        optimum.to_csv(
            os.path.join(
                path_out, "veg", f"{self.prefix}_{self.extent}_{self.name}_optimalT.csv"
            )
        )


class SensitivityWaterPixel:
    """Calculate and plot the sensitivity of EVI in each season to SPI & VPD at pixel level."""

    def __init__(self):
        self.prefix = "percity_per_pixel"
        self.extent = Setup().extent
        self.name = Setup().name  # veg data

    def calc(self):
        def _corr_calc(var, evi):
            if len(var) == 0:
                return np.nan, np.nan
            elif np.isnan(evi[0]):
                return np.nan, np.nan
            else:
                filt = ~(np.isnan(var) | np.isnan(evi))
                evi = evi[filt]
                var = var[filt]

                if len(evi) < 10:
                    return np.nan, np.nan

                corr, pval = spearmanr(evi, var)
                return corr, pval

        hsave = pd.HDFStore(
            os.path.join(
                path_out,
                "veg",
                f"{self.prefix}_{self.extent}_{self.name}_sensitivity_water.h5",
            ),
            mode="w",
        )
        # optima_tmax, optima_tmin

        for fid in tqdm(range(85)):
            evi = read_evi(fid, self.name, self.extent).load()

            with xr.open_dataset(
                os.path.join(path_intrim, "Daymet", self.extent, f"spi_{fid}.nc")
            ) as hr:
                spi = hr["spi"].copy(deep=True)
                spi = spi.loc[spi["time"].to_index().year >= 2000, :, :].load()

            vpd = read_daymet(fid, "vpd", self.extent).resample({"time": "1MS"}).mean()
            vpd = vpd.loc[vpd["time"].to_index().year >= 2000, :, :].load()

            temp = (
                evi["time"]
                .to_index()
                .intersection(spi["time"].to_index())
                .intersection(vpd["time"].to_index())
            )

            F = Filter(fid, "daymet")
            evi = F.apply(evi.loc[temp, :, :])
            vpd = F.apply(vpd.loc[temp, :, :])
            spi = F.apply(spi.loc[temp, :, :])

            # de-seasonalize
            evi = evi.groupby("time.month") - evi.groupby("time.month").mean()
            vpd = vpd.groupby("time.month") - vpd.groupby("time.month").mean()

            season_ind = temp.month.map(month_to_season)

            for varname, var in zip(["spi", "vpd"], [spi, vpd]):
                corr_seasons = {}
                pval_seasons = {}
                for season in ["DJF", "MAM", "JJA", "SON"]:
                    var_ = var[season_ind == season, :, :]
                    evi_ = evi[season_ind == season, :, :]
                    corr, pval = xr.apply_ufunc(
                        _corr_calc,
                        var_,
                        evi_,
                        input_core_dims=[["time"], ["time"]],
                        output_core_dims=[[], []],
                        vectorize=True,
                        dask="parallelized",
                    )

                    corr_seasons[season] = da_to_df(corr)
                    pval_seasons[season] = da_to_df(pval)

                corr_seasons = pd.DataFrame(corr_seasons)
                corr_seasons["fid"] = fid
                corr_seasons = corr_seasons.reset_index().set_index(
                    ["fid", "row", "col"]
                )

                pval_seasons = pd.DataFrame(pval_seasons)
                pval_seasons["fid"] = fid
                pval_seasons = pval_seasons.reset_index().set_index(
                    ["fid", "row", "col"]
                )

                hsave.append(f"corr_{varname}", corr_seasons)
                hsave.append(f"pval_{varname}", pval_seasons)

        hsave.close()

    def calc_mean(self):
        seasonal_average = pd.DataFrame(
            np.nan,
            index=pd.MultiIndex.from_product(
                [["DJF", "MAM", "JJA", "SON"], range(85), ["urban", "rural", "total"]],
                names=["season", "fid", "location"],
            ),
            columns=["corr_spi", "corr_vpd"],
        )

        h = pd.HDFStore(
            os.path.join(
                path_out,
                "veg",
                f"{self.prefix}_{self.extent}_{self.name}_sensitivity_water.h5",
            ),
            mode="r",
        )
        for use in ["corr_spi", "corr_vpd"]:
            temp = add_urlabel_all(h.select(use), self.extent)
            temp_1 = temp.groupby(["location", "fid"]).mean()
            temp_2 = temp.mean(axis=0)
            # Calculate the urban & rural average temperature in each season
            for season in ["DJF", "MAM", "JJA", "SON"]:
                for location in ["urban", "rural"]:
                    seasonal_average.loc[
                        (season, slice(None), location), use
                    ] = temp_1.loc[location, season].values
                seasonal_average.loc[(season, slice(None), "total"), use] = temp_2.loc[
                    season
                ]
        self.seasonal_average = seasonal_average
        h.close()
