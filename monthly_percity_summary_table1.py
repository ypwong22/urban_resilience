""" Fit Generalized Additive Model between each individual predictor and the sign & magnitude of the resilience metrics.
"""
import pandas as pd
import os
import numpy as np
from utils.analysis import *
from utils.constants import *
from utils.paths import *
from utils.extremes import *
from utils.plotting import *
from utils.regression import *
from scipy.stats import mannwhitneyu, spearmanr, pearsonr, wilcoxon, linregress
from statsmodels.tools.tools import add_constant
from scipy.interpolate import UnivariateSpline
from matplotlib.colors import BoundaryNorm
import itertools as it
from tqdm import tqdm
from scipy.stats import linregress, norm
from pygam import LinearGAM, s
import pingouin


##############################################################################
# Setup
##############################################################################
self = Plotter("heat_wave")
self.prefix = "percity_per_pixel"
self.extent = Setup().extent
self.name = Setup().name  # veg data
self.heat_wave_thres = Setup().heat_wave_thres
self.hot_and_dry_thres = Setup().hot_and_dry_thres

use_gap_to_optimalT = False
if use_gap_to_optimalT:
    folder = "gap_to_optimalT"
else:
    folder = "optimalT"

no_crop = True
if no_crop:
    suffix = "_nocrop"
else:
    suffix = ""

# Uncomment this to re-run
# self.get_city_summary(no_crop)
# self.get_city_summary_by_event(no_crop)


##############################################################################
# Read data
##############################################################################
data = pd.read_csv(
    os.path.join(
        path_out,
        "measures",
        "regression_per_pixel",
        "summary",
        f"{self.prefix}_{self.extent}_{self.name}_{self.heat_wave_thres}_{self.hot_and_dry_thres}_{self.extreme}_city_average{suffix}.csv",
    ),
    index_col=[0, 1, 2],
    header=[0, 1],
)
predictors_, luc_clist = self._get_predictors(use_gap_to_optimalT)
predictors_["season"] = predictors_.index.get_level_values("end").month.map(
    month_to_season
)
predictors = predictors_.groupby(["season", "use", "fid"]).mean()

luc_list = [
    "Crop",
    "Deciduous forest",
    "Evergreen forest",
    "Grass",
    "Mixed forest",
    "Shrub",
    "Wetland",
]
predictors_list = list(predictors.columns)
for luc in luc_list:
    predictors_list.remove(luc)
predictors_list.remove("Developed")

luc_main = predictors.loc[:, luc_list]  # .idxmax(axis = 1)


##############################################################################
# Find the relationships
##############################################################################
# (parameter search for GAM; had to reduce this for too long)
params_grid = dict(lam=np.logspace(-3, 3, 11))
# n_splines = np.array([5, 10, 20, 30, 40]),
# spline_order = np.arange(1, 7),

results = pd.DataFrame(
    np.nan,
    index=pd.MultiIndex.from_product(
        [["Resistance", "Recovery"], ["Sign", "Mag"], predictors_list + luc_list]
    ),
    columns=pd.MultiIndex.from_product(
        [
            ["Spearman", "Spearman_p", "GAM_p", "GAM_AIC", "GAM_R2"],
            ["DJF", "MAM", "JJA", "SON"],
        ]
    ),
)
gam_params = pd.DataFrame(
    np.nan,
    index=pd.MultiIndex.from_product(
        [
            ["Resistance", "Recovery"],
            ["Sign", "Mag"],
            predictors_list + luc_list,
            ["daymet", "topowx", "yyz"],
        ]
    ),
    columns=pd.MultiIndex.from_product(
        [list(params_grid.keys()), ["DJF", "MAM", "JJA", "SON"]]
    ),
)
for which, stat, aux, season in tqdm(
    it.product(
        ["Sign", "Mag"],
        ["Resistance", "Recovery"],
        predictors_list,
        ["DJF", "MAM", "JJA", "SON"],
    )
):
    if which == "Sign":
        y = (
            data.loc[season, (stat, "frac_pos_urban")]
            - data.loc[season, (stat, "frac_pos_rural")]
        )
    else:
        y = (
            data.loc[season, (stat, "median_abs_urban")]
            - data.loc[season, (stat, "median_abs_rural")]
        )

    for aux in predictors_list:
        if stat == "Resistance":
            if "post_event" in aux:
                continue  # irrelevant
        elif stat == "Recovery":
            if "in_event" in aux:
                continue
        x = predictors.loc[season, aux]

        rho = [None] * 3
        rho_p = [None] * 3
        gam_p = [None] * 3
        gam_aic = [None] * 3
        gam_r2 = [None] * 3
        for u, use in enumerate(["daymet", "topowx", "yyz"]):
            x_ = x.loc[use]  # .groupby('fid').mean()
            y_ = y.loc[use]  # .groupby('fid').mean()

            rho[u], rho_p[u] = spearmanr(x_, y_)

            """random_gam = LinearGAM(s(0)).gridsearch(
                x_.values.reshape(-1, 1), y_.values, **params_grid
            )"""
            random_gam = LinearGAM(s(0)).fit(x_, y_)
            gam_p[u] = random_gam.statistics_["p_values"][0]
            gam_aic[u] = random_gam.statistics_["AIC"]
            gam_r2[u] = random_gam.statistics_["pseudo_r2"]["explained_deviance"]

            """
            params = random_gam._get_terms().info
            for kk in params_grid.keys():
                gam_params.loc[(stat, which, aux, use), (kk, season)] = params["terms"][
                    0
                ][kk][0]"""

        results.loc[(stat, which, aux), ("Spearman", season)] = np.median(rho)
        results.loc[(stat, which, aux), ("Spearman_p", season)] = np.median(rho_p)
        results.loc[(stat, which, aux), ("GAM_p", season)] = np.median(gam_p)
        results.loc[(stat, which, aux), ("GAM_AIC", season)] = np.median(gam_aic)
        results.loc[(stat, which, aux), ("GAM_R2", season)] = np.median(gam_r2)

    # LUC: use multiple regression
    temp = pd.concat([luc_main.loc[season, :], y.to_frame("y")], axis=1)
    for aux in luc_list:
        covar = luc_list.copy()
        covar.remove(aux)
        res = pingouin.partial_corr(temp, x=aux, y="y", covar=covar, method="spearman")
        results.loc[(stat, which, aux), ("Spearman", season)] = float(
            res.loc["spearman", "r"]
        )
        results.loc[(stat, which, aux), ("Spearman_p", season)] = float(
            res.loc["spearman", "p-val"]
        )

    # calculate the percent explained variance
    """random_gam = LinearGAM(
        s(0) + s(1) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6)
    ).gridsearch(luc_main.loc[season, :].values, y.values, **params_grid)"""
    random_gam = LinearGAM(s(0) + s(1) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6)).fit(
        luc_main.loc[season, :].values, y.values
    )
    for ll, luc in enumerate(luc_list):
        results.loc[(stat, which, luc), ("GAM_p", season)] = random_gam.statistics_[
            "p_values"
        ][ll]
        results.loc[(stat, which, luc), ("GAM_AIC", season)] = random_gam.statistics_[
            "AIC"
        ]
        results.loc[(stat, which, luc), ("GAM_R2", season)] = random_gam.statistics_[
            "pseudo_r2"
        ]["explained_deviance"]

fix = Setup().format_prefix().replace(f"{Setup().use}_", "")
results.to_csv(
    os.path.join(
        path_out,
        "measures",
        "regression_per_pixel",
        "summary",
        f"percity_spatial_avg_summary_{fix}_{self.extreme}_fig1_table1{suffix}.csv",
    )
)
gam_params.to_csv(
    os.path.join(
        path_out,
        "measures",
        "regression_per_pixel",
        "summary",
        f"percity_spatial_avg_summary_{fix}_{self.extreme}_fig1_table1{suffix}_gam_params.csv",
    )
)
