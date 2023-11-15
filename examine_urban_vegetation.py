import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from utils.paths import *
from utils.analysis import *
import seaborn as sns


fid_list = [7, 60, 79, 78, 65, 51]
city_list = ["Boston", "Augusta", "Tampa", "Orlando", "Dallas", "Los Angeles"]


collect = pd.DataFrame(
    np.nan,
    index=city_list,
    columns=pd.MultiIndex.from_product(
        [
            ["Grass", "Evergreen forest", "Deciduous forest", "Crop", "Shrub"],
            ["GLC_FCS30D urban", "NLCD urban", "NLCD rural"],
        ],
    ),
)


result = {}
for yy in range(2001, 2020):
    data = pd.read_csv(os.path.join(path_intrim, "GLC_FCS30D", f"{yy}_final.csv"))
    result[yy] = data.set_index(["city_name", "lc_descri"]).loc[:, "percentage"]
result = pd.DataFrame(result).mean(axis=1).unstack()
result.index = ["Augusta", "Boston", "Dallas", "Los Angeles", "Orlando", "Tampa"]
collect.loc[:, ("Grass", "GLC_FCS30D urban")] = result["Grassland"]
collect.loc[:, ("Evergreen forest", "GLC_FCS30D urban")] = result.loc[
    :,
    [
        col
        for col in result.columns
        if "evergreen" in col.lower() and "forest" in col.lower()
    ],
].sum(axis=1)
collect.loc[:, ("Deciduous forest", "GLC_FCS30D urban")] = result.loc[
    :,
    [
        col
        for col in result.columns
        if "deciduous" in col.lower() and "forest" in col.lower()
    ],
].sum(axis=1)
collect.loc[:, ("Crop", "GLC_FCS30D urban")] = result.loc[
    :,
    [col for col in result.columns if "cropland" in col.lower()],
].sum(axis=1)
collect.loc[:, ("Shrub", "GLC_FCS30D urban")] = result.loc[
    :,
    [col for col in result.columns if "shrubland" in col.lower()],
].sum(axis=1)


for fid, city in zip(fid_list, city_list):
    urban = agg_nlcd(read_nlcd(fid, "core", "tiff_3x").mean(axis=0))
    urban = da_to_df(urban, urban["band"].to_index()).mean(axis=1)
    urban.index = [modis_luc_agg_names[i] for i in urban.index]
    collect.loc[city, (slice(None), "NLCD urban")] = urban.loc[
        collect.columns.levels[0]
    ].values

    rural = agg_nlcd(read_nlcd(fid, "rural", "tiff_3x").mean(axis=0))
    rural = da_to_df(rural, rural["band"].to_index()).mean(axis=1)
    rural.index = [modis_luc_agg_names[i] for i in rural.index]
    collect.loc[city, (slice(None), "NLCD rural")] = rural.loc[
        collect.columns.levels[0]
    ].values


df_long = collect.T.reset_index().melt(
    id_vars=["level_0", "level_1"],
    value_vars=collect.T.columns,
    var_name="City",
    value_name="Value",
)
df_long.columns = ["Class", "Dataset", "City", "Value"]

g = sns.FacetGrid(df_long, col="City", col_wrap=2, height=2, aspect=2, sharey=True)
g.map(
    sns.barplot,
    "Class",
    "Value",
    "Dataset",
    order=collect.columns.levels[0],
    hue_order=["GLC_FCS30D urban", "NLCD urban", "NLCD rural"],
    palette="Spectral",
    errorbar=None,
)
g.set_titles("{col_name}")
g.set_xticklabels(rotation=90)
g.set_axis_labels("", "")

# Adjusting the legend
g.add_legend(title="", ncol=3, bbox_to_anchor=(0.5, -0.2))

g.fig.savefig(
    os.path.join(path_intrim, "GLC_FCS30D", "plot.png"),
    dpi=600.0,
    bbox_inches="tight",
)
