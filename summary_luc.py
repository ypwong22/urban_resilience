""" For each city, summarize the temporal evolution of percentage area of each land cover. Save to HDF. 
    Also, count the number of dominant land cover pixels (>50%) . 
    There are too few pure grids. Do not go forward with this one. Instead, go with aggregating the
    land cover classes to reduce the number of covariates. 
"""
import os
import numpy as np
import pandas as pd
from utils.paths import *
from utils.constants import *
from utils.plotting import *
from utils.analysis import *
import rasterio as rio
import matplotlib.pyplot as plt
import matplotlib as mpl


def summarize(data):
    new = {}
    for key in data['band'].values:
        val = data.loc[key, :, :]
        new[key] = np.nansum(val.values > 0.5)
    return pd.Series(new)


extent = 'tiff_3x'


##########################################################################################
# Save the land cover area (km2) of the urban & rural areas separately to csv file
# Use the original land cover types since it'll be easy to add up
##########################################################################################
collect = {}
for fid in range(85):
    for which in ['core', 'rural']:
        # Remove pixels with >40% water or >80% impervious area
        dall = mask_water(fid,
                          mask_impervious(fid,
                                          read_nlcd(fid, which, extent).mean(dim = 'year'), 80, extent, which),
                                          extent, which)
        collect[(which,fid)] = summarize(dall)
collect = pd.DataFrame(collect).T
collect.to_csv(os.path.join(path_out, 'luc', 'luc.csv'))

##########################################################################################
# Diagnostic plots of land cover change and temporal average
# Use aggregated land cover types in these diagnostic plots
##########################################################################################
for fid in range(85):
    # Remove pixels with >40% water or >80% impervious area
    dall = mask_water(fid, mask_impervious(fid, agg_nlcd(read_nlcd(fid, 'both', extent)),
                                           80, extent, 'both'), extent, 'both')


    # Plot the percentage land cover change
    fig, ax = plt.subplots(figsize = (10, 10))
    #ax.set_prop_cycle(color=['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
    #                         '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
    #                         '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
    #                         '#17becf', '#9edae5'])
    ax.stackplot(dall['year'], dall.mean(dim = ['row', 'col']).T)
    ax.legend([modis_luc_agg_names[i] for i in dall['band'].values], loc = 'upper right', bbox_to_anchor = (1.3, 1.))
    ax.set_ylim([0,1])
    ax.set_xlim([2001, 2020])
    ax.set_xticks(dall['year'])
    fig.savefig(os.path.join(path_out, 'luc', f'luc_change_{fid}.png'), dpi = 600., bbox_inches = 'tight')
    plt.close(fig)


    # Plot the map of dominant land cover, averaged
    # set the pixels where no land cover type is >50% to -1 (mixed)
    lcmax = dominant_nlcd(dall.mean(dim = 'year'))

    data = pd.DataFrame(lcmax.values.reshape(-1), columns = ['val'],
                        index = pd.MultiIndex.from_product([lcmax['row'].values, lcmax['col'].values]))
    data = data.dropna(axis = 0, how = 'all')
    data = add_grid_coords(fid, data, extent)

    bands = np.insert(dall['band'].values, 0, -1)
    bands_wrapped = np.append(np.insert(bands, 0, -2), 100)
    levels = (bands_wrapped[1:] + bands_wrapped[:-1]) / 2
    norm = mpl.colors.BoundaryNorm(levels, ncolors = len(bands))

    fig, ax = plt.subplots(figsize = (6, 6), subplot_kw = {'projection': crs_daymet})
    cf = ax.scatter(data['x'].values, data['y'].values, c = data['val'].values, s = 10,
                    lw = 0., marker = 's', transform = crs_daymet, cmap = 'tab20',
                    norm = norm)
    add_core_boundary(ax, fid, 'k')
    cb = plt.colorbar(cf, ax = ax, shrink = 0.5, ticks = (levels[1:] + levels[:-1])/2)
    cb.set_ticklabels(['No land cover >50%'] + [modis_luc_agg_names[i] for i in bands[1:]])
    ax.set_title('Dominant land cover')
    fig.savefig(os.path.join(path_out, 'luc', f'luc_map_{fid}.png'), dpi = 600.,
                bbox_inches = 'tight')
    plt.close(fig)
