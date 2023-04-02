""" Boxplots of the EVI, summarized by season and land cover type """
import pandas as pd
import os
import numpy as np
from utils.paths import *
from utils.constants import *
from utils.analysis import *
from utils.plotting import *
import itertools as it
import matplotlib.pyplot as plt
from extract_post_extreme_veg_pergrid import get_veg_data, deseasonalize


def stage_data(fid, name, thres, extent):
    veg_data = get_veg_data(name, fid, extent)
    veg_data_quarter = veg_data['time'].to_index().to_period('Q-NOV').quarter

    lu_data = mask_water(fid, mask_impervious(fid, read_nlcd(fid, 'both', True, extent),
                                              thres, extent), extent)
    # convert to dominant land cover
    lu_data = lu_data.idxmax('band')

    grid_list = pd.read_csv(os.path.join(path_intrim, 'Daymet', 'coords',
                                         extent + '_' + str(fid) + '.csv'), index_col = [0,1])
    col_list, row_list = np.meshgrid(np.arange(veg_data.shape[2]),
                                     np.arange(veg_data.shape[1]))
    rowcol_list = list(zip(row_list.reshape(-1), col_list.reshape(-1)))

    lu_data = pd.Series(lu_data.values.reshape(-1),
                        index = pd.MultiIndex.from_tuples(rowcol_list))
    lu_data = lu_data.dropna()

    veg_data = pd.DataFrame(veg_data.values.reshape(veg_data.shape[0], -1),
                            index = veg_data['time'].to_index(),
                            columns = pd.MultiIndex.from_tuples(rowcol_list,
                                                                names = ['row','col']))
    veg_data = veg_data.dropna(axis = 1, how = 'all')

    subset_index = veg_data.columns.intersection(grid_list.index).intersection(lu_data.index)
    veg_data = veg_data.loc[:, subset_index]
    lu_data = lu_data.loc[subset_index].astype(int)
    xy_pos   = grid_list.loc[subset_index, ['x','y','rural']]

    return veg_data, veg_data_quarter, lu_data, xy_pos


if __name__ == '__main__':
    lu_names = list(modis_luc_agg.keys())
    name = 'MOD09Q1G_EVI'
    if name == 'MOD09Q1G_EVI':
        thres = 80
    else:
        thres = 30
    fid_list = range(85) # default is range(85)
    extent = 'tiff_3x'

    for fid in fid_list:
        veg_data, veg_data_quarter, lu_data, xy_pos = stage_data(fid, name, thres, extent)

        # rows: season; cols: urban core vs rural
        fig, axes = plt.subplots(4, 2, figsize = (10, 10), sharex = False, sharey = True)
        for i, season in zip([1,2,3,4], ['DJF','MAM','JJA','SON']):
            veg_data_subset = veg_data.loc[veg_data_quarter == i, :]

            for j, pos in enumerate(['Urban core','Rural']):
                ax = axes[i-1,j]

                n_pixels = [None] * len(lu_names)
                for l, lun in enumerate(lu_names):
                    veg_data_subset2 = veg_data_subset.loc[:, 
                        (lu_data.isin(modis_luc_agg[lun])) & (xy_pos['rural'].astype(int) == j)]
                    n_pixels[l] = veg_data_subset2.shape[1]

                    veg_data_subset2 = veg_data_subset2.values.reshape(-1)
    
                    if len(veg_data_subset2) > 0:
                        ax.boxplot(veg_data_subset2, positions = [l], whis = [5,95],
                                   flierprops = {'markersize': 1, 'linestyle': 'none', 
                                                 'markerfacecolor': 'green'})
    
                ax.set_xticks(np.arange(len(lu_names)))
                if i == 4:
                    ax.set_xticklabels([f'{a} ({b} pixels)' for a,b in zip(lu_names, n_pixels)],
                                       rotation = 90)
                else:
                    ax.set_xticklabels([])
                if i == 1:
                    ax.set_title(pos)
                if j == 0:
                    ax.set_ylabel(season)
        fig.savefig(os.path.join(path_out, 'veg', extent,
                                 'summary_by_luc_' + name + '_' + str(fid) + '.png'),
                    dpi = 600., bbox_inches = 'tight')
        plt.close(fig)
