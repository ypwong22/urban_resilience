import os
import pandas as pd
import numpy as np
from utils.analysis import *
from utils.constants import *
from utils.paths import *
from utils.plotting import *
import matplotlib.pyplot as plt
import rasterio as rio
from glob import glob
from time import time


################################################################################################
# Maps of annual and seasonal UHI
################################################################################################
extent = 'tiff_3x'
fid = REPLACE # site 22 needs extra memory

for use in ['daymet', 'topowx', 'yyz']:
    start = time()
    
    if use == 'daymet':
        tmax = read_daymet(fid, 'tmax', extent)
        tmin = read_daymet(fid, 'tmin', extent)
    elif use == 'topowx':
        tmax, tmin = read_topowx(fid, True, extent)
        # Remove Feb 29
        tmax = tmax[(tmax['time'].to_index().month != 2) | (tmax['time'].to_index().day != 29), :, :]
        tmin = tmin[(tmin['time'].to_index().month != 2) | (tmin['time'].to_index().day != 29), :, :]
    elif use == 'yyz':
        tmax, tmin = read_yyz(fid, True, extent)
    else:
        raise 'Not implemented'

    end  = time()
    print(f'Finished reading data @ fid = {fid} in {end - start:.06f} seconds.')


    tvec0 = tmax['time'].to_index().to_period('Q-NOV')

    tmax_anomalies = {}
    tmin_anomalies = {}
    for qtr, season in enumerate(['DJF', 'MAM', 'JJA', 'SON'], 1):
        tmax_q = tmax.loc[tvec0.quarter == qtr, :, :]
        tmin_q = tmax.loc[tvec0.quarter == qtr, :, :]

        # remove the first incomplete DJF and the last incomplete DJF
        if qtr == 1:
            tmax_q = tmax_q[1:-1, :, :]
            tmin_q = tmin_q[1:-1, :, :]
            tvec   = tvec0 [tvec0.quarter == qtr][1:-1]
        else:
            tvec   = tvec0 [tvec0.quarter == qtr]

        # subset the year to post-2000s to match vegetation
        tmax_q = tmax_q[tvec.year >= 2001, :, :]
        tmin_q = tmin_q[tvec.year >= 2001, :, :]

        # subtract the climatology
        tmax_anomalies[season] = (tmax_q.mean(dim = 'time') - tmax_q.mean())
        tmin_anomalies[season] = (tmin_q.mean(dim = 'time') - tmin_q.mean())

    # subset the year to post-2000s to match vegetation
    tmax_q = tmax[tvec0.year >= 2001, :, :]
    tmin_q = tmin[tvec0.year >= 2001, :, :]
        
    # subtract the climatology
    tmax_anomalies['annual'] = (tmax_q.mean(dim = 'time') - tmax_q.mean())
    tmin_anomalies['annual'] = (tmin_q.mean(dim = 'time') - tmin_q.mean())

    # get generic spatial projection
    flist = sorted(glob(os.path.join(path_data, 'Meteorological', 'TOPOWx', extent,
                                     f'reproj_fid{fid}_*.tif')))
    ds = rio.open(flist[0])
    profile = dict(ds.profile)
    profile['count'] = 2
    ds.close()

    # save to geotiff
    for season in ['annual', 'DJF', 'MAM', 'JJA', 'SON']:
        tmax = tmax_anomalies[season]
        tmin = tmin_anomalies[season]

        ds = rio.open(os.path.join(path_out, 'uhi', extent, f'{use}_{season}_{fid}.tif'),
                      'w', **profile)
        ds.write(tmax.values, indexes = 1)
        ds.set_band_description(1, 'tmax')
        ds.write(tmin.values, indexes = 2)
        ds.set_band_description(2, 'tmin')
        ds.close()
    
        # make plot
        grid_list = pd.read_csv(os.path.join(path_intrim, 'Daymet', 'coords',
                                             extent + '_' + str(fid) + '.csv'), index_col = [0,1])
        rowcol_list = pd.MultiIndex.from_product([np.arange(tmax.shape[0]), np.arange(tmax.shape[1])])
        data = pd.DataFrame({'tmax': tmax.values.reshape(-1), 'tmin': tmin.values.reshape(-1)},
                            index = rowcol_list)
        data = pd.concat([data, grid_list], axis = 1, join = 'inner')
    
        # plot stuff
        fig, axes = plt.subplots(2, 1, subplot_kw = {'projection': crs_daymet}, figsize = (8, 12))
        ax = axes[0]
        cf = ax.scatter(data['x'], data['y'], marker = 's',
                        lw = 0., s = 20, vmin = -3.5, vmax = 3.5,
                        c = data['tmax'], cmap = 'RdBu_r', transform = crs_daymet)
        plt.colorbar(cf, ax = ax, shrink = 0.7)
        ax.set_title(f'tmax {fid}')
        add_core_boundary(ax, fid)
        ax = axes[1]
        cf = ax.scatter(data['x'], data['y'], marker = 's',
                        lw = 0.1, s = 20, vmin = -3.5, vmax = 3.5,
                        c = data['tmin'], cmap = 'RdBu_r', transform = crs_daymet)
        plt.colorbar(cf, ax = ax, shrink = 0.7)
        ax.set_title(f'tmin {fid}')
        add_core_boundary(ax, fid)
        fig.savefig(os.path.join(path_out, 'uhi', extent, f'uhi_{use}_{season}_{fid}.png'), dpi = 600.,
                    bbox_inches = 'tight')
        plt.close(fig)
