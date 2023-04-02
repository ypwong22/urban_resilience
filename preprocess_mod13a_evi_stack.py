"""
Reproject the EVI data to Daymet 1km grid using nearest neighbor

3. Stack up the reprojected files
4. Check how the reprojected files look
"""
import os
import numpy as np
from utils.paths import *
from utils.constants import *
from utils.plotting import *
from utils.analysis import *
from glob import glob
from osgeo import gdal,osr
import matplotlib.pyplot as plt
from datetime import datetime


modis_wkt = 'PROJCS["unknown",GEOGCS["unknown",DATUM["unknown",SPHEROID["unknown",6371007.181,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Sinusoidal"],PARAMETER["longitude_of_center",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
src_srs = osr.SpatialReference()
src_srs.ImportFromWkt(modis_wkt)


name = 'MYD13A3.061'
extent = 'tiff_3x' # 'tiff', 'tiff_3x'

#
if name == 'MOD13A3.061':
    ymin = 2001
elif name == 'MYD13A3.061':
    ymin = 2003
for fid in range(85):
    print(fid)
    for yr in range(ymin, 2021, 5):
        outvrt = os.path.join(path_data, 'Vegetation', 'EVI', 'tempdir', extent, 'vsimem',
                              f'{name}_stacked_{fid}_{yr}.vrt')
        outtif = os.path.join(path_intrim, 'vegetation', f'{name}_EVI', extent,
                              f'reproj_fid{fid}_{yr}-{min(yr+4,2020)}.tif')
        tifs = sorted(glob(os.path.join(path_data, 'Vegetation', 'EVI', 'tempdir', extent,
                                        f'{name}_reproj_{fid}_{yr}*.tif'))) + \
               sorted(glob(os.path.join(path_data, 'Vegetation', 'EVI', 'tempdir', extent,
                                        f'{name}_reproj_{fid}_{yr+1}*.tif'))) + \
               sorted(glob(os.path.join(path_data, 'Vegetation', 'EVI', 'tempdir', extent,
                                        f'{name}_reproj_{fid}_{yr+2}*.tif'))) + \
               sorted(glob(os.path.join(path_data, 'Vegetation', 'EVI', 'tempdir', extent,
                                        f'{name}_reproj_{fid}_{yr+3}*.tif'))) + \
               sorted(glob(os.path.join(path_data, 'Vegetation', 'EVI', 'tempdir', extent,
                                        f'{name}_reproj_{fid}_{yr+4}*.tif')))

        outds = gdal.BuildVRT(outvrt, tifs, separate=True)
        outds = gdal.Translate(outtif, outds, creationOptions = ["TILED=YES"])


#
for fid in range(85):
    values = read_evi(fid, name + '_EVI', extent)

    fig, axes = plt.subplots(2, 1, figsize = (6, 12))

    ax = axes[0]
    ax.plot(np.nanmean(np.nanmean(values, axis = 2), axis = 1))
    ax.set_title('time series')

    ax = axes[1]
    ax.imshow(np.nanmean(values, axis = 0))
    ax.set_title('mean')

    fig.savefig(os.path.join(path_intrim, 'vegetation', f'{name}_EVI', extent,
                             f'check_plot_{fid}.png'), dpi = 600., bbox_inches = 'tight')
    plt.close(fig)


if name != 'MOD09Q1G': # exclude the gap-filled data set

    # Check the percent missing data in each season
    for fid in range(85):
        values = mask_water(fid, mask_impervious(fid, read_evi(fid, name + '_EVI', extent), 0.8, extent, 'both'), extent, 'both')

        fig, axes = plt.subplots(2, 4, figsize = (20, 12))
        for i, season in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
            filt = values['time'].to_index().month.isin(season_to_month[season])
            temp = values[filt, :, :]
            mask = np.broadcast_to(np.isnan(values).all(axis = 0, keepdims = True), temp.shape)
            is_nan = np.ma.masked_where(mask, np.isnan(temp.values).astype(int))
            pct_nans_ts  = is_nan.mean(axis = 2).mean(axis = 1) * 100.
            pct_nans_map = is_nan.mean(axis = 0) * 100.

            ax = axes[0, i]
            ax.plot(pct_nans_ts)
            ax.set_title('time series')
            ax.set_ylim([0, 100.])

            ax = axes[1, i]
            ax.imshow(pct_nans_map, vmin = 0, vmax = 100, cmap = 'inferno_r')
            ax.set_title('mean')

        fig.savefig(os.path.join(path_intrim, 'vegetation', f'{name}_EVI', extent,
                                 f'check_plot2_{fid}.png'), dpi = 600., bbox_inches = 'tight')
        plt.close(fig)

    # Check the percent missing data in each extreme event
    heat_wave = {}
    for use in ['daymet', 'topowx', 'yyz']:
        filename = os.path.join(path_out, 'extreme_events', f'percity_{extent}_{use}_MOD09Q1G_EVI_90_85.h5')
        with pd.HDFStore(filename) as hf:
            heat_wave[use] = hf.select('heat_wave')

    for fid in range(85):
        evi = mask_water(fid, mask_impervious(fid, read_evi(fid, name + '_EVI', extent), 80, extent, 'both'), extent, 'both')

        ncols = 0
        heat_wave_fid = {}
        for use in ['daymet', 'topowx', 'yyz']:
            heat_wave_fid[use] = heat_wave[use].loc[fid, :]
            heat_wave_fid[use] = heat_wave_fid[use].loc[heat_wave_fid[use].index.get_level_values('start') >= evi['time'].to_index()[0]]
            ncols = max(ncols, heat_wave_fid[use].shape[0])

        nrows = 3 # int(np.ceil(evi_event.shape[0] / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize = (ncols * 3, nrows * 3))
        for i, use in enumerate(['daymet', 'topowx', 'yyz']):
            for eid in range(heat_wave_fid[use].shape[0]):
                evi_event = evi.loc[heat_wave_fid[use].index[eid][0]:heat_wave_fid[use].index[eid][1], :, :]

                ## do not plot every single one; count as missing if any single month is missing
                #for mon in range(evi_event.shape[0]):
                #    ax = axes.flat[mon]
                #    ax.imshow(evi_event[mon, :, :], vmin = -0.2, vmax = 1., cmap = 'viridis_r')
                #    ax.set_title(datetime.strftime(evi_event['time'].to_index()[mon], '%Y-%m'))
                #    # calculates percent missing
                #    # np.isnan(evi_event[mon, :, :])
                #for i in range(evi_event.shape[0], ncols * nrows):
                #    ax.axis('off')

                ax = axes[i, eid]
                cf = ax.imshow(np.isnan(evi_event).sum(axis = 0) / evi_event.shape[0] * 100., vmin = 0., vmax = 100., cmap = 'inferno_r')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(datetime.strftime(evi_event['time'].to_index()[0], '%Y-%m') + f' + {evi_event.shape[0]} months')
            
            for eid in range(heat_wave_fid[use].shape[0], ncols):
                ax = axes[i, eid]
                ax.axis('off')
        cax = fig.add_axes([0.15, 0.07, 0.7, 0.02])
        plt.colorbar(cf, cax = cax, orientation = 'horizontal', label = 'Percentage missing months')
        fig.savefig(os.path.join(path_intrim, 'vegetation', f'{name}_EVI', extent, 
                                f'check_heat_wave_{fid}.png'), dpi = 600., bbox_inches = 'tight')
        plt.close(fig)
