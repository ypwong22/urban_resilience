""" Climatological EVI in each season & during each extreme event.
    Threshold = 0.05 will be used to mask out the no-vegetation data points. 
    This threshold is taken from Zhao et al. 2016 Prevalent vegetation growth enhancement in urban environment.
"""
import pandas as pd
import os
import numpy as np
import rasterio as rio
from utils.paths import *
from utils.constants import *
from utils.analysis import *
from utils.plotting import *
from datetime import datetime

from summary_veg import read_events, calc_evi_event_normals, calc_evi_normals


if __name__ == '__main__':
    name = 'MOD09Q1G_EVI'
    extent = 'tiff_3x'

    for fid in range(85):
        print(fid)

        events_list = read_events(fid, name, extent)
        veg0 = mask_water(fid, mask_impervious(fid, read_evi(fid, name, extent), 0.8, extent, 'both'), extent, 'both')
        seasonal_mean, _ = calc_evi_normals(veg0)
        events_mean_in, _, _, _, _ = calc_evi_event_normals(veg0, events_list)

        # steal from Daymet profile
        h = rio.open(os.path.join(path_intrim, 'Daymet', extent, f'tmax_{fid}_2001.tif'))
        profile = dict(h.profile)
        profile['width'] = seasonal_mean.shape[2]
        profile['height'] = seasonal_mean.shape[1]
        profile['count'] = seasonal_mean.shape[0] + events_mean_in['daymet'].shape[0] + events_mean_in['topowx'].shape[0] + events_mean_in['yyz'].shape[0]
        h.close()

        ds = rio.open(os.path.join(path_out, 'veg', extent, f'{name}_{fid}_mask.tif'), 'w', **profile)
        for i, season in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
            ds.write(seasonal_mean[i, :, :].values, i + 1)
            ds.set_band_description(i + 1, f'seasonal_mean_{season}')

        layer = 5
        for use in ['daymet', 'topowx', 'yyz']:
            for j in range(events_mean_in[use].shape[0]):
                ds.write(events_mean_in[use][j, :, :].values, layer)
                end_of_event = datetime.strftime(events_mean_in[use]['time'].to_index()[j], '%Y%m%d')
                ds.set_band_description(layer, f'event_mean_{end_of_event}')
                layer += 1

        ds.close()
