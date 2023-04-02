""" 
Subset the extracted vegetation response using land cover (or consider all land cover types),
  and season of the year.

 Exclude the following types of data points
 (2) Pixel-events when the EVI at the start or end of the event is < 0.1, or the pixel's EVI
     lack interannual variability (0.01)

     Pixel-events outliers:
       BNU_LAI: The maximum LAI in the post-event time series is > 8 or the minimum is < -8.
                This is based on visual inspection of the outliers, which exists in the daymet 
                and yyz extracted outcomes.
       EVI    : The same threshold = 8 can be used. 
"""
import pandas as pd
import os
import numpy as np
from utils.paths import *
from utils.constants import *
from utils.analysis import *
from utils.plotting import *
import itertools as it
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from summary_veg_by_luc import stage_data
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


name = 'REPLACE1' # 'MOD09Q1G_EVI', 'BNU_LAI'
opt = 'REPLACE2' # 'percity', 'pergrid'
extent = 'REPLACE3' # 'tiff', 'tiff_3x'
use = 'REPLACE4' # 'topowx', 'daymet', 'yyz'


if name == 'BNU_LAI':
    outlier = 8 # this threshold is determined by visually examining the response time series plots
else:
    outlier = 8
M = 80
N = 546
lu_names = list(modis_luc_agg.keys())
fid_list = range(85)


if opt == 'percity':
    from extract_post_extreme_veg_percity import get_extreme_data
else:
    #del get_extreme_data
    from extract_post_extreme_veg_pergrid import get_extreme_data


for ex, fid in it.product(['hw'], fid_list): # 'comp_hw_dr'
    print(ex, fid)

    if name == 'MOD09Q1G_EVI':
        veg_data, _, _, _ = stage_data(fid, name, 80, extent)
    else:
        # still need to use EVI, but threshold = 30
        veg_data, _, _, _ = stage_data(fid, 'MOD09Q1G_EVI', 30, extent)
    veg_std = veg_data.std(axis = 0)

    if opt == 'percity':
        extremes = get_extreme_data(ex, fid, extent, use)
    else:
        extremes = get_extreme_data(name, ex, fid, extent, use)

    extremes = extremes.loc[(extremes['start'] >= veg_data.index[0 ]) & \
                            (extremes['end'  ] <= veg_data.index[-1]), :]

    if opt == 'percity':
        extremes_indicator = pd.MultiIndex.from_tuples([(i,j,k,m) for i,j in \
            zip(veg_data.columns.get_level_values('row'),
                veg_data.columns.get_level_values('col')) for k,m in \
            zip(extremes['start'], extremes['end'])],
            names = ['row','col','start','end']).to_frame().reset_index(drop = True)
    else:
        extremes_indicator = extremes[['row','col','start','end']]

    # (i) check if the EVI/LAI is < 0.1
    extremes_indicator['VegTooSmall'] = False

    for i, row in tqdm(extremes_indicator.iterrows()):
        if not (row['row'], row['col']) in veg_data.columns:
            extremes_indicator.loc[i, 'VegTooSmall'] = True
            continue

        if veg_std.loc[(row['row'], row['col'])] < 0.01:
            extremes_indicator.loc[i, 'VegTooSmall'] = True
            continue

        veg_row = veg_data.loc[:, (row['row'], row['col'])]
        if len(np.where(veg_row.index < row['start'])[0]) == 0:
            continue
        if len(np.where(veg_row.index > row['end'])[0]) == 0:
            continue
    
        # if the heat wave is outside the growing season (start & end < 0.1), also skip
        which  = np.where(veg_row.index < row['start'])[0][-1]
        which2 = np.where(veg_row.index > row['end'  ])[0][ 0]
        if (veg_row.iloc[which] < 0.1) and (veg_row.iloc[which2] < 0.1):
            extremes_indicator.loc[i, 'VegTooSmall'] = True

    # (ii) check if the EVI & LAI has outliers
    extremes_indicator = extremes_indicator.set_index(['row','col','start','end'])
    extremes_indicator['VegTooLarge'] = False

    post_data = pd.read_csv(os.path.join(path_out, 'veg_response', opt, extent, use,
                                         f'{ex}_{name}_{M}_{N}_{fid}.csv'),
                            index_col = [0, 1, 2, 3], parse_dates = True \
                            ).drop(['succeded','preceded'], axis = 1)
    # interpolate to day 0 value, and subtract
    def _interp(*args):
        i = args[0][0]
        row = args[0][1]
        temp_row = row.dropna()
        x   = [int(xx.replace('day ','')) for xx in temp_row.index]
        y   = temp_row.values
        benchmark = interp1d(x,y, fill_value='extrapolate')(0)
        return (i, (row - benchmark).abs().max())
    result = process_map(_interp, post_data.iterrows())
    result = pd.Series([rr[1] for rr in result], index = pd.MultiIndex.from_tuples([rr[0] for rr in result]))
    too_large = result.index[result > outlier]
    extremes_indicator.loc[too_large.intersection(extremes_indicator.index), 'VegTooLarge'] = True

    # (iii) check if the duration of the extreme event is longer than 8 days
    extremes_indicator.reset_index(inplace = True)
    extremes_indicator['Duration > 8 days'] = \
        (extremes_indicator['end'] - extremes_indicator['start']).apply(lambda x: x.days) >= 8

    extremes_indicator = extremes_indicator.drop('end', axis = 1)
    extremes_indicator.to_csv(os.path.join(path_out, 'veg_response', opt, extent, use,
                                           'filter', f'per_event_{name}_{ex}_{fid}.csv'),
                              index = False)
