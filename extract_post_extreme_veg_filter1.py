""" 
Subset the extracted vegetation response using land cover (or consider all land cover types),
  and season of the year.

 Exclude the following types of data points
 (1) Cities that do not have sufficient number of data points, for all land cover types,
     or just one land cover type.
     Pixels that fall into the urban core of another city are not a problem because the tif 
     urban mask already considered that problem.

The results suggest that the pixels dominated by the "Barren" land type cannot be compared
    on its own because there are too few data points.

Since this does not concern any meteorological data, percity/pergrid & daymet/topowx/yyz 
    distinctions are not needed.
"""
#%load_ext autoreload
#%autoreload 2
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


lu_names = list(modis_luc_agg.keys())
fid_list = range(85)
name = 'MOD09Q1G_EVI' # 'MOD09Q1G_EVI', 'BNU_LAI'
extent = 'tiff_3x'
thres = 0.1 # this threshold of vegetation existence is determined by manually examining
            # "summary_veg_by_luc.py"

            
# (1) summarize the number of pixels in each land cover type for urban and rural pixels
#     use the summary to derive a filter for spatial locations
redo = True
if redo:
    summary_of_luc = pd.DataFrame(np.nan, index = range(85),
                                  columns = pd.MultiIndex.from_product([['Urban core', 'Rural'], lu_names]))
    for fid in fid_list:
        if name == 'MOD09Q1G_EVI':
            veg_data, _, lu_data, xy_pos = stage_data(fid, name, 80, extent)
        else:
            # still need to use EVI, but threshold = 30
            veg_data, _, lu_data, xy_pos = stage_data(fid, 'MOD09Q1G_EVI', 30, extent)
    
        # Further filter the pixels to veg_data.max(axis = 0) >= thres
        # and the standard deviation > 0.01 to ensure vegetation exists. The 
        veg_data_mean = (veg_data.max(axis = 0) > thres) & (veg_data.std(axis = 0) > 0.01)
        lu_data = lu_data.loc[veg_data_mean]
        xy_pos  = xy_pos.loc[veg_data_mean, :]
    
        for j, pos in enumerate(['Urban core','Rural']):
            for l, lun in enumerate(lu_names):
                summary_of_luc.loc[fid, (pos,lun)] = sum((lu_data.isin(modis_luc_agg[lun])) & \
                                                         (xy_pos['rural'].astype(int) == j))
    summary_of_luc.to_csv(os.path.join(path_out, 'luc', extent, f'{name}_summary_of_luc.csv'))


summary_of_luc = pd.read_csv(os.path.join(path_out, 'luc', extent, f'{name}_summary_of_luc.csv'),
                             index_col = 0, header = [0,1])
store = pd.HDFStore(os.path.join(path_out, 'luc', extent, f'{name}_summary_of_luc_filter.h5'))
for fid in fid_list:
    if name == 'MOD09Q1G_EVI':
        veg_data, _, lu_data, xy_pos = stage_data(fid, name, 80, extent)
    else:
        # still need to use EVI, but threshold = 30
        veg_data, _, lu_data, xy_pos = stage_data(fid, 'MOD09Q1G_EVI', 30, extent)

    grid_list = pd.read_csv(os.path.join(path_intrim, 'Daymet', 'coords',
                                         extent + '_' + str(fid) + '.csv'), index_col = [0,1])

    luc_indicator = pd.DataFrame(True, index = grid_list.index,
                                 columns = ['Use_all', 'Use_Developed',
                                            'Use_Forest', 'Use_Shrub', 'Use_Grass',
                                            'Use_Crop', 'Use_Wetland'])

    # skip all the pixels where veg_data.max(axis = 0) < thres
    # or standard deviation < 0.01
    veg_data_mean = (veg_data.max(axis = 0) > thres) & (veg_data.std(axis = 0) > 0.01)
    veg_data_mean = veg_data_mean.index[~veg_data_mean.values]
    luc_indicator.loc[veg_data_mean, :] = False

    # skip all baren pixels
    baren = lu_data.index[lu_data == 31]
    luc_indicator.loc[baren, 'Use_all'] = False

    # for individual land cover types, skip all the irrelevant land cover types.
    # moreover, skip the cities where the number of pixels in
    # the urban core is < 0.1 * the number of pixels in the rural area, or
    # > 10 * the number of pixels in the rural aarea
    for l, lun in enumerate(luc_list):
        luc_indicator.loc[:, f'Use_{lun}'] = False

        temp = (summary_of_luc.loc[fid, ('Urban core'  , lun)] < \
                (0.1 * summary_of_luc.loc[fid, ('Rural', lun)] + 1e-10)) | \
               (summary_of_luc.loc[fid, ('Urban core'  , lun)] > 10 * \
                summary_of_luc.loc[fid, ('Rural'       , lun)])
        if ~temp:
            lu_pixel = lu_data.index[lu_data.isin(modis_luc_agg[lun])]
            luc_indicator.loc[lu_pixel, f'Use_{lun}'] = True

    luc_indicator['fid'] = fid
    luc_indicator = luc_indicator.reset_index().set_index(['fid','row','col'])

    store.append('df', luc_indicator, data_columns = list(range(luc_indicator.shape[1])))
store.close()
