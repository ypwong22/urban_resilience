""" Extract daymet data using fid masks """
import os
import pandas as pd
import xarray as xr
import numpy as np
import rasterio as rio
from rasterio.transform import Affine
from utils.constants import *
from utils.paths import *
from utils.analysis import *
import itertools as it


ds = rio.open(os.path.join(path_intrim, 'urban_mask', 'city_boundary_3x_merged.tif'))
mask = ds.read()[0, :, :]
ds.close()

# for vv, yr in it.product(['prcp', 'tmax', 'tmin', 'srad', 'vp'], range(1980, 2021)):
vv = 'REPLACE1'
yr = REPLACE2

print(vv, yr)

hr = xr.open_dataset(os.path.join(path_data, 'Meteorological', 'Daymet', 'v4', 
                                  f'daymet_v4_daily_na_{vv}_{yr}.nc'))
tvec = hr['time'].to_index()
tvec = tvec.year * 10000 + tvec.month * 100 + tvec.day

data_collect = {}
col_start    = {}
row_start    = {}
for i in range(hr[vv].shape[0]):
    print(i)

    data = hr[vv][i, :, :].load()

    for fid in range(85):
        subset = data.where(mask == fid)
        subset_cols = np.where(~np.all(np.isnan(subset.values), axis = 0))[0]
        subset_cols = np.arange(subset_cols[0], subset_cols[-1]+1)
        subset_rows = np.where(~np.all(np.isnan(subset.values), axis = 1))[0]
        subset_rows = np.arange(subset_rows[0], subset_rows[-1]+1)
        subset = subset.values[:, subset_cols][subset_rows, :]

        if i == 0:
            data_collect[fid] = np.empty([hr[vv].shape[0],
                                          len(subset_rows), len(subset_cols)])

            col_start   [fid] = hr['x'].values[subset_cols][0]
            row_start   [fid] = hr['y'].values[subset_rows][0]

        data_collect[fid][i, :, :] = subset

for fid in range(85):
    profile = {'driver': 'GTiff', 'dtype': 'float64', 'nodata': -1e20,
               'width': data_collect[fid].shape[2],
               'height': data_collect[fid].shape[1],
               'count': data_collect[fid].shape[0],
               'crs': wkt_daymet,
               'transform': Affine(1000, 0, col_start[fid], 0, -1000, row_start[fid]),
               'tiled': False, 'compress': 'lzw', 'interleave': 'band'}

    ds = rio.open(os.path.join(path_intrim, 'Daymet', 'tiff_3x', f'{vv}_{fid}_{yr}.tif'), 'w',
                  **profile)
    for i in range(hr[vv].shape[0]):
        ds.write(data_collect[fid][i, :, :], i + 1)
        ds.set_band_description(i + 1, str(tvec[i]))
    ds.close()
hr.close()
