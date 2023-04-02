""" Reproject the urban area extent to Daymet 3x grid """
import os
# os.environ['PROJDIR'] = '/gpfs/alpine/cli146/proj-shared'
import pandas as pd
import numpy as np
from glob import glob
import rasterio as rio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
from utils.paths import *
from utils.constants import *
from utils.plotting import *
from utils.analysis import *
# %matplotlib inline

ds = rio.open(os.path.join(path_intrim, 'urban_mask', 'city_boundary_3x_merged.tif'))
mask = ds.read()[0, :, :]
ds.close()

ds            = rio.open(os.path.join(path_data, 'Masks', 'annual_urbanMap',
                                      'sQ_urbanMap_global_stackTS.tif'))
src_transform = ds.profile['transform']
src_crs       = ds.profile['crs']
src_data      = ds.read()[0, :, :]
ds.close()

print(np.unique(src_data.reshape(-1)))
print(ds.xy(1,1), ds.xy(0,0))

for fid in range(85):
    print(fid)

    ds            = rio.open(os.path.join(path_intrim, 'gee_single', 'Impervious', 'tiff_3x',
                                          f'impervious_{fid:02d}.tif'))
    dst_crs       = ds.profile['crs']
    dst_transform = ds.profile['transform']
    dst_shape     = ds.read().shape[1:]
    ds.close()

    dst_temp      = np.empty(dst_shape)

    reproject(
        src_data,
        dst_temp,
        src_transform = src_transform,
        src_crs = src_crs,
        dst_transform = dst_transform,
        dst_crs = dst_crs,
        resampling = Resampling.nearest)

    dst_years     = list(range(1992, 2021))
    dst_n         = len(dst_years)
    dst_temp      = np.around(dst_temp).astype(int)
    dst_data      = np.zeros((dst_n,) + dst_shape)
    for i in range(1, dst_n+1):
        dst_data[i-1,:] = (dst_temp >= (dst_n+1-i)).astype(int)

    with rio.open(
            os.path.join(path_intrim, 'urban_mask', 'annualUrbanExtent_1992-2020',
                         f'{fid}.tif'),
            'w',
            driver ='GTiff',
            width  = dst_shape[1],
            height = dst_shape[0],
            count  = dst_n,
            dtype  = np.int16,
            transform=dst_transform,
            crs=dst_crs) as dst:
        for i in range(1,dst_n+1):
            dst.write(dst_data[i-1,:,:], i)
            dst.set_band_description(i, str(dst_years[i-1]))
