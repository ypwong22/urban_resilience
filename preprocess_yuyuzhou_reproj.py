"""
Reproject Zhou's temperature data to Daymet 1km grid using nearest neighbor
"""
import os
import pandas as pd
import xarray as xr
import numpy as np
import rasterio as rio
from rasterio.warp import reproject, Resampling
from utils.constants import *
from utils.paths import *
from utils.analysis import *
import itertools as it
from glob import glob


extent = 'tiff_3x'

ds = rio.open(os.path.join(path_intrim, 'urban_mask', 'city_boundary_3x_merged.tif'))
mask = ds.read()[0, :, :]
ds.close()

for fid,vv in it.product([81], ['tmax', 'tmin']):
    city    = f'city{fid_to_yuyu[fid]}'

    ds      = rio.open(os.path.join(path_intrim, 'Daymet', extent, f'{vv}_{fid}_1980.tif'))
    profile = dict(ds.profile)
    ds.close()

    for yy in range(2003, 2021):
        flist      = sorted(glob(os.path.join(path_data, 'Meteorological',
                                              'Ta_USA_cities-selected',
                                              str(yy), city, f'{city}_{vv.upper()}*.tif')))

        data_bands = -32768 * np.ones([len(flist), profile['height'], profile['width']])

        for j,f in enumerate(flist):
            ds            = rio.open(f)
            src_transform = ds.profile['transform']
            src_crs       = ds.profile['crs']
            source        = ds.read()
            ds.close()

            dst           = -32768 * np.ones([profile['height'], profile['width']])

            reproject(
                source,
                dst,
                src_transform = src_transform,
                src_crs       = src_crs,
                dst_transform = profile['transform'],
                dst_crs       = profile['crs'],
                resampling    = Resampling.nearest,
                src_nodata    = -32768,
                dst_nodata    = -32768
            )

            data_bands[j, :, :] = dst

        # write it out to a file.
        with rio.open(
                os.path.join(path_intrim, 'Ta_USA_cities-selected', extent,
                             f'{vv}_{fid}_{yy}.tif'), 'w',
                driver    = 'GTiff',
                width     = data_bands.shape[2],
                height    = data_bands.shape[1],
                count     = data_bands.shape[0],
                dtype     = np.int16,
                nodata    = -32768,
                transform = profile['transform'],
                crs       = profile['crs']) as dst:
            dst.write(data_bands, indexes = range(1, data_bands.shape[0] + 1))
