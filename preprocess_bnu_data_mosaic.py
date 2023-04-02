import os
import pandas as pd
import numpy as np
from utils.paths import *
from utils.constants import *
from utils.plotting import *
from utils.analysis import *
from glob import glob
from osgeo import gdal,osr
from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt

import itertools as it

modis_wkt = 'PROJCS["MODIS Sinusoidal",GEOGCS["Custom",DATUM["Custom",SPHEROID["Custom",6371007.181,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Sinusoidal"],PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],PARAMETER["central_meridian",0.0],PARAMETER["semi_major",6371007.181],PARAMETER["semi_minor",6371007.181],UNIT["m",1.0]]'
src_srs = osr.SpatialReference()
src_srs.ImportFromWkt(modis_wkt)

yd_list = sorted(os.listdir(os.path.join(path_data, 'Vegetation', 'LAI', 'MODIS_BNU')))
yd_list = np.unique([i.split('.')[1].replace('A','') for i in yd_list if 'hdf' in i])
# print(doy_list)
# print(len(doy_list)) = 920


for yd in yd_list:
    all_data = np.full([2400 * 3, 2400 * 6], -0.3) # Set the Fill Value to be -0.3
    all_data = all_data.astype(np.float32)

    count = 0
    for i,h in enumerate([7,8,9,10,11,12]):
        for j,v in enumerate([4,5,6]):
            filename = os.path.join(path_data, 'Vegetation', 'LAI', 'MODIS_BNU',
                                    f'MCD15A2H.A{yd}.h{h:02d}v{v:02d}.tsf.sat.hdf')
            if not os.path.exists(filename):
                continue

            ds = SD(filename, SDC.READ)
            data = ds.select('Lai_500m')[:,:].astype(np.float32)
            scale_factor = 0.1 # appears to be
            data = data * scale_factor

            # geotransformation
            for k in ds.attributes()['StructMetadata.0'].split('\n'):
                if k.split('=')[0].replace('\t','') == 'UpperLeftPointMtrs':
                    UpperLeftPointMtrs = np.array(k.split('=')[1][1:-1].split(',')).astype(float)
                if k.split('=')[0].replace('\t','') == 'LowerRightMtrs':
                    LowerRightMtrs = np.array(k.split('=')[1][1:-1].split(',')).astype(float)
                    break

            if count == 0:
                left = UpperLeftPointMtrs[0]
                upper = UpperLeftPointMtrs[1]
                right = LowerRightMtrs[0]
                lower = LowerRightMtrs[1]
                print(lower, upper, left, right)
                print((upper - lower) / 4800, (right - left) / 4800)
            else:
                left = min(left, UpperLeftPointMtrs[0])
                upper = max(upper, UpperLeftPointMtrs[1])
                right = max(right, LowerRightMtrs[0])
                lower = min(lower, LowerRightMtrs[1])

            ds = None

            all_data[(j*2400):(j*2400+2400), (i*2400):(i*2400+2400)] = data

            count += 1

        #print(lower, upper, left, right)
        #print((upper - lower) / 2400 / 3, (right - left) / 2400 / 6)

    # Save the mosaic to file
    newfile = os.path.join(path_data, 'Vegetation', 'LAI', 'MODIS_BNU', 'tempdir', f'orig_{yd}.tif')
    driver = gdal.GetDriverByName('GTiff')
    temp_calc = driver.Create(newfile, all_data.shape[1], all_data.shape[0], 1, gdal.GDT_Float32)
    #temp_calc.SetSpatialRef(src_srs)
    temp_calc.SetProjection(src_srs.ExportToWkt())
    # take note the Y-spacing must be negative because the origin is in the upper left
    #temp_calc.SetGeoTransform(np.array([-11119505.196667, 231.65635826395862, 0,
    #                                    5559752.598333, 0, -231.65635826375006]))
    temp_calc.SetGeoTransform(np.array([left, (right-left)/2400/6, 0, upper, 0, -(upper-lower)/2400/3]))
    temp_b = temp_calc.GetRasterBand(1)
    temp_b.SetNoDataValue(-0.3)
    temp_b.WriteArray(all_data)
    temp_calc.FlushCache() # this step is important because otherwise the data may not show on disk
    temp_b = None
    driver = None