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

extent = 'tiff_3x'
if extent == 'tiff':
    f = gdal.Open(os.path.join(path_intrim, 'urban_mask', 'US_urbanCluster_buffer.tif'),
                  gdal.GA_ReadOnly)
    dst_srs = f.GetProjection()
    f = None
else:
    f = gdal.Open(os.path.join(path_intrim, 'urban_mask', 'city_boundary_3x_merged.tif'),
                  gdal.GA_ReadOnly)
    dst_srs = f.GetProjection()
    f = None

# for yd in yd_list:
yd = yd_list[REPLACE]

filename = os.path.join(path_data, 'Vegetation', 'LAI', 'MODIS_BNU', 'tempdir', 
                        f'orig_{yd}.tif')

f = gdal.Open(filename, gdal.GA_ReadOnly)
src_srs = f.GetProjection()
f = None

for fid in range(85):
    f = gdal.Open(os.path.join(path_intrim, 'gee_single', 'NLCD', extent,
                               f'NLCD_{fid:02d}.tif'), gdal.GA_ReadOnly)
    geoTransform = f.GetGeoTransform()
    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * f.RasterXSize
    miny = maxy + geoTransform[5] * f.RasterYSize
    f = None

    print('\t', yd, fid, minx, maxy, maxx, miny)

    # reproject
    newfile = os.path.join(path_data, 'Vegetation', 'LAI', 'MODIS_BNU', 'tempdir',
                           extent, f'temp_reproj_{fid}_{yd}.tif')
    gdal.Warp(
        newfile,
        filename, 
        format = 'GTiff',
        xRes = 1000, yRes = 1000, targetAlignedPixels = True,
        srcSRS = src_srs, dstSRS = dst_srs,
        srcNodata = -0.3, dstNodata = -0.3, 
        resampleAlg = gdal.GRA_NearestNeighbour
    )

    # cut to the same extent
    newnewfile = newfile.replace('temp_reproj_', 'reproj_')
    f = gdal.Open(newfile)
    f = gdal.Translate(newnewfile, f, projWin = [minx, maxy, maxx, miny])
    f.FlushCache()
    f = None

    os.system(f'rm {newfile}')
