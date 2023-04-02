"""
Reproject the EVI data to Daymet 1km grid using nearest neighbor. Reproject to Daymet grid and cut to small cities.
"""
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
import rasterio as rio

modis_wkt = 'PROJCS["unknown",GEOGCS["unknown",DATUM["unknown",SPHEROID["unknown",6371007.181,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Sinusoidal"],PARAMETER["longitude_of_center",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
src_srs = osr.SpatialReference()
src_srs.ImportFromWkt(modis_wkt)

#dst_srs = osr.SpatialReference()
#dst_srs.SetFromUserInput('EPSG:4326')
extent = 'tiff_3x' # 'tiff', 'tiff_3x'
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


for name in ['MOD13A3.061', 'MYD13A3.061']:
    if name == 'MOD13A3.061':
        ymin = 2001
    elif name == 'MYD13A3.061':
        ymin = 2003

    for year in range(2003, 2021):

        if (name == 'MYD13A3.061') or (year < 2018):
            continue

        flist = sorted(glob(os.path.join(path_data, 'Vegetation', 'EVI', name,
                                         f'{name}__1_km_monthly_EVI_doy{year}???_aid0001.tif')))

        for i, filename in enumerate(flist):
            f = gdal.Open(filename, gdal.GA_ReadOnly)
            src_srs = f.GetProjection()
            f = None
        
            for fid in range(85):
                # get the transform information
                f = gdal.Open(os.path.join(path_intrim, 'gee_single', 'NLCD', extent,
                                           f'NLCD_{fid:02d}.tif'), gdal.GA_ReadOnly)
                geoTransform = f.GetGeoTransform()
                minx = geoTransform[0]
                maxy = geoTransform[3]
                maxx = minx + geoTransform[1] * f.RasterXSize
                miny = maxy + geoTransform[5] * f.RasterYSize
                f = None
        
                print('\t', year, i, fid, minx, maxy, maxx, miny)
        
                # reproject
                newfile = os.path.join(path_data, 'Vegetation', 'EVI', 'tempdir', extent,
                                       f'temp_reproj_{fid}_{year}_{i+1:02d}.tif')
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
                newnewfile = newfile.replace('temp_reproj_', 'temp_temp_reproj_')
                f = gdal.Open(newfile)
                f = gdal.Translate(newnewfile, f, projWin = [minx, maxy, maxx, miny])
                f.FlushCache()
                f = None
        
                os.system(f'rm {newfile}')
        
                # reproject the quality check file
                qa_newfile = os.path.join(path_data, 'Vegetation', 'EVI', 'tempdir', extent,
                                          f'temp_reproj_qa_{fid}_{year}_{i+1:02d}.tif')
                gdal.Warp(
                    qa_newfile,
                    filename.replace('monthly_EVI', 'monthly_pixel_reliability'), 
                    format = 'GTiff',
                    xRes = 1000, yRes = 1000, targetAlignedPixels = True,
                    srcSRS = src_srs, dstSRS = dst_srs,
                    srcNodata = -0.3, dstNodata = -0.3, 
                    resampleAlg = gdal.GRA_NearestNeighbour
                )

                # cut to the same extent
                qa_newnewfile = qa_newfile.replace('temp_reproj_', 'temp_temp_reproj_')
                f = gdal.Open(qa_newfile)
                f = gdal.Translate(qa_newnewfile, f, projWin = [minx, maxy, maxx, miny])
                f.FlushCache()
                f = None

                os.system(f'rm {qa_newfile}')

                # remove the bad quality data
                h = rio.open(newnewfile)
                h2 = rio.open(qa_newnewfile)
        
                temp = h.read()
                temp2 = h2.read()
                temp = np.where((temp == -3000) | (temp2 != 0), np.nan, temp)
                temp = temp * 0.0001
        
                # save the updated data to file
                profile = dict(h.profile)
                profile['dtype'] = 'float64'
                ds = rio.open(os.path.join(path_data, 'Vegetation', 'EVI', 'tempdir', extent,
                                           f'{name}_reproj_{fid}_{year}_{i+1:02d}.tif'), 'w', **profile)
                ds.write(temp[0, :, :], 1)
                ds.close()
        
                h.close()
                h2.close()
        
                os.system(f'rm {newnewfile}')
                os.system(f'rm {qa_newnewfile}')
