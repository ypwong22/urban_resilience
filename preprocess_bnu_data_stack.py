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


modis_wkt = 'PROJCS["MODIS Sinusoidal",GEOGCS["Custom",DATUM["Custom",SPHEROID["Custom",6371007.181,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Sinusoidal"],PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],PARAMETER["central_meridian",0.0],PARAMETER["semi_major",6371007.181],PARAMETER["semi_minor",6371007.181],UNIT["m",1.0]]'
src_srs = osr.SpatialReference()
src_srs.ImportFromWkt(modis_wkt)

yd_list = sorted(os.listdir(os.path.join(path_data, 'Vegetation', 'LAI', 'MODIS_BNU')))
yd_list = np.unique([i.split('.')[1].replace('A','') for i in yd_list if 'hdf' in i])

extent = 'tiff_3x'


for fid in range(85):
    print(fid)
    for yr in range(2001, 2020, 5):
        outvrt = os.path.join(path_data, 'Vegetation', 'LAI', 'MODIS_BNU', 'tempdir', extent,
                              'vsimem', f'stacked_{fid}_{yr}.vrt')
        outtif = os.path.join(path_data, 'Vegetation', 'LAI', 'MODIS_BNU', 'processed',
                              extent, f'reproj_fid{fid}_{yr}-{min(yr+4,2019)}.tif')
        tifs = sorted(glob(os.path.join(path_data, 'Vegetation', 'LAI', 'MODIS_BNU', 'tempdir', 
                                        extent, f'reproj_{fid}_{yr}*.tif'))) + \
               sorted(glob(os.path.join(path_data, 'Vegetation', 'LAI', 'MODIS_BNU', 'tempdir', 
                                        extent, f'reproj_{fid}_{yr+1}*.tif'))) + \
               sorted(glob(os.path.join(path_data, 'Vegetation', 'LAI', 'MODIS_BNU', 'tempdir', 
                                        extent, f'reproj_{fid}_{yr+2}*.tif'))) + \
               sorted(glob(os.path.join(path_data, 'Vegetation', 'LAI', 'MODIS_BNU', 'tempdir', 
                                        extent, f'reproj_{fid}_{yr+3}*.tif'))) + \
               sorted(glob(os.path.join(path_data, 'Vegetation', 'LAI', 'MODIS_BNU', 'tempdir', 
                                        extent, f'reproj_{fid}_{yr+4}*.tif')))

        outds = gdal.BuildVRT(outvrt, tifs, separate=True)
        outds = gdal.Translate(outtif, outds, creationOptions = ["TILED=YES"])


#
for fid in range(85):
    values = read_lai(fid, False, extent)

    fig, axes = plt.subplots(2, 1, figsize = (6, 12))

    ax = axes[0]
    ax.plot(np.nanmean(np.nanmean(values, axis = 2), axis = 1))
    ax.set_title('time series')

    ax = axes[1]
    ax.imshow(np.nanmean(values, axis = 0))
    ax.set_title('mean')

    fig.savefig(os.path.join(path_data, 'Vegetation', 'LAI', 'MODIS_BNU', 'processed',
                             extent, f'check_plot_{fid}.png'), dpi = 600.,
                bbox_inches = 'tight')
    plt.close(fig)
