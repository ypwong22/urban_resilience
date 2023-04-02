""" Reproject the EVI data to Daymet 1km grid using nearest neighbor
1. Loop through the data to create mosaics that cover the CONUS
"""
import os
import numpy as np
from utils.paths import *
from utils.constants import *
from utils.plotting import *
from utils.analysis import *
from glob import glob
from osgeo import gdal,osr
from pyhdf.SD import SD, SDC
import itertools as it

modis_wkt = 'PROJCS["MODIS Sinusoidal",GEOGCS["Custom",DATUM["Custom",SPHEROID["Custom",6371007.181,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Sinusoidal"],PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],PARAMETER["central_meridian",0.0],PARAMETER["semi_major",6371007.181],PARAMETER["semi_minor",6371007.181],UNIT["m",1.0]]'
src_srs = osr.SpatialReference()
src_srs.ImportFromWkt(modis_wkt)

doy_list = sorted(os.listdir(os.path.join(path_data, 'Vegetation', 'EVI', 'MOD09Q1G_EVI', '2001')))
print(doy_list)

for year, doy in it.product(range(2001,2020), doy_list):
    # year = 2002 # print(year, doy)
    # doy = '225'

    folder = os.path.join(path_data, 'Vegetation', 'EVI', 'MOD09Q1G_EVI', str(year), doy)

    # Create a mosaic
    all_data = np.full([4800 * 3, 4800 * 6], -0.3) # Fill Value for Composed_EVI is -3000
    all_data = all_data.astype(np.float32)
    for i,h in enumerate(range(8,14)):
        for j,v in enumerate(range(4,7)):
            filename = glob(os.path.join(folder,
                            f'MOD09Q1G_EVI.A{year}{doy}.h{h:02d}v{v:02d}.006.*.hdf'))

            print(filename)

            if len(filename) == 0:
                continue

            filename = filename[0]
            ds = SD(filename, SDC.READ)

            data = ds.select('Composed_EVI')[:,:].astype(np.float32)
            # data[data < -2001] = np.nan
            attrs = ds.select('Composed_EVI').attributes(full = 1)
            scale_factor = attrs['scale_factor'][0]
            data = data * scale_factor
            #if (i == 0) & (j == 0):
            #    print(scale_factor)

            # geotransformation
            for k in ds.attributes()['StructMetadata.0'].split('\n'):
                if k.split('=')[0].replace('\t','') == 'UpperLeftPointMtrs':
                    UpperLeftPointMtrs = np.array(k.split('=')[1][1:-1].split(',')).astype(float)
                if k.split('=')[0].replace('\t','') == 'LowerRightMtrs':
                    LowerRightMtrs = np.array(k.split('=')[1][1:-1].split(',')).astype(float)
                    break
            if (i == 0) and (j == 0):
                left = UpperLeftPointMtrs[0]
                upper = UpperLeftPointMtrs[1]
                right = LowerRightMtrs[0]
                lower = LowerRightMtrs[1]
                #print(lower, upper, left, right)
                #print((upper - lower) / 4800, (right - left) / 4800)
            else:
                left = min(left, UpperLeftPointMtrs[0])
                upper = max(upper, UpperLeftPointMtrs[1])
                right = max(right, LowerRightMtrs[0])
                lower = min(lower, LowerRightMtrs[1])

            ds = None

            all_data[(j*4800):(j*4800+4800), (i*4800):(i*4800+4800)] = data

    print(lower, upper, left, right)

    # Correct wrong specifications in the origina hdf files
    if ((year == 2001) & (doy in ['057','105','129','289'])) | \
    ((year == 2002) & (doy in ['049','161','169','225','329'])) | \
    ((year == 2003) & (doy in ['049','145'])) | \
    ((year == 2012) & (doy in ['273'])):
        lower = 2223901.039332999847829341888427734375
        upper = 5559752.598333000205457210540771484375
        left = -11119505.19666700065135955810546875
        right = -4447802.07866699993610382080078125

    print((upper - lower) / 4800 / 3, (right - left) / 4800 / 6)

    # Save the mosaic to file
    newfile = os.path.join(path_data, 'Vegetation', 'EVI', 'tempdir', f'orig_{year}_{doy}.tif')

    driver = gdal.GetDriverByName('GTiff')
    temp_calc = driver.Create(newfile, all_data.shape[1], all_data.shape[0], 1, gdal.GDT_Float32)
    #temp_calc.SetSpatialRef(src_srs)
    temp_calc.SetProjection(src_srs.ExportToWkt())
    # take note the Y-spacing must be negative because the origin is in the upper left
    #temp_calc.SetGeoTransform(np.array([-11119505.196667, 231.65635826395862, 0,
    #                                    5559752.598333, 0, -231.65635826375006]))
    temp_calc.SetGeoTransform(np.array([left, (right-left)/4800/6, 0, upper, 0, -(upper-lower)/4800/3]))
    temp_b = temp_calc.GetRasterBand(1)
    temp_b.SetNoDataValue(-0.3)
    temp_b.WriteArray(all_data)
    temp_calc.FlushCache() # this step is important because otherwise the data may not show on disk
    temp_b = None
    driver = None