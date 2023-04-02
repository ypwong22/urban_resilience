import numpy as np
import rasterio as rio
from rasterio import Affine as A
from rasterio.warp import reproject, Resampling
from utils.constants import *
from utils.paths import *
import os


dst_transform = A(1000.0000000000008, 0.0, -4560750.0, 0.0, -1000.0000000000006, 4984500.0)
dst_crs = rio.crs.CRS.from_wkt('PROJCS["North_America_Lambert_Conformal_Conic_2SP",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101004,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["latitude_of_origin",42.5],PARAMETER["central_meridian",-100],PARAMETER["standard_parallel_1",25],PARAMETER["standard_parallel_2",60],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]')
dst_shape = [8075, 7814]
destination = -32768 * np.ones(dst_shape, np.int16)


for fid in range(85):
    ds = rio.open(os.path.join(path_data, 'Meteorological', 'Ta_USA_cities-selected', 
                               '2003', f'city{fid_to_yuyu[fid]}',
                               f'city{fid_to_yuyu[fid]}_TMIN2003103.tif'))
    src_transform = ds.profile['transform']
    src_crs = ds.profile['crs']
    source = ds.read()
    ds.close()

    dest_temp = -32768 * np.ones(dst_shape, np.int16)

    reproject(
        source,
        dest_temp,
        src_transform = src_transform,
        src_crs = src_crs,
        dst_transform = dst_transform,
        dst_crs = dst_crs,
        resampling = Resampling.nearest,
        src_nodata = -32768,
        dst_nodata = -32768)

    print(np.sum(dest_temp > -32768))

    destination[dest_temp > -32768] = fid


# Write it out to a file.
with rio.open(
        os.path.join(path_intrim, 'city_boundary_3x_merged.tif'),
        'w',
        driver='GTiff',
        width=dst_shape[1],
        height=dst_shape[0],
        count=1,
        dtype=np.int16,
        nodata=-32768,
        transform=dst_transform,
        crs=dst_crs) as dst:
    dst.write(destination, indexes=1)
