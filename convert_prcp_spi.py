import pandas as pd
import numpy as np
import os
from utils.constants import *
from utils.paths import *
from utils.extremes import *
from utils.analysis import *
from utils.spi import *
import multiprocessing as mp


vv = 'prcp'
extent = 'tiff_3x'

# range(85)
for fid in [REPLACE]:
    print(fid)

    ## temporarily skip unfinished fid's
    #if fid in [18, 22, 31, 51, 65, 61, 75]:
    #    continue

    values = read_daymet(fid, vv, extent)
    spi = xr.apply_ufunc(calc_spi, values.chunk({'time': -1}),
                         input_core_dims = [['time']],
                         output_core_dims = [['time2']], 
                         vectorize = True,
                         dask = 'parallelized',
                         dask_gufunc_kwargs = {'output_sizes': {'time2': 41 * 12}} \
    ).compute()
    spi = spi.transpose('time2', 'row', 'col')

    spi = spi.rename({'time2': 'time'})

    spi['time'] = pd.date_range('1980-01-01', '2020-12-31', freq = 'MS')

    spi.to_dataset(name = 'spi').to_netcdf(os.path.join(path_intrim, 'Daymet', extent,
                                                        'spi_' + str(fid) + '.nc'))

"""
for fid in range(85):
    hr = xr.open_dataset(os.path.join(path_intrim, 'Daymet', extent, 'temp', 'spi_' + str(fid) + '.nc'))
    spi = hr['spi'].copy(deep = True)
    hr.close()

    spi = spi.rename({'time2': 'time'})

    spi['time'] = pd.date_range('1980-01-01', '2020-12-31', freq = 'MS')

    spi.to_dataset(name = 'spi').to_netcdf(os.path.join(path_intrim, 'Daymet', extent,
                                                        'spi_' + str(fid) + '.nc'))
"""