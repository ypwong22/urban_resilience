import os
import pandas as pd
import numpy as np
import rasterio as rio
from utils.constants import *
from utils.paths import *
from utils.analysis import *


################################################################################
# Convert vp to VPD
################################################################################
for yr in range(1980, 2021):
    tvec = pd.date_range(f'{yr}-01-01', f'{yr}-12-31', freq = 'D')
    tvec = tvec.year * 10000 + tvec.month * 100 + tvec.day

    for fid in range(85):
        h = rio.open(os.path.join(path_intrim, 'Daymet', 'tiff_3x', f'tmax_{fid}_{yr}.tif'), 'r')
        tmax = h.read()
        h.close()

        h = rio.open(os.path.join(path_intrim, 'Daymet', 'tiff_3x', f'tmin_{fid}_{yr}.tif'), 'r')
        tmin = h.read()
        h.close()

        tmean = (tmax + tmin) / 2

        e_sat = 0.611 * np.exp( 17.27 * tmean / (tmean + 237.3))
        
        h = rio.open(os.path.join(path_intrim, 'Daymet', 'tiff_3x', f'vp_{fid}_{yr}.tif'), 'r')
        profile = dict(h.profile)
        vp = h.read()
        h.close()

        vpd = np.clip(e_sat * 1000 - vp, a_min = 0., a_max = None)

        ds = rio.open(os.path.join(path_intrim, 'Daymet', 'tiff_3x', f'vpd_{fid}_{yr}.tif'), 'w', **profile)
        for i in range(vpd.shape[0]):
            ds.write(vpd[i, :, :], i + 1)
            ds.set_band_description(i + 1, str(tvec[i]))
        ds.close()
