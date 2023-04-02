import os
import pandas as pd
import numpy as np
from utils.analysis import *
from utils.constants import *
from utils.paths import *
from utils.plotting import *
import matplotlib.pyplot as plt
import rasterio as rio
from glob import glob
from time improt time


################################################################################################
# Calculate the difference in mean T_{max} and T_{min} between the urban core and the rural average
# runtime = ~1 hr
################################################################################################
extent = 'tiff_3x'
uhi = pd.DataFrame(np.nan, index = range(85),
                   columns = pd.MultiIndex.from_product([['daymet', 'topowx', 'yyz'], 
                                                         ['delta_tmax', 'delta_tmin']]))
for use in ['daymet', 'topowx', 'yyz']:
    for fid in range(85):
        print(use, fid)

        if use == 'daymet':
            hw = read_daymet(fid, 'tmax', extent)
            cw = read_daymet(fid, 'tmin', extent)
        elif use == 'topowx':
            hw, cw = read_topowx(fid, True, extent)
            filter = ~((hw['time'].to_index().month == 2) & (hw['time'].to_index().day == 29))
            hw = hw[filter, :, :]
            filter = ~((cw['time'].to_index().month == 2) & (cw['time'].to_index().day == 29))
            cw = cw[filter, :, :]
        elif use == 'yyz':
            hw, cw = read_yyz(fid, True, extent)
        else:
            raise 'Not implemented'
    
        mask_core = get_mask(fid, 'core', True, extent)
        mask_rural = get_mask(fid, 'rural', True, extent)
    
        delta_tmax = hw.where(mask_core).mean() - hw.where(mask_rural).mean()
        delta_tmin = cw.where(mask_core).mean() - cw.where(mask_rural).mean()

        uhi.loc[fid, (use, 'delta_tmax')] = delta_tmax
        uhi.loc[fid, (use, 'delta_tmin')] = delta_tmin
uhi.to_csv(os.path.join(path_out, 'uhi', extent, 'uhi_city.csv'))


#############################################################################################
# Plot the annual UHI against climatology
################################################################################################
extent = 'tiff_3x'
uhi = pd.read_csv(os.path.join(path_out, 'uhi', extent, 'uhi_city.csv'), index_col = 0, header = [0,1])

fig, axes = plt.subplots(3, 2, figsize = (12,12), sharex = True, sharey = True)
for i,use in enumerate(['daymet', 'topowx', 'yyz']):
    clim = pd.read_csv(os.path.join(path_out, 'clim', extent, f'{use}_clim.csv'), index_col = 0)

    for j,t,d in zip(range(2), ['delta_tmax', 'delta_tmin'], ['Daytime $\Delta$T', 'Nighttime $\Delta$T']):
        ax = axes[i,j]

        cf = ax.scatter(clim['tmean'].values, clim['prcp'].values * 365.25, 
                        c = uhi.loc[:, (use,t)].values, cmap = 'jet', vmin = -.5, vmax = 2)
        if i == 0:
            ax.set_title(d)
        if i == 2:
            ax.set_xlabel('Annual mean temperature of the urban area ($^o$C)')
        if j == 0:
            if i == 1:
                ax.set_ylabel('Annual total precipitation of the urban area (mm)\n' + use)
            else:
                ax.set_ylabel(use)
cax = fig.add_axes([0.95, 0.1, 0.01, 0.8])
plt.colorbar(cf, cax = cax, orientation = 'vertical', label = 'UHI ($^o$C)')
fig.savefig(os.path.join(path_out, 'uhi', extent, f'climUHI.png'), dpi = 600., bbox_inches = 'tight')
plt.close(fig)
