import pandas as pd
import xarray as xr
import os
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import seaborn as sns

from monthly_percity_fit_per_pixel import *

from utils.analysis import *
from utils.constants import *
from utils.paths import *
from utils.extremes import *
from utils.get_monthly_data import *

extreme = 'heat_wave'
season = 'MAM'

#############################################################################
# Summarize the relationship between land cover types
#############################################################################
bins = (np.array([5, 25, 50, 75, 100]), np.array([5, 25, 50, 75, 100]))

for key, subset in modis_luc_city_groups.items():
    p = PixelData(extreme, season, (key, subset))
    luc = p.collect_luc() * 100. # convert fraction to percentages

    luc_agg = pd.DataFrame({
        'Total forest': luc[['Evergreen forest', 'Mixed forest', 'Deciduous forest']].sum(axis = 1),
        'Total short veg': luc[['Shrub', 'Grass']].sum(axis = 1), 
        'Crop': luc['Crop'],
        'Wetland': luc['Wetland'],
    })

    luc['impervious_frac'] = p.collect_impervious()
    luc_agg['impervious_frac'] = p.collect_impervious()

    g = sns.pairplot(luc, kind="hist", plot_kws = {'bins': bins})
    g.savefig(os.path.join(path_out, 'luc', 'plots', f'percity_{p.format_prefix()}_luc_{key}.png'), dpi = 600., bbox_inches = 'tight')
    g = sns.pairplot(luc_agg, kind="hist", plot_kws = {'bins': bins})
    g.savefig(os.path.join(path_out, 'luc', 'plots', f'percity_{p.format_prefix()}_luc_agg_{key}.png'), dpi = 600., bbox_inches = 'tight')
