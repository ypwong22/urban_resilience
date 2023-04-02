import rasterio as rio
from utils.constants import *
from utils.paths import *
from utils.extremes import *
from utils.analysis import *
from utils.plotting import *
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_latlon(fid, extent):
    # Only using isel works.
    yy = 1980

    if extent == 'tiff':
        ds = rio.open(os.path.join(path_intrim, 'Daymet', extent,
                                   f'DAYMETv4_tmax_{fid}_{yy}.tif'))
    else:
        ds = rio.open(os.path.join(path_intrim, 'Daymet', extent, 
                                   f'tmax_{fid}_{yy}.tif'))

    row, col, x, y = np.array_split(np.array([(i, j, ) + ds.xy(i,j) \
                    for i in np.arange(ds.shape[0]) for j in np.arange(ds.shape[1])]),
                                     4, axis = 1)

    mask_core = get_mask(fid, 'core', True, extent)
    row_core, col_core = np.where(mask_core)
    rowcol_core = list(zip(row_core, col_core))
    mask_core = np.array([(r,c) in rowcol_core for r,c in zip(row,col)])

    mask_rural = get_mask(fid, 'rural', True, extent)
    row_rural, col_rural = np.where(mask_rural)
    rowcol_rural = list(zip(row_rural, col_rural))
    mask_rural = np.array([(r,c) in rowcol_rural for r,c in zip(row,col)])

    filter = mask_core | mask_rural
    row         = row[filter]
    y           = y  [filter]
    col         = col[filter]
    x           = x  [filter]
    mask_core   = mask_core[filter]
    mask_rural  = mask_rural[filter]

    proj_orig = CRS(wkt_daymet)
    proj_new = CRS('epsg:4326')
    transformer = Transformer.from_crs(proj_orig, proj_new)
    lats, lons = transformer.transform(x, y)
    coord = pd.DataFrame({'row': row.astype(int).reshape(-1),
                          'col': col.astype(int).reshape(-1),
                          'lat': lats.reshape(-1), 'lon': lons.reshape(-1),
                          'x': x.reshape(-1), 'y': y.reshape(-1),
                          'core': mask_core.reshape(-1),
                          'rural': mask_rural.reshape(-1)},
                         index = range(len(lats)))
    return coord


extent = 'tiff_3x'
for fid in range(85):
    coord = get_latlon(fid, extent)
    coord.to_csv(os.path.join(path_intrim, 'Daymet', 'coords', f'{extent}_{fid}.csv'),
                 index = False)

    if fid in [28,29]:
        fig, ax = plt.subplots(subplot_kw = {'projection': crs_daymet})
        cf1 = ax.scatter(coord['x'], coord['y'], c = coord['core'], 
                            s = 10, lw = 0., marker = 's',
                            transform = crs_daymet)
        add_core_boundary(ax, fid, 'k')
        plt.colorbar(cf1, ax = ax, shrink = 0.5)
        fig.savefig(os.path.join(path_intrim, 'Daymet', 'coords',
                                 f'check_{extent}_{fid}.png'))


# overlapping pixels are not a problem because the tiff mask of cities already accounted for it
## fid = 22
## coords = pd.read_csv(os.path.join(path_intrim, 'Daymet', 'coords', f'{extent}_{fid}.csv'))
## coords['x'] = coords['x'].astype(int).astype(str)
## coords['y'] = coords['y'].astype(int).astype(str)
## #coords_coords = coords.loc[coords['rural'], :].set_index(['x','y'])
## coords_coords = coords.set_index(['x','y'])
## 
## 
## fid_other = 14
## coords_other = pd.read_csv(os.path.join(path_intrim, 'Daymet', 'coords', f'{extent}_{fid_other}.csv'))
## coords_other['x'] = coords_other['x'].astype(int).apply(str)
## coords_other['y'] = coords_other['y'].astype(int).apply(str)
## #coords_other_coords = coords_other.loc[coords_other['core'], :].set_index(['x','y'])
## coords_other_coords = coords_other.set_index(['x','y'])
## 
## overlap = coords_coords.index.intersection(coords_other_coords.index)
