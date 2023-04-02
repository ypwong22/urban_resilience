""" 
Collect the fitted vegetation measures and predictors for the data points into HDF5 file

Time varying predictand and the predictors:
  - Resistance, resilience, and recovery
  - Intensity and duration of the extreme event
  - Average SPI during the extreme event (this is 30-day rolling average)

Static predictors:
  - Urban heat island intensity: day & night
  - Climatology of temperature and precipitation
  - symlog(Elevation + 1)
  - Broad land cover types (modis_luc_agg, exclude Barren pixels; note Developed already indicates impervious area)

Skip soil texture & soil water storage capacity (gNATSGO), because these are likely confounded with the
  land cover types, and are likely not accurate in urban areas. 
"""
import rasterio as rio
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from utils.paths import *
from utils.constants import *
from utils.plotting import *
from utils.analysis import *
import pickle


def rowcol_to_urlabel(fid, df_orig, extent):
    grid_list = pd.read_csv(os.path.join(path_intrim, 'Daymet', 'coords',
                                         extent + '_' + str(fid) + '.csv'), index_col = [0,1])
    if 'row' in df_orig.columns:
        df_new = df_orig.set_index(['row','col'])
    else:
        df_new = df_orig.copy(deep = True)

    df_new = df_new.loc[df_new.index.intersection(grid_list.index), :]
    df_new['Is_urban_core'] = (~grid_list.loc[df_new.index, 'rural']).astype(int)

    return df_new


def read_scalar(met, extent):
    # P & T climatology and the number of 1km2 pixels in developed areas
    data = pd.read_csv(os.path.join(path_out, 'clim', extent, f'{met}_clim.csv'),
                       index_col = 0)
    data.columns = ['Prcp', 'Tmean', 'Size']
    return data


def read_static(fid, met, extent):
    data  = {}

    # Elevation
    f     = rio.open(os.path.join(path_intrim, 'gee_single', 'DEM', extent,
                                  'DEM_{:02d}.tif'.format(fid)))
    nrows = f.read().shape[1]
    ncols = f.read().shape[2]
    ind   = np.where([i == 'elevation_mean' for i in f.descriptions])[0][0]
    elev  = f.read()[ind, :, :].reshape(-1)
    # symlog transformation
    data['Elevation'] = np.sign(elev) * np.log(np.abs(elev) + 1)
    f.close()

    # UHI level: do seasonal, separately average over the day and night
    for season in ['annual', 'DJF', 'MAM', 'JJA', 'SON']:
        f = rio.open(os.path.join(path_out, 'uhi', extent, f'{met}_{season}_{fid}.tif'))
        tmax = f.read()[0, :, :]
        tmin = f.read()[1, :, :]
        data[f'UHI_{season}_day'  ] = tmax.reshape(-1)
        data[f'UHI_{season}_night'] = tmax.reshape(-1)
    f.close()

    # NLCD land cover types
    nlcd = read_nlcd(fid, 'both', True, extent)
    modis_luc_list = ['Developed', 'Forest', 'Shrub', 'Grass', 'Crop', 'Wetland']
    for luc in modis_luc_list:
        data[luc] = np.where(np.isnan(nlcd[0, :, :]), np.nan,
                             nlcd.loc[modis_luc_agg[luc], :, :].sum(dim = 'band').values)
        data[luc] = data[luc].reshape(-1)

    #
    data = pd.DataFrame(data)
    col_list, row_list = np.meshgrid(np.arange(ncols), np.arange(nrows))
    data['row'] = row_list.reshape(-1)
    data['col'] = col_list.reshape(-1)

    # subset impervious area and add urban/rural flag
    if name == 'MOD09Q1G_EVI':
        thres = 80
    else:
        thres = 30
    data = subset_water(fid, subset_impervious(fid, subset_mask(fid, data, extent),
                                               thres, extent), extent)
    data = rowcol_to_urlabel(fid, data, extent)
    data = data.dropna(how = 'any')

    return data


def read_extreme_properties(ex, fid, opt, extent, met):
    # event intensity & duration & season
    if opt == 'pergrid':
        data             = pd.read_csv(os.path.join(path_out, 'extreme_events', extent, met,
                                                    f'{ex}_all_{fid}.csv'))
        data_start       = pd.DatetimeIndex([datetime.strptime('1980-01-01', '%Y-%m-%d') + \
                                             timedelta(days = i) \
                                             for i in data['start (days from 1980-01-01)']])
        data_date_middle = pd.DatetimeIndex([datetime.strptime('1980-01-01', '%Y-%m-%d') + \
                                             timedelta(days = int((i+j)/2)) \
                    for i,j in zip(data['start (days from 1980-01-01)'],
                                   data['end (days from 1980-01-01)'])])
        data['Season']   = data_date_middle.to_period('Q-NOV').quarter
        data['start']    = data_start
        data             = data.drop(['start (days from 1980-01-01)',
                                      'end (days from 1980-01-01)'], axis = 1)

        # subset impervious area and add urban/rural flag
        if name == 'MOD09Q1G_EVI':
            thres = 80
        else:
            thres = 30
        data             = subset_water(fid, subset_impervious(fid,
                                                               subset_mask(fid, data, extent),
                                                               thres, extent), extent)
        if ex == 'comp_hw_dr':
            data.columns = data.columns.map({'row': 'row', 'col': 'col', 'Year': 'Year',
                                             'Season': 'Season', 'start': 'start',
                                             'intensity (hw)': 'Intensity (hw)',
                                             'intensity (spi)': 'Intensity (spi)',
                                             'duration': 'Duration'})
        elif ex == 'comp_dr_dr':
            data.columns = data.columns.map({'row': 'row', 'col': 'col', 'Year': 'Year',
                                             'Season': 'Season', 'start': 'start',
                                             'intensity (vpd)': 'Intensity (vpd)',
                                             'intensity (spi)': 'Intensity (spi)',
                                             'duration': 'Duration'})
        else:
            data.columns = data.columns.map({'row': 'row', 'col': 'col', 'Year': 'Year',
                                             'Season': 'Season', 'start': 'start',
                                             'intensity': 'Intensity', 'duration': 'Duration'})
        data['fid'] = fid
        data = data.set_index(['fid', 'row', 'col', 'start'])

    else:
        data             = pd.read_csv(os.path.join(path_out, 'extreme_events', extent, met, 
                                                    f'all_all_{fid}.csv'))
        data             = data.loc[data['type'] == ex, :].drop('type', axis = 1)
        if ex == 'comp_hw_dr':
            data         = data.drop(['intensity', 'severity', 'intensity (vpd)',
                                      'severity (vpd)', 'severity (spi)', 'severity (hw)'], axis = 1)
        elif ex == 'comp_dr_dr':
            data         = data.drop(['intensity', 'severity', 'intensity (hw)',
                                      'severity (hw)', 'severity (spi)', 'severity (vpd)'], axis = 1)
        else:
            data         = data.drop(['severity', 'intensity (hw)', 'severity (hw)',
                                      'intensity (vpd)', 'severity (vpd)',
                                      'intensity (spi)', 'severity (spi)'], axis = 1)
        data_start       = pd.DatetimeIndex([datetime.strptime('1980-01-01', '%Y-%m-%d') + \
                                             timedelta(days = i) \
                                             for i in data['start (days from 1980-01-01)']])
        data_date_middle = pd.DatetimeIndex([datetime.strptime('1980-01-01', '%Y-%m-%d') + \
                                             timedelta(days = int((i+j)/2)) \
                                    for i,j in zip(data['start (days from 1980-01-01)'],
                                                   data['end (days from 1980-01-01)'])])
        data['Season']   = data_date_middle.to_period('Q-NOV').quarter
        data['start']    = data_start
        data             = data.drop(['start (days from 1980-01-01)',
                                      'end (days from 1980-01-01)'], axis = 1)
        if ex == 'comp_hw_dr':
            data.columns = data.columns.map({'Year': 'Year', 'Season': 'Season',
                                             'start': 'start',
                                             'intensity (hw)': 'Intensity (hw)',
                                             'intensity (spi)': 'Intensity (spi)',
                                             'duration': 'Duration'})
        elif ex == 'comp_dr_dr':
            data.columns = data.columns.map({'Year': 'Year', 'Season': 'Season',
                                             'start': 'start',
                                             'intensity (vpd)': 'Intensity (vpd)',
                                             'intensity (spi)': 'Intensity (spi)',
                                             'duration': 'Duration'})
        else:
            data.columns = data.columns.map({'Year': 'Year', 'Season': 'Season',
                                             'start': 'start',
                                             'intensity': 'Intensity',
                                             'duration': 'Duration'})

        # copy and past these for all the rows & columns
        grid_list = pd.read_csv(os.path.join(path_intrim, 'Daymet', 'coords',
                                             f'{extent}_{fid}.csv'))[['row','col']]

        data = pd.concat([data.loc[data.index.repeat(grid_list.shape[0])].reset_index(drop = True),
                          pd.concat([grid_list] * data.shape[0], axis = 0).reset_index(drop = True)],
                         axis = 1)
        data['fid'] = fid
        data = data.set_index(['fid','row','col','start'])

    if ex == 'hw':
        # add spi level during the heat wave event
        with pd.HDFStore(os.path.join(path_out, 'extreme_events', extent, met,
                                      f'spi_in_hw_{opt}_{extent}_{met}.h5')) as hf:
            df = hf.select('df').loc[[fid], :]
            data = pd.concat([data, df], axis = 1, join = 'inner')
    return data



def read_veg_measures(ex, var, m, n, s, fid, opt, extent, met):
    filename = os.path.join(path_out, 'veg_response', opt, extent, met, 'fitted', 
                            f'{ex}_{var}_{m}_{n}_{s}_{fid}_metrics.csv')
    if not os.path.exists(filename):
        return None

    data = pd.read_csv(filename)
    # remove the rows when resistance, resilience, and recovery are all NaN (these are where
    # vegetation lacks variability during the extreme)
    data = data.dropna(subset = ['resistance','resilience','recovery'], axis = 0, how = 'all')

    data['start'] = pd.DatetimeIndex(data['start'])
    data['fid']   = fid

    if name == 'MOD09Q1G_EVI':
        thres = 80
    else:
        thres = 30
    data = subset_water(fid, subset_impervious(fid, subset_mask(fid, data, extent),
                                               thres, extent), extent)
    data = data.set_index(['fid','row','col','start'])
    data['has_outlier'] = data['has_outlier'].astype(bool)

    return data


def assemble_Xy(fid, scalar, static, ext_prop, me):
    me_extend = me.copy().reset_index()

    for k in scalar.keys():
        me_extend[k] = scalar.loc[fid, k]

    for i, row in static.iterrows():
        temp = (me_extend['row'] == i[0]) & (me_extend['col'] == i[1])
        if sum(temp) > 0:
            me_extend.loc[temp, row.index] = row.values
    me_extend = me_extend.set_index(['fid', 'row', 'col', 'start'])

    me_extend = pd.concat([me_extend, ext_prop], axis = 1, join = 'inner')

    me_extend['Season'  ] = me_extend['Season'  ].astype(int)
    me_extend['Duration'] = me_extend['Duration'].astype(int)

    return me_extend


if __name__ == '__main__':
    name    = response_list[REPLACE1]
    extreme = extremes_list[REPLACE2]
    opt     = 'REPLACE3' # percity, pergrid
    extent  = 'REPLACE4' # tiff_3x, tiff
    met     = 'REPLACE5' # daymet, topowx, yyz
    M = 80
    N = 546
    S = 180

    filename = os.path.join(path_out, 'measures', opt, extent, met,
                            f'df_{extreme}_{name}_{M}_{N}_{S}.h5')
    if os.path.exists(filename):
        os.remove(filename)
    store    = pd.HDFStore(filename)

    scalar   = read_scalar(met, extent)
    for fid in range(85):
        from time import time
        start = time()

        static           = read_static(fid, met, extent)
        ext_prop         = read_extreme_properties(extreme, fid, opt, extent, met)
        me               = read_veg_measures(extreme, name, M, N, S, fid, opt, extent, met)
        if me is None:
            continue
        df               = assemble_Xy(fid, scalar, static, ext_prop, me)
        df               = filter_events(filter_points(df, fid, name, extent),
                                         extreme, fid, name, False, opt, extent, met)

        store.append('df',df,data_columns=list(range(df.shape[1])))

        end = time()
        print(f'Finished {fid} in {(end - start)/60:.6f} minutes')

    store.close()

    """
    store=pd.HDFStore(os.path.join(path_out, 'measures', f'df_{extreme}_{name}_{M}_{N}_{S}.h5'))
    df=store.select('df')
    df=df.drop_duplicates()
    store2=pd.HDFStore(os.path.join(path_out, 'measures', f'df_{extreme}_{name}_{M}_{N}_{S}_2.h5'))
    store2.append('df',df,data_columns=list(range(df.shape[1])))
    store2.close()
    store.close()
    """
