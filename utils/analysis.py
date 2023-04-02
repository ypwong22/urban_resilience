import rasterio as rio
from pyproj.transformer import Transformer, CRS
from .constants import *
from .paths import *
import xarray as xr
import os
import numpy as np
from datetime import datetime,timedelta
import pandas as pd
from glob import glob
import scipy.odr as odr
from scipy.stats import t


def latlon_to_xy(lats, lons, crs = wkt_daymet):
    inProj = CRS('epsg:4326')
    outProj = CRS(crs, preserve_units=True)
    transformer = Transformer.from_crs(inProj, outProj)
    x, y = transformer.transform(lons, lats)
    return x,y


def xy_to_latlon(xs, ys, crs = wkt_daymet):
    inProj = CRS(crs, preserve_units=True)
    outProj = CRS('epsg:4326')
    transformer = Transformer.from_crs(inProj, outProj)
    lons, lats = transformer.transform(xs,ys)
    return lons, lats


def add_grid_coords(fid, df, extent):
    """ Add the x-y coordinates to a data frame, which has row & col properties """
    grid_list = pd.read_csv(os.path.join(path_intrim, 'Daymet', 'coords',
                                         extent + '_' + str(fid) + '.csv'), index_col = [0,1])

    if df.index.names[0] is None:
        if isinstance(df.index, pd.MultiIndex):
            df_index_name = [f'level_{n}' for n in range(len(df.index.levels))]
        else:
            df_index_name_original = [None]
            df_index_name = 'index'
    else:
        df_index_name_original = df.index.names
        df_index_name = df.index.names

    df = df.reset_index().set_index(['row', 'col'])
    df = df.loc[df.index.intersection(grid_list.index), :]
    df['x'] = grid_list.loc[df.index, 'x'].values
    df['y'] = grid_list.loc[df.index, 'y'].values

    df = df.reset_index().set_index(df_index_name)
    df.index.names = df_index_name_original
    return  df


def add_urlabel(fid, df_orig, extent):
    """ Add the label if Is_urban_core to a data frame, which has row & col properties """
    grid_list = pd.read_csv(os.path.join(path_intrim, 'Daymet', 'coords', extent + '_' + str(fid) + '.csv'), index_col = [0,1])
    if 'row' in df_orig.columns:
        df_new = df_orig.set_index(['row','col'])
    else:
        df_new = df_orig.copy(deep = True)

    df_new = df_new.loc[df_new.index.intersection(grid_list.index), :]
    df_new['Is_urban_core'] = grid_list.loc[df_new.index, 'core'].astype(bool)

    return df_new


def add_urlabel_all(df, extent):
    """ Add the label if Is_urban_core to a data frame, which has row & col properties.
        Assumes df_orig has 'fid', 'row', 'col' properties.
    """
    if 'row' in df.index.names or 'col' in df.index.names or 'fid' in df.index.names:
        df_index_names = df.index.names
        df = df.reset_index()
    else:
        df_index_names = None
    df['location'] = np.full(df.shape[0], '                               ')
    for fid in np.unique(df['fid']):
        grid_list = pd.read_csv(os.path.join(path_intrim, 'Daymet', 'coords',
                                            extent + '_' + str(fid) + '.csv'), index_col = [0,1])
        df.loc[df['fid'] == fid, 'location'] = ['urban' if grid_list.loc[(i,j), 'core'] else 'rural' for i,j in zip(df.loc[df['fid'] == fid, 'row'], df.loc[df['fid'] == fid, 'col'])]
    if df_index_names is not None:
        df = df.set_index(df_index_names, drop = True)
    return df


def get_mask(fid, which = ['both', 'core', 'rural'], clip = [True, False], opt = ['tiff', 'tiff_3x']):
    if which == 'core':
        ds = rio.open(os.path.join(path_intrim, 'urban_mask', 'US_urbanCluster.tif'))
        mask = ds.read()[0, :, :] == fid
        ds.close()
    else:
        if opt == 'tiff':
            ds = rio.open(os.path.join(path_intrim, 'urban_mask', 'US_urbanCluster_merged.tif'))
            mask = ds.read()[0, :, :] == fid
            ds.close()
        else:
            ds = rio.open(os.path.join(path_intrim, 'urban_mask',
                                       'city_boundary_3x_merged.tif'))
            mask = ds.read()[0, :, :] == fid
            ds.close()

        if which == 'rural':
            ds = rio.open(os.path.join(path_intrim, 'urban_mask', 'US_urbanCluster.tif'))
            mask2 = ds.read()[0, :, :] == fid
            ds.close()
            mask = mask & (~mask2)

    #fig, ax = plt.subplots()
    #ax.imshow(mask)
    #fig.savefig(os.path.join(path_intrim, 'test.png'))

    if clip:
        # Clip mask to like GEE exports: one extra column to the right, one extra row to the bottom
        if opt == 'tiff':
            ds = rio.open(os.path.join(path_intrim,'urban_mask','US_urbanCluster_merged.tif'))
        else:
            ds = rio.open(os.path.join(path_intrim,'urban_mask','city_boundary_3x_merged.tif'))
        temp = ds.read()[0, :, :] == fid
        retain = np.where(temp)
        ds.close()
        mask = mask[min(retain[0]):(max(retain[0])+2), min(retain[1]):(max(retain[1]+2))]

    return mask


def read_daymet(fid, vv, opt = ['tiff', 'tiff_3x']):
    for yy in range(1980, 2021):
        if opt == 'tiff':
            ds = rio.open(os.path.join(path_intrim, 'Daymet', opt, 'DAYMETv4_' + str(vv) + \
                                       '_' + str(fid) + '_' + str(yy) + '.tif'))
        else:
            ds = rio.open(os.path.join(path_intrim, 'Daymet', opt, str(vv) + \
                                       '_' + str(fid) + '_' + str(yy) + '.tif'))
        bands = ds.read()

        if opt == 'tiff_3x':
            # quick fix: pad columns
            bands = np.concatenate([bands,
                                    np.full([bands.shape[0], 1, bands.shape[2]], np.nan)],
                                   axis = 1)
            bands = np.concatenate([bands,
                                    np.full([bands.shape[0], bands.shape[1], 1], np.nan)],
                                   axis = 2)

        #tvec = [datetime.strptime(ii.split('_')[1], '%Y%m%d') for ii in ds.descriptions]
        tvec = [datetime(yy,1,1) + timedelta(days = ii) \
                for ii in range((datetime(yy,12,31) - datetime(yy,1,1)).days + 1)]
        tvec = [ii for ii in tvec if not ((ii.month == 2) & (ii.day == 29))]
        ds.close()

        if yy == 1980:
            bands_all = bands
            tvec_all = tvec
        else:
            bands_all = np.concatenate([bands_all, bands], axis = 0)
            tvec_all = tvec_all + tvec

    mask = get_mask(fid, 'both', True, opt)
    mask = np.broadcast_to(mask, bands_all.shape)
    bands_all = np.where(mask, bands_all, np.nan)

    if vv == 'prcp':
        unit = 'mm day-1'
    elif vv == 'srad':
        unit = 'W m-2'
    elif vv == 'swe':
        unit = 'kg m-2'
    elif (vv == 'tmax') | (vv == 'tmin'):
        unit = 'degC'
    elif (vv == 'vp') | (vv == 'vpd'):
        unit = 'Pa'
    else:
        raise 'Unrecognized unit'

    bands = xr.DataArray(bands_all, dims = ['time', 'row', 'col'],
                         coords = {'time': tvec_all,
                                   'row': range(bands_all.shape[1]),
                                   'col': range(bands_all.shape[2])},
                         attrs = {'unit': unit})

    return bands


def read_topowx(fid, opt = ['tiff', 'tiff_3x']):
    flist = sorted(glob(os.path.join(path_data, 'Meteorological', 'TOPOWx', opt,
                                     f'reproj_fid{fid}_*.tif')))

    for i, filename in enumerate(flist):
        t0, t1 = filename.split('_')[-1].split('-')
        if '20191231' in t1:
            t1 = t1.replace('20191231', '20161231') # fix a naming error
        t0 = datetime.strptime(t0, '%Y%m%d')
        t1 = datetime.strptime(t1.replace('.tif',''), '%Y%m%d')
        tvec = [t0 + timedelta(days = i) for i in range((t1-t0).days+1)]
        # print(tvec[0], tvec[-1], len(tvec))

        f = rio.open(filename)
        values = f.read() * 100 # scale factor
        values[np.abs(values) >= 1e+37] = np.nan
        f.close()

        dims = ['time','row','col']
        coords = {'time': tvec, 'row': range(values.shape[1]),
                  'col': range(values.shape[2])}

        # print(i, filename, values[:len(tvec), :, :].shape)

        v1 = xr.DataArray(values[:len(tvec), :, :], dims = dims, coords = coords)
        v2 = xr.DataArray(values[len(tvec):, :, :], dims = dims, coords = coords)

        if i == 0:
            tmax = v1
            tmin = v2
        else:
            tmax = xr.concat([tmax, v1], dim = 'time')
            tmin = xr.concat([tmin, v2], dim = 'time')

    mask = get_mask(fid, 'both', True, opt)
    mask = np.broadcast_to(mask, tmax.shape)
    tmax = tmax.where(mask)
    tmin = tmin.where(mask)

    return tmax, tmin


def read_yyz(fid, opt = ['tiff', 'tiff_3x']):
    for vv in ['tmax', 'tmin']:
        flist = sorted(glob(os.path.join(path_intrim, 'Ta_USA_cities-selected', opt,
                                         f'{vv}_{fid}_*.tif')))

        for i, filename in enumerate(flist):
            yy = int(filename.split('_')[-1].replace('.tif',''))
            tvec = [datetime(yy,1,1) + timedelta(days = ii) \
                    for ii in range((datetime(yy,12,31) - datetime(yy,1,1)).days + 1)]
            tvec = [ii for ii in tvec if not ((ii.month == 2) & (ii.day == 29))]

            f = rio.open(filename)
            values = f.read() * 0.1 # scale factor
            values[np.abs(values) >= 100] = np.nan
            f.close()

            if opt == 'tiff_3x':
                # quick fix: pad columns
                values = np.concatenate([values, np.full([values.shape[0], 1,
                                                          values.shape[2]], np.nan)],
                                        axis = 1)
                values = np.concatenate([values, np.full([values.shape[0],
                                                          values.shape[1], 1], np.nan)],
                                        axis = 2)

            dims = ['time','row','col']
            coords = {'time': tvec, 'row': range(values.shape[1]),
                      'col': range(values.shape[2])}

            # print(i, filename, values[:len(tvec), :, :].shape)

            var = xr.DataArray(values, dims = dims, coords = coords)

            if i == 0:
                if vv == 'tmax':
                    tmax = var
                else:
                    tmin = var
            else:
                if vv == 'tmax':
                    tmax = xr.concat([tmax, var], dim = 'time')
                else:
                    tmin = xr.concat([tmin, var], dim = 'time')

    mask = get_mask(fid, 'both', True, opt)
    mask = np.broadcast_to(mask, tmax.shape)
    tmax = tmax.where(mask)
    tmin = tmin.where(mask)

    return tmax, tmin


def read_ssm(fid, mask = False):
    flist = sorted(glob(os.path.join(path_intrim, 'gee', 'SMAP10km',
                                     'SMAP10KM_ssm_' + str(fid) + '_*.tif')))
    for i, fn in enumerate(flist):
        ds = rio.open(fn)
        bands = ds.read()
        # every 3 days
        tvec = [datetime.strptime(x[-8:], '%Y%m%d') for x in ds.descriptions]
        ds.close()

        di = xr.DataArray(bands, dims = ['time', 'row', 'col'],
                          coords = {'time': tvec, 'row': np.arange(bands.shape[1]),
                                    'col': np.arange(bands.shape[2])})
        if i == 0:
            dall = di
        else:
            dall = xr.concat([dall, di], dim = 'time')

    if mask:
        mask = get_mask(fid, 'both', clip = True, opt = 'tiff_3x')
        mask = np.broadcast_to(mask, dall.shape)
        dall = dall.where(mask)

    return dall


def read_pml(varname, fid, mask = False, extent = 'tiff'):
    flist = sorted(glob(os.path.join(path_intrim, 'gee', 'PMLv2', extent,
                                     varname + '_' + str(fid) + '_*.tif')))
    for i, fn in enumerate(flist):
        ds = rio.open(fn)
        bands = ds.read()
        tvec = [datetime.strptime(x[-8:], '%Y%m%d') for x in ds.descriptions]
        ds.close()

        di = xr.DataArray(bands, dims = ['time', 'row', 'col'],
                          coords = {'time': tvec, 'row': np.arange(bands.shape[1]),
                                    'col': np.arange(bands.shape[2])})
        if i == 0:
            dall = di
        else:
            dall = xr.concat([dall, di], dim = 'time')
    if mask:
        mask = get_mask(fid, 'both', clip = True, opt = extent)
        mask = np.broadcast_to(mask, dall.shape)
        dall = dall.where(mask)
    return dall


def read_evi(fid, name, opt = ['tiff', 'tiff_3x']):
    flist = sorted(glob(os.path.join(path_intrim, 'vegetation', name, opt, f'reproj_fid{fid}_*.tif')))
    for i, fn in enumerate(flist):
        ds = rio.open(fn)
        bands = ds.read()
        bands = np.where(bands >= -0.2, bands, np.nan)

        y1 = int(fn.split('_')[-1].replace('.tif','').split('-')[0])
        y2 = int(fn.split('_')[-1].replace('.tif','').split('-')[1])
        # print(y1, y2)

        if name == 'MOD09Q1G_EVI':
            freq = '8D'
        else:
            freq = '1MS'

        tvec = []
        for yy in range(y1, y2+1):
            tvec += list(pd.date_range(str(yy)+'-01-01', str(yy)+'-12-31', freq = freq))

        di = xr.DataArray(bands, dims = ['time', 'row', 'col'],
                          coords = {'time': tvec,
                                    'row': np.arange(bands.shape[1]),
                                    'col': np.arange(bands.shape[2])})

        # downsample to monthly using maximum value composition
        if name == 'MOD09Q1G_EVI':
            di = di.resample(time = '1MS').max()

        if i == 0:
            dall = di
        else:
            dall = xr.concat([dall, di], dim = 'time')
        ds.close()

    mask = get_mask(fid, 'both', True, opt)
    mask = np.broadcast_to(mask, dall.shape)
    dall = dall.where(mask)

    return dall


def read_lai(fid, mask = False, opt = ['tiff', 'tiff_3x']):
    flist = sorted(glob(os.path.join(path_data, 'Vegetation', 'LAI', 'MODIS_BNU', 'processed',
                                     opt, f'reproj_fid{fid}_*.tif')))

    for i, fn in enumerate(flist):
        ds = rio.open(fn)
        bands = ds.read()

        y1 = int(fn.split('_')[-1].replace('.tif','').split('-')[0])
        y2 = int(fn.split('_')[-1].replace('.tif','').split('-')[1])
        # print(y1, y2)

        tvec = []
        for yy in range(y1, y2+1):
            tvec += list(pd.date_range(str(yy)+'-01-01', str(yy)+'-12-31', freq = '8D'))

        di = xr.DataArray(bands, dims = ['time', 'row', 'col'],
                          coords = {'time': tvec, 'row': np.arange(bands.shape[1]),
                                    'col': np.arange(bands.shape[2])})
        if i == 0:
            dall = di
        else:
            dall = xr.concat([dall, di], dim = 'time')
        ds.close()

    if mask:
        mask = get_mask(fid, 'both', True, opt)
        mask = np.broadcast_to(mask, dall.shape)
        dall = dall.where(mask)

    return dall


def read_gpp(fid, mask = False, opt = ['tiff', 'tiff_3x']):
    h = rio.open(os.path.join(path_intrim, 'slope_gpp', f'{opt}_{fid}.tif'))
    bands = h.read() * 0.001 # scale factor
    h.close()

    tvec = pd.date_range('2000-01-01', '2019-12-31')
    bands = xr.DataArray(bands, dims = ['time', 'row', 'col'], 
                         coords = {'time': tvec, 'row': np.arange(bands.shape[1]),
                                   'col': np.arange(bands.shape[2])})

    bands = bands.resample({'time': '1MS'}).mean()

    if mask:
        mask = get_mask(fid, 'both', True, opt)
        mask = np.broadcast_to(mask, bands.shape)
        bands = bands.where(mask)

    return bands


def read_et(fid, mask = False, opt = ['tiff', 'tiff_3x']):
    flist = sorted(glob(os.path.join(path_intrim, 'gee', 'PMLv2', opt, 'PMLv2_ET_' + str(fid) + '_*.tif')))
    for i, fn in enumerate(flist):
        ds = rio.open(fn)
        bands = ds.read()
        tvec = [datetime.strptime(x[-8:], '%Y%m%d') for x in ds.descriptions]
        ds.close()

        di = xr.DataArray(bands, dims = ['time', 'row', 'col'],
                          coords = {'time': tvec, 'row': np.arange(bands.shape[1]),
                                    'col': np.arange(bands.shape[2])})
        if i == 0:
            dall = di
        else:
            dall = xr.concat([dall, di], dim = 'time')

    if mask:
        mask = get_mask(fid, 'both', True, opt)
        mask = np.broadcast_to(mask, dall.shape)
        dall = dall.where(mask)

    return dall


def read_rh(fid, opt = ['tiff', 'tiff_3x']):
    """ Compute partial vapor pressure from tmean and vp. Method from

     Huang, J. (2018). A Simple 2Accurate Formula for Calculating Saturation Vapor Pressure of Water and Ice, Journal of Applied Meteorology and Climatology, 57(6), 1265-1272. Retrieved Mar 2, 2022, from https://journals.ametsoc.org/view/journals/apme/57/6/jamc-d-17-0334.1.xml 
    """
    tmean = (read_daymet(fid, 'tmin', True, opt) + \
             read_daymet(fid, 'tmax', True, opt).values) / 2
    ps = np.exp(34.494 - 4924.99/(tmean + 237.1)) / np.power(tmean + 105, 1.57)
    vp = read_daymet(fid, 'vp', True, opt)
    return vp / ps


def val_to_percentile(vector):
    """ Find the percentile corresponding to each value in an array """
    ind = np.argsort(vector)
    rank = np.empty(len(vector), dtype = int)
    for j, i in enumerate(ind):
        rank[i] = j
    percentile = (rank + 0.5) / len(rank)
    return percentile


def read_nlcd(fid, which = ['core','rural','both'], opt = ['tiff', 'tiff_3x']):
    f = rio.open(os.path.join(path_intrim, 'gee_single', 'NLCD', opt, f'NLCD_{fid:02d}.tif'))
    temp = f.read()
    f.close()

    dall = xr.DataArray(temp.reshape(8, -1, temp.shape[1], temp.shape[2]), dims = ['year','band','row','col'],
                        coords = {'year': [2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019],
                                  'band': [11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82, 90, 95],
                                  'row' : range(temp.shape[1]),
                                  'col' : range(temp.shape[2])})

    # remove the land cover types only in Alaska since they are irrelevant to CONUS
    dall = dall[:, ~np.isin(dall['band'].values, [51, 72, 73, 74]), :, :]

    # interpolate intermediate years
    dall = dall.interp({'year': range(2001, 2020)}, method = 'linear')

    # year 2020 = 2019
    dall = dall.interp({'year': range(2001, 2021)}, method = 'nearest', kwargs = {'fill_value': 'extrapolate'})

    mask = get_mask(fid, which, True, opt)
    dall = dall.where(mask)
    return dall


def agg_nlcd(da):
    """ Convert the original NLCD classes' percentages to broader ones """
    coords = dict(da.coords)
    coords['band'] = sorted(modis_luc_agg.keys())
    da_agg = xr.DataArray(np.nan, dims = da.dims, coords = coords)
    for i in coords['band']:
        detail = modis_luc_agg[i]
        if len(da.shape) == 3:
            da_agg.loc[i, :, :] = da.loc[detail, :, :].sum(axis = 0, skipna = False).values
        elif len(da.shape) == 4:
            da_agg.loc[:, i, :, :] = da.loc[:, detail, :, :].sum(axis = 1, skipna = False).values
        else:
            raise 'Not implemented'
    return da_agg


def dominant_nlcd(da):
    """ Identify the dominant land cover in each pixel. 
        The mixed forest type requires some special handling: 
        Neither deciduous nor evergreen species + 0.5* mixed forest are greater than 75% of total tree cover.
    """
    # the land cover with largest fraction in the pixel
    lcmax = da.idxmax(dim = 'band')

    # move mixed evergreen and deciduous pixels to the mixed forest type
    if len(da.shape) == 4:
        tot_tree = (da.loc[:, 2, :, :] + da.loc[:, 3, :, :] + da.loc[:, 4, :, :])
        to_mixed = ((da.loc[:, 2, :, :] + 0.5 * da.loc[:, 3, :, :]) < 0.75 * tot_tree) & \
                   ((da.loc[:, 4, :, :] + 0.5 * da.loc[:, 3, :, :]) < 0.75 * tot_tree) & \
                   ((lcmax == 2) | (lcmax == 4))
        lcmax    = lcmax.where(~to_mixed, 3)
        lcmax    = lcmax.where(np.isnan(lcmax.values[0,:,:]) | (da.max(dim = 'band') > 0.5), -1)
    elif len(da.shape) == 3:
        tot_tree = (da.loc[2, :, :] + da.loc[3, :, :] + da.loc[4, :, :])
        to_mixed = ((da.loc[2, :, :] + 0.5 * da.loc[3, :, :]) < 0.75 * tot_tree) & \
                   ((da.loc[4, :, :] + 0.5 * da.loc[3, :, :]) < 0.75 * tot_tree) & \
                   ((lcmax == 2) | (lcmax == 4))
        lcmax    = lcmax.where(~to_mixed, 3)
        lcmax    = lcmax.where(np.isnan(lcmax.values) | (da.max(dim = 'band') > 0.5), -1)

    return lcmax


def read_impervious(fid, which = ['core', 'rural', 'both'], opt = ['tiff', 'tiff_3x']):
    f = rio.open(os.path.join(path_intrim, 'gee_single', 'Impervious', opt, f'impervious_{fid:02d}.tif'))
    temp = f.read() / 100 # convert from percentage to fraction
    f.close()

    dall = xr.DataArray(temp, dims = ['year','row','col'],
                        coords = {'year': [2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019],
                                  'row' : range(temp.shape[1]),
                                  'col' : range(temp.shape[2])})

    # interpolate intermediate years
    dall = dall.interp({'year': range(2001, 2020)}, method = 'linear')

    # year 2020 = 2019
    dall = dall.interp({'year': range(2001, 2021)}, method = 'nearest', kwargs = {'fill_value': 'extrapolate'})

    mask = get_mask(fid, which, True, opt)
    dall = dall.where(mask)
    return dall


def read_elevation(fid, which = ['core','rural','both'], opt = ['tiff', 'tiff_3x']):
    f     = rio.open(os.path.join(path_intrim, 'gee_single', 'DEM', opt,
                                    'DEM_{:02d}.tif'.format(fid)))
    ind   = np.where([i == 'elevation_mean' for i in f.descriptions])[0][0]
    elev  = f.read()[ind, :, :]
    f.close()

    elev = xr.DataArray(elev, dims = ['row','col'],
                        coords = {'row' : range(elev.shape[0]),
                                  'col' : range(elev.shape[1])})

    mask = get_mask(fid, which, True, opt)
    elev = elev.where(mask)
    return elev


def mask_water(fid, da, opt = ['tiff', 'tiff_3x'], which = ['urban', 'rural', 'both']):
    """ Use NLCD to remove the pixels with > 40% water """
    nlcd = read_nlcd(fid, which, opt).mean(axis = 0) # average percentage land cover over area over 2001-2020
    mask = nlcd.loc[11, :, :].values < 0.4
    if isinstance(da, xr.DataArray):
        da = da.where(mask)
    else:
        mask = np.where(mask, 1, np.nan)
        mask = np.broadcast_to(mask, da.shape)
        da = da * mask
    return da


def mask_impervious(fid, da, thres, opt = ['tiff', 'tiff_3x'], which = ['urban', 'rural', 'both']):
    dall = read_impervious(fid, which, opt)
    mask = dall.mean(axis = 0) < thres # average impervious area over the time period
    if isinstance(da, xr.DataArray):
        da = da.where(mask)
    else:
        mask = np.where(mask, 1, np.nan)
        mask = np.broadcast_to(mask, da.shape)
        da = da * mask
    return da


def mask_crop(fid, da, opt = ['tiff', 'tiff_3x'], which = ['urban', 'rural', 'both']):
    nlcd = read_nlcd(fid, which, opt).mean(axis = 0) # average percentage land cover over area over 2001-2020
    mask = nlcd.loc[82, :, :].values < 0.5
    if isinstance(da, xr.DataArray):
        da = da.where(mask)
    else:
        mask = np.where(mask, 1, np.nan)
        mask = np.broadcast_to(mask, da.shape)
        da = da * mask
    return da


def mask_low_evi_seasonal(da_evi, fid, name, extent = ['tiff', 'tiff_3x'], season = None):
    """ Use the "create_low_evi_masks.py" output to remove the pixels
        that have seasonal mean EVI <= 0.05.
        This is to ensure that the extreme event fell on enough vegetation
        to have an impact on vegetation. """
    h = rio.open(os.path.join(path_out, 'veg', extent, f'{name}_{fid}_mask.tif'))
    mask = h.read()[:4, :, :]
    h.close()

    if season is None:
        for i, _ in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
            filt = da_evi['time'].to_index().to_period('Q-NOV').quarter == (i + 1)
            da_evi[filt, :, :] = da_evi[filt, :, :].where(mask[i, :, :] > 0.05).values
    else:
        if season == 'DJF':
            quarter = 1
        elif season == 'MAM':
            quarter = 2
        elif season == 'JJA':
            quarter = 3
        elif season == 'SON':
            quarter = 4
        if isinstance(da_evi, xr.DataArray):
            da_evi = da_evi.where(mask[quarter - 1, :, :] > 0.05)
        else:
            da_evi = np.where(mask[quarter - 1, :, :] > 0.05, da_evi, np.nan)
    return da_evi


def subset_mask(fid, df, opt = ['tiff', 'tiff_3x'], which = 'both'):
    mask = get_mask(fid, which, True, opt)
    rowlist, collist = np.where(mask)
    rowcol_list = pd.MultiIndex.from_tuples(list(zip(rowlist, collist)), names = ['row', 'col'])

    if df.index.names[0] is None:
        if isinstance(df.index, pd.MultiIndex):
            df_index_name = [f'level_{n}' for n in range(len(df.index.levels))]
        else:
            df_index_name_original = [None]
            df_index_name = 'index'
    else:
        df_index_name_original = df.index.names
        df_index_name = df.index.names

    df = df.reset_index().set_index(['row', 'col'])
    df = df.loc[df.index.intersection(rowcol_list), :]
    df = df.reset_index().set_index(df_index_name)
    df.index.names = df_index_name_original
    return df


def subset_water(fid, df, opt = ['tiff', 'tiff_3x']):
    mask = (read_nlcd(fid, 'both', opt).mean(axis = 0).loc[11,:,:].values < 0.5)
    rowlist, collist = np.where(mask)
    rowcol_list = pd.MultiIndex.from_tuples(list(zip(rowlist, collist)))
    format = 'row' in df.columns
    if format:
        df = df.set_index(['row','col'])
    df = df.loc[df.index.intersection(rowcol_list), :]
    if format:
        df = df.reset_index().rename({'level_0':'row','level_1':'col'}, axis = 1)
    return df


def subset_impervious(fid, df, thres, opt = ['tiff', 'tiff_3x']):
    """ Based on row & col info from df, subset the df to only impervious < thres% pixels """
    dall = read_impervious(fid, 'both', opt)
    mask = dall.mean(axis = 0) < thres
    rowlist, collist = np.where(mask)
    rowcol_list = pd.MultiIndex.from_tuples(list(zip(rowlist, collist)))
    format = 'row' in df.columns
    if format:
        df = df.set_index(['row','col'])
    df = df.loc[df.index.intersection(rowcol_list), :]
    if format:
        df = df.reset_index().rename({'level_0':'row','level_1':'col'}, axis = 1)
    return df


def normalize(ma_array):
    temp   = np.where  (ma_array.mask, np.nan, ma_array.data)
    n_mean = np.nanmean(temp, axis = 0, keepdims = True     )
    n_std  = np.nanstd (temp, axis = 0, keepdims = True     )

    #temp = (temp - n_mean) / np.where((n_std > 0.) | np.isnan(n_std), n_std, 
    #                                  np.nanmin(n_std[n_std > 0.]) * 1e-3)

    # when STD is zero at a point, set to NaN; since we don't want to deal with
    # static vegetation/meteorological factors anyway. 
    temp   = (temp - n_mean) / np.where(n_std > 0., n_std, np.nan)
    temp   = np.ma.masked_where(ma_array.mask, temp)
    return temp, n_mean, n_std


def olsTensor(Y, x):
    """ Repeated calculation of linear regression in the spatial dimensions.
    Parameters
    ----------
    Y : np.ma.array
        The variable of interest. The first dimension will be assumed to be
        time (replicate observations).
    x : np.array or np.ma.array
        The time variable of interest. If one-dimensional, will be propagated
        to the dimensionality of Y. If having the same dimensionality as Y,
        must be a masked array.
    Returns
    -------
    r : np.ma.array
        The trend. If x only has a time dimension, `r` is a scalar.
        Otherwise, `r` has the same dimensionality as x[1:].
    p : np.ma.array
        The two-sided p-values of the trend. If x only has a time 
        dimension, `p` is a scalar. Otherwise, `p` has the same 
        dimensionality as x[1:].
    """
    if type(Y) != np.ma.core.MaskedArray:
        raise TypeError('Y must be a masked array')
    if Y.shape[0] < 3:
        raise ValueError('At least three observations are needed')

    if (type(x) != np.ma.core.MaskedArray) and (type(x) != np.ndarray):
        raise TypeError('x must be either masked or ordinary numpy array')
    if (not np.allclose(x.shape, Y.shape)) and (len(x.shape) != 1):
        raise ValueError('x must be either 1-dimensional or has the same shape as Y')

    # homogenize the shape and mask of x and Y
    if type(Y.mask) == bool:
        Y.mask = np.full(Y.shape, Y.mask)
    if type(x) == np.ma.core.MaskedArray:
        if type(x.mask) == bool:
            x.mask = np.full(x.shape, x.mask)
    else:
        x = np.ma.array(x, mask = np.full(x.shape, False))

    orig_shape = Y.shape
    Y = Y.reshape(Y.shape[0], 1, int(np.prod(Y.shape[1:])))
    if len(x.shape) != 1:
        x = x.reshape(Y.shape)
    else:
        x = np.ma.array(np.broadcast_to(x.data.reshape(-1,1,1), Y.shape),
                        mask = np.broadcast_to(x.mask.reshape(-1,1,1), Y.shape))
    x = np.ma.array(x.data, mask = x.mask | Y.mask)
    Y = np.ma.array(Y, mask = x.mask)

    # normalize
    x, _, x_scale = normalize(x)
    Y, _, Y_scale = normalize(Y)

    # add constant term
    x = np.ma.concatenate([np.ma.array(np.ones(Y.shape), mask = Y.mask), x], axis = 1)

    # calculate the regression coefficients; treating the masked points as if zero.
    xx = np.where(x.mask == False, x.data, 0.)
    yy = np.where(Y.mask == False, Y.data, 0.)

    beta = np.einsum('ijk,jlk->ilk',
                     np.einsum('ijk,ljk->ilk',
                               np.linalg.pinv(np.einsum('ijk,ilk->jlk',xx,xx \
                                                       ).transpose(2,0,1)).transpose(1,2,0),
                               xx), yy)

    # calculate the p-value
    from scipy.stats import t
    if not isinstance(Y.mask, np.ndarray):
        dof = Y.shape[0] - 2
    else:
        dof = np.sum(Y.mask == False, axis = 0) - 2
    resid = yy - np.einsum('ijk,jlk->ilk', xx, beta)
    mse = np.sum(np.power(resid,2), axis=0) / dof

    # somehow, unable to apply np.ma.mean on x[:,[1],:]
    temp = x[:,[1],:]
    temp.data[temp.mask] = np.nan
    temp = temp.data
    std = np.nansum(np.power(temp - np.nanmean(temp, axis = 0, keepdims = True), 2), axis = 0)

    # somehow, using masked array here results in underflow error; had to use np.nan
    np.seterr(divide='ignore', invalid='ignore')
    beta = beta[1, :] # discard intercept
    tval = beta / np.sqrt(mse/std)
 
    np.seterr(divide='raise', invalid='raise')
    pval = 2 * t.sf(np.abs(tval), dof)

    # scale the beta
    beta = beta * Y_scale / x_scale

    # mask the data
    tval = np.ma.masked_invalid(tval)
    pval = np.ma.array(pval, mask = tval.mask)
    beta = np.ma.array(beta, mask = tval.mask)

    # restore shape
    if len(orig_shape) > 1:

        beta = beta.reshape(orig_shape[1:])
        pval = pval.reshape(orig_shape[1:])
    else:
        beta = float(beta.data)
        pval = float(pval.data)
    return beta, pval


def convert_to_rank(sortindex):
    rank = np.empty_like(sortindex)
    rank[sortindex] = np.arange(len(sortindex))
    return rank


def da_to_sr(da):
    """ Convert static xarray map to pandas series """
    col_list, row_list = np.meshgrid(np.arange(da.shape[1]), np.arange(da.shape[0]))
    da = pd.Series(da.values.reshape(-1),
                   index = pd.MultiIndex.from_arrays([row_list.reshape(-1),
                                                      col_list.reshape(-1)],
                                                     names = ['row','col']))
    return da


def da_to_df(da, time = None):
    """ Convert spatiotemporal series xarray to pandas data frame """
    if time is not None:
        tvec = time
    else:
        if len(da.shape) == 3:
            try:
                tvec = da['time'].to_index()
            except:
                raise ValueError('Must specify time if input array is 3d')
        else:
            tvec = time

    if isinstance(da, xr.DataArray):
        da = da.values

    if len(da.shape) == 3:
        col_list, row_list = np.meshgrid(np.arange(da.shape[2]), np.arange(da.shape[1]))
        da = pd.DataFrame(da.reshape(da.shape[0], -1), index = tvec,
                          columns = pd.MultiIndex.from_arrays([row_list.reshape(-1),
                                                               col_list.reshape(-1)],
                                                              names = ['row','col']))
        da = da.dropna(axis = 1, how = 'any')
    elif len(da.shape) == 2:
        col_list, row_list = np.meshgrid(np.arange(da.shape[1]), np.arange(da.shape[0]))
        if tvec is None:
            da = pd.Series(da.reshape(-1),
                           index = pd.MultiIndex.from_arrays([row_list.reshape(-1),
                                                              col_list.reshape(-1)],
                                                             names = ['row','col']))
            da = da.dropna()
        else:
            da = pd.DataFrame(da.reshape(1,-1), index = [tvec],
                              columns = pd.MultiIndex.from_arrays([row_list.reshape(-1),
                                                                 col_list.reshape(-1)],
                                                                names = ['row','col']))
            da = da.dropna(axis = 1, how = 'any')
    else:
        raise 'Not implemented'

    return da


def round_to_magnitude(vector, nsig):
    """ round to 'nsig' significant digits; works on vectors """
    def _round_to_digits(a, magnitude):
        return np.around(a / np.power(10, magnitude), nsig - 1) * np.power(10, magnitude)

    order_of_magnitude = np.floor(np.log10(np.abs(vector)))
    order_of_magnitude[np.abs(vector) < 1e-10] = 0
    func = np.vectorize(_round_to_digits)
    vector_round = func(vector, order_of_magnitude)
    return vector_round


def create_reasonable_bins(vec, nbins):
    vec_bins = np.percentile(vec, np.linspace(0, 100, nbins))

    # check how big difference there is; we need 2 more significant digits
    vec_bins_abs = np.abs(vec_bins)
    nsig = 2 + int(np.floor(np.log10( (np.max(vec_bins_abs) - np.min(vec_bins_abs))/np.max(vec_bins_abs) )))

    vec_bins = np.unique(round_to_magnitude(vec_bins, nsig))
    vec_bins[0] = vec_bins[0] - 1e-10
    vec_bins[-1] = vec_bins[-1] + 1e-10
    return vec_bins


def add_luc_label(fid, df, extent):    
    if 'row' in df.index.names or 'col' in df.index.names:
        df_index_names = df.index.names
        df = df.reset_index()
    else:
        df_index_names = None
        df_index = df.index

    luc = agg_nlcd(read_nlcd(fid, 'both', extent))
    luc = luc.mean('year').idxmax('band')
    luc = da_to_df(luc).astype(int).map(modis_luc_agg_names)

    df = df.set_index(['row', 'col'])
    df['luc'] = luc.loc[df.index]
    df = df.reset_index()

    if df_index_names is not None:
        df = df.set_index(df_index_names, drop = True)
    else:
        df.index = df_index
    return df


def add_luc_label_all(df, extent):
    if 'row' in df.index.names or 'col' in df.index.names or 'fid' in df.index.names:
        df_index_names = df.index.names
        df = df.reset_index()
    else:
        df_index_names = None

    df['luc'] = pd.Series('                                           ', index = df.index)
    for fid in df['fid'].unique():
        df.loc[df['fid'] == fid, 'luc'] = add_luc_label(fid, df.loc[df['fid'] == fid], extent)['luc']

    if df_index_names is not None:
        df = df.set_index(df_index_names, drop = True)
    return df


def orthogonal_regression(x, y, num_preds = 1):
    """ Traditional linear regression assumes x has no measurement errors. 
        This makes the slope always <1 no matter if I put urban or rural on the y-axis!
        Note x must use  row_stack, not column_stack.
    """
    if num_preds == 1:
        def f(B, x):
            """Linear function y = m*x + b"""
            # B is a vector of the parameters.
            # x is an array of the current x values.
            # x is in the same format as the x passed to Data or RealData.
            #
            # Return an array in the same format as y passed to Data or RealData.
            return B[0]*x + B[1]
    elif num_preds == 2:
        def f(B, x):
            """ y = a1*x1 + a2*x2 + b"""
            return B[0]*x[0] + B[1]*x[1] + B[2]
    elif num_preds == 3:
        def f(B, x):
            """ y = a1*x1 + a2*x2 + a3*x3 + b"""
            return B[0]*x[0] + B[1]*x[1] + B[2]*x[2] + B[3]
    else:
        raise 'Not implemented'

    linear = odr.Model(f)
    if num_preds == 1:
        x_std = x.std()
    else:
        x_std = x.std(axis = 1)
    mydata = odr.Data(x, y, wd = 1./np.power(x_std, 2), we = 1./np.power(y.std(),2))
    myodr = odr.ODR(mydata, linear, beta0 = [1. for n in range(num_preds+1)])
    myoutput = myodr.run()

    beta = myoutput.beta

    tinv = lambda p, df: abs(t.ppf(p/2, df))
    ts = tinv(0.05, len(x)-2)
    beta_ci = ts * myoutput.sd_beta

    beta_sig = np.abs(beta) > beta_ci

    return beta, beta_ci, beta_sig
