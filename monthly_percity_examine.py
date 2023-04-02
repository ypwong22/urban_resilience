""" Preliminary examination of the monthly times series of heat waves & hot-dry events.
    Not necessarily consistent with the "official" analysis scripts. 
"""
import pandas as pd
import xarray as xr
import os
import numpy as np
from utils.analysis import *
from utils.constants import *
from utils.paths import *
from utils.extremes import *
from scipy.stats import boxcox_normplot, boxcox
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as stats
from tqdm import tqdm


def draw_extremes(ax, vector, is_extreme, time):
    ax.plot(time, vector, '-ob')
    ax.plot(time[is_extreme], vector[is_extreme], 'or')
    return ax


def draw_dual_extremes(ax, vector, is_extreme_high, is_extreme_low, time):
    ax.plot(time, vector, '-ob')
    ax.plot(time[is_extreme_high], vector[is_extreme_high], 'or')
    ax.plot(time[is_extreme_low], vector[is_extreme_low], 'og')
    return ax


def draw_extreme_pair(ax, a_orig, b_orig, a_is_extreme, b_is_extreme_high = None, b_is_extreme_low = None):
    ax.plot(a_orig, b_orig, 'ob')
    ax.plot(a_orig[a_is_extreme], b_orig[a_is_extreme], 'o', color = '#2c7fb8')
    if b_is_extreme_high is not None:
        ax.plot(a_orig[a_is_extreme & b_is_extreme_high], b_orig[a_is_extreme & b_is_extreme_high], 'or')
    if b_is_extreme_low is not None:
        ax.plot(a_orig[a_is_extreme & b_is_extreme_low], b_orig[a_is_extreme & b_is_extreme_low], 'og')
    return ax

def get_veg_data(varname, fid, extent):
    """ open the vegetation file """   
    if varname in ['PMLv2_GPP', 'PMLv2_ET']:
        extent = 'tiff'
        dall = read_pml(varname, fid, True, extent)
    elif varname == 'BNU_LAI':
        dall = read_lai(fid, True, extent)
    elif varname == 'MOD09Q1G_EVI':
        dall = read_evi(fid, 'MOD09Q1G_EVI', extent)
    # also mask the pixels where impervious > thres or water >50%
    if varname == 'BNU_LAI':
        thres = 30
    else:
        thres = 80
    dall = mask_water(fid, mask_impervious(fid, dall, thres, extent, 'both'), extent, 'both')
    return dall

def get_monthly_variables(fid, extent, use, name):
    if use == 'daymet':
        tmax = read_daymet(fid, 'tmax', extent)
        #tmin = read_daymet(fid, 'tmin', extent)
    elif use == 'topowx':
        tmax, _ = read_topowx(fid, True, extent)
        # Remove Feb 29
        tmax = tmax[(tmax['time'].to_index().month != 2) | (tmax['time'].to_index().day != 29), :, :]
        #tmin = tmin[(tmin['time'].to_index().month != 2) | (tmin['time'].to_index().day != 29), :, :]
    elif use == 'yyz': 
        tmax, _ = read_yyz(fid, True, extent)
    else:
        raise 'Not implemented'
    with xr.open_dataset(os.path.join(path_intrim, 'Daymet', extent, 'spi_' + str(fid) + '.nc')) as hr:
        if use != 'yyz':
            spi = hr['spi'][30:, :, :].copy(deep = True)
        else:
            spi = hr['spi'].copy(deep = True)

    tmax = mask_water(fid, mask_impervious(fid, tmax, 0.8, extent, 'both'), extent, 'both')
    spi  = mask_water(fid, mask_impervious(fid, spi , 0.8, extent, 'both'), extent, 'both')

    # (water & impervious mask already inside)
    veg0 = get_veg_data(name, fid, extent)

    # Convert to monthly average
    tmax = tmax.resample({'time': '1M'}).mean()
    #tmin = tmin.resample({'time': '1M'}).mean()
    spi = spi.resample({'time': '1M'}).last() # since SPI was calculated on 30 day rolling windows
    veg0 = veg0.resample({'time': '1M'}).max()

    # Remove the same climatology for the three temperature datasets
    tmax_clim = tmax[( tmax['time'].to_index().year >= 2003) & ( tmax['time'].to_index().year <= 2016),
                :, :].mean(dim = 'time')
    #tmin_clim = tmin[( tmin['time'].to_index().year >= 2003) & ( tmin['time'].to_index().year <= 2016),
    #            :, :].mean(dim = 'time')
    tmax = tmax - tmax_clim
    #tmin = tmin - tmin_clim

    # Slight inconsistencies in start & end year in order to maximize the usable data
    ymin = max(2001, tmax['time'].to_index().year[0]) # start year of EVI data
    ymax = min(2019, tmax['time'].to_index().year[-1]) # end year of EVI data
    tmax  =  tmax[( tmax['time'].to_index().year >= ymin) & ( tmax['time'].to_index().year <= ymax), :, :]
    #tmin  =  tmin[( tmin['time'].to_index().year >= ymin) & ( tmin['time'].to_index().year <= ymax), :, :]
    spi = spi[(spi['time'].to_index().year >= ymin) & (spi['time'].to_index().year <= ymax), :, :]
    veg0 = veg0[(veg0['time'].to_index().year >= ymin) & (veg0['time'].to_index().year <= ymax), :, :]

    # Convert meteorological factors to spatial average
    tmax  =  tmax.mean(dim = ['row','col'])
    #tmin  =  tmin.mean(dim = ['row','col'])
    spi = spi.mean(dim = ['row','col'])

    # Convert vegetation to urban/rural/per land cover average
    urban_mask = get_mask(fid, 'core', True, extent)
    rural_mask = get_mask(fid, 'rural', True, extent)
    veg = {'urban': veg0.where(urban_mask).mean(dim = ['row', 'col']),
           'rural': veg0.where(rural_mask).mean(dim = ['row', 'col'])}

    # Identify the monthly extremes
    is_heat_wave = np.full(len(tmax.values), False)
    is_hot_and_dry = np.full(len(tmax.values), False)
    is_pos_veg = {'urban': np.full(len(tmax.values), False), 'rural': np.full(len(tmax.values), False)}
    is_neg_veg = {'urban': np.full(len(tmax.values), False), 'rural': np.full(len(tmax.values), False)}
    for mon in range(12):
        is_heat_wave[mon::12] = identify_high_extremes(tmax.values[mon::12])

        temp = identify_low_extremes(spi.values[mon::12])
        is_hot_and_dry[mon::12] = is_heat_wave[mon::12] & temp

        for loc in ['urban','rural']: 
            is_pos_veg[loc][mon::12] = identify_high_extremes(veg[loc].values[mon::12])
            is_neg_veg[loc][mon::12] = identify_low_extremes(veg[loc].values[mon::12])

    ## In order to treat the magnitude difference between urban and rural vegetation, convert 
    ## the urban & rural EVI values to normal distribution using box-cox transformation, and then
    ## standardize the normal distribution to mean and std during the study period.
    ##veg_transformed = {'urban': np.empty(len(tmax.values)), 'rural': np.empty(len(tmax.values))}
    ##for mon in range(12):
        #veg_transformed['urban'][mon::12], _ = boxcox(veg['urban'].values[mon::12])
        #veg_transformed['rural'][mon::12], _ = boxcox(veg['rural'].values[mon::12])

    # In order to treat the magnitude different between urban and rural vegetation, 
    # convert the urban & rural EVI values to their percentiles, which proved to be more steady 
    # than the Box-Cox transformed values. 
    veg_transformed = {'urban': np.empty(len(tmax.values)), 'rural': np.empty(len(tmax.values))}
    for mon in range(12):
        veg_transformed['urban'][mon::12] = val_to_percentile(veg['urban'].values[mon::12])
        veg_transformed['rural'][mon::12] = val_to_percentile(veg['rural'].values[mon::12])

    return tmax, spi, veg, veg_transformed, is_heat_wave, is_hot_and_dry, is_pos_veg, is_neg_veg


if __name__ == '__main__':

    extent = 'tiff_3x'
    fid = REPLACE
    name = 'MOD09Q1G_EVI'

    for use in ['daymet', 'topowx', 'yyz']:
        print(use)

        tmax, spi, veg, veg_transformed, is_heat_wave, is_hot_and_dry, is_pos_veg, is_neg_veg = get_monthly_variables(fid, extent, use, name)

        # Plot the time series of the vegetation and meteorological extremes
        fig, axes = plt.subplots(4, 1, sharex = True, sharey = False, figsize = (8, 10))
        draw_extremes(axes.flat[0], tmax.values, is_heat_wave, tmax['time'].to_index())
        axes.flat[0].set_title('tmax & heat waves')
        draw_extremes(axes.flat[1], spi.values, is_hot_and_dry, spi['time'].to_index())
        axes.flat[1].set_title('spi & hot and dry')
        draw_dual_extremes(axes.flat[2], veg['urban'].values, is_pos_veg['urban'], is_neg_veg['urban'], veg['urban']['time'].to_index())
        axes.flat[2].set_title('urban evi extremes')
        draw_dual_extremes(axes.flat[3], veg['rural'].values, is_pos_veg['rural'], is_neg_veg['rural'], veg['rural']['time'].to_index())
        axes.flat[3].set_title('rural evi extremes')
        fig.savefig(os.path.join(path_out, 'monthly', extent, use, f'city_level_extremes_ts_{fid}.png'),
                    dpi = 600., bbox_inches = 'tight')
        plt.close(fig)


        # Plot the relationship between the extremes, separately for the urban and for the rural areas
        fig, axes = plt.subplots(2, 2, sharex = False, sharey = True, figsize = (10, 10))
        for i, loc in enumerate(['urban', 'rural']):
            ax = axes[0, i]
            draw_extreme_pair(ax, tmax.values, veg[loc].values, is_heat_wave, is_pos_veg[loc], is_neg_veg[loc])
            if i == 0:
                ax.legend(['All', 'Heat wave', 'Extreme veg high | Heat wave', 'Extreme veg low | Heat wave'],
                        loc = 'upper left') # , bbox_to_anchor = [0,-1.6], ncol = 1)
            ax.set_title(loc)
            ax.set_xlabel('Tmax ($^o$C)')
            ax = axes[1, i]
            draw_extreme_pair(ax, spi.values, veg[loc].values, is_hot_and_dry, is_pos_veg[loc], is_neg_veg[loc])
            if i == 0:
                ax.legend(['All', 'Hot and dry', 'Extreme veg high | Hot and dry', 'Extreme veg low | Hot and dry'],
                        loc = 'upper left') # , bbox_to_anchor = [1.2,-0.4], ncol = 1)
            ax.set_xlabel('SPI')
        fig.savefig(os.path.join(path_out, 'monthly', extent, use, f'city_level_extremes_scatter_{fid}.png'),
                    dpi = 600., bbox_inches = 'tight')
        plt.close(fig)


        # Plot the two kdes of the urban & rural area averaged EVI.
        fig, axes = plt.subplots(4, 3, sharex = False, sharey = False, figsize = (12,12))
        for mon in range(1, 13):
            ax = axes.flat[mon - 1]
            ax.set_title(f'Month {mon}')

            filt = tmax['time'].to_index().month == mon
            df_urban = pd.DataFrame({'EVI': veg['urban'].values[filt], 'urban': 'Yes'})
            df_rural = pd.DataFrame({'EVI': veg['rural'].values[filt], 'urban': 'No' })
            df = pd.concat([df_urban, df_rural], axis = 0).reset_index(drop = True)

            sns.histplot(data = df, x = 'EVI', hue = 'urban', bins = 15, kde = True, ax = ax)
            ax.set_xlabel('EVI')
        fig.savefig(os.path.join(path_out, 'monthly', extent, use, f'city_level_kde_{fid}.png'),
                    dpi = 600., bbox_inches = 'tight')
        plt.close(fig)


        # Plot the Box-Cox transform parameters of the urban & rural area averaged EVI.
        fig, axes = plt.subplots(4, 3, sharex = False, sharey = False, figsize = (12,12))
        for mon in range(1, 13):
            ax = axes.flat[mon - 1]
            ax.set_title(f'Month {mon}')

            filt = tmax['time'].to_index().month == mon
            lmbdas_urban, ppcc_urban = boxcox_normplot(veg['urban'].values[filt], la = -10, lb = 10)
            lmbdas_rural, ppcc_rural = boxcox_normplot(veg['rural'].values[filt], la = -10, lb = 10)

            ax.plot(lmbdas_urban, ppcc_urban, 'xb')
            ax.plot(lmbdas_rural, ppcc_rural, 'xr')

            ax.axvline(lmbdas_urban[np.argmax(ppcc_urban)], color = 'b', lw = 2)
            ax.axvline(lmbdas_rural[np.argmax(ppcc_rural)], color = 'r', lw = 2)

            ax.set_xlabel('$\lambda$')
            ax.set_ylabel('Prob Plot Corr. Coef.')
            ax.set_ylim([0.8,1])
            ax.legend(['urban','rural'])
        fig.savefig(os.path.join(path_out, 'monthly', extent, use, f'city_level_boxcox_{fid}.png'),
                    dpi = 600., bbox_inches = 'tight')
        plt.close(fig)


        # Plot the EVI values, transformed by box-cox, and converted to z-scores on the
        # standardized normal distribution during & outside heat waves, for each season.
        fig, axes = plt.subplots(2, 2, figsize = (12, 12))
        for i, season, month_list in zip(range(4), ['DJF', 'MAM', 'JJA', 'SON'], 
                                         [[12,1,2], [3,4,5], [6,7,8], [9,10,11]]):
            ax = axes.flat[i]
            ax.set_title(season)

            filt = tmax['time'].to_index().month.isin(month_list)

            df_urban = pd.DataFrame({'is_heat_wave': is_heat_wave[filt], 
                                     'veg_transformed': veg_transformed['urban'][filt],
                                     'urban': 'Yes'})
            df_rural = pd.DataFrame({'is_heat_wave': is_heat_wave[filt],
                                     'veg_transformed': veg_transformed['rural'][filt],
                                     'urban': 'No'})
            df = pd.concat([df_urban, df_rural], axis = 0)

            if df.shape[0] > 0:
                sns.boxplot(x = 'is_heat_wave', hue = 'urban', y = 'veg_transformed', ax = ax, data = df)
        fig.savefig(os.path.join(path_out, 'monthly', extent, use, f'city_level_boxplots_heat_wave_{fid}.png'),
                    dpi = 600., bbox_inches = 'tight')
        plt.close(fig)


        fig, axes = plt.subplots(2, 2, figsize = (12, 12))
        for i, season, month_list in zip(range(4), ['DJF', 'MAM', 'JJA', 'SON'], 
                                         [[12,1,2], [3,4,5], [6,7,8], [9,10,11]]):
            ax = axes.flat[i]
            ax.set_title(season)

            filt = tmax['time'].to_index().month.isin(month_list)

            df_urban = pd.DataFrame({'is_hot_dry': is_hot_and_dry[filt], 
                                     'veg_transformed': veg_transformed['urban'][filt], 
                                    'urban': 'Yes'})
            df_rural = pd.DataFrame({'is_hot_dry': is_hot_and_dry[filt],
                                     'veg_transformed': veg_transformed['rural'][filt],
                                     'urban': 'No'})
            df = pd.concat([df_urban, df_rural], axis = 0)

            if df.shape[0] > 0:
                sns.boxplot(x = 'is_hot_dry', hue = 'urban', y = 'veg_transformed', ax = ax, data = df)
        fig.savefig(os.path.join(path_out, 'monthly', extent, use, f'city_level_boxplots_hot_dry_{fid}.png'),
                    dpi = 600., bbox_inches = 'tight')
        plt.close(fig)
