from operator import xor
from re import L
import pandas as pd
import os
import numpy as np
from utils.paths import *
from utils.constants import *
from utils.analysis import *
from utils.plotting import *
from utils.get_monthly_data import Filter
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap
from dateutil.relativedelta import relativedelta
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import rcParams
from datetime import date
from tqdm import tqdm
import itertools as it
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from scipy.interpolate import UnivariateSpline
import piecewise_regression
import statsmodels.api as sm
import multiprocessing as mp
from dateutil.relativedelta import relativedelta
import math


class ByPixel():
    """
    Seasonal mean & std, 
    mean & std during & 3 months after the heat wave, 
    pay attention to where these are smaller than 0.05 // 0.1 // 0.2.

    Plot these for each city separately. 
    """
    def __init__(self):
        pass


    def read_events(self, fid, name, extent):
        heat_wave = {}
        for use in ['daymet', 'topowx', 'yyz']:
            filename = os.path.join(path_out, 'extreme_events', f'percity_{extent}_{use}_{name}_90_85.h5')
            with pd.HDFStore(filename) as hf:
                heat_wave[use] = hf.select('heat_wave').loc[fid, :]
        return heat_wave


    def calc_evi_normals(self, veg):
        # veg = read_evi(fid, name, extent).resample(time = 'Q-NOV').mean()
        # mean_monthly = veg.groupby('time.quarter').mean()
        # std_monthly  = veg.groupby('time.quarter').std()
        veg = veg.resample(time = 'Q-NOV').mean()
        mean = veg.groupby('time.quarter').mean()
        std  = veg.groupby('time.quarter').std()
        return mean, std


    def calc_evi_event_normals(self, veg, events_list):
        mean_in   = {}
        std_in    = {}
        mean_post = {}
        std_post  = {}
        label = {}
        for use in ['daymet', 'topowx', 'yyz']:
            mean_in  [use] = []
            std_in   [use] = []
            mean_post[use] = []
            std_post [use] = []
            tdim           = []
            label    [use] = []
            for eid in range(events_list[use].shape[0]):
                start = events_list[use].index[eid][0]
                end   = events_list[use].index[eid][1]

                if (start < veg['time'].to_index()[0]) | ((end + relativedelta(months = 3)) > veg['time'].to_index()[-1]):
                    continue

                veg_temp = veg.loc[start:end, :, :]
                mean_in  [use].append(veg_temp.mean(axis = 0))
                std_in   [use].append(veg_temp.std (axis = 0))
                tdim.append(end) # ending month to facilitate extracting quarters
                label    [use].append(datetime.strftime(start, '%Y-%m') + f' + {veg_temp.shape[0]} months')

                veg_temp = veg.loc[(end + relativedelta(months = 1)):(end + relativedelta(months = 3)), :, :]
                mean_post[use].append(veg_temp.mean(axis = 0))
                std_post [use].append(veg_temp.std (axis = 0))

            mean_in  [use]   = xr.concat(mean_in  [use], dim = 'time')
            std_in   [use]   = xr.concat(std_in   [use], dim = 'time')
            mean_post[use]   = xr.concat(mean_post[use], dim = 'time')
            std_post [use]   = xr.concat(std_post [use], dim = 'time')

            mean_in  [use]['time'] = tdim
            std_in   [use]['time'] = tdim
            mean_post[use]['time'] = tdim
            std_post [use]['time'] = tdim

        return mean_in, std_in, mean_post, std_post, label


    def make_seasonal_plot(self, name, mean, std, fid, extent):
        city_size = mean.shape[0] * mean.shape[1]

        norm_mean = BoundaryNorm([0., 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 256, extend = 'both')
        norm_std = BoundaryNorm(np.arange(0., 0.11, 0.01), 256, extend = 'max')

        fig, axes = plt.subplots(2, 4, figsize = (16, 8), subplot_kw = {'projection': crs_daymet})
        for i, season in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
            ax = axes[0, i]
            cf = map_of_city(mean[i, :, :], fid, extent, ax, s = np.power(city_size, 0.3), cmap = 'Spectral', norm = norm_mean)
            plt.colorbar(cf, ax = ax, shrink = 0.5)
            ax.set_title(season)
            if i == 0:
                ax.text(-0.05, 0.5, 'EVI mean', rotation = 90, verticalalignment = 'center', transform = ax.transAxes)

            ax = axes[1, i]
            cf = map_of_city(std[i, :, :], fid, extent, ax, s = np.power(city_size, 0.3), cmap = 'Spectral', norm = norm_std)
            plt.colorbar(cf, ax = ax, shrink = 0.5)
            ax.set_title(season)
            if i == 0:
                ax.text(-0.05, 0.5, 'EVI interannual std', rotation = 90, verticalalignment = 'center', transform = ax.transAxes)

        fig.savefig(os.path.join(path_out, 'veg', f'summary_seasonal_{name}_{fid}.png'), dpi = 600., bbox_inches = 'tight')
        plt.close(fig)


    def make_events_plot(self, mean_std_in_post, label, fid, extent, mean = False):
        season_list = ['DJF', 'MAM', 'JJA', 'SON']

        ncols_by_season = [0, 0, 0, 0]
        for use in ['daymet', 'topowx', 'yyz']:
            for i, _ in enumerate(season_list):
                filt = mean_std_in_post[use]['time'].to_index().to_period('Q-NOV').quarter == (i + 1)
                ncols_by_season[i] = max(sum(filt), ncols_by_season[i])

        city_size = mean_std_in_post['daymet'].shape[0] * mean_std_in_post['daymet'].shape[1]
        if mean:
            norm = BoundaryNorm([0., 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 256, extend = 'both')
            clabel = 'EVI mean'
        else:
            norm = BoundaryNorm(np.arange(0., 0.11, 0.01), 256, extend = 'max')
            clabel = 'EVI interannual std'

        fig = plt.figure(figsize = (3 * np.sum(ncols_by_season), 9))
        gs = GridSpec(1, 4, hspace = 0, width_ratios = ncols_by_season)
        for i, season in enumerate(season_list):
            sub_gs = GridSpecFromSubplotSpec(3, ncols_by_season[i], subplot_spec = gs[i], hspace = 0., wspace = 0.)
            axes = np.empty(shape = (3, ncols_by_season[i]), dtype=object)
            for j, use in enumerate(['daymet', 'topowx', 'yyz']):

                count = 0

                for k in range(mean_std_in_post[use].shape[0]):
                    quarter = mean_std_in_post[use]['time'].to_index().to_period('Q-NOV')[k].quarter
                    if (quarter - 1) != i:
                        continue

                    ax = plt.subplot(sub_gs[j, count], sharex = axes[0,0], sharey = axes[0,0], projection = crs_daymet)
                    fig.add_subplot(ax)
                    axes[j, count] = ax

                    cf = map_of_city(mean_std_in_post[use][k, :, :], fid, extent, ax, s = np.power(city_size, 0.3) * 1.5, cmap = 'Spectral', norm = norm)
                    ax.set_title(label[use][k])
                    ax.set_ylabel(use)

                    count = count + 1

            # Ghost axes for setting title; somehow I cannot put this before the for loop
            position = list(gs[i].get_position(fig).bounds)
            ax_shared = make_ghost(fig.add_axes(position))
            ax_shared.set_title(season, pad = 20)

        cax = fig.add_axes([0.15, 0.07, 0.7, 0.02])
        plt.colorbar(cf, cax = cax, orientation = 'horizontal', label = clabel)
        return fig


    def run(self):
        name = 'MOD09Q1G'
        extent = 'tiff_3x'

        for fid in range(85):
            print(fid)

            events_list = self.read_events(fid, name, extent)
            veg0 = mask_water(fid, mask_impervious(fid, read_evi(fid, name, extent), 0.8, extent, 'both'), extent, 'both')
            seasonal_mean, seasonal_std = self.calc_evi_normals(veg0)
            events_mean_in, events_std_in, events_mean_post, events_std_post, events_label = self.calc_evi_event_normals(veg0, events_list)

            self.make_seasonal_plot(name, seasonal_mean, seasonal_std, fid, extent)    

            for d, data in zip(['events_mean_in', 'events_std_in', 'events_mean_post', 'events_std_post'],
                                [events_mean_in, events_std_in, events_mean_post, events_std_post]):
                fig = self.make_events_plot(data, events_label, fid, extent, mean = 'mean' in d)
                fig.savefig(os.path.join(path_out, 'veg', f'summary_event_{name}_{d}_{fid}.png'), dpi = 600., bbox_inches = 'tight')
                plt.close(fig)


class ByCity():
    """ Plot the urban & rural & urban-rural difference in EVI in each event, for each season and region separately. """
    def __init__(self):
        self.name = 'MOD09Q1G_EVI'
        self.extent = 'tiff_3x'
        self.region_list = ['Northeast', 'Southeast_top', 'Southeast_hot', 'West']


    def stagedata(self):
        data = pd.DataFrame(np.nan, index = range(85),
                            columns = pd.MultiIndex.from_product([['DJF', 'MAM', 'JJA', 'SON'], ['urban', 'rural', 'diff']], names = ['season', 'location']))
        for fid in range(85):
            F = Filter(fid) 

            for s, season in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
                h = rio.open(os.path.join(path_out, 'veg', self.extent, f'{self.name}_{fid}_mask.tif'))
                temp = h.read()[s, :, :]
                h.close()
                temp = F.apply(temp, season)
                temp = da_to_df(temp).to_frame('evi')
                temp = add_urlabel(fid, temp, self.extent)
                data.loc[fid, (season, 'urban')] = temp.loc[temp['Is_urban_core'], 'evi'].mean()
                data.loc[fid, (season, 'rural')] = temp.loc[~temp['Is_urban_core'], 'evi'].mean()
                data.loc[fid, (season, 'diff' )] = data.loc[fid, (season, 'urban')] - data.loc[fid, (season, 'rural')]
        
        data.to_csv(os.path.join(path_out, 'veg', 'summary_seasonal_evi.csv'))
    
    def readdata(self):
        self.data = pd.read_csv(os.path.join(path_out, 'veg', 'summary_seasonal_evi.csv'), index_col = 0, header = [0, 1])

    def barplot(self):
        fig, axes = plt.subplots(4, len(self.region_list), sharex = False, sharey = True)
        for i, season in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
            for j, region in enumerate(self.region_list):
                temp = self.data.loc[modis_luc_city_groups[region], (season, 'diff')]

                ax = axes[j, i]
                ax.bar(range(len(temp)), temp.values)
                ax.set_xticks(range(len(temp)))
                ax.set_xticklabels(temp.index)

                ax.set_title(season)
                ax.set_ylabel(region)
        fig.savefig(os.path.join(path_out, 'veg', 'summary_seasonal_difference_bar.png'), dpi = 600., bbox_inches = 'tight')
        plt.close(fig)

    def map(self):
        rcParams['font.size'] = 6
        rcParams['axes.titlesize'] = 6

        for which in ['urban', 'rural', 'diff']:
            if which == 'diff':
                map_args = dict(vmin = -0.15, vmax = 0.15, cmap = cmap_div())
            elif which == 'urban':
                map_args = dict(vmin = 0.05, vmax = 0.7, cmap = 'viridis_r')
            elif which == 'rural':
                map_args = dict(vmin = 0.05, vmax = 0.7, cmap = 'viridis_r')

            fig, axes = plt.subplots(2, 2, figsize = (10, 8), sharex = True, sharey = True,
                                    subplot_kw = {'projection': ccrs.AlbersEqualArea(central_longitude = -100, central_latitude = 35)})
            fig.subplots_adjust(hspace = 0.1, wspace = 0.01)
            for i, season in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
                ax = axes.flat[i]

                map = MapOfColors('point', self.data[(season, which)])
                cf = map.plot(inset_bar = False, ax = ax, map_args = map_args)
                ax.set_title(season)
                plt.colorbar(cf, ax = ax, orientation = 'horizontal', extend = 'both', shrink = 0.9)
            fig.savefig(os.path.join(path_out, 'veg', f'summary_seasonal_difference_map_{which}.png'), dpi = 600., bbox_inches = 'tight')
            plt.close(fig)


class SensitivitySummerPixel():
    def __init__(self, extreme):
        self.extreme = extreme
        self.prefix = 'percity_per_pixel'
        self.extent = Setup().extent
        self.name = Setup().name # veg data
        self.heat_wave_thres = Setup().heat_wave_thres
        self.hot_and_dry_thres = Setup().hot_and_dry_thres


    def calc(self):
        """ Compare the sensitivity to SPI and VPD at pixel level during
            - the event & recovery periods of the non "+-" events
            - non heat waves
            - the event & recovery period of the +- events
        """
        h = pd.HDFStore(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'{self.prefix}_{self.extent}_{self.name}_{self.heat_wave_thres}_{self.hot_and_dry_thres}_{self.extreme}_city_average_events.h5'), mode = 'r')
        data = h.select('data').copy()
        h.close()
        summary_varname_urban = data.loc['urban', :]
        summary_varname_rural = data.loc['rural', :]
        temp = summary_varname_urban.index.intersection(summary_varname_rural.index)
        summary_varname_urban = summary_varname_urban.loc[temp, :]
        summary_varname_rural = summary_varname_rural.loc[temp, :]

        # ++: both > 0
        # -+: resistance <= 0, but recovered to >= 0
        # +-: resistance > 0, but recovered to < 0 (JJA is especially of interest)
        # --: both < 0
        pct_urban = pd.Series('                 ', index = summary_varname_urban.index)
        pct_urban.loc[(summary_varname_urban['Resistance'] > 0) & (summary_varname_urban['Recovery'] >= 0)] = '++'
        pct_urban.loc[(summary_varname_urban['Resistance'] <= 0) & (summary_varname_urban['Recovery'] >= 0)] = '\u2212+'
        pct_urban.loc[(summary_varname_urban['Resistance'] > 0) & (summary_varname_urban['Recovery'] < 0)] = '+\u2212'
        pct_urban.loc[(summary_varname_urban['Resistance'] <= 0) & (summary_varname_urban['Recovery'] < 0)] = '\u2212\u2212'

        pct_rural = pd.Series('                 ', index = summary_varname_rural.index)
        pct_rural.loc[(summary_varname_rural['Resistance'] > 0) & (summary_varname_rural['Recovery'] >= 0)] = '++'
        pct_rural.loc[(summary_varname_rural['Resistance'] <= 0) & (summary_varname_rural['Recovery'] >= 0)] = '\u2212+'
        pct_rural.loc[(summary_varname_rural['Resistance'] > 0) & (summary_varname_rural['Recovery'] < 0)] = '+\u2212'
        pct_rural.loc[(summary_varname_rural['Resistance'] <= 0) & (summary_varname_rural['Recovery'] < 0)] = '\u2212\u2212'

        season_ind = pd.DatetimeIndex(pct_urban.index.get_level_values(2)).month.map(month_to_season)

        hsave = pd.HDFStore(os.path.join(path_out, 'veg', f'{self.prefix}_{self.extent}_{self.name}_{self.heat_wave_thres}_{self.hot_and_dry_thres}_{self.extreme}_sensitivity_extreme.h5'), mode = 'w')
        # variables: <spi/vpd>_corr_urban_<season>,
        #            <spi/vpd>_corr_rural_<season>,
        #            <spi/vpd>_corr_lower_urban_<season>,
        #            <spi/vpd>_corr_lower_rural_<season>,
        #            <spi/vpd>_corr_upper_urban_<season>,
        #            <spi/vpd>_corr_upper_rural_<season>
        for fid in tqdm(range(85)):
            evi0 = read_evi(fid, self.name, self.extent).load()
            evi0 = mask_low_evi_seasonal(mask_water(fid, mask_impervious(fid, evi0, 0.8, self.extent, 'both'), self.extent, 'both'), fid, self.name, self.extent)
            with xr.open_dataset(os.path.join(path_intrim, 'Daymet', self.extent, f'spi_{fid}.nc')) as hr:
                spi0 = hr['spi'].copy(deep = True)
                spi0 = spi0.loc[spi0['time'].to_index().year >= 2000, :, :].load()
            vpd0 = read_daymet(fid, 'vpd', self.extent).resample({'time': '1MS'}).mean()
            vpd0 = vpd0.loc[vpd0['time'].to_index().year >= 2000, :, :].load()

            all_events0 = evi0['time'].to_index().intersection(spi0['time'].to_index()).intersection(vpd0['time'].to_index())

            evi0 = evi0.loc[all_events0, :, :]
            vpd0 = vpd0.loc[all_events0, :, :]
            spi0 = spi0.loc[all_events0, :, :]

            # de-seasonalize
            evi0 = evi0.groupby('time.month') - evi0.groupby('time.month').mean()
            vpd0 = vpd0.groupby('time.month') - vpd0.groupby('time.month').mean()

            for loc, pct_data in zip(['urban', 'rural'], [pct_urban, pct_rural]):
                if loc == 'urban':
                    mask = get_mask(fid, 'core', True, self.extent)
                else:
                    mask = get_mask(fid, 'rural', True, self.extent)
                evi = evi0.where(mask)
                spi = spi0.where(mask)
                vpd = vpd0.where(mask)

                for corrname, corrvar in zip(['spi', 'vpd'], [spi, vpd]):
                    corr = {}
                    corr_p = {}

                    for season in ['DJF', 'MAM', 'JJA', 'SON']:
                        all_events = all_events0 # note: temporarily needed for pm_events, etc. below

                        pm_events = list(np.unique([ee for e in \
                            pct_data.index[(season_ind == season) & (pct_data.index.get_level_values('fid') == fid) & (pct_data == '\u2212\u2212')] \
                            for ee in pd.date_range(e[1], e[2], freq = '1MS')]))
                        pm_events_after = list(np.unique([ee for e in \
                            pct_data.index[(season_ind == season) & (pct_data.index.get_level_values('fid') == fid) & (pct_data == '\u2212\u2212')] \
                            for ee in [e[2] + relativedelta(months = 1), e[2] + relativedelta(months = 2), e[2] + relativedelta(months = 3)] \
                            if ee <= all_events[-1]]))

                        other_events = list(np.unique([ee for e in \
                            pct_data.index[(season_ind == season) & (pct_data.index.get_level_values('fid') == fid) & (pct_data != '\u2212\u2212')] \
                            for ee in pd.date_range(e[1], e[2], freq = '1MS')]))
                        other_events_after = list(np.unique([ee for e in \
                            pct_data.index[(season_ind == season) & (pct_data.index.get_level_values('fid') == fid) & (pct_data != '\u2212\u2212')] \
                            for ee in [e[2] + relativedelta(months = 1), e[2] + relativedelta(months = 2), e[2] + relativedelta(months = 3)] \
                            if ee <= all_events[-1]]))

                        all_events = all_events[all_events.month.map(month_to_season) == season]
                        non_events = [ee for ee in all_events if not (ee in pm_events or ee in pm_events_after or ee in other_events or ee in other_events_after)]

                        def spearmanr_catch(evi, y):
                            if len(evi) == 0:
                                return np.nan, np.nan
                            else:
                                filt = ~(np.isnan(evi) | np.isnan(y))
                                evi = evi[filt]
                                y = y[filt]

                                if len(evi) < 10:
                                    # below minimum required sample size for Spearman's rho
                                    return np.nan, np.nan

                                rho, pval = spearmanr(evi, y)
                                ## https://stats.stackexchange.com/questions/18887/how-to-calculate-a-confidence-interval-for-spearmans-rank-correlation
                                ## Bonnett and Wright (2000)
                                # delta = 1.96 * math.sqrt(1 + rho * rho / 2) / math.sqrt(len(evi) - 3)
                                # lower = math.tanh(math.atanh(rho) - delta)
                                # upper = math.tanh(math.atanh(rho) + delta)

                                return rho, pval

                        corr[(season, 'non_events')], corr_p[(season, 'non_events')] = \
                            xr.apply_ufunc(spearmanr_catch, evi.loc[non_events, :, :], corrvar.loc[non_events, :, :],
                                           input_core_dims = [['time'], ['time']], output_core_dims = [[], []],
                                           vectorize = True, dask = 'parallel')
                        corr[(season, 'non_events')].compute()
                        corr_p[(season, 'non_events')].compute()

                        corr[(season, 'non_events')] = da_to_df(corr[(season, 'non_events')])
                        corr_p[(season, 'non_events')] = da_to_df(corr_p[(season, 'non_events')])

                        for name, extra in zip(['pm_events', 'pm_events_after', 'other_events', 'other_events_after'], [pm_events, pm_events_after, other_events, other_events_after]):
                            try:
                                corr[(season, name)], corr_p[(season, name)] = \
                                    xr.apply_ufunc(spearmanr_catch, evi.loc[non_events + extra, :, :], corrvar.loc[non_events + extra, :, :],
                                                   input_core_dims = [['time'], ['time']], output_core_dims = [[], []],
                                                   vectorize = True, dask = 'parallel')
                            except:
                                import pdb; pdb.set_trace()
                            corr[(season, name)].compute()
                            corr_p[(season, name)].compute()

                            corr[(season, name)] = da_to_df(corr[(season, name)])
                            corr_p[(season, name)] = da_to_df(corr_p[(season, name)])

                    corr = pd.DataFrame(corr).reset_index()
                    corr['fid'] = fid
                    corr = corr.set_index(['fid', 'row', 'col'])
                    corr.columns.names = ['season', 'type']

                    corr_p = pd.DataFrame(corr_p).reset_index()
                    corr_p['fid'] = fid
                    corr_p = corr_p.set_index(['fid', 'row', 'col'])
                    corr_p.columns.names = ['season', 'type']

                    hsave.append(f'corr_{corrname}_{loc}', corr.stack())
                    hsave.append(f'corr_p_{corrname}_{loc}', corr_p.stack())

        hsave.close()


    def plot(self):
        map_args = {
            'diff': {'norm': BoundaryNorm(np.arange(-0.075, 0.076, 0.025), 256, extend = 'both'), 'cmap': 'PiYG'},
            'abs' : {'norm': BoundaryNorm(np.arange(-0.2, 0.21, 0.05), 256, extend = 'both'), 'cmap': 'RdBu'}
        }
        bar_args = {'color': ['#313695', '#4575b4', '#a50026', '#d73027']}
        lab = 'abcdefghijklmnopqrstuvwxyz'
        season = 'JJA'

        for var in ['spi', 'vpd']:
            fig, axes = plt.subplots(4, 3, figsize = (15, 15), subplot_kw = {'projection': ccrs.AlbersEqualArea(central_longitude = -100, central_latitude = 35)})
            fig.subplots_adjust(hspace = 0.1, wspace = 0.01)
            for j, loc in enumerate(['urban', 'rural']):
                h = pd.HDFStore(os.path.join(path_out, 'veg', f'{self.prefix}_{self.extent}_{self.name}_{self.heat_wave_thres}_{self.hot_and_dry_thres}_{self.extreme}_sensitivity_extreme.h5'), mode = 'r')
                corr = h.select(f'corr_{var}_{loc}')
                corr_p = h.select(f'corr_p_{var}_{loc}')
                h.close()

                corr_season = corr[season].unstack()
                corr_p_season = (corr_p[season].unstack() <= 0.05)
                corr_season = pd.DataFrame(np.where(corr_p_season, corr_season, 0.), index = corr_season.index, columns = corr_season.columns)
                corr_season = corr_season.groupby('fid').mean()
                corr_p_season = corr_p_season.groupby('fid').mean() > 0.5

                for i, event in enumerate(['pm_events', 'pm_events_after', 'other_events', 'other_events_after', 'non_events', 'non_events']):
                    if i <= 1:
                        temp = pd.DataFrame({'value': corr_season[event], 'pval': corr_p_season[event]})
                        if i == 0:
                            title = '+\u2212 events'
                        else:
                            title = 'Post +\u2212 events'
                    elif i == 2:
                        temp = corr_season['pm_events'] - corr_season[event]
                        title = '$\Delta$ +\u2212, other'
                    elif i == 3:
                        temp = corr_season['pm_events_after'] - corr_season[event]
                        title = '$\Delta$ Post +\u2212, post other'
                    elif i == 4:
                        temp = corr_season['pm_events'] - corr_season[event]
                        title = '$\Delta$ +\u2212, non'
                    elif i == 5:
                        temp = corr_season['pm_events_after'] - corr_season[event]
                        title = '$\Delta$ Post +\u2212, non'

                    ax = axes.flat[j*6 + i]
                    if i <= 1:
                        mag = map_args['abs']
                    else:
                        mag = map_args['diff']

                    map = MapOfColors('point', temp)
                    cf = map.plot(ax = ax, inset_bar = True, map_args = mag, bar_args = bar_args)

                    ax.set_title(title)
                    ax.text(0.02, 0.93, lab[j*6 + i], fontweight = 'bold', transform = ax.transAxes)
                    if np.mod(i, 3) == 0:
                        ax.text(-0.1, 0.5, f'{loc.capitalize()}', rotation = 90, verticalalignment = 'center', transform = ax.transAxes)

                    if i >= 2:
                        res = wilcoxon(temp.values)
                        if res.pvalue <= 0.05:
                            fontweight = 'bold'
                        else:
                            fontweight = 'normal'
                        ax.text(0.55, 0.85, f'p = {res.pvalue:.2f}', fontweight = fontweight, transform = ax.transAxes)

                    if j == 1:
                        if i == 0:
                            cax = fig.add_axes([0.1, 0.08, 0.4, 0.01])
                            plt.colorbar(cf, cax = cax, orientation = 'horizontal')
                        elif i == 2:
                            cax = fig.add_axes([0.51, 0.08, 0.4, 0.01])
                            plt.colorbar(cf, cax = cax, orientation = 'horizontal')
            fig.savefig(os.path.join(path_out, 'veg', f'{self.prefix}_{self.extent}_{self.name}_{self.heat_wave_thres}_{self.hot_and_dry_thres}_{self.extreme}_sensitivity_summer_{var}.png'), dpi = 600., bbox_inches = 'tight')
            plt.close(fig)



if __name__ == '__main__':
    #v = ByPixel()
    #v.run()

    #v = ByCity()
    #v.stagedata() # takes a while
    #v.readdata()
    #v.map()

    #s = Sensitivity()
    #s.stagedata()
    #s.rel_temperature()
    #s.rel_temperature_numbers()
    #s.rel_temperature_numbers_plot()
    #s.rel_water()
    #s.rel_water_numbers()
    #s.rel_water_numbers_plot()

    #s = SensitivitySummerPixel('heat_wave')
    #s.calc()
    #s.plot()


    #s = SensitivityWaterPixel()
    #s.calc()
    #s.plot()
