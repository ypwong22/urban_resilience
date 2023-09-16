import pandas as pd
import xarray as xr
import os
import numpy as np
from utils.analysis import *
from utils.constants import *
from utils.paths import *
from utils.extremes import *
from utils.plotting import *
from utils.regression import *
from matplotlib import rcParams
from matplotlib.patches import Rectangle
from matplotlib.cm import get_cmap
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, spearmanr, pearsonr, wilcoxon, linregress
from statsmodels.tools.tools import add_constant
from scipy.interpolate import UnivariateSpline
from matplotlib.colors import BoundaryNorm, Normalize, LinearSegmentedColormap
from statsmodels.discrete.discrete_model import Logit
import statsmodels.api as sm
import statsmodels.formula.api as smf
import itertools as it
from tqdm import tqdm

from monthly_percity_fit_per_pixel import Norm


def _parula_map():
    """ cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
        [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
        [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
        0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
        [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
        0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
        [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
        0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
        [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
        0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
        [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
        0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
        [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
        0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
        0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
        [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
        0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
        [0.0589714286, 0.6837571429, 0.7253857143], 
        [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
        [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
        0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
        [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
        0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
        [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
        0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
        [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
        0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
        [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
        [0.7184095238, 0.7411333333, 0.3904761905], 
        [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
        0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
        [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
        [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
        0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
        [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
        0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
        [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
        [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
        [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
        0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
        [0.9763, 0.9831, 0.0538]] """
    cm_data = get_cmap('viridis')(range(256))
    cm_data = [list(cm_data[i][:3]) + [1 - np.power(i,1.5)/np.power(len(cm_data),1.5)] for i in range(len(cm_data))]
    parula_map = LinearSegmentedColormap.from_list('parula', cm_data)
    return parula_map.reversed()


class Fig1(Plotter):
    def __init__(self, extreme):
        super().__init__(extreme)
        self.prefix = 'percity_per_pixel'

        self.extent = Setup().extent
        self.name = Setup().name # veg data
        self.heat_wave_thres = Setup().heat_wave_thres
        self.hot_and_dry_thres = Setup().hot_and_dry_thres


    def plot_diff(self, no_crop = True):
        """ By each season and whole U.S., the urban-rural difference in sign and magnitude.
        """
        if no_crop:
            suffix = '_nocrop'
        else:
            suffix = ''

        data = pd.read_csv(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'{self.prefix}_{self.extent}_{self.name}_{self.heat_wave_thres}_{self.hot_and_dry_thres}_{self.extreme}_city_average{suffix}.csv'), 
                        index_col = [0,1,2], header = [0, 1])

        rcParams['font.size'] = 6
        rcParams['axes.titlesize'] = 6
        lab = 'abcdefghijklmn'
        clist = ['#0000ff', '#87cefa', '#ff0000', '#e69b00']

        fig, axes = plt.subplots(figsize = (6.5, 3), sharex = True, sharey = True)
        fig.subplots_adjust(hspace = 0.05, wspace = 0.05)
        for i, season in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
            ax = axes.flat[i]

            huelist = ['frac_pos_urban', 'frac_pos_rural', 'median_abs_urban', 'median_abs_rural']

            temp = data.loc[season, data.columns.get_level_values(1).isin(huelist)]
            temp.columns.names = ['stat', 'which']

            temp2 = temp.stack().stack().to_frame('values').reset_index()
            g = sns.boxplot(x = 'stat', hue = 'which', y = 'values', data = temp2, ax = ax, showfliers = False, whis = [10, 90], palette = clist, hue_order = huelist)

            if i < 3:
                ax.legend([],[], frameon=False)
            else:
                sns.move_legend(g, 'lower left', bbox_to_anchor = (-1, -0.2), ncol = 4, title = None)

            ax.text(0.05, 0.9, lab[i], transform = ax.transAxes, fontweight = 'bold')
            ax.set_xlabel('')
            ax.set_ylabel('')

            # calculate the significance of difference between the urban and rural metrics
            pvalues = pd.DataFrame(np.nan, index = pd.MultiIndex.from_product([['Resistance', 'Recovery'], ['median_abs', 'frac_pos']]),
                                columns = ['daymet', 'topowx', 'yyz'])
            diff = pd.DataFrame(np.nan, index = pd.MultiIndex.from_product([['Resistance', 'Recovery'], ['median_abs', 'frac_pos']]),
                                columns = ['daymet', 'topowx', 'yyz'])
            for stat in ['Resistance', 'Recovery']:
                for which in ['median_abs', 'frac_pos']:
                    for use in ['daymet', 'topowx', 'yyz']:
                        diff.loc[(stat, which), use] = np.median(temp.loc[use, (stat, f'{which}_urban')] - temp.loc[use, (stat, f'{which}_rural')])
                        pvalues.loc[(stat, which), use] = wilcoxon(temp.loc[use, (stat, f'{which}_urban')], temp.loc[use, (stat, f'{which}_rural')]).pvalue
            pvalues = pvalues.median(axis = 1)
            diff = diff.median(axis = 1)

            yloc = [0.75, 0.1, 0.35, 0.6]
            for i, stat in enumerate(['Resistance','Recovery']):
                for j, which in enumerate(['frac_pos', 'median_abs']):
                    median = diff.loc[(stat, which)]
                    pval = pvalues.loc[(stat, which)]

                    if pval <= 0.05:
                        kwargs = {'bbox': dict(facecolor = 'none', edgecolor = 'k', pad = 1, lw = 0.5)}
                    else:
                        kwargs = {}
                    ax.text(i - 0.28 + 0.4 * j, yloc[i*2 + j], f'{median:.3f}', **kwargs)
        fix = Setup().format_prefix().replace(f'{Setup().use}_', '')
        fig.savefig(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'percity_spatial_avg_summary_{fix}_{self.extreme}_fig1{suffix}.png'), dpi = 600., bbox_inches = 'tight')
        plt.close(fig)


    def diff_table_regional(self, no_crop = True):
        """ By each season and each region, the urban-rural difference in sign and magnitude.
        """
        if no_crop:
            suffix = '_nocrop'
        else:
            suffix = ''

        data = pd.read_csv(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'{self.prefix}_{self.extent}_{self.name}_{self.heat_wave_thres}_{self.hot_and_dry_thres}_{self.extreme}_city_average{suffix}.csv'), 
                        index_col = [0,1,2], header = [0, 1])

        regional_diff = pd.DataFrame(np.nan,
                                    index = pd.MultiIndex.from_product([self.region_list, ['Resistance', 'Recovery'], ['median_abs', 'frac_pos'], ['DJF', 'MAM', 'JJA', 'SON']]),
                                    columns = pd.MultiIndex.from_product([['diff', 'pvalue'], ['daymet', 'topowx', 'yyz']]))
        for region, stat, which, season, use in it.product(self.region_list, ['Resistance', 'Recovery'], ['median_abs', 'frac_pos'], 
                                                        ['DJF','MAM','JJA','SON'], ['daymet', 'topowx', 'yyz']):
            temp = data.loc[(season, use), (stat, f'{which}_urban')] - data.loc[(season, use), (stat, f'{which}_rural')]
            temp = temp.loc[temp.index.isin(modis_luc_city_groups[region])]
            regional_diff.loc[(region, stat, which, season), ('diff', use)] = np.median(temp)
            regional_diff.loc[(region, stat, which, season), ('pvalue', use)] = wilcoxon(temp).pvalue
        fix = Setup().format_prefix().replace(f'{Setup().use}_', '')
        regional_diff.to_csv(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'percity_spatial_avg_summary_{fix}_{self.extreme}_fig1{suffix}.csv'))
        regional_diff.groupby(level = 0, axis = 1).median().to_csv(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'percity_spatial_avg_summary_{fix}_{self.extreme}_fig1_simplified{suffix}.csv'))


    def plot_city_summary(self, no_crop = False):
        """ Plot the city-level average in the per pixel resistance and recovery. 
            Plot the urban-rural differences. 
            As maps.
        """
        if no_crop:
            suffix = '_nocrop'
        else:
            suffix = ''

        data = pd.read_csv(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'{self.prefix}_{self.extent}_{self.name}_{self.heat_wave_thres}_{self.hot_and_dry_thres}_{self.extreme}_city_average{suffix}.csv'),
                           index_col = [0, 1, 2], header = [0, 1])

        # draw
        cmap = cmap_div() # sns.diverging_palette(10, 240, s = 100, l = 45, as_cmap = True)
        lab = 'abcdefghijklmnopqrstuvwxyz'
        rcParams['font.size'] = 5
        rcParams['axes.titlesize'] = 5

        map_args = {
            'diff_frac_pos': {'norm': BoundaryNorm([-0.09, -0.06, -0.03, -0.015, 0, 0.015, 0.03, 0.06, 0.09], 256, extend = 'both'), 'cmap': cmap},
            'diff_median_abs': {'norm': BoundaryNorm([-0.09, -0.06, -0.03, -0.015, 0, 0.015, 0.03, 0.06, 0.09], 256, extend = 'both'), 'cmap': cmap},
            'abs' : {'norm': BoundaryNorm([-0.24, -0.12, -0.06, 0, 0.06, 0.12, 0.24], 256, extend = 'both'), 'cmap': cmap}
        }
        bar_args = {'color': ['#313695', '#4575b4', '#a50026', '#d73027']}

        for f in range(2):
            fig, axes = plt.subplots(4, 4, figsize = (6.5, 5.5), subplot_kw = {'projection': ccrs.AlbersEqualArea(central_longitude = -100, central_latitude = 35)})
            fig.subplots_adjust(hspace = 0.01, wspace = 0.01)
            for i, season in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
                if f == 0:
                    mag = map_args['abs']
                    for j, name in enumerate(['urban', 'rural']):
                        for k, target in enumerate(['Resistance', 'Recovery']):
                            ax = axes[k*2 + j, i]
                            temp = data.loc[season, (target, f'median_{name}')].groupby('fid').median()
                            pval = data.loc[season, (target, f'median_{name}_pval')].groupby('fid').median()
                            temp = pd.DataFrame({'value' : temp.values, 'pvalue': pval.values <= 0.05})
                            map = MapOfColors('point', temp)
                            cf = map.plot(ax = ax, inset_bar = True, map_args = mag, bar_args = bar_args, point_scale = 0.2)

                            if (j == 0) & (k == 0):
                                ax.set_title(season)
                            if i == 0:
                                ax.text(-0.1, 0.5, f'{target} {name}', rotation = 90, verticalalignment = 'center', transform = ax.transAxes)
                            if (i == 3) & (k == 0):
                                cax = fig.add_axes([0.92, 0.54 - 0.4 * j, 0.01, 0.3])
                                plt.colorbar(cf, cax = cax, orientation = 'vertical')

                            if name == 'rural':
                                # rural: Pearson correlation with the urban results
                                xtemp = data.loc[season, (target, f'median_abs_urban')]
                                ytemp = data.loc[season, (target, f'median_abs_rural')]
                                rho = [None] * 3
                                pval = [None] * 3
                                for m, use in enumerate(['daymet', 'topowx', 'yyz']):
                                    rho[m], pval[m] = pearsonr(xtemp.loc[xtemp.index.get_level_values('use') == use].values, 
                                                               ytemp.loc[ytemp.index.get_level_values('use') == use].values)
                                rho = np.median(rho)
                                pval = np.median(pval)
                                if pval <= 0.05:
                                    fontweight = 'bold'
                                else:
                                    fontweight = 'normal'
                                ax.text(0.6, 0.84, f'$r=${rho:.2g}', fontweight = fontweight, transform = ax.transAxes)
                            ax.text(0.02, 0.91, lab[j*8 + k*4 + i], fontweight = 'bold', transform = ax.transAxes)
                else:
                    for j, name in enumerate(['frac_pos', 'median_abs']):
                        for k, target in enumerate(['Resistance', 'Recovery']):
                            ax = axes[k + j*2, i]
                            temp = data.loc[season, (target, f'{name}_urban')].groupby('fid').median() - data.loc[season, (target, f'{name}_rural')].groupby('fid').median()
                            pval = data.loc[season, (target, f'{name}_pval')].groupby('fid').median()
                            temp = pd.DataFrame({'value' : temp.values, 'pvalue': pval.values <= 0.05})
                            map = MapOfColors('point', temp)
                            cf = map.plot(ax = ax, inset_bar = True, map_args = map_args[f'diff_{name}'], bar_args = bar_args, point_scale = 0.2)

                            if (j == 0) & (k == 0):
                                ax.set_title(season)
                            if i == 0:
                                ax.text(-0.1, 0.5, f'{target} urban\u2212rural', rotation = 90, verticalalignment = 'center', transform = ax.transAxes)
                                if k == 0:
                                    if name == 'median_abs':
                                        ax.text(-0.2, 0, f'$\Delta$Absolute value', rotation = 90, verticalalignment = 'center', transform = ax.transAxes)
                                    else:
                                        ax.text(-0.2, 0, f'$\Delta$Fraction positive', rotation = 90, verticalalignment = 'center', transform = ax.transAxes)
                            if (i == 3) & (k == 0):
                                cax = fig.add_axes([0.92, 0.54 - 0.4 * j, 0.01, 0.3])
                                plt.colorbar(cf, cax = cax, orientation = 'vertical')

                            ax.text(0.02, 0.91, lab[j*8 + k*4 + i], fontweight = 'bold', transform = ax.transAxes)

            fix = Setup().format_prefix().replace(f'{Setup().use}_', '')
            fig.savefig(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'{self.prefix}_{fix}_{self.extreme}_fig1_S{f+1}{suffix}.png'), dpi = 600., bbox_inches = 'tight')
            plt.close(fig)


    def plot_diff_in_optim(self, no_crop = True):
        """ By each season and resilience metric, 
            (1) the urban-rural difference for the whole U.S. and each part
            (2) relationship between the difference and optimal temperature

            Also plot the urban-rural difference in gap to optimal temperature
            (1) tmax
            (2) tmin
        """
        if no_crop:
            suffix = '_nocrop'
        else:
            suffix = ''

        lab = 'abcdefghijklmnopqrstuvwxyz'
        rcParams['font.size'] = 5.
        rcParams['axes.titlesize'] = 5.
        huelist = ['#757575', '#ff9d00', '#fd785c', '#f00030', '#0690e2']
        clist = ['#0000ff', '#87cefa', '#ff0000', '#e69b00']

        map_args = {
            'diff': {'norm': BoundaryNorm(np.arange(-1.5, 1.51, 0.3), 256, extend = 'both'), 'cmap': cmap_div(thres = 0.1)},
            'urban' : {'norm': BoundaryNorm(np.arange(-20., 20.1, 5), 256, extend = 'both'), 'cmap': cmap_div(thres = 0.1)},
            'rural' : {'norm': BoundaryNorm(np.arange(-20., 20.1, 5), 256, extend = 'both'), 'cmap': cmap_div(thres = 0.1)}
        }
        bar_args = {'color': ['#313695', '#4575b4', '#a50026', '#d73027']}

        fig = plt.figure(figsize = (6.5, 6.5))
        gs = GridSpec(2, 1, height_ratios = [2, 1.4], hspace = 0.2, wspace = 0)
        gs_1 = GridSpecFromSubplotSpec(2, 2, subplot_spec = gs[0], wspace = 0.15, hspace = 0.15)
        gs_2 = GridSpecFromSubplotSpec(2, 4, subplot_spec = gs[1], hspace = 0.01, wspace = 0.01)
        axes_1 = np.empty([2, 2], dtype = object)
        axes_1[0,0] = fig.add_subplot(gs_1[0,0])
        axes_1[0,1] = fig.add_subplot(gs_1[0,1])
        axes_1[1,0] = fig.add_subplot(gs_1[1,0])
        axes_1[1,1] = fig.add_subplot(gs_1[1,1])
        axes_2 = np.empty([2, 4], dtype = object)
        for i, j in it.product(range(2), range(4)):
            axes_2[i, j] = fig.add_subplot(gs_2[i,j], projection = ccrs.AlbersEqualArea(central_longitude = -100, central_latitude = 35))


        #########################################
        # Relationship between urban-rural differences and gap to optimal temperature
        # but using tmin
        #########################################
        # Retrieve the urban-rural difference in resilience and predictors
        predictors, _ = self._get_predictors()
        summary_varname_diff, summary_varname_diff_mag, _, _ = self._get_summary()

        intersect = summary_varname_diff.index.intersection(predictors.index)
        season_ind = pd.DatetimeIndex(intersect.get_level_values('end')).month.map(month_to_season)
        summary_varname_diff = summary_varname_diff.loc[intersect, :]
        summary_varname_diff_mag = summary_varname_diff.loc[intersect, :]
        predictors = predictors.loc[intersect, :]

        # Add season
        predictors.index = pd.MultiIndex.from_tuples(((season_ind[p], *item) for p, item in enumerate(predictors.index)),
                                                     names = ['season'] + predictors.index.names)
        summary_varname_diff.index = pd.MultiIndex.from_tuples(((season_ind[p], *item) for p, item in enumerate(summary_varname_diff.index)),
                                                                 names = ['season'] + summary_varname_diff.index.names)
        summary_varname_diff_mag.index = pd.MultiIndex.from_tuples(((season_ind[p], *item) for p, item in enumerate(summary_varname_diff.index)),
                                                                    names = ['season'] + summary_varname_diff.index.names)


        tvar = 'tmin'
        for i, stat in enumerate(['Resistance', 'Recovery']):
            data = pd.DataFrame({'x': predictors.loc[:, f'dgap_to_optimal_{tvar}'],
                                 'y': summary_varname_diff.loc[:, stat],
                                 'yerr': summary_varname_diff.loc[:, stat]})
            data = data.dropna(axis = 0, how = 'any')
            data['month'] = predictors.index.get_level_values('end').month
            data['region'] = predictors['region']
            data1 = data.groupby(['region', 'month', 'use']).mean()
            data1['hue'] = data1.index.get_level_values('region')
            data2 = data.copy().groupby(['month', 'use']).mean().reset_index()
            data2['region'] = 'U.S.'
            data2['hue'] = 'U.S.'
            data2 = data2.set_index(['region', 'month', 'use'])
            data = pd.concat([data1, data2], axis = 0)

            ax = axes_1[0, i]
            sns.scatterplot(x = 'x', y = 'y', hue = 'hue', data = data, ax = ax, legend = False,
                            hue_order = ['U.S.', 'Northeast', 'Southeast_top', 'Southeast_hot', 'West'], palette = huelist, alpha = 0.5, s = 8)
            if i == 0:
                h = [None] * 5
                for dm in range(5):
                    h[dm], = ax.plot(np.nan, np.nan, '-', color = huelist[dm])
                ax.legend(h, ['U.S.', 'Northeast', 'Southeast_top', 'Southeast_hot', 'West'], ncol = 3, loc = 'upper right', columnspacing = 3)
            for g, region in enumerate(['U.S.', 'Northeast', 'Southeast_top', 'Southeast_hot', 'West']):
                slope = [None] * 3
                intercept = [None] * 3
                pr = [None] * 3
                pval = [None] * 3
                rho = [None] * 3
                pval2 = [None] * 3
                for m, met in enumerate(['daymet', 'topowx', 'yyz']):
                    filt = (data.index.get_level_values('use') == met) & (data['hue'] == region)
                    x = data.loc[filt, 'x']
                    y = data.loc[filt, 'y']

                    res = linregress(x, y)
                    slope[m] = res.slope
                    intercept[m] = res.intercept

                    pr[m], pval[m] = pearsonr(x, y)
                    rho[m], pval2[m] = spearmanr(x, y)

                slope = np.median(slope)
                intercept = np.median(intercept)
                pr = np.median(pr)
                pval = np.median(pval)
                rho = np.median(rho)
                pval2 = np.median(pval2)

                if pval <= 0.05:
                    kwargs = {'bbox': dict(facecolor = 'none', edgecolor = huelist[g], pad = 1, lw = 0.5)}
                else:
                    kwargs = {}

                ax.text(0.05 + 0.18*g, 0.72, f'{pr:.2g}', transform = ax.transAxes, color = huelist[g], **kwargs)
                ax.plot([-6.5, 2.8], slope * np.array([-6.5, 2.8]) + intercept, ls = '-', color = huelist[g], lw = 0.5)
                if pval2 <= 0.05:
                    kwargs = {'bbox': dict(facecolor = 'none', edgecolor = huelist[g], pad = 1, lw = 0.5)}
                else:
                    kwargs = {}
                ax.text(0.05 + 0.18*g, 0.62, f'({rho:.2g})', transform = ax.transAxes, color = huelist[g], **kwargs)
            ax.text(-0.05, 1.05, lab[i], fontweight = 'bold', transform = ax.transAxes)
            ax.set_xlim([-1.7, 1.])
            ax.set_ylim([-0.12, 0.2])
            if i == 1:
                ax.set_yticklabels([])
            #ax.set_xlabel('$\Delta$T$_{gap, ' + tvar[1:] + '}$ ($\degree$C)')
            ax.set_xlabel('')
            ax.set_ylabel('$\Delta$' + f'{stat}' + '$_{urban\u2212rural}$')


            data = pd.DataFrame({'x': predictors.loc[:, f'dgap_to_optimal_{tvar}'],
                                 'y': summary_varname_diff.loc[:, stat],
                                 'yerr': summary_varname_diff.loc[:, stat],
                                 'hue': summary_varname_diff.index.get_level_values(0)})
            data = data.dropna(axis = 0, how = 'any')
            ax = axes_1[1, i]
            sns.scatterplot(x = 'x', y = 'y', hue = 'hue', data = data, ax = ax, legend = False,
                            hue_order = ['DJF', 'MAM', 'JJA', 'SON'], palette = clist, alpha = 0.5, s = 8)
            if i == 0:
                h = [None] * 4
                for dm in range(4):
                    h[dm], = ax.plot(np.nan, np.nan, '-', color = clist[dm])
                ax.legend(h, ['DJF', 'MAM', 'JJA', 'SON'], ncol = 4, loc = 'upper right')
            for s, season in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
                slope = [None] * 3
                intercept = [None] * 3
                pr = [None] * 3
                pval = [None] * 3
                rho = [None] * 3
                pval2 = [None] * 3
                for m, met in enumerate(['daymet', 'topowx', 'yyz']):
                    filt = (predictors.index.get_level_values('use') == met)
                    x = predictors.loc[filt, f'dgap_to_optimal_{tvar}'].loc[season]
                    y = summary_varname_diff.loc[filt, stat].loc[season]

                    res = linregress(x, y)
                    slope[m] = res.slope
                    intercept[m] = res.intercept

                    pr[m], pval[m] = pearsonr(x, y)
                    rho[m], pval2[m] = spearmanr(x, y)

                slope = np.median(slope)
                intercept = np.median(intercept)
                pr = np.median(pr)
                pval = np.median(pval)
                rho = np.median(rho)
                pval2 = np.median(pval2)

                if pval <= 0.05:
                    kwargs = {'bbox': dict(facecolor = 'none', edgecolor = clist[s], pad = 1, lw = 0.5)}
                else:
                    kwargs = {}

                ax.text(0.05 + 0.22*s, 0.75, f'{pr:.2g}', transform = ax.transAxes, color = clist[s], **kwargs)
                ax.plot([-6.5, 2.8], slope * np.array([-6.5, 2.8]) + intercept, ls = '-', color = clist[s], lw = 0.5)

                if pval2 <= 0.05:
                    kwargs = {'bbox': dict(facecolor = 'none', edgecolor = clist[s], pad = 1, lw = 0.5)}
                else:
                    kwargs = {}
                ax.text(0.05 + 0.22*s, 0.65, f'({rho:.2g})', transform = ax.transAxes, color = clist[s], **kwargs)
            ax.text(-0.05, 1.05, lab[2+i], fontweight = 'bold', transform = ax.transAxes)
            ax.set_xlim([-6.5, 2.8])
            ax.set_ylim([-0.6, 1])
            if i == 1:
                ax.set_yticklabels([])
            ax.set_xlabel('$\Delta$T$_{gap, ' + tvar[1:] + '}$ urban\u2212rural ($\degree$C)')
            ax.set_ylabel('$\Delta$' + f'{stat}' + '$_{urban\u2212rural}$')


        #########################################
        # Diff in gap to optimal temperature
        #########################################
        # Retrieve the urban & rural average temperature in each season
        seasonal_average_all = pd.DataFrame(np.nan, index = range(85), columns = pd.MultiIndex.from_product([['urban', 'rural'], ['tmax', 'tmin'], ['DJF', 'MAM', 'JJA', 'SON'], ['daymet', 'topowx', 'yyz']]))
        for fid, varname, use in it.product(range(85), ['tmax', 'tmin'], ['daymet', 'topowx', 'yyz']):
            var_series = pd.read_csv(os.path.join(path_out, 'veg', f'summary_series_keep_seasonality_{varname}_{use}_{fid}.csv'), index_col = 0, parse_dates = True)
            var_series = var_series.loc[var_series.index.year >= 2001, :]
            for season in ['DJF', 'MAM', 'JJA', 'SON']:
                for location in ['urban', 'rural']:
                    seasonal_average_all.loc[fid, (location, varname, season, use)] = var_series.loc[var_series.index.month.map(month_to_season) == season, location].mean()
        seasonal_average = seasonal_average_all.groupby(level = [0,1,2], axis = 1).median()

        # Retrieve the optimal temperature
        optimum = pd.read_csv(os.path.join(path_out, 'veg', f'{self.prefix}_{self.extent}_{self.name}_optimalT.csv'), index_col = 0, header = [0, 1])

        predictors, _, _ = self._get_predictors()
        region = predictors['region'].groupby('fid').first()

        # Plot
        for i, season in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
            location = 'diff'
            for k, varname in enumerate(['tmax', 'tmin']):
                ax = axes_2[k, i]

                if location == 'diff':
                    result = (seasonal_average.loc[:, ('urban', varname, season)] - optimum.loc[:, (varname, 'urban')]) - \
                             (seasonal_average.loc[:, ('rural', varname, season)] - optimum.loc[:, (varname, 'rural')])
                else:
                    result = seasonal_average.loc[:, (location, varname, season)] - optimum.loc[:, (varname, location)]

                m = MapOfColors('point', result)
                cf = m.plot(ax = ax, inset_bar = True, bar_args = bar_args, map_args = map_args[location], point_scale = 0.2)

                """
                ax_inset = inset_axes(ax, width = 0.25, height = 0.21,
                                      bbox_to_anchor = ccrs.AlbersEqualArea(central_longitude = -100, central_latitude = 35).transform_point(-95, 53.1, ccrs.PlateCarree()),
                                      bbox_transform = ax.transData, borderpad = 0)
                ax_inset.bar([1,2,3,4], result.groupby(region).median())
                ax_inset.set_xticks([1,2,3,4])
                ax_inset.set_xticklabels(['N','S$_1$','S$_2$','W'])
                ax_inset.tick_params('both', length = 2, pad = 0.5)
                if location == 'diff':
                    ax_inset.set_ylim([-1.3, .8])
                else:
                    ax_inset.set_ylim([-30, 15])
                """

                if k == 0:
                    ax.set_title(season)
                if i == 0:
                    if location == 'diff':
                        ax.text(-0.1, 0.5, '${\Delta}T_{gap,' + varname[1:] + '}$ urban\u2212rural', rotation = 90, verticalalignment = 'center', transform = ax.transAxes)
                    else:
                        ax.text(-0.1, 0.5, 'T$_{gap,' + varname[1:] + '}$ ' + location, rotation = 90, verticalalignment = 'center', transform = ax.transAxes)

                ax.text(0.02, 0.9, lab[4 + i + k*4], transform = ax.transAxes, weight = 'bold')

                if (i == 3) & (k == 0):
                    cax = fig.add_axes([0.1, 0.07, 0.8, 0.01])
                    plt.colorbar(cf, cax = cax, orientation = 'horizontal')

                if location == 'rural':
                    # rural: Pearson correlation with the urban results
                    rho = [None] * 3
                    pval = [None] * 3
                    for m, use in enumerate(['daymet', 'topowx', 'yyz']):
                        xtemp = seasonal_average_all.loc[:, ('urban', varname, season, use)] - optimum.loc[:, (varname, 'urban')]
                        ytemp = seasonal_average_all.loc[:, ('rural', varname, season, use)] - optimum.loc[:, (varname, 'rural')]
                        rho[m], pval[m] = pearsonr(xtemp.values, ytemp.values)
                    rho = np.median(rho)
                    pval = np.median(pval)
                    if pval <= 0.05:
                        fontweight = 'bold'
                    else:
                        fontweight = 'normal'
                    ax.text(0.6, 0.84, f'$r=${rho:.2g}', fontweight = fontweight, transform = ax.transAxes)

        fig.savefig(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', 
                                 f'{self.prefix}_{self.extent}_{self.name}_{self.heat_wave_thres}_{self.hot_and_dry_thres}_{self.extreme}_Fig1_S1{suffix}.png'), dpi = 600., bbox_inches = 'tight')
        plt.close(fig)


    def table_diff2(self, no_crop = True):
        """ By each season and resilience metric, the relationship between urban-rural difference
            and city-level predictors
            - For continuous variables
                - slope of Pearson correlation, Spearman correlation, and R-squared
            - For land use and region
                - box plot
        """
        if no_crop:
            suffix = '_nocrop'
        else:
            suffix = ''

        data = pd.read_csv(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary',
                                        f'{self.prefix}_{self.extent}_{self.name}_{self.heat_wave_thres}_{self.hot_and_dry_thres}_{self.extreme}_city_average{suffix}.csv'),
                            index_col = [0, 1, 2], header = [0, 1])

        predictors_, luc_clist = self._get_predictors()
        predictors_['season'] = predictors_.index.get_level_values('end').month.map(month_to_season)
        predictors = predictors_.groupby(['season', 'use', 'fid']).mean()
        # predictors['region'] = predictors_[['season','region']].groupby(['season', 'use', 'fid']).first()

        ####################################
        # Only predictors dynamic has seasonality
        ####################################
        predictors_dynamic = list(predictors.columns)
        luc_list = ['Developed', 'Crop', 'Deciduous forest', 'Evergreen forest', 'Grass', 'Mixed forest', 'Shrub', 'Wetland']
        for luc in luc_list:
            predictors_dynamic.remove(luc)
        predictors_dynamic.remove('region')
        luc_list.remove('Developed')

        # (impervious_frac is dynamic annually, not seasonally)
        predictors_static = ['city_size_log', 'elev_diff', 'impervious_frac']
        for p in predictors_static:
            predictors_dynamic.remove(p)

        print(predictors_static)
        print(predictors_dynamic)
        print(luc_list)

        ####################################
        # By season and resilience metric: spatial regression
        ####################################
        results = pd.DataFrame(np.nan,
                            index = pd.MultiIndex.from_product([['Resistance', 'Recovery'], ['Sign', 'Mag'], predictors_dynamic + predictors_static + ['luc', 'region']]),
                            columns = pd.MultiIndex.from_product([['R', 'R_p', 'Rho', 'Rho_p', 'R2'], ['DJF', 'MAM', 'JJA', 'SON'], ]))
        for which, stat, aux, season in it.product(['Sign', 'Mag'], ['Resistance', 'Recovery'], predictors_dynamic + predictors_static, ['DJF', 'MAM', 'JJA', 'SON']):
            if stat == 'Resistance':
                if 'post_event' in aux:
                    continue # irrelevant
            elif stat == 'Recovery':
                if 'in_event' in aux:
                    continue
            x = predictors.loc[season, aux]

            if which == 'Sign':
                y = data.loc[season, (stat, 'frac_pos_urban')] - data.loc[season, (stat, 'frac_pos_rural')]
            else:
                y = data.loc[season, (stat, 'median_abs_urban')] - data.loc[season, (stat, 'median_abs_rural')]

            slope = [None] * 3
            slope_p = [None] * 3
            r2 = [None] * 3
            rho = [None] * 3
            rho_p = [None] * 3
            for u, use in enumerate(['daymet', 'topowx', 'yyz']):
                x_ = x.loc[use] # .groupby('fid').mean()
                y_ = y.loc[use] # .groupby('fid').mean()
                res = linregress(x_, y_)
                r2[u] = res.rvalue **2
                slope[u], slope_p[u] = pearsonr(x_, y_)
                rho[u], rho_p[u] = spearmanr(x_, y_)

            results.loc[(stat, which, aux), ('R2', season)] = np.median(r2)
            results.loc[(stat, which, aux), ('R', season)] = np.median(slope)
            results.loc[(stat, which, aux), ('R_p', season)] = np.median(slope_p)
            results.loc[(stat, which, aux), ('Rho', season)] = np.median(rho)
            results.loc[(stat, which, aux), ('Rho_p', season)] = np.median(rho_p)

        fix = Setup().format_prefix().replace(f'{Setup().use}_', '')
        results.to_csv(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'percity_spatial_avg_summary_{fix}_{self.extreme}_fig1_table1{suffix}.csv'))


        results = pd.read_csv(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'percity_spatial_avg_summary_{fix}_{self.extreme}_fig1_table1{suffix}.csv'),
                            index_col = [0,1,2], header = [0, 1])


        ####################################
        # Boxplots of by land cover differences
        ####################################
        luc_main = predictors.loc[:, luc_list].idxmax(axis = 1)

        fig, axes = plt.subplots(4, 4, figsize = (8, 8), sharex = True, sharey = True)
        for i, which in enumerate(['Sign', 'Mag']):
            for j, stat in enumerate(['Resistance', 'Recovery']):
                for k, season in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
                    ax = axes[i*2 + j, k]
                    if which == 'Sign':
                        y = data.loc[season, (stat, 'frac_pos_urban')] - data.loc[season, (stat, 'frac_pos_rural')]
                    else:
                        y = data.loc[season, (stat, 'median_abs_urban')] - data.loc[season, (stat, 'median_abs_rural')]
                    temp = pd.DataFrame({'x': luc_main.loc[season, :], 'y': y})
                    sns.boxplot(x = 'x', y = 'y', data = temp, order = luc_list, ax = ax, whis = [10, 90], showfliers = False)
                    ax.axhline(0., ls = ':', lw = 0.5)

                    # calculate the percent explained variance
                    residuals = temp.copy()
                    standard = temp.groupby('x').median()
                    for ind in standard.index:
                        residuals.loc[temp['x'] == ind, 'y'] = residuals.loc[temp['x'] == ind, 'y'] - standard.loc[ind, 'y']

                    r2 = 1 - np.sum(np.power(residuals['y'].values, 2)) / np.sum(np.power(temp['y'].values, 2))
                    ax.text(0.05, 0.88, f'$R^2 = ${r2:.2f}', transform = ax.transAxes)
                    results.loc[(stat, which, 'luc'), ('R2', season)] = r2

                    plt.setp(ax.get_xticklabels(), rotation = 90)
                    ax.set_xlabel('')
                    if (i == 0) & (j == 0):
                        ax.set_title(season)
                    if k == 0:
                        ax.set_ylabel(f'{which} {stat}')
                    else:
                        ax.set_ylabel('')

        fix = Setup().format_prefix().replace(f'{Setup().use}_', '')
        fig.savefig(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'percity_spatial_avg_summary_{fix}_{self.extreme}_fig1_S1{suffix}.png'), dpi = 600., bbox_inches = 'tight')
        plt.close(fig)


        ####################################
        # Boxplots of regional differences
        ####################################
        fig, axes = plt.subplots(4, 4, figsize = (8, 8))
        for i, which in enumerate(['Sign', 'Mag']):
            for j, stat in enumerate(['Resistance', 'Recovery']):
                for k, season in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
                    ax = axes[i*2 + j, k]
                    if which == 'Sign':
                        y = data.loc[season, (stat, 'frac_pos_urban')] - data.loc[season, (stat, 'frac_pos_rural')]
                    else:
                        y = data.loc[season, (stat, 'median_abs_urban')] - data.loc[season, (stat, 'median_abs_rural')]
                    x = predictors.loc[season, 'region']
                    temp = pd.DataFrame({'x': x, 'y': y})
                    sns.boxplot(x = 'x', y = 'y', data = temp, order = region_names, ax = ax, whis = [10, 90], showfliers = False)
                    ax.axhline(0., ls = ':', lw = 0.5)

                    # calculate the percent explained variance
                    residuals = temp.copy()
                    standard = temp.groupby('x').median()
                    for ind in standard.index:
                        residuals.loc[temp['x'] == ind, 'y'] = residuals.loc[temp['x'] == ind, 'y'] - standard.loc[ind, 'y']

                    r2 = 1 - np.sum(np.power(residuals['y'].values, 2)) / np.sum(np.power(temp['y'].values, 2))
                    ax.text(0.05, 0.88, f'$R^2 = ${r2:.2f}', transform = ax.transAxes)
                    results.loc[(stat, which, 'region'), ('R2', season)] = r2

                    plt.setp(ax.get_xticklabels(), rotation = 90)
                    ax.set_xlabel('')
                    if (i == 0) & (j == 0):
                        ax.set_title(season)
                    if k == 0:
                        ax.set_ylabel(f'{which} {stat}')
                    else:
                        ax.set_ylabel('')

        fix = Setup().format_prefix().replace(f'{Setup().use}_', '')
        fig.savefig(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'percity_spatial_avg_summary_{fix}_{self.extreme}_fig1_S2{suffix}.png'), dpi = 600., bbox_inches = 'tight')
        plt.close(fig)

        fix = Setup().format_prefix().replace(f'{Setup().use}_', '')
        results.to_csv(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'percity_spatial_avg_summary_{fix}_{self.extreme}_fig1_table1{suffix}.csv'))


        ####################################
        # Visualize the differences in R-squared across the predictors
        ####################################
        results = pd.read_csv(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'percity_spatial_avg_summary_{fix}_{self.extreme}_fig1_table1{suffix}.csv'),
                              index_col = [0,1,2], header = [0, 1])
        rcParams['font.size'] = 6
        rcParams['axes.titlesize'] = 6
        fig, axes = plt.subplots(1, 4, figsize = (6.5, 6.5), sharex = True, sharey = True)
        for i, stat in enumerate(['Resistance', 'Recovery']):
            for j, which in enumerate(['Sign', 'Mag']):
                ax = axes[j*2 + i]

                temp = results.loc[(stat, which), 'R2']

                ax.imshow(temp, vmin = 0., vmax = 0.3, cmap = 'RdYlBu', aspect = 0.7)

                for a,b in it.product(range(temp.shape[0]), range(temp.shape[1])):
                    val = temp.values[a, b]
                    if ~np.isnan(val):
                        ax.text(b, a, f'{val:.2f}', ha = 'center', va = 'center')

                ax.set_title(stat)
                if i == 0:
                    if which == 'Sign':
                        ax.text(1., 1.07, '$\Delta$Fraction positive', transform = ax.transAxes, horizontalalignment = 'center')
                    else:
                        ax.text(1., 1.07, '$\Delta$Absolute value', transform = ax.transAxes, horizontalalignment = 'center')

                ax.set_yticks(range(temp.shape[0]))
                ax.set_yticklabels(temp.index)
                ax.set_xticks(range(temp.shape[1]))
                ax.set_xticklabels(['DJF', 'MAM', 'JJA', 'SON'], rotation = 90)
        fig.savefig(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'percity_spatial_avg_summary_{fix}_{self.extreme}_fig1_S3{suffix}.png'), dpi = 600., bbox_inches = 'tight')
        plt.close(fig)


        ####################################
        # By region and resilience metric: monthly regression
        ####################################
        modis_fid_to_region = dict([(k,i) for i,j in modis_luc_city_groups.items() for k in j])

        results = pd.DataFrame(np.nan,
                                index = pd.MultiIndex.from_product([['Resistance', 'Recovery'], ['Sign', 'Mag'], predictors_dynamic]),
                                columns = pd.MultiIndex.from_product([['Slope', 'Slope_p'], ['U.S.', 'Northeast', 'Southeast_top', 'Southeast_hot', 'West']]))
        for which, stat, aux, region in it.product(['Sign', 'Mag'], ['Resistance', 'Recovery'], predictors_dynamic,
                                                    ['U.S.', 'Northeast', 'Southeast_top', 'Southeast_hot', 'West']):
            if region == 'U.S.':
                filt = np.full(predictors.shape[0], True)
            else:
                filt = pd.Series(predictors.index.get_level_values('fid')).map(modis_fid_to_region).values == region

            if stat == 'Resistance':
                if 'post_event' in aux:
                    continue # irrelevant
            elif stat == 'Recovery':
                if 'in_event' in aux:
                    continue

            x = predictors.loc[filt, aux]

            if region == 'U.S.':
                filt = np.full(data.shape[0], True)
            else:
                filt = pd.Series(data.index.get_level_values('fid')).map(modis_fid_to_region).values == region

            if which == 'Sign':
                y = data.loc[filt, (stat, 'frac_pos_urban')] - data.loc[filt, (stat, 'frac_pos_rural')]
            else:
                y = data.loc[filt, (stat, 'median_abs_urban')] - data.loc[filt, (stat, 'median_abs_rural')]

            slope = [None] * 3
            slope_p = [None] * 3
            for u, use in enumerate(['daymet', 'topowx', 'yyz']):
                x_ = x.loc[x.index.get_level_values('use') == use]
                y_ = y.loc[y.index.get_level_values('use') == use]
                grp_ = x_.index.get_level_values('fid')
                temp = pd.DataFrame({'x': x_.values, 'y': y_.values, 'grp': grp_.values})
                md = smf.mixedlm(f'y ~ x', temp, groups = temp['grp'])
                mdf = md.fit().summary()
                try:
                    slope[u] = float(mdf.tables[1].loc['x', 'Coef.'])
                    slope_p[u] = float(mdf.tables[1].loc['x', 'P>|z|'])
                except:
                    slope[u] = np.nan
                    slope_p[u] = np.nan

            results.loc[(stat, which, aux), ('Slope', region)] = np.median(slope)
            results.loc[(stat, which, aux), ('Slope_p', region)] = np.median(slope_p)

        fix = Setup().format_prefix().replace(f'{Setup().use}_', '')
        results.to_csv(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'percity_spatial_avg_summary_{fix}_{self.extreme}_fig1_table2{suffix}.csv'))


    def city_level_regression(self, no_crop = True):
        pass


class Fig2(Plotter):
    def __init__(self, extreme, region_list = None, n_estimators = None, subset_pred = None):
        super().__init__(extreme)

        # override if new values are provided
        if region_list is not None:
            self.region_list = region_list
        if n_estimators is not None:
            self.n_estimators = n_estimators
        if subset_pred is not None:
            self.subset_pred = subset_pred


    def stagedata(self, calc = False):
        """ Divide everything into urban and rural averages. """
        if calc:
            y_test_diff = {}
            y_pred_test_diff = {}
            y_pred_test_std  = {}
            shap_values_diff = {}
            for region, target, season, use in it.product(self.region_list, ['Resistance', 'Recovery'], ['DJF', 'MAM', 'JJA', 'SON'], ['daymet', 'topowx', 'yyz']):
                setup = Setup()
                setup.set_use(use)
                run_suffix = f'{setup.format_prefix()}_{self.extreme}_{season}_{self.subset_pred}_{region}_{target}_{self.n_estimators}'

                filename = os.path.join(path_out, 'measures', 'regression_per_pixel', 'shap', f'{self.prefix}_{run_suffix}.h5')

                with pd.HDFStore(filename, mode = 'r') as hf:
                    try:
                        y_test = hf.select('y_test')[target].sort_index(axis = 0)
                        y_pred_test = hf.select('y_pred_test')[target].sort_index(axis = 0)
                        shap_values = hf.select('shap_values').sort_index(axis = 0)
                    except:
                        continue

                    y_pred_test_std[(region, target, season, use)] = y_pred_test.groupby(['fid', 'start', 'end']).std()

                    location = add_urlabel_all(shap_values.index.to_frame(index = False), setup.extent)['location'].values

                    y_test_urban = y_test.loc[location == 'urban'].groupby(['fid', 'start', 'end']).mean()
                    y_test_rural = y_test.loc[location == 'rural'].groupby(['fid', 'start', 'end']).mean()
                    y_pred_test_urban = y_pred_test.loc[location == 'urban'].groupby(['fid', 'start', 'end']).mean()
                    y_pred_test_rural = y_pred_test.loc[location == 'rural'].groupby(['fid', 'start', 'end']).mean()
                    shap_values_urban = shap_values.loc[location == 'urban', :].groupby(['fid', 'start', 'end']).mean()
                    shap_values_rural = shap_values.loc[location == 'rural', :].groupby(['fid', 'start', 'end']).mean()

                    y_test_diff[(region, target, season, use)] = y_test_urban - y_test_rural
                    y_pred_test_diff[(region, target, season, use)] = y_pred_test_urban - y_pred_test_rural
                    shap_values_diff[(region, target, season, use)] = (shap_values_urban - shap_values_rural).stack()

            y_test_diff = pd.DataFrame(y_test_diff).stack() # series
            y_test_diff.index.names = ['fid', 'start', 'end', 'use']
            y_pred_test_diff = pd.DataFrame(y_pred_test_diff).stack()
            y_pred_test_diff.index.names = ['fid', 'start', 'end', 'use']
            y_pred_test_std  = pd.DataFrame(y_pred_test_std).stack()
            y_pred_test_std.index.names = ['fid', 'start', 'end', 'use']
            shap_values_diff = pd.DataFrame(shap_values_diff).unstack()
            shap_values_diff.columns = shap_values_diff.columns.reorder_levels([0, 1, 2, 4, 3])
            shap_values_diff = shap_values_diff.stack()
            shap_values_diff.index.names = ['fid', 'start', 'end', 'use']

            y_test_diff.to_csv(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'{self.prefix}_{self.extent}_{self.name}_{self.heat_wave_thres}_{self.hot_and_dry_thres}_{self.extreme}_y_test_diff.csv'))
            y_pred_test_diff.to_csv(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'{self.prefix}_{self.extent}_{self.name}_{self.heat_wave_thres}_{self.hot_and_dry_thres}_{self.extreme}_y_pred_test_diff.csv'))
            y_pred_test_std.to_csv(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'{self.prefix}_{self.extent}_{self.name}_{self.heat_wave_thres}_{self.hot_and_dry_thres}_{self.extreme}_y_pred_test_std.csv'))
            shap_values_diff.to_csv(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'{self.prefix}_{self.extent}_{self.name}_{self.heat_wave_thres}_{self.hot_and_dry_thres}_{self.extreme}_shap_values_diff.csv'))

        self.y_test_diff = pd.read_csv(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'{self.prefix}_{self.extent}_{self.name}_{self.heat_wave_thres}_{self.hot_and_dry_thres}_{self.extreme}_y_test_diff.csv'), index_col = [0,1,2,3], header = [0, 1, 2], parse_dates = True)
        self.y_pred_test_diff = pd.read_csv(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'{self.prefix}_{self.extent}_{self.name}_{self.heat_wave_thres}_{self.hot_and_dry_thres}_{self.extreme}_y_pred_test_diff.csv'), index_col = [0,1,2,3], header = [0, 1, 2], parse_dates = True)
        self.y_pred_test_std = pd.read_csv(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'{self.prefix}_{self.extent}_{self.name}_{self.heat_wave_thres}_{self.hot_and_dry_thres}_{self.extreme}_y_pred_test_std.csv'), index_col = [0,1,2,3], header = [0, 1, 2], parse_dates = True)
        self.shap_values_diff = pd.read_csv(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'{self.prefix}_{self.extent}_{self.name}_{self.heat_wave_thres}_{self.hot_and_dry_thres}_{self.extreme}_shap_values_diff.csv'), index_col = [0,1,2,3], header = [0, 1, 2, 3], parse_dates = True)


    def urban_minus_rural(self):
        """
        More useful ways to summarize the SHAP values obtained by regression.
        """
        norm_levels = [-1, -0.75, -0.5, -0.25, -0.1, -0.05, -0.01, 0., 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1] # np.around(np.power(10, np.linspace(-2, 0, 15)), 2)
        norm = {
            'Resistance': BoundaryNorm(norm_levels, ncolors = 256, extend = 'neither'), # LogNorm(0.01, 1, clip = True),
            'Recovery': BoundaryNorm(norm_levels, ncolors = 256, extend = 'neither'), # LogNorm(0.01, 1, clip = True),
            'Resilience': BoundaryNorm(norm_levels, ncolors = 256, extend = 'neither'), # LogNorm(0.01, 1, clip = True)
        }
        # norm = {
        #     'Resistance': SymLogNorm(vmin = -1, vmax = 1, linthresh = 0.01),
        #     'Recovery': SymLogNorm(vmin = -1, vmax = 1, linthresh = 0.01),
        #     'Recovery': SymLogNorm(vmin = -1, vmax = 1, linthresh = 0.01)
        # }
        lab = 'abcdefghijklmnopqrstuvwxyz'
        rcParams['font.size'] = 6.
        rcParams['axes.titlesize'] = 6.

        fig, axes = plt.subplots(1, 2 * len(self.region_list), figsize = (8, 4), sharex = True, sharey = True)
        # fig.subplots_adjust(hspace = 0.40)

        for j, target in enumerate(['Resistance', 'Recovery']):
            if self.extreme == 'heat_wave':
                factor_list = ['event_intensity', 'event_duration']
            elif self.extreme == 'hot_and_dry':
                factor_list = ['hot_intensity', 'dry_intensity', 'event_duration']
            if target == 'Recovery':
                factor_list = factor_list + ['dtmax_post_event', 'dtmin_post_event', 'spi_post_event', 'vpd_post_event']
            else:
                factor_list = factor_list + ['dtmax_in_event', 'dtmin_in_event', 'spi_in_event', 'vpd_in_event']
            factor_list = factor_list + ['impervious_frac', 'elev', 'background_tmean', 'background_prcp',
                                         'Evergreen forest', 'Mixed forest', 'Deciduous forest', 'Shrub', 'Grass', 'Crop', 'Wetland']

            fig.text(0.3 + 0.4 * j, 0.92, s = target, horizontalalignment = 'center')

            for i, key in enumerate(self.region_list):
                shap_values_diff_mean = self.shap_values_diff.loc[:, (key, target)].mean(axis = 0).unstack().T
                shap_values_diff_mean = shap_values_diff_mean.loc[factor_list, :]

                # Mask the pts that pushes < 1/10 of the mean urban-rural difference
                # threshold = np.broadcast_to(shap_values_diff_mean.abs().max(axis = 0).values.reshape(1, -1) * 0.1, shap_values_diff_mean.shape)
                # threshold = np.broadcast_to(self.y_pred_test_diff.loc[:, (key, target)].mean(axis = 0).values.reshape(1, -1) * 0.1, shap_values_diff_mean.shape)

                contribution_score = shap_values_diff_mean.abs()
                contribution_score = (contribution_score - contribution_score.min(axis = 0)) / (contribution_score.max(axis = 0) - contribution_score.min(axis = 0))
                contribution_score = contribution_score * (shap_values_diff_mean / shap_values_diff_mean.abs())

                ax = axes[j*4 + i]
                cf = ax.imshow(contribution_score, norm = norm[target], cmap = cmap_div(thres = 0.05))
                ax.set_yticks(range(shap_values_diff_mean.shape[0]))
                ax.set_xticks(range(shap_values_diff_mean.shape[1]))
                ax.set_xticklabels(['DJF', 'MAM', 'JJA', 'SON'], rotation = 90)
                if i == 0 and j == 0:
                    ax.set_yticklabels([f.replace('_post_', '_in/post_').replace('_in_', '_in/post_') for f in factor_list])
                ax.set_title(key)

                if i == 0 and j == 0:
                    cax = fig.add_axes([0.1, 0.03, 0.8, 0.012])
                    cbar = plt.colorbar(cf, cax = cax, orientation = 'horizontal', label = 'Contribution score', extend = 'neither')
                    cbar.ax.set_xticks(norm_levels) # add the labels
                    cbar.ax.set_xticklabels(norm_levels) # add the labels
                ax.text(-0.2, 1.02, lab[i + j*len(self.region_list)], fontweight = 'bold', transform = ax.transAxes)

                mask = (contribution_score.abs() >= 1/10) & (contribution_score.abs() < 1/2) # shap_values_diff_mean.abs() < threshold
                for a,b in it.product(range(shap_values_diff_mean.shape[0]), range(shap_values_diff_mean.shape[1])):
                    if mask.iloc[a, b]:
                        ax.text(b, a, 'x', fontsize = 6, verticalalignment = 'center', horizontalalignment = 'center')
                mask = (contribution_score.abs() >= 1/2) # shap_values_diff_mean.abs() < threshold
                for a,b in it.product(range(shap_values_diff_mean.shape[0]), range(shap_values_diff_mean.shape[1])):
                    if mask.iloc[a, b]:
                        ax.text(b, a, '$\emptyset$', fontsize = 6, verticalalignment = 'center', horizontalalignment = 'center')

        fig.savefig(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'{self.prefix}_{self.extent}_{self.name}_{self.heat_wave_thres}_{self.hot_and_dry_thres}_{self.extreme}_contribution_to_diff.png'), bbox_inches = 'tight', dpi = 600.)
        plt.close(fig)


    def actual(self):
        """ Importance scores and Spearman correlation coefficients summarized partial dependence relationships """
        #norm_importance_levels = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
        norm_importance = {
            'Resistance': LogNorm(0.01, 1, clip = True),
            'Recovery': LogNorm(0.01, 1, clip = True),
            'Resilience': LogNorm(0.01, 1, clip = True)
        }
        norm_importance_levels = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1] # np.around(np.power(10, np.linspace(-2, 0, 15)), 2)
        #norm_importance = {
        #    'Resistance': BoundaryNorm(norm_importance_levels, ncolors = 256, extend = 'neither'), # LogNorm(0.01, 1, clip = True),
        #    'Recovery': BoundaryNorm(norm_importance_levels, ncolors = 256, extend = 'neither'), # LogNorm(0.01, 1, clip = True),
        #    'Resilience': BoundaryNorm(norm_importance_levels, ncolors = 256, extend = 'neither'), # LogNorm(0.01, 1, clip = True)
        #}
        norm_correlation = {
            'Resistance': BoundaryNorm(np.arange(-0.8, 0.81, 0.1), ncolors = 256, extend = 'both'), #SymLogNorm(vmin = -0.5, vmax = 0.5, linthresh = 0.0005),
            'Recovery': BoundaryNorm(np.arange(-0.8, 0.81, 0.1), ncolors = 256, extend = 'both'), #SymLogNorm(vmin = -0.5, vmax = 0.5, linthresh = 0.0005),
            'Resilience': BoundaryNorm(np.arange(-0.8, 0.81, 0.1), ncolors = 256, extend = 'both') #SymLogNorm(vmin = -0.5, vmax = 0.5, linthresh = 0.0005),
        }
        lab = 'abcdefghijklmnopqrstuvwxyz'
        rcParams['font.size'] = 6.
        rcParams['axes.titlesize'] = 6.

        fig, axes = plt.subplots(2, 2 * len(self.region_list), figsize = (8, 7), sharex = True, sharey = True)
        fig.subplots_adjust(hspace = 0.40)

        for j, target in enumerate(['Resistance', 'Recovery']):
            if self.extreme == 'heat_wave':
                factor_list = ['event_intensity', 'event_duration']
            elif self.extreme == 'hot_and_dry':
                factor_list = ['hot_intensity', 'dry_intensity', 'event_duration']
            if target == 'Recovery':
                factor_list = factor_list + ['dtmax_post_event', 'dtmin_post_event', 'spi_post_event', 'vpd_post_event']
            else:
                factor_list = factor_list + ['dtmax_in_event', 'dtmin_in_event', 'spi_in_event', 'vpd_in_event']
            factor_list = factor_list + ['impervious_frac', 'elev', 'background_tmean', 'background_prcp',
                                         'Evergreen forest', 'Mixed forest', 'Deciduous forest', 'Shrub', 'Grass', 'Crop', 'Wetland']

            fig.text(0.3 + 0.4 * j, 0.92, s = target, horizontalalignment = 'center')

            # importances_collect = {}
            for i, key in tqdm(enumerate(self.region_list)):
                setup = Setup()

                importances = pd.DataFrame(np.nan, columns = ['daymet', 'topowx', 'yyz'], 
                                           index = pd.MultiIndex.from_product([['DJF', 'MAM', 'JJA', 'SON'], factor_list]))
                for use in ['daymet', 'topowx', 'yyz']:
                    setup.set_use(use)
                    for season in ['DJF', 'MAM', 'JJA', 'SON']:
                        run_suffix = f'{self.extreme}_{season}_{self.subset_pred}_{key}_{target}_{self.n_estimators}'
                        filename = os.path.join(path_out, 'measures', 'regression_per_pixel', 'shap', f'percity_per_pixel_{setup.format_prefix()}_{run_suffix}.h5')
                        with pd.HDFStore(filename, mode = 'r') as hf:
                            for factor in factor_list:
                                importances.loc[(season, factor), use] = hf.select('shap_values').abs().mean(axis = 0).loc[factor]
                # importances_collect[key] = importances.mean(axis = 1).unstack().T.loc[factor_list, :]
                importances = importances.mean(axis = 1).unstack().T.loc[factor_list, :]
                importance_score = (importances - importances.min(axis = 0)) / (importances.max(axis = 0) - importances.min(axis = 0))

                correlations = pd.DataFrame(np.nan, index = factor_list, columns = ['DJF', 'MAM', 'JJA', 'SON'])
                significance = pd.DataFrame(np.nan, index = factor_list, columns = ['DJF', 'MAM', 'JJA', 'SON'])
                for season in ['DJF', 'MAM', 'JJA', 'SON']:
                    for factor in factor_list:
                        shap_values = [None] * 3
                        X_test_norm = [None] * 3
                        for k, use in enumerate(['daymet', 'topowx', 'yyz']):
                            setup.set_use(use)
                            run_suffix = f'{self.extreme}_{season}_{self.subset_pred}_{key}_{target}_{self.n_estimators}'
                            filename = os.path.join(path_out, 'measures', 'regression_per_pixel', 'shap', f'percity_per_pixel_{setup.format_prefix()}_{run_suffix}.h5')
                            with pd.HDFStore(filename, mode = 'r') as hf:
                                shap_values[k] = hf.select('shap_values')[factor].values
                                X_test_norm[k] = hf.select('X_test_norm')[factor].values
                        shap_values = np.concatenate(shap_values)
                        X_test_norm = np.concatenate(X_test_norm)
                        corr, pval = spearmanr(X_test_norm, shap_values)
                        correlations.loc[factor, season] = corr
                        significance.loc[factor, season] = pval

                # correlations_unimportant = correlations.copy()
                ## Following the convention of Cohen's d, if the effect is < 0.2 * standard deviation, consider negligible
                # threshold = 0.2 * np.sqrt(np.mean(np.power(self.y_pred_test_std.loc[:, (key, target, season)].values, 2)))
                # threshold = np.broadcast_to(importances.max(axis = 0) * 0.1, correlations.shape)
                # correlations_unimportant.loc[:, :] = np.where(importances.values > threshold, np.nan, correlations.values) # mask out the variables that are not important
                # correlations.loc[:, :] = np.where(importances.values <= threshold, np.nan, correlations.values) # mask out the variables that are not important

                ax = axes[0, j*4 + i]
                cf = ax.imshow(importance_score, norm = norm_importance[target], cmap = 'Blues') # _parula_map())
                ax.set_yticks(range(importance_score.shape[0]))
                ax.set_xticks(range(importance_score.shape[1]))
                if i == 0 and j == 0:
                    ax.set_yticklabels([f.replace('_post_', '_in/post_').replace('_in_', '_in/post_') for f in factor_list])
                ax.set_title(key)

                if i == 0 and j == 0:
                    cax = fig.add_axes([0.1, 0.53, 0.8, 0.012])
                    cbar = plt.colorbar(cf, cax = cax, orientation = 'horizontal', label = 'Importance score', extend = 'min')
                    cbar.ax.set_xticks(norm_importance_levels) # add the labels
                    cbar.ax.set_xticklabels(norm_importance_levels) # add the labels
                ax.text(-0.2, 1.02, lab[i + j*len(self.region_list)], fontweight = 'bold', transform = ax.transAxes)

                ax = axes[1, j*4 + i]
                cf = ax.imshow(correlations, norm = norm_correlation[target], cmap = cmap_div(thres = 0.05))
                # cf2 = ax.imshow(correlations_unimportant, norm = norm_correlation[target], cmap = 'PRGn')
                for a, b in it.product(range(correlations.shape[0]), range(correlations.shape[1])):
                    if significance.values[a, b] <= 0.05:
                        ax.text(b, a, 'x', fontsize = 6, verticalalignment = 'center', horizontalalignment = 'center')
                ax.set_yticks(range(correlations.shape[0]))
                ax.set_xticks(range(correlations.shape[1]))
                ax.set_xticklabels(['DJF', 'MAM', 'JJA', 'SON'], rotation = 90)
                if i == 0 and j == 0:
                    ax.set_yticklabels([f.replace('_post_', '_in/post_').replace('_in_', '_in/post_') for f in factor_list])
                ax.set_title(key)

                if i == 0 and j == 0:
                    cax = fig.add_axes([0.1, 0.035, 0.8, 0.012])
                    cbar = plt.colorbar(cf, cax = cax, orientation = 'horizontal', label = 'Spearman correlation between predictor and predictor\'s SHAP')

                #    cax = fig.add_axes([0.1, -0.025, 0.8, 0.012])
                #    cbar = plt.colorbar(cf2, cax = cax, orientation = 'horizontal', label = 'Spearman correlation between predictor and predictor\'s SHAP')

                ax.text(-0.2, 1.02, lab[i + j*len(self.region_list) + len(self.region_list)*2], fontweight = 'bold', transform = ax.transAxes)

        fig.savefig(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'{self.prefix}_{self.extent}_{self.name}_{self.heat_wave_thres}_{self.hot_and_dry_thres}_{self.extreme}_importance.png'), bbox_inches = 'tight', dpi = 600.)
        plt.close(fig)


    def optima_relationship(self):
        """ Takes 20 minutes to run. Relationship between the gap to optimal temperature and the local temperature anomaly at pixel level. """
        rcParams['font.size'] = 5.5
        rcParams['axes.titlesize'] = 5.5
        lab = 'abcdefghijklmnopqrst'

        all_coefs = pd.DataFrame(np.nan, 
                                 index = pd.MultiIndex.from_product([['Slope', 'Rho', 'Gap', 'Slope_p', 'Rho_p', 'Diff_p'], ['DJF', 'MAM', 'JJA', 'SON']]),
                                 columns = pd.MultiIndex.from_product([self.region_list, ['tmax', 'tmin']]))

        for season in ['DJF', 'MAM', 'JJA', 'SON']:

            fig, axes = plt.subplots(len(self.region_list), 4, figsize = (6.5, 6.5), sharex = True, sharey = True)
            fig.subplots_adjust(hspace = 0.05, wspace = 0.1)
            for j, region in enumerate(self.region_list):
                print(season, region)

                fid_list = modis_luc_city_groups[region]

                # Read the seasonal mean temperatures and optimal temperatures per pixel
                h = pd.HDFStore(os.path.join(path_out, 'veg', f'{self.prefix}_{self.extent}_{self.name}_optimalT.h5'), mode = 'r')
                temp_optim = pd.DataFrame({'tmax': h.select('optima_tmax').loc[:, 0], 'tmin': h.select('optima_tmin').loc[:, 0]})
                temp_optim = temp_optim.loc[fid_list, :]
                h.close()

                h2 = pd.HDFStore(os.path.join(path_out, 'clim', f'seasonal_average_{self.extent}_{self.heat_wave_thres}_{self.hot_and_dry_thres}.h5'), 'r')
                temp_clim = pd.DataFrame({'daymet': h2.select('daymet').loc[season, :].loc[fid_list, :].stack(), 
                                          'topowx': h2.select('topowx').loc[season, :].loc[fid_list, :].stack(), 
                                          'yyz': h2.select('yyz').loc[season, :].loc[fid_list, :].stack()}).unstack()
                temp_clim.columns = temp_clim.columns.reorder_levels([1, 0])
                temp_clim = temp_clim.stack()
                temp_clim.index.names = ['fid', 'row', 'col', 'use']
                h2.close()

                # Calculate the gap to optimal temperature
                gap_to_optim = temp_clim.copy()
                for use in ['daymet', 'topowx', 'yyz']:
                    temp_temp_clim = temp_clim.loc[(slice(None), slice(None), slice(None), use), :]
                    temp_temp_clim.index = temp_temp_clim.index.droplevel(3)
                    temp = temp_temp_clim - temp_optim
                    temp = temp.loc[temp_temp_clim.index, :] # make sure the index match
                    gap_to_optim.loc[(slice(None), slice(None), slice(None), use), :] = temp.values
                gap_to_optim = gap_to_optim.dropna(axis = 0, how = 'any')

                del temp_optim, temp_clim

                # Read the local temperature anomalies  
                temp_tmax = {}
                temp_tmin = {}
                for i, use in enumerate(['daymet', 'topowx', 'yyz']):
                    h3 = pd.HDFStore(os.path.join(path_out, 'clim', f'{self.prefix}_tmax_{self.extent}_{use}_{self.heat_wave_thres}_{self.hot_and_dry_thres}.h5'), 'r')
                    h4 = pd.HDFStore(os.path.join(path_out, 'clim', f'{self.prefix}_tmin_{self.extent}_{use}_{self.heat_wave_thres}_{self.hot_and_dry_thres}.h5'), 'r')
                    temp_tmax[use] = h3.select('heat_wave').loc[fid_list, ['in_event', 'post_event']]
                    temp_tmax[use] = temp_tmax[use].loc[temp_tmax[use].index.get_level_values('end').month.map(month_to_season) == season, :]
                    temp_tmax[use] = temp_tmax[use].rename({'in_event': 'dtmax_in_event', 'post_event': 'dtmax_post_event'}, axis = 1).stack()

                    temp_tmin[use] = h4.select('heat_wave').loc[fid_list, ['in_event', 'post_event']]
                    temp_tmin[use] = temp_tmin[use].loc[temp_tmin[use].index.get_level_values('end').month.map(month_to_season) == season, :]
                    temp_tmin[use] = temp_tmin[use].rename({'in_event': 'dtmin_in_event', 'post_event': 'dtmin_post_event'}, axis = 1).stack()
                    h3.close()
                    h4.close()
                temp_tmax = pd.DataFrame(temp_tmax).unstack()
                temp_tmax.columns = temp_tmax.columns.reorder_levels([1, 0])
                temp_tmax = temp_tmax.stack()
                temp_tmax.index.names = ['fid', 'start', 'end', 'row', 'col', 'use']
                temp_tmin = pd.DataFrame(temp_tmin).unstack()
                temp_tmin.columns = temp_tmin.columns.reorder_levels([1, 0])
                temp_tmin = temp_tmin.stack()
                temp_tmin.index.names = ['fid', 'start', 'end', 'row', 'col', 'use']

                # Match the seasonal gaps to local temperature anomalies
                data_collect = pd.DataFrame(np.nan, index = temp_tmax.index, columns = ['dtmax_in_event', 'dtmax_post_event', 'dtmin_in_event', 'dtmin_post_event', 'gap_to_optim_tmax', 'gap_to_optim_tmin'])
                data_collect.loc[:, ['dtmax_in_event', 'dtmax_post_event']] = temp_tmax
                data_collect.loc[:, ['dtmin_in_event', 'dtmin_post_event']] = temp_tmin

                dates = data_collect.reset_index()[['start', 'end']].drop_duplicates()
                for _, row in tqdm(dates.iterrows()):
                    filt = (data_collect.index.get_level_values('start') == row['start']) & \
                           (data_collect.index.get_level_values('end'  ) == row['end'  ])
                    temp = pd.DataFrame(np.nan, 
                                        index = data_collect.loc[filt, :].index.droplevel(['start', 'end']), 
                                        columns = gap_to_optim.columns)
                    filt2 = gap_to_optim.index.intersection(temp.index)
                    temp.loc[filt2, :] = gap_to_optim.loc[filt2, :]

                    data_collect.loc[filt, 'gap_to_optim_tmax'] = temp.loc[:, 'tmax'].values
                    data_collect.loc[filt, 'gap_to_optim_tmin'] = temp.loc[:, 'tmin'].values

                # Plot the gap to optimal temperature against the local temperature anomalies
                for k, tvar in enumerate(['tmax', 'tmin']):
                    for m, rr in enumerate(['in_event', 'post_event']):
                        ax = axes[j, k*2 + m]

                        temp = data_collect.loc[:, [f'd{tvar}_{rr}', f'gap_to_optim_{tvar}']].dropna(axis = 0)

                        x = temp[f'd{tvar}_{rr}']
                        y = temp[f'gap_to_optim_{tvar}']
                        sns.histplot(x = x.values, y = y.values, bins = (np.arange(-4, 4, 0.5), np.arange(-40, 11, 2.5)), ax = ax)

                        slope = [None]*3
                        pval = [None]*3
                        intercept = [None]*3
                        rho = [None]*3
                        pval2 = [None]*3
                        for n, use in enumerate(['daymet', 'topowx', 'yyz']):
                            x_ = x.loc[(slice(None), slice(None), slice(None), slice(None), slice(None), use)].values
                            y_ = y.loc[(slice(None), slice(None), slice(None), slice(None), slice(None), use)].values
                            res = linregress(x_, y_)
                            slope[n] = res.slope
                            pval[n] = res.pvalue
                            intercept[n] = res.intercept
                            rho[n], pval2[n] = spearmanr(x_, y_)
                        slope = np.median(slope)
                        pval = np.median(pval)
                        intercept = np.median(intercept)
                        rho = np.median(rho)
                        pval2 = np.median(pval2)

                        all_coefs.loc[('Slope', season), (region, tvar)] = slope
                        all_coefs.loc[('Rho', season), (region, tvar)] = rho
                        all_coefs.loc[('Slope_p', season), (region, tvar)] = pval
                        all_coefs.loc[('Rho_p', season), (region, tvar)] = pval2

                        ax.plot([-4, 4], slope * np.array([-4,4]) + intercept, '-k')
                        if pval <= 0.05:
                            kwargs = {'bbox': dict(facecolor = 'none', edgecolor = 'k', pad = 1, lw = 0.5)}
                        else:
                            kwargs = {}
                        ax.text(0.05, 0.3, f'{slope:.3f}', transform = ax.transAxes, color = 'k', **kwargs)
                        if pval2 <= 0.05:
                            kwargs = {'bbox': dict(facecolor = 'none', edgecolor = 'r', pad = 1, lw = 0.5)}
                        else:
                            kwargs = {}
                        ax.text(0.05, 0.15, f'{rho:.3f}', transform = ax.transAxes, color = 'r', **kwargs)

                        if (k == 0) and (m == 0):
                            ax.set_ylabel(f'{region}\nGap to optim {tvar}')
                        else:
                            ax.set_ylabel(f'Gap to optim {tvar}')
                        if j == (len(self.region_list) - 1):
                            ax.set_xlabel(f'd{tvar}_{rr}')

                        ax.text(0.05, 0.92, lab[(k*2 + m)*len(self.region_list) + j], transform = ax.transAxes, fontweight = 'bold')


                for m, rr in enumerate(['in_event', 'post_event']):
                    avg_gap_max = [None]*3
                    avg_gap_min = [None]*3
                    pval3 = [None]*3

                    for n, use in enumerate(['daymet', 'topowx', 'yyz']):
                        temp = data_collect.loc[(slice(None), slice(None), slice(None), slice(None), slice(None), use), 
                                                ['gap_to_optim_tmax', 'gap_to_optim_tmin']].dropna(axis = 0)
                        x = temp['gap_to_optim_tmax'].values
                        y = temp['gap_to_optim_tmin'].values
                        avg_gap_max[n] = np.mean(x)
                        avg_gap_min[n] = np.mean(y)
                        res = wilcoxon(x, y)
                        pval3[n] = res.pvalue
                    avg_gap_max[n] = np.median(avg_gap_max)
                    avg_gap_min[n] = np.median(avg_gap_min)
                    pval3 = np.median(pval3)
                    all_coefs.loc[('Gap'   , season), (region, 'tmax')] = np.median(avg_gap_max)
                    all_coefs.loc[('Gap'   , season), (region, 'tmin')] = np.median(avg_gap_min)
                    all_coefs.loc[('Diff_p', season), (region, 'tmin')] = np.median(pval3)

            fig.savefig(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'pixel_optima_relationship_{self.prefix}_{setup.extent}_{setup.name}_{setup.heat_wave_thres}_{setup.hot_and_dry_thres}_{season}.png'), dpi = 600., bbox_inches = 'tight')
            plt.close(fig)

        all_coefs.to_csv(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'pixel_optima_relationship_{self.prefix}_{setup.extent}_{setup.name}_{setup.heat_wave_thres}_{setup.hot_and_dry_thres}.csv'))


class Fig3(Plotter):
    def __init__(self, extreme, region_list = None, n_estimators = None, subset_pred = None):
        super().__init__(extreme)

        # override if new values are provided
        if region_list is not None:
            self.region_list = region_list
        if n_estimators is not None:
            self.n_estimators = n_estimators
        if subset_pred is not None:
            self.subset_pred = subset_pred


    def rr_rel(self, no_crop = False):
        """ Plot the dependence of urban & rural resistance and recovery on heat wave duration and intensity (2 covariates),
            and plot the urban-rural difference. """
        if no_crop:
            suffix = '_nocrop'
        else:
            suffix = ''

        summary_varname_diff, summary_varname_urban, summary_varname_rural = self._get_summary()
        predictors, _, _ = self._get_predictors()

        temp = predictors.index.intersection(summary_varname_urban.index).intersection(summary_varname_rural.index)
        predictors = predictors.loc[temp, :]
        summary_varname_urban = summary_varname_urban.loc[temp, :]
        summary_varname_rural = summary_varname_rural.loc[temp, :]
        summary_varname_diff  = summary_varname_diff.loc[temp, :]

        season_ind = pd.DatetimeIndex(summary_varname_urban.index.get_level_values('end')).month.map(month_to_season)

        lab = 'abcdefghijklmnopqrstuvwxyz'
        rcParams['font.size'] = 5.5
        rcParams['axes.titlesize'] = 5.5
        map_args = {
            'diff': {'norm': BoundaryNorm([-0.06, -0.03, -0.015, 0, 0.015, 0.03, 0.06], 256, extend = 'both'), 'cmap': cmap_div(thres = 0.1)},
            'abs' : {'norm': BoundaryNorm([-0.48, -0.36, -0.24, -0.12, -0.06, 0, 0.06, 0.12, 0.24, 0.36, 0.48], 256, extend = 'both'), 'cmap': cmap_div(thres = 0.1)}
        }

        fig, axes = plt.subplots(6, 4, figsize = (6.5, 8), sharex = True, sharey = True)
        fig.subplots_adjust(hspace = 0.01, wspace = 0.1)
        for i, data in enumerate([summary_varname_urban, summary_varname_rural, summary_varname_diff]):
            for j, stat in enumerate(['Resistance', 'Recovery']):
                for s, season in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
                    ax = axes[i*2 + j, s]

                    x = np.linspace(0.5, 9.5, 10)
                    x_mid = [f'{xx:.0f}' for xx in 0.5 * (x[1:] + x[:-1])]
                    y = np.linspace(-0.25, 5.25, 12)
                    y_mid = [f'{yy:.0f}' for yy in 0.5 * (y[1:] + y[:-1])]
                    z = np.full((len(y)-1, len(x)-1), fill_value = np.nan)
                    for a, b in it.product(range(len(x)-1), range(len(y)-1)):
                        filt = (predictors['event_duration' ].values >= x[a]) & (predictors['event_duration' ].values < x[a+1]) & \
                               (predictors['event_intensity'].values >= y[b]) & (predictors['event_intensity'].values < y[b+1]) & \
                               (season_ind == season)
                        z[b, a] = data.loc[filt, stat].groupby('use').mean().median()
                    if i == 2:
                        cf = ax.imshow(z[::-1, :], **map_args['diff'])
                    else:
                        cf = ax.imshow(z[::-1, :], **map_args['abs'])
                    ax.set_xticks(range(0, len(x_mid), 1))
                    ax.set_xticklabels(x_mid)
                    ax.set_yticks(range(0, len(y_mid), 2))
                    ax.set_yticklabels(y_mid[::-1][::2])
                    ax.text(-0.3, 0.95, lab[i*4 + j *8 + s], transform = ax.transAxes, fontweight = 'bold')
                    ax.axvline(3, ls = ':', color = 'k', lw = 0.5)
                    ax.axhline(6, ls = ':', color = 'k', lw = 0.5)

                    if (i == 0) & (j == 0):
                        ax.set_title(season)
                    if s == 0:
                        if i == 0:
                            titletext = f'{stat} urban'
                        elif i == 1:
                            titletext = f'{stat} rural'
                        else:
                            titletext = f'$\Delta${stat}' + '$_{urban-rural}$'
                        ax.set_ylabel(f'{titletext}\nIntensity ($^o$C)')

                    if (i == 2) & (j == 1):
                        ax.set_xlabel('Duration (months)')
                    else:
                        ax.set_xticklabels([])
                    
                    if j == 0:
                        if i == 0:
                            cax = fig.add_axes([0.9, 0.38, 0.01, 0.5])
                            plt.colorbar(cf, cax = cax, orientation = 'vertical', ticks = map_args['abs']['norm'].boundaries)
                        elif i == 2:
                            cax = fig.add_axes([0.9, 0.1, 0.01, 0.25])
                            plt.colorbar(cf, cax = cax, orientation = 'vertical')

        fix = Setup().format_prefix().replace(f'{Setup().use}_', '')
        fig.savefig(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'percity_spatial_avg_summary_{fix}_{self.extreme}_fig3_S1.png'), dpi = 600., bbox_inches = 'tight')
        plt.close(fig)


    def rr_urban_minus_rural(self):
        def _subplotter(ax, temp, hue_order):
            sns.boxplot(data = temp, x = 'season', hue = 'duration', y = 'stat', 
                        order = ['DJF', 'MAM', 'JJA', 'SON'], hue_order = hue_order, 
                        palette = ['#fc9272', '#9ecae1'], ax = ax, showfliers = False, whis = [10, 90],
                        linewidth = 1)

            diff = temp.groupby(['use', 'season', 'duration']).apply(lambda x: wilcoxon(x['stat'].values).pvalue)
            ind = pd.MultiIndex.from_product([['DJF', 'MAM', 'JJA', 'SON'], hue_order])
            diff = diff.groupby(['season', 'duration']).median().loc[ind]

            top = temp.groupby(['season', 'duration']).apply(lambda x: np.percentile(x['stat'], 90)).loc[ind]
            ylim = ax.get_ylim()
            yrng = ylim[1] - ylim[0]

            xticks = ax.get_xticks()
            for t in xticks:
                ax.text(t-0.25, top.values[t*2] + yrng * 0.08, f'{diff.values[t*2]:.3f}', 
                        horizontalalignment = 'center', verticalalignment = 'top')
                ax.text(t+0.25, top.values[t*2+1] + yrng * 0.08, f'{diff.values[t*2+1]:.3f}',
                        horizontalalignment = 'center', verticalalignment = 'top')

            ax.axhline(0., ls = '--', lw = 0.5)
            ax.legend([], [], frameon=False)
            ax.set_ylabel('')
            ax.set_xlabel('')
            res1 = ax.add_patch(Rectangle([0,0], width = 0, height = 0, facecolor = '#fc9272'))
            res2 = ax.add_patch(Rectangle([0,0], width = 0, height = 0, facecolor = '#9ecae1'))
            ax.legend([res1, res2], hue_order, ncol = 2, loc = 'lower right')


        lab = 'abcdefghijklmnopqrstuvwxyz'
        rcParams['font.size'] = 5.5
        rcParams['axes.titlesize'] = 5.5
        clist = ['#0000ff', '#87cefa', '#ff0000', '#e69b00']


        summary_varname_diff, _, summary_varname_rural = self._get_summary()
        predictors, _, _ = self._get_predictors()

        temp = predictors.index.intersection(summary_varname_diff.index).intersection(summary_varname_diff.index)
        predictors = predictors.loc[temp, :]
        summary_varname_rural = summary_varname_rural.loc[temp, :]
        summary_varname_diff  = summary_varname_diff.loc[temp, :]

        season_ind = pd.DatetimeIndex(summary_varname_diff.index.get_level_values('end')).month.map(month_to_season)


        s = Fig2(self.extreme)
        s.stagedata(calc = False)
        intersect_index = s.shap_values_diff.index.intersection(predictors.index)
        factor = 'impervious_frac'
        shap_values_diff = s.shap_values_diff.loc[intersect_index, :]

        fig, axes = plt.subplots(2, 4, figsize = (6.5, 3), sharex = False, sharey = False)
        fig.subplots_adjust(hspace = 0.05, wspace = 0.05)
        for i, stat in enumerate(['Resistance', 'Recovery']):
            for j, side in enumerate(['event_duration', 'event_intensity']):
                ax = axes[0, j*2 + i]

                h = [None] * 4
                for k, season in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
                    x = predictors.loc[season_ind == season, side].values
                    y = summary_varname_diff.loc[season_ind == season, stat].values
                    ind = np.argsort(x)
                    spl = UnivariateSpline(x[ind], y[ind])
                    x2 = np.linspace(np.percentile(x, 5), np.percentile(x, 95), 21)
                    y2 = spl(x2)
                    h[k], = ax.plot(x2, y2, '-', color = clist[k])

                if (i == 0) & (j == 0):
                    ax.set_ylabel('Urban - rural')
                else:
                    ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.set_xlabel(side)
                ax.set_title(f'$\Delta${stat}')
                if (i == 0) & (j == 0):
                    ax.legend(h, ['DJF', 'MAM', 'JJA', 'SON'])
                if j == 0:
                    ax.set_xlim([1, 9])
                else:
                    ax.set_xlim([0, 6])
                ax.axhline(0., ls = '--', lw = 0.5, color = 'k')
                ax.set_ylim([-0.08, 0.06])

                ax = axes[1, j*2 + i]

                h = [None] * 4
                for k, season in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
                    # note this is the test set
                    contrib = shap_values_diff.loc[:, (slice(None), stat, season, factor)]
                    contrib.columns = contrib.columns.droplevel([1, 2, 3])
                    contrib.columns.names = ['Region']
                    contrib = contrib.stack().dropna()
                    contrib = contrib.to_frame('stat')
                    contrib['side'] = predictors.loc[contrib.index.droplevel(4), side].values
                    x = contrib['side'].values
                    y = contrib['stat'].values
                    ind = np.argsort(x)
                    spl = UnivariateSpline(x[ind], y[ind])
                    x2 = np.linspace(np.percentile(x, 5), np.percentile(x, 95), 21)
                    y2 = spl(x2)
                    h[k], = ax.plot(x2, y2, '-', color = clist[k])

                if (i == 0) & (j == 0):
                    ax.set_ylabel(f'Contrib. {factor}')
                else:
                    ax.set_yticklabels([])
                ax.set_xlabel(side)
                ax.axhline(0., ls = '--', lw = 0.5, color = 'k')
                if j == 0:
                    ax.set_xlim([1, 9])
                else:
                    ax.set_xlim([0, 6])
                ax.set_ylim([-0.035, 0.019])

        fix = Setup().format_prefix().replace(f'{Setup().use}_', '')
        fig.savefig(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'percity_spatial_avg_summary_{fix}_{self.extreme}_fig3.png'), dpi = 600., bbox_inches = 'tight')
        plt.close(fig)


    def rr_other_factors(self):
        def _subplotter(ax, temp, hue_order):
            sns.boxplot(data = temp, x = 'season', hue = 'duration', y = 'stat', 
                        order = ['DJF', 'MAM', 'JJA', 'SON'], hue_order = hue_order, 
                        palette = ['#fc9272', '#9ecae1'], ax = ax, showfliers = False, whis = [10, 90],
                        linewidth = 1)

            diff = temp.groupby(['use', 'season', 'duration']).apply(lambda x: wilcoxon(x['stat'].values).pvalue)
            ind = pd.MultiIndex.from_product([['DJF', 'MAM', 'JJA', 'SON'], hue_order])
            diff = diff.groupby(['season', 'duration']).median().loc[ind]

            top = temp.groupby(['season', 'duration']).apply(lambda x: np.percentile(x['stat'], 90)).loc[ind]
            ylim = ax.get_ylim()
            yrng = ylim[1] - ylim[0]

            xticks = ax.get_xticks()
            for t in xticks:
                ax.text(t-0.25, min(top.values[t*2] + yrng * 0.12, ylim[1] - yrng*0.03), f'{diff.values[t*2]:.3f}', 
                        horizontalalignment = 'center', verticalalignment = 'top')
                ax.text(t+0.25, min(top.values[t*2 + 1] + yrng * 0.12, ylim[1] - yrng*0.03), f'{diff.values[t*2+1]:.3f}',
                        horizontalalignment = 'center', verticalalignment = 'top')

            ax.axhline(0., ls = '--', lw = 0.5)
            ax.legend([], [], frameon=False)
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_ylim(ylim[0] - yrng * 0.1, ylim[1] + yrng * 0.1)


        lab = 'abcdefghijklmnopqrstuvwxyz'
        rcParams['font.size'] = 5.5
        rcParams['axes.titlesize'] = 5.5

        summary_varname_diff, _, summary_varname_rural = self._get_summary()
        predictors, _, _ = self._get_predictors()

        temp = predictors.index.intersection(summary_varname_diff.index)
        predictors = predictors.loc[temp, :]
        summary_varname_diff = summary_varname_diff.loc[temp, :]

        season_ind = pd.DatetimeIndex(summary_varname_diff.index.get_level_values('end')).month.map(month_to_season)


        s = Fig2(self.extreme)
        s.stagedata(calc = False)
        factor_list = ['dtmax', 'dtmin', 'Deciduous forest', 'Shrub', 'Grass', 'Crop', 'Wetland']

        fig, axes = plt.subplots(len(factor_list), 2, figsize = (6.5, 8), sharex = True, sharey = False)
        for k, stat in enumerate(['Resistance', 'Recovery']):
            for i, factor in enumerate(factor_list):
                if factor in ['dtmax', 'dtmin', 'spi', 'vpd']:
                    if stat == 'Recovery':
                        factor_ = factor + '_post_event'
                    else:
                        factor_ = factor + '_in_event'
                else:
                    factor_ = factor

                ax = axes[i, k]
                if factor in ['dtmax', 'dtmin', 'spi', 'vpd']:
                    if stat == 'Recovery':
                        factor_ = factor + '_post_event'
                    else:
                        factor_ = factor + '_in_event'
                else:
                    factor_ = factor
                # note this is the test set
                intersect_index = s.shap_values_diff.index.intersection(predictors.index)
                contrib = s.shap_values_diff.loc[intersect_index, (slice(None), stat, slice(None), factor_)]
                contrib.columns = contrib.columns.droplevel([1,3])
                contrib.columns.names = ['Region', 'Season']
                contrib = contrib.stack().stack().dropna()
                season_ = contrib.index.get_level_values(4)
                contrib.index = contrib.index.droplevel(4)
                contrib = contrib.to_frame('stat')
                contrib['season'] = season_
                contrib['duration'] = ((predictors.loc[contrib.index.droplevel(4), 'event_duration'] > 4) & \
                    (predictors.loc[contrib.index.droplevel(4), 'event_intensity'] > 2)).map({True: 'Long & intense', False: 'Otherwise'}).values

                if factor == 'dtmax':
                    ax.set_ylim([-0.015, 0.01])
                elif factor == 'dtmin':
                    ax.set_ylim([-0.01, 0.02])
                elif factor == 'Deciduous forest':
                    ax.set_ylim([-0.017, 0.005])
                elif factor == 'Shrub':
                    ax.set_ylim([-0.001, 0.001])
                elif factor == 'Grass':
                    ax.set_ylim([-0.005, 0.01])
                elif factor == 'Crop':
                    ax.set_ylim([-0.02, 0.02])

                _subplotter(ax, contrib, ['Long & intense', 'Otherwise'])
                if k == 0:
                    if factor in ['dtmax', 'dtmin', 'spi', 'vpd']:
                        ax.set_ylabel(f'{factor}_in/post_event')
                    else:
                        ax.set_ylabel(f'{factor}')
                ax.text(0.05, 1.05, lab[k*len(factor_list) + i], transform = ax.transAxes, fontweight = 'bold')


        res1 = ax.add_patch(Rectangle([0,0], width = 0, height = 0, facecolor = '#fc9272'))
        res2 = ax.add_patch(Rectangle([0,0], width = 0, height = 0, facecolor = '#9ecae1'))
        ax.legend([res1, res2], ['Long & intense', 'Otherwise'], ncol = 2, loc = [-0.5, -0.5])
        fix = Setup().format_prefix().replace(f'{Setup().use}_', '')
        fig.savefig(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'percity_spatial_avg_summary_{fix}_{self.extreme}_fig4_S2.png'), dpi = 600., bbox_inches = 'tight')
        plt.close(fig)


if __name__ == '__main__':
    setup = Setup()
    setup.set_use('yyz')
    print(setup)
    extreme = 'heat_wave'

    # Retrieve data; takes a long time
    #s = Fig1(extreme)
    #s.get_city_summary(no_crop = True)
    #s.get_luc_summary()

    s = Fig1(extreme)
    s.plot_diff(no_crop = True)
    #s.plot_city_summary(no_crop = True)
    #s.table_diff2(no_crop = True)

    #s = Fig2(extreme)
    #s.stagedata(calc = True)
    #s.stagedata(calc = False)
    #s.urban_minus_rural()
    #s.actual()
    #s.interaction('impervious_frac', 'dtmax_in/post_event')
    #s.interaction('impervious_frac', 'dtmin_in/post_event')

    #s = Validation(extreme)
    #s.metrics()
    #s.corr_diff()

    #s = Fig3(extreme)
    #s.rr_rel()
    #s.rr_urban_minus_rural()
    #s.rr_other_factors()