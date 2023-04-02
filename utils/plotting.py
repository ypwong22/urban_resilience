import numpy as np
from matplotlib import colors
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os
import pandas as pd
import shapefile as shp  # Requires the pyshp package
from sklearn.neighbors import KernelDensity
from scipy.stats import norm, gaussian_kde, wilcoxon
import warnings
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
import seaborn as sns

from .analysis import *
from .constants import *
from .paths import *
from .get_monthly_data import Setup


def cmap_div(rev = False, thres = 0.2):
    """
    Colormap with clear division between positive and negative values in the middle.
    """
    c = cm.get_cmap('Reds', 256)
    new1 = c(np.linspace(1, thres, 128))
    c = cm.get_cmap('Blues', 256)
    new2 = c(np.linspace(thres, 1., 128))
    newcmp = colors.ListedColormap(np.concatenate([new1, new2], axis = 0))
    if rev:
        newcmp.colors = newcmp.colors[::-1, :]
    return newcmp


def cmap_discrete(n, map):
    """ Colormap discretized to categories """
    c   = cm.get_cmap(map, 256)
    new = colors.ListedColormap(c(np.arange(1/n, 1 - 0.9/n , 1/n)))
    return new


def cmap_clist(n, map):
    """ Colormap pick colors """
    c   = cm.get_cmap(map, 256)
    new = [c(int(i)) for i in np.linspace(0, 256, n)]
    return new


def add_core_boundary(ax, fid, color = '#dd1c77', **kwargs):
    """ Plot the x-y series that consist of the poly vertices of the urban core. """
    sf = shp.Reader(os.path.join(path_intrim, 'urban_mask', 
                                 'US_urbanCluster_500km_Polyline.shp'))
    for shape in sf.shapeRecords():
        if shape.record[1] == fid:
            x = [i[0] for i in shape.shape.points[:]]
            y = [i[1] for i in shape.shape.points[:]]
            ax.plot(x, y, '-', color = color, lw = 0.5, **kwargs)
    sf.close()
    return ax


def ax_histogram(ax, vector, bins, dist = 'norm',
                 show = 'both',
                 args_hist = {'color': '#bdbdbd', 'edgecolor': '#636363'},
                 args_line = {'color': '#2b8cbe'}):
    """
    Plot a histogram in panel with fitted Gaussian distribution.
    Parameters
    ----------
      vector: 1-d array
          The data set.
      bins: int or a sequence of increasing numbers
          The same as input into **numpy.histogram**.
          Defines the number of bins or the bin edges, from left-most to
          right-most.
      dist: str
          If *norm*, plot fitted normal distribution.
          If *kde*, plot fitted Gaussian kernal density.
      show: str
          If *both* (default), plot both the histogram bars and fitted
          distribution.
          If *bar*, plot only the histogram bars.
          If *line*, plot only the fitted distribution using lines.
    """
    vector = vector[~np.isnan(vector)]
    hist, bin_edges = np.histogram(vector, bins = bins, density = True)

    h = []

    if (show == 'bar') | (show == 'both'):
        h1 = ax.bar(bin_edges[:-1], hist,
                    width = bin_edges[1:] - bin_edges[:-1],
                    align = 'edge', **args_hist)
        h.append(h1)
 
        # convenience to insure edge is not transparent.
        if 'alpha' in args_hist:
            del args_hist['alpha']
            args_hist['facecolor'] = 'None'
            ax.bar(bin_edges[:-1], hist,
                   width = bin_edges[1:] - bin_edges[:-1],
                   align = 'edge', **args_hist)

    if (show == 'line') | (show == 'both'):
        x = np.linspace(bin_edges[0], bin_edges[-1], 100)

        if dist == 'norm':
            mean, std = norm.fit(vector)
            h2, = ax.plot(x, norm.pdf(x, mean, std), **args_line)
        elif dist == 'kde':
            kde = KernelDensity(bandwidth = np.std(vector)/5, kernel='gaussian')
            kde.fit(vector.reshape(-1,1))
            prob = np.exp(kde.score_samples(x.reshape(-1,1)))
            h2, = ax.plot(x, prob, '-', **args_line)
        h.append(h2)

    return h


def ax_scatter_density(ax, x, y, cmap = 'jet'):
    """Plot 2D scatter plot, colored by local density."""
    loc = np.vstack([x.ravel(), y.ravel()])
    f = gaussian_kde(loc)
    density = f(loc)
    h = ax.scatter(x, y, c = density, cmap = cmap,
                   edgecolor = None)
    return h


def ax_colored_density(ax, x, y, cmap = 'jet'):
    """Plot 2D Gaussian kde density plot."""
    loc = np.vstack([x.ravel(), y.ravel()])
    f = gaussian_kde(loc)

    x_range = np.linspace(np.min(x) - (np.max(x) - np.min(x))/10,
                          np.max(x) + (np.max(x) - np.min(x))/10, 100)
    y_range = np.linspace(np.min(y) - (np.max(y) - np.min(y))/10,
                          np.max(y) + (np.max(y) - np.min(y))/10, 100)
    x_mesh, y_mesh = np.meshgrid(x_range, y_range)
    z_mesh = f(np.vstack([x_mesh.ravel(), y_mesh.ravel()]))

    cf = ax.contourf(x_range, y_range, 
                     z_mesh.T.reshape(x_mesh.shape), cmap = cmap)
    return cf


def ax_shade(ax, time, matrix, ts_label = '', ts_col = 'red',
             shade_col = 'red', alpha = 0.2, skipna = False):
    """
    Plot a shaded ensemble of time series.
    """
    if skipna:
        ts_min = np.nanmin(matrix, axis = 1)
        ts_mean = np.nanmean(matrix, axis = 1)
        ts_max = np.nanmax(matrix, axis = 1)
    else:
        ts_min = np.min(matrix, axis = 1)
        ts_mean = np.mean(matrix, axis = 1)
        ts_max = np.max(matrix, axis = 1)

    hl, = ax.plot(time, ts_mean, '-', color = ts_col, linewidth = 2,
                  label = ts_label)
    ax.plot(time, ts_min, '--', color = shade_col, linewidth = 1)
    ax.plot(time, ts_max, '--', color = shade_col, linewidth = 1)
    ax.fill_between(time, ts_min, ts_max, where = ts_max > ts_min,
                    facecolor = shade_col, alpha = alpha)
    hfill, = ax.fill(np.nan, np.nan, facecolor = shade_col, alpha = alpha)
    return hl, hfill


class MapOfColors(Setup):
    def __init__(self, style, df):
        super().__init__()

        if style == 'point':
            if isinstance(df, pd.DataFrame):
                # insure all 85 points are in the index
                for fid in range(85):
                    if not fid in df.index:
                        df.loc[fid, df.columns[0]] = np.nan
                        df.loc[fid, df.columns[1]] = True
                warnings.warn('Treating the second column of pd.DataFrame as True/False indicator of significance')
                self.series = df.iloc[:, 0].values
                self.series_sig = df.iloc[:, 1].values.astype(bool)
            elif isinstance(df, pd.Series):
                # insure all 85 points are in the index
                for fid in range(85):
                    if not fid in df.index:
                        df.loc[fid] = np.nan
                self.series = df.values
                self.series_sig = np.full(len(self.series), True) # only to pick the plot style with borders
            else:
                raise 'df must be a pandas.DataFrame or pandas.Series object if plotting scatter plot'
        elif style == 'pie':
            if not isinstance(df, pd.DataFrame):
                raise 'df must be a pandas.DataFrame object if plotting pie plot'
            else:
                self.data = df
        else:
            raise f'style = {style} is not implemented'

        self.style = style # point / bars on each city

        self.trans_map = ccrs.AlbersEqualArea(central_longitude = -100, central_latitude = 35)
        self.trans_data = ccrs.PlateCarree()

    def _get_coord(self, fid):
        coord = pd.read_csv(os.path.join(path_intrim, 'Daymet', 'coords', f'{self.extent}_{fid}.csv'))
        lat = float(coord['lat'].mean())
        lon = float(coord['lon'].mean())
        return lat, lon, coord.shape[0]

    def _get_coords(self):
        xvec = np.empty(85)
        yvec = np.empty(85)
        size = np.empty(85)
        for fid in range(85):
            lat, lon, size[fid] = self._get_coord(fid)
            x, y = self.trans_map.transform_point(lon, lat, self.trans_data)
            xvec[fid] = x
            yvec[fid] = y
        return xvec, yvec, size

    def _apanel(self, ax, xvec, yvec, size, point_scale, **norm_args):
        ax.coastlines(lw = 0.5)
        ax.set_extent([-124, -73, 24.69, 52], crs = self.trans_data)
        cf = ax.scatter(xvec[self.series_sig], yvec[self.series_sig], c = self.series[self.series_sig],
                        edgecolors = 'k', linewidths = 1.5 * point_scale, s = np.sqrt(size[self.series_sig] + 1) * point_scale, alpha = 0.7,
                        transform = self.trans_map, **norm_args)
        ax.scatter(xvec[~self.series_sig], yvec[~self.series_sig], c = self.series[~self.series_sig],
                   edgecolors = 'none', s = np.sqrt(size[~self.series_sig] + 1) * point_scale, linewidths = 0.5 * point_scale, alpha = 0.7,
                   transform = self.trans_map, **norm_args)
        return cf

    def _inset_bar(self, ax, **kwargs):
        """ Count the number of positive & negative pts, and positive & negative significant pts. """
        def _abar(ax, series, series_p, **kwargs):
            bar1 = np.sum(series > 0)
            bar1_insig = np.sum((series > 0) & (~series_p))
            bar2 = np.sum(series < 0)
            bar2_insig = np.sum((series < 0) & (~series_p                                                                                                 ))
            ax.bar([0,0,1,1], [bar1, bar1_insig, bar2, bar2_insig], **kwargs)
            ax.set_xticks([0,1])
            ax.set_xticklabels(['Pos','Neg'])
            ax.tick_params('both', length = 2, pad = 0.5)

            # p-value of Wilcoxon signed rank test of whether the data is symmetric about zero
            """
            res = wilcoxon(series)
            if res.pvalue <= 0.05:
                ax.set_title('p $\leq$ 0.05')
            elif res.pvalue <= 0.1:
                ax.set_title('p $\leq$ 0.1')
            else:
                ax.set_title('p > 0.1')
            """

        if 'color' not in kwargs:
            kwargs['color'] = ['#31a354', '#74c476', '#993404', '#fe9929']

        ax_inset = inset_axes(ax, width = 0.25, height = 0.21,
                              bbox_to_anchor = self.trans_map.transform_point(-95, 53.1, self.trans_data),
                              bbox_transform = ax.transData, borderpad = 0)
        _abar(ax_inset, self.series, self.series_sig, **kwargs)

    def _apie(self, ax, xvec, yvec, legend, legend_args, **kwargs):
        ax.coastlines(lw = 0.5)
        ax.set_extent([-124, -75, 24.69, 49.67], crs = self.trans_data)

        for fid, x, y in zip(range(85), xvec, yvec):
            if not fid in self.data.index:
                continue

            x0 = x
            y0 = y

            shift_list = {
                28: [-200000,   70000], 
                29: [-200000, -100000],
                51: [-100000,  100000],
                58: [-100000, -100000],
                30: [-150000, -100000], 
                76: [-200000,  -50000],
                73: [ 100000, -100000],
                72: [ 100000, -200000],
                79: [-200000,   50000],
                81: [-200000,  -50000],
                82: [-200000, -150000],
                83: [ 200000, -240000],
                80: [ 200000, -100000],
                77: [ 200000,       0],
                74: [ 300000,  300000],
                12: [ 0     ,  200000],
                10: [ 0     ,  200000],
                 9: [ 0     ,  400000],
                13: [  50000,  200000],
                 8: [ 0     ,  150000],
                17: [ 0     ,  150000], 
                40: [ 150000, -100000],
                36: [-100000,       0],
                36: [ 400000,       0],
                22: [ 300000,       0],
                14: [ 200000, -200000],
                20: [ 0     ,  130000],
                 8: [  10000,  300000],
                32: [ 0     ,  150000],
                56: [ 300000,  0     ],
                43: [-100000,  300000],
                48: [-100000,  150000], 
                60: [-100000, -100000]
            }

            if np.sum(self.data.loc[fid, :].values) > 0:
                if fid in shift_list.keys():
                    dx = shift_list[fid][0]
                    dy = shift_list[fid][1]
                    x = x + dx
                    y = y + dy
                    ax.plot([x0, x], [y0, y], lw = 0.5, color = 'b')
                    ax.scatter(x0, y0, s = 6, color = 'b', zorder = 0)
                ax_inset = inset_axes(ax, width = 0.12, height = 0.12,
                                    bbox_to_anchor = (x, y), loc = 'center',
                                    bbox_transform = ax.transData, borderpad = 0)
                h, _= ax_inset.pie(self.data.loc[fid, :].values, **kwargs)
        if legend: 
            ax.legend(h, self.data.columns, **legend_args)

    def plot(self, inset_bar = True, ax = None, map_args = {}, bar_args = {}, point_scale = 1, legend = True, legend_args = {}, pie_args = {}, annotate = False):
        if ax is None:
            fig, ax = plt.subplots(figsize = (8,6), subplot_kw = {'projection': self.trans_map})
            return_fig = True
        else:
            self.trans_map = ax.projection
            return_fig = False

        xvec, yvec, size = self._get_coords()

        if self.style == 'point':
            cf = self._apanel(ax, xvec, yvec, size, point_scale, **map_args)
            if inset_bar:
                self._inset_bar(ax, **bar_args)
            if annotate:
                for fid in range(85):
                    ax.text(xvec[fid], yvec[fid], str(fid), fontsize = 3, zorder = -1)
            if return_fig:
                return fig, ax, cf
            else:
                return cf

        elif self.style == 'pie':
            self._apie(ax, xvec, yvec, legend, legend_args, **pie_args)
            if annotate:
                new_ax = ax.get_figure().add_axes(ax.get_position())
                new_ax.set_xlim(ax.get_xlim())
                new_ax.set_ylim(ax.get_ylim())
                new_ax.patch.set_alpha(0.)
                new_ax.get_xaxis().set_visible(False)
                new_ax.get_yaxis().set_visible(False)
                for fid in range(85):
                    new_ax.text(xvec[fid], yvec[fid], str(fid), fontsize = 3)

            if return_fig:
                return fig, ax


def map_of_city(array, fid, extent, ax = None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize = (6, 6), subplot_kw = {'projection': crs_daymet})
        return_fig = True
    else:
        return_fig = False
    
    if isinstance(array, xr.DataArray):
        array = da_to_df(array).to_frame('data')
    elif isinstance(array, pd.Series):
        array = array.to_frame('data')
    else:
        array.columns = ['data'] # rename; assume index to be ['row', 'col']

    data = add_grid_coords(fid, array, extent)

    cf = ax.scatter(data['x'], data['y'], c = data['data'], lw = 0., marker = 's', transform = crs_daymet, **kwargs)
    add_core_boundary(ax, fid, 'k')

    if return_fig:
        plt.colorbar(cf, ax = ax, shrink = 0.7)
        return fig, cf
    else:
        return cf


def make_ghost(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_facecolor('none')
    return ax


class TwoDCmap():
    def __init__(self, norm_1, norm_2):
        self.N = 3
        #self.cmap_1 = colors.ListedColormap([(1 - i, 0, 1, 1) for i in np.linspace(0.5/self.N, 1 - 0.5/self.N, N)])
        #self.cmap_2 = colors.ListedColormap([(1, 0, 1 - i, 1) for i in np.linspace(0.5/self.N, 1 - 0.5/self.N, N)])
        self.cmap_1 = colors.ListedColormap(['#fffafa', '#ef85a3', '#e53869'])
        self.cmap_2 = colors.ListedColormap(['#e5eaff', '#61b8ff', '#0087f5'])
        self.norm_1 = norm_1
        self.norm_2 = norm_2

    def gen_colors(self, list_1, list_2):
        clist = np.array([(np.asarray(self.cmap_1(self.norm_1(list_1[i]))) * np.asarray(self.cmap_2(self.norm_2(list_2[i])))) for i in range(len(list_1))])
        return clist

    def draw_legend(self, cax, xlim = None, ylim = None, xticks = None, yticks = None):
        """ Either give lim or ticks """
        if xlim is None:
            xlim = self.norm_1.vmin, self.norm_1.vmax
        if ylim is None:
            ylim = self.norm_2.vmin, self.norm_2.vmax
        if xticks is None:
            xticks = np.linspace(0, 1, self.N + 1) * (xlim[1] - xlim[0]) + xlim[0]
        if yticks is None:
            yticks = np.linspace(0, 1, self.N + 1) * (ylim[1] - ylim[0]) + ylim[0]

        x_val = np.arange(0.5, self.N) / self.N * (xlim[1] - xlim[0]) + xlim[0]
        y_val = np.arange(0.5, self.N) / self.N * (ylim[1] - ylim[0]) + ylim[0]
        x_val, y_val = np.meshgrid(x_val, y_val, indexing = 'xy')
        ind = np.arange(self.N * self.N).reshape(self.N, self.N) / self.N / self.N
        cmap = colors.ListedColormap(self.gen_colors(x_val.reshape(-1), y_val.reshape(-1)))

        #print(cmap.colors)

        cax.imshow(ind, cmap = cmap)
        cax.set_xticks(np.arange(-0.5, self.N + 0.5))
        cax.set_yticks(np.arange(-0.5, self.N + 0.5))
        cax.set_xticklabels([f'{i:.1f}' for i in xticks], rotation = 90)
        cax.set_yticklabels([f'{i:.1f}' for i in yticks])
        cax.set_xlim([-0.5, self.N - 0.5])
        cax.set_ylim([-0.5, self.N - 0.5])