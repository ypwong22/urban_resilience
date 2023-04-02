"""
Fit regressor of pixel-level recovery and resistance using the predictors identified from `monthly_percity_predictors.py`.
"""
from random import Random
import pandas as pd
import xarray as xr
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import norm, pearsonr
import itertools as it
import warnings
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
# import xgboost as xgb
import fasttreeshap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.interpolate import UnivariateSpline
import joblib
import multiprocessing as mp
from time import time

from utils.analysis import *
from utils.constants import *
from utils.paths import *
from utils.extremes import *
from utils.get_monthly_data import *


class PixelData(Setup):
    def __init__(self, use, extreme, season, land_cover, city_list):
        super().__init__(use)

        self.extreme = extreme # 'heat_wave', 'hot_and_dry'
        self.season = season # 'DJF', 'MAM', 'JJA', 'SON'

        self.predictors = None
        self.predictand = None

        self.land_cover = land_cover # The rural land cover to focus on (pixel area > 50%)
        self.city_list = city_list # The cities to focus on for this rural land cover

        self.prefix = 'percity_per_pixel'
        self.suffix = f'{self.extreme}_{self.season}'


    def collect_uhi(self):
        """ local heat island intensity & spi & vpd during/after the event"""
        with pd.HDFStore(os.path.join(path_out, 'clim', f'{self.prefix}_tmax_{self.format_prefix_noveg()}.h5'), mode = 'r') as h:
            tmax = h.select(self.extreme).loc[self.city_list, :].drop('of_season', axis = 1)
            filt = tmax.index.get_level_values('end').month.isin(season_to_month[self.season])
            tmax = tmax.loc[filt, :].rename({'in_event': 'dtmax_in_event', 'post_event': 'dtmax_post_event'}, axis = 1)

        with pd.HDFStore(os.path.join(path_out, 'clim', f'{self.prefix}_tmin_{self.format_prefix_noveg()}.h5'), mode = 'r') as h:
            tmin = h.select(self.extreme).loc[self.city_list, :].drop('of_season', axis = 1)
            filt = tmin.index.get_level_values('end').month.isin(season_to_month[self.season])
            tmin = tmin.loc[filt, :].rename({'in_event': 'dtmin_in_event', 'post_event': 'dtmin_post_event'}, axis = 1)

        with pd.HDFStore(os.path.join(path_out, 'clim', f'{self.prefix}_spi_{self.format_prefix_noveg()}.h5'), mode = 'r') as h:
            spi = h.select(self.extreme).loc[self.city_list, :].drop('of_season', axis = 1)
            filt = spi.index.get_level_values('end').month.isin(season_to_month[self.season])
            spi = spi.loc[filt, :].rename({'in_event': 'spi_in_event', 'post_event': 'spi_post_event'}, axis = 1)

        with pd.HDFStore(os.path.join(path_out, 'clim', f'{self.prefix}_vpd_{self.format_prefix_noveg()}.h5'), mode = 'r') as h:
            vpd = h.select(self.extreme).loc[self.city_list, :].drop('of_season', axis = 1)
            filt = vpd.index.get_level_values('end').month.isin(season_to_month[self.season])
            vpd = vpd.loc[filt, :].rename({'in_event': 'vpd_in_event', 'post_event': 'vpd_post_event'}, axis = 1)

        return tmax, tmin, spi, vpd


    def collect_luc(self):
        """ Pixel level land cover percentages. Include only urban Developed & rural self.land_cover
            Pixel level percentage area """
        with pd.HDFStore(os.path.join(path_out, 'luc', f'{self.prefix}_{self.format_prefix_noveg()}.h5'), mode = 'r') as h:
            data = h.select(self.extreme).loc[self.city_list, :]

            filt = data.index.get_level_values('end').month.isin(season_to_month[self.season])
            data = data.loc[filt, :]

            data = data.loc[data['impervious_frac'] <= 0.8, :] # mask impervious area
            data = data.rename(modis_luc_agg_names, axis = 1)
            data = data.loc[data['Crop'] < 0.5, :] # remove croplands

            # Keep only the desired pixels based on LUC and urban/rural
            data = add_urlabel_all(data, self.extent)
            filt = ((data['location'] == 'rural') & (data[self.land_cover] >= 0.5)) | \
                   ((data['location'] == 'urban') & (data['Developed'] >= 0.5))
            data = data.loc[filt, :].drop('location', axis = 1)

            # Drop the null land cover types
            data = data.loc[:, data.mean(axis = 0) > 1e-6]

            # Drop the Developed type since is in impervious_frac
            luc = data.drop(['impervious_frac', 'Developed'], axis = 1)
            impervious = data[['impervious_frac']]

        return luc, impervious


    def collect_event(self):
        """ intensity & duration of the event """
        with pd.HDFStore(os.path.join(path_out, 'extreme_events', f'percity_{self.format_prefix_noveg()}.h5'), mode = 'r') as h:
            data = h.select(self.extreme).loc[self.city_list, :]
            filt = data.index.get_level_values(2).month.isin(season_to_month[self.season])
            if self.extreme == 'heat_wave':
                extreme_property = data.loc[filt, :].rename({'intensity': 'event_intensity', 'duration': 'event_duration'}, axis = 1)
            else:
                extreme_property = data.loc[filt, :].rename({'hot_intensity': 'hot_intensity', 'dry_intensity': 'dry_intensity', 'duration': 'event_duration'}, axis = 1)
        return extreme_property


    def collect_elev(self):
        """ pixel level elevation """
        with pd.HDFStore(os.path.join(path_out, 'elev', f'{self.prefix}_{self.extent}.h5'), mode = 'r') as h:
            elev = h.select('elev').copy().loc[self.city_list, :]
        return elev


    def collect_optimalT(self):
        """ pixel optimal temperature. fid x row x col """
        h = pd.HDFStore(os.path.join(path_out, 'veg', f'{self.prefix}_{self.extent}_{self.name}_optimalT.h5'), mode = 'r')
        optima = pd.DataFrame({'optimal_tmax': h['optima_tmax'].loc[self.city_list, 0],
                               'optimal_tmin': h['optima_tmin'].loc[self.city_list, 0]})
        h.close()
        return optima


    def collect_sensWater(self):
        """ pixel sensitivity to spi and vpd. fid x row x col | [spi, vpd]

        Although urban-rural difference does not exist at U.S. level, individual cities 
        (esp. when separately considering eastern and western U.S.) exhibit clear differences.
        """
        h = pd.HDFStore(os.path.join(path_out, 'veg', f'{self.prefix}_{self.extent}_{self.name}_sensitivity_water.h5'), mode = 'r')
        corr = {}
        for varname in ['corr_spi', 'corr_vpd']:
            corr[varname] = h.select(varname).copy(deep = True).loc[self.city_list, self.season]
        corr = pd.DataFrame(corr)
        h.close()
        return corr


    def collect_predictand(self):
        with pd.HDFStore(os.path.join(path_out, 'veg_response', f'{self.prefix}_{self.format_prefix()}.h5'), mode = 'r') as h:
            data = h.select(self.extreme).loc[self.city_list, :]
            filt = data.index.get_level_values('end').month.isin(season_to_month[self.season])
            predictand = data.loc[filt, ['Resistance', 'Recovery']]
        return predictand


    def collect_predictors(self):
        """ Collect the predictors into the same data frame """
        tmax, tmin, spi, vpd = self.collect_uhi()
        luc, impervious = self.collect_luc()
        extreme_property = self.collect_event()
        elev = self.collect_elev()
        optimalT = self.collect_optimalT()
        corr = self.collect_sensWater()

        index = tmax.index.intersection(tmin.index).intersection(spi.index).intersection( \
            vpd.index).intersection(luc.index).intersection(impervious.index)
        collection = pd.DataFrame(np.nan, index = index, 
                                columns = ['dtmax_in_event', 'dtmax_post_event', 'dtmin_in_event', 'dtmin_post_event', 
                                            'spi_in_event', 'spi_post_event', 'vpd_in_event', 'vpd_post_event'] + \
                                            list(luc.columns) + ['impervious_frac', 'event_intensity', 'event_duration'] + \
                                            ['elevation', 'optimal_tmax', 'optimal_tmin', 'corr_spi', 'corr_vpd'])
        collection.loc[:, ['dtmax_in_event', 'dtmax_post_event']] = tmax.loc[index, :]
        collection.loc[:, ['dtmin_in_event', 'dtmin_post_event']] = tmin.loc[index, :]
        collection.loc[:, ['spi_in_event', 'spi_post_event']] = spi.loc[index, :]
        collection.loc[:, ['vpd_in_event', 'vpd_post_event']] = vpd.loc[index, :]
        collection.loc[:, luc.columns] = luc.loc[index, :]
        collection.loc[:, 'impervious_frac'] = impervious.loc[index, 'impervious_frac']

        index_first3 = pd.MultiIndex.from_frame(index.to_frame()[['fid', 'start', 'end']])
        for ind in extreme_property.index:
            if ind in index_first3:
                collection.loc[ind, 'event_intensity'] = extreme_property.loc[ind, 'event_intensity']
                collection.loc[ind, 'event_duration'] = extreme_property.loc[ind, 'event_duration']

        index_first3 = pd.MultiIndex.from_frame(index.to_frame()[['fid', 'row', 'col']])
        ind2 = elev.index.intersection(optimalT.index).intersection(corr.index)
        for ind in ind2:
            if ind in index_first3:
                filt = index_first3 == ind
                collection.loc[filt, 'elevation'] = elev.loc[ind, 'elev']
                collection.loc[filt, 'optimal_tmax'] = optimalT.loc[ind, 'optimal_tmax']
                collection.loc[filt, 'optimal_tmin'] = optimalT.loc[ind, 'optimal_tmin']
                collection.loc[filt, 'corr_spi'] = corr.loc[ind, 'corr_spi']
                collection.loc[filt, 'corr_vpd'] = corr.loc[ind, 'corr_vpd']


        self.predictors = collection.dropna(axis = 0, how = 'any')
        self.predictand = self.collect_predictand().dropna(axis = 0, how = 'any')

        ind = self.predictand.index.intersection(self.predictors.index)

        self.predictand = self.predictand.loc[ind, :]
        self.predictors = self.predictors.loc[ind, :]

        return self.predictors, self.predictand


class Norm():
    def __init__(self, df = None):
        if df is not None:
            self.min = df.min(axis = 0)
            self.max = df.max(axis = 0)
        else:
            self.min = None
            self.max = None

    def apply(self, new_df):
        return (new_df - self.min) / (self.max - self.min)

    def reverse(self, df):
        return (self.max - self.min) * df + self.min

    def save(self, filename):
        pd.DataFrame({'min': self.min, 'max': self.max}).to_csv(filename)

    def load(self, filename):
        data = pd.read_csv(filename, index_col = 0)
        self.min = data['min']
        self.max = data['max']


class Regression(PixelData):
    def __init__(self, use, extreme, season, land_cover, city_list, target, n_estimators):
        super().__init__(use, extreme, season, land_cover, city_list)
        self.target = target # Resistance/Recovery
        self.n_estimators = n_estimators
        self.reg = None
        self.norm = None
        self.run_suffix = f'{self.format_prefix()}_{self.extreme}_{self.season}_{self.land_cover}_{self.target}_{self.n_estimators}'
        self.predictors, self.predictand = self.collect_predictors()


    def read_Xy(self, target):
        y = self.predictand[target] * 100 # inflate the values for better numerical accuracy?

        if target == 'Resistance':
            X = self.predictors.loc[:, [c for c in self.predictors.columns if not 'post_event' in c]]
        else:
            X = self.predictors.loc[:, [c for c in self.predictors.columns if not 'in_event' in c]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 12)

        # normalize X using min-max
        self.norm = Norm(X_train)
        X_train_norm = self.norm.apply(X_train)
        X_test_norm = self.norm.apply(X_test)

        return X_train_norm, X_test_norm, y_train, y_test


    def run_reset(self, alt_n = None):
        if alt_n is not None:
            self.n_estimators = alt_n
        self.run_suffix = f'{self.format_prefix()}_{self.extreme}_{self.season}_{self.land_cover}_{self.target}_{self.n_estimators}'


    def run_oob(self, alt_n = None):
        """ Find the hyper parameter using OOB score """
        self.run_reset(alt_n)
        X_train_norm, _, y_train, _ = self.read_Xy(self.target)
        self.reg = RandomForestRegressor(n_estimators = self.n_estimators, max_depth = X_train_norm.shape[1] - 1,
                                         n_jobs = mp.cpu_count() // 2, oob_score = True)
        self.reg = self.reg.fit(X_train_norm.values, y_train.values)

        ## json format saves the optimal number of trees
        #self.reg.save_model(os.path.join(path_out, 'measures', 'regression_per_pixel', 'fit', 
        #                    f'{self.prefix}_{self.run_suffix}.json'))
        joblib.dump(self.reg, os.path.join(path_out, 'measures', 'regression_per_pixel', 'fit', 
                                            f'{self.prefix}_{self.run_suffix}.sav'))
        # need to save the data norm
        self.norm.save(os.path.join(path_out, 'measures', 'regression_per_pixel', 'fit', 
                                    f'{self.prefix}_norm_{self.run_suffix}.csv'))
        return self.reg.oob_score_


    def load_model(self):
        # self.reg = xgb.Booster()
        # self.reg.load_model(os.path.join(path_out, 'measures', 'regression_per_pixel', 'fit', f'{self.prefix}_{self.run_suffix}.json'))
        self.reg = joblib.load(os.path.join(path_out, 'measures', 'regression_per_pixel', 'fit', f'{self.prefix}_{self.run_suffix}.sav'))
        self.norm = Norm()
        self.norm.load(os.path.join(path_out, 'measures', 'regression_per_pixel', 'fit', f'{self.prefix}_norm_{self.run_suffix}.csv'))


    def run(self, alt_n = None, refit = False):
        self.run_reset(alt_n)
        X_train_norm, X_test_norm, y_train, y_test = self.read_Xy(self.target)

        if refit:
            self.reg = RandomForestRegressor(n_estimators = self.n_estimators, max_depth = X_train_norm.shape[1] - 1, n_jobs = mp.cpu_count() // 2)
            self.reg = self.reg.fit(X_train_norm.values, y_train.values)
        else:
            self.load_model()

        y_pred_train = self.reg.predict(X_train_norm.values)
        y_pred_test  = self.reg.predict(X_test_norm.values)

        rmse = (np.sqrt(np.mean(np.power(y_pred_test - y_test.values, 2))), np.sqrt(np.mean(np.power(y_pred_train - y_train.values, 2))))
        corr = (pearsonr(y_pred_test, y_test.values)[0], pearsonr(y_pred_train, y_train.values)[0])
        bias = (np.mean(y_pred_test) - np.mean(y_test.values), np.mean(y_pred_train) - np.mean(y_train.values))
        self.performance = pd.DataFrame({'rmse': rmse, 'corr': corr, 'bias': bias}, index = ['test','train'])

        if refit:
            # If a new model is fitted, save the model
            joblib.dump(self.reg, os.path.join(path_out, 'measures', 'regression_per_pixel', 'fit', 
                                               f'{self.prefix}_{self.run_suffix}.sav'))
            self.norm.save(os.path.join(path_out, 'measures', 'regression_per_pixel', 'fit', 
                                        f'{self.prefix}_norm_{self.run_suffix}.csv'))

        # Save for SHAP values calculation
        filename = os.path.join(path_out, 'measures', 'regression_per_pixel', 'shap', f'{self.prefix}_{self.run_suffix}.h5')
        with pd.HDFStore(filename, mode = 'w') as hf:
            hf.append('X_test_norm', X_test_norm)
            hf.append('y_test', y_test.to_frame(self.target))
            hf.append('y_pred_test', pd.DataFrame(y_pred_test, index = y_test.index, columns = [self.target]))
            hf.append('y_train', y_train.to_frame(self.target))
            hf.append('y_pred_train', pd.DataFrame(y_pred_train, index = y_train.index, columns = [self.target]))
            hf.append('performance', self.performance)

        return self.reg, self.performance


    def shap(self):
        """ Calculate the SHAP values """
        filename = os.path.join(path_out, 'measures', 'regression_per_pixel', 'shap', f'{self.prefix}_{self.run_suffix}.h5')
        with pd.HDFStore(filename, mode = 'a') as hf:
            X_test_norm = hf.select('X_test_norm')
            explainer = fasttreeshap.TreeExplainer(self.reg, algorithm = 'auto', n_jobs = -1)
            shap_values = explainer(X_test_norm.values).values
            sv = pd.DataFrame(shap_values, index = X_test_norm.index, columns = X_test_norm.columns)
            hf.append(f'shap_values', sv)


if __name__ == '__main__':
    #--------------------------------------------------------------------------------
    # Get the list of cities to investigate each LUC
    setup = Setup('daymet')
    with pd.HDFStore(os.path.join(path_out, 'luc', f'percity_spatial_avg_{setup.format_prefix_noveg()}.h5'), mode = 'r') as h:
        df = h.select('heat_wave').loc[(slice(None), slice(None), slice(None), 'rural'), :].reset_index().groupby('fid').mean().drop(['impervious_frac', 'impervious_size'], axis = 1)
        df2 = pd.DataFrame(np.where(df > 0.2, df, np.nan), index = df.index, columns = df.columns)
        for fid in range(85):
          if np.all(np.isnan(df2.loc[fid, :].values)):
            df2.loc[fid, df.loc[fid, :].idxmax()] = df.loc[fid, df.loc[fid, :].idxmax()]
        df2['Developed'] = 0. # ignore this type
        df = df2.fillna(0.)

        # Skip the following cities
        df.loc[[0, 1, 2, 3, 26, 28, 29, 32, 34], :] = 0.
        df.loc[[51, 58, 6, 5, 8, 19, 27, 30, 43, 48, 49, 45, 46, 38, 33, 68, 74, 77, 78, 79, 80, 81, 82, 83, 31, 20], 'Grass'] = 0.
        df.loc[84, 'Crop']= 0.
        df.loc[[23, 25, 35, 54, 47], 'Deciduous forest'] = 0.
        df.loc[[12, 40], 'Wetland'] = 0.
        df.loc[:, 'Mixed forest'] = 0.

    luc_list = ['Evergreen forest', 'Deciduous forest', 'Shrub', 'Grass', 'Wetland']
    luc_cities = {}
    for luc in luc_list:
        luc_cities[luc] = list(df.loc[df[luc] > 1e-6, :].index)
    #--------------------------------------------------------------------------------

    use = 'topowx' # 'REPLACE0'
    opt = 'shap' # 'REPLACE1'
    ex = 'heat_wave' # 'REPLACE2' # 'heat_wave'
    season = 'DJF' # 'REPLACE3' # 'JJA'
    land_cover = 'Shrub' # luc_list[REPLACE4] # 'Evergreen'
    target = 'Resistance' # 'REPLACE5' # 'Resistance'


    # Note: loading the Deciduous data set takes 1 hour; the rest are several minutes
    # Number of data points (JJA)
    # Evergreen forest, 28695
    # Deciduous forest, 356334
    # Shrub, 108733
    # Grass, 112572
    # Crop, 44898
    # Wetland, 93557


    # Perform regression fit
    if opt == 'regression':
        r = Regression(use, ex, season, land_cover, luc_cities[land_cover], target, 1000)
        oob_score = pd.Series(np.nan, index = [1000, 5000, 10000, 50000])
        for n_estimators in [1000, 5000, 10000, 50000]:
            oob_score.loc[n_estimators] = r.run_oob(alt_n = n_estimators)
        suffix = r.run_suffix.replace('_50000', '')
        oob_score.to_csv(os.path.join(path_out, 'measures', 'regression_per_pixel', 'fit', f'{r.prefix}_oob_score_{suffix}.csv'))


    # Get the shap using the best 
    if opt == 'shap':
        # MODIFY THIS MANNUALLY
        # Determined value based on hyperparameter optimization
        n_estimators = 1000

        r = Regression(use, ex, season, land_cover, luc_cities[land_cover], target, n_estimators)
        r.run(refit = True)

        start = time()
        r.shap()
        end = time()
        print(f'The SHAP took {(end - start) / 60} minutes.')
