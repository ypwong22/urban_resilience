import xarray as xr
import os
from glob import glob
import geopandas as gpd
import rioxarray
import rasterio as rio
from osgeo import gdal, gdalconst
from shapely.geometry import mapping
from utils.paths import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np


class NLCD():
    def __init__(self, fid):
        self.fid = fid
        self.pft_names = {
            11: 'Open water',
            21: 'Developed, open space',
            22: 'Developed, low intensity',
            23: 'Developed, medium intensity',
            24: 'Developed high intensity',
            31: 'Barren land',
            41: 'Deciduous forest',
            42: 'Evergreen forest',
            43: 'Mixed forest',
            51: 'Dwarf scrub',
            52: 'Shrub/scrub',
            71: 'Grassland/herbaceous',
            81: 'Pasture/hay',
            82: 'Cultivated crops',
            90: 'Woody wetlands',
            95: 'Emergent herbaceous wetlands'
        }
        self.pft_aggr = {
            'Barren': [31], 
            'Broadleaf deciduous tree': [41],
            'Broad/needleleaf evergreen tree': [42],
            'Crop': [82],
            'Grass': [71, 81], 
            'Mixed tree': [43],
            'Shrub': [51, 52], 
            'Urban': [21,22,23,24], 
            'Water': [11, 90, 95]
        }
        self.pft_aggr_rev = dict([(c,j) for j in self.pft_aggr.keys() for c in self.pft_aggr[j]])


    def extract_data(self):
        filename = os.path.join(path_intrim, 'gee_single', 'NLCD', 'tiff_3x', f'NLCD_{self.fid:02g}.tif')
        h = rio.open(filename)
        array = h.read()
        index = h.descriptions
        h.close()

        data = pd.Series(0., index = self.pft_aggr.keys())
        for pft in self.pft_aggr.keys():
            for n in self.pft_aggr[pft]:
                loc = np.where([i == f'2019_{n:02g}' for i in index])[0]
                data.loc[pft] += np.nanmean(array[loc, :, :]) * 100.
        data = data.to_frame('2019')
        return data


class GCAM_Demeter():
    
    def __init__(self, scenario, model, fid):
        self.path_gcam_input = '/gpfs/alpine/cli146/proj-shared/future_land_use/gcam'
        self.scenario = scenario
        self.model = model
        self.fid = fid
        self.pft_names = {
            0 : ('Water', 'Water'),
            1 : ('Needleleaf evergreen tree - temperate', 'NET_tem'),
            2 : ('Needleleaf evergreen tree - boreal', 'NET_bor'),
            3 : ('Needleleaf deciduous tree - boreal', 'NDT_bor'),
            4 : ('Broadleaf evergreen tree - tropical', 'BET_tro'),
            5 : ('Broadleaf evergreen tree - temperate', 'BET_tem'),
            6 : ('Broadleaf deciduous tree - tropical', 'BDT_tro'),
            7 : ('Broadleaf deciduous tree - temperate', 'BDT_tem'),
            8 : ('Broadleaf deciduous tree - boreal', 'BDT_bor'),
            9 : ('Broadleaf evergreen shrub - temperate', 'BES_tem'),
            10: ('Broadleaf deciduous shrub - temperate', 'BDS_tem'),
            11: ('Broadleaf deciduous shrub - boreal', 'BDS_bor'),
            12: ('C3 Arctic', 'C3_gra_arc'),
            13: ('C3 Grass', 'C3_gra'),
            14: ('C4 Grass', 'C4_gra'),
            15: ('Corn: rainfed', 'Corn_rf'),
            16: ('Corn: irrigated', 'Corn_irr'),
            17: ('Wheat: rainfed', 'Wheat_rf'),
            18: ('Wheat: irrigated', 'Wheat_irr'),
            19: ('Soybean: rainfed', 'Soy_rf'),
            20: ('Soybean: irrigated', 'Soy_irr'),
            21: ('Cotton: rainfed', 'Cotton_rf'),
            22: ('Cotton: irrigated', 'Cotton_irr'),
            23: ('Rice: rainfed', 'Rice_rf'),
            24: ('Rice: irrigated', 'Rice_irr'),
            25: ('Sugar crop: rainfed', 'Sugarcrop_rf'),
            26: ('Sugar crop: irrigated', 'Sugarcrop_irr'),
            27: ('Other crop: rainfed', 'OtherCrop_rf'),
            28: ('Other crop: irrigated', 'OtherCrop_irr'),
            29: ('Bioenergy crop: rainfed', 'Bioenergy_rf'),
            30: ('Bioenergy crop: irrigated', 'Bioenergy_irr'),
            31: ('Urban', 'Urban'),
            32: ('Barren', 'Barren')
        }
        self.pft_aggr = {
            'Water': [0],
            'Needleleaf tree': [1, 2, 3],
            'Broadleaf evergreen tree': [4, 5],
            'Broadleaf deciduous tree': [6, 7, 8],
            'Shrub': [9, 10, 11], 
            'Grass': [12, 13, 14], 
            'Crop': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
            'Urban': [31], 
            'Barren': [32]
        }
        self.pft_aggr_rev = dict([(c,j) for j in self.pft_aggr.keys() for c in self.pft_aggr[j]])


    def extract_data(self):
        for i in range(33):
            pft = f'PFT{i}'

            infile_list = sorted(glob(os.path.join(self.path_gcam_input, f'GCAM_Demeter_LU_{self.scenario}_{self.model}_*.nc')))
            outfile = os.path.join(path_out, 'future_luc_summary', 'gcam', f'gcam_{self.scenario}_{self.model}_{pft}_{self.fid}.nc')

            with xr.open_mfdataset(infile_list, combine = 'nested', concat_dim = 'year') as h:
                data = h[pft].transpose('year', 'latitude', 'longitude')[:, ::-1, :]
                data.rio.set_spatial_dims(x_dim = 'longitude', y_dim = 'latitude', inplace = True)
                data.rio.write_crs('epsg:4326', inplace = True)

                shap = gpd.read_file(os.path.join(path_intrim, 'urban_mask', 'city_boundary_3x_merged.shp'))
                geom = shap.geometry[self.fid]

                clipped = data.rio.clip([mapping(geom)], shap.crs, drop = True)
                clipped.to_dataset(name = pft).to_netcdf(outfile)


    def plot_timeseries(self):
        data = pd.DataFrame(index = range(2015, 2101, 5), columns = range(33))
        for i in range(33):
            pft = f'PFT{i}'
            with xr.open_dataset(os.path.join(path_out, 'future_luc_summary', 'gcam', f'gcam_{self.scenario}_{self.model}_{pft}_{self.fid}.nc')) as hr:
                weights = np.cos(np.deg2rad(hr.latitude))
                weights.name = 'weights'
                ts = hr[pft].weighted(weights).mean(('latitude', 'longitude'))
                data.iloc[:, i] = ts.values
 
        data2 = data.stack()
        data2.index.names = ['year', 'pft']
        data2 = data2.reset_index()
        data2['pft'] = data2['pft'].map(self.pft_aggr_rev)
        data2 = data2.groupby(['year', 'pft']).sum().iloc[:, 0].unstack()

        obj2 = NLCD(self.fid)
        data3 = obj2.extract_data()
 
        fig = plt.figure(figsize = (11, 10))
        gs = fig.add_gridspec(nrows = 1, ncols = 2, width_ratios = (1, 9))

        ax_1 = fig.add_subplot(gs[0])
        data3.T.plot(kind='bar', stacked = True, ax = ax_1)
        ax_1.legend(bbox_to_anchor = [5.5, -0.05], ncol = 2)
        ax_1.set_ylim([0, 100])
        ax_1.set_xticklabels([2019], rotation = 0)

        ax = fig.add_subplot(gs[1])
        ax.stackplot(range(2015, 2101, 5), data2.values.T, labels = data2.columns)
        ax.set_xlim([2015, 2100])
        ax.set_ylim([0, 100])
        ax.legend(bbox_to_anchor = [1, -0.05], ncol = 2)
        fig.savefig(os.path.join(path_out, 'future_luc_summary', f'gcam_{self.scenario}_{self.model}_{self.fid}.png'), dpi = 600., bbox_inches = 'tight')
        plt.close(fig)


class CellularAutomata():

    def __init__(self, scenario, fid):
        self.path_infiles = '/gpfs/alpine/cli146/proj-shared/future_land_use/cellular_automata'
        self.scenario = scenario
        self.fid = fid
        self.year_list = list(range(2020, 2101, 5))
        self.pft_names = {
            1 : 'Water',
            2 : 'Broadleaf evergreen tree, tropical',
            3 : 'Broadleaf evergreen tree, temperate',
            4 : 'Broadleaf deciduous tree, tropical',
            5 : 'Broadleaf deciduous tree, temperate',
            6 : 'Broadleaf deciduous tree, boreal',
            7 : 'Needleleaf evergreen tree, temperate',
            8 : 'Needleleaf evergreen tree, boreal',
            9 : 'Needleleaf deciduous tree',
            10: 'Broadleaf evergreen shrub, temperate',
            11: 'Broadleaf deciduous shrub, temperate',
            12: 'Broadleaf deciduous shrub, boreal',
            13: 'C3 grass, arctic',
            14: 'C3 grass',
            15: 'C4 grass',
            16: 'Mixed C3/C4 grass',
            17: 'Barren',
            18: 'Cropland',
            19: 'Urban'
            # 20: 'Permanent snow and ice' // does not exist in the study region
        }
        self.pft_aggr = {
            'Water': [1],
            'Needleleaf tree': [7, 8, 9],
            'Broadleaf evergreen tree': [2, 3],
            'Broadleaf deciduous tree': [4, 5, 6],
            'Shrub': [10, 11, 12], 
            'Grass': [13, 14, 15, 16], 
            'Crop': [18],
            'Urban': [19], 
            'Barren': [17]
        }
        self.pft_aggr_rev = dict([(c,j) for j in self.pft_aggr.keys() for c in self.pft_aggr[j]])


    def extract_data(self):
        for yr in self.year_list:
            filename = os.path.join(self.path_infiles, self.scenario, f'global_PFT_{self.scenario}_{yr}.tif')

            f = gdal.Open(filename, gdal.GA_ReadOnly)
            src_srs = f.GetProjection()
            f = None

            f = gdal.Open(os.path.join(path_intrim, 'urban_mask', 'city_boundary_3x_merged.tif'), gdal.GA_ReadOnly)
            dst_srs = f.GetProjection()
            f = None

            f = gdal.Open(os.path.join(path_intrim, 'gee_single', 'NLCD', 'tiff_3x', f'NLCD_{self.fid:02d}.tif'), gdal.GA_ReadOnly)
            geoTransform = f.GetGeoTransform()
            minx = geoTransform[0]
            maxy = geoTransform[3]
            maxx = minx + geoTransform[1] * f.RasterXSize
            miny = maxy + geoTransform[5] * f.RasterYSize
            nodatamask = ~np.isnan(f.GetRasterBand(1).ReadAsArray())
            f = None

            # print('\t', filename, fid, minx, maxy, maxx, miny)

            # reproject and cut to the same extent
            newfile = os.path.join(path_out, 'future_luc_summary', 'cellular_automata', f'temp_{self.scenario}_{yr}_{self.fid}.tif')
            gdal.Warp(
                newfile,
                filename,
                format = 'GTiff',
                outputBounds = [minx, miny, maxx, maxy],
                xRes = 1000, yRes = 1000, targetAlignedPixels = True,
                srcSRS = src_srs, dstSRS = dst_srs,
                srcNodata = -128, dstNodata = -128,
                outputType = gdalconst.GDT_Int16,
                resampleAlg = gdal.GRA_NearestNeighbour
            )

        outvrt = os.path.join(path_out, 'future_luc_summary', 'cellular_automata', f'temp_{self.scenario}_{self.fid}.vrt')
        outtif = os.path.join(path_out, 'future_luc_summary', 'cellular_automata', f'temp_{self.scenario}_{self.fid}.tif')
        tifs = [os.path.join(path_out, 'future_luc_summary', 'cellular_automata', f'temp_{self.scenario}_{yr}_{self.fid}.tif') for yr in self.year_list]
        outds = gdal.BuildVRT(outvrt, tifs, separate=True)
        outds = gdal.Translate(outtif, outds, creationOptions = ["TILED=YES"])
        outds = None

        for tif in tifs:
            os.remove(tif)
        os.remove(outvrt)

        f = rio.open(outtif, mode = 'r')
        profile = dict(f.profile)
        array = f.read()
        array = np.where(np.broadcast_to(nodatamask[np.newaxis, ...], array.shape), array, -128)
        f.close()
        with rio.open(os.path.join(path_out, 'future_luc_summary', 'cellular_automata', f'{self.scenario}_{self.fid}.tif'), 'w', **profile) as dst:
            dst.write(array, indexes = list(range(1, len(array) + 1)))
        os.remove(outtif)


    def plot_timeseries(self):
        data = pd.DataFrame(index = range(2020, 2101, 5), columns = range(1, 20))

        f = rio.open(os.path.join(path_out, 'future_luc_summary', 'cellular_automata', f'{self.scenario}_{self.fid}.tif'))
        temp = f.read()
        for i in range(1, 20):
            data.loc[:, i] = (temp == i).sum(axis = 2).sum(axis = 1) / (temp != -128).sum(axis = 2).sum(axis = 1) * 100.
        f.close()

        data2 = data.stack()
        data2.index.names = ['year', 'pft']
        data2 = data2.reset_index()
        data2['pft'] = data2['pft'].map(self.pft_aggr_rev)
        data2 = data2.groupby(['year', 'pft']).sum().iloc[:, 0].unstack()

        obj2 = NLCD(self.fid)
        data3 = obj2.extract_data()
 
        fig = plt.figure(figsize = (11, 10))
        gs = fig.add_gridspec(nrows = 1, ncols = 2, width_ratios = (1, 9))

        ax_1 = fig.add_subplot(gs[0])
        data3.T.plot(kind='bar', stacked = True, ax = ax_1)
        ax_1.legend(bbox_to_anchor = [5.5, -0.05], ncol = 2)
        ax_1.set_ylim([0, 100])
        ax_1.set_xticklabels([2019], rotation = 0)

        ax = fig.add_subplot(gs[1])
        ax.stackplot(range(2020, 2101, 5), data2.values.T, labels = data2.columns)
        ax.set_xlim([2020, 2100])
        ax.set_ylim([0, 100])
        ax.legend(bbox_to_anchor = [1, -0.05], ncol = 2)
        fig.savefig(os.path.join(path_out, 'future_luc_summary', f'cellular_automata_{self.scenario}_{self.fid}.png'), dpi = 600., bbox_inches = 'tight')
        plt.close(fig)


if __name__ == '__main__':
    #########################################################################
    scenario_list = ['ssp1_rcp26', 'ssp1_rcp45', 'ssp1_rcp60',
                     'ssp2_rcp26', 'ssp2_rcp45', 'ssp2_rcp60', 
                     'ssp3_rcp45', 'ssp3_rcp60',
                     'ssp4_rcp26', 'ssp4_rcp45', 'ssp4_rcp60', 
                     'ssp5_rcp26', 'ssp5_rcp45', 'ssp5_rcp60', 'ssp5_rcp85']
    model_list = ['modelmean', 'gfdl', 'hadgem', 'ipsl', 'miroc', 'noresm']

    model = 'modelmean'
    for scenario in ['ssp5_rcp85', 'ssp2_rcp45']:
        for fid in [18, 61]:
            obj = GCAM_Demeter(scenario, model, fid)
            #obj.extract_data()
            obj.plot_timeseries()

    #########################################################################
    scenario_list = ['SSP1_RCP19', 'SSP1_RCP26', 'SSP2_RCP45', 'SSP3_RCP70',
                     'SSP4_RCP34', 'SSP4_RCP60', 'SSP5_RCP34', 'SSP5_RCP85']
    for scenario in ['SSP2_RCP45', 'SSP5_RCP85']:
        for fid in [18, 61]:
            obj = CellularAutomata(scenario, fid)
            #obj.extract_data()
            obj.plot_timeseries()