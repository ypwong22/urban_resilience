from osgeo import gdal, ogr, osr, gdalconst
import os
from utils.paths import *

# extent and properties // following Daymet
x_min = -4560250
y_max = 4984000
x_max = 3252750
y_min = -3090000
x_res = 7814
y_res = 8075
pixel_width = 1000
ref = osr.SpatialReference()
ref.ImportFromProj4('+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m no_defs')

# open shapefile and create layer
shpFile = os.path.join(path_data, 'Masks', 'US_urbanCluster', 'US_urbanCluster_500km.shp')
driver = ogr.GetDriverByName('ESRI Shapefile')

srcData = driver.Open(shpFile, 0)
srcLayer = srcData.GetLayer()
srcRef = srcLayer.GetSpatialRef()

nr_features = srcLayer.GetFeatureCount()
for feat_id in range(nr_features):
    feature = srcLayer.GetFeature(feat_id)
    geom = feature.GetGeometryRef()
    #refTrans = osr.CoordinateTransformation(srcLayer.GetSpatialRef(), ref)
    #geom.Transform(refTrans)

    outShapefile = os.path.join(path_intrim, 'urban_mask',
                                'urban_mask_temp_' + str(feat_id) + '.shp')
    # Remove output shapefile if it already exists
    if os.path.exists(outShapefile):
        driver.DeleteDataSource(outShapefile)
    # Create the output shapefile
    outDataSource = driver.CreateDataSource(outShapefile)
    outLayer = outDataSource.CreateLayer('feature' + str(feat_id), srs = srcRef, 
                                         geom_type=ogr.wkbPolygon)

    # Add fields
    srcLayerDefn = srcLayer.GetLayerDefn()
    for i in range(0, srcLayerDefn.GetFieldCount()):
        fieldDefn = srcLayerDefn.GetFieldDefn(i)
        outLayer.CreateField(fieldDefn)
 
    # Create the feature and set values
    featureDefn = outLayer.GetLayerDefn()
    feat = ogr.Feature(featureDefn)
    feat.SetGeometry(geom)
    for i in range(0, featureDefn.GetFieldCount()):
        feat.SetField(featureDefn.GetFieldDefn(i).GetNameRef(), feature.GetField(i))
    outLayer.CreateFeature(feat)

    # Save and close DataSource
    feat = None
    outDataSource = None

    # Save to raster file
    output = os.path.join(path_intrim, 'urban_mask', 'urban_mask_' + str(feat_id)+'.tif')
    target_ds = gdal.GetDriverByName('GTiff').Create(output, x_res, y_res, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((x_min, pixel_width, 0, y_max, 0, -pixel_width))
    target_ds.SetProjection(ref.ExportToProj4())

    new_dataSource = driver.Open(outShapefile, 0)
    new_layer = new_dataSource.GetLayer()

    # this works now
    gdal.RasterizeLayer(target_ds, [1], new_layer)
    target_ds = None

    new_dataSource = None

dataSource = None
