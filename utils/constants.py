import numpy as np
import cartopy.crs as ccrs

esa_cci_luc_legend = {10: 'Cropland, rainfed',
                      11: 'Herbaceous cover',
                      12: 'Tree or shrub cover',
                      20: 'Cropland, irrigated or post-flooding',
                      30: 'Mosaic cropland (>50%) / natural vegetation (tree, shrub, herbaceous cover) (<50%)',
                      40: 'Mosaic natural vegetation (tree, shrub, herbaceous cover) (>50%) / cropland (<50%) ',
                      50: 'Tree cover, broadleaved, evergreen, closed to open (>15%)',
                      60: 'Tree cover, broadleaved, deciduous, closed to open (>15%)',
                      61: 'Tree cover, broadleaved, deciduous, closed (>40%)',
                      62: 'Tree cover, broadleaved, deciduous, open (15-40%)',
                      70: 'Tree cover, needleleaved, evergreen, closed to open (>15%)',
                      71: 'Tree cover, needleleaved, evergreen, closed (>40%)',
                      72: 'Tree cover, needleleaved, evergreen, open (15-40%)',
                      80: 'Tree cover, needleleaved, deciduous, closed to open (>15%)',
                      81: 'Tree cover, needleleaved, deciduous, closed (>40%)',
                      82: 'Tree cover, needleleaved, deciduous, open (15-40%)',
                      90: 'Tree cover, mixed leaf type (broadleaved and needleleaved)',
                      100: 'Mosaic tree and shrub (>50%) / herbaceous cover (<50%)',
                      110: 'Mosaic herbaceous cover (>50%) / tree and shrub (<50%)',
                      120: 'Shrubland',
                      121: 'Shrubland evergreen',
                      122: 'Shrubland deciduous',
                      130: 'Grassland',
                      140: 'Lichens and mosses',
                      150: 'Sparse vegetation (tree, shrub, herbaceous cover) (<15%)',
                      151: 'Sparse tree (<15%)',
                      152: 'Sparse shrub (<15%)',
                      153: 'Sparse herbaceous cover (<15%)',
                      160: 'Tree cover, flooded, fresh or brakish water',
                      170: 'Tree cover, flooded, saline water',
                      180: 'Shrub or herbaceous cover, flooded, fresh/saline/brakish water',
                      190: 'Urban areas',
                      200: 'Bare areas',
                      201: 'Consolidated bare areas',
                      202: 'Unconsolidated bare areas',
                      210: 'Water bodies',
                      220: 'Permanent snow and ice'}
esa_cci_luc_labels = sorted(esa_cci_luc_legend.keys())


modis_luc_legend = {
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


modis_luc_agg_names = {
    1: 'Developed',
    2: 'Evergreen forest',
    3: 'Mixed forest',
    4: 'Deciduous forest',
    5: 'Shrub',
    6: 'Grass',
    7: 'Crop',
    8: 'Wetland'
}


modis_luc_agg_id = {
    'Developed': 1, 
    'Evergreen forest': 2,
    'Mixed forest': 3,
    'Deciduous forest': 4,
    'Shrub': 5,
    'Grass': 6,
    'Crop': 7,
    'Wetland': 8
}

modis_luc_agg = {
    1: [21, 22, 23, 24],
    2: [42],
    3: [43],
    4: [41],
    5: [52],
    6: [71,81],
    7: [82],
    8: [90, 95]
} # ignore barren (31), but there's no need to specifically remove, because implicit in EVI < 0.1.


modis_luc_agg_rev = {
    (21, 22, 23, 24): 1,
    (42            ): 2,
    (43            ): 3,
    (41            ): 4,
    (52            ): 5,
    (71, 81        ): 6,
    (82            ): 7,
    (90, 95        ): 8
}


# By Koppen climate zone

#modis_luc_city_groups = {
#    'Crop': [4, 21, 23, 41, 39, 55, 11, 12, 18, 10, 16, 30, 9, 13, 15, 33, 27, 14], # 84 is too far south
#    'Nat_Veg_East': [54, 65, 47, 75, 35, 57, 38, 46, 63, 52, 45, 61, 49, 60, 56, 48, 43, 44, 37, 36, 19, 17, 24, 8, 6, 5, 7, 20, 22, 31],
#    'Wetland': [40, 62, 68, 74, 78, 77, 79, 81, 80, 82, 83, 73, 70, 72, 69, 66],
#    'Shrub': [42, 51, 50, 53, 58, 59, 64, 67, 71, 76],
#    'Northwest': [0, 1, 2, 3, 25, 26, 28, 29, 32, 34]
#}

modis_luc_city_groups = {
    'Northeast': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 27, 30, 33], # relatively homogeneous climate
    'Southeast_top': [41, 47, 54, 35, 57, 39, 55, 46, 38, 45, 61, 49, 48, 43, 44, 37, 40, 31, 52, 36, 63], # relatively homogeneous climate
    'Southeast_hot': [56, 60, 62, 68, 74, 78, 77, 80, 83, 79, 81, 82, 69, 66, 70, 72, 65, 71, 76, 75, 73, 84], # relatively homogeneous climate
    'West': [0, 1, 2, 3, 28, 26, 29, 51, 58, 53, 25, 32, 34, 42, 59, 64, 50, 67], # add background temperature and precipitation
}


modis_luc_short_list = ['urban_Developed', 'rural_Evergreen forest', 'rural_Mixed forest', 'rural_Deciduous forest', 'rural_Shrub', 'rural_Grass', 'rural_Crop', 'rural_Wetland']


wkt_daymet = 'PROJCS["unknown",GEOGCS["unknown",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["latitude_of_origin",42.5],PARAMETER["central_meridian",-100],PARAMETER["standard_parallel_1",25],PARAMETER["standard_parallel_2",60],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metres",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
crs_daymet = ccrs.LambertConformal(central_longitude=-100, central_latitude=42.5,
                                   standard_parallels= (25,60), globe = ccrs.Globe(datum="NAD83"))


city_names = {
    5: 'Albany, NY',
    6: 'Rochester, NY',
    7: 'Boston, MA',
    8: 'Buffalo, NY',
    9: 'Flint, MI',
    10: 'Grand Rapids, MI',
    11: 'Madison, WI',
    12: 'Milwaukee, WI',
    13: 'Detroit, MI',
    14: 'Allentown, PA',
    15: 'Toledo, OH',
    16: 'South Blend, IN',
    17: 'Youngston, OH',
    18: 'Chicago, IL',
    19: 'Cleveland, OH',
    20: 'Harrisburg, PA',
    22: 'New York, NY',
    24: 'Pittsburgh, PA',
    27: 'Columbus, OH',
    30: 'Bloomington, IN',
    31: 'Washington, DC',
    33: 'Cincinnati, OH',
    35: 'Kansas City, MI',
    36: 'Hampton, VA',
    37: 'Richmond, VA',
    38: 'Louisville, KY',
    39: 'St Louis, IL',
    40: 'Portsmouth, VA',
    43: 'Greensboro, NC',
    44: 'Raleigh, NC',
    45: 'Knoxville, TN',
    46: 'Nashville, TN', 
    48: 'Charlotte, NC',
    49: 'Greenville, SC',
    52: 'Chattanooga, TN',
    55: 'Memphis, TN',
    56: 'Columbia, SC',
    57: 'Little Rock, AR',
    60: 'Augusta, SC',
    61: 'Atlanta, GA',
    62: 'Charleston, SC',
    63: 'Birmingham, AL',
    66: 'Jackson, MI',
    68: 'Jacksonville, FL',
    69: 'Mobile, AL',
    70: 'Baton Rouge, LA',
    72: 'New Orleans, LA',
    74: 'Ocala, FL',
    77: 'Melborne, FL',
    78: 'Orlando, FL',
    79: 'Tampa, FL', 
    80: 'Port St Lucie',
    81: 'Sarasota, FL',
    82: 'Bonita Springs, FL',
    83: 'Miami, FL'
}


elm_pft_names = {
    0: 'not_vegetated',
    1: 'needleleaf_evergreen_temperate_tree',
    2: 'needleleaf_evergreen_boreal_tree',
    3: 'needleleaf_deciduous_boreal_tree',
    4: 'broadleaf_evergreen_tropical_tree',
    5: 'broadleaf_evergreen_temperate_tree',
    6: 'broadleaf_deciduous_tropical_tree',
    7: 'broadleaf_deciduous_temperate_tree',
    8: 'broadleaf_deciduous_boreal_tree',
    9: 'broadleaf_evergreen_shrub',
    10: 'broadleaf_deciduous_temperate_shrub',
    11: 'broadleaf_deciduous_boreal_shrub',
    12: 'c3_arctic_grass',
    13: 'c3_non-arctic_grass',
    14: 'c4_grass',
    15: 'c3_crop',
    16: 'c3_irrigated'
}


fid_to_yuyu = {
    0: 355,
    1: 641,
    2: 244,
    3: 631,
    4: 3060,
    5: 7668,
    6: 6819,
    7: 7889,
    8: 6475,
    9: 5443,
    10: 4913,
    11: 3958,
    12: 4330,
    13: 5481,
    14: 7525,
    15: 5518,
    16: 4846,
    17: 6127,
    18: 4365,
    19: 5850,
    20: 7116,
    21: 3044,
    22: 7515,
    23: 2420,
    24: 6267,
    25: 833,
    26: 117,
    27: 5729,
    28: 36,
    29: 17,
    30: 4861,
    31: 7105,
    32: 1231,
    33: 5377,
    34: 1245,
    35: 2767,
    36: 7447,
    37: 7183,
    38: 5103,
    39: 3723,
    40: 7559,
    41: 2081,
    42: 586,
    43: 6552,
    44: 6935,
    45: 5642,
    46: 4868,
    47: 2469,
    48: 6305,
    49: 6020,
    50: 1107,
    51: 219,
    52: 5367,
    53: 473,
    54: 2019,
    55: 3942,
    56: 6401,
    57: 3403,
    58: 395,
    59: 697,
    60: 6172,
    61: 5581,
    62: 6797,
    63: 4951,
    64: 781,
    65: 2052,
    66: 3991,
    67: 1080,
    68: 6435,
    69: 4698,
    70: 3755,
    71: 1935,
    72: 3921,
    73: 3029,
    74: 6377,
    75: 2541,
    76: 1792,
    77: 6862,
    78: 6457,
    79: 6256,
    80: 7039,
    81: 6337,
    82: 6601,
    83: 7099,
    84: 1810}
yuyu_to_fid = dict([(j, i) for i,j in fid_to_yuyu.items()])


month_to_season = {12: 'DJF',
                   1 : 'DJF',
                   2 : 'DJF',
                   3 : 'MAM',
                   4 : 'MAM',
                   5 : 'MAM',
                   6 : 'JJA',
                   7 : 'JJA',
                   8 : 'JJA',
                   9 : 'SON',
                   10: 'SON',
                   11: 'SON'}
season_to_month = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [7, 8, 9], 'SON': [9, 10, 11]}