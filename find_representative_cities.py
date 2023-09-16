"""
In the eastern U.S. that has 
dResistance > 0 in winter, dResistance < 0 in summer, dRecovery > 0 in winter, and dRecovery <0 in summer

In the western U.S. that has 
dResistance > 0 and dRecovery > 0 in summer

Pick large cities that satisfy these conditions
"""
import pandas as pd
import os
import numpy as np
from utils.analysis import *
from utils.constants import *
from utils.paths import *
from utils.extremes import *
from utils.get_monthly_data import *
from utils.plotting import *
import matplotlib.pyplot as plt
import matplotlib as mpl


prefix = 'percity_per_pixel'
extent = 'tiff_3x'
name = 'MOD09Q1G_EVI'
heat_wave_thres = 90
hot_and_dry_thres = 85
extreme = 'heat_wave'
suffix = '_nocrop'


h = pd.HDFStore(os.path.join(path_out, 'measures', 'regression_per_pixel', 'summary', f'{prefix}_{extent}_{name}_{heat_wave_thres}_{hot_and_dry_thres}_{extreme}_city_average_events{suffix}.h5'), mode = 'r')
data = h.select('data').copy()
summary_varname_diff = data.loc['urban', :] - data.loc['rural', :]
h.close()


is_western = summary_varname_diff.index.get_level_values('fid').isin(modis_luc_city_groups['West'])
is_summer = summary_varname_diff.index.get_level_values('end').month.isin([6, 7, 8])
is_winter = summary_varname_diff.index.get_level_values('end').month.isin([12, 1, 2])


west_criteria = is_western & is_summer & (summary_varname_diff['Resistance'] > 0) & (summary_varname_diff['Recovery'] > 0)
west_satisfy = west_criteria.groupby('fid').mean()
west_satisfy = west_satisfy.loc[west_satisfy > 0]


"""
In [46]: west_satisfy.sort_values()
Out[46]:
fid
58    0.020408
64    0.057692
51    0.062500
25    0.075000
50    0.081633
67    0.086957
59    0.092593
29    0.093023
32    0.105263
1     0.116279
2     0.121212
53    0.137255
26    0.148148
42    0.155172 # Pick Las Vegas
28    0.156863
34    0.170732
3     0.243243
dtype: float64
"""


# urban_size = pd.read_csv(os.path.join(path_out, 'uhi', 'urban_size.csv'))


west_criteria = is_western & is_summer & (summary_varname_diff['Resistance'] > 0) & (summary_varname_diff['Recovery'] > 0)
west_satisfy = west_criteria.groupby('fid').mean()
west_satisfy = west_satisfy.loc[west_satisfy > 0]


east_criteria = (~is_western) & \
    (is_summer & (summary_varname_diff['Resistance'] < 0) & (summary_varname_diff['Recovery'] < 0))
east_criteria2 = (~is_western) & \
    (is_winter & (summary_varname_diff['Resistance'] > 0) & (summary_varname_diff['Recovery'] > 0))
east_satisfy = east_criteria.groupby('fid').mean() + east_criteria2.groupby('fid').mean()
east_satisfy = east_satisfy.loc[east_satisfy > 0]


"""
In [71]: east_satisfy.sort_values()
Out[71]:
fid
75    0.052632
18    0.068182
84    0.097561
40    0.100000
12    0.100000
        ...
66    0.333333
8     0.333333
46    0.354839 # Pick Nashville
17    0.428571
76    0.468750
Length: 67, dtype: float64
"""


# Get the dates of the heat waves
summary_varname_diff.loc[42, :].reorder_levels(['use', 'start', 'end']).sort_index()
"""
daymet 2001-05-01 2001-05-01
       2001-09-01 2001-09-01
       2002-04-01 2002-07-01
       2003-01-01 2003-01-01
       2003-07-01 2003-10-01
       2004-03-01 2004-03-01
       2005-12-01 2005-12-01
       2007-11-01 2007-11-01
       2009-05-01 2009-05-01
       2011-08-01 2011-08-01
       2014-01-01 2014-01-01
       2014-10-01 2014-10-01
       2015-02-01 2015-03-01
       2016-02-01 2016-02-01
       2016-06-01 2016-06-01
       2017-06-01 2017-06-01
       2017-11-01 2017-12-01
       2018-04-01 2018-04-01
       2018-09-01 2018-09-01
       2019-08-01 2019-08-01
topowx 2002-04-01 2002-04-01
       2003-01-01 2003-01-01
       2003-07-01 2003-10-01
       2004-03-01 2004-03-01
       2005-07-01 2005-07-01
       2005-12-01 2005-12-01
       2007-11-01 2007-11-01
       2009-05-01 2009-05-01
       2009-09-01 2009-09-01
       2010-12-01 2010-12-01
       2011-08-01 2011-08-01
       2013-04-01 2013-06-01
       2014-01-01 2014-01-01
       2014-10-01 2014-10-01
       2015-02-01 2015-03-01
       2016-02-01 2016-02-01
       2016-06-01 2016-06-01
       2016-11-01 2016-11-01
yyz    2003-01-01 2003-01-01
       2003-07-01 2003-10-01
       2004-03-01 2004-03-01
       2005-07-01 2005-07-01
       2005-12-01 2005-12-01
       2006-05-01 2006-05-01
       2007-11-01 2007-11-01
       2009-05-01 2009-05-01
       2009-09-01 2009-09-01
       2011-08-01 2011-08-01
       2013-04-01 2013-06-01
       2014-01-01 2014-01-01
       2014-10-01 2014-10-01
       2015-02-01 2015-03-01
       2016-02-01 2016-02-01
       2016-06-01 2016-06-01
       2017-11-01 2017-12-01
       2018-04-01 2018-04-01
       2018-09-01 2018-09-01
       2019-08-01 2019-08-01
"""


summary_varname_diff.loc[(summary_varname_diff.index.get_level_values('fid') == 42) & is_summer & (summary_varname_diff['Resistance'] > 0) & (summary_varname_diff['Recovery'] > 0), :]
"""
fid start      end        use
42  2002-04-01 2002-07-01 daymet  0.043736    0.128705
    2011-08-01 2011-08-01 daymet  0.070178    0.009784
                          topowx  0.070178    0.009784
                          yyz     0.070178    0.009784
    2016-06-01 2016-06-01 daymet  0.068521    0.003602
                          topowx  0.068521    0.003602
                          yyz     0.068521    0.003602
    2019-08-01 2019-08-01 daymet  0.134913    0.089253
                          yyz     0.134913    0.089253
"""


summary_varname_diff.loc[46, :].reorder_levels(['use', 'start', 'end']).sort_index()
"""
daymet 2001-11-01 2001-11-01
       2006-01-01 2006-01-01
       2007-03-01 2007-03-01
       2007-08-01 2007-08-01
       2010-04-01 2010-10-01
       2012-01-01 2012-07-01
       2012-12-01 2012-12-01
       2015-12-01 2015-12-01
       2016-09-01 2017-04-01
       2018-02-01 2018-06-01
       2019-09-01 2019-09-01
topowx 2001-02-01 2001-02-01
       2001-11-01 2001-11-01
       2005-02-01 2005-02-01
       2006-01-01 2006-04-01
       2007-03-01 2007-10-01
       2010-04-01 2010-08-01
       2012-01-01 2012-07-01
       2012-12-01 2012-12-01
       2015-12-01 2015-12-01
       2016-06-01 2016-11-01
yyz    2006-01-01 2006-04-01
       2007-03-01 2007-08-01
       2010-06-01 2010-10-01
       2012-03-01 2012-03-01
       2012-07-01 2012-07-01
       2012-12-01 2012-12-01
       2015-11-01 2015-12-01
       2016-09-01 2017-04-01
       2018-02-01 2018-05-01
       2019-09-01 2019-09-01
"""


summary_varname_diff.loc[(summary_varname_diff.index.get_level_values('fid') == 46) & (east_criteria | east_criteria2), :]
"""
                                  Recovery  Resistance
fid start      end        use
46  2005-02-01 2005-02-01 topowx  0.053529    0.070469
    2006-01-01 2006-01-01 daymet  0.086760    0.062237
    2012-12-01 2012-12-01 daymet  0.003344    0.028650
                          topowx  0.003344    0.028650
                          yyz     0.003344    0.028650
    2015-11-01 2015-12-01 yyz     0.005800    0.053626
    2010-04-01 2010-08-01 topowx -0.017317   -0.096736
    2012-01-01 2012-07-01 daymet -0.044629   -0.144994
                          topowx -0.044629   -0.144994
    2012-07-01 2012-07-01 yyz    -0.136975   -0.016639
    2018-02-01 2018-06-01 daymet -0.085613   -0.029192
"""