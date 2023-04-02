from scipy.stats import gamma, norm
import warnings
import numpy as np

def calc_spi(vector):
    """ Assumes vector starts from day 1 of the year and ends on day 365 of the year. 365-day calendar. """

    if len(vector) < 90:
        return np.full(len(vector) // 365 * 12, np.nan)
    
    if sum(np.isnan(vector)) > 0:
        warnings.warn('Unable to handle missing values. Return nan')
        return np.full(len(vector) // 365 * 12, np.nan)

    # 1. Rolling monthly average
    vector2 = np.full(len(vector), np.nan)
    # note skipping the leading 90 days because of rolling average
    for ii in range(90, len(vector)):
        vector2[ii] = np.mean(vector[(ii-90):ii])
    # vector2 = vector2[90:]

    # for each last day of the month
    vector3 = np.full([len(vector2) // 365, 12], np.nan)

    for m, doy in enumerate([ 31,  59,  90, 120, 151, 181, 212, 243, 273, 304, 334, 365]):
        # 2. fit distribution
        temp = vector2[(doy-1)::365]
        filter = ~np.isnan(temp)
        temp = temp[filter]
        a, loc, scale = gamma.fit(temp)

        # 3. Convert to quantile
        pct = gamma.cdf(temp, a, loc, scale)
        pct = np.clip(pct, 1e-6, 1-1e-6) # sometimes the pct = 1., resulting in np.inf

        # 4. Convert to z-score
        vector3[filter, m] = norm.ppf(pct, 0, 1)

    return vector3.reshape(-1)