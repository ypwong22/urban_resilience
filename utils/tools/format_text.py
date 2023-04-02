import numpy as np

def ppp(p_value):
    """ Use atsterisks to denote p-values. """
    if p_value <= 0.01:
        return r'^{***}'
    elif p_value <= 0.05:
        return r'^{**}'
    elif p_value <= 0.1:
        return r'^{*}'
    else:
        return ''


def ppf(slope, intercept, p_slope, p_intercept):
    """ Pretty format for regression equations. """

    if (np.abs(slope) < 1e-1) | (np.abs(slope) > 10):
        fslope = '{:.2e}'.format(slope)
    else:
        fslope = '%.2f' % slope
    fslope += ppp(p_slope)

    if (np.abs(intercept) < 1e-1) | (np.abs(intercept) > 10):
        fintercept = '{:.2e}'.format(np.abs(intercept))
    else:
        fintercept = '%.2f' % (np.abs(intercept))
    if intercept > 0:
        fintercept = ' + ' + fintercept
    else:
        fintercept = ' - ' + fintercept
    fintercept += ppp(p_intercept)

    return r'$y = ' + fslope + r' \times x' + fintercept + r'$'
