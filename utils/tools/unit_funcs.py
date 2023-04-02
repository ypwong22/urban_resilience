import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import linregress


####
def unit_trend(vector):
    x = np.arange(len(vector))
    y = vector
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    if len(y) > 3:
        result = linregress(x, y)
        return result.slope
    else:
        return np.nan


def unit_trend_pval(vector):
    x = np.arange(len(vector))
    y = vector
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    if len(y) > 3:
        result = linregress(x, y)
        return np.array([result.slope, result.pvalue])
    else:
        return np.array([np.nan, np.nan])


####
def _relative_entropy(vector, r_max):
    r"""
    Calculates the relative entropy of the climatology of precipitation.
    
    Parameters
    ----------
    vector: 1-d array
        n * 12-element vector corresponding to a 12-month climatology of the
        hydrological year.
    r_max: float
        whether to normalize the annual total by the global maximum in order to
        distinguish between different locations across the world
        (norm_frac = 7932 mm yr^{-1} at Tabubil, Papua New Guinea in the 
        original paper).

    Feng, X., A. Porporato, and I. Rodriguez-Iturbe, 2013: Changes in 
        rainfall seasonality in the tropics. Nature Clim Change, 3, 811-815,
        https://doi.org/10.1038/nclimate1907.
    """
    vector = vector.reshape(-1, 12)

    frac_month = vector / np.sum(vector, axis = 1, keepdims = True)
    if not (r_max is None):
        frac_month *= r_max

    # handle zero values when all seasons are zero: uniform
    frac_month[np.sum(vector, axis = 1) < 1e-10, :] = 1 / 12

    temp = np.log2(frac_month * 12)
    temp[~np.isfinite(temp)] = 0. # handle zero values
    
    D = np.sum(frac_month * temp, axis = 1, keepdims = False)
    return D


def longterm_seasonality(vector, r_max):
    r"""
    The long-term seasonality index defined in

    Feng, X., A. Porporato, and I. Rodriguez-Iturbe, 2013: Changes in
        rainfall seasonality in the tropics. Nature Clim Change, 3, 811-815,
        https://doi.org/10.1038/nclimate1907.
    
    Meausures the evenness of the distribution of precipitation over months.
    Uses relative entropy and ranges between 0 and log_2(12) = 3.585, with 
    larger values meaning that precipitation is more concentrated in the
    wet season.

    Parameters
    ----------
    vector: 1-d array
        n * 12-element vector corresponding to a 12-month climatology of the
        hydrological year.
    r_max: float
        whether to normalize the annual total by the global maximum in order to
        distinguish between different locations across the world
        (norm_frac = 7932 mm yr^{-1} at Tabubil, Papua New Guinea in the 
         original paper).
    """
    D = _relative_entropy(vector, r_max)
    S = D * np.sum(vector.reshape(-1, 12), axis = 1) / r_max
    return S


def centroid(vector):
    r"""
    Measures the timing of rainfall using the first moment.

    Feng, X., A. Porporato, and I. Rodriguez-Iturbe, 2013: Changes in
        rainfall seasonality in the tropics. Nature Clim Change, 3, 811-815,
        https://doi.org/10.1038/nclimate1907.

    Parameters
    ----------
    vector: 1-d array
        n * 12-element vector corresponding to a 12-month climatology of the
        hydrological year.
    """
    vector = vector.reshape(-1, 12)
    C = np.sum( vector * np.arange(1, 13).reshape(1, -1), axis = 1 ) \
        / np.sum(vector, axis = 1)

    # handle zero values when all seasons are zero: uniform
    C[np.std(vector, axis = 1) < 1e-10] = np.mean(np.arange(1, 13))
    return C


def spread(vector, C = None):
    r"""
    Measures the duration of the wet season using the spread around the
    centroid.

    Feng, X., A. Porporato, and I. Rodriguez-Iturbe, 2013: Changes in
        rainfall seasonality in the tropics. Nature Clim Change, 3, 811-815,
        https://doi.org/10.1038/nclimate1907.

    Parameters
    ----------
    vector: 1-d array
        12-element vector corresponding to a 12-month climatology of the
        hydrological year.
    """
    if C is None:
        C = centroid(vector)
    vector = vector.reshape(-1, 12)
    Z = np.sqrt( np.sum( np.power( np.arange(1,13).reshape(1,-1) - \
                                   C.reshape(-1, 1), 2 ) * \
                         vector, axis = 1 ) / np.sum(vector, axis = 1) )
    # handle zero values when all seasons are zero: uniform
    Z[np.sum(vector, axis = 1) < 1e-10] = np.std(np.arange(1, 13))
    return Z


def entropic_spread(vector):
    r"""
    An information theory-based measure for the support of the monthly
    rainfall distribution in each year. Therefore, it is a measure of the
    duration of the rainy season and is defined based on the information 
    entropy of each year.
    
    Analogous to spread.

    Feng, X., A. Porporato, and I. Rodriguez-Iturbe, 2013: Changes in
        rainfall seasonality in the tropics. Nature Clim Change, 3, 811-815,
        https://doi.org/10.1038/nclimate1907.

    Parameters
    ----------
    vector: 1-d array
        12-element vector corresponding to a 12-month climatology of the
        hydrological year.
    """
    frac_month = vector / np.sum(vector)
    H = - np.sum( frac_month * np.log2(frac_month) ) # information entropy
    E = np.sqrt( (np.power(2, 2 * H) - 1) / 12 )
    return E


def demodulated_amplitude_n_phase(x, time_series):
    r"""
    The demodulated amplitude is akin to mean, and the demodulated phase is
    akin to centroid. But they are derived based on localized harmonic 
    analysis. 

    The time series was multiplied by two sinusoidal functions, a low pass 
    filter applied to isolate the low frequencies from the high frequencies, 
    and the amplitude and phase of the low frequency component identified. 
    
    Also, it appears, based on the unit test, that there is a burn-in period 
    in the phase estimation. The first 240-360 theta (20-30 years) won't be
    accurate.

    Feng, X., A. Porporato, and I. Rodriguez-Iturbe, 2013: Changes in
        rainfall seasonality in the tropics. Nature Clim Change, 3, 811-815,
        https://doi.org/10.1038/nclimate1907.

    Parameters
    ----------
    x: 1-d array
        Numeric time step; Must match with the phase of the time series, or the
        estimated theta will be phase-shifted. 
    time_series: 1-d array
        Time series of, e.g., monthly precipitation. Must be a multiple of 12.

    Unit test
    ----------
    n = 240
    trend = 0.05; trend2 = -0.00035
    x = np.arange(250, n + 250)
    time_series = 100 + 50 * trend * np.arange(n) * \
        np.cos(2 * np.pi / 12 * (x + trend2 * np.arange(n))) + \
        np.random.rand(n)
    A, theta = demodulated_amplitude_n_phase(x, time_series)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(x, trend * np.arange(n) * 50, '-r')
    ax[0].plot(x[::12], A, '-b')
    ax[1].plot(x, trend2 * np.arange(n), '-r')
    ax[1].plot(x[::12], theta, '-b')
    plt.close(fig)
    """
    T = 12

    # Transform
    y = time_series * np.exp(-1j * 2 * np.pi / T * x)
    
    # Hanning window
    weight = np.append(np.insert(np.ones(T - 1), 0, 0.5), 0.5)
    F = np.append(np.insert( 1/T * np.convolve(y, weight, mode = 'valid'), 0, 
                             np.full(T//2, np.nan) ), np.full(T//2, np.nan))
    A = 2 * np.sqrt( np.power(F.real, 2) + np.power(F.imag, 2) )
    theta = np.arctan( F.imag / F.real )

    # Resample to annual
    A = A[::12]
    theta = theta[::12] / np.pi / 2 * T # convert from angular frequency to
                                        # period

    return A, theta


