# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

__all__ = ['gelman_rubin',
           'lagAutocorrelation',
           'geweke']
# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------ 
def gelman_rubin(x):    
    """
    Returns the Gelman-Rubin statistical test.
    
    """
    if np.shape(x) < (2,):
        raise ValueError(
            'Gelman-Rubin diagnostic requires multiple chains of the same length.')

    try:
        m, n = np.shape(x)
    except ValueError:
        print("Bad shape for the chains")
        return

    # Calculate between-chain variance
    B_over_n = np.sum((np.mean(x, 1) - np.mean(x)) ** 2) / (m - 1)

    # Calculate within-chain variances
    W = np.sum([(x[i] - xbar) ** 2 for i, xbar in enumerate(np.mean(x,1))]) / (m * (n - 1))

    # (over) estimate of variance
    s2 = W * (n - 1) / n + B_over_n

    # Pooled posterior variance estimate
    V = s2 + B_over_n / m

    # Calculate PSRF
    R = V / W

    return R
# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------ 
def lagAutocorrelation(chain, n):
    """
    """
    l = len(chain)
    #n = 10;
    lag = np.floor(np.array(np.linspace(0,0.75,n))*l) 
    mean = np.mean(chain)
    chainMean = chain-mean
    output = np.zeros(n)
    j = 0
    for k in lag:
        a = chainMean[:l-k]
        b = chainMean[k:]
        output[j] = np.sum(a*b) / np.sum(chainMean**2)
        j += 1
    return output

# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------ 
def geweke(x, first=.1, last=.5, intervals=20, maxlag=20):
    """Return z-scores for convergence diagnostics.

    Compare the mean of the first % of series with the mean of the last % of
    series. x is divided into a number of segments for which this difference is
    computed. If the series is converged, this score should oscillate between
    -1 and 1.

    Parameters
    ----------
    x : array-like
      The trace of some stochastic parameter.
    first : float
      The fraction of series at the beginning of the trace.
    last : float
      The fraction of series at the end to be compared with the section
      at the beginning.
    intervals : int
      The number of segments.
    maxlag : int
      Maximum autocorrelation lag for estimation of spectral variance

    Returns
    -------
    scores : list [[]]
      Return a list of [i, score], where i is the starting index for each
      interval and score the Geweke score on the interval.

    Notes
    -----

    The Geweke score on some series x is computed by:

      .. math:: \frac{E[x_s] - E[x_e]}{\sqrt{V[x_s] + V[x_e]}}

    where :math:`E` stands for the mean, :math:`V` the variance,
    :math:`x_s` a section at the start of the series and
    :math:`x_e` a section at the end of the series.

    References
    ----------
    Geweke (1992)
    """
    try:
        from statsmodels.regression.linear_model import yule_walker
        has_sm = True
    except ImportError:
        has_sm = False

    if not has_sm:
        print("statsmodels not available. Geweke diagnostic cannot be calculated.")
        return

    if np.ndim(x) > 1:
        return [geweke(y, first, last, intervals) for y in np.transpose(x)]

    # Filter out invalid intervals
    if first + last >= 1:
        raise ValueError(
            "Invalid intervals for Geweke convergence analysis",
            (first, last))

    # Initialize list of z-scores
    zscores = [None] * intervals

    # Starting points for calculations
    starts = np.linspace(0, int(len(x)*(1.-last)), intervals).astype(int)

    # Loop over start indices
    for i,s in enumerate(starts):

        # Size of remaining array
        x_trunc = x[s:]
        n = len(x_trunc)

        # Calculate slices
        first_slice = x_trunc[:int(first * n)]
        last_slice = x_trunc[int(last * n):]

        z = (first_slice.mean() - last_slice.mean())
        z /= np.sqrt(spec(first_slice)/len(first_slice) +
                     spec(last_slice)/len(last_slice))
        zscores[i] = len(x) - n, z

    return zscores

        