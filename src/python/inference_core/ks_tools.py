
import numpy as np
import array
import matplotlib.pyplot as plt
from math import erf,sqrt
from scipy.stats import lognorm

def empirical_cdf(sample):
    """ caclulates empirical cdf"""
    x = np.sort(sample)
    y_direct = np.arange(1, len(x) +1 )
    y_len = np.full(len(x),len(x))
    y_i = np.divide(y_direct,y_len)

    return x,y_i
