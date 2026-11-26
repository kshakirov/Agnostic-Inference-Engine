import numpy as np
from math import sqrt
from scipy.stats import norm

def mean_difference_standard_error(data_1, data_2):
    length_1 = len(data_1)
    length_2 = len(data_2)
    std_1 = np.std(data_1)
    # parameters for the second half
    std_2 = np.std(data_2)
    return sqrt(std_1 **2/length_1 + std_2**2/length_2)


def wald_test(data_1,data_2):
    """ runs wald test on both collections"""
    # parameters for the first half
    mean_1 = np.mean(data_1)
    mean_2 = np.mean(data_2)
    se = mean_difference_standard_error(data_1,data_2)
    return  (mean_1 - mean_2) / se


def confidence_interval(data_1, data_2, critical_value=1.96):
    """ takes data1 and data 2 and SE"""
    mean_1 = np.mean(data_1)
    mean_2 = np.mean(data_2)
    means_diff = mean_1 - mean_2
    se = mean_difference_standard_error(data_1,data_2)
    confidence_interval_l = means_diff -  se * critical_value
    confidence_interval_r = means_diff + se * critical_value
    return confidence_interval_l, confidence_interval_r

def p_value(wald):
    return 2 * norm.cdf(-(abs(wald)))

def power(val, data_1, data_2, critical_value=1.96):
    se = mean_difference_standard_error(data_1, data_2)
    wt_i = val / se
    left_interval_i = - critical_value - wt_i
    left_zeros_i = norm.cdf(left_interval_i)
    right_interval_2 = critical_value  - wt_i
    right_zeros_i = 1 - norm.cdf(right_interval_2)
    return left_zeros_i + right_zeros_i
