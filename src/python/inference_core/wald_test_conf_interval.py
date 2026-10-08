import numpy as np
from math import sqrt

def mean_difference_standard_error(data_1, data_2):
    length_1 = len(data_1)
    length_2 = len(data_2)
    std_1 = np.std(data_1)
    # parameters for the second half
    std_2 = np.std(data_2)
    return sqrt(std_1 **2/length_1 + std_2**2/length_2)


def wald_test(data_1,data_2):
    """ runs wald test on both collections must be of the same size """
    # parameters for the first half
    mean_1 = np.mean(data_1)
    mean_2 = np.mean(data_2)
    se = mean_difference_standard_error(data_1,data_2)
    return  (mean_1 - mean_2) / se


def conf_interval(data_1, data_2, critical_value=1.96):
    """ takes data1 and data 2 and SE"""
    mean_1 = np.mean(data_1)
    mean_2 = np.mean(data_2)
    means_diff = mean_1 - mean_2
    se = mean_difference_standard_error(data_1,data_2)
    confidence_interval = means_diff -  se * critical_value
    confidential_interval_r = means_diff + se * critical_value
    return confidence_interval, confidential_interval_r
