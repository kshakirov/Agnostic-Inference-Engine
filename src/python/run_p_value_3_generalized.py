from  inference_core.wasserman import calculate_empirical_moments
import numpy as np
from math import erf,sqrt,floor
from scipy.stats import norm
from scipy.stats import norm



from inference_core.ks_tools import empirical_cdf, bootstrap,bootstrap_normal_step, partial_cdf_at_point, ks_distance,  bootstrap_p_value
from inference_core.utils import prep_data

FILE_NAME_1 = "/Users/kiryloshakirov/Documents/ChatGPT/Agnostic-Inference-Engine/experiments/latency/2026-10-05-baseline-c/baseline-c.csv"

LENGTH = 600
np_data_1 = prep_data(FILE_NAME_1, LENGTH)

FILE_NAME_2 = "/Users/kiryloshakirov/Documents/ChatGPT/Agnostic-Inference-Engine/experiments/latency/2026-09-24-baseline-b/baseline-b.csv"


np_data_2 = prep_data(FILE_NAME_2, LENGTH)


# parameters for the first half
mean_1 = np.mean(np_data_1)
var_1  = np.var(np_data_1)
std_1 = np.std(np_data_1)


# parameters for the second half
mean_2 = np.mean(np_data_2)
var_2  = np.var(np_data_2)
std_2 = np.std(np_data_2)


se = sqrt(std_1 **2/LENGTH + std_2**2/LENGTH)

w = (mean_1 - mean_2) / se
print(f"mean_a = {mean_1}, mean_b = {mean_2}, std_a is {std_1}, std_b is {std_2} Wald  is {w}")


h0_prob = 1 - norm.cdf(w)
print(f"The probability to get Wald stats not less than {w} is equal {h0_prob}")


means_diff = mean_1 - mean_2
interval_l =means_diff - se 
interval_r = means_diff + se 


confidential_interval_l = means_diff -  se * 1.96
confidential_interval_r = means_diff + se * 1.96 

print(f"building interval 0.95 means we take the diff of means {means_diff}  the real in millis  width interval  is  from {interval_l} to {interval_r}, but confidentiality interval is 1.96 times real interval, thus we return back to millis from abstract Wald number from {confidential_interval_l} to {confidential_interval_r} ")

