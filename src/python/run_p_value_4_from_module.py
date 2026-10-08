from  inference_core.wasserman import calculate_empirical_moments
from inference_core.wald_test_conf_interval import wald_test, confidence_interval
import numpy as np
from math import erf,sqrt,floor
from scipy.stats import norm
from scipy.stats import norm

from inference_core.utils import prep_data

FILE_NAME_1 = "/Users/kiryloshakirov/Documents/ChatGPT/Agnostic-Inference-Engine/experiments/latency/2026-10-05-baseline-c/baseline-c.csv"

LENGTH = 600
np_data_1 = prep_data(FILE_NAME_1, LENGTH)

FILE_NAME_2 = "/Users/kiryloshakirov/Documents/ChatGPT/Agnostic-Inference-Engine/experiments/latency/2026-09-24-baseline-b/baseline-b.csv"


np_data_2 = prep_data(FILE_NAME_2, LENGTH)


w = wald_test(np_data_1, np_data_2)
print(f" Wald  is {w}")


h0_prob = 1 - norm.cdf(w)
print(f"The probability to get Wald stats not less than {w} is equal {h0_prob}")




confidential_interval_l,confidential_interval_r = confidence_interval(np_data_1, np_data_2, 1.96)


print(f"building interval 0.95 means we take the diff of means  the real in millis  width interval, but confidentiality interval is 1.96 times real interval, thus we return back to millis from abstract Wald number from {confidential_interval_l} to {confidential_interval_r} ")

