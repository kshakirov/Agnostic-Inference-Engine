from  inference_core.wasserman import calculate_empirical_moments
from inference_core.wald_test_conf_interval import wald_test, confidence_interval
import numpy as np
from math import erf,sqrt,floor,log
from scipy.stats import norm


from inference_core.utils import prep_data

FILE_NAME_TEMPLATE = "/Users/kiryloshakirov/Documents/ChatGPT/Agnostic-Inference-Engine/experiments/latency/sclerotix/post-body-matrix/body-16b-run-001.csv"

LENGTH = 6
np_data_1 = prep_data(FILE_NAME_TEMPLATE, LENGTH)
print(np_data_1)
