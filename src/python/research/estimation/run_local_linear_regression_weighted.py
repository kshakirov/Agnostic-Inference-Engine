from  inference_core.estimation.wasserman import calculate_empirical_moments
from inference_core.uncertainty_quantification.wald_test_conf_interval import wald_test, confidence_interval
import numpy as np
from math import erf,sqrt,floor,log
from scipy.stats import norm
from inference_core.utils import difference_matrix

from inference_core.utils import prep_data
from inference_core.estimation.local_linear_regression_tools import local_linear_regression_weighted

def file_template(size,run="001"):
    return f"/Users/kiryloshakirov/Documents/ChatGPT/Agnostic-Inference-Engine/experiments/latency/sclerotix/post-body-matrix/body-{size}b-run-{run}.csv"


body_sizes = np.array([
    16, 32, 64, 128, 256, 512, 1024,
    2048, 4096, 8192, 16384, 32768, 65536,
])


def log_body_sizes(b_s):
    return np.log2(b_s)

LENGTH = 1000
xx = []
for s in body_sizes:
    d = prep_data(file_template(s),LENGTH)
    xx.append(np.average(d))
x = log_body_sizes(body_sizes)
y = np.array(xx)
#print(x)
#print(y)


local_linear_regression_weighted(x, y, [0.1,0.2,0.4,0.8,1,2,3,4,5,6])
