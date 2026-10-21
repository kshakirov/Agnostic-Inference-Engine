from  inference_core.wasserman import calculate_empirical_moments
from inference_core.wald_test_conf_interval import wald_test, confidence_interval
from inference_core.ks_tools import empirical_cdf
from inference_core.nonparametric import calculate_plug_in_quantile
import numpy as np
from math import erf,sqrt,floor,log
from scipy.stats import norm


from inference_core.utils import prep_data

FILE_NAME_1 = "/Users/kiryloshakirov/Documents/ChatGPT/Agnostic-Inference-Engine/experiments/latency/sclerotix/post_index_16b/20261019T034154Z/run-001/samples.csv"

LENGTH = 600
test_data = prep_data(FILE_NAME_1, LENGTH)






diff_matrix = test_data[:, None] - test_data[None, :]
#print(diff_matrix)

h = 1

def kde(dm,h):
    kernel_matrix = norm.pdf(dm / h)
#    print(f"kde batch for h = 1 is {kernel_matrix}")
    np.fill_diagonal(kernel_matrix,0)

    kde_loop = np.sum(kernel_matrix, axis=1) / (h * (LENGTH -1))



    return np.mean(np.log(kde_loop))

ln_avg_h = kde(diff_matrix, h)
print(f"mean log kde with h=1 {ln_avg_h}")

for i  in [0.02, 0.05, 0.1, 0.2, 0.4]:
    ln_avg_h = kde(diff_matrix, i)
    print(f"mean log kde with h={i} {ln_avg_h}")


x,y = empirical_cdf(test_data)
#print(f"cdf is {y}")
q25 = calculate_plug_in_quantile(test_data, 0.25)

q75 = calculate_plug_in_quantile(test_data, 0.75)

IQR = q75 - q25

s = np.std(test_data)


h_star = 0.9 * min(s, IQR / 1.34) * LENGTH **(-0.2)
print(f"h* is {h_star}")
