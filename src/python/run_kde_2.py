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


print("Choosing candidates")



def choose_h_candidates(h_star, r):
    cs = []
    ks = range(-r,r,1)
    print(ks)
    for k in ks:
        h_i= h_star * (2 ** (k/2))
        cs.append(h_i)
    return cs
    


        
def cross_validate_kde(diff_matrix, candidates):
    ln_agv_h =-100
    h_max = 0
    i_max = 0
    for i in range(len(candidates)):
        ln_i =kde(diff_matrix, candidates[i])
        if ln_i > ln_agv_h:
            ln_agv_h = ln_i
            h_max = candidates[i]
            i_max = i
            
        print(f"for h ={i} avg ln is {ln_i} but max is {ln_agv_h}")
    print(f"maximum is at h= {h_max} and is {ln_agv_h}")
    print(f"the closest are {candidates[i_max -1]},{candidates[i_max + 1]}")
    new_step = abs(candidates[i_max -1] - candidates[i_max + 1])/ 10.0
    print(new_step)
    for i in range(10):
        s = candidates[i_max - 1] + new_step * i
        ln_i =kde(diff_matrix, s)
        if ln_i > ln_agv_h:
            ln_agv_h = ln_i
            h_max = s

            
        print(f"for h ={s} avg ln is {ln_i} but max is {ln_agv_h}")
    print(f"maximum is at h= {h_max} and is {ln_agv_h}")
#searching for maximum not at the extremes
cross_validate_kde(diff_matrix, choose_h_candidates(h_star, 10))
