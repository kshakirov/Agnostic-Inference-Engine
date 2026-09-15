import numpy as np
import array
from math import erf,sqrt
from scipy.stats import lognorm
from inference_core.ks_tools import empirical_cdf, bootstrap_normal,bootstrap_normal_step, partial_cdf_at_point, get_dstar, bootstrap_dstar, get_ro, prep_data

################ЭМПИРИЧЕСКИЕ ДАННЫе #################
#констаны для получения данных

FILE_NAME = "/Users/kiryloshakirov/Documents/ChatGPT/Agnostic-Inference-Engine/experiments/latency/2026-08-31-baseline-a/baseline-a.csv"
LENGTH = 600
BOOTSTRAP_SIZE=1000


#пока будет здесь может быть уникальна для типа данных 


np_data = prep_data(FILE_NAME, LENGTH)
mean = np.mean(np_data)

var  = np.var(np_data)
std = np.std(np_data)

print(f"mean = {mean}\n var =  {var}\n std =  {std}")


x, y = empirical_cdf(np_data)
print(f"ecdf x 10  = {x[0:10]} y 10 = {y[0:10]}")





##################LOGNORMAL STARTD #########
