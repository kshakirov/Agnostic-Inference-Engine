from  inference_core.wasserman import calculate_empirical_moments
import numpy as np
from math import erf,sqrt,floor
from scipy.stats import norm



from inference_core.ks_tools import empirical_cdf, bootstrap,bootstrap_normal_step, partial_cdf_at_point, ks_distance,  bootstrap_p_value
from inference_core.utils import prep_data


################ЭМПИРИЧЕСКИЕ ДАННЫе #################
#констаны для получения данных

FILE_NAME = "/Users/kiryloshakirov/Documents/ChatGPT/Agnostic-Inference-Engine/experiments/latency/2026-08-31-baseline-a/baseline-a.csv"
LENGTH = 600
BOOTSTRAP_SIZE=1000


FILE_NAME_b = "/Users/kiryloshakirov/Documents/ChatGPT/Agnostic-Inference-Engine/experiments/latency/2026-09-24-baseline-b/baseline-b.csv"
#пока будет здесь может быть уникальна для типа данных 


np_data = prep_data(FILE_NAME, LENGTH)
mean = np.mean(np_data)

var  = np.var(np_data)
std = np.std(np_data)

print(f"For all set mean = {mean} var =  {var} std =  {std}")

#################### Wald test #################

np_data_b = prep_data(FILE_NAME_b, LENGTH)
mean_b = np.mean(np_data_b)

var_b  = np.var(np_data_b)
std_b = np.std(np_data_b)

print(f"For all set B mean = {mean_b} var =  {var_b} std =  {std_b}")






print("H0 is  mean = mean_b")

#mean_1 = np.mean(np_data[0:(floor(LENGTH/2))])
#std_1= np.std(np_data[0:(floor(LENGTH/2))])

#mean_2 = np.mean(np_data[(floor(LENGTH/2)):LENGTH])
#std_2 = np.std(np_data[(floor(LENGTH/2)):LENGTH])

se = sqrt(std_b **2/LENGTH + std**2/LENGTH)

w = (mean_b - mean) / se
print(f"mean = {mean}, mean_b = {mean_b} Wald is {w}")


############################ Wasserman by the book ######

print(f"Now computing the probability H0")



h0_prob = 1 - norm.cdf(5.738)
print(f"The probability to get Wald stats not less than 5.738 is equal {h0_prob}")
