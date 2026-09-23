from  inference_core.wasserman import calculate_empirical_moments
import numpy as np
from math import erf,sqrt,floor

from inference_core.ks_tools import empirical_cdf, bootstrap,bootstrap_normal_step, partial_cdf_at_point, ks_distance,  bootstrap_p_value
from inference_core.utils import prep_data


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

print(f"For all set mean = {mean} var =  {var} std =  {std}")

#################### Wald test #################

print("H0 is  mean1 = mean2")

mean_1 = np.mean(np_data[0:(floor(LENGTH/2))])
std_1= np.std(np_data[0:(floor(LENGTH/2))])

mean_2 = np.mean(np_data[(floor(LENGTH/2)):LENGTH])
std_2 = np.std(np_data[(floor(LENGTH/2)):LENGTH])

se = sqrt(std_1**2/floor(LENGTH/2) + std_2**2/floor(LENGTH/2))

w = (mean_1 - mean_2) / se
print(f"mean_1 = {mean_1}, mean_2 = {mean_2} Wald is {w}")


