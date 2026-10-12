from  inference_core.wasserman import calculate_empirical_moments
from inference_core.wald_test_conf_interval import wald_test, confidence_interval
import numpy as np
from math import erf,sqrt,floor
from scipy.stats import norm


from inference_core.utils import prep_data

FILE_NAME_1 = "/Users/kiryloshakirov/Documents/ChatGPT/Agnostic-Inference-Engine/experiments/latency/2026-10-05-baseline-c/baseline-c.csv"

LENGTH = 600
np_data_1 = prep_data(FILE_NAME_1, LENGTH)

FILE_NAME_2 = "/Users/kiryloshakirov/Documents/ChatGPT/Agnostic-Inference-Engine/experiments/latency/2026-09-24-baseline-b/baseline-b.csv"


np_data_2 = prep_data(FILE_NAME_2, LENGTH)


test_data = [1,2,2,3,7]
x =2
h=1
def norm_kernel(x):
    return norm.pdf(x)

def kde_part(x,x_i,h):
    """ exclude x  from data, build a hill over x """
    return norm_kernel((x - x_i) /h)
    

print(f"testing kde of {x} is {norm_kernel(x)} ")

for i in test_data:
    print(f"kde for {i} is {norm_kernel(i)} ")

print(f"for 2 we have 2 points with h = 1 x_i=1 {kde_part(2,1,1)} and x_i=3 {kde_part(2, 3, 1)}")

print("now the whole part")


for i in test_data:
    n=len(test_data)
    h=1
    f_i=0
    for y in test_data:
        k_i = kde_part(i, y, 1)
        #print(k_i,end=" ")
        f_i+=k_i
    f_i= 1/(h * n) * f_i
    print(f"for {i} we have kde = {f_i}")
