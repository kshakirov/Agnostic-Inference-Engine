from  inference_core.wasserman import calculate_empirical_moments
from inference_core.wald_test_conf_interval import wald_test, confidence_interval
import numpy as np
from math import erf,sqrt,floor,log
from scipy.stats import norm


from inference_core.utils import prep_data

FILE_NAME_1 = "/Users/kiryloshakirov/Documents/ChatGPT/Agnostic-Inference-Engine/experiments/latency/sclerotix/post_index_16b/20261019T034154Z/run-001/samples.csv"

LENGTH = 6
np_data_1 = prep_data(FILE_NAME_1, LENGTH)

#FILE_NAME_2 = "/Users/kiryloshakirov/Documents/ChatGPT/Agnostic-Inference-Engine/experiments/latency/2026-09-24-baseline-b/baseline-b.csv"


#np_data_2 = prep_data(FILE_NAME_2, LENGTH)


test_data = np.array([1,2,2,3,7])
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


def kde_step(data, x,h,n):
    f_i=0
    for y in data:
        k_i = kde_part(x, y, h)
        #print(k_i,end=" ")
        f_i+=k_i
    f_i= 1/(h * n) * f_i
    return f_i

def kde(data, h):
    n=len(data)
    for i in data:
        f_i =kde_step(data, i, h, n)
        print(f"for {i} we have kde = {f_i}")
    return f_i


kde(test_data, 1)


def cross_validate_kde_step(data, x,index,h):
    n = len(data)
#    return kde_step(data[:index] + data[index+1:],x, h,n - 1)
    return kde_step(np.delete(data, index), x, h, n - 1)
print(f"cross validation step for 0 is {cross_validate_kde_step(test_data, 1,0,1)}")

def cross_validate_kde(data,h):
    log_cvs=0
    n = len(data)
    for index in range(len(data)):
        cv= cross_validate_kde_step(data, data[index], index, h)
        log_cvs += log(cv)
#        print(f"cross validation step for {data[index]} is {cv}, ln is {log(cv)}")
    return (log_cvs / n)

ln_agv_h = cross_validate_kde(test_data, 1)
#print(f"for h =1 avg ln is {ln_agv_h}")


def max_kde(data):
    ln_agv_h =-100
    for i in range(len(data)):
        ln_i =cross_validate_kde(data, data[i])
        if ln_i > ln_agv_h:
            ln_agv_h = ln_i
            
        print(f"for h ={data[i]} avg ln is {ln_i} but max is {ln_agv_h}")
    
max_kde(test_data)
#max_kde(np_data_1)
