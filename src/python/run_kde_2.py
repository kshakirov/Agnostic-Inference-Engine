from  inference_core.wasserman import calculate_empirical_moments
from inference_core.wald_test_conf_interval import wald_test, confidence_interval
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


kernel_matrix = norm.pdf(diff_matrix / h)    
print(f"kde batch for h = 1 is {kernel_matrix}")
np.fill_diagonal(kernel_matrix,0)

kde_loop = np.sum(kernel_matrix, axis=1) / (h * (LENGTH -1))



ln_avg_h = np.mean(np.log(kde_loop))
print(f"mean log kde with h=1 {ln_avg_h}")


#print(f"testing kde of {x} is {norm_kernel(x)} ")

# for i in test_data:
#     print(f"kde for {i} is {norm_kernel(i)} ")

#print(f"for 2 we have 2 points with h = 1 x_i=1 {kde_part(diff_matrix, 2,1,1)} and x_i=3 {kde_part(diff_matrix,2, 3, 1)}")


#print("now the whole part")


# def kde_step(data, dm, x_i,h,n):
#     f_i=0
#     for y_i, y in  enumerate(data):
#         k_i = kde_part(dm, x_i, y_i, h)
#         #print(k_i,end=" ")
#         f_i+=k_i
#     f_i= 1/(h * n) * f_i
#     return f_i

# def kde(data,dm, h):
#     n=len(data)
#     for x_i, x in enumerate(data):
#         f_i =kde_step(data, dm, x_i, h, n)
#         print(f"for {x} we have kde = {f_i}")
#     return f_i

#kde_step(test_data, diff_matrix, 0, h, len(test_data))
#kde(test_data, diff_matrix, 1)

#x is notneedded 
def cross_validate_kde_step(data,dm, x,index,h):
    n = len(data)
#    return kde_step(data[:index] + data[index+1:],x, h,n - 1)
    return kde_step(np.delete(data, index),dm, index, h, n - 1)
#print(f"cross validation step for 0 is {cross_validate_kde_step(test_data,diff_matrix, 1,0,1)}")

def cross_validate_kde(data,dm,h):
    log_cvs=0
    n = len(data)
    for index in range(len(data)):
        cv= cross_validate_kde_step(data,dm, data[index], index, h)
        log_cvs += log(cv)
#        print(f"cross validation step for {data[index]} is {cv}, ln is {log(cv)}")
    return (log_cvs / n)

#ln_agv_h = cross_validate_kde(test_data,diff_matrix, 1)
#print(f"for h =1 avg ln is {ln_agv_h}")


def max_kde(data,dm):
    ln_agv_h =-100
    for i in range(len(data)):
        ln_i =cross_validate_kde(data,dm, data[i])
        if ln_i > ln_agv_h:
            ln_agv_h = ln_i
            
        print(f"for h ={data[i]} avg ln is {ln_i} but max is {ln_agv_h}")

#max_kde(test_data,diff_matrix)
