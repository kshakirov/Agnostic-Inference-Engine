from  inference_core.wasserman import calculate_empirical_moments
from inference_core.wald_test_conf_interval import wald_test, confidence_interval
import numpy as np
from math import erf,sqrt,floor,log
from scipy.stats import norm
from inference_core.utils import difference_matrix

from inference_core.utils import prep_data


def file_template(size,run="001"):
    return f"/Users/kiryloshakirov/Documents/ChatGPT/Agnostic-Inference-Engine/experiments/latency/sclerotix/post-body-matrix/body-{size}b-run-{run}.csv"



LENGTH = 3
x = np.array([1, 2, 3])
y = np.array([1, 3, 3])



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
print(x)
print(y)



d_m = -1 * difference_matrix(x, x)

print(d_m)

h = 1
W = np.exp(-(d_m ** 2) / (2 * h ** 2))
print(W)
a_j = np.sum(np.multiply(W, np.square(d_m)),axis=1) 
print(a_j)


b_j = np.sum(W,axis=1)
c_j = np.sum((2 * np.multiply(d_m,W)), axis=1)
print(c_j)

d_j = - 2 * np.sum( y * d_m * W, axis=1)

print(d_j)


e_j = -2 * np.sum(y * W, axis=1)
print(e_j)


f_j= np.sum(np.square(y) * W, axis=1)
print(f_j)

#mA = np.array([[2*a_j, c_j],[c_j, 2*b_j]]).transpose(2,0,1)
#b = np.array([-d_j, -e_j])


mA = np.array([
    [2*a_j, c_j],
    [c_j, 2*b_j]
]).transpose(2, 0, 1)

rhs = np.array([
    -d_j,
    -e_j
]).T

solution = np.linalg.solve(
    mA,
    rhs[..., None]
).squeeze(-1)

print(solution)
print(d_m)
print(W)
