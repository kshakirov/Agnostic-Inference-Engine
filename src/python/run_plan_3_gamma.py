import numpy as np
import array
from math import erf,sqrt
from scipy.stats import gamma
from inference_core.ks_tools import empirical_cdf, bootstrap,bootstrap_normal_step, partial_cdf_at_point, get_dstar, bootstrap_dstar, get_ro, prep_data, bootstrap_gamma_step

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





##################GAMMA  STARTD #########
# подбираем сначала параметры

shape, loc, scale = gamma.fit(np_data, floc=0)

print(f"shape is {shape} loc is {loc} scale {scale}")


k_mean = shape * scale

k_var= shape * scale * scale
k_std = sqrt(k_var)

print(f"mean is {k_mean} k_std {k_var} k_std {k_std}")


################## KS #####################
g_cdf = gamma.cdf(x,shape,loc,scale)

d_star_real = get_dstar(g_cdf, y)

print(f"d_star is {d_star_real}")

b_d_star = bootstrap_gamma_step(shape, scale, LENGTH)

print(f"b d_star is {b_d_star}")

b_d_stars = bootstrap(BOOTSTRAP_SIZE, shape, scale, LENGTH,bootstrap_gamma_step)

print(b_d_stars[10:20])

ro =get_ro(b_d_stars, d_star_real)

print(f"ro is {ro} ")
