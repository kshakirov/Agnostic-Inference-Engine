import numpy as np
from math import erf,sqrt
from inference_core.ks_tools import empirical_cdf, bootstrap,bootstrap_normal_step, partial_cdf_at_point, get_dstar,  ks_distance
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

print(f"mean = {mean}\n var =  {var}\n std =  {std}")


x, y = empirical_cdf(np_data)
print(f"ecdf x 10  = {x[0:10]} y 10 = {y[0:10]}")





##################LOGNORMAL STARTD #########



x_log =np.log(x)

x_log_mean = np.mean(x_log)

x_log_var = np.var(x_log)

x_log_std = np.std(x_log)

print(f"x log, mean {x_log_mean}, var is {x_log_var} std is {x_log_std}")

print(f"building log normal cdf for each elem in ECDF we get its probability based on our x_log_mean and x_log_std, for it we use vectorized functionality of numpy")


lognorm_func = partial_cdf_at_point(x_log_mean, x_log_std)

lognorm_func_vectorized = np.vectorize(lognorm_func)
x_log_cdf =lognorm_func_vectorized(x_log)

d_star_real_log = get_dstar(x_log_cdf, y)


d_star_s_log = bootstrap(BOOTSTRAP_SIZE, (x_log_mean, x_log_std), LENGTH)


ro = ks_distance(d_star_s_log, d_star_real_log)

print(f"lognormal ro is {ro}")
