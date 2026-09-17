import numpy as np
from math import sqrt
from scipy.stats import weibull_min
from inference_core.ks_tools import empirical_cdf, bootstrap, ks_distance,  bootstrap_p_value, bootstrap_weibull_step
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


##################### Подбираем параметры  то бишь fit

k_hat, loc, lam_hat = weibull_min.fit(np_data, floc=0)

print(f"params k_hat, loc, lam_hat  {k_hat, loc, lam_hat }")


#################### KS distance то бишь по Колмоговорову

w_cdf = weibull_min.cdf(x,  k_hat, loc, lam_hat )

d_star_real = ks_distance(w_cdf, y)

print(f"D real  is  {d_star_real}")


b_d_stars = bootstrap(BOOTSTRAP_SIZE, (k_hat, lam_hat), LENGTH,bootstrap_weibull_step)

#print(b_d_stars[10:20])

ro =bootstrap_p_value(b_d_stars, d_star_real)

print(f"ro is {ro} ")
