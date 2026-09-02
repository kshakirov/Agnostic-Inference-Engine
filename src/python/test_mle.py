from inference_core.mle.generator import generate_normal_stream, mean_std
from math import sqrt
import numpy as np

n = 10000
mu = 1000
sigma = 150
estms = []
viborka =1000
for i in range(viborka):
    s = generate_normal_stream(mu, sigma, n,i )
    m,std = mean_std(s)
#    print(m,std)
    estms.append((m,std))


means = [m for (m,std) in estms]
stds = [std for (m,std) in estms]

params_of_means = mean_std(means)
params_of_stds = mean_std(stds)

print(f"params of means {params_of_means}")
print(f"params of stds {params_of_stds}")

std_theoretic = sigma / sqrt(n)

print(std_theoretic)

fisher_info = 1 / sigma ** 2

fisher_info_n = fisher_info * n
se_fisher_info = 1 /sqrt(fisher_info_n)
print(f"Fisher Info is {fisher_info}, for n is {fisher_info_n}, SE Fisher Info {se_fisher_info}")

disps = [std*std for std in stds]
disps[0:5]
disp_mean = mean_std(disps)
disp_mean
