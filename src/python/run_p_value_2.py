from  inference_core.wasserman import calculate_empirical_moments
from inference_core.wald_test_conf_interval import wald_test, confidence_interval, p_value,power
import numpy as np
from math import erf,sqrt,floor
from scipy.stats import norm
from scipy.stats import norm



from inference_core.ks_tools import empirical_cdf, bootstrap,bootstrap_normal_step, partial_cdf_at_point, ks_distance,  bootstrap_p_value
from inference_core.utils import prep_data

FILE_NAME = "/Users/kiryloshakirov/Documents/ChatGPT/Agnostic-Inference-Engine/experiments/latency/2026-09-24-baseline-b/baseline-b.csv"

LENGTH = 600
np_data = prep_data(FILE_NAME, LENGTH)
LENGTH_HALF = LENGTH / 2

# checking if rows are in consecutive order 
data = np.genfromtxt(FILE_NAME, delimiter=',')
timestamp = data[:,0]
for idx in range(len(timestamp)- 1 ):
    if timestamp[idx] > timestamp[idx + 1]:
        print(f"order is broken for {idx} {timestamp[idx]} vs {timestamp[idx + 1]}")

    
# only one row wasn't in order


np_data_1=np_data[0:300]
np_data_2=np_data[300:600]

# parameters for the first half
mean_1 = np.mean(np_data_1)
var_1  = np.var(np_data_1)
std_1 = np.std(np_data_1)


# parameters for the second half
mean_2 = np.mean(np_data_2)
var_2  = np.var(np_data_2)
std_2 = np.std(np_data_2)


se = sqrt(std_1 **2/LENGTH_HALF + std_2**2/LENGTH_HALF)

w = (mean_1 - mean_2) / se
print(f"mean_a = {mean_1}, mean_b = {mean_2}, std_a is {std_1}, std_b is {std_2} Wald  is {w}")


h0_prob = 1 - norm.cdf(w)
print(f"The probability to get Wald stats not less than {w} is equal {h0_prob}")


means_diff = mean_1 - mean_2
interval_l =means_diff - se 
interval_r = means_diff + se 


confidential_interval_l = means_diff -  se * 1.96
confidential_interval_r = means_diff + se * 1.96 

print(f"building interval 0.95 means we take the diff of means {means_diff}  the real in millis  width interval  is  from {interval_l} to {interval_r}, but confidentiality interval is 1.96 times real interval, thus we return back to millis from abstract Wald number from {confidential_interval_l} to {confidential_interval_r} ")


print(f"calculating the Wald test power  for 2 ms  instead of mean_diff/SE we take 2 ms/ SE")

wt_i = 2 / se
left_interval_i = -1.96 - wt_i
left_zeros_i = norm.cdf(left_interval_i)
right_interval_2 = 1.96  - wt_i
right_zeros_i = 1 - norm.cdf(right_interval_2)
print(f"It is {wt_i}, now we calculate  how probablity to get 0 mean diff changed, the same way by comparing intervalus for 0 we have -1.96 and + 1.96. Now the center move to 1.03 so the original interval is at -1.96 but the new one from the left is at - {left_interval_i}. It still contains zero but the probability to get it decreased we should take now CDF for values less then {left_interval_i}. It is  {left_zeros_i}. \n Now we do the same with the right interval  right interval now is {right_interval_2} the cdf is {right_zeros_i}" )



grades = np.linspace(0.0, 5, 10)     # 11 точек, 0.0 .. 1.0 включительно
for i in grades:
    wt_i = i / se
    left_interval_i = -1.96 - wt_i
    left_zeros_i = norm.cdf(left_interval_i)
    right_interval_2 = 1.96  - wt_i
    right_zeros_i = 1 - norm.cdf(right_interval_2)
    prob =  left_zeros_i + right_zeros_i
    print(f"power({i}) = {prob}")


print(f"testing module ")

w_test = wald_test(np_data_1, np_data_2)

conf_int = confidence_interval(np_data_1, np_data_2, 1.96)

p_v = p_value(w_test)

pw_2 = power(2,np_data_1,np_data_2,1.96)
pw_0 = power(0,np_data_1,np_data_2,1.96)

print(f" w test {w_test}, confidence interval is {conf_int} p value is {p_v} power for 2 is {pw_2} power for 0 is {pw_0}")
