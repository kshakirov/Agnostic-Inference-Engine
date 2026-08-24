from  inference_core.bootstrap import get_metrics, bootstrap_naive
from inference_core.nonparametric import calculate_empirically_cdf, calculate_plug_in_quantile, calculate_plug_in_conf_interval
from pathlib import Path

metrics = get_metrics("samples.dat")
good_metrics = [m for m in metrics if not m[0] ]# у нас метрика наоборот это метрика падений, ну да бог  с ним


print(f"moment quantile is {calculate_plug_in_quantile(good_metrics, 0.99)}")
print(f"moment cdf for t = 163 is {calculate_empirically_cdf(good_metrics, 163)}")

def curried_calculate_empirically_cdf(samples):
    return calculate_empirically_cdf(samples, 9.0)

def curried_calculate_plug_in_quantile(samples):
    return  calculate_plug_in_quantile(samples, 0.99)

b_samples = bootstrap_naive(10,  good_metrics, curried_calculate_plug_in_quantile)


quantile = calculate_plug_in_quantile(b_samples, 0.5)
print(f"Monte Carlo quantile is {quantile}")

l_end, r_end = calculate_plug_in_conf_interval(b_samples, 0.95)
print(l_end,r_end)
