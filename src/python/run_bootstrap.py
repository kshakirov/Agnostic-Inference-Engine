from  inference_core.bootstrap import get_metrics, bootstrap_naive
from inference_core.nonparametric import calculate_empirically_cdf, calculate_plug_in_quantile
from pathlib import Path

metrics = get_metrics("samples.dat")
good_metrics = [m for m in metrics if not m[0] ]# у нас метрика наоборот это метрика падений, ну да бог  с ним
good_metrics[89:100]

print(f"moment quantile is {calculate_plug_in_quantile(good_metrics, 0.99)}")
print(f"moment cdf for t = 163 is {calculate_empirically_cdf(good_metrics, 163)}")

def carried_calculate_empirically_cdf(samples):
    return calculate_empirically_cdf(samples, 9.0)

def carried_calculate_plug_in_quantile(samples):
    return  calculate_plug_in_quantile(samples, 0.99)

b_samples = bootstrap_naive(1000,  good_metrics[0:10000], carried_calculate_plug_in_quantile)


quantile = calculate_plug_in_quantile(b_samples, 0.99)
print(f"Monte Carlo quantile is {quantile}")
