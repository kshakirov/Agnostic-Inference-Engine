import numpy as np
from math import ceil,floor

    
def calculate_empirically_cdf(sample, t):
    print(f"calculate_empirically_cdf: caculating cdf from [{len(sample)}] sample with threshold {t}")
    return len([s for s in sample if s[1] <= t]) / len(sample)

def calculate_plug_in_quantile(sample, q_value):
    if len(sample) == 0:
        raise ValueError("sample must not be empty")
    if not 0 <= q_value <= 1:
        raise ValueError("q_value must be between 0 and 1")

    index = min(max(ceil(q_value * len(sample)) - 1, 0), len(sample) - 1)
    return sorted(sample)[index]

def calculate_plug_in_conf_interval(sample, confidentiality):
    """  рассчет дов интервала для бутстрапа"""
    v = (1 - confidentiality)/2
    l_end =     min(ceil(v * len(sample)),len(sample)-1)
    r_end =     min(ceil((1 -v ) * len(sample)),len(sample)-1)
    sorted_sample = sorted(sample)
    return sorted_sample[l_end], sorted_sample[r_end]


