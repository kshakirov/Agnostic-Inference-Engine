import numpy as np
from math import ceil,floor

    
def calculate_empirically_cdf(sample, t, total):
    print(f"calculate_empirically_cdf: caculating cdf from [{len(sample)}] sample with threshold {t}")
    return len([s for s in sample if s[1] <= t]) / len(sample)

def calculate_plug_in_quantile(sample, q_value):
    index =     min(ceil(q_value * len(sample)),len(sample)-1)
    return sorted(sample)[index]


