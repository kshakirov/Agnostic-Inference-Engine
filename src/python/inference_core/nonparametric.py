import numpy as np
from math import ceil,floor

    
def calculate_empirically_cdf(sample, t, total):
    print(f"calculate_empirically_cdf: caculating cdf from [{len(sample)}] sample with threshold {t}")
    return len([s for s in sorted(sample) if s[1] < t]) / total

def calculate_plug_in_quantile(sample, q_value):
        
    return sorted(sample, key=lambda i: i[1])[floor(q_value * len(sample))]


