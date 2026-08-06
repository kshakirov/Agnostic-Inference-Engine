import numpy as np
print("issue-2 estimating non-parametrically")

#in milliseconds
num = 10000
bad_num_1 = int(num * 0.0002)
bad_num_2 = int(num * 0.0023)

def generate_sample(mu, sigma, num, status):
    print(f"generate_sample: generating {num} samples with status [{status}]")
    samples = np.random.lognormal(mean=mu, sigma=sigma, size=num)
    return  [(status, s* 100) for s in sorted(samples)]

    
def calculate_empirically_cdf(sample, t):
    print(f"calculate_empirically_cdf: caculating cdf from [{len(sample)}] sample with threshold {t}")
    return len([s for s in sample if s[1] < t])

def calculate_plug_in_quantile(sample, q_value):
    print("calculate_plug_in_quantile: calculating ")


good_sample = generate_sample(0.0, 1.0, num, False)
print(good_sample[:5])

bad_1_sample = generate_sample(0.0, 10.0, bad_num_1, True)
#print(bad_1_sample)
bad_2_sample = generate_sample(0.0, 3.0, bad_num_2, True)
#print(bad_2_sample[:5])

distr =calculate_empirically_cdf(good_sample, 50)
print(distr)

quantile = calculate_plug_in_quantile(sample, 0.99)
