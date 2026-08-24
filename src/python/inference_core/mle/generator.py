import numpy as np

def generate_normal_stream(mu,sigma,size, seed):
    """ generating normal stream """
    rng=     np.random.default_rng(seed)
    return  rng.normal(loc=mu, scale=sigma, size=size)

def mean_std(data):
    return  np.mean(data), np.std(data)

def add_noise(data, mu, level):
    """ adding noise """
    noise_abs = level * mu
    noisy_data = np.random.normal(0, noise_abs / 2, len(data))
    return np.add(data, noisy_data)
