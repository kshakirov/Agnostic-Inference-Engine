
import numpy as np
import array
import matplotlib.pyplot as plt
from math import erf,sqrt
from scipy.stats import gamma

def empirical_cdf(sample):
    """ caclulates empirical cdf"""
    x = np.sort(sample)
    y_direct = np.arange(1, len(x) +1 )
    y_len = np.full(len(x),len(x))
    y_i = np.divide(y_direct,y_len)

    return x,y_i

def normal_cdf_at_point(x, mu_hat, sigma_hat):
    z = (x - mu_hat) /sigma_hat
    return 0.5 * ( 1 + erf(z/sqrt(2)))

def partial_cdf_at_point(mu_hat, sigma_hat):
    def ncdf_at_point(x):
        z = (x - mu_hat) /sigma_hat
        return 0.5 * ( 1 + erf(z/sqrt(2)))
    return ncdf_at_point

def partial_gammf_cdf_at_point(mu_hat, sigma_hat):
    def gamma_cdf_at_point(x):
        return gamma.cdf(x, a=mu_hat, scale=sigma_hat)
    return gamma_cdf_at_point


def get_dstar(n_x,y):
#    d_after_array = np.abs(np.subtract(n_x, y))
    length = len(y)
    d_after = np.max(np.abs(np.subtract(n_x, y)))
    y_before = np.arange(length) *  1/length
    d_before_array = np.abs(np.subtract(n_x, y_before))
    d_before = np.max(np.abs(np.subtract(n_x, y_before)))
    d_real = max(d_before,d_after)
    return d_real


# def bootstrap_dstar(params,length, bootstrap_size):
#     d_star_one = bootstrap_normal_step(*params, length)
#     d_star_s =bootstrap(bootstrap_size, params, length)
#     d_star_max = np.max(d_star_s)
#     return d_star_s 


def bootstrap_normal_step(mu,sigma, length):
    "из названия понятно бутcтрап нормального распределиния один шаг"
    rng = np.random.default_rng()
    np_array = rng.normal(loc=mu, scale=sigma, size=length)
    x_cdf,y_after_cdf= empirical_cdf(np_array)
    m = np.mean(np_array)
    s = np.std(np_array)
    ncdf_at_point = partial_cdf_at_point(m, s)
    vectorized_func = np.vectorize(ncdf_at_point)
    n_x = vectorized_func(x_cdf)
    return get_dstar(n_x, y_after_cdf)



def bootstrap_gamma_step(mu,sigma, length):
    "из названия понятно бутcтрап нормального распределиния один шаг"
    rng = np.random.default_rng()
    np_array = rng.gamma(shape=mu, scale=sigma, size=length)
    x_cdf,y_after_cdf= empirical_cdf(np_array)
    m, loc, s = gamma.fit(np_array, floc=0)
    n_x = gamma.cdf(x_cdf, a=m,scale=s)
    return get_dstar(n_x, y_after_cdf)



def bootstrap(size, params,length,step=bootstrap_normal_step):
    d_s = np.full(size,0.0)
    for i in range(size):
        d_s[i] = step(*params, length)
    return d_s


#later change for names n_x 


def get_ro(d_star_s, d_real):
    return (1 + len([d for d in d_star_s if d >= d_real])) / (len(d_star_s) + 1)


