import numpy as np
from math import erf,sqrt,floor,log
from scipy.stats import norm

data = [(500,1.0), (900,1.4), (1100,1.8), (1500,2.5)]

def norm_kernel(x):
    return norm.pdf(x)

def kernel_regression_step(data,x,h):
    w = 0
    kw=0
    for p in  data:
        x_i,y = p
        w_i = norm_kernel((x - x_i)/h)
        w += w_i
        kw += w_i * y
#        print(f" {x} for {x_i} k({x_i}) = {w_i} kw_i = {w_i * y}")
    m= kw /w
#    print(f"m({x})={m}")
    return m
#m_avg(data, 1000,100)
#m_avg(data, 1000,500)
kernel_regression_step(data[:0]+ data[1:], 500, 100)

def kernel_regression(data,h):
    errs = []
    for index in range(len(data)):
        m_i = kernel_regression_step(data[:index] + data[index + 1:], data[index][0], h)
        err_i = m_i - data[index][1]
        #print(f"err_i is {err_i} for {data[index][0]}")
        errs.append(err_i)
    errs= np.array(errs)
    var = np.var(errs)
    print(f"leave out error for {h}  is {var}")


kernel_regression(data, 100)
kernel_regression(data, 500)

