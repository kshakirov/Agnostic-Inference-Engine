from inference_core.estimation.linear_regression_tools import simple_linear_regression
import numpy as np

rng = np.random.default_rng()
x = rng.integers(low=0, high=1000, size=100)   # 10 целых в [0, 100)

n_repititions = 100
a_hat =np.zeros(n_repititions)
b_hat = np.zeros(n_repititions)
for i in range(n_repititions):
    rng_normal = np.random.default_rng()
    v = rng_normal.standard_normal(size=100)   # N(0, 1), 10 элементов
    y = 2*x + 5 + v
    A,B = simple_linear_regression(x, y)
    a_hat[i]= A
    b_hat[i] =B
    print(f"A = {A},B={B}")

bias_a = np.mean(a_hat) - 2
bias_b = np.mean(b_hat ) - 5

print(f"bias A = {bias_a}, bias B={bias_b}")
