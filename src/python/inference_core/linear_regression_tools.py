import numpy as np
from inference_core.local_linear_regression_tools import local_linear_regression


x = np.array([1., 2., 3., 4.])
y = np.array([2., 3., 5., 4.])

def simple_linear_regression(x,y):

    x_0 = np.mean(x)
   # print(f"x_0 = {x_0}")

    d_vector =  x - x_0
    #print(f"d_i {d_vector}")
    d_vector_sqr = np.square(d_vector)
    #print(f"d_i ** 2 = {d_vector_sqr}")

    a = np.sum(d_vector_sqr)
    #print(f"A = {a}")

    b = len(x)
    #print(f"B = {b}")

    c =0

    d = -2 * np.sum(y * d_vector)
    #print(f"D = {d}")


    e = -2 * np.sum(y)
    #print(f"E = {e}")

    m_aa = 2 *a
    m_bb = 2 *b

    r_a  = - d
    r_b = - e

    A = r_a / m_aa
    B = r_b /m_bb - x_0 * A
    return A,B
    #print(f"solution is {A}x + {B}")
