from inference_core.utils import difference_matrix
import numpy as np

def local_linear_regression(x, y):
    LENGTH = len(x)
    h = 1
    d_m = -1 * difference_matrix(x, x)

    a_j = np.sum(np.square(d_m),axis=1)
    b_j = np.repeat(LENGTH, LENGTH)
    c_j = np.sum(np.multiply(d_m,2), axis=1)
    d_j = - 2 * np.sum( np.multiply(y,d_m), axis=1)
    e_j = -2 * np.sum(y)
    f_j= np.sum(np.square(y))
    mA = np.array([
        [2*a_j, c_j],
        [c_j, 2*b_j]
    ]).transpose(2, 0, 1)

    rhs = np.array([
        -d_j,
        np.full_like(d_j, -e_j)
    ]).T

    solution = np.linalg.solve(
        mA,
        rhs[..., None]
    ).squeeze(-1)

    return solution
