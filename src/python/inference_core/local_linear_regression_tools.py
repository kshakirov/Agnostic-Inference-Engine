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

def local_linear_regression_weighted(x,y,hs):

    d_m = -1 * difference_matrix(x, x)

   #print(d_m)
    np.set_printoptions(suppress=True)
    for h in hs:
        #  h = 1
        W = np.exp(-(d_m ** 2) / (2 * h ** 2))
        np.fill_diagonal(W,0)
        #print(W)
        a_j = np.sum(np.multiply(W, np.square(d_m)),axis=1)
        #print(a_j)
        b_j = np.sum(W,axis=1)
        c_j = np.sum((2 * np.multiply(d_m,W)), axis=1)
        #print(c_j)
        d_j = - 2 * np.sum( y * d_m * W, axis=1)
        #print(d_j)
        e_j = -2 * np.sum(y * W, axis=1)
        #print(e_j)
        f_j= np.sum(np.square(y) * W, axis=1)
        #print(f_j)
        mA = np.array([
            [2*a_j, c_j],
            [c_j, 2*b_j]
        ]).transpose(2, 0, 1)

        rhs = np.array([
            -d_j,
            -e_j
        ]).T

        solution = np.linalg.solve(
            mA,
            rhs[..., None]
        ).squeeze(-1)

        #  print(solution)

        prediction = solution[:,1]
        #print(prediction)


        cv = np.average( np.square(np.subtract(y,prediction)))
        print(f"h={h}, cv= {cv} max cond = {np.max(np.linalg.cond(mA))}")
