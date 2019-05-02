import numpy as np
import scipy as sp
import scipy.linalg


def solve(x_0: np.ndarray, func, jac, eps: np.float64):
    f = func(x_0)
    x = x_0
    d = np.full_like(x_0, 500)
    while sp.linalg.norm(d, ord=2) > eps:
        f = func(x)
        J = jac(x)
        d = sp.linalg.solve(J, f)
        x -= d
    return x

