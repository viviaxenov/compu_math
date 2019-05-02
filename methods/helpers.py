import numpy as np
import scipy as sp
import scipy.linalg


def verify_args(A: np.ndarray, f: np.ndarray) -> bool:
    shape = A.shape
    if shape[0] != shape[1]:
        raise ValueError("Matrix should have shape [n,n]; Having [{0:d},{1:d}]".format(*shape))
    if shape[0] != f.shape[0]:
        raise ValueError("Matrix and right part should have same size; Having {0:d} and {1:d}"
                         .format(shape[0], f.shape[0]))
    return True


def linear_ode_analytical(A: np.ndarray, y_0: np.ndarray):
    """"
    Gets analytical solution for ODE system dy/dt = Ay, y(0) = y_0, where A=const
    Args:
        A - right part matrix
        y_0 - initial condition
    Returns:
        u - function  : y(t)
    Warning: works only if A has basis of real eigenvalues. Other cases weren't considered
    """
    eigvals, eigvecs = sp.linalg.eig(A)

    def solution(t: np.ndarray) -> np.ndarray:
        consts = sp.linalg.solve(eigvecs, y_0)
        l = np.diag(eigvals)
        times = np.stack((t, t))
        exps = np.exp(l @ times)
        consts = np.diag(consts)
        cords = consts @ exps
        return eigvecs @ cords

    return solution


def lorenz_attractor(sigma: np.float64, r: np.float64, b: np.float64):
    def f(t: np.float64, x: np.ndarray):
        if x.shape[0] != 3:
            raise ValueError(f'y should have 3 dimensions, having shape {x.shape}')

        res = np.zeros_like(x)
        res[0] = sigma * (x[1] - x[0])
        res[1] = x[0] * (r - x[2]) - x[1]
        res[2] = x[0] * x[1] - b * x[2]
        return res

    def jac(t: np.float64, x: np.ndarray):
        if x.shape[0] != 3:
            raise ValueError(f'y should have 3 dimensions, having shape {x.shape}')
        return np.array([[-sigma, sigma, 0.],
                         [r - x[2], -1., -x[0]],
                         [x[1], x[0], -b]])

    return f, jac


def rossler_attractor(mu: np.float64):
    def f(t, x):
        if x.shape[0] != 3:
            raise ValueError(f'y should have 3 dimensions, having shape {x.shape}')
        res = np.zeros_like(x)
        res[0] = -x[1] - x[2]
        res[1] = x[0] + x[1]/5.
        res[2] = 1/5. + x[2]*(x[0] - mu)
        return res

    def jac(t, x):
        return np.array([[0.,    -1.,    -1.],
                        [1.,    .2,     0.],
                        [x[2],  0.,     x[0] - mu]])

    return f, jac


def rikitake_attractor(mu: np.float64, gamma_1: np.float64, gamma_2: np.float64):
    def f(t, x):
        if x.shape[0] != 4:
            raise ValueError(f'y should have 3 dimensions, having shape {x.shape}')
        return np.array([   -mu*x[0] + x[1]*x[2],
                            -mu*x[1] + x[0]*x[3],
                            1. - x[0]*x[1] - gamma_1*x[2],
                            1. - x[0]*x[1] - gamma_2*x[3]])

    def jac(t, x):
        return np.array([[-mu,  x[2],   x[1],   0.],
                         [x[3], -mu,    0.,     x[0]],
                         [-x[1], -x[0], -gamma_1,   0.],
                         [-x[1], -x[0], 0.,   -gamma_2]])

    return f, jac

