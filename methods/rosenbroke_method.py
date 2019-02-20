import numpy as np
import  scipy as sp
import scipy.linalg

from methods.helpers import verify_args


def linear_odeint(A : np.ndarray, x_0 : np.ndarray, tau : np.float64, n_iter : int) -> np.ndarray:
    """
    Makes iterations with Rosenbroke's method (3-rd order)
    Args:
        A - right part matrix
        x_0 - initial conditions
        tau - timestep
        n_iter - number of iterations
    Returns:
        x - solution (x(tau*n_iter))
    Raises:
        ValueError - if shapes of A, x_0 don't correspond
    """

    verify_args(A, x_0)

    alpha = 1.077
    beta = - 0.372          # controlling params that provide 3-rd order accuracy
    gamma = -0.577

    x = x_0

    for i in range(n_iter):
        b = np.dot(A, (x + gamma*tau*np.dot(A, x)))
        X = np.eye(*A.shape) - alpha*tau*A - beta*tau**2*np.dot(A, A)

        grad = sp.linalg.solve(X, b)

        x = x + tau*grad
    return x

