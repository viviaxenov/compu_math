import numpy as np
import  scipy as sp
import scipy.linalg
import numpy.linalg

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

#    alpha = 1.077
#    beta = - 0.372          # controlling params that provide 3-rd order accuracy
#    gamma = -0.577

    gamma = -1/np.sqrt(3)
    alpha = 1/2 - gamma
    beta = 1/6 - alpha*(alpha + gamma)

    x = x_0

    for i in range(n_iter):
        b = np.dot(A, (x + gamma*tau*np.dot(A, x)))
        X = np.eye(*A.shape) - alpha*tau*A - beta*tau**2*np.dot(A, A)

        grad = np.linalg.solve(X, b)

        x = x + tau*grad
    return x


def get_analytical(A : np.ndarray, y_0 : np.ndarray):
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

    def solution(t : np.ndarray) -> np.ndarray :
        consts = sp.linalg.solve(eigvecs, y_0)
        l = np.diag(eigvals)
        times = np.stack((t,t))
        exps = np.exp(l@times)
        consts = np.diag(consts)
        cords = consts@exps
        return eigvecs@cords
    return solution

