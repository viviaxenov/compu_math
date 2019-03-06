import numpy as np
from methods.helpers import *


def solve(f, x_0 : np.float64, y_0 : np.float64, h : np.float64, 
        x_max : np.float64) -> (np.float64, np.float64):
    """
    Solves a first-order ODE dy/dx = f(x, y) with implicit Euler method
    Args:
        f(float, float) -> float - function in right part
        x_0, y_0 - initial conditions
        h - grid step
        x_max - right boundary of the domain of solution
    Returns:
        (x, y) - tuple of numpy.array - regular grid with step h for x
        and values of solution y in those points
    """
    x = np.arange(start=x_0, stop=x_max, step=h)
    y = np.zeros_like(x)

    y[0] = y_0
    for i in range(1, x.shape[0]):
        f_prev = f(x[i - 1], y[i - 1])
        y_pred = y[i - 1] + h*f_prev
        y[i] = y[i - 1] + 0.5*h*(f_prev + f(x[i], y_pred))
    return (x, y)


def solve_lin_syst(u_0 : np.ndarray, A : np.ndarray, h : np.float64, n_steps : int, explicit=False) -> np.ndarray:
    verify_args(A, u_0)
    u = u_0
    if explicit:
        R = np.eye(*A.shape) + h*A
    else:
        R = np.linalg.inv(np.eye(*A.shape) - h*A)

    for _ in range(n_steps):
        u = R@u
    return u
