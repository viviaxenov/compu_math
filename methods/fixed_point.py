import numpy as np
import scipy as sp
import scipy.linalg
import scipy.optimize as spopt

from abc import ABCMeta, abstractmethod

class FixedPointSolver:
    __metaclass__ = ABCMeta
    @abstractmethod
    def iteration(self, *args, **kwargs):
        """Iteration, depends on method. Must calculate new point and update residual"""
    def solve(self, eps:np.float64, max_iter: int = 10000):
        iter = 0
        while(self.residual >= eps):
            self.iteration()
            iter += 1
            if(iter >= max_iter):
                raise ValueError(f"Number of iterations exceeded maximal number of iterations {max_iter:d}")

        return iter


class SimpleIterationSolver(FixedPointSolver):
    def __init__(self, f, x_0: np.ndarray, norm=2):
        """
            Solves fixed-point problem f(x) = x with simple iteration method
            Args:
                f - function, R^n -> R^n
                x_0 - initial guess
                norm - the norm to estimate residual. Same rules as for np.linalg.norm (optional,
                        default --- 2-norm (euclidean))
        """
        self.f = f
        self.val = x_0
        self.x = x_0
        self.norm = norm
        self.residual = np.inf              # difference btw x and f(x)

    def iteration(self):
        self.x = self.val
        self.val = self.f(self.x)
        self.residual = np.linalg.norm(self.x - self.val, ord=self.norm)


class NewtonSolver(FixedPointSolver):
    def __init__(self, f, jac, x_0: np.ndarray, norm=2):
        """
            Solves fixed-point problem x = f(x) with Newton's method
            Args:
                f - function
                jac - jacobian of f
                x_0 - initial guess
        """
        self.f = f
        self.jac = jac
        self.norm = norm
        self.x = x_0
        self.val = self.f(x_0)
        self.residual = np.inf              # difference btw x and f(x)

    def iteration(self, *args, **kwargs):
        J = np.identity(self.x.shape[0]) - self.jac(self.x)
        f = self.x - self.val
        d = sp.linalg.solve(J, f)
        self.x -= d
        self.val = self.f(self.x)
        self.residual = np.linalg.norm(self.x - self.val, ord=self.norm)



class AndersonSolver(FixedPointSolver):
    """"
        Solves fixed-point problem x = g(x) with Anderson's accelerated method
        Args:
            :arg g - function
            :arg x_0 - initial guess
            :arg m - number of points which give secant information
    """
    def __init__(self, g, x_0: np.ndarray, m: int = 2):
        self.function = g
        self.m = m
        # number of iterations
        self.k = 0

        # rows of the following matrices are points, respective function values and residuals
        self.x = np.array([x_0])
        self.g = np.array([self.function(x_0)])
        self.f = self.g - self.x

        # first step is done via Picard method (simple iteration)
        self.x = np.vstack((self.x, self.g[0]))
        self.g = np.vstack((self.g, self.function(self.x[1])))
        self.f = self.g - self.x

        # composing matrices with variations of f and x
        self.X = (self.x[1:] - self.x[:-1])
        self.F = (self.f[1:] - self.f[:-1])
        self.k += 1

    def iteration(self, *args, **kwargs):
        # TODO: do this optimization problem with QR decomposition
        # transposition because in the original paper the COLUMNS of those matrices are x and f respectively
        lstq_solution = spopt.lsq_linear(self.F.T, self.f[-1])
        if not lstq_solution.success:
            raise RuntimeError("Least square problem did not converge. Reason: " + lstq_solution.message)

        gamma = lstq_solution.x
        # calculating new point
        # transposition because in the original paper the COLUMNS of those matrices are x and f respectively
        x_new = self.g[-1] - (self.X.T + self.F.T)@gamma
        g_new = self.function(x_new)
        f_new = g_new - x_new
        # updating matrices
        # truncating if necessary
        self.x = np.vstack((self.x[-(self.m - 1)], x_new))
        self.g = np.vstack((self.g[-(self.m - 1)], g_new))
        self.f = np.vstack((self.f[-(self.m - 1)], f_new))

        self.X = np.vstack((self.X[-(self.m - 1)], self.x[-1] - self.x[-2]))
        self.F = np.vstack((self.F[-(self.m - 1)], self.f[-1] - self.f[-2]))



