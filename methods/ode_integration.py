import numpy as np

import newton_method as newt

from abc import ABCMeta, abstractmethod


class ODEProblem:
    __metaclass__=ABCMeta
    def __init__(self, f, x_0: np.ndarray, t_0: np.float64, T: np.float64, n_steps: int):
        self.n_steps = n_steps
        self.f = f
        self.t = np.linspace(t_0, T, n_steps + 1, endpoint=True)
        self.x = np.zeros((x_0.shape[0], self.t.shape[0]))
        self.x[:, 0] = x_0

    @abstractmethod
    def step(self, i: int):
        """Integration step, depends on method"""

    def solve(self):
        for i in range(1, self.t.shape[0]):
            self.x[:, i] = self.step(i)

    def get_sample(self, n_samples: int):
        step = self.n_steps//n_samples
        return np.vstack((self.t[::step], self.x[:, ::step]))


class RK45Solver(ODEProblem):
    def step(self, i: int):
        x = self.x[:, i - 1]
        t = self.t[i - 1]
        h = self.t[i] - self.t[i - 1]
        k_1 = self.f(t,         x)
        k_2 = self.f(t + h/2.,  x + h*k_1/2.)
        k_3 = self.f(t + h/2.,  x + h*k_2/2.)
        k_4 = self.f(t + h,     x + h*k_3)

        return x + h*(k_1 + 2*k_2 + 2*k_3 + k_4)/6.

class ImplicitEulerSolver(ODEProblem):
    def __init__(self, f, jac, x_0: np.ndarray, t_0: np.float64, T: np.float64, n_steps: int):
        super().__init__(f, x_0, t_0, T, n_steps)
        self.jac = jac

    def step(self, i: int):
        x = self.x[:, i - 1]
        t = self.t[i - 1]
        h = self.t[i] - self.t[i - 1]
        def f(k: np.ndarray):
            return k - self.f(t + h, x + h*k)
        def jac(k: np.ndarray):
            return np.identity(k.shape[0]) - h*self.jac(t + h, x + h*k)

        k = newt.solve(self.f(t, x), f, jac, 1e-12)
        # TODO: estimate error in newton's method
        return x + h*k
