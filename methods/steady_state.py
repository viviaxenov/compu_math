import numpy as np
import methods.shuttle as s

class SteadyStateProblem:
    """"
        Class for solving 1-D steady state problem for equation
            d/dx(k(x) du/dx) - q(x)u = f(x)
        Args:
            right_cond, left_cond - array [a, b, c] that boundary condition is
                a*du/dx + b*u = c
            k, q, f - equation coefs
            n_steps  - number of grid steps
    """
    def __init__(self, left_cond : tuple, right_cond : tuple, k, q, f, n_steps : int, left_bound : np.float64=0.0, right_bound : np.float64 = 1.0):
        self.left_cond = left_cond
        self.right_cond = right_cond
        self.left_bound = left_bound
        self.right_bound = right_bound

        self.n_steps = n_steps
        self.h = (right_bound - left_bound)/(float(n_steps))

        self.k = k
        self.q = q
        self.f = f
        self.domain = np.linspace(self.left_bound, self.right_bound, n_steps + 1, endpoint=True)
        self.solution = []

    def solve(self):
        """Solves steady-state problem"""

        # preparing matrices form TDM
        main = np.zeros(self.n_steps + 1)
        upper = np.zeros(self.n_steps)
        lower = np.zeros(self.n_steps)
        f = np.zeros(self.n_steps + 1)

        # filling conditions for inner part
        tmp = self.k(self.domain[:-1] + 0.5*self.h)
        main[1:-1] = -self.q(self.domain[1:-1]) - (tmp[:-1] + tmp[1:])/self.h**2
        upper[1:] = tmp[1:]/self.h**2
        lower[:-1] = tmp[:-1]/self.h**2
        f[1:-1] = self.f(self.domain[1:-1])

        # filling left boundary condition
        alpha, beta, gamma = self.left_cond
        main[0] = beta - alpha/self.h
        upper[0] = alpha/self.h
        f[0] = gamma

        # filling right boundary condition
        alpha, beta, gamma = self.right_cond
        main[-1] = beta + alpha/self.h
        lower[-1] = -alpha/self.h
        f[-1] = gamma

        self.solution = s.solve(upper, main, lower, f)



