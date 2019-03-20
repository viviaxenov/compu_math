import numpy as np

import methods.shuttle as s


class PeriodicProblem:
    def __init__(self, P, f, n_steps : int, left_bound : np.float64=0.0, right_bound : np.float64 = 1.0):
        self.P = P
        self.f = f
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.domain = np.linspace(self.left_bound, self.right_bound, n_steps)

        self.n_steps = n_steps
        self.h = self.domain[1] - self.domain[0]
        self.solution = []

    def solve(self):
        dom = self.domain[:-1]
        f = self.f(dom)
        m = -self.P(dom)-2.0/self.h**2
        u = np.full_like(dom[:-1], 1.0/self.h**2)
        l = np.full_like(dom[:-1], 1.0/self.h**2)

        bl = 1.0/self.h**2
        tr = 1.0/self.h**2

        self.solution = np.zeros_like(self.domain)
        self.solution[:-1] = s.solve_cyclic(u, m, l, f, bl=bl, ur=tr)
        self.solution[-1] = self.solution[0]


