import numpy as np


class PoissonProblem:
    def __init__(self, n_x: int, n_y: int, boundary, u_0, f):
        self.n_x = n_x
        self.n_y = n_y
        x = np.linspace(boundary[0], boundary[1], n_x + 1, endpoint=True)
        y = np.linspace(boundary[2], boundary[3], n_y + 1, endpoint=True)

        self.h_x = x[1] - x[0]
        self.h_y = y[1] - y[0]

        self.xx, self.yy = np.meshgrid(x, y)
        self.solution = u_0(self.xx, self.yy)

        self.solution[self.xx == boundary[0]] = 0
        self.solution[self.xx == boundary[1]] = 0
        self.solution[self.yy == boundary[2]] = 0
        self.solution[self.yy == boundary[3]] = 0
        self.f = f
        self.err = np.inf

    def residual(self, w_error: bool = False):
        s = self.solution
        res = ((s[0:-2, 1:-1] - 2.0*s[1:-1, 1:-1] + s[2:, 1:-1])/self.h_x**2
                                       + (s[1:-1, 0:-2] - 2.0*s[1:-1, 1:-1] + s[1:-1, 2:])/self.h_y**2
                                       - self.f(self.xx[1:-1, 1:-1], self.yy[1:-1, 1:-1]))
        if w_error:
            self.err = np.linalg.norm(res.flatten(), ord=np.inf)
        return res

    def step(self, tau: np.float64, w_error: bool = False):
        self.solution[1:-1, 1:-1] += tau*self.residual(w_error)

    def iterate_chebyshev(self, n_iter: int = None, eps: np.float64=1e-3):
        # operator's minimal and maximal eigvals
        l = 4*(np.sin(np.pi/2/self.n_x)**2/self.h_x**2 + np.sin(np.pi/2/self.n_y)**2/self.h_y**2)
        L = 4*(np.cos(np.pi/2/self.n_x)**2/self.h_x**2 + np.cos(np.pi/2/self.n_y)**2/self.h_y**2)
        mu = L/l

        if n_iter is None:
            n_iter = int(np.ceil(-np.sqrt(mu)/2*np.log(eps))) + 1

        for i in range(1, n_iter + 1):
            tau = 1./((L + l)/2 + (L - l)/2*np.cos(np.pi*(2*i - 1)/2/n_iter))
            self.step(tau, w_error=(i == n_iter))
        return n_iter

    def solve(self, n_iter: int, eps: np.float64):
        total_iter = 0
        while self.err > eps:
            total_iter += self.iterate_chebyshev(n_iter=n_iter)
            if total_iter >= 8000:
                raise(RuntimeError(f'Residual is {self.err:.2e} after {total_iter:d} steps, probably instability'))
        return total_iter

