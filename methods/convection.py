import numpy as np

class ConvectionProblem:
    """"
        Class for solving 1-D convection problem
            u_t + a*u_x = 0         a < 0
            u(0, x) = u_0(x)
            u(t, L) = phi(t)
        with 3-rd order scheme (thus requires 3 time derivatives of phi)
        Args:
            a <
    """

    def __init__(self, a: np.float64, u_0, phi, phi_t, phi_tt, phi_ttt, c: np.float64,
                 n_steps: np.float64, left_bound : np.float64=0.0, right_bound : np.float64 = 1.0):
        if a >= 0.0:
            raise ValueError(f'a should be negative, you have {a} >= 0')

        self.a = a
        self.domain = np.linspace(left_bound, right_bound, n_steps + 1, endpoint=True)
        self.h = self.domain[1] - self.domain[0]
        self.c = c                                          # Courant parameter
        self.tau = np.fabs(self.h*self.c/self.a)
        self.T = 0.0
        self.solution = u_0(self.domain)

        self.u_0 = u_0
        self.phi = phi
        self.phi_t = phi_t
        self.phi_tt = phi_tt
        self.phi_ttt = phi_ttt

    def update_courant(self, c: np.float64):
        """"Sets tau in such way that Courant parameter is equal to the given (number of steps does not change)"""
        self.c = c                                          # Courant parameter
        self.tau = np.fabs(self.h*self.c/self.a)

    def reset(self):
        self.T = 0.0
        self.solution = self.u_0(self.domain)


    def step(self, tau=None):
        old_tau = self.tau
        if tau is not None:
            self.tau = tau

        self.T += self.tau
        ts = lambda kh: self.phi(self.T) + (-self.phi_t(self.T)/self.a)*kh \
                        + (self.phi_tt(self.T)/self.a**2)*kh**2/2.0 + (-self.phi_ttt(self.T)/self.a**3)*kh**3/6.0
        u = self.solution[:-3]
        u1 = self.solution[1:-2]
        u2 = self.solution[2:-1]
        u3 = self.solution[3:]

        tau = self.tau
        h = self.h

        c = self.c

        s1 = (2*u3 - 9*u2 + 18*u1 - 11*u)*c/6
        s2 = (-u3 + 4*u2 - 5*u1 + 2*u)/2*c**2
        s3 = (u3 - 3*u2 + 3*u1 - u)*c**3/6

        self.solution[:-3] += s1 + s2 + s3
        self.solution[-3:] = ts(np.array([-2.0*h, -h, 0.0]))

        if tau is not None:
            self.tau = old_tau

    def get_sample(self, n_samples:int):
        step = (self.domain.shape[0] - 1)//n_samples
        return np.stack((self.domain[::step], self.solution[::step]))
