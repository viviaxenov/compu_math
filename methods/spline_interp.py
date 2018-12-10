import numpy as np

import methods.shuttle as sh


class CubicSpline:
    """
    Implements cubic spline interpolation
    """

    def __init__(self, t_0: np.array, f_0: np.array):
        """
        Creates cubic spline S  interpolating given function with condition d^2S/dt^2(t_0) = d^2S/dt^2(t_N) = 0
        Args:
            t - interpolation points
            f - values of function in points t
        """
        # sorting input points
        perm = np.argsort(t_0)
        t = np.copy(t_0[perm])
        f = np.copy(f_0[perm])

        h = t[1:] - t[:-1]

        a = f[:-1]

        l = h[1:]
        u = l
        m = 2 * (h[1:] + h[:-1])
        F = 3 * ((f[2:] - f[1:-1]) / h[1:] - (f[1:-1] - f[:-2]) / h[:-1])
        k = F.shape[0]
        A = np.zeros((k,k))
        for i in range(k):
            A[i, i] = m[i]
        for i in range(1, k):
            A[i, i - 1] = l[i]
            A[i - 1, i] = u[i]
#        c = sh.solve(u, m, l, F)
        c = np.linalg.solve(A,F)
        c = np.concatenate(([0], c, [0]))
        b = (f[1:] - f[:-1])/h - h*(c[1:] + 2*c[:-1])/3
        d = (c[1:] - c[:-1])/(3*h)
        print(c)
        c = c[:-1]

        self.t = t
        self.coefs = np.column_stack((a,b,c,d))

    def __call__(self, t: np.float64):
        """
        Calculates spline value at point t in [t_0; t_N]
        Args:
            t - point to interpolate
        Returns:
            f - spline value
        Raises:
            ValueError - if t is out of [t_0; t_N]
        """
        if t < self.t[0] or t > self.t[-1]:
            raise (ValueError('Point is out of bounds [t_0; t_N]'))
        # find the segment t belongs to
        i = np.searchsorted(self.t, t, side='left') - 1
        if i == -1:
            return self.coefs[0, 0]                                 # problem with left boundary
        z = t - self.t[i]
        pows = np.array([1, z, z**2, z**3])
        return self.coefs[i]@pows
