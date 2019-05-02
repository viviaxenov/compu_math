import numpy as np
import scipy as sp
import scipy.integrate

from methods.ode_integration import *
from methods.helpers import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# -----------------------------------------------------------
# Lorentz attractor                                         |
# -----------------------------------------------------------

T = 20.0
n_steps = T/1e-2

sigma = 10.
r = 28.
bs = [8./3, 10., 20.]

xs = [1.0, 2.0, 3.0, 5.0, 10.0]

for b in bs:
    f, jac = lorenz_attractor(sigma, r, b)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for ind, x_0 in enumerate(xs):
        r_0 = np.array([x_0, 1.0, 1.0])
        solver = RK45Solver(f, r_0, 0., T, n_steps=n_steps)
        solver.solve()

        ax.plot(solver.x[0], solver.x[1], solver.x[2], color=f'C{ind}', label=f'$x_0 = {x_0:.1f}$', linewidth=.7)

    ax.set_title(f'Lorentz attractor, $b = {b:.2f}$')
    ax.legend()
    fig.tight_layout()
    plt.show()

# -------------------------------------------------------------------------------------------
# Rossler's atrractor
# -------------------------------------------------------------------------------------------

T = 30.
mus = np.linspace(0.01, 10., 5, endpoint=True)
y_0 = [1.0, 2.0, 3.0]
starts = [np.array([0., x, y]) for x in y_0 for y in y_0]
for mu in mus:
    f, jac = rossler_attractor(mu)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for ind, r_0 in enumerate(starts):
        solver = RK45Solver(f, r_0, 0., T, n_steps=n_steps)
        solver.solve()

        ax.plot(solver.x[0], solver.x[1], solver.x[2], color=f'C{ind}',
                label=f'$y_0 = {r_0[1]:.1f}, z_0 = {r_0[2]:.1f}$', linewidth=.7)

    fig.suptitle(f'Rössler attractor, $\\mu = {mu:.2f}$')
    ax.legend()
    fig.tight_layout()
    plt.show()



# -------------------------------------------------------------------------------------------
# Rikitake atrractor
# -------------------------------------------------------------------------------------------

T = 20.
mus = [0.2, 1., 2.]
gammas = [0.002, 0.003, 0.004]
gamma_2 = 0.002
starts = [np.array([0., x, y, 1.0]) for x in y_0 for y in y_0]
for mu in mus:
    for gamma_1 in gammas:
        f, jac = rikitake_attractor(mu, gamma_1, gamma_2)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for ind, r_0 in enumerate(starts):
            solver = RK45Solver(f, r_0, 0., T, n_steps=n_steps)
            solver.solve()

            ax.plot(solver.x[0], solver.x[1], solver.x[2], color=f'C{ind}',
                    label=f'$y_0 = {r_0[1]:.1f}, z_0 = {r_0[2]:.1f}$', linewidth=.7)

        fig.suptitle(f'Rikitake attractor (projected on $xyz$), $\\mu = {mu:.2f},\\ \\gamma_1 = {gamma_1:.2f}$')
        ax.legend()
        fig.tight_layout()
        plt.show()

