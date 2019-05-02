import numpy as np
import scipy as sp
import scipy.integrate

from methods.ode_integration import *
from methods.helpers import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --------------------------------------------------------------------------------------------
# Convergence rates
# --------------------------------------------------------------------------------------------

sigma = 10.
r = 28.
bs = [8./3, 10., 20.]

mu = 5.

attr_funcs = [lorenz_attractor]#, rossler_attractor, rikitake_attractor]
attr_args = [(sigma, r, bs[0]), [mu], (0.2, 0.003, 0.002)]
starts = [np.array([2., 1., 1.]), np.array([0.0, 1.0, 1.0]), np.array([0.] + [1.]*3)]
names = ['Lorentz', 'Rössler', 'Rikitake']
max_half = 11
T = 1.
fit_slice = [-1, -2, -3]


for ind, attractor in enumerate(attr_funcs):
    f, jac = attractor(*(attr_args[ind]))

    ref_solverRK = RK45Solver(f, starts[ind], 0., T, 10*2**max_half)
    ref_solverIE = ImplicitEulerSolver(f, jac, starts[ind], 0., T, 10 *2**max_half)
    ref_solverIE.solve()
    ref_solverRK.solve()
    ref_IE = ref_solverIE.get_sample(10)[1:]
    ref_RK = ref_solverRK.get_sample(10)[1:]

    steps = [10*2**k for k in range(1, max_half)]
    errIE = []
    errRK = []
    for n_steps in steps:
        solverRK = RK45Solver(f, starts[ind], 0., T, n_steps)
        solverIE = ImplicitEulerSolver(f, jac, starts[ind], 0., T, n_steps)
        solverRK.solve()
        solverIE.solve()
        nsl_IE = solverIE.get_sample(10)[1:]
        nsl_RK = solverRK.get_sample(10)[1:]
        errIE += [np.linalg.norm((nsl_IE - ref_IE).flatten(), ord=np.inf)]
        errRK += [np.linalg.norm((nsl_RK - ref_RK).flatten(), ord=np.inf)]

    x = np.log10(T)-np.log10(np.array(steps))
    y_IE = np.log10(np.array(errIE))
    y_RK = np.log10(np.array(errRK))
    pIE = np.polyfit(x, y_IE, deg=1)
    pRK = np.polyfit(x[:fit_slice[ind]], y_RK[:fit_slice[ind]], deg=1)
    lineIE = np.poly1d(pIE)
    lineRK = np.poly1d(pRK)

    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y_IE, 'bs', label='Implicit Euler')
    ax.plot(x, lineIE(x), 'b--', label=f'$\\frac{{ d(\\log||u - u_r||) }}{{ d(\\log h) }} = {pIE[0]:.2f}$')
    ax.plot(x, y_RK, 'rs', label='\"Classic\" Runge-Kutta')
    ax.plot(x, lineRK(x), 'r--', label=f'$\\frac{{ d(\\log||u - u_r||) }}{{ d(\\log h) }} = {pRK[0]:.2f}$')
    ax.set_xlabel('$\\log_{{10}}{h}$')
    ax.set_ylabel('$\\log_{{10}}{||u_h - u_{ref}||_{\\infty}}$')
    ax.legend()
    ax.grid()
    fig.suptitle(f'Convergence rate for {names[ind]} attractor')
    save_path = f'img/{names[ind]}.png'
    fig.savefig(save_path)
