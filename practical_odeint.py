import numpy as np
import scipy as sp
import scipy.linalg

import matplotlib.pyplot as plt

import methods.rosenbroke_method as rsb


def get_analytical(A : np.ndarray, y_0 : np.ndarray):
    """"
    Gets analytical solution for ODE system dy/dt = Ay, y(0) = y_0, where A=const
    Args:
        A - right part matrix
        y_0 - initial condition
    Returns:
        u - function  : y(t)
    Warning: works only if A has basis of real eigenvalues. Other cases weren't considered
    """
    eigvals, eigvecs = sp.linalg.eig(A)

    def solution(t : np.ndarray) -> np.ndarray :
        consts = sp.linalg.solve(eigvecs, y_0)
        l = np.diag(eigvals)
        times = np.stack((t,t))
        exps = np.exp(l@times)
        consts = np.diag(consts)
        cords = consts@exps
        return eigvecs@cords
    return solution


def stability_function(z : np.ndarray):
    return (1 - 0.0772*z - 0.205*z**2)/(1 - 1.0772*z + 0.372*z**2)


# xs = np.linspace(-100, 10, 100, endpoint=True)
# fig, ax = plt.subplots(1,1)
# ax.plot(xs, stability_function(xs), 'r-')
# ax.axhline(1)
# ax.axhline(-1)
# ax.set_xlabel('$z$')
# ax.set_ylabel('$R(z)$ - stability function')
# ax.grid(True)
# plt.show()

A = np.array([[-101, 250], [40, -101]])
eigvals, eigvecs = sp.linalg.eig(A)
e = np.abs(eigvals)
L = e.max()/e.min()
print(f'Eigenvalues are {eigvals[0]:.1f} and {eigvals[1]:.1f}')
print(f'|\\lambda_max/\lambda_min| = {L:.1f} >> 1 => system is stiff')

b = np.array([-5e9, 2e9])
u = get_analytical(A, b)

T = 0.1

n_iter = 1024
n_samples = 11
sample_points = np.linspace(0, T, n_samples, endpoint=True)
samples = np.array([b])
tau = (sample_points[1] - sample_points[0])/n_iter
x = b

for i in range(n_samples - 1):
    x = rsb.linear_odeint(A, x, tau, n_iter)
    samples = np.vstack((samples, x))
samples = samples.T

analytical_solution = np.real(u(sample_points))
delta = np.linalg.norm((analytical_solution - samples)/analytical_solution, ord=1)

error = np.abs(samples - analytical_solution)

print(samples[0], '\n', analytical_solution[0], '\n', error[0])
print(samples[1], '\n', analytical_solution[1], '\n', error[1])

print('To solve the system numerically, we use Rosenbroke method. It is A-stable and has 3-rd order of accuracy.')
print(f'With given step tau = {tau:.2e} the stability function for both eigenvalues is {stability_function(tau*eigvals)}')
print(f'We search for solution in {n_samples:d} sample points and compare the values with analytical solution')
print(f'On each section between samples we make {n_iter:d} iterations of the method with time step {tau:.1e}')
print(f'Relative difference (c_1 norm) is {delta:.1e}')


t = np.linspace(0, T, 10*n_samples, endpoint=True)
analytical_solution = u(t)

# Phase trajectories

fig, ax = plt.subplots(1, 1)

ax.plot(samples[0], samples[1], 'bs', label='Numeric')
ax.plot(analytical_solution[0], analytical_solution[1], 'r-', label='Analytical')
ax.set_xlabel('$y_1(t)$')
ax.set_ylabel('$y_2(t)$')
ax.legend()
ax.grid(True)
plt.show()

fig, axs = plt.subplots(1, 2, sharex=True)

ax = axs[0]
ax.plot(sample_points, np.real(u(sample_points)[0]), 'r-', label='Analytical')
ax.plot(sample_points, samples[0], 'bs', label='Numerical')
ax.set_xlabel('$t$')
ax.set_ylabel('$y_1(t)$')
ax.legend()
ax.grid(True)

ax = axs[1]
ax.plot(sample_points, np.real(u(sample_points)[1]), 'r-', label='Analytical')
ax.plot(sample_points, samples[1], 'bs', label='Numerical')
ax.set_xlabel('$t$')
ax.set_ylabel('$y_2(t)$')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()

# Actual convergence rate

times = []
diffs = []
analytical_solution = np.real(u(sample_points))

fin_diff = []

for n_iter in [1024*2**k for k in range(5)]:
    tau = (sample_points[1] - sample_points[0])/n_iter
    samples = np.array([b])
    x = b
    for i in range(n_samples - 1):
        x = rsb.linear_odeint(A, x, tau, n_iter)
        samples = np.vstack((samples, x))
    samples = samples.T

    delta = np.linalg.norm((analytical_solution - samples)/analytical_solution, ord=1)
    times.append(tau)
    diffs.append(delta)

    fin_diff += [np.abs(samples[:, -1] - analytical_solution[:, -1])]

times = np.array(times)
diffs = np.array(diffs)

fin_diff = np.array(fin_diff)
splits = range(5)


# x = np.log(times)
# y = np.log(diffs)
print('\n\nError in final point T=0.1 is:')
print(fin_diff)
# x = splits
# y = np.log2(fin_diff)
#
# p = np.polyfit(x, y, deg=1)
# line = np.poly1d(p)
#
# fig, ax = plt.subplots(1, 1)
# ax.plot(x, y, 'bs', label='Actual')
# ax.plot(x, line(x), 'r-', label='Fit')
#
# ax.set_xlabel(r'$\log{\tau}$')
# ax.set_ylabel(r'$\log{||y - y_{analyt.}||}$')
# ax.text(x[3] - 1.5, y[3], f'$\\frac{{dx}}{{dy}} \\approx {p[0] : .2f}$')
# ax.grid(True)
# ax.legend()
# plt.show()
#
# print('We solve the equation using smaller timesteps and plot the difference from analytical solution in log scale.')
# print(f'The slope (~{p[0] : .2f}) denotes actual convergence rate')
