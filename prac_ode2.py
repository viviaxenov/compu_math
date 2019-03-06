import numpy as np
import scipy as sp
import scipy.linalg

import matplotlib.pyplot as plt

import methods.rosenbroke_method as rsb

A = np.array([[-101, 250], [40, -101]])             # matrix
b = np.array([-5, 2])*1e9                           # initial condition
u = rsb.get_analytical(A, b)

T = 0.1
n_samples = 11
init_iter = 1024                                      # starting number of iterations
n_splits = 8

samples = np.linspace(0, T, n_samples, endpoint=True)
analytical = np.real(u(samples))
with np.printoptions(precision=2, linewidth=1000):
    print("Analytical solution is:")
    print(analytical)

numeric = b
x = b
tau = (samples[1] - samples[0])/init_iter
n_iter = init_iter


tp = np.dtype([('tau', np.float64), ('dx', np.float64), ('dy', np.float64), ('ie', np.float64)])
fin_err = np.array([], dtype=tp)

for k in range(n_splits):
    numeric = b
    x = b
    n_iter = init_iter*2**k
    tau = (samples[1] - samples[0])/n_iter
    for i in range(n_samples - 1):
        x = rsb.linear_odeint(A, x, tau, n_iter)
        numeric = np.vstack((numeric, x))
    numeric = numeric.T

    ie = sp.linalg.norm(numeric - analytical, ord=2)

    fe = np.fabs(numeric[:, -1] - analytical[:, -1])

    fin_err = np.insert(fin_err, len(fin_err), (tau, fe[0], fe[1], ie))
    with np.printoptions(precision=2, linewidth=1000):
        print(f"\nh = {tau} ({n_iter} iterations)\nNumeric solution is:")
        print(numeric)
        print("Error is:")
        print(numeric - analytical)

x = np.log(fin_err['tau'])
y1 = np.log(fin_err['dx'])
y2 = np.log(fin_err['dy'])

fit = 5
p1 = np.polyfit(x[:fit], y1[:fit], deg=1)
p2 = np.polyfit(x[:fit], y2[:fit], deg=1)

line1 = np.poly1d(p1)
line2 = np.poly1d(p2)
print(p1[0], p2[0])

fig, axs = plt.subplots(1, 3)

ax = axs[0]
ax.plot(fin_err['tau'], fin_err['dx'], 'bs', label='$|y_{1}^{num} - y_1^{a}|$')
ax.plot(fin_err['tau'], fin_err['dy'], 'r^', label='$|y_{2}^{num} - y_2^{a}|$')
ax.set_xlabel('$h$')
ax.grid(True)
ax.legend(loc='upper left')

ax = axs[1]
ax.plot(x, y1, 'bs', label='$\\log{|y_{1}^{num} - y_1^{a}|}$')
ax.plot(x, y2, 'r^', label='$\\log{|y_{2}^{num} - y_2^{a}|}$')
ax.plot(x[:fit], line1(x[:fit]), 'b-')
ax.plot(x[:fit], line2(x[:fit]), 'r-')
ax.set_xlabel('$\\log h$')
ax.grid(True)
ax.legend(loc='upper left')

ax = axs[2]
x = np.log(fin_err['tau'])
y = np.log(fin_err['ie'])

p = np.polyfit(x[:-3], y[:-3], deg=1)
line = np.poly1d(p)
print(p[0])
ax.plot(x, y, 'go')
ax.plot(x, line(x), 'g--')
ax.grid(True)

plt.tight_layout()
plt.show()
