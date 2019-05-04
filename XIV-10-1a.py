import numpy as np
import matplotlib.pyplot as plt

from methods.convection import LaxSchemePeriodic

T = 9.
L = 10.
a = 1.
c = 1.
n_steps = 100
max_splits = 9
u_0 = lambda x: np.sin(2*np.pi*x/L)

steps = [n_steps*2**k for k in range(max_splits)]
courants = [1.0, 0.6, 0.3]


fig, ax = plt.subplots(1, 1)

for c in courants:
    solvers = [LaxSchemePeriodic(a, u_0, c, s, right_bound=L) for s in steps]
    for s in solvers:
        s.integrate(T)
    samples = [s.get_sample(10) for s in solvers]
    errors = [np.linalg.norm(s[1] - u_0(s[0] - T), ord=1) for s in samples]
    x = np.log10(np.array([s.h for s in solvers]))
    y = np.log10(np.array(errors))

    p = np.polyfit(x[y > -8], y[y > -8], deg=1)
#    p = np.polyfit(x, y, deg=1)
    line = np.poly1d(p)

    points = ax.plot(x, y, label=f'$\sigma = {c:.1f}$', linestyle='None', marker='s')
    ax.plot(x, line(x), linestyle='--', color=points[0].get_color(),
            label=f'$\\frac{{ d(\\log_{{10}}||u_h - U||) }}{{ d(\\log_{{10}} h ) }} \\approx {p[0]:.2f}$')

ax.grid()
ax.legend()
ax.set_title('Convergence rate for Lax scheme with periodic border conditions')
ax.set_xlabel('$\\log_{{10}} h$')
ax.set_ylabel('$\\log_{{10}}||u_h - U||$')
plt.show()

