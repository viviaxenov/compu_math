
import numpy as np
import matplotlib.pyplot as plt
from methods.convection import ConvectionProblem

phi = lambda x: np.cos(1 + 2.0*x)
phi_t = lambda x: -2.0*np.sin(1 + 2.0*x)
phi_tt = lambda x: -4.0*np.cos(1 + 2.0*x)
phi_ttt = lambda x: 8.0*np.sin(1 + 2.0*x)


def analytical(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    c = x + 2*t
    return np.cos(c)


n_steps = 100
period = 1.0
c = 1.2


solver = ConvectionProblem(-2.0, np.cos, phi, phi_t, phi_tt, phi_ttt, c, n_steps)

while solver.T < period:
    solver.step()

smp = solver.get_sample(10)
asl = analytical(smp[0], np.full_like(smp[0], solver.T))
err = smp[1] - asl

bound_analytical = analytical(solver.domain[-3:], np.full_like(solver.domain[-3:], solver.T))
bound_err = solver.solution[-3:] - bound_analytical

with np.printoptions(precision=6, linewidth=1000):
    print(f'Courant parameter = {solver.c}')
    print('+----------------------------------------------------------------------------------------------------------')
    print(f'Domain      : {smp[0]}')
    print(f'Analytical  : {asl}')
    print(f'Numeric     : {smp[1]}')
    print(f'Difference  : {err}')
    print()
    print(f'Maximal absolute difference: {np.linalg.norm(err, ord=np.inf):.5e}')
    print()
    print(f'Boundary    : {solver.domain[-3:]}')
    print(f'Analytical  : {bound_analytical}')
    print(f'Numeric     : {solver.solution[-3:]}')
    print(f'Difference  : {bound_err}')
    print('+----------------------------------------------------------------------------------------------------------')

exit()

plt.plot(solver.domain, solver.solution, 'r-')
plt.plot(solver.domain, analytical(solver.domain, np.full_like(solver.domain, solver.T)), 'b--')
plt.grid()
plt.show()



# stability area is something like [1; 2] (from experiment)

courants = np.linspace(0.1, 4.0, 20, endpoint=True)
stab_status = np.full_like(courants, 1.0)

solver = ConvectionProblem(-2.0, np.cos, phi, phi_t, phi_tt, phi_ttt, 1.0, n_steps)

# checking stability range
for ind, c in enumerate(courants):
    solver.reset()
    solver.update_courant(c)
    while solver.T < period:
        solver.step()
        if np.max(np.fabs(solver.solution) > 1.0):
            stab_status[ind] = -.0
            break

plt.plot(courants, stab_status, 'bs')
plt.grid()
plt.show()

# Convergence rate

courants = np.linspace(1.0, 2.0, 5, endpoint=True)
steps = np.array([50*2**k for k in range(7)])
n_samples = 50
errors = []

period = 1.0

for n_steps in steps:
    solver = ConvectionProblem(-2.0, np.cos, phi, phi_t, phi_tt, phi_ttt, 1.0, n_steps)
    err = []
    for c in courants:
        solver.reset()
        solver.update_courant(c)
        while solver.T < period:
            solver.step()
        smp = solver.get_sample(n_samples)
        asl = analytical(smp[0], np.full_like(smp[0], solver.T))
        err_norm = np.linalg.norm(asl - smp[1], ord=np.inf)
        err += [err_norm]
    errors += [err]

ys = np.log(np.array(errors).T)
x = -np.log(np.array(steps))

fig, ax = plt.subplots(1, 1)
fig.suptitle('Convergence rate')
ax.set_xlabel('$\log{h}$')
ax.set_ylabel('$\log{|| u - u_{analyt}||_{\infty}}$')
colors = ['red', 'blue', 'yellow', 'green']

for ind, y in enumerate(ys):
    p = np.polyfit(x[1:-1], y[1:-1], deg=1)
    line = np.poly1d(p)
    ax.plot(x, y, linestyle='None', marker='^', color=f'C{ind}', label=f'$c = {courants[ind]:.2f},\\ slope\\ {p[0]:.2e}$')
    ax.plot(x, line(x), linestyle='--', color=f'C{ind}')
ax.grid()
ax.legend()
plt.show()
exit()





