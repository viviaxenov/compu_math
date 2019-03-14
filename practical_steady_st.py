import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt

import methods.steady_state as ss

# solving model problem


k = ss.get_array_const(1.25)
q = ss.get_array_const(2.5)
f = ss.get_array_const(np.cos(0.5))

l_cond = (k(0.0), -100.0, .0)
r_cond = (-k(1.0), .0, .0)

#l_cond = (0.0, 1.0, 0.0)
#r_cond = (0.0, 1.0, 1.0)

analytical = ss.get_analytical(l_cond, r_cond, 1.25, 2.5, np.cos(0.5))

model_problem = ss.SteadyStateProblem(l_cond, r_cond, k, q, f, n_steps=40000)
model_problem.solve()

fig, ax = plt.subplots(1, 1)
ax.plot(model_problem.domain, model_problem.solution, 'b-', label='Numeric')
ax.plot(model_problem.domain, analytical(model_problem.domain), 'r--', label='Analytical')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(x)$')

plt.grid(True)
plt.show()

s = model_problem.get_sample(10)
err = (s[1] - analytical(s[0]))/s[1]

with np.printoptions(precision=5, linewidth=1000):
    print('-------------------------- Model problem ------------------------------------------------------------------\n')
    print('Domain sample:')
    print(s[0])
    print('Analytical solution:')
    print(analytical(s[0]))
    print('Numeric solution:')
    print(s[1])
    print('Relative difference:')
    print(err)
    print(f'Error norm = {np.linalg.norm(err, ord=1):.1e}')
    print('-----------------------------------------------------------------------------------------------------------\n\n')

# Solving problem with varying coeffs

k = lambda x: x**2 + 1
q = lambda x: x + 2
f = np.cos

l_cond = (k(0.0), -100.0, .0)
r_cond = (-k(1.0), .0, .0)

max_prec = 7

ref_problem = ss.SteadyStateProblem(l_cond, r_cond, k, q, f, n_steps=1000000)
ref_problem.solve()
ref = ref_problem.get_sample(10)[1]

with np.printoptions(precision=5, linewidth=1000):
    print('-------------------------- Varying coefs problem ----------------------------------------------------------\n')
    print('Domain sample:')
    print(ref_problem.get_sample(10)[0])
    print('Numeric solution:')
    print(reft)

errors = []
steps = []

for n_steps in [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000]:
    pr = ss.SteadyStateProblem(l_cond, r_cond, k, q, f, n_steps=n_steps)
    pr.solve()
    s = pr.get_sample(10)[1]
    err = np.linalg.norm(ref - s, ord=1)
    step = 1./n_steps

    errors += [err]
    steps += [step]

errors = np.array(errors)
steps = np.array(steps)

p = np.polyfit(np.log(steps)[:-1], np.log(errors)[:-1], deg=1)
line = np.poly1d(p)

fig, ax = plt.subplots(1, 1)

ax.plot(np.log(steps), np.log(errors), 'bs')
ax.plot(np.log(steps), line(np.log(steps)), 'b--', label=f'$\\frac{{  d(\\log||x - x_r||) }}{{ d(\\log{{h}}) }} \\approx {p[0]:.1f}$')
ax.set_xlabel(r'$\log{h}$')
ax.set_xlabel(r'$\log{||u - u_r||_1}$')
ax.legend()
ax.grid()
plt.show()