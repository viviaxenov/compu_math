import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt

import methods.steady_state as ss

# solving model problem

x_0 = 1/np.sqrt(5)
q_1 = x_0
q_2 = x_0**3
# разрыв f!!!
f_1 = 1
f_2 = x_0**2 - 1
k = np.sin(x_0)**2 + 1
u_0 = 1.0
u_1 = 2.0

analytical = ss.get_analytical_with_break(u_0, u_1, x_0, q_1, q_2, k, f_1, f_2)

k = ss.get_array_const(k)
q = ss.get_step_function(q_1, q_2, x_0)
f = ss.get_step_function(f_1, f_2, x_0)

l_cond = (0.0, 1.0, u_0)
r_cond = (0.0, 1.0, u_1)

#l_cond = (0.0, 1.0, 0.0)
#r_cond = (0.0, 1.0, 1.0)


model_problem = ss.SteadyStateProblem(l_cond, r_cond, k, q, f, n_steps=1000)
model_problem.solve()

fig, ax = plt.subplots(1, 1)
ax.plot(model_problem.domain, model_problem.solution, 'b-', label='Numeric')
ax.plot(model_problem.domain, analytical(model_problem.domain), 'r--', label='Analytical')
ax.axvline(x_0)
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

k = lambda x: np.sin(x)**2 + 1


def f(x: np.ndarray):
    res = np.zeros_like(x)
    res[x < x_0] = 1.0
    res[x >= x_0] = x[x >= x_0]**2 - 1
    return res


def q(x: np.ndarray):
    res = np.zeros_like(x)
    res[x < x_0] = x[x < x_0]
    res[x >= x_0] = x[x >= x_0]**3
    return res


max_prec = 7

test_problem = ss.SteadyStateProblem(l_cond, r_cond, k, q, f, n_steps=1000)
test_problem.solve()
ref_problem = ss.SteadyStateProblem(l_cond, r_cond, k, q, f, n_steps=1000000)
ref_problem.solve()
ref = ref_problem.get_sample(10)[1]

with np.printoptions(precision=5, linewidth=1000):
    print('-------------------------- Varying coefs problem ----------------------------------------------------------\n')
    print('Domain sample:')
    print(ref_problem.get_sample(10)[0])
    print('Numeric solution:')
    print(test_problem.get_sample(10)[1])


errors = []
steps = []

for n_steps in [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000]:
    pr = ss.SteadyStateProblem(l_cond, r_cond, k, q, f, n_steps=n_steps)
    pr.solve()
    s = pr.get_sample(10)[1]
    err = np.linalg.norm(ref - s, ord=2)
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
ax.set_ylabel(r'$\log{||u - u_r||_1}$')
ax.legend()
ax.grid()
plt.show()
