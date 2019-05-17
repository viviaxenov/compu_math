import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt

import methods.steady_state as ss

# solving model problem

x_0 = 1/np.sqrt(2)  # точка разрыва

k_1 = 1.
q_1 = 1.            # до разрыва
f_1 = np.exp(x_0)
k_2 = np.sin(x_0)
q_2 = 2.            # после
f_2 = f_1   # разрыва f нету
q = ss.get_step_function(q_1, q_2, x_0)
k = ss.get_step_function(k_1, k_2, x_0)
f = ss.get_array_const(np.exp(x_0))


u_0 = 1.0
u_1 = 0.0

analytical = ss.get_analytical_with_break(u_0, u_1, x_0, q_1, q_2, k_1, k_2, f_1, f_2)


l_cond = (0.0, 1.0, u_0)
r_cond = (0.0, 1.0, u_1)
# НА СДАЧЕ НУЖНО КРУТИТЬ ЭТО!
n_steps = 100   # число шагов, точек на 1 больше

model_problem = ss.SteadyStateProblem(l_cond, r_cond, k, q, f, n_steps=n_steps, breakpoint=x_0)
model_problem.solve()

# fig, ax = plt.subplots(1, 1)
# ax.plot(model_problem.domain, model_problem.solution, 'b-', label='Numeric')
# ax.plot(model_problem.domain, analytical(model_problem.domain), 'r--', label='Analytical')
# ax.axvline(x_0)
# ax.set_xlabel('$x$')
# ax.set_ylabel('$u(x)$')
#
# plt.grid(True)
# plt.show()

s = model_problem.get_sample(10)
asl = analytical(s[0])
err = (s[1][asl != 0] - asl[asl != 0])/asl[asl != 0]

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
    print(f'Error norm = {np.linalg.norm(err, ord=np.inf):.1e}')
    print('-----------------------------------------------------------------------------------------------------------\n\n')


# Checking convergence rate

# errors = []
# steps = []
#
# for n_steps in [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000]:
#     pr = ss.SteadyStateProblem(l_cond, r_cond, k, q, f, n_steps=n_steps, breakpoint=x_0)
#     pr.solve()
#     s = pr.get_sample(10)
#     rel_err = (s[1] - analytical(s[0]))
#     err = np.linalg.norm(rel_err, ord=np.inf)
#     step = 1./n_steps
#
#     errors += [err]
#     steps += [step]
#
# errors = np.array(errors)
# steps = np.array(steps)
#
# p = np.polyfit(np.log(steps)[:-1], np.log(errors)[:-1], deg=1)
# line = np.poly1d(p)
#
# fig, ax = plt.subplots(1, 1)
#
# fig.suptitle('Скорость сходимости для задачи с постоянными коэффициентами')
# ax.plot(np.log(steps), np.log(errors), 'bs')
# ax.plot(np.log(steps), line(np.log(steps)), 'b--', label=f'$\\frac{{  d(\\log||x - x_r||) }}{{ d(\\log{{h}}) }} \\approx {p[0]:.1f}$')
# ax.set_xlabel(r'$\log{h}$')
# ax.set_ylabel(r'$\log{||u - u_r||_{\infty}}$')
# ax.legend()
# ax.grid()
# plt.show()

# ----------------------------------------------------------------------------------------------------------------------
#           ЗАДАЧА С ПЕРЕМЕННЫМИ КОЭФФИЦИЕНТАМИ
# ----------------------------------------------------------------------------------------------------------------------


def k(x: np.ndarray):
    res = np.zeros_like(x)
    res[x < x_0] = 1.0
    res[x >= x_0] = np.exp(np.sin(x[x >= x_0]))
    return res
f = lambda x: np.exp(x)
q = ss.get_step_function(q_1, q_2, x_0)


max_prec = 7

test_problem = ss.SteadyStateProblem(l_cond, r_cond, k, q, f, n_steps=n_steps , breakpoint=x_0)
test_problem.solve()
# ref_problem = ss.SteadyStateProblem(l_cond, r_cond, k, q, f, n_steps=1000000, breakpoint=x_0)
# ref_problem.solve()
# ref = ref_problem.get_sample(10)[1]

with np.printoptions(precision=5, linewidth=1000):
    print('-------------------------- Varying coefs problem ----------------------------------------------------------\n')
    print('Domain sample:')
    print(test_problem.get_sample(10)[0])
    print('Numeric solution:')
    print(test_problem.get_sample(10)[1])


# errors = []
# steps = []
#
# for n_steps in [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000]:
#     pr = ss.SteadyStateProblem(l_cond, r_cond, k, q, f, n_steps=n_steps, breakpoint=x_0)
#     pr.solve()
#     s = pr.get_sample(10)[1]
#     err = np.linalg.norm(ref - s, ord=2)
#     step = 1./n_steps
#
#     errors += [err]
#     steps += [step]
#
# errors = np.array(errors)
# steps = np.array(steps)
#
# p = np.polyfit(np.log(steps)[:-1], np.log(errors)[:-1], deg=1)
# line = np.poly1d(p)
#
# fig, ax = plt.subplots(1, 1)
#
# fig.suptitle('Скорость сходимости в задаче с переменными коэффициентами')
# ax.plot(np.log(steps), np.log(errors), 'bs')
# ax.plot(np.log(steps), line(np.log(steps)), 'b--', label=f'$\\frac{{  d(\\log||x - x_r||) }}{{ d(\\log{{h}}) }} \\approx {p[0]:.1f}$')
# ax.set_xlabel(r'$\log{h}$')
# ax.set_ylabel(r'$\log{||u - u_r||_{\infty}}$')
# ax.legend()
# ax.grid()
# plt.show()
