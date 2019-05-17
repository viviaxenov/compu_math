import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from methods.poisson_problem import PoissonProblem


def analytical(xx, yy):
    return -yy*(1.0 - yy)*(0.5 - xx)*(0.5 + xx)


def f(xx, yy):
    return 2*yy*(1.0 - yy) + 2*(0.5 - xx)*(0.5 + xx)


sz = 10         # количество шагов сетки на сторону (узлов на 1 больше)
ch_steps = 22   # количество итераций с чебвшёвским параметром на шаге
u_0 = lambda xx, yy: np.zeros_like(xx)

solver = PoissonProblem(sz, sz, (-0.5, 0.5, 0.0, 1.0), u_0, f)

total_it = solver.solve(n_iter=ch_steps, eps=1e-3)
# делает шаги по n_iter итерций с чебышёвским параметром
# пока невязка не станет меньше eps
# возвращает общее число сделанных шагов
# нужно подобрать (лапками) максимальное n_iter, при котором решение устойчиво, тогда будет работать быстро

asl = analytical(solver.xx, solver.yy)[1:-1, 1:-1]
nsl = solver.solution[1:-1, 1:-1]
diff = (nsl - asl)/asl
err = np.linalg.norm(diff.flatten(), ord=np.inf)

print(f'({solver.n_x + 1}x{solver.n_y + 1}) points, h_x = {solver.h_x:.2e}, h_y = {solver.h_y:.2e}')
print(f'{ch_steps:-2d} Chebyshov steps | Total iterations: {total_it:5d} | Residual: {solver.err:.2e} | Relative error: {err:.2e}|')
print()

with np.printoptions(precision=5, linewidth=10000):
    print(solver.solution)

# если не хочешь смотреть графики, раскомментируй
# exit()

fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')
surf = ax.plot_surface(solver.xx, solver.yy, solver.solution, cmap=cm.coolwarm)
ax.set_title('Numeric solution')
ax = fig.add_subplot(212, projection='3d')
ax.set_title('Analytical solution')
surf = ax.plot_surface(solver.xx, solver.yy, analytical(solver.xx, solver.yy), cmap=cm.coolwarm)
plt.show()

