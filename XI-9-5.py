import numpy as np
import matplotlib.pyplot as plt

import methods.periodic_problem as pp

P = lambda x: np.sin(2*np.pi*x) + np.full_like(x, 10.0)
f = lambda x: np.cos(2*np.pi*x)

pr = pp.PeriodicProblem(P, f, 200)
pr.solve()


fig, ax = plt.subplots()
ax.plot(pr.domain, pr.solution, 'b-')
ax.set_xlabel('x')
ax.set_ylabel('y(x)')
ax.grid()
plt.show()


