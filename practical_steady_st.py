import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt


import methods.steady_state as ss
from methods.shuttle import solve as shsolve

k = lambda x: np.full_like(x, 1.0)
q = lambda x: np.full_like(x, 1.0)
f = lambda x: np.full_like(x, 0.0)

def get_k_model(k_1, k_2, x_0):
    def k_model(x):
        a = np.zeros_like(x)
        a[x >= x_0] = k_2
        a[x < x_0] = k_1
        return a
    return k_model

problem = ss.SteadyStateProblem((0, 1.0, 0.0), (0, 1.0, 10.0), get_k_model(1.0, 5.0, 0.5), q, f, 2000)
problem.solve()

plt.plot(problem.domain, problem.solution)
plt.grid()
plt.show()