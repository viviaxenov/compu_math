import numpy as np
import methods.shuttle as sh

u = np.array([1.0]*4)
m = np.array([2.0]*5)
l = np.array([1.0]*4)

tr = .0
bl = 0.0

A = np.array([
        [2.0, 1.0, 0.0, 0.0, tr],
        [1.0, 2.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 2.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 2.0, 1.0],
        [ bl, 0.0, 0.0, 1.0, 2.0],
    ])

f = np.array([6.0]*5)
sol = (np.linalg.solve(A, f))
shuttle = (sh.solve_cyclic(u, m, l, f, tr, bl))


with np.printoptions(linewidth=1000):
    print(np.linalg.solve(A,f))
    print(sh.solve(u, m, l, f))
    print(shuttle)
    print(shuttle - sh.solve(u, m, l, f))
