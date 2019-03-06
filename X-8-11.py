import numpy as np
import matplotlib.pyplot as plt

import methods.euler_method as em

A = np.array([[0, 1],
              [-100, -101]])
u_0 = np.array([2, -2])

eigval, eigvec = np.linalg.eig(A)

L = np.max(np.abs(eigval))/np.min(np.abs(eigval))

print(f'Stiffness (\\lambda_max/\\lambda_min) = {L:.1e} => system is stiff')

T = 5.5
n_samples = 10
n_iter = 25
h = T/n_samples/n_iter

coarse = np.zeros([u_0.shape[0], n_samples])
fine = np.zeros([u_0.shape[0], n_samples])

coarse[:, 0] = u_0
fine[:, 0] = u_0

for i in range(1, n_samples):
    coarse[:, i] = em.solve_lin_syst(coarse[:, i-1], A, h, n_iter, explicit=True)
    fine[:, i] = em.solve_lin_syst(fine[:, i-1], A, h/10, n_iter*10, explicit=True)

fig, ax = plt.subplots(1, 1)
ax.plot(coarse[0], coarse[1], 'b^', label=f'$h = {h:.1e}$')
ax.plot(fine[0], fine[1], 'r^', label=f'$h = {h/10:.1e}$')
ax.set_title('Explicit Euler method')
ax.set_xlabel('$y$')
ax.set_ylabel('$y\'$')
ax.grid(True)
ax.legend()
plt.show()


for i in range(1, n_samples):
    coarse[:, i] = em.solve_lin_syst(coarse[:, i-1], A, h, n_iter)
    fine[:, i] = em.solve_lin_syst(fine[:, i-1], A, h/10, n_iter*10)

fig, ax = plt.subplots(1, 1)
ax.plot(coarse[0], coarse[1], 'b^')
ax.plot(fine[0], fine[1], 'r^')
ax.set_title('Implicit Euler method')
ax.set_xlabel('$y$')
ax.set_ylabel('$y\'$')
ax.grid(True)
plt.show()

