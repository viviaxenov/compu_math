import numpy as np
import matplotlib.pyplot as plt

import methods.euler_method as em


def f(x : np.float64, y : np.float64) -> np.float64:
	return y

x_0 = 1
y_0 = 30
x_max = 2
h = 0.01


x, y = em.solve(f, x_0, y_0, h, x_max)

plt.plot(x, y, 'b-', x, 30*np.exp(x - 1), 'r--')
plt.grid(True)
plt.show()
