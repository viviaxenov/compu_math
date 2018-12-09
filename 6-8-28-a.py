import numpy as np
import matplotlib.pyplot as plt
import scipy as sp



xs = np.linspace(-1, 1, 50)
x = xs[::4]
y = np.sin(x)

ref_sp = sp.interpolate.CubicSpline(x, y, bc_type='natural')
my_sp = 
plt.plot(xs, np.sin(xs), 'b-', xs, ref_sp(xs), 'r--')
plt.show()
