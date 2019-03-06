import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import methods.spline_interp as si

x = np.array([0,1,2,3,4])
y = np.array([0.0, .05, .86603, 1.0, .86603])
sp = si.CubicSpline(x, y)
ref_sp = CubicSpline(x, y, bc_type='natural')
xs = np.linspace(0, 4, 50, endpoint=True)
y_int = np.array([sp(t) for t in xs])
plt.plot(xs, y_int, 'b-',
         x, y, 'rs',
         xs, ref_sp(xs), 'g--')

plt.grid(True)
plt.show()
exit()

xs = np.linspace(-np.pi, np.pi, 200, endpoint=True)
x = np.linspace(-np.pi, np.pi, 10, endpoint=True)
y = np.cos(x)
sp = si.CubicSpline(x, y)
y_int = np.array([sp(x) for x in xs])

ref_sp = CubicSpline(x, y, bc_type='natural')
plt.plot(xs, ref_sp(xs) - y_int, 'r--',
            xs, y_int, 'b--',
            xs, np.cos(xs), 'g-',
            x, y, 'gs')
plt.grid(True)
plt.show()

