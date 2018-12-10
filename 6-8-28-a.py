import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import methods.spline_interp as si


xs = np.linspace(-np.pi, np.pi, 100, endpoint=True)
x = np.linspace(-np.pi, np.pi, 10, endpoint=True)
y = np.cos(x)
sp = si.CubicSpline(x, y)
y_int = np.array([sp(x) for x in xs])

ref_sp = CubicSpline(x, y, bc_type='natural')
plt.plot(xs, ref_sp(xs), 'b-',
            xs, y_int, 'g--')
plt.show()
print(sp.t[:2])
print(sp.coefs[:2])
