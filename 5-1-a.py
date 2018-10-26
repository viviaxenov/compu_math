import numpy as np
import matplotlib.pyplot as plt

import methods.least_square_method as lstsq

#table = np.genfromtxt('data/5-1-a.csv', delimiter=',', names=True)
#x, y = table['x'], table['y']

p = np.array([6.4, -18.3])
line0 = np.poly1d(p)
x = np.linspace(-4, 4, num=15)
y = line0(x)
delta_y = np.random.normal(0.0, 3, size=15)
y += delta_y

coefs = lstsq.lin_fit(x, y)
p = np.polyfit(x, y, deg=1)
print(coefs)
print(p)

line = np.poly1d(coefs)

fig, axs = plt.subplots()
axs.errorbar(x, y, yerr=delta_y.max(), fmt='bs')
axs.plot(x, line(x))
axs.plot(x, line0(x), linestyle='--')
axs.grid(True)
fig.show()
input()
