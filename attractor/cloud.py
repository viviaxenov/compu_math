import numpy as np
import numpy.random
import scipy as sp

from methods.ode_integration import *
from methods.helpers import lorenz_attractor

import pyvtk
import os

f, jac = lorenz_attractor(10., 28., 8./3)
T = 40.
tau = 1e-2
ipf = 4
n_steps = int(T/tau)
n_particles = 10000
deviation = 0.01
start_point = np.array([0., 0., 0.])
starts = np.random.normal(start_point, deviation, (n_particles, 3))
r_0 = np.linalg.norm(starts - start_point, axis=1, ord=2)

cloud = [RK45Solver(f, x_0, 0., T, n_steps) for x_0 in starts]

for particle in cloud:
    particle.solve()

dirpath = 'vtk/000_10k/'
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

for i in range(0, n_steps + 1, ipf):
    points = [p.x[:, i] for p in cloud]
    velocities = [f(i, x) for x in points]
    vtk = pyvtk.VtkData(
            pyvtk.UnstructuredGrid(points),
            pyvtk.PointData(pyvtk.Vectors(velocities, name='Velocity'),
                            pyvtk.Scalars(r_0, name='r_0'))
            )
    vtk.tofile(dirpath + f'f{i}')

